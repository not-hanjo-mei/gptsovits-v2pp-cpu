// vits_bench: tile-level A/B for conv1d_fast on synthetic shapes
// (the PCONV census leaders). For each shape: correctness of the
// NEON tile against the generic tile (max abs diff), then timed
// runs of both tiles through the real conv1d_fast path, with the
// dequant/compute split read off vc_stats deltas. Tile selection is
// switched in-process via packed::vits_tile_override.
//
// usage: vits_bench [threads] [reps]
#include "vits_kernels.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace packed;
using Clk = std::chrono::steady_clock;

struct Shape {
    const char* name;
    int64_t Ci, O, K, stride, dil, L;
    int mode;
};

// Filled from the PCONV census (see REPORT): executed-shape leaders
// of the shipping vits graph at a medium-sentence length scale.
static const Shape kShapes[] = {
    // name, Ci, O, K, stride, dil, L, mode
    {"prov_512x256_k3", 512, 256, 3, 1, 1, 2048, 0},     // provisional (pre-census)
    {"lead_192_k11_d1", 192, 192, 11, 1, 1, 12000, 0},   // census #1: 11.1% of cp
    {"lead_192_k7_d1", 192, 192, 7, 1, 1, 12000, 0},     // census #2: 8.1%
    {"lead_192_k3_d1", 192, 192, 3, 1, 1, 12000, 0},     // census #3: 6.3%
    {"lead_384_k11_d1", 384, 384, 11, 1, 1, 1500, 0},    // census #4: 5.4%
    {"lead_96_k11_d1_m1", 96, 96, 11, 1, 1, 24000, 1},   // census #5: 5.0%
    {"lead_192_k11_d5", 192, 192, 11, 1, 5, 12000, 0},   // census: 3.5% (wide window)
    {"tail_24_k11_d1_m1", 24, 24, 11, 1, 1, 96000, 1},   // census: 1.6% (full-rate tail)
    // Knife 4 (dil=3/5 register window): dilated variants of the
    // census families, mirroring the dil=1 leaders above.
    {"lead_192_k11_d3", 192, 192, 11, 1, 3, 12000, 0},
    {"lead_192_k7_d3", 192, 192, 7, 1, 3, 12000, 0},
    {"lead_192_k7_d5", 192, 192, 7, 1, 5, 12000, 0},
    {"lead_192_k3_d3", 192, 192, 3, 1, 3, 12000, 0},
    {"lead_192_k3_d5", 192, 192, 3, 1, 5, 12000, 0},
    {"lead_384_k11_d3", 384, 384, 11, 1, 3, 1500, 0},
    {"lead_96_k11_d3_m1", 96, 96, 11, 1, 3, 24000, 1},
    {"lead_96_k7_d5_m1", 96, 96, 7, 1, 5, 24000, 1},
};

int main(int argc, char** argv) {
    int threads = argc > 1 ? std::atoi(argv[1]) : 1;
    int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    set_kernel_threads(threads);
    std::mt19937 rng(11);
    auto rf = [&] { return std::uniform_real_distribution<float>(-1, 1)(rng); };

    for (const Shape& s : kShapes) {
        const int64_t C = s.Ci * s.K;
        const int64_t pad_l = (s.K - 1) * s.dil / 2;
        const int64_t Lout = (s.L + 2 * pad_l - s.dil * (s.K - 1) - 1) / s.stride + 1;
        std::vector<uint8_t> pk;
        std::vector<float> sc;
        std::vector<uint8_t> ze;
        if (s.mode == 1) {
            pk.resize((size_t)s.O * C);
            sc.resize((size_t)s.O);
            for (auto& v : sc) v = 0.01f;
        } else {
            pk.resize((size_t)s.O * (C / 64) * 32);
            sc.resize((size_t)s.O * (C / 64));
            ze.resize((size_t)s.O * (C / 64));
            for (auto& v : sc) v = 0.01f;
            for (auto& b : ze) b = 8;
        }
        for (auto& b : pk) b = (uint8_t)rng();
        std::vector<float> X((size_t)s.Ci * s.L), bias((size_t)s.O);
        for (auto& v : X) v = rf();
        for (auto& v : bias) v = rf() * 0.1f;
        std::vector<float> Yg((size_t)s.O * Lout), Yn((size_t)s.O * Lout);
        auto run_once = [&](float* Y) {
            conv1d_fast(X.data(), 1, s.Ci, s.L, pk.data(), sc.data(), ze.empty() ? nullptr : ze.data(),
                        s.mode, s.O, s.K, s.stride, pad_l, s.dil, Lout, bias.data(), Y);
        };
        // correctness: generic vs NEON through the same entry point
        vits_tile_override(0);
        run_once(Yg.data());
        vits_tile_override(1);
        run_once(Yn.data());
        double maxdiff = 0, maxref = 0;
        for (size_t i = 0; i < Yg.size(); i++) {
            maxdiff = std::max(maxdiff, (double)std::fabs(Yg[i] - Yn[i]));
            maxref = std::max(maxref, (double)std::fabs(Yg[i]));
        }
        // timing
        double t_gen = 0, t_neon = 0, dq_gen = 0, dq_neon = 0;
        for (int tile = 0; tile < 2; tile++) {
            vits_tile_override(tile);
            run_once(tile ? Yn.data() : Yg.data());  // warmup (also pool spin-up)
            long long dq0 = vc_stats().dequant_us.load(), cp0 = vc_stats().compute_us.load();
            auto t0 = Clk::now();
            for (int r = 0; r < reps; r++) run_once(tile ? Yn.data() : Yg.data());
            double dt = std::chrono::duration<double>(Clk::now() - t0).count() / reps;
            long long dqs = (vc_stats().dequant_us.load() - dq0) / reps;
            if (tile == 0) { t_gen = dt; dq_gen = dqs / 1e3; }
            else { t_neon = dt; dq_neon = dqs / 1e3; }
            (void)cp0;
        }
        vits_tile_override(-1);
        double gmac = (double)s.O * Lout * C / 1e9;
        std::printf(
            "%s Ci=%lld O=%lld K=%lld dil=%lld s=%lld L=%lld mode=%d | check maxdiff %.3g (maxref "
            "%.3g) | generic %.1f ms %.3f GMAC/s dq %.1f ms | neon %.1f ms %.3f GMAC/s dq %.1f ms "
            "| speedup %.2fx (threads=%d)\n",
            s.name, (long long)s.Ci, (long long)s.O, (long long)s.K, (long long)s.dil,
            (long long)s.stride, (long long)s.L, s.mode, maxdiff, maxref, t_gen * 1e3,
            gmac / t_gen, dq_gen, t_neon * 1e3, gmac / t_neon, dq_neon,
            t_neon > 0 ? t_gen / t_neon : 0.0, threads);
        std::fflush(stdout);
    }
    return 0;
}
