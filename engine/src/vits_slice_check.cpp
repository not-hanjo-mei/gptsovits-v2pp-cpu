// vits_slice_check: validates decoder-level streaming for the
// host's chunked VITS mode. Semantic-level slicing is unsound
// (the vits front aligns text and semantic globally; slicing the
// semantic input changes the output everywhere, corr ~0.16 on the
// noiseless graph). The split used here runs the front of the
// graph (tools/split_vits.py) once to get the latent z, then runs
// the purely convolutional decoder (/vq_model/dec/) on z slices
// with a few frames of context at each joint.
//
// On the noiseless fp32 split graphs this reports:
//   * dec one-shot vs the unsplit graph (split sanity, ~exact)
//   * dec chunked vs one-shot for several context widths:
//     overall / boundary / interior agreement

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cpu.h"
#include "free_engine.h"
#include "npz.h"
#include "packed_core.h"

static freeeng::Tensor npy_tensor(const NpyTensor& t) {
    freeeng::Tensor ft;
    ft.data = t.data.data();
    ft.shape = t.shape;
    if (t.descr == "<f4") ft.dtype = 1;
    else if (t.descr == "<i8") ft.dtype = 7;
    else if (t.descr == "<i4") ft.dtype = 6;
    else throw std::runtime_error("npz: unexpected descr " + t.descr);
    return ft;
}

static std::vector<float> copy_f32(const freeeng::Tensor& t) {
    if (t.dtype != 1) throw std::runtime_error("expected f32 tensor");
    const float* p = (const float*)t.data;
    if (t.contiguous()) return std::vector<float>(p, p + t.numel());
    std::vector<int64_t> coord(t.shape.size(), 0);
    std::vector<float> v;
    v.reserve((size_t)t.numel());
    for (int64_t i = 0; i < t.numel(); i++) {
        int64_t off = 0;
        for (size_t d = 0; d < coord.size(); d++) off += coord[d] * t.strides[d];
        v.push_back(p[off]);
        for (int d = (int)coord.size() - 1; d >= 0; d--) {
            if (++coord[d] < t.shape[d]) break;
            coord[d] = 0;
        }
    }
    return v;
}

struct Stats {
    double maxerr = 0, corr = 1;
};

static Stats compare(const float* a, const float* b, size_t n) {
    Stats s;
    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; i++) {
        s.maxerr = std::max(s.maxerr, std::abs((double)a[i] - b[i]));
        sa += a[i]; sb += b[i]; sab += (double)a[i] * b[i];
        saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i];
    }
    double cov = sab - sa * sb / (double)n;
    double va = saa - sa * sa / (double)n, vb = sbb - sb * sb / (double)n;
    s.corr = (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : (s.maxerr == 0 ? 1.0 : 0.0);
    return s;
}

static void report(const char* tag, const std::vector<float>& joined,
                   const std::vector<float>& full, const std::vector<int64_t>& joints) {
    Stats all = compare(joined.data(), full.data(), full.size());
    std::vector<char> near(full.size(), 0);
    for (int64_t j : joints)
        for (int64_t i = std::max<int64_t>(0, j - 1280); i < std::min<int64_t>((int64_t)full.size(), j + 1280); i++)
            near[(size_t)i] = 1;
    std::vector<float> ja, fa, ji, fi;
    for (size_t i = 0; i < full.size(); i++)
        (near[i] ? ja : ji).push_back(joined[i]), (near[i] ? fa : fi).push_back(full[i]);
    std::printf("%s overall corr %.6f maxerr %.6g", tag, all.corr, all.maxerr);
    if (!ja.empty()) {
        Stats b = compare(ja.data(), fa.data(), ja.size());
        std::printf(" | boundary corr %.6f maxerr %.6g", b.corr, b.maxerr);
    }
    if (!ji.empty()) {
        Stats in = compare(ji.data(), fi.data(), ji.size());
        std::printf(" | interior corr %.6f maxerr %.6g", in.corr, in.maxerr);
    }
    std::printf("\n");
}

int main(int argc, char** argv) {
    std::string root;
    int threads = 4;
    int64_t chunk_tokens = 25;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--root") root = argv[++i];
        else if (a == "--threads") threads = std::atoi(argv[++i]);
        else if (a == "--chunk-tokens") chunk_tokens = std::atoll(argv[++i]);
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    if (root.empty()) { std::cerr << "--root required\n"; return 2; }
    cpu::log_features("vits_slice_check");
    try {
        packed::set_kernel_threads(threads);
        Npz npz;
        npz.load(root + "/work/memopt/vits_feed.npz");
        std::map<std::string, freeeng::Tensor> feed;
        for (const char* nm : {"text_seq", "pred_semantic", "ge", "ge_advanced"})
            feed.emplace(nm, npy_tensor(npz.entries.at(nm)));
        const int64_t T = feed.at("pred_semantic").shape.back();

        auto compile = [&](const std::string& rel, const char* name) {
            onnxm::Graph g = onnxm::load_model(root + rel);
            return freeeng::Plan::compile(g, name);
        };
        auto full_plan = compile("/work/fp32_full/vits_fp32_noiseless.onnx", "vits-full");
        auto front_plan = compile("/work/fp32_full/vits_front_noiseless.onnx", "vits-front");
        auto dec_plan = compile("/work/fp32_full/vits_dec_noiseless.onnx", "vits-dec");
        const std::string ZN = "/vq_model/Mul_2_output_0";

        auto full_out = full_plan->run(feed);
        std::vector<float> full = copy_f32(full_out.at("audio"));
        full_plan->release_arena();
        auto front_out = front_plan->run(feed);
        const freeeng::Tensor& zt = front_out.at(ZN);
        std::vector<int64_t> zshape = zt.shape;
        std::vector<float> z = copy_f32(zt);
        front_plan->release_arena();
        const int64_t C = zshape[1], F = zshape[2];
        if (full.size() % (size_t)F) throw std::runtime_error("audio not divisible by frames");
        const int64_t spf = (int64_t)full.size() / F;  // samples per latent frame
        std::printf("T=%lld z shape [%lld, %lld, %lld] frames/token %.3f samples/frame %lld audio %zu\n",
                    (long long)T, (long long)zshape[0], (long long)C, (long long)F,
                    (double)F / (double)T, (long long)spf, full.size());

        // z is [1, C, F] channel-major: a frame slice is a
        // per-channel row copy, never a flat pointer offset.
        std::vector<float> zslice;
        auto run_dec = [&](int64_t a, int64_t b) {
            zslice.assign((size_t)(b - a) * C, 0);
            for (int64_t c = 0; c < C; c++)
                std::copy(z.begin() + (size_t)c * F + a, z.begin() + (size_t)c * F + b,
                          zslice.begin() + (size_t)c * (b - a));
            std::map<std::string, freeeng::Tensor> df;
            freeeng::Tensor zv;
            zv.dtype = 1;
            zv.data = (const uint8_t*)zslice.data();
            zv.shape = {1, C, b - a};
            df.emplace(ZN, zv);
            df.emplace("ge", feed.at("ge"));
            auto outs = dec_plan->run(df);
            std::vector<float> out = copy_f32(outs.at("audio"));
            dec_plan->release_arena();
            return out;
        };

        std::vector<float> oneshot = run_dec(0, F);
        if (oneshot.size() != full.size()) throw std::runtime_error("dec one-shot length mismatch");
        report("dec one-shot:", oneshot, full, {});
        std::vector<float> oneshot2 = run_dec(0, F);
        report("dec one-shot run2:", oneshot2, full, {});
        {
            // Prefix slice: output must match the full output's
            // prefix except near the right edge (receptive field).
            int64_t pf = 100;
            std::vector<float> pre = run_dec(0, pf);
            std::printf("dec prefix[0,100) out size: %zu (expect %lld)\n",
                        pre.size(), (long long)(pf * spf));
            for (int w = 0; w < 6; w++) {
                Stats sw = compare(pre.data() + (size_t)w * spf, full.data() + (size_t)w * spf, (size_t)spf);
                std::printf("  frame %d window corr %.4f maxerr %.4g\n", w, sw.corr, sw.maxerr);
            }
            size_t keep = (size_t)(pf - 40) * spf;
            Stats s = compare(pre.data(), full.data(), keep);
            std::printf("dec prefix[0,100) first %zu samples: corr %.6f maxerr %.6g\n",
                        keep, s.corr, s.maxerr);
            // Middle slice starting at frame 50, no context: its
            // output should match full samples [50*spf, ...) away
            // from both edges if the decoder is shift-equivariant.
            std::vector<float> mid = run_dec(50, 100);
            size_t off = (size_t)(50 + 20) * spf, len = (size_t)10 * spf;
            Stats sm = compare(mid.data() + (size_t)20 * spf, full.data() + off, len);
            std::printf("dec mid[50,100) samples at +20..30 frames: corr %.6f maxerr %.6g\n",
                        sm.corr, sm.maxerr);
        }

        {
            // Length vs content: same length (150) with the tail
            // zeroed, and prefix content at full length via zero
            // padding. If early frames move in the first case the
            // coupling is content-global; if only length matters,
            // the padded run matches the full run's early frames.
            auto run_zvec = [&](const std::vector<float>& zv, int64_t frames) {
                std::map<std::string, freeeng::Tensor> df;
                freeeng::Tensor t;
                t.dtype = 1;
                t.data = (const uint8_t*)zv.data();
                t.shape = {1, C, frames};
                df.emplace(ZN, t);
                df.emplace("ge", feed.at("ge"));
                auto outs = dec_plan->run(df);
                std::vector<float> out = copy_f32(outs.at("audio"));
                dec_plan->release_arena();
                return out;
            };
            std::vector<float> ztail = z;
            for (int64_t c = 0; c < C; c++)
                for (int64_t f = 100; f < F; f++) ztail[(size_t)c * F + f] = 0;
            std::vector<float> o_tail = run_zvec(ztail, F);
            size_t keep = (size_t)60 * spf;
            Stats s1 = compare(o_tail.data(), full.data(), keep);
            std::printf("tail-zeroed (len 150) first 60 frames: corr %.6f maxerr %.6g\n", s1.corr, s1.maxerr);
            std::vector<float> zpad(z.size(), 0);
            for (int64_t c = 0; c < C; c++)
                std::copy(z.begin() + (size_t)c * F, z.begin() + (size_t)c * F + 100,
                          zpad.begin() + (size_t)c * F);
            std::vector<float> o_pad = run_zvec(zpad, F);
            Stats s2 = compare(o_pad.data(), o_tail.data(), keep);
            std::printf("zero-pad vs tail-zeroed first 60 frames: corr %.6f maxerr %.6g\n", s2.corr, s2.maxerr);
        }

        const int64_t fpt = F / T;  // frames per token (exact for this model family)
        const int64_t chunk = chunk_tokens * fpt;
        for (int64_t ctx : {0, 8, 16, 32, 64}) {
            std::vector<float> joined;
            joined.reserve(full.size());
            std::vector<int64_t> joints;
            for (int64_t s = 0; s < F; s += chunk) {
                int64_t e = std::min(F, s + chunk);
                int64_t cs = std::max<int64_t>(0, s - ctx);
                int64_t ce = std::min(F, e + ctx);
                std::vector<float> raw = run_dec(cs, ce);
                size_t off = (size_t)(s - cs) * spf;
                size_t len = (size_t)(e - s) * spf;
                if (off + len > raw.size()) throw std::runtime_error("dec slice shorter than spf mapping");
                joined.insert(joined.end(), raw.begin() + off, raw.begin() + off + len);
                if (e < F) joints.push_back((int64_t)joined.size());
            }
            if (joined.size() != full.size()) throw std::runtime_error("joined length mismatch");
            char tag[64];
            std::snprintf(tag, 64, "dec chunked ctx=%lld:", (long long)ctx);
            report(tag, joined, full, joints);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
