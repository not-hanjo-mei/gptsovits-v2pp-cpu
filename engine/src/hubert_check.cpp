// hubert_check: run the HuBERT graph (converted fp32 external
// weights, see tools/convert_fp16_external.py) on the free engine
// and compare last_hidden_state against the ORT output and the
// shipped refpack tensor for the same reference clip.
//
//   hubert_check DIR
//
// DIR holds hubert/chinese-hubert-base.onnx (+ its weights bin),
// audio_16k.npy (the exact waveform the ORT run consumed),
// ssl_ort.npy and ssl_rpk.npy (the two oracles; the prep step
// verified they agree to one fp16 ulp).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cpu.h"
#include "free_engine.h"
#include "onnx_model.h"

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)sz);
    if (sz > 0 && fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        throw std::runtime_error("short read " + path);
    }
    fclose(f);
    return buf;
}

static std::vector<float> read_npy_f32(const std::string& path) {
    auto bytes = read_file(path);
    if (bytes.size() < 10 || memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
        throw std::runtime_error("bad npy " + path);
    uint16_t hlen = (uint16_t)bytes[8] | ((uint16_t)bytes[9] << 8);
    size_t off = 10 + hlen;
    size_t nbytes = bytes.size() - off;
    std::vector<float> out(nbytes / 4);
    memcpy(out.data(), bytes.data() + off, nbytes - nbytes % 4);
    return out;
}

// Plan outputs may be strided views (a graph ending in Transpose
// returns the view, not a materialized copy). Flatten in logical
// order before comparing.
static std::vector<float> materialize(const freeeng::Tensor& t) {
    std::vector<float> out((size_t)t.numel());
    const float* p = (const float*)t.data;
    if (t.strides.empty()) {
        memcpy(out.data(), p, out.size() * 4);
        return out;
    }
    int r = (int)t.shape.size();
    std::vector<int64_t> coord(r, 0);
    int64_t off = 0;
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = p[off];
        for (int d = r - 1; d >= 0; d--) {
            off += t.strides[d];
            if (++coord[d] < t.shape[d]) break;
            coord[d] = 0;
            off -= t.strides[d] * t.shape[d];
        }
    }
    return out;
}

struct Stats { double maxerr = 0, corr = 1, relrms = 0; };
static Stats compare(const float* a, const float* b, size_t n) {
    Stats s;
    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0, num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        s.maxerr = std::max(s.maxerr, std::abs(d));
        num += d * d;
        den += (double)b[i] * b[i];
        sa += a[i]; sb += b[i]; sab += (double)a[i] * b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i];
    }
    s.relrms = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    double cov = sab - sa * sb / (double)n;
    double va = saa - sa * sa / (double)n, vb = sbb - sb * sb / (double)n;
    s.corr = (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : (s.maxerr == 0 ? 1.0 : 0.0);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: hubert_check DIR | GRAPH.onnx AUDIO.npy REF.npy\n"; return 2; }
    cpu::log_features("hubert_check");
    if (argc == 4) {
        // Bisect mode: run one (sub)graph, compare its single
        // output against the ORT reference in REF.npy.
        try {
            auto audio = read_npy_f32(argv[2]);
            auto ref = read_npy_f32(argv[3]);
            onnxm::Graph g = onnxm::load_model(argv[1]);
            auto plan = freeeng::Plan::compile(g, "bisect");
            freeeng::Tensor in;
            in.dtype = 1;
            in.data = (const uint8_t*)audio.data();
            in.shape = {1, (int64_t)audio.size()};
            std::map<std::string, freeeng::Tensor> ins;
            ins.emplace(g.inputs[0].name, in);
            auto outs = plan->run(ins);
            const auto& out = outs.begin()->second;
            std::printf("output shape:");
            for (auto d : out.shape) std::printf(" %lld", (long long)d);
            std::printf(" (ref numel %zu)\n", ref.size());
            auto flat = materialize(out);
            size_t n = std::min(flat.size(), ref.size());
            Stats s = compare(flat.data(), ref.data(), n);
            std::printf("{\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}\n", s.corr, s.relrms, s.maxerr);
            if (const char* save = std::getenv("HUBERT_SAVE")) {
                // Minimal .npy writer (f4, C order, up to 3 dims).
                std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
                for (size_t i = 0; i < out.shape.size(); i++) {
                    if (i) hdr += ", ";
                    hdr += std::to_string(out.shape[i]);
                }
                if (out.shape.size() == 1) hdr += ",";
                hdr += "), }";
                while ((10 + hdr.size() + 1) % 64) hdr += " ";
                hdr += "\n";
                FILE* f = fopen(save, "wb");
                const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
                fwrite(magic, 1, 6, f);
                uint8_t ver[2] = {1, 0};
                fwrite(ver, 1, 2, f);
                uint16_t hl = (uint16_t)hdr.size();
                fwrite(&hl, 2, 1, f);
                fwrite(hdr.data(), 1, hdr.size(), f);
                fwrite(flat.data(), 4, flat.size(), f);
                fclose(f);
            }
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "hubert_check: " << e.what() << "\n";
            return 1;
        }
    }
    std::string dir = argv[1];
    try {
        auto audio = read_npy_f32(dir + "/audio_16k.npy");
        auto ssl_ort = read_npy_f32(dir + "/ssl_ort.npy");
        auto ssl_rpk = read_npy_f32(dir + "/ssl_rpk.npy");
        auto t0 = Clock::now();
        onnxm::Graph g = onnxm::load_model(dir + "/hubert/chinese-hubert-base.onnx");
        auto plan = freeeng::Plan::compile(g, "hubert");
        double t_compile = ms_since(t0);
        freeeng::Tensor in;
        in.dtype = 1;
        in.data = (const uint8_t*)audio.data();
        in.shape = {1, (int64_t)audio.size()};
        std::map<std::string, freeeng::Tensor> ins;
        ins.emplace("input_values", in);
        t0 = Clock::now();
        auto outs = plan->run(ins);
        double t_run = ms_since(t0);
        const auto& out = outs.at("last_hidden_state");
        std::printf("output shape:");
        for (auto d : out.shape) std::printf(" %lld", (long long)d);
        std::printf(" (ort/rpk numel %zu)\n", ssl_ort.size());
        auto flat = materialize(out);
        const float* o = flat.data();
        size_t n = std::min(flat.size(), ssl_ort.size());
        Stats so = compare(o, ssl_ort.data(), n);
        Stats sr = compare(o, ssl_rpk.data(), n);
        std::printf("{\"free_vs_ort\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"free_vs_rpk\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"compile_ms\": %.0f, \"run_ms\": %.0f}\n",
                    so.corr, so.relrms, so.maxerr, sr.corr, sr.relrms, sr.maxerr, t_compile, t_run);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "hubert_check: " << e.what() << "\n";
        return 1;
    }
}
