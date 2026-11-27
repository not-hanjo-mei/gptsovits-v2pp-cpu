// roberta_check: run the RoBERTa graph (GenieData/RoBERTa) on the
// free engine with Genie-frontend inputs for the neutral1 prompt
// and compare phone_features against the ORT run and the shipped
// refpack's ref_bert.
//
// usage: roberta_check DIR   (DIR = refextract dir holding
//   hubert/../roberta_*.npy + bert_ort.npy + bert_rpk.npy; the
//   graph is read from <tts-root>/GenieData/RoBERTa/RoBERTa.onnx,
//   with tts-root = DIR/../../.. -- pass the refextract dir under
//   work/, i.e. ~/.../tts-engine/work/refextract)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cpu.h"
#include "free_engine.h"
#include "onnx_model.h"

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    std::vector<uint8_t> buf;
    uint8_t chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
    fclose(f);
    return buf;
}

struct NpyI64 {
    std::vector<int64_t> shape;
    std::vector<int64_t> data;
};

static NpyI64 read_npy_i64(const std::string& path) {
    auto buf = read_file(path);
    if (buf.size() < 10 || memcmp(buf.data(), "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "%s: not a .npy\n", path.c_str());
        exit(2);
    }
    uint16_t hlen = buf[8] | (buf[9] << 8);
    std::string hdr((const char*)buf.data() + 10, hlen);
    if (hdr.find("'<i8'") == std::string::npos && hdr.find("\"<i8\"") == std::string::npos &&
        hdr.find("'<i8'") == std::string::npos && hdr.find("<i8") == std::string::npos) {
        std::fprintf(stderr, "%s: expected <i8, header: %s\n", path.c_str(), hdr.c_str());
        exit(2);
    }
    NpyI64 r;
    auto sp = hdr.find("'shape':");
    auto open = hdr.find('(', sp), close = hdr.find(')', open);
    std::string sh = hdr.substr(open + 1, close - open - 1);
    size_t pos = 0;
    int64_t total = 1;
    while (pos < sh.size()) {
        auto comma = sh.find(',', pos);
        std::string tok = sh.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (tok.find_first_not_of(" \t") != std::string::npos) {
            int64_t d = std::stoll(tok);
            r.shape.push_back(d);
            total *= d;
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    const uint8_t* p = buf.data() + 10 + hlen;
    r.data.resize((size_t)total);
    memcpy(r.data.data(), p, (size_t)total * 8);
    return r;
}

static std::vector<float> read_npy_f32(const std::string& path) {
    auto buf = read_file(path);
    if (buf.size() < 10 || memcmp(buf.data(), "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "%s: not a .npy\n", path.c_str());
        exit(2);
    }
    uint16_t hlen = buf[8] | (buf[9] << 8);
    size_t bytes = buf.size() - 10 - hlen;
    std::vector<float> out(bytes / 4);
    memcpy(out.data(), buf.data() + 10 + hlen, out.size() * 4);
    return out;
}

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
    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0, se = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        if (std::fabs(d) > s.maxerr) s.maxerr = std::fabs(d);
        se += d * d;
        sa += a[i]; sb += b[i]; sab += (double)a[i] * b[i];
        saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i];
    }
    double cov = sab - sa * sb / (double)n;
    double va = saa - sa * sa / (double)n, vb = sbb - sb * sb / (double)n;
    s.corr = (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : 0;
    s.relrms = std::sqrt(se / (double)n) / (std::sqrt(sbb / (double)n) + 1e-30);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: roberta_check DIR\n"; return 2; }
    std::string dir = argv[1];
    cpu::log_features("roberta_check");
    try {
        auto ids = read_npy_i64(dir + "/roberta_input_ids.npy");
        auto mask = read_npy_i64(dir + "/roberta_attention_mask.npy");
        auto rep = read_npy_i64(dir + "/roberta_repeats.npy");
        auto bert_ort = read_npy_f32(dir + "/bert_ort.npy");
        auto bert_rpk = read_npy_f32(dir + "/bert_rpk.npy");
        // Graph: DIR is <root>/work/refextract.
        std::string graph_path = dir + "/../../GenieData/RoBERTa/RoBERTa.onnx";
        onnxm::Graph g = onnxm::load_model(graph_path);
        auto t0 = std::chrono::steady_clock::now();
        auto plan = freeeng::Plan::compile(g, "roberta");
        auto t1 = std::chrono::steady_clock::now();
        auto mk = [](const NpyI64& a) {
            freeeng::Tensor t;
            t.dtype = 7;
            t.data = (const uint8_t*)a.data.data();
            t.shape = a.shape;
            return t;
        };
        std::map<std::string, freeeng::Tensor> ins;
        ins.emplace("input_ids", mk(ids));
        ins.emplace("attention_mask", mk(mask));
        ins.emplace("repeats", mk(rep));
        auto outs = plan->run(ins);
        auto t2 = std::chrono::steady_clock::now();
        const auto& out = outs.at("phone_features");
        std::printf("output shape:");
        for (auto d : out.shape) std::printf(" %lld", (long long)d);
        std::printf(" (ort/rpk numel %zu)\n", bert_ort.size());
        auto flat = materialize(out);
        size_t n = std::min(flat.size(), bert_ort.size());
        Stats so = compare(flat.data(), bert_ort.data(), n);
        Stats sr = compare(flat.data(), bert_rpk.data(), n);
        std::printf("{\"free_vs_ort\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"free_vs_rpk\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"compile_ms\": %lld, \"run_ms\": %lld}\n",
                    so.corr, so.relrms, so.maxerr, sr.corr, sr.relrms, sr.maxerr,
                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "roberta_check: " << e.what() << "\n";
        return 1;
    }
}
