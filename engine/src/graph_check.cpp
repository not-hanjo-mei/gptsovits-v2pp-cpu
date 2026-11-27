// graph_check: generic free-engine vs ORT bisect tool. Runs one
// ONNX graph with .npy inputs and compares named outputs against
// .npy references (saved from an ORT run, cast to float32).
//
// usage: graph_check GRAPH.onnx --in name=path.npy ...
//        --ref name=path.npy ... [--save name=path.npy ...]
// Input .npy dtypes: <i8, <i4 (widened to i8), <f4, <f2 (decoded
// to f4). Ref .npy must be <f4. Every graph output's shape and
// absmax are printed, ref or not.

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

static void write_file(const std::string& path, const uint8_t* p, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    fwrite(p, 1, n, f);
    fclose(f);
}

struct Npy {
    std::string descr;  // <i8 <i4 <f4 <f2
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    std::vector<uint8_t> buf;  // owns the file bytes
    int64_t numel = 1;
};

static Npy read_npy(const std::string& path) {
    Npy r;
    r.buf = read_file(path);
    if (r.buf.size() < 10 || memcmp(r.buf.data(), "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "%s: not a .npy\n", path.c_str());
        exit(2);
    }
    uint16_t hlen = r.buf[8] | (r.buf[9] << 8);
    std::string hdr((const char*)r.buf.data() + 10, hlen);
    auto dp = hdr.find("'descr':");
    auto q1 = hdr.find("'", dp + 8), q2 = hdr.find("'", q1 + 1);
    r.descr = hdr.substr(q1 + 1, q2 - q1 - 1);
    auto sp = hdr.find("'shape':");
    auto open = hdr.find('(', sp), close = hdr.find(')', open);
    std::string sh = hdr.substr(open + 1, close - open - 1);
    size_t pos = 0;
    while (pos < sh.size()) {
        auto comma = sh.find(',', pos);
        std::string tok = sh.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (tok.find_first_not_of(" \t") != std::string::npos) {
            int64_t d = std::stoll(tok);
            r.shape.push_back(d);
            r.numel *= d;
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (r.shape.empty()) { r.shape.push_back(1); r.numel = 1; }
    r.data = r.buf.data() + 10 + hlen;
    return r;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3ff;
            bits = sign | ((uint32_t)(e + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | (0xffu << 23) | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Materialize any engine output tensor to flat float (row-major),
// following strides when present.
static std::vector<float> materialize_f(const freeeng::Tensor& t) {
    std::vector<float> out((size_t)t.numel());
    auto rd = [&](int64_t off) -> float {
        switch (t.dtype) {
            case 1: return ((const float*)t.data)[off];
            case 10: return f16_to_f32(((const uint16_t*)t.data)[off]);
            case 7: return (float)((const int64_t*)t.data)[off];
            case 6: return (float)((const int32_t*)t.data)[off];
            case 11: return (float)((const double*)t.data)[off];
            default: return 0;
        }
    };
    if (t.strides.empty()) {
        for (size_t i = 0; i < out.size(); i++) out[i] = rd((int64_t)i);
        return out;
    }
    int r = (int)t.shape.size();
    std::vector<int64_t> coord(r, 0);
    int64_t off = 0;
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = rd(off);
        for (int d = r - 1; d >= 0; d--) {
            off += t.strides[d];
            if (++coord[d] < t.shape[d]) break;
            coord[d] = 0;
            off -= t.strides[d] * t.shape[d];
        }
    }
    return out;
}

static void save_npy_f32(const std::string& path, const std::vector<float>& v,
                          const std::vector<int64_t>& shape) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) hdr += ", ";
        hdr += std::to_string(shape[i]);
    }
    if (shape.size() == 1) hdr += ",";
    hdr += "), }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - total % 64) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    std::vector<uint8_t> out;
    const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    out.insert(out.end(), magic, magic + 6);
    out.push_back(1); out.push_back(0);
    uint16_t hl = (uint16_t)hdr.size();
    out.push_back(hl & 0xff); out.push_back(hl >> 8);
    out.insert(out.end(), hdr.begin(), hdr.end());
    const uint8_t* p = (const uint8_t*)v.data();
    out.insert(out.end(), p, p + v.size() * 4);
    write_file(path, out.data(), out.size());
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
    if (argc < 2) { std::cerr << "usage: graph_check GRAPH [--in n=p.npy] [--ref n=p.npy] [--save n=p.npy]\n"; return 2; }
    std::string graph_path = argv[1];
    std::map<std::string, std::string> in_paths, ref_paths, save_paths;
    {
        std::map<std::string, std::string>* cur = &in_paths;
        for (int i = 2; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--in") { cur = &in_paths; continue; }
            if (a == "--ref") { cur = &ref_paths; continue; }
            if (a == "--save") { cur = &save_paths; continue; }
            auto eq = a.find('=');
            if (eq == std::string::npos) { std::cerr << "bad arg " << a << "\n"; return 2; }
            (*cur)[a.substr(0, eq)] = a.substr(eq + 1);
        }
    }
    cpu::log_features("graph_check");
    try {
        // Keep Npy objects (and converted buffers) alive for the run.
        std::map<std::string, Npy> npys;
        std::map<std::string, std::vector<int64_t>> i64_bufs;
        std::map<std::string, std::vector<float>> f32_bufs;
        std::map<std::string, freeeng::Tensor> ins;
        for (auto& kv : in_paths) {
            Npy n = read_npy(kv.second);
            freeeng::Tensor t;
            t.shape = n.shape;
            if (n.descr == "<i8") {
                t.dtype = 7;
                t.data = n.data;
            } else if (n.descr == "<i4") {
                auto& b = i64_bufs[kv.first];
                b.resize((size_t)n.numel);
                const int32_t* p = (const int32_t*)n.data;
                for (int64_t j = 0; j < n.numel; j++) b[j] = p[j];
                t.dtype = 7;
                t.data = (const uint8_t*)b.data();
            } else if (n.descr == "<f4") {
                t.dtype = 1;
                t.data = n.data;
            } else if (n.descr == "<f2") {
                auto& b = f32_bufs[kv.first];
                b.resize((size_t)n.numel);
                const uint16_t* p = (const uint16_t*)n.data;
                for (int64_t j = 0; j < n.numel; j++) b[j] = f16_to_f32(p[j]);
                t.dtype = 1;
                t.data = (const uint8_t*)b.data();
            } else {
                std::cerr << kv.first << ": unsupported descr " << n.descr << "\n";
                return 2;
            }
            npys.emplace(kv.first, std::move(n));
            ins.emplace(kv.first, t);
        }
        onnxm::Graph g = onnxm::load_model(graph_path);
        auto t0 = std::chrono::steady_clock::now();
        auto plan = freeeng::Plan::compile(g, "graph_check");
        auto t1 = std::chrono::steady_clock::now();
        auto outs = plan->run(ins);
        auto t2 = std::chrono::steady_clock::now();
        std::printf("compile_ms %lld run_ms %lld\n",
                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        for (auto& kv : outs) {
            const auto& t = kv.second;
            auto flat = materialize_f(t);
            double amax = 0;
            for (float v : flat) amax = std::max(amax, (double)std::fabs(v));
            std::printf("out %s shape:", kv.first.c_str());
            for (auto d : t.shape) std::printf(" %lld", (long long)d);
            std::printf(" dtype %d absmax %.6g", t.dtype, amax);
            auto rp = ref_paths.find(kv.first);
            if (rp != ref_paths.end()) {
                Npy rn = read_npy(rp->second);
                if (rn.descr != "<f4") { std::cerr << "ref " << rp->second << " not <f4\n"; return 2; }
                const float* rb = (const float*)rn.data;
                size_t n = std::min(flat.size(), (size_t)rn.numel);
                Stats s = compare(flat.data(), rb, n);
                std::printf(" vs_ref corr %.6f relrms %.6f maxerr %.6g (ref numel %lld)",
                            s.corr, s.relrms, s.maxerr, (long long)rn.numel);
            }
            std::printf("\n");
            auto sp = save_paths.find(kv.first);
            if (sp != save_paths.end()) save_npy_f32(sp->second, flat, t.shape);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "graph_check: " << e.what() << "\n";
        return 1;
    }
}
