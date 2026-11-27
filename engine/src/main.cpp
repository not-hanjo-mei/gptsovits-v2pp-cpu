// TTS host: GPT-SoVITS v2ProPlus synthesis with a host-controlled
// decode loop. Loads weight-only packs (W4 or W8 GPT, W8 vits), dequantizes
// them in-process and injects the fp32 tensors into the fp32 graph templates
// from Stage 1. Reference conditioning (incl. prompt-encoder outputs) comes
// from offline .rpk packs. A request may instead carry
// "ref_wav" + "ref_text": the five tensors are then extracted
// at runtime (refextract.h, free engine only) and cached by
// content hash for the life of the process. With
// "ref_save": true the tensors are also written to
// host/refpacks/<hash>.rpk and registered in the sidecar
// rt_index.json, so later processes can use them as
// {"tag": "rt:<hash>"} like any offline pack.
// Interface: resident process, stdin one JSON request per line, one wav file
// per request into --outdir, one JSON result line on stdout per request.
#ifndef POTATO_FREE_ONLY
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>

#include "cpu.h"
#include "deess.h"
#include "free_engine.h"
#include "free_pool.h"
#include "g2p.h"
#include "json_min.h"
#include "npz.h"
#ifdef POTATO_FREE_ONLY
// Free-only build: packed.h / vits_packed.h are the ORT custom-op glue
// and are bypassed entirely; the free engine needs only the ORT-free
// core (BlobStore, kernel pool) from packed_core.h.
#include "packed_core.h"
#else
#include "packed.h"
#include "vits_packed.h"
#endif
#include "patch.h"
#include "refextract.h"
#include "rpk.h"

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
    return -1;
}
static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_file(const std::string& path, const uint8_t* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write: " + path);
    f.write((const char*)data, n);
}

// ---------------- model loading ----------------
#ifndef POTATO_FREE_ONLY
struct SessionPack {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> in_names, out_names;
    std::vector<const char*> in_ptrs, out_ptrs;
};

static std::string load_and_patch(const std::string& template_path, const std::string& pack_path,
                                  const Json& manifest_sec, const char* scheme,
                                  std::string& report) {
    auto modelb = read_file(template_path);
    std::string model(modelb.begin(), modelb.end());
    std::map<std::string, RestoredTensor> repl;
    if (!pack_path.empty()) {
        Npz npz;
        npz.load(pack_path);
        std::vector<std::pair<std::string, std::string>> name_scheme;
        std::map<std::string, std::vector<int64_t>> shapes;
        const Json* entries = manifest_sec.find("manifest");
        if (!entries || entries->type != Json::Arr) throw std::runtime_error("manifest: no entries list");
        for (auto& e : entries->arr) {
            std::string name = e.find("name")->as_str();
            std::string sch = e.find("scheme")->as_str();
            std::vector<int64_t> shp;
            for (auto& d : e.find("shape")->arr) shp.push_back((int64_t)d.as_num());
            name_scheme.push_back({name, sch});
            shapes[name] = shp;
        }
        repl = restore_pack(npz, name_scheme, shapes);
    }
    int replaced = 0;
    size_t bytes = 0;
    auto patched = pb::patch_model(model, repl, replaced, bytes);
    std::ostringstream rs;
    rs << template_path.substr(template_path.find_last_of('/') + 1) << ": template " << model.size()
       << " B, pack " << (pack_path.empty() ? std::string("(none)") : pack_path.substr(pack_path.find_last_of('/') + 1))
       << " scheme " << scheme << ", " << repl.size() << " pack entries, " << replaced
       << " initializers replaced (" << bytes << " B fp32 injected)";
    report = rs.str();
    return patched;
}

static SessionPack make_session(Ort::Env& env, const std::string& model, int threads,
                                bool no_arena = false, const std::string& opt_level = "all") {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    if (no_arena) opts.DisableCpuMemArena();
    opts.SetGraphOptimizationLevel(opt_level == "basic" ? GraphOptimizationLevel::ORT_ENABLE_BASIC
                                                        : GraphOptimizationLevel::ORT_ENABLE_ALL);
    SessionPack sp;
    sp.session = std::make_unique<Ort::Session>(env, model.data(), model.size(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < sp.session->GetInputCount(); i++) sp.in_names.push_back(sp.session->GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < sp.session->GetOutputCount(); i++) sp.out_names.push_back(sp.session->GetOutputNameAllocated(i, alloc).get());
    for (auto& n : sp.in_names) sp.in_ptrs.push_back(n.c_str());
    for (auto& n : sp.out_names) sp.out_ptrs.push_back(n.c_str());
    return sp;
}

static Ort::Value f32_tensor(Ort::AllocatorWithDefaultOptions& alloc, const float* data, std::vector<int64_t> shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    auto v = Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
    memcpy(v.GetTensorMutableData<float>(), data, n * 4);
    return v;
}
static Ort::Value i64_tensor(Ort::AllocatorWithDefaultOptions& alloc, const int64_t* data, std::vector<int64_t> shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    auto v = Ort::Value::CreateTensor<int64_t>(alloc, shape.data(), shape.size());
    memcpy(v.GetTensorMutableData<int64_t>(), data, n * 8);
    return v;
}
#endif  // !POTATO_FREE_ONLY

// ---------------- ORT-free engine state (--engine free) ----------------
// The free engine runs the same packed-residency configuration as the
// ORT --packed path (W4 GPT pools + E vits pool over one BlobStore),
// with plans compiled from the same graph files. --vits-baked in free
// mode runs the baked fp32 vits graph on the free engine instead: the
// template's initializers are replaced in place by the E pack's
// restored fp32 tensors (the same values load_and_patch injects for
// ORT), so the plan sees exactly the ORT baked model's weights.
struct FreeHost {
    std::shared_ptr<packed::BlobStore> store;
    std::unique_ptr<FreePool> pool_f, pool_s, pool_v;
    std::shared_ptr<freeeng::Plan> enc, first, stage_a, stage_b, vits;
    // Streaming (packed path only): split vits plans, compiled
    // lazily on the first stream request. vits_split marks that the
    // split graphs belong to this host's vits path.
    std::shared_ptr<freeeng::Plan> vits_front, vits_dec;
    bool vits_baked = false;
    bool vits_split = false;
};

static std::shared_ptr<freeeng::Plan> compile_plan_file(const std::string& path, const char* label) {
    auto t0 = Clock::now();
    onnxm::Graph g = onnxm::load_model(path);
    auto p = freeeng::Plan::compile(g, label);
    std::cerr << "[load] free plan " << label << " compiled in " << (int)ms_since(t0)
              << " ms, " << p->step_count() << " steps; rss " << rss_kb() / 1024 << " MB\n";
    return p;
}

static void patch_graph_inplace(onnxm::Graph& g, const std::string& root, const std::string& pack_path,
                                const Json& manifest_sec) {
    Npz npz;
    npz.load(pack_path);
    std::vector<std::pair<std::string, std::string>> name_scheme;
    std::map<std::string, std::vector<int64_t>> shapes;
    const Json* entries = manifest_sec.find("manifest");
    if (!entries || entries->type != Json::Arr) throw std::runtime_error("manifest: no entries list");
    for (auto& e : entries->arr) {
        std::string name = e.find("name")->as_str();
        name_scheme.push_back({name, e.find("scheme")->as_str()});
        std::vector<int64_t> shp;
        for (auto& d : e.find("shape")->arr) shp.push_back((int64_t)d.as_num());
        shapes[name] = shp;
    }
    auto repl = restore_pack(npz, name_scheme, shapes);
    int replaced = 0;
    for (auto& t : g.initializers) {
        auto it = repl.find(t.name);
        if (it == repl.end()) continue;
        t.data_type = 1;
        t.dims = it->second.shape;
        t.data.assign((const uint8_t*)it->second.data.data(),
                      (const uint8_t*)it->second.data.data() + it->second.data.size() * 4);
        replaced++;
    }
    if (replaced != (int)repl.size())
        throw std::runtime_error("patch_graph_inplace: replaced " + std::to_string(replaced) + " of " +
                                 std::to_string(repl.size()) + " pack entries");
}

static freeeng::Tensor ft_from_rpk(const RpkTensor& t) {
    freeeng::Tensor ft;
    ft.dtype = t.dtype == 1 ? 7 : 1;
    ft.data = t.data.data();
    ft.shape = t.shape;
    return ft;
}
static std::vector<float> ft_to_vec(const freeeng::Tensor& t) {
    std::vector<float> v;
    v.reserve((size_t)t.numel());
    const float* p = (const float*)t.data;
    if (t.contiguous()) {
        v.assign(p, p + t.numel());
        return v;
    }
    std::vector<int64_t> coord(t.shape.size(), 0);
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

// ---------------- sampling ----------------
static int64_t sample_token(const float* logits, int n, int top_k, double top_p, double temperature,
                            std::mt19937_64& rng, bool greedy) {
    if (greedy || temperature <= 0.0) {
        return std::max_element(logits, logits + n) - logits;
    }
    std::vector<std::pair<float, int>> cand;
    cand.reserve(n);
    float mx = *std::max_element(logits, logits + n);
    for (int i = 0; i < n; i++) cand.push_back({(logits[i] - mx) / (float)temperature, i});
    if (top_k > 0 && top_k < n) {
        std::nth_element(cand.begin(), cand.begin() + top_k, cand.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        cand.resize(top_k);
    }
    double sum = 0;
    std::vector<std::pair<double, int>> sp;
    for (auto& c : cand) { double p = std::exp((double)c.first); sp.push_back({p, c.second}); sum += p; }
    for (auto& pr : sp) pr.first /= sum;
    std::sort(sp.begin(), sp.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::pair<double, int>> kept;
    double cum = 0;
    for (auto& pr : sp) {
        kept.push_back(pr);
        cum += pr.first;
        if (cum >= top_p) break;
    }
    double ksum = 0;
    for (auto& pr : kept) ksum += pr.first;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double u = uni(rng) * ksum, acc = 0;
    for (auto& pr : kept) { acc += pr.first; if (u <= acc) return pr.second; }
    return kept.back().second;
}

// ---------------- wav ----------------
static void write_wav(const std::string& path, const float* audio, size_t n, int sr) {
    std::ofstream f(path, std::ios::binary);
    auto w32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    auto w16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    f.write("RIFF", 4); w32(36 + (uint32_t)n * 2); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    f.write("data", 4); w32((uint32_t)n * 2);
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; i++) {
        float x = std::max(-1.0f, std::min(1.0f, audio[i]));
        pcm[i] = (int16_t)std::lrint(x * 32767.0f);
    }
    f.write((char*)pcm.data(), (std::streamsize)n * 2);
}

// ---- minimal wav IO for the offline --deess-file verification mode ----
static bool read_wav_any(const std::string& path, std::vector<float>& out, int& sr) {
    auto buf = read_file(path);
    if (buf.size() < 44 || memcmp(buf.data(), "RIFF", 4) != 0) return false;
    uint16_t fmt = 0, chans = 0, bits = 0;
    size_t pos = 12;
    const uint8_t* data = nullptr;
    size_t data_len = 0;
    while (pos + 8 <= buf.size()) {
        uint32_t csz = rd32(&buf[pos + 4]);
        if (memcmp(&buf[pos], "fmt ", 4) == 0) {
            fmt = rd16(&buf[pos + 8]); chans = rd16(&buf[pos + 10]);
            sr = (int)rd32(&buf[pos + 12]); bits = rd16(&buf[pos + 22]);
        } else if (memcmp(&buf[pos], "data", 4) == 0) {
            data = &buf[pos + 8]; data_len = csz;
        }
        pos += 8 + csz + (csz & 1);
    }
    if (!data || chans == 0) return false;
    if (fmt == 1 && bits == 16) {
        size_t frames = data_len / 2 / chans;
        out.resize(frames);
        for (size_t i = 0; i < frames; i++) out[i] = (int16_t)rd16(data + (i * chans) * 2) / 32768.0f;
    } else if (fmt == 3 && bits == 32) {
        size_t frames = data_len / 4 / chans;
        out.resize(frames);
        for (size_t i = 0; i < frames; i++) { float v; memcpy(&v, data + (i * chans) * 4, 4); out[i] = v; }
    } else return false;
    return true;
}
static void write_wav_f32(const std::string& path, const float* audio, size_t n, int sr) {
    std::ofstream f(path, std::ios::binary);
    auto w32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    auto w16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    f.write("RIFF", 4); w32(36 + (uint32_t)n * 4); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(3); w16(1); w32(sr); w32(sr * 4); w16(4); w16(32);
    f.write("data", 4); w32((uint32_t)n * 4);
    f.write((const char*)audio, n * 4);
}

// Trim leading/trailing near-silence, each side capped (shell behavior: up to 1.3 s).
static std::pair<size_t, size_t> trim_range(const float* a, size_t n, double max_trim_s, int sr) {
    float peak = 0;
    for (size_t i = 0; i < n; i++) peak = std::max(peak, std::fabs(a[i]));
    float thr = peak * 0.0032f;  // about -50 dB of peak
    size_t cap = (size_t)(max_trim_s * sr);
    size_t lead = 0;
    while (lead < n && std::fabs(a[lead]) <= thr && lead < cap) lead++;
    if (lead >= n) return {0, n};
    size_t trail = 0;
    while (trail < n - lead && std::fabs(a[n - 1 - trail]) <= thr && trail < cap) trail++;
    return {lead, n - trail};
}

int main(int argc, char** argv) {
    std::string root;  // --root is required (no baked-in default)
    std::string outdir, emit_dir;
    int gpt_bits = 4, threads = 4;
    // --ort-threads: ORT intra-op threads for the PACKED sessions only
    // (baked sessions always use --threads). The packed graphs' heavy ops
    // run on the kernel pool (--threads); ORT's own spinning intra-op
    // threads oversubscribe the cores and roughly double packed vits
    // time (measured S10), so the packed default is 1. Numerics are
    // unaffected (thread count changes no result for these ops).
    int ort_threads = 1;
    // Stage 3: --packed runs the two GPT decoder sessions from packed graphs
    // (weights stay packed in per-graph pools, fused-dequant kernels; W4 only).
    // --no-arena / --opt-level exist to attribute RSS in the stage-3 ledger;
    // defaults reproduce the stage-2 session configuration exactly.
    bool use_packed = false, no_arena = false, vits_baked = false;
    bool fp32_graphs = false;  // --fp32-graphs: standard-precision load path (free engine)
    bool fp16_weights = false;  // --fp16-weights: fp16 weight storage for the GPT plans (fp32 path only)
    std::string opt_level = "all";
    // --engine free: the ORT-free engine (ORT-FREE.md phase 2). GPT runs
    // packed residency (W4) and vits packed residency by default, or the
    // baked fp32 vits graph with --vits-baked -- all on free_engine
    // plans. The ORT path stays the default and is unchanged.
    std::string engine = "ort";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--root") root = next();
        else if (a == "--engine") engine = next();
        else if (a == "--outdir") outdir = next();
        else if (a == "--gpt-bits") gpt_bits = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--packed") use_packed = true;
        else if (a == "--ort-threads") ort_threads = std::stoi(next());
        // --vits-baked: with --packed, keep vits on the baked path (stage-3
        // behaviour) while GPT stays packed. Attribution/debugging only;
        // the default with --packed is vits packed residency (S10).
        else if (a == "--vits-baked") vits_baked = true;
        else if (a == "--fp32-graphs") fp32_graphs = true;
        else if (a == "--fp16-weights") fp16_weights = true;
        else if (a == "--no-arena") no_arena = true;
        else if (a == "--opt-level") opt_level = next();
        else if (a == "--emit-models") emit_dir = next();
        else if (a == "--g2p-stdin") { /* handled below */ }
        else if (a == "--deess-file") { i += 2; /* handled below */ }
        else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
    }
    // Offline verification mode: --deess-file in.wav out.wav applies the
    // deess2air output stage to a wav file (no models loaded).
    for (int i = 1; i + 2 < argc; i++) {
        if (std::string(argv[i]) == "--deess-file") {
            std::vector<float> sig;
            int fsr = 0;
            if (!read_wav_any(argv[i + 1], sig, fsr)) { std::cerr << "deess-file: cannot read wav\n"; return 2; }
            std::vector<float> out(sig.size());
            deess::deess2air(sig.data(), sig.size(), fsr, out.data());
            write_wav_f32(argv[i + 2], out.data(), out.size(), fsr);
            std::cerr << "deess-file: " << sig.size() << " samples @ " << fsr << " Hz processed\n";
            return 0;
        }
    }
    bool g2p_stdin = false;
    for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--g2p-stdin") g2p_stdin = true;
    if (g2p_stdin) {
        // Validation helper: one input line of text -> one JSON array of phone ids.
        G2P g2p;
        g2p.load(root + "/host/data");
        std::string l;
        while (std::getline(std::cin, l)) {
            if (l.empty()) continue;
            int fb = 0;
            auto ids = g2p.text_to_ids("." + l, fb);
            std::cout << "{\"ids\": [";
            for (size_t i = 0; i < ids.size(); i++) { if (i) std::cout << ","; std::cout << ids[i]; }
            std::cout << "], \"fallback\": " << fb << "}" << std::endl;
        }
        return 0;
    }
    if (outdir.empty() && emit_dir.empty()) { std::cerr << "--outdir required\n"; return 2; }
    if (gpt_bits != 4 && gpt_bits != 8) { std::cerr << "--gpt-bits must be 4 or 8\n"; return 2; }
    if (use_packed && gpt_bits != 4) { std::cerr << "--packed supports --gpt-bits 4 only (packed graphs are W4; W8 stays on the baked path)\n"; return 2; }
    if (opt_level != "all" && opt_level != "basic") { std::cerr << "--opt-level must be all or basic\n"; return 2; }
    if (engine != "ort" && engine != "free") { std::cerr << "--engine must be ort|free\n"; return 2; }
    bool engine_free = engine == "free";
#ifdef POTATO_FREE_ONLY
    if (!engine_free) { std::cerr << "this is a free-only build: --engine ort is not compiled in\n"; return 2; }
#endif
    if (fp32_graphs && !engine_free) { std::cerr << "--fp32-graphs requires --engine free\n"; return 2; }
    if (fp16_weights && !fp32_graphs) { std::cerr << "--fp16-weights requires --fp32-graphs\n"; return 2; }
    if (engine_free && !fp32_graphs && gpt_bits != 4) { std::cerr << "--engine free supports --gpt-bits 4 only (packed graphs are W4)\n"; return 2; }
    if (engine_free && !emit_dir.empty()) { std::cerr << "--emit-models is ORT-path only\n"; return 2; }
    std::string scheme = gpt_bits == 4 ? "w4" : "w8";

    try {
        auto t_boot = Clock::now();
        Json man, man_hybrid;
        if (!fp32_graphs) {
            man = JsonParser(read_text(root + "/wo-packs/manifest.json")).parse();
            man_hybrid = JsonParser(read_text(root + "/wo-packs/manifest_vits_w4hybrid.json")).parse();
        }
        // vits always loads the variant-E hybrid pack (enc_p + quantizer codebook
        // in W8, the rest W4; see VITS-W4.md). GPT packs follow --gpt-bits.
        struct GSpec { const char* tpl; const char* sec; const char* packfmt; const char* schemekey; const Json* manifest; };
        std::vector<GSpec> specs = {
            {"t2s_encoder_fp32.onnx", "gpt_encoder", "gpt_encoder_w%d.npz", nullptr, &man},
            {"t2s_first_stage_decoder_fp32.onnx", "gpt_first", "gpt_first_w%d.npz", nullptr, &man},
            {"t2s_stage_decoder_fp32.onnx", "gpt_stage", "gpt_stage_w%d.npz", nullptr, &man},
            {"vits_wnfold.onnx", "sovits_vits", "sovits_vits_w4hybrid.npz", "w4hybrid", &man_hybrid},
        };
#ifndef POTATO_FREE_ONLY
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "potato-tts-host");
        std::vector<SessionPack> sessions;
#endif
        // Packed GPT state (stage 3): one descriptor pool per graph, all
        // backed by ONE content-addressed BlobStore (see packed.h), plus
        // name-bound sessions for specs 1/2.
#ifndef POTATO_FREE_ONLY
        auto blob_store = std::make_shared<packed::BlobStore>();
        packed::WeightPool pool_f(blob_store), pool_s(blob_store), pool_v(blob_store);
        packed::NamedSession gpt_first_p, gpt_stage_p, vits_p;
        bool vits_packed = false;
        Ort::CustomOpDomain domain(nullptr);
        bool pool_ready = false;
#endif
        cpu::log_features("host");
        std::cerr << "[load] gpt_bits=" << gpt_bits << " threads=" << threads
                  << " packed=" << (use_packed ? 1 : 0) << " arena=" << (no_arena ? "off" : "on")
                  << " opt=" << opt_level << " engine=" << engine << "\n";
        // ---- free-engine load (--engine free) ----
        FreeHost fh;
        if (engine_free) {
            packed::set_kernel_threads(threads);
            if (fp32_graphs) {
                // Standard-precision path: compile the raw fp32 graphs
                // from <root>/graphs-fp32/ (the stage decoder file holds
                // the logits variant; see DESIGN.md). No packs, no
                // manifests, no packed lookups.
                auto compile = [&](const std::string& path, const std::string& label, bool f16 = false) {
                    auto t0 = Clock::now();
                    onnxm::Graph g = onnxm::load_model(path);
                    auto p = freeeng::Plan::compile(g, label, f16 && fp16_weights);
                    std::cerr << "[load] free plan " << label << " compiled in " << (int)ms_since(t0)
                              << " ms, " << p->step_count() << " steps; rss " << rss_kb() / 1024 << " MB\n";
                    return p;
                };
                fh.enc = compile(root + "/graphs-fp32/t2s_encoder_fp32.onnx", "gpt_encoder", true);
                fh.first = compile(root + "/graphs-fp32/t2s_first_stage_decoder_fp32.onnx", "gpt_first", true);
                fh.stage_a = compile(root + "/graphs-fp32/t2s_stage_decoder_fp32.onnx", "gpt_stage_a", true);
                fh.stage_b = compile(root + "/graphs-fp32/t2s_stage_decoder_fp32.onnx", "gpt_stage_b", true);
                fh.vits = compile(root + "/graphs-fp32/vits_fp32.onnx", "vits");
            } else {
                fh.store = std::make_shared<packed::BlobStore>();
                fh.pool_f = std::make_unique<FreePool>(fh.store);
                fh.pool_s = std::make_unique<FreePool>(fh.store);
                fh.pool_v = std::make_unique<FreePool>(fh.store);
                std::map<std::string, bool> filter;
                for (auto f : {"t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx"}) {
                    Json side = JsonParser(read_text(root + "/host/packed_graphs/" + f + ".json")).parse();
                    for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
                }
                fh.pool_f->filter = filter;
                fh.pool_f->load(root, man, "w4", "gpt_first");
                fh.pool_s->filter = filter;
                fh.pool_s->load(root, man, "w4", "gpt_stage");
                {
                    Json side = JsonParser(read_text(root + "/host/packed_graphs/vits_wnfold.onnx.json")).parse();
                    for (auto& pi : side.find("packed_inputs")->arr) fh.pool_v->filter[pi.find("tensor")->as_str()] = true;
                }
                fh.pool_v->load(root, man_hybrid, "w4hybrid", "sovits_vits");
                std::cerr << "[pool] free: first entries " << fh.pool_f->entries << "; stage entries "
                          << fh.pool_s->entries << "; vits entries " << fh.pool_v->entries
                          << "; store unique " << fh.store->unique_bytes() << " B; rss " << rss_kb() / 1024
                          << " MB\n";
                auto compile = [&](const std::string& path, const std::string& label) {
                    auto t0 = Clock::now();
                    onnxm::Graph g = onnxm::load_model(path);
                    auto p = freeeng::Plan::compile(g, label);
                    std::cerr << "[load] free plan " << label << " compiled in " << (int)ms_since(t0)
                              << " ms, " << p->step_count() << " steps; rss " << rss_kb() / 1024 << " MB\n";
                    return p;
                };
                fh.enc = compile(root + "/work/wo_graphs/w4/t2s_encoder_fp32.onnx", "gpt_encoder");
                fh.first = compile(root + "/host/packed_graphs/t2s_first_stage_decoder_fp32.onnx", "gpt_first");
                fh.first->set_packed_lookup(fh.pool_f->lookup());
                fh.stage_a = compile(root + "/host/packed_graphs/t2s_stage_decoder_fp32.onnx", "gpt_stage_a");
                fh.stage_a->set_packed_lookup(fh.pool_s->lookup());
                fh.stage_b = compile(root + "/host/packed_graphs/t2s_stage_decoder_fp32.onnx", "gpt_stage_b");
                fh.stage_b->set_packed_lookup(fh.pool_s->lookup());
                fh.vits_baked = vits_baked;
                if (vits_baked) {
                    auto t0 = Clock::now();
                    onnxm::Graph g = onnxm::load_model(root + "/work/folded/vits_wnfold.onnx");
                    const Json* secnode = man_hybrid.find("sovits_vits")->find("w4hybrid");
                    patch_graph_inplace(g, root, root + "/wo-packs/sovits_vits_w4hybrid.npz", *secnode);
                    fh.vits = freeeng::Plan::compile(g, "vits_baked");
                    std::cerr << "[load] free plan vits_baked compiled in " << (int)ms_since(t0) << " ms, "
                              << fh.vits->step_count() << " steps; rss " << rss_kb() / 1024 << " MB\n";
                } else {
                    fh.vits = compile(root + "/host/packed_graphs/vits_wnfold.onnx", "vits");
                    fh.vits->set_packed_lookup(fh.pool_v->lookup());
                    fh.vits_split = true;  // split graphs live next to the packed graph
                }
            }  // fp32_graphs / packed load paths
        }
#ifndef POTATO_FREE_ONLY
        int spec_idx = 0;
        for (auto& gs : specs) {
            if (engine_free) break;
            int idx = spec_idx++;
            if (use_packed && (idx == 1 || idx == 2)) {
                if (!pool_ready) {
                    packed::set_kernel_threads(threads);
                    std::map<std::string, bool> filter;
                    for (auto f : {"t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx"}) {
                        Json side = JsonParser(read_text(root + "/host/packed_graphs/" + f + ".json")).parse();
                        for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
                    }
                    pool_f.set_filter(filter);
                    pool_f.load(root, man, "w4", {"gpt_first"});
                    pool_s.set_filter(filter);
                    pool_s.load(root, man, "w4", {"gpt_stage"});
                    domain = packed::make_domain_full();
                    pool_ready = true;
                    std::cerr << "[pool] first: entries " << pool_f.entries()
                              << "; stage: entries " << pool_s.entries()
                              << "; store unique " << blob_store->unique_bytes()
                              << " B; logical " << pool_f.logical_bytes() + pool_s.logical_bytes()
                              << " B; rss " << rss_kb() / 1024 << " MB\n";
                }
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(ort_threads);
                if (no_arena) opts.DisableCpuMemArena();
                opts.SetGraphOptimizationLevel(opt_level == "basic" ? GraphOptimizationLevel::ORT_ENABLE_BASIC
                                                                    : GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.Add(domain);
                auto bytes = read_file(root + "/host/packed_graphs/" + gs.tpl);
                auto t0 = Clock::now();
                (idx == 1 ? gpt_first_p : gpt_stage_p).init(env, bytes.data(), bytes.size(), opts);
                std::cerr << "[load] packed session " << gs.tpl << " ready in " << (int)ms_since(t0)
                          << " ms; rss " << rss_kb() / 1024 << " MB\n";
                continue;
            }
            if (use_packed && idx == 3 && !vits_baked) {
                // vits packed residency (S10 step 2): E-pack tensors stay
                // packed in pool_v (shared store); the graph runs
                // potato::PackedConv / PackedConvTranspose / PackedGather.
                if (!pool_ready) { std::cerr << "internal: domain not ready for vits\n"; return 1; }
                Json side = JsonParser(read_text(root + "/host/packed_graphs/vits_wnfold.onnx.json")).parse();
                std::map<std::string, bool> filter;
                for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
                pool_v.set_filter(filter);
                pool_v.load(root, man_hybrid, "w4hybrid", {"sovits_vits"});
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(ort_threads);
                if (no_arena) opts.DisableCpuMemArena();
                opts.SetGraphOptimizationLevel(opt_level == "basic" ? GraphOptimizationLevel::ORT_ENABLE_BASIC
                                                                    : GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.Add(domain);
                auto bytes = read_file(root + "/host/packed_graphs/vits_wnfold.onnx");
                auto t0 = Clock::now();
                vits_p.init(env, bytes.data(), bytes.size(), opts);
                vits_packed = true;
                std::cerr << "[load] packed session vits_wnfold.onnx ready in " << (int)ms_since(t0)
                          << " ms; pool_v entries " << pool_v.entries() << "; store unique "
                          << blob_store->unique_bytes() << " B; rss " << rss_kb() / 1024 << " MB\n";
                continue;
            }
            char packname[64];
            if (strstr(gs.packfmt, "%d")) snprintf(packname, 64, gs.packfmt, gpt_bits);
            else snprintf(packname, 64, "%s", gs.packfmt);
            std::string rep;
            const Json* secnode = gs.manifest->find(gs.sec)->find(gs.schemekey ? gs.schemekey : scheme.c_str());
            auto patched = load_and_patch(root + "/work/folded/" + gs.tpl, root + "/wo-packs/" + packname,
                                          *secnode, gs.schemekey ? gs.schemekey : scheme.c_str(), rep);
            std::cerr << "[load] " << rep << "\n";
            if (!emit_dir.empty()) {
                write_file(emit_dir + "/patched_" + gs.tpl, (const uint8_t*)patched.data(), patched.size());
                continue;
            }
            auto t0 = Clock::now();
            sessions.push_back(make_session(env, patched, threads, no_arena, opt_level));
            std::cerr << "[load] session ready in " << (int)ms_since(t0) << " ms; rss " << rss_kb() / 1024 << " MB\n";
        }
#endif  // !POTATO_FREE_ONLY
        // In packed mode with baked vits, sessions holds [encoder, vits];
        // with vits packed residency it holds [encoder] only; baked mode
        // holds all 4. vits_sess is only dereferenced when !vits_packed
        // (and never in free mode, which has no ORT sessions at all).
#ifndef POTATO_FREE_ONLY
        SessionPack* vits_sess_p = nullptr;
        if (!engine_free) vits_sess_p = &(vits_packed ? sessions[0] : (use_packed ? sessions[1] : sessions[3]));
#endif
        if (!emit_dir.empty()) { std::cerr << "[emit] patched models written to " << emit_dir << "\n"; return 0; }

        G2P g2p;
        g2p.load(root + "/host/data");
        Json index_doc = JsonParser(read_text(root + "/host/refpacks/index.json")).parse();
        // Sidecar index for runtime-extracted packs saved with
        // "ref_save": tag -> {file, audio, text}. Kept apart
        // from index.json so the host never rewrites it.
        struct RtIndexEntry { std::string file, audio, text; };
        std::map<std::string, RtIndexEntry> rt_index;
        {
            std::ifstream f(root + "/host/refpacks/rt_index.json");
            if (f) {
                Json doc = JsonParser(std::string((std::istreambuf_iterator<char>(f)),
                                                  std::istreambuf_iterator<char>())).parse();
                if (doc.type == Json::Obj)
                    for (auto& kv : doc.obj) {
                        const Json* file = kv.second.find("file");
                        if (!file) continue;
                        RtIndexEntry e;
                        e.file = file->as_str();
                        if (const Json* a = kv.second.find("audio")) e.audio = a->as_str();
                        if (const Json* t = kv.second.find("text")) e.text = t->as_str();
                        rt_index[kv.first] = e;
                    }
            }
        }
        auto find_tag_file = [&](const std::string& tag) -> std::string {
            if (const Json* node = index_doc.find(tag)) return node->find("file")->as_str();
            auto it = rt_index.find(tag);
            return it == rt_index.end() ? "" : it->second.file;
        };
        auto persist_rt_index = [&]() {
            auto esc = [](const std::string& s) {
                std::string o;
                for (char c : s) {
                    if (c == '"' || c == '\\') { o += '\\'; o += c; }
                    else if ((uint8_t)c < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", c); o += b; }
                    else o += c;
                }
                return o;
            };
            std::string out = "{\n";
            bool first = true;
            for (auto& kv : rt_index) {
                if (!first) out += ",\n";
                first = false;
                out += " \"" + esc(kv.first) + "\": {\"file\": \"" + esc(kv.second.file) +
                       "\", \"audio\": \"" + esc(kv.second.audio) + "\", \"text\": \"" +
                       esc(kv.second.text) + "\"}";
            }
            out += "\n}\n";
            write_file(root + "/host/refpacks/rt_index.json", (const uint8_t*)out.data(), out.size());
        };
        std::cerr << "[load] g2p + refpack index ready (" << rt_index.size()
                  << " saved runtime refs); boot " << (int)ms_since(t_boot) << " ms; rss " << rss_kb() / 1024 << " MB\n";

#ifndef POTATO_FREE_ONLY
        Ort::AllocatorWithDefaultOptions alloc;
#endif
        std::map<std::string, std::map<std::string, RpkTensor>> refcache;
        auto get_ref = [&](const std::string& tag) -> const std::map<std::string, RpkTensor>& {
            auto it = refcache.find(tag);
            if (it != refcache.end()) return it->second;
            std::string file = find_tag_file(tag);
            if (file.empty()) throw std::runtime_error("tag not in index: " + tag);
            auto rp = load_rpk(root + "/host/refpacks/" + file);
            refcache[tag] = std::move(rp);
            return refcache[tag];
        };

        // Runtime-extracted references (M3-3d): lazily
        // initialized on the first ref_wav request; results
        // share the pack cache under an "rt:<hash>" key.
        RefExtractor rt_ext;
        bool rt_ext_ready = false, rt_ext_failed = false;
        std::set<std::string> rt_saved;
        auto get_ref_rt = [&](const std::string& wav, const std::string& ptext, bool save)
            -> const std::map<std::string, RpkTensor>& {
            auto wb = read_file(wav);
            uint64_t h = 1469598103934665603ULL;
            auto mix = [&](const uint8_t* p, size_t n) {
                for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
            };
            mix(wb.data(), wb.size());
            mix((const uint8_t*)ptext.data(), ptext.size());
            char key[40];
            snprintf(key, sizeof key, "rt:%016llx", (unsigned long long)h);
            auto it = refcache.find(key);
            if (it == refcache.end()) {
                if (!engine_free) throw std::runtime_error("ref_wav requires --engine free");
                if (!rt_ext_ready) {
                    if (rt_ext_failed) throw std::runtime_error("runtime reference extraction unavailable");
                    std::string err;
                    if (!rt_ext.init(root + "/host/refextract", err)) {
                        rt_ext_failed = true;
                        throw std::runtime_error("refextract init: " + err);
                    }
                    rt_ext_ready = true;
                }
                auto t0 = Clock::now();
                auto tensors = rt_ext.extract(wav, ptext);
                std::cerr << "[ref] runtime reference " << key << " extracted in " << (int)ms_since(t0)
                          << " ms; rss " << rss_kb() / 1024 << " MB\n";
                refcache[key] = std::move(tensors);
                it = refcache.find(key);
            }
            if (save && !rt_saved.count(key)) {
                std::string file = std::string(key + 3) + ".rpk";  // strip "rt:"
                write_rpk(root + "/host/refpacks/" + file, it->second);
                RtIndexEntry e;
                e.file = file;
                e.audio = wav.substr(wav.find_last_of('/') + 1);
                e.text = ptext;
                rt_index[key] = e;
                persist_rt_index();
                rt_saved.insert(key);
                std::cerr << "[ref] saved runtime reference as pack " << file << " (tag " << key << ")\n";
            }
            return it->second;
        };

        const int SR = 32000, EOS_ID = 1024, VOCAB = 1025, MAX_STEPS = 500;
        std::string line;
        long req_no = 0;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            req_no++;
            auto t_req = Clock::now();
            try {
                Json r = JsonParser(line).parse();
                auto getstr = [&](const char* k, const std::string& d) { const Json* j = r.find(k); return j ? j->as_str(d) : d; };
                auto getnum = [&](const char* k, double d) { const Json* j = r.find(k); return j ? j->as_num(d) : d; };
                std::string text = getstr("text", "");
                std::string tag = getstr("tag", "neutral1");
                std::string ref_wav = getstr("ref_wav", "");
                std::string ref_text = getstr("ref_text", "");
                double speed = getnum("speed", 1.0);
                (void)speed;
                int top_k = (int)getnum("top_k", 20);
                double top_p = getnum("top_p", 0.9);
                double temperature = getnum("temperature", 0.85);
                long long seed = (long long)getnum("seed", 42);
                const Json* gj = r.find("greedy");
                bool greedy = gj && gj->type == Json::Bool && gj->b;
                std::vector<std::string> notes;
                if (r.find("speed")) notes.push_back("speed accepted but not applied: no graph input carries it");
                const Json* rsj = r.find("ref_save");
                bool ref_save = rsj && rsj->type == Json::Bool && rsj->b;
                const std::map<std::string, RpkTensor>* refp = nullptr;
                std::string tag_used = tag;
                if (!ref_wav.empty()) {
                    refp = &get_ref_rt(ref_wav, ref_text, ref_save);
                    tag_used = "(runtime)";
                    notes.push_back("runtime-extracted reference from " + ref_wav +
                                    (ref_save ? ", saved as a pack" : ""));
                } else {
                    if (find_tag_file(tag).empty()) { tag_used = "neutral1"; notes.push_back("tag '" + tag + "' missing, fell back to neutral1"); }
                    refp = &get_ref(tag_used);
                }
                const auto& ref = *refp;

                // ---- G2P (shell behavior: prepend ".") ----
                auto t0 = Clock::now();
                int fb = 0;
                auto text_ids = g2p.text_to_ids("." + text, fb);
                double t_g2p = ms_since(t0);
                int64_t L = text_ids.size();
                std::vector<float> text_bert((size_t)L * 1024, 0.0f);
                const auto& ref_seq = ref.at("ref_seq");
                const auto& ref_bert = ref.at("ref_bert");
                const auto& ssl = ref.at("ssl_content");

                // ---- GPT encoder ----
                t0 = Clock::now();
#ifndef POTATO_FREE_ONLY
                std::vector<Ort::Value> enc_out;
#endif
                std::map<std::string, freeeng::Tensor> fenc_out;
                int64_t plen = 0;
                if (engine_free) {
                    std::map<std::string, freeeng::Tensor> ein;
                    ein.emplace("ref_seq", ft_from_rpk(ref_seq));
                    freeeng::Tensor tt; tt.dtype = 7; tt.data = (const uint8_t*)text_ids.data(); tt.shape = {1, L};
                    ein.emplace("text_seq", tt);
                    ein.emplace("ref_bert", ft_from_rpk(ref_bert));
                    freeeng::Tensor tb; tb.dtype = 1; tb.data = (const uint8_t*)text_bert.data(); tb.shape = {L, 1024};
                    ein.emplace("text_bert", tb);
                    ein.emplace("ssl_content", ft_from_rpk(ssl));
                    fenc_out = fh.enc->run(ein);
                    plen = fenc_out.at("prompts").shape[1];
                }
#ifndef POTATO_FREE_ONLY
                else {
                    std::vector<Ort::Value> enc_in;
                    enc_in.push_back(i64_tensor(alloc, ref_seq.i64(), ref_seq.shape));
                    enc_in.push_back(i64_tensor(alloc, text_ids.data(), {1, L}));
                    enc_in.push_back(f32_tensor(alloc, ref_bert.f32(), ref_bert.shape));
                    enc_in.push_back(f32_tensor(alloc, text_bert.data(), {L, 1024}));
                    enc_in.push_back(f32_tensor(alloc, ssl.f32(), ssl.shape));
                    enc_out = sessions[0].session->Run(Ort::RunOptions{nullptr}, sessions[0].in_ptrs.data(), enc_in.data(), enc_in.size(), sessions[0].out_ptrs.data(), sessions[0].out_names.size());
                    plen = enc_out[1].GetTensorTypeAndShapeInfo().GetShape()[1];
                }
#endif
                double t_enc = ms_since(t0);

                // ---- first stage + decode loop ----
                std::vector<int64_t> gen;
                int steps = 0;
                double t_first = 0, t_dec = 0;
                if (engine_free) {
                    // Name-bound protocol on free plans (same as
                    // tf_check_free): stage state ping-pongs between
                    // two Plan instances so the previous step's outputs
                    // feed back as inputs without copies.
                    t0 = Clock::now();
                    std::map<std::string, freeeng::Tensor> fin;
                    fin.emplace("x", fenc_out.at("x"));
                    fin.emplace("prompts", fenc_out.at("prompts"));
                    auto fstate = fh.first->run(fin);
                    t_first = ms_since(t0);
                    fh.enc->release_arena();  // x/prompts were consumed by the first run
                    {
                        auto& y0 = fstate.at("y");
                        const int64_t* yd = (const int64_t*)y0.data;
                        gen.push_back(yd[y0.shape[1] - 1]);
                    }
                    t0 = Clock::now();
                    std::mt19937_64 rng((uint64_t)seed);
                    // Indexed decode: per-plan slot permutations
                    // (built once) replace the per-step name maps.
                    // first_out -> stage_in renames y->iy,
                    // y_emb->iy_emb, present_*->past_*; stage outputs
                    // feed the other stage plan the same way.
                    auto slot_of = [](const std::vector<std::string>& names, const std::string& n) {
                        for (size_t i = 0; i < names.size(); i++)
                            if (names[i] == n) return (int)i;
                        return -1;
                    };
                    auto feed_perm = [&](const std::vector<std::string>& src_out,
                                         const std::vector<std::string>& dst_in) {
                        std::vector<int> perm(dst_in.size(), -1);
                        for (size_t i = 0; i < dst_in.size(); i++) {
                            const std::string& dn = dst_in[i];
                            std::string sn = dn;
                            if (dn == "iy") sn = "y";
                            else if (dn == "iy_emb") sn = "y_emb";
                            else if (dn.rfind("past_k_", 0) == 0) sn = "present_k_" + dn.substr(7);
                            else if (dn.rfind("past_v_", 0) == 0) sn = "present_v_" + dn.substr(7);
                            perm[i] = slot_of(src_out, sn);
                            if (perm[i] < 0) throw std::runtime_error("stage feed: no source for " + dn);
                        }
                        return perm;
                    };
                    // Stateful KV: one shared cache for both stage
                    // plans, fresh per request. The first stage run
                    // receives the real past tensors (copied in);
                    // later runs pass them as missing and the plans
                    // read/append the cache buffers directly.
                    static const bool use_kv = std::getenv("POTATO_NO_KV") == nullptr;
                    auto kvc = use_kv ? freeeng::make_kv_cache() : nullptr;
                    if (kvc) {
                        fh.stage_a->set_kv_cache(kvc);
                        fh.stage_b->set_kv_cache(kvc);
                    }
                    auto past_mask_of = [](const std::vector<std::string>& names) {
                        std::vector<char> m(names.size(), 0);
                        for (size_t i = 0; i < names.size(); i++)
                            m[i] = names[i].rfind("past_", 0) == 0;
                        return m;
                    };
                    const auto& a_in = fh.stage_a->input_names();
                    const auto& b_in = fh.stage_b->input_names();
                    std::vector<char> past_mask_a = past_mask_of(a_in);
                    std::vector<char> past_mask_b = past_mask_of(b_in);
                    const auto& a_out = fh.stage_a->output_names();
                    const auto& b_out = fh.stage_b->output_names();
                    std::vector<int> perm_first_a = feed_perm(fh.first->output_names(), a_in);
                    std::vector<int> perm_ab = feed_perm(a_out, b_in);
                    std::vector<int> perm_ba = feed_perm(b_out, a_in);
                    int y_slot_a = slot_of(a_out, "y"), y_slot_b = slot_of(b_out, "y");
                    int lg_slot_a = slot_of(a_out, "logits_penalized");
                    int lg_slot_b = slot_of(b_out, "logits_penalized");
                    if (y_slot_a < 0 || y_slot_b < 0 || lg_slot_a < 0 || lg_slot_b < 0)
                        throw std::runtime_error("stage outputs missing y/logits_penalized");
                    std::vector<freeeng::Tensor> state(a_in.size());
                    {
                        // fstate is in fh.first's output order.
                        const auto& fo = fh.first->output_names();
                        std::vector<freeeng::Tensor> fvec(fo.size());
                        for (size_t i = 0; i < fo.size(); i++) fvec[i] = fstate.at(fo[i]);
                        for (size_t i = 0; i < state.size(); i++) state[i] = fvec[perm_first_a[i]];
                    }
                    bool use_a = true;
                    while (gen.back() != EOS_ID && steps < MAX_STEPS) {
                        std::vector<freeeng::Tensor> out =
                            (use_a ? fh.stage_a : fh.stage_b)->run_vec(state);
                        steps++;
                        const auto& lg = out[use_a ? lg_slot_a : lg_slot_b];
                        const float* logits = (const float*)lg.data;
                        size_t lnum = (size_t)lg.numel();
                        const float* last_row = logits + (lnum - VOCAB);
                        int64_t nxt = sample_token(last_row, VOCAB, top_k, top_p, temperature, rng, greedy);
                        auto& yv = out[use_a ? y_slot_a : y_slot_b];
                        int64_t* yd = (int64_t*)yv.data;
                        yd[yv.shape[1] - 1] = nxt;
                        gen.push_back(nxt);
                        const std::vector<int>& perm = use_a ? perm_ab : perm_ba;
                        const std::vector<char>& pmask = use_a ? past_mask_b : past_mask_a;
                        state.assign(perm.size(), freeeng::Tensor{});
                        for (size_t i = 0; i < perm.size(); i++)
                            state[i] = (use_kv && pmask[i]) ? freeeng::Tensor{} : out[perm[i]];
                        use_a = !use_a;
                    }
                    t_dec = ms_since(t0);
                    fh.first->release_arena();  // stage plans own the live state now
                }
#ifndef POTATO_FREE_ONLY
                else if (use_packed) {
                    // Name-bound protocol (same as tf_check): state tensors are
                    // carried by graph input/output name, weights come from
                    // each graph's own pool as zero-copy views on every Run.
                    t0 = Clock::now();
                    std::map<std::string, Ort::Value> pstate;
                    {
                        std::map<std::string, Ort::Value> fin;
                        std::vector<std::string> want;
                        for (auto& n : gpt_first_p.in_names) if (!pool_f.has(n)) want.push_back(n);
                        if (want.size() != 2) throw std::runtime_error("packed first stage: unexpected state inputs");
                        fin.emplace(want[0], std::move(enc_out[0]));
                        fin.emplace(want[1], std::move(enc_out[1]));
                        pstate = gpt_first_p.run(fin, &pool_f);
                    }
                    t_first = ms_since(t0);
                    {
                        auto& y0 = pstate.at(gpt_first_p.out_names[0]);
                        auto* yd = y0.GetTensorMutableData<int64_t>();
                        auto ys = y0.GetTensorTypeAndShapeInfo().GetShape();
                        gen.push_back(yd[ys[1] - 1]);
                    }
                    t0 = Clock::now();
                    std::mt19937_64 rng((uint64_t)seed);
                    const std::string& yname = gpt_stage_p.out_names[0];
                    while (gen.back() != EOS_ID && steps < MAX_STEPS) {
                        // Stage graph input names differ from the state/output
                        // names: iy <- y, iy_emb <- y_emb, past_* <- present_*
                        // (same rename tf_check uses; baked mode hides it by
                        // feeding positionally).
                        std::map<std::string, Ort::Value> sin;
                        sin["iy"] = std::move(pstate.at("y"));
                        sin["iy_emb"] = std::move(pstate.at("y_emb"));
                        for (auto& [n, v] : pstate) {
                            if (n.rfind("present_k_", 0) == 0) sin["past_k_" + n.substr(10)] = std::move(v);
                            else if (n.rfind("present_v_", 0) == 0) sin["past_v_" + n.substr(10)] = std::move(v);
                        }
                        pstate.clear();
                        pstate = gpt_stage_p.run(sin, &pool_s);
                        steps++;
                        auto& lg = pstate.at("logits_penalized");
                        const float* logits = lg.GetTensorData<float>();
                        size_t lnum = lg.GetTensorTypeAndShapeInfo().GetElementCount();
                        const float* last_row = logits + (lnum - VOCAB);
                        int64_t nxt = sample_token(last_row, VOCAB, top_k, top_p, temperature, rng, greedy);
                        auto& yv = pstate.at(yname);
                        auto* yd = yv.GetTensorMutableData<int64_t>();
                        auto ys = yv.GetTensorTypeAndShapeInfo().GetShape();
                        yd[ys[1] - 1] = nxt;
                        gen.push_back(nxt);
                    }
                    t_dec = ms_since(t0);
                } else {
                    t0 = Clock::now();
                    std::vector<Ort::Value> first_in;
                    first_in.push_back(std::move(enc_out[0]));
                    first_in.push_back(std::move(enc_out[1]));
                    auto cur = sessions[1].session->Run(Ort::RunOptions{nullptr}, sessions[1].in_ptrs.data(), first_in.data(), first_in.size(), sessions[1].out_ptrs.data(), sessions[1].out_names.size());
                    t_first = ms_since(t0);
                    // cur: [y, y_emb, kv x48]
                    {
                        auto* yd = cur[0].GetTensorMutableData<int64_t>();
                        auto ys = cur[0].GetTensorTypeAndShapeInfo().GetShape();
                        gen.push_back(yd[ys[1] - 1]);
                    }
                    t0 = Clock::now();
                    std::mt19937_64 rng((uint64_t)seed);
                    while (gen.back() != EOS_ID && steps < MAX_STEPS) {
                        std::vector<Ort::Value> st_in;
                        st_in.reserve(cur.size());
                        for (auto& v : cur) st_in.push_back(std::move(v));
                        auto out = sessions[2].session->Run(Ort::RunOptions{nullptr}, sessions[2].in_ptrs.data(), st_in.data(), st_in.size(), sessions[2].out_ptrs.data(), sessions[2].out_names.size());
                        steps++;
                        // out: [y, y_emb, stop, kv x48, logits]
                        // logits_penalized is a flat [1025] vector (last position only).
                        const float* logits = out.back().GetTensorData<float>();
                        size_t lnum = out.back().GetTensorTypeAndShapeInfo().GetElementCount();
                        const float* last_row = logits + (lnum - VOCAB);
                        int64_t nxt = sample_token(last_row, VOCAB, top_k, top_p, temperature, rng, greedy);
                        auto* yd = out[0].GetTensorMutableData<int64_t>();
                        auto ys = out[0].GetTensorTypeAndShapeInfo().GetShape();
                        yd[ys[1] - 1] = nxt;
                        gen.push_back(nxt);
                        cur.clear();
                        cur.push_back(std::move(out[0]));
                        cur.push_back(std::move(out[1]));
                        for (size_t k = 3; k + 1 < out.size(); k++) cur.push_back(std::move(out[k]));  // skip stop (idx2) and logits (last)
                    }
                    t_dec = ms_since(t0);
                }
#endif

                // pred_semantic = tokens before first EOS
                std::vector<int64_t> pred;
                for (auto tk : gen) { if (tk >= EOS_ID) break; pred.push_back(tk); }
                if (pred.empty()) throw std::runtime_error("empty semantic sequence");

                // Streaming (free engine, packed vits only):
                // "stream": true runs the vits front once for the
                // latent z, then synthesizes the purely convolutional
                // decoder (split graphs, tools/split_vits.py) in
                // chunks of chunk_tokens tokens, each chunk appended
                // to reqNNN.pcm (s16le raw vits output, pre
                // output-chain) as it lands, with one JSON line per
                // chunk on stdout. Slicing the semantic input itself
                // is unsound (the front aligns text and semantic
                // globally), but the decoder is exactly local:
                // chunk_ctx tokens of latent context on both sides
                // (16 frames at the default) reproduce the one-shot
                // audio bit-exactly (vits_slice_check). The final WAV
                // still goes through the normal output chain, built
                // from the concatenated chunk audio.
                const Json* stj = r.find("stream");
                bool want_stream = stj && stj->type == Json::Bool && stj->b;
                const Json* ctj = r.find("chunk_tokens");
                int64_t chunk_tokens = ctj ? (int64_t)ctj->as_num(25) : 25;
                const Json* ccj = r.find("chunk_ctx");
                int64_t chunk_ctx = ccj ? (int64_t)ccj->as_num(8) : 8;
                if (chunk_tokens < 1) chunk_tokens = 25;
                if (chunk_ctx < 0) chunk_ctx = 0;
                bool streamed = false;
                double first_audio_ms = -1;
                int n_chunks = 0;

                // ---- vits ----
                t0 = Clock::now();
#ifndef POTATO_FREE_ONLY
                std::vector<Ort::Value> vits_in;
                vits_in.push_back(i64_tensor(alloc, text_ids.data(), {1, L}));
                vits_in.push_back(i64_tensor(alloc, pred.data(), {1, 1, (int64_t)pred.size()}));
                vits_in.push_back(f32_tensor(alloc, ref.at("ge").f32(), ref.at("ge").shape));
                vits_in.push_back(f32_tensor(alloc, ref.at("ge_advanced").f32(), ref.at("ge_advanced").shape));
                std::vector<Ort::Value> vout;
                Ort::Value vout_packed(nullptr);
#endif
                std::vector<float> audio_free;
                const float* audio = nullptr;
                size_t n_audio = 0;
                if (engine_free) {
                    auto make_vits_feed = [&]() {
                        std::map<std::string, freeeng::Tensor> vin;
                        freeeng::Tensor vt; vt.dtype = 7; vt.data = (const uint8_t*)text_ids.data(); vt.shape = {1, L};
                        vin.emplace("text_seq", vt);
                        freeeng::Tensor vp; vp.dtype = 7; vp.data = (const uint8_t*)pred.data(); vp.shape = {1, 1, (int64_t)pred.size()};
                        vin.emplace("pred_semantic", vp);
                        vin.emplace("ge", ft_from_rpk(ref.at("ge")));
                        vin.emplace("ge_advanced", ft_from_rpk(ref.at("ge_advanced")));
                        return vin;
                    };
                    auto run_vits_full = [&]() {
                        auto vout_f = fh.vits->run(make_vits_feed());
                        std::vector<float> out = ft_to_vec(vout_f.at("audio"));
                        fh.vits->release_arena();  // audio is copied out; drop the vits peak
                        return out;
                    };
                    const int64_t T = (int64_t)pred.size();
                    if (want_stream && fh.vits_split && T > chunk_tokens) {
                        try {
                            if (!fh.vits_front) {
                                fh.vits_front = compile_plan_file(root + "/host/packed_graphs/vits_front.onnx", "vits_front");
                                fh.vits_front->set_packed_lookup(fh.pool_v->lookup());
                                fh.vits_dec = compile_plan_file(root + "/host/packed_graphs/vits_dec.onnx", "vits_dec");
                                fh.vits_dec->set_packed_lookup(fh.pool_v->lookup());
                            }
                            const std::string ZN = "/vq_model/Mul_2_output_0";
                            auto fout = fh.vits_front->run(make_vits_feed());
                            const freeeng::Tensor& zt = fout.at(ZN);
                            if (zt.dtype != 1 || zt.shape.size() != 3)
                                throw std::runtime_error("vits front: unexpected latent shape");
                            const int64_t C = zt.shape[1], F = zt.shape[2];
                            std::vector<float> z = ft_to_vec(zt);
                            fh.vits_front->release_arena();
                            if (F % T) throw std::runtime_error("vits front: latent frames not a multiple of tokens");
                            const int64_t fpt = F / T;              // latent frames per token (2)
                            const int64_t spf = 1280 / fpt;         // samples per latent frame (640)
                            const int64_t chunk_f = chunk_tokens * fpt;
                            const int64_t ctx_f = chunk_ctx * fpt;
                            char pcmname[64];
                            snprintf(pcmname, 64, "req%03ld.pcm", req_no);
                            std::string pcmpath = outdir + "/" + pcmname;
                            std::ofstream pcm(pcmpath, std::ios::binary);
                            if (!pcm) throw std::runtime_error("cannot open stream file " + pcmpath);
                            audio_free.reserve((size_t)T * 1280);
                            std::vector<int16_t> s16;
                            std::vector<float> zslice;
                            for (int64_t s = 0; s < F; s += chunk_f) {
                                int64_t e = std::min(F, s + chunk_f);
                                int64_t cs = std::max<int64_t>(0, s - ctx_f);
                                int64_t ce = std::min(F, e + ctx_f);
                                // z is [1, C, F] channel-major: a frame
                                // slice is a per-channel row copy.
                                zslice.assign((size_t)(ce - cs) * C, 0);
                                for (int64_t c = 0; c < C; c++)
                                    std::copy(z.begin() + (size_t)c * F + cs, z.begin() + (size_t)c * F + ce,
                                              zslice.begin() + (size_t)c * (ce - cs));
                                std::map<std::string, freeeng::Tensor> din;
                                freeeng::Tensor zv; zv.dtype = 1; zv.data = (const uint8_t*)zslice.data(); zv.shape = {1, C, ce - cs};
                                din.emplace(ZN, zv);
                                din.emplace("ge", ft_from_rpk(ref.at("ge")));
                                auto dout = fh.vits_dec->run(din);
                                std::vector<float> raw = ft_to_vec(dout.at("audio"));
                                fh.vits_dec->release_arena();
                                size_t off = (size_t)(s - cs) * spf;
                                size_t len = (size_t)(e - s) * spf;
                                if (raw.size() != (size_t)(ce - cs) * spf)
                                    throw std::runtime_error("vits dec chunk length off the spf mapping");
                                s16.resize(len);
                                for (size_t i = 0; i < len; i++) {
                                    double v = raw[off + i];
                                    if (v > 1.0) v = 1.0; else if (v < -1.0) v = -1.0;
                                    s16[i] = (int16_t)std::lrint(v * 32767.0);
                                }
                                pcm.write((const char*)s16.data(), (std::streamsize)(len * 2));
                                pcm.flush();
                                audio_free.insert(audio_free.end(), raw.begin() + off, raw.begin() + off + len);
                                n_chunks++;
                                if (first_audio_ms < 0) first_audio_ms = ms_since(t_req);
                                std::cout << "{\"chunk\": " << (n_chunks - 1) << ", \"file\": \"" << pcmpath
                                          << "\", \"samples_total\": " << audio_free.size()
                                          << ", \"elapsed_ms\": " << (long long)ms_since(t_req) << "}" << std::endl;
                            }
                            streamed = true;
                        } catch (const std::exception& e) {
                            notes.push_back(std::string("stream failed, one-shot used: ") + e.what());
                            audio_free.clear();
                            n_chunks = 0;
                            first_audio_ms = -1;
                        }
                    }
                    if (!streamed) {
                        if (want_stream && !fh.vits_split)
                            notes.push_back("stream not used: split vits graphs only exist on the packed free path");
                        else if (want_stream && T <= chunk_tokens)
                            notes.push_back("stream not used: sequence fits one chunk");
                        audio_free = run_vits_full();
                    }
                    audio = audio_free.data();
                    n_audio = audio_free.size();
                }
#ifndef POTATO_FREE_ONLY
                else if (vits_packed) {
                    std::map<std::string, Ort::Value> vin;
                    vin.emplace("text_seq", std::move(vits_in[0]));
                    vin.emplace("pred_semantic", std::move(vits_in[1]));
                    vin.emplace("ge", std::move(vits_in[2]));
                    vin.emplace("ge_advanced", std::move(vits_in[3]));
                    auto pout = vits_p.run(vin, &pool_v);
                    auto it = pout.find("audio");
                    if (it == pout.end()) throw std::runtime_error("packed vits: no audio output");
                    vout_packed = std::move(it->second);
                    audio = vout_packed.GetTensorData<float>();
                    n_audio = vout_packed.GetTensorTypeAndShapeInfo().GetElementCount();
                } else {
                    vout = vits_sess_p->session->Run(Ort::RunOptions{nullptr}, vits_sess_p->in_ptrs.data(), vits_in.data(), vits_in.size(), vits_sess_p->out_ptrs.data(), 1);
                    audio = vout[0].GetTensorData<float>();
                    n_audio = vout[0].GetTensorTypeAndShapeInfo().GetElementCount();
                }
#endif
                double t_vits = ms_since(t0);

                // Output stage (DEESS.md v2): deess2air before silence trimming.
                // Per-request field "deess" (default true) toggles it.
                const Json* dj = r.find("deess");
                bool use_deess = !(dj && dj->type == Json::Bool && !dj->b);
                std::vector<float> processed;
                const float* audio_out = audio;
                double t_deess = 0;
                if (use_deess) {
                    processed.assign(audio, audio + n_audio);
                    auto td = Clock::now();
                    deess::deess2air(processed.data(), n_audio, SR, processed.data());
                    t_deess = ms_since(td);
                    audio_out = processed.data();
                }

                auto [lo, hi] = trim_range(audio_out, n_audio, 1.3, SR);

                // Final stage (chain amendment): tanh soft clip on the trimmed
                // segment, RMS-preserving; per sentence, no carried state.
                // Request field "softclip" (default true) toggles it for A/B.
                const Json* sj = r.find("softclip");
                bool use_softclip = !(sj && sj->type == Json::Bool && !sj->b);
                std::vector<float> seg(audio_out + lo, audio_out + hi);
                deess::LevelStats lv{};
                if (use_softclip) {
                    lv = deess::softclip(seg.data(), seg.size());
                } else if (!seg.empty()) {
                    double pk = 0, acc = 0;
                    for (float v : seg) { double a = std::fabs((double)v); if (a > pk) pk = a; acc += (double)v * v; }
                    lv.peak_in = lv.peak_out = pk;
                    lv.rms_in = lv.rms_out = std::sqrt(acc / (double)seg.size());
                }
                auto crest_db = [](double pk, double rms) { return rms > 0 ? 20.0 * std::log10(pk / rms) : 0.0; };
                auto rms_db = [](double rms) { return rms > 0 ? 20.0 * std::log10(rms) : -120.0; };

                char fname[64];
                snprintf(fname, 64, "req%03ld.wav", req_no);
                std::string fpath = outdir + "/" + fname;
                write_wav(fpath, seg.data(), seg.size(), SR);

                std::ostringstream os;
                os << "{\"file\": \"" << fpath << "\", \"duration_s\": " << (hi - lo) / (double)SR
                   << ", \"elapsed_ms\": " << (long long)ms_since(t_req)
                   << ", \"steps\": " << steps << ", \"tokens\": [";
                for (size_t i = 0; i < gen.size(); i++) { if (i) os << ","; os << gen[i]; }
                os << "], \"tag_used\": \"" << tag_used << "\", \"g2p_fallback\": " << fb
                   << ", \"timing_ms\": {\"g2p\": " << t_g2p << ", \"encoder\": " << t_enc
                   << ", \"first\": " << t_first << ", \"decode\": " << t_dec
                   << ", \"decode_per_step\": " << (steps ? t_dec / steps : 0) << ", \"vits\": " << t_vits
                   << ", \"deess\": " << t_deess
                   << ", \"total\": " << ms_since(t_req) << "}, \"deess_applied\": " << (use_deess ? "true" : "false")
                   << ", \"streamed\": " << (streamed ? "true" : "false")
                   << ", \"first_audio_ms\": " << (long long)first_audio_ms
                   << ", \"chunks\": " << n_chunks
                   << ", \"softclip_applied\": " << (use_softclip ? "true" : "false")
                   << ", \"levels\": {\"peak_in\": " << lv.peak_in << ", \"peak_out\": " << lv.peak_out
                   << ", \"rms_in_db\": " << rms_db(lv.rms_in) << ", \"rms_out_db\": " << rms_db(lv.rms_out)
                   << ", \"crest_in_db\": " << crest_db(lv.peak_in, lv.rms_in)
                   << ", \"crest_out_db\": " << crest_db(lv.peak_out, lv.rms_out)
                   << ", \"drive\": " << lv.drive << "}"
                   << ", \"rss_mb\": " << rss_kb() / 1024.0
                   << ", \"notes\": [";
                for (size_t i = 0; i < notes.size(); i++) { if (i) os << ", "; os << "\"" << json_escape(notes[i]) << "\""; }
                os << "]}";
                std::cout << os.str() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "{\"error\": \"" << json_escape(e.what()) << "\"}" << std::endl;
            }
        }
        if (engine_free) {
            auto mb = [](size_t b) { return b / 1048576.0; };
            std::cerr << "[free] final arenas MB: enc " << mb(fh.enc->arena_bytes()) << ", first "
                      << mb(fh.first->arena_bytes()) << ", stage_a " << mb(fh.stage_a->arena_bytes())
                      << ", stage_b " << mb(fh.stage_b->arena_bytes()) << ", vits "
                      << mb(fh.vits->arena_bytes()) << "; rss " << rss_kb() / 1024 << " MB\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
