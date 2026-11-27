// C++ port of g2pM (the polyphone predictor Genie's Chinese
// G2P uses): a cedict lookup for single-pronunciation chars
// plus a one-layer BiLSTM (hidden 32, embedding 64) with a
// two-layer FC head that picks the pronunciation class of
// polyphonic chars from context. Mirrors g2pM.py's forward
// pass (gate order i|f|g|o, both directions) and its
// __call__(char_split=True) flow.
//
// Assets come from tools-exported g2pm.bin (see the header
// comment in the export script): cedict, char2idx, idx2class
// and the float32 weights. Validated by g2pm_check against
// the original on the reference-text corpus.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "wordpiece.h"  // utf8 helpers

namespace g2pm {

class G2pM {
  public:
    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        std::vector<uint8_t> buf;
        uint8_t chunk[65536];
        size_t n;
        while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
        fclose(f);
        size_t p = 0;
        auto rd = [&](void* dst, size_t k) {
            if (p + k > buf.size()) return false;
            std::memcpy(dst, buf.data() + p, k);
            p += k;
            return true;
        };
        auto u32 = [&]() { uint32_t v = 0; rd(&v, 4); return v; };
        auto u16 = [&]() { uint16_t v = 0; rd(&v, 2); return v; };
        auto u8 = [&]() { uint8_t v = 0; rd(&v, 1); return v; };
        auto str = [&]() {
            uint16_t len = u16();
            std::string s(len, '\0');
            if (len) rd(&s[0], len);
            return s;
        };
        char magic[4];
        if (!rd(magic, 4) || std::memcmp(magic, "G2PM", 4) != 0) return false;
        if (u32() != 1) return false;
        uint32_t nc = u32();
        for (uint32_t i = 0; i < nc; i++) {
            uint32_t cp = u32();
            uint8_t np = u8();
            auto& v = cedict_[cp];
            for (uint8_t k = 0; k < np; k++) v.push_back(str());
        }
        uint32_t ni = u32();
        for (uint32_t i = 0; i < ni; i++) {
            std::string k = str();
            int32_t v = 0;
            rd(&v, 4);
            char2idx_[k] = v;
        }
        uint32_t ncl = u32();
        idx2class_.resize(ncl);
        for (uint32_t i = 0; i < ncl; i++) idx2class_[i] = str();
        uint32_t nw = u32();
        for (uint32_t i = 0; i < nw; i++) {
            std::string name = str();
            uint8_t dt = u8();
            uint8_t nd = u8();
            std::vector<uint32_t> dims(nd);
            size_t cnt = 1;
            for (uint8_t k = 0; k < nd; k++) { dims[k] = u32(); cnt *= dims[k]; }
            Tensor t;
            t.dims = dims;
            t.data.resize(cnt);
            if (dt == 0) {
                if (!rd(t.data.data(), cnt * 4)) return false;
            } else {
                std::vector<double> tmp(cnt);
                if (!rd(tmp.data(), cnt * 8)) return false;
                for (size_t k = 0; k < cnt; k++) t.data[k] = (float)tmp[k];
            }
            weights_[name] = std::move(t);
        }
        return p == buf.size();
    }

    // Per-character pronunciations, mirroring
    // G2pM.__call__(sent, char_split=True) with tone=True.
    std::vector<std::string> operator()(const std::string& text) const {
        std::vector<uint32_t> cps = wptok::utf8_decode(text);
        std::vector<std::string> pros(cps.size());
        std::vector<int> ids(cps.size());
        std::vector<int> poly;
        for (size_t i = 0; i < cps.size(); i++) {
            std::string ch;
            wptok::utf8_encode(cps[i], ch);
            auto ii = char2idx_.find(ch);
            ids[i] = ii != char2idx_.end() ? ii->second : char2idx_.at("<UNK>");
            auto ci = cedict_.find(cps[i]);
            if (ci != cedict_.end()) {
                if (ci->second.size() > 1) {
                    poly.push_back((int)i);
                    pros[i] = "\xE2\x96\x81";  // SPLIT_TOKEN placeholder
                } else {
                    pros[i] = ci->second[0];
                }
            } else {
                pros[i] = ch;
            }
        }
        if (!poly.empty()) {
            std::vector<int> seq;
            seq.push_back(char2idx_.at("\xEC\x8B\x9C"));  // BOS '시'
            for (int id : ids) seq.push_back(id);
            seq.push_back(char2idx_.at("\xEB\x81\x9D"));  // EOS '끝'
            std::vector<int> targets;
            for (int i : poly) targets.push_back(i + 1);
            std::vector<int> preds = predict(seq, targets);
            for (size_t k = 0; k < poly.size(); k++) pros[poly[k]] = idx2class_[preds[k]];
        }
        return pros;
    }

  private:
    struct Tensor {
        std::vector<uint32_t> dims;
        std::vector<float> data;
    };
    const Tensor& w(const char* name) const { return weights_.at(name); }

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    // One LSTM direction over the embedding rows of seq.
    // Wih [4H, E], Whh [4H, H], gate order i|f|g|o.
    std::vector<std::array<float, 32>> lstm_dir(const std::vector<int>& seq, bool rev) const {
        const Tensor& emb = w("embedding.weight");
        const Tensor& wih = w(rev ? "lstm.weight_ih_l0_reverse" : "lstm.weight_ih_l0");
        const Tensor& whh = w(rev ? "lstm.weight_hh_l0_reverse" : "lstm.weight_hh_l0");
        const Tensor& bih = w(rev ? "lstm.bias_ih_l0_reverse" : "lstm.bias_ih_l0");
        const Tensor& bhh = w(rev ? "lstm.bias_hh_l0_reverse" : "lstm.bias_hh_l0");
        const int E = 64, H = 32;
        std::vector<std::array<float, 32>> hs(seq.size());
        std::array<float, 32> h{}, c{};
        for (size_t t = 0; t < seq.size(); t++) {
            int id = rev ? seq[seq.size() - 1 - t] : seq[t];
            const float* x = &emb.data[(size_t)id * E];
            float gates[128];
            for (int r = 0; r < 4 * H; r++) {
                double acc = (double)bih.data[r] + bhh.data[r];
                const float* wr = &wih.data[(size_t)r * E];
                for (int k = 0; k < E; k++) acc += (double)x[k] * wr[k];
                const float* hr = &whh.data[(size_t)r * H];
                for (int k = 0; k < H; k++) acc += (double)h[k] * hr[k];
                gates[r] = (float)acc;
            }
            std::array<float, 32> hn;
            for (int k = 0; k < H; k++) {
                float i = sigmoid(gates[k]);
                float f = sigmoid(gates[H + k]);
                float g = std::tanh(gates[2 * H + k]);
                float o = sigmoid(gates[3 * H + k]);
                c[k] = f * c[k] + i * g;
                hn[k] = o * std::tanh(c[k]);
            }
            h = hn;
            hs[rev ? seq.size() - 1 - t : t] = h;
        }
        return hs;
    }

    std::vector<int> predict(const std::vector<int>& seq, const std::vector<int>& targets) const {
        auto fw = lstm_dir(seq, false);
        auto bw = lstm_dir(seq, true);
        const Tensor& w0 = w("logit_layer.0.weight");
        const Tensor& b0 = w("logit_layer.0.bias");
        const Tensor& w1 = w("logit_layer.2.weight");
        const Tensor& b1 = w("logit_layer.2.bias");
        std::vector<int> out;
        for (int t : targets) {
            float x[64];
            for (int k = 0; k < 32; k++) { x[k] = fw[t][k]; x[32 + k] = bw[t][k]; }
            float hid[32];
            for (int r = 0; r < 32; r++) {
                double acc = b0.data[r];
                for (int k = 0; k < 64; k++) acc += (double)x[k] * w0.data[(size_t)r * 64 + k];
                hid[r] = acc > 0 ? (float)acc : 0.0f;
            }
            int best = 0;
            double bestv = -1e300;
            for (size_t r = 0; r < b1.data.size(); r++) {
                double acc = b1.data[r];
                for (int k = 0; k < 32; k++) acc += (double)hid[k] * w1.data[r * 32 + k];
                if (acc > bestv) { bestv = acc; best = (int)r; }
            }
            out.push_back(best);
        }
        return out;
    }

    std::unordered_map<uint32_t, std::vector<std::string>> cedict_;
    std::unordered_map<std::string, int32_t> char2idx_;
    std::vector<std::string> idx2class_;
    std::unordered_map<std::string, Tensor> weights_;
};

}  // namespace g2pm
