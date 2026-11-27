// Runtime reference extraction (M3-3d stage d): build the
// five reference-pack tensors in memory from a wav file and
// its prompt text, mirroring precompute_refs.py / Genie's
// ReferenceAudio + get_phones_and_bert('chinese'):
//   wav -> audio_32k (resample + 0.3 s silence append)
//       -> audio_16k (resample)
//       -> HuBERT      -> ssl_content [1, 768, T]
//       -> speaker enc -> sv embedding [1, 20480]
//       -> prompt enc  -> ge [1, 1024, 1], ge_advanced [1, 512, 1]
//   text -> normalize + Chinese g2p -> ref_seq [1, L] (ids),
//           word2ph -> RoBERTa (WordPiece ids/mask + repeats)
//           -> ref_bert [n_phones, 1024]
// With lang "en" the text side is the English G2P port
// (eng_g2p.h) and ref_bert is zeros, exactly as Genie's
// get_phones_and_bert('english') does; the audio side is
// unchanged. All four model stages run on the free engine;
// each output is copied out of the plan arena immediately.
// Extraction is once per new reference, so stage speed is
// secondary.
//
// Asset directory layout (host/refextract/):
//   hubert/chinese-hubert-base.onnx (+ converted fp32 bin)
//   RoBERTa.onnx  speaker_encoder.onnx
//   prompt_encoder_fp32.onnx  vocab.txt
//   g2p/{g2pm.bin, dict.txt, jieba_posseg.bin, finalseg.bin,
//        pypinyin.bin, sandhi.bin, polyphonic.bin,
//        opencpop-strict.txt, symbols_v2.txt, t2s.bin}
//   eng/{eng.bin, taggers/..., wordsegment/...} (English only,
//        loaded lazily on the first English request)
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "chn_g2p.h"
#include "eng_g2p.h"
#include "free_engine.h"
#include "npz.h"  // rd16/rd32 for rpk.h
#include "resample.h"
#include "rpk.h"
#include "wavread.h"
#include "wordpiece.h"

class RefExtractor {
   public:
    bool init(const std::string& dir, std::string& err) {
        dir_ = dir;
        try {
            hubert_ = compile(dir + "/hubert/chinese-hubert-base.onnx", "ref_hubert");
            roberta_ = compile(dir + "/RoBERTa.onnx", "ref_roberta");
            sv_ = compile(dir + "/speaker_encoder.onnx", "ref_sv");
            pe_ = compile(dir + "/prompt_encoder_fp32.onnx", "ref_prompt_encoder");
            if (!tok_.load_vocab(dir + "/vocab.txt")) {
                err = "cannot load vocab.txt";
                return false;
            }
            std::string g = dir + "/g2p";
            if (!g2p_.load(g + "/g2pm.bin", g + "/dict.txt", g + "/jieba_posseg.bin", g + "/finalseg.bin",
                           g + "/pypinyin.bin", g + "/sandhi.bin", g + "/polyphonic.bin",
                           g + "/opencpop-strict.txt", g + "/symbols_v2.txt", g + "/t2s.bin")) {
                err = "cannot load g2p assets";
                return false;
            }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
        ready_ = true;
        return true;
    }
    bool ready() const { return ready_; }

    bool ensure_eng(std::string& err) {
        if (eng_ready_) return true;
        if (eng_failed_) { err = "english g2p assets unavailable"; return false; }
        std::vector<std::string> symbols;
        {
            std::ifstream f(dir_ + "/g2p/symbols_v2.txt");
            if (!f) { eng_failed_ = true; err = "cannot open symbols_v2.txt"; return false; }
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                symbols.push_back(line);
            }
            while (!symbols.empty() && symbols.back().empty()) symbols.pop_back();
        }
        if (!eng_.load(dir_ + "/eng", symbols, err)) { eng_failed_ = true; return false; }
        eng_ready_ = true;
        return true;
    }

    std::map<std::string, RpkTensor> extract(const std::string& wav_path, const std::string& prompt_text,
                                             const std::string& lang = "zh") {
        if (!ready_) throw std::runtime_error("refextract: not initialized");
        const bool en = (lang == "en");
        if (en) {
            std::string err;
            if (!ensure_eng(err)) throw std::runtime_error("refextract eng: " + err);
        }
        refaudio::WavData w;
        std::string werr;
        if (!refaudio::read_wav(wav_path, w, werr)) throw std::runtime_error("refextract wav: " + werr);
        std::vector<float> a32 = w.sample_rate == 32000
                                     ? w.mono
                                     : refaudio::resample_sinc(w.mono, w.sample_rate, 32000);
        a32.resize(a32.size() + 9600, 0.0f);  // ReferenceAudio appends 0.3 s
        std::vector<float> a16 = refaudio::resample_sinc(a32, 32000, 16000);

        std::vector<int64_t> ref_ids;
        chng2p::ChineseG2P::Processed proc;
        if (en) {
            ref_ids = eng_.phones_to_ids(eng2p::u8_to_w(prompt_text));
            if (ref_ids.empty()) throw std::runtime_error("refextract: prompt text produced no phones");
        } else {
            proc = g2p_.process(prompt_text);
            if (proc.ids.empty()) throw std::runtime_error("refextract: prompt text produced no phones");
            ref_ids = proc.ids;
        }

        std::map<std::string, RpkTensor> out;

        // RoBERTa -> ref_bert (English: zeros, as upstream)
        if (en) {
            RpkTensor t;
            t.dtype = 0;
            t.shape = {(int64_t)ref_ids.size(), 1024};
            t.data.assign(ref_ids.size() * 1024 * 4, 0);
            out["ref_bert"] = std::move(t);
        } else {
            std::vector<int64_t> ids, mask;
            tok_.encode(proc.clean, ids, mask);
            std::vector<int64_t> repeats(proc.word2ph.begin(), proc.word2ph.end());
            std::map<std::string, freeeng::Tensor> in;
            in.emplace("input_ids", mk_i64(ids, {1, (int64_t)ids.size()}));
            in.emplace("attention_mask", mk_i64(mask, {1, (int64_t)mask.size()}));
            in.emplace("repeats", mk_i64(repeats, {(int64_t)repeats.size()}));
            auto outs = roberta_->run(in);
            out["ref_bert"] = copy_f32_rpk(outs.at("phone_features"));
            roberta_->release_arena();
        }
        // HuBERT -> ssl_content
        {
            std::map<std::string, freeeng::Tensor> in;
            in.emplace("input_values", mk_f32(a16, {1, (int64_t)a16.size()}));
            auto outs = hubert_->run(in);
            out["ssl_content"] = copy_f32_rpk(outs.at("last_hidden_state"));
            hubert_->release_arena();
        }
        // speaker encoder -> prompt encoder -> ge, ge_advanced
        {
            std::map<std::string, freeeng::Tensor> in;
            in.emplace("waveform", mk_f32(a16, {1, (int64_t)a16.size()}));
            auto outs = sv_->run(in);
            std::vector<int64_t> sv_shape;
            std::vector<float> sv_emb = copy_f32(outs.at("embedding"), sv_shape);
            sv_->release_arena();
            std::map<std::string, freeeng::Tensor> pin;
            pin.emplace("ref_audio", mk_f32(a32, {1, (int64_t)a32.size()}));
            pin.emplace("sv_emb", mk_f32(sv_emb, sv_shape));
            auto pouts = pe_->run(pin);
            out["ge"] = copy_f32_rpk(pouts.at("ge"));
            out["ge_advanced"] = copy_f32_rpk(pouts.at("ge_advanced"));
            pe_->release_arena();
        }
        // ref_seq
        {
            RpkTensor t;
            t.dtype = 1;
            t.shape = {1, (int64_t)ref_ids.size()};
            t.data.assign((const uint8_t*)ref_ids.data(),
                          (const uint8_t*)ref_ids.data() + ref_ids.size() * 8);
            out["ref_seq"] = std::move(t);
        }
        return out;
    }

   private:
    static std::shared_ptr<freeeng::Plan> compile(const std::string& path, const char* label) {
        onnxm::Graph g = onnxm::load_model(path);
        return freeeng::Plan::compile(g, label);
    }
    static freeeng::Tensor mk_f32(const std::vector<float>& v, std::vector<int64_t> shape) {
        freeeng::Tensor t;
        t.dtype = 1;
        t.data = (const uint8_t*)v.data();
        t.shape = std::move(shape);
        return t;
    }
    static freeeng::Tensor mk_i64(const std::vector<int64_t>& v, std::vector<int64_t> shape) {
        freeeng::Tensor t;
        t.dtype = 7;
        t.data = (const uint8_t*)v.data();
        t.shape = std::move(shape);
        return t;
    }
    static float f16_to_f32(uint16_t h) {
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp = (h >> 10) & 0x1f;
        uint32_t man = h & 0x3ff;
        uint32_t bits;
        if (exp == 0) {
            if (man == 0) {
                bits = sign;
            } else {
                exp = 1;
                while (!(man & 0x400)) {
                    man <<= 1;
                    exp--;
                }
                man &= 0x3ff;
                bits = sign | ((uint32_t)(exp + 112) << 23) | (man << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7f800000 | (man << 13);
        } else {
            bits = sign | ((uint32_t)(exp + 112) << 23) | (man << 13);
        }
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    // Copy a plan output (f32, or fp16-decoded) into a flat
    // vector, honoring strides for view outputs.
    static std::vector<float> copy_f32(const freeeng::Tensor& t, std::vector<int64_t>& shape_out) {
        shape_out = t.shape;
        std::vector<float> v;
        v.reserve((size_t)t.numel());
        std::vector<int64_t> coord(t.shape.size(), 0);
        for (int64_t i = 0; i < t.numel(); i++) {
            int64_t off = 0;
            for (size_t d = 0; d < coord.size(); d++) off += coord[d] * (t.strides.empty() ? 0 : t.strides[d]);
            if (t.strides.empty()) off = i;
            if (t.dtype == 1)
                v.push_back(((const float*)t.data)[off]);
            else if (t.dtype == 10)
                v.push_back(f16_to_f32(((const uint16_t*)t.data)[off]));
            else
                throw std::runtime_error("refextract: unexpected output dtype " + std::to_string(t.dtype));
            for (int d = (int)coord.size() - 1; d >= 0; d--) {
                if (++coord[d] < t.shape[d]) break;
                coord[d] = 0;
            }
        }
        return v;
    }
    static RpkTensor copy_f32_rpk(const freeeng::Tensor& t) {
        RpkTensor r;
        r.dtype = 0;
        std::vector<float> v = copy_f32(t, r.shape);
        r.data.assign((const uint8_t*)v.data(), (const uint8_t*)v.data() + v.size() * 4);
        return r;
    }

    std::shared_ptr<freeeng::Plan> hubert_, roberta_, sv_, pe_;
    wptok::WordPieceTokenizer tok_;
    chng2p::ChineseG2P g2p_;
    eng2p::EngG2P eng_;
    std::string dir_;
    bool ready_ = false, eng_ready_ = false, eng_failed_ = false;
};
