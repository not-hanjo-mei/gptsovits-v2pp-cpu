// The pypinyin subset Genie's Chinese G2P uses, ported:
//   * to_initials / to_finals_tone3 over tone3 strings
//     (the g2pM pronunciation path), and
//   * lazy_pinyin(style=FINALS_TONE3,
//     neutral_tone_with_five=True) over jieba words (the
//     ToneSandhi merge decisions).
// Mirrors pypinyin/style/_tone_convert.py, style/_utils.py,
// standard.py (convert_finals), core/converter (phrase dict
// first, else per-char), seg/simpleseg + seg/mmseg
// (no_non_phrases=True). Data from the exported pypinyin.bin
// (PINYIN_DICT, PHRASES_DICT, phonetic symbol table).
// Validated by pypinyin_check against pypinyin oracles.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "jieba.h"  // detail::UStr

namespace pypin {

class PinyinMini {
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
        auto u32 = [&]() { uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
        auto u8 = [&]() { return buf[p++]; };
        auto str = [&]() {
            uint16_t len;
            std::memcpy(&len, buf.data() + p, 2);
            p += 2;
            std::string s((const char*)buf.data() + p, len);
            p += len;
            return s;
        };
        if (buf.size() < 4 || std::memcmp(buf.data(), "PYP1", 4) != 0) return false;
        p = 4;
        uint32_t n1 = u32();
        for (uint32_t i = 0; i < n1; i++) {
            uint32_t cp = u32();
            uint8_t cnt = u8();
            auto& v = singles_[cp];
            for (uint8_t k = 0; k < cnt; k++) v.push_back(str());
        }
        uint32_t n2 = u32();
        for (uint32_t i = 0; i < n2; i++) {
            std::string word = str();
            uint8_t nch = u8();
            auto& v = phrases_[word];
            for (uint8_t c = 0; c < nch; c++) {
                uint8_t cnt = u8();
                std::vector<std::string> prons;
                for (uint8_t k = 0; k < cnt; k++) prons.push_back(str());
                v.push_back(prons);
            }
            jieba::detail::UStr u(word);
            for (size_t k = 1; k <= u.size(); k++) phrase_prefixes_.insert(u.slice(0, k));
        }
        uint32_t n3 = u32();
        for (uint32_t i = 0; i < n3; i++) {
            std::string k = str();
            std::string v = str();
            auto cps = wptok::utf8_decode(k);
            if (cps.size() == 1)
                symbol1_[cps[0]] = v;
            else
                symbol_multi_.push_back({k, v});
        }
        return p == buf.size();
    }

    // ---- tone conversion (tone3 / tone-style inputs) ----

    std::string to_initials(const std::string& pinyin) const { return get_initials(pinyin, true); }

    std::string to_finals_tone3(const std::string& pinyin, bool neutral_with_five) const {
        std::string p2;
        for (char c : pinyin)
            if (c != '5') p2 += c;
        std::string finals = to_finals(p2);
        if (finals.empty()) return "";
        std::string num = first_digit(symbol_to_number(p2));
        if (num.empty()) return neutral_with_five ? finals + "5" : finals;
        return finals + num;
    }

    // ---- lazy_pinyin(FINALS_TONE3, neutral_tone_with_five=True) ----
    std::vector<std::string> lazy_finals_tone3(const std::string& word) const {
        std::vector<std::string> out;
        jieba::detail::UStr u(word);
        size_t i = 0, N = u.size();
        while (i < N) {
            size_t j = i;
            bool han = is_hans(u.cp[i]);
            while (j < N && is_hans(u.cp[j]) == han) j++;
            if (!han) {
                out.push_back(u.slice(i, j));
            } else {
                for (const std::string& seg : mmseg_cut(u, i, j)) {
                    auto it = phrases_.find(seg);
                    std::vector<std::string> syls;
                    if (it != phrases_.end()) {
                        for (auto& prons : it->second)
                            if (!prons.empty()) syls.push_back(prons[0]);
                    } else {
                        jieba::detail::UStr su(seg);
                        for (size_t k = 0; k < su.size(); k++) {
                            auto si = singles_.find(su.cp[k]);
                            if (si != singles_.end() && !si->second.empty())
                                syls.push_back(si->second[0]);
                            else
                                syls.push_back(su.slice(k, k + 1));
                        }
                    }
                    for (auto& s : syls) out.push_back(to_finals_tone3(s, true));
                }
            }
            i = j;
        }
        return out;
    }

    static bool is_hans(uint32_t c) {
        return c == 0x3007 || (c >= 0xE815 && c <= 0xE864) || c == 0xFA18 ||
               (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF) ||
               (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x20000 && c <= 0x2A6DF) ||
               (c >= 0x2A703 && c <= 0x2B73F) || (c >= 0x2B740 && c <= 0x2B81D) ||
               (c >= 0x2B825 && c <= 0x2BF6E) || (c >= 0x2C029 && c <= 0x2CE93) ||
               c == 0x2D016 || (c >= 0x2D11B && c <= 0x2EBD9) ||
               (c >= 0x2F80A && c <= 0x2FA1F) || (c >= 0x30000 && c <= 0x3134A) ||
               (c >= 0x300F7 && c <= 0x31288) || c == 0x30EDD || c == 0x30EDE ||
               (c >= 0x31350 && c <= 0x32389);
    }

  private:
    static const std::vector<std::string>& initials_strict() {
        static const std::vector<std::string> v = {"b", "p", "m", "f", "d", "t", "n", "l", "g",
                                                   "k", "h", "j", "q", "x", "zh", "ch", "sh", "r",
                                                   "z", "c", "s"};
        return v;
    }
    static const std::unordered_set<std::string>& finals_set() {
        static const std::unordered_set<std::string> s = {
            "i", "u", "ü", "a", "ia", "ua", "o", "uo", "e", "ie", "üe", "ai", "uai", "ei",
            "uei", "ao", "iao", "ou", "iou", "an", "ian", "uan", "üan", "en", "in", "uen",
            "ün", "ang", "iang", "uang", "eng", "ing", "ueng", "ong", "iong", "er", "ê"};
        return s;
    }

    std::string get_initials(const std::string& pinyin, bool strict) const {
        for (const std::string& i : initials_strict())
            if (pinyin.compare(0, i.size(), i) == 0) return i;
        if (!strict) {
            if (pinyin.compare(0, 1, "y") == 0) return "y";
            if (pinyin.compare(0, 1, "w") == 0) return "w";
        }
        return "";
    }

    std::string symbol_to_number(const std::string& s) const {
        std::string out;
        for (uint32_t cp : wptok::utf8_decode(s)) {
            auto it = symbol1_.find(cp);
            if (it != symbol1_.end())
                out += it->second;
            else
                wptok::utf8_encode(cp, out);
        }
        for (auto& kv : symbol_multi_) out = replace_all(out, kv.first, kv.second);
        return out;
    }
    static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
        if (from.empty()) return s;
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
        return s;
    }
    static std::string first_digit(const std::string& s) {
        for (char c : s)
            if (c >= '0' && c <= '9') return std::string(1, c);
        return "";
    }

    std::string to_finals(const std::string& pinyin) const {
        std::string s = symbol_to_number(pinyin);
        std::string nodigit;
        for (char c : s)
            if (!(c >= '0' && c <= '9')) nodigit += c;
        s = replace_all(nodigit, "v", "ü");
        std::string f = get_finals(s);
        return replace_all(f, "ü", "v");
    }

    std::string get_finals(const std::string& pinyin) const {
        std::string s = convert_finals(pinyin);
        std::string init = get_initials(s, true);
        std::string fin = s.substr(init.size());
        if (!finals_set().count(fin)) {
            std::string init2 = get_initials(s, false);
            std::string fin2 = s.substr(init2.size());
            if (finals_set().count(fin2)) return fin2;
            return "";
        }
        return fin;
    }

    // pypinyin.standard.convert_finals on symbol-free text
    // (toned variants of the maps collapse to the plain ones).
    std::string convert_finals(const std::string& in) const {
        std::string p = convert_zero_consonant(in);
        p = convert_uv(p);
        p = convert_iou(p);
        p = convert_uei(p);
        p = convert_uen(p);
        return p;
    }
    static bool is_u_tone(uint32_t c) {
        return c == 'u' || c == 0x016B || c == 0x00FA || c == 0x01D4 || c == 0x00F9;
    }
    static bool is_i_tone(uint32_t c) {
        return c == 'i' || c == 0x012B || c == 0x00ED || c == 0x01D0 || c == 0x00EC;
    }
    static std::string uv_map(uint32_t c) {
        switch (c) {
            case 'u': return "ü";
            case 0x016B: return "ǖ";
            case 0x00FA: return "ǘ";
            case 0x01D4: return "ǚ";
            case 0x00F9: return "ǜ";
        }
        return "";
    }
    std::string convert_zero_consonant(const std::string& in) const {
        std::string p = in;
        auto cps = wptok::utf8_decode(p);
        if (!cps.empty() && cps[0] == 'y') {
            if (cps.size() >= 2 && is_u_tone(cps[1])) {
                std::string rest;
                for (size_t i = 2; i < cps.size(); i++) wptok::utf8_encode(cps[i], rest);
                p = uv_map(cps[1]) + rest;
            } else if (cps.size() >= 2 && is_i_tone(cps[1])) {
                p = p.substr(1);
            } else {
                p = "i" + p.substr(1);
            }
            cps = wptok::utf8_decode(p);
        }
        if (!in.empty() && in[0] == 'w') {
            auto c2 = wptok::utf8_decode(p);
            if (c2.size() >= 2 && is_u_tone(c2[1])) {
                p = p.substr(1);
            } else {
                p = "u" + p.substr(1);
            }
        }
        if (!finals_set().count(p)) return in;
        return p;
    }
    static std::string convert_uv(const std::string& p) {
        auto cps = wptok::utf8_decode(p);
        if (cps.size() >= 2 && (cps[0] == 'j' || cps[0] == 'q' || cps[0] == 'x') && is_u_tone(cps[1])) {
            std::string out;
            wptok::utf8_encode(cps[0], out);
            out += uv_map(cps[1]);
            for (size_t i = 2; i < cps.size(); i++) wptok::utf8_encode(cps[i], out);
            return out;
        }
        return p;
    }
    // Shared shape of the iou/uei/uen rules: the string ends
    // with a two-codepoint key; iou is start-anchored ([a-z]+
    // before the key), uei/uen need one [a-z] before the key.
    static std::string convert_tail(const std::string& p, const std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::string>>& maps, bool anchored) {
        auto cps = wptok::utf8_decode(p);
        if (cps.size() < 2) return p;
        for (auto& m : maps) {
            if (cps[cps.size() - 2] == m.first.first && cps[cps.size() - 1] == m.first.second) {
                if (anchored) {
                    bool ok = cps.size() > 2;
                    for (size_t i = 0; i + 2 < cps.size() && ok; i++)
                        if (!(cps[i] >= 'a' && cps[i] <= 'z')) ok = false;
                    if (!ok) return p;
                } else {
                    if (cps.size() < 3 || !(cps[cps.size() - 3] >= 'a' && cps[cps.size() - 3] <= 'z')) return p;
                }
                std::string out;
                for (size_t i = 0; i + 2 < cps.size(); i++) wptok::utf8_encode(cps[i], out);
                out += m.second;
                return out;
            }
        }
        return p;
    }
    static std::string convert_iou(const std::string& p) {
        static const std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::string>> maps = {
            {{'i', 'u'}, "iou"}, {{'i', 0x016B}, "ioū"}, {{'i', 0x00FA}, "ioú"},
            {{'i', 0x01D4}, "ioǔ"}, {{'i', 0x00F9}, "ioù"}};
        return convert_tail(p, maps, true);
    }
    static std::string convert_uei(const std::string& p) {
        static const std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::string>> maps = {
            {{'u', 'i'}, "uei"}, {{'u', 0x012B}, "ueī"}, {{'u', 0x00ED}, "ueí"},
            {{'u', 0x01D0}, "ueǐ"}, {{'u', 0x00EC}, "ueì"}};
        return convert_tail(p, maps, false);
    }
    static std::string convert_uen(const std::string& p) {
        static const std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::string>> maps = {
            {{'u', 'n'}, "uen"}, {{'u', 0x016B}, "ūen"}, {{'u', 0x00FA}, "úen"},
            {{'u', 0x01D4}, "ǔen"}, {{'u', 0x00F9}, "ùen"}};
        return convert_tail(p, maps, false);
    }

    std::vector<std::string> mmseg_cut(const jieba::detail::UStr& u, size_t b, size_t e) const {
        std::vector<std::string> out;
        size_t pos = b;
        while (pos < e) {
            size_t last_valid = 0;  // length in codepoints
            bool broke = false;
            for (size_t idx = pos; idx < e; idx++) {
                std::string w = u.slice(pos, idx + 1);
                if (phrase_prefixes_.count(w)) {
                    if (phrases_.count(w)) last_valid = idx + 1 - pos;
                } else {
                    if (last_valid) {
                        out.push_back(u.slice(pos, pos + last_valid));
                        pos += last_valid;
                    } else {
                        out.push_back(u.slice(pos, pos + 1));
                        pos += 1;
                    }
                    broke = true;
                    break;
                }
            }
            if (!broke) {
                if (last_valid) {
                    out.push_back(u.slice(pos, pos + last_valid));
                    pos += last_valid;
                } else {
                    std::string rest = u.slice(pos, e);
                    if (!phrases_.count(rest)) {
                        for (size_t k = pos; k < e; k++) out.push_back(u.slice(k, k + 1));
                    } else {
                        out.push_back(rest);
                    }
                    pos = e;
                }
            }
        }
        return out;
    }

    std::unordered_map<uint32_t, std::vector<std::string>> singles_;
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> phrases_;
    std::unordered_set<std::string> phrase_prefixes_;
    std::unordered_map<uint32_t, std::string> symbol1_;
    std::vector<std::pair<std::string, std::string>> symbol_multi_;
};

}  // namespace pypin
