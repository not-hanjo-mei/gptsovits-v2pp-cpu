// C++ port of Genie's Chinese G2P (genie_tts/G2P/Chinese):
// text -> phones + word2ph via jieba posseg, g2pM, pypinyin
// tone conversion, the ToneSandhi rule stack, erhua merging,
// the polyphonic correction dict and the opencpop phone map.
// Mirrors ChineseG2P.py, ToneSandhi.py, Erhua.py and
// CorrectPronunciation.py. Two documented divergences, both
// inputs where the Python code raises IndexError and
// produces nothing at all (several shipped reference texts
// hit the first one, which is why only 65 of 69 clips have
// packs): an empty lazy-pinyin final in the three-tone merge
// checks counts as "not tone 3" instead of crashing, and an
// empty final with a non-empty initial is skipped like a
// KeyError instead of crashing.
// Validated by chn_g2p_check against Genie oracles.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "g2pm.h"
#include "jieba.h"
#include "pypinyin_mini.h"

namespace chng2p {

struct WordSets {
    std::unordered_set<std::string> must_neural, must_not_neural, must_erhua, not_erhua;
    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        std::vector<uint8_t> buf;
        uint8_t chunk[65536];
        size_t n;
        while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
        fclose(f);
        if (buf.size() < 4 || std::memcmp(buf.data(), "SHD1", 4) != 0) return false;
        size_t p = 4;
        auto u32 = [&]() { uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
        auto str = [&]() {
            uint16_t len;
            std::memcpy(&len, buf.data() + p, 2);
            p += 2;
            std::string s((const char*)buf.data() + p, len);
            p += len;
            return s;
        };
        std::unordered_set<std::string>* sets[4] = {&must_neural, &must_not_neural, &must_erhua, &not_erhua};
        for (int i = 0; i < 4; i++) {
            uint32_t cnt = u32();
            for (uint32_t k = 0; k < cnt; k++) sets[i]->insert(str());
        }
        return p == buf.size();
    }
};

namespace detail {
inline std::string tone_set(const std::string& s, char digit) {
    if (s.empty()) return s;
    size_t i = s.size() - 1;
    while (i > 0 && (s[i] & 0xC0) == 0x80) i--;
    return s.substr(0, i) + digit;
}
inline std::string last_cp_str(const std::string& s) {
    if (s.empty()) return "";
    size_t i = s.size() - 1;
    while (i > 0 && (s[i] & 0xC0) == 0x80) i--;
    return s.substr(i);
}
inline bool isnumeric_cp(uint32_t c) {
    if (c >= '0' && c <= '9') return true;
    if (c >= 0xFF10 && c <= 0xFF19) return true;
    switch (c) {
        case 0x3007: case 0x4E00: case 0x4E8C: case 0x4E09: case 0x56DB: case 0x4E94:
        case 0x516D: case 0x4E03: case 0x516B: case 0x4E5D: case 0x5341: case 0x767E:
        case 0x5343: case 0x4E07: case 0x4EBF: case 0x4E24: case 0x96F6: case 0x58F9:
        case 0x8D30: case 0x53C1: case 0x8086: case 0x4F0D: case 0x9646: case 0x67D2:
        case 0x634C: case 0x7396: case 0x62FE: case 0x4F70: case 0x4EDF: case 0x00B2:
        case 0x00B3: case 0x00B9:
            return true;
    }
    return false;
}
inline bool isalpha_cp(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c >= 0xFF21 && c <= 0xFF3A) return true;
    if (c >= 0xFF41 && c <= 0xFF5A) return true;
    if (c >= 0xC0 && c <= 0xFF && c != 0xD7 && c != 0xF7) return true;
    if (c >= 0x0100 && c <= 0x017F) return true;
    if (c >= 0x0391 && c <= 0x03C9 && c != 0x03A2) return true;
    return pypin::PinyinMini::is_hans(c);
}
inline std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}
}  // namespace detail

class ToneSandhi {
  public:
    void init(const WordSets* sets, const jieba::Jieba* jb, const pypin::PinyinMini* pm) {
        sets_ = sets;
        jb_ = jb;
        pm_ = pm;
    }

    std::vector<jieba::Pair> pre_merge_for_modify(std::vector<jieba::Pair> seg) const {
        seg = merge_bu(seg);
        seg = merge_yi(seg);
        seg = merge_reduplication(seg);
        seg = merge_continuous_three(seg, false);
        seg = merge_continuous_three(seg, true);
        seg = merge_er(seg);
        return seg;
    }

    std::vector<std::string> modified_tone(const std::string& word, const std::string& pos,
                                           std::vector<std::string> finals) const {
        finals = bu_sandhi(word, finals);
        finals = yi_sandhi(word, finals);
        finals = neural_sandhi(word, pos, finals);
        finals = three_sandhi(word, finals);
        return finals;
    }

  private:
    static bool all_tone_three(const std::vector<std::string>& finals) {
        for (auto& x : finals)
            if (x.empty() || x.back() != '3') return false;
        return true;
    }
    static bool is_reduplication(const std::string& word) {
        jieba::detail::UStr u(word);
        return u.size() == 2 && u.cp[0] == u.cp[1];
    }
    std::vector<std::string> split_word(const std::string& word) const {
        std::vector<std::string> wl = jb_->cut_for_search(word);
        std::stable_sort(wl.begin(), wl.end(), [](const std::string& a, const std::string& b) {
            return wptok::utf8_decode(a).size() < wptok::utf8_decode(b).size();
        });
        if (wl.empty()) return {word, ""};
        const std::string& first = wl[0];
        jieba::detail::UStr u(word), fu(first);
        size_t idx = std::string::npos;
        for (size_t i = 0; i + fu.size() <= u.size(); i++) {
            bool eq = true;
            for (size_t k = 0; k < fu.size(); k++)
                if (u.cp[i + k] != fu.cp[k]) { eq = false; break; }
            if (eq) { idx = i; break; }
        }
        if (idx == 0) return {first, u.slice(fu.size(), u.size())};
        return {u.slice(0, u.size() - fu.size()), first};
    }

    static std::vector<jieba::Pair> merge_bu(const std::vector<jieba::Pair>& seg) {
        std::vector<jieba::Pair> out;
        std::string last_word;
        for (auto& pr : seg) {
            std::string word = pr.word;
            if (last_word == "不") word = last_word + word;
            if (word != "不") out.push_back({word, pr.pos});
            last_word = word;
        }
        if (last_word == "不") out.push_back({last_word, "d"});
        return out;
    }
    static std::vector<jieba::Pair> merge_yi(const std::vector<jieba::Pair>& seg) {
        std::vector<jieba::Pair> news;
        size_t i = 0;
        while (i < seg.size()) {
            const std::string& word = seg[i].word;
            bool merged = false;
            if (i >= 1 && word == "一" && i + 1 < seg.size()) {
                jieba::Pair last = !news.empty() ? news.back() : seg[i - 1];
                if (last.word == seg[i + 1].word && last.pos == "v" && seg[i + 1].pos == "v") {
                    news.back() = {last.word + "一" + seg[i + 1].word, last.pos};
                    i += 2;
                    merged = true;
                }
            }
            if (!merged) {
                news.push_back(seg[i]);
                i += 1;
            }
        }
        std::vector<jieba::Pair> out;
        for (auto& pr : news) {
            if (!out.empty() && out.back().word == "一")
                out.back().word += pr.word;
            else
                out.push_back(pr);
        }
        return out;
    }
    static std::vector<jieba::Pair> merge_reduplication(const std::vector<jieba::Pair>& seg) {
        std::vector<jieba::Pair> out;
        for (auto& pr : seg) {
            if (!out.empty() && pr.word == out.back().word)
                out.back().word += pr.word;
            else
                out.push_back(pr);
        }
        return out;
    }
    std::vector<jieba::Pair> merge_continuous_three(const std::vector<jieba::Pair>& seg, bool second) const {
        std::vector<std::vector<std::string>> sub_finals;
        for (auto& pr : seg) sub_finals.push_back(pm_->lazy_finals_tone3(pr.word));
        std::vector<jieba::Pair> out;
        std::vector<char> merge_last(seg.size(), 0);
        for (size_t i = 0; i < seg.size(); i++) {
            bool cond = false;
            if (i >= 1 && !merge_last[i - 1]) {
                if (!second) {
                    cond = all_tone_three(sub_finals[i - 1]) && all_tone_three(sub_finals[i]);
                } else {
                    // Python indexes [-1][-1] here and crashes
                    // on empty finals; count those as not-tone-3.
                    auto& a = sub_finals[i - 1];
                    auto& b = sub_finals[i];
                    cond = !a.empty() && !b.empty() && !a.back().empty() && !b.front().empty() &&
                           a.back().back() == '3' && b.front().back() == '3';
                }
            }
            if (cond) {
                jieba::detail::UStr up(seg[i - 1].word), uc(seg[i].word);
                if (!is_reduplication(seg[i - 1].word) && up.size() + uc.size() <= 3) {
                    out.back().word += seg[i].word;
                    merge_last[i] = 1;
                } else {
                    out.push_back(seg[i]);
                }
            } else {
                out.push_back(seg[i]);
            }
        }
        return out;
    }
    static std::vector<jieba::Pair> merge_er(const std::vector<jieba::Pair>& seg) {
        std::vector<jieba::Pair> out;
        for (size_t i = 0; i < seg.size(); i++) {
            if (i >= 1 && seg[i].word == "儿" && seg[i - 1].word != "#")
                out.back().word += seg[i].word;
            else
                out.push_back(seg[i]);
        }
        return out;
    }

    static std::vector<std::string> bu_sandhi(const std::string& word, std::vector<std::string> finals) {
        jieba::detail::UStr u(word);
        if (u.size() == 3 && u.cp[1] == 0x4E0D) {  // 不
            if (finals.size() > 1) finals[1] = detail::tone_set(finals[1], '5');
        } else {
            for (size_t i = 0; i < u.size(); i++) {
                if (u.cp[i] == 0x4E0D && i + 1 < u.size() && i + 1 < finals.size() &&
                    !finals[i + 1].empty() && finals[i + 1].back() == '4' && i < finals.size())
                    finals[i] = detail::tone_set(finals[i], '2');
            }
        }
        return finals;
    }
    std::vector<std::string> yi_sandhi(const std::string& word, std::vector<std::string> finals) const {
        jieba::detail::UStr u(word);
        bool has_yi = false;
        for (uint32_t c : u.cp)
            if (c == 0x4E00) has_yi = true;
        if (has_yi) {
            bool all_num = true;
            for (uint32_t c : u.cp)
                if (c != 0x4E00 && !detail::isnumeric_cp(c)) all_num = false;
            if (all_num) return finals;
        }
        if (u.size() == 3 && u.cp[1] == 0x4E00 && u.cp[0] == u.cp[2]) {
            if (finals.size() > 1) finals[1] = detail::tone_set(finals[1], '5');
        } else if (word.compare(0, 6, "第一") == 0) {
            if (finals.size() > 1) finals[1] = detail::tone_set(finals[1], '1');
        } else {
            static const std::unordered_set<uint32_t> punc = {
                0xFF1A, 0xFF0C, 0xFF1B, 0x3002, 0xFF1F, 0xFF01, 0x201C, 0x201D, 0x2018,
                0x2019, '\'', ':', ',', ';', '.', '?', '!'};
            for (size_t i = 0; i < u.size(); i++) {
                if (u.cp[i] == 0x4E00 && i + 1 < u.size() && i < finals.size() && i + 1 < finals.size()) {
                    if (!finals[i + 1].empty() && finals[i + 1].back() == '4')
                        finals[i] = detail::tone_set(finals[i], '2');
                    else if (!punc.count(u.cp[i + 1]))
                        finals[i] = detail::tone_set(finals[i], '4');
                }
            }
        }
        return finals;
    }
    std::vector<std::string> neural_sandhi(const std::string& word, const std::string& pos,
                                           std::vector<std::string> finals) const {
        jieba::detail::UStr u(word);
        size_t n = u.size();
        if (!pos.empty() && (pos[0] == 'n' || pos[0] == 'v' || pos[0] == 'a') &&
            !sets_->must_not_neural.count(word)) {
            for (size_t j = 1; j < n; j++)
                if (u.cp[j] == u.cp[j - 1] && j < finals.size()) finals[j] = detail::tone_set(finals[j], '5');
        }
        auto last_in = [&](const char* chars) {
            if (n == 0) return false;
            std::unordered_set<uint32_t> s;
            for (uint32_t c : wptok::utf8_decode(chars)) s.insert(c);
            return s.count(u.cp[n - 1]) > 0;
        };
        auto set_last = [&]() {
            if (!finals.empty()) finals.back() = detail::tone_set(finals.back(), '5');
        };
        int ge_idx = -1;
        for (size_t i = 0; i < n; i++)
            if (u.cp[i] == 0x4E2A) { ge_idx = (int)i; break; }  // 个
        bool done = true;
        if (n >= 1 && last_in("吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶")) {
            set_last();
        } else if (n >= 1 && last_in("的地得")) {
            set_last();
        } else if (n == 1 && last_in("了着过") && (pos == "ul" || pos == "uz" || pos == "ug")) {
            set_last();
        } else if (n > 1 && last_in("们子") && (pos == "r" || pos == "n") &&
                   !sets_->must_not_neural.count(word)) {
            set_last();
        } else if (n > 1 && last_in("上下里") && (pos == "s" || pos == "l" || pos == "f")) {
            set_last();
        } else if (n > 1 && last_in("来去") && [&] {
                       std::unordered_set<uint32_t> s;
                       for (uint32_t c : wptok::utf8_decode("上下进出回过起开")) s.insert(c);
                       return s.count(u.cp[n - 2]) > 0;
                   }()) {
            set_last();
        } else if ((ge_idx >= 1 && (detail::isnumeric_cp(u.cp[ge_idx - 1]) || [&] {
                        std::unordered_set<uint32_t> s;
                        for (uint32_t c : wptok::utf8_decode("几有两半多各整每做是")) s.insert(c);
                        return s.count(u.cp[ge_idx - 1]) > 0;
                    }())) ||
                   word == "个") {
            if (ge_idx >= 0 && (size_t)ge_idx < finals.size())
                finals[ge_idx] = detail::tone_set(finals[ge_idx], '5');
        } else {
            done = false;
        }
        if (!done) {
            std::string last2 = n >= 2 ? u.slice(n - 2, n) : word;
            if (sets_->must_neural.count(word) || sets_->must_neural.count(last2)) set_last();
        }
        // Split-word conventional neutral tone.
        std::vector<std::string> wl = split_word(word);
        if (wl.size() == 2) {
            size_t first_len = jieba::detail::UStr(wl[0]).size();
            std::vector<std::vector<std::string>> fl = {
                std::vector<std::string>(finals.begin(), finals.begin() + std::min(first_len, finals.size())),
                std::vector<std::string>(finals.begin() + std::min(first_len, finals.size()), finals.end())};
            for (size_t i = 0; i < 2; i++) {
                jieba::detail::UStr su(wl[i]);
                std::string sub_last2 = su.size() >= 2 ? su.slice(su.size() - 2, su.size()) : wl[i];
                if ((sets_->must_neural.count(wl[i]) || sets_->must_neural.count(sub_last2)) && !fl[i].empty())
                    fl[i].back() = detail::tone_set(fl[i].back(), '5');
            }
            finals = fl[0];
            finals.insert(finals.end(), fl[1].begin(), fl[1].end());
        }
        return finals;
    }
    std::vector<std::string> three_sandhi(const std::string& word, std::vector<std::string> finals) const {
        jieba::detail::UStr u(word);
        size_t n = u.size();
        if (n == 2 && all_tone_three(finals)) {
            if (!finals.empty()) finals[0] = detail::tone_set(finals[0], '2');
        } else if (n == 3) {
            std::vector<std::string> wl = split_word(word);
            size_t first_len = wl.empty() ? 0 : jieba::detail::UStr(wl[0]).size();
            if (all_tone_three(finals)) {
                if (first_len == 2 && finals.size() > 1) {
                    finals[0] = detail::tone_set(finals[0], '2');
                    finals[1] = detail::tone_set(finals[1], '2');
                } else if (first_len == 1 && finals.size() > 1) {
                    finals[1] = detail::tone_set(finals[1], '2');
                }
            } else {
                std::vector<std::vector<std::string>> fl = {
                    std::vector<std::string>(finals.begin(), finals.begin() + std::min(first_len, finals.size())),
                    std::vector<std::string>(finals.begin() + std::min(first_len, finals.size()), finals.end())};
                for (size_t i = 0; i < fl.size(); i++) {
                    auto& sub = fl[i];
                    if (all_tone_three(sub) && sub.size() == 2) {
                        sub[0] = detail::tone_set(sub[0], '2');
                    } else if (i == 1 && !all_tone_three(sub) && !sub.empty() && !sub[0].empty() &&
                               sub[0].back() == '3' && !fl[0].empty() && !fl[0].back().empty() &&
                               fl[0].back().back() == '3') {
                        fl[0].back() = detail::tone_set(fl[0].back(), '2');
                    }
                    finals = fl[0];
                    finals.insert(finals.end(), fl[1].begin(), fl[1].end());
                }
            }
        } else if (n == 4) {
            std::vector<std::vector<std::string>> fl = {
                std::vector<std::string>(finals.begin(), finals.begin() + std::min<size_t>(2, finals.size())),
                std::vector<std::string>(finals.begin() + std::min<size_t>(2, finals.size()), finals.end())};
            finals.clear();
            for (auto& sub : fl) {
                if (all_tone_three(sub) && !sub.empty()) sub[0] = detail::tone_set(sub[0], '2');
                finals.insert(finals.end(), sub.begin(), sub.end());
            }
        }
        return finals;
    }

    const WordSets* sets_ = nullptr;
    const jieba::Jieba* jb_ = nullptr;
    const pypin::PinyinMini* pm_ = nullptr;
};

class Erhua {
  public:
    void init(const WordSets* sets) { sets_ = sets; }
    std::pair<std::vector<std::string>, std::vector<std::string>> merge_erhua(
        std::vector<std::string> initials, std::vector<std::string> finals, const std::string& word,
        const std::string& pos) const {
        jieba::detail::UStr u(word);
        for (size_t i = 0; i < finals.size(); i++) {
            if (i == finals.size() - 1 && i < u.size() && u.cp[i] == 0x513F && finals[i] == "er1")
                finals[i] = "er2";
        }
        if (!sets_->must_erhua.count(word) &&
            (sets_->not_erhua.count(word) || pos == "a" || pos == "j" || pos == "nr"))
            return {initials, finals};
        if (finals.size() != u.size()) return {initials, finals};
        std::vector<std::string> ni, nf;
        std::string last2 = u.size() >= 2 ? u.slice(u.size() - 2, u.size()) : word;
        for (size_t i = 0; i < finals.size(); i++) {
            std::string phn = finals[i];
            if (i == finals.size() - 1 && u.cp[i] == 0x513F && (phn == "er2" || phn == "er5") &&
                !sets_->not_erhua.count(last2) && !nf.empty())
                phn = "er" + detail::last_cp_str(nf.back());
            ni.push_back(initials[i]);
            nf.push_back(phn);
        }
        return {ni, nf};
    }

  private:
    const WordSets* sets_ = nullptr;
};

class ChineseG2P {
  public:
    bool load(const std::string& g2pm_bin, const std::string& dict_txt, const std::string& posseg_bin,
              const std::string& finalseg_bin, const std::string& pypinyin_bin, const std::string& sandhi_bin,
              const std::string& polyphonic_bin, const std::string& opencpop_txt, const std::string& symbols_txt) {
        if (!g2pm_.load(g2pm_bin)) return false;
        if (!jieba_.load_dict(dict_txt) || !jieba_.load_posseg(posseg_bin) || !jieba_.load_finalseg(finalseg_bin))
            return false;
        if (!pinyin_.load(pypinyin_bin)) return false;
        if (!sets_.load(sandhi_bin)) return false;
        if (!load_polyphonic(polyphonic_bin)) return false;
        if (!load_opencpop(opencpop_txt)) return false;
        if (!load_symbols(symbols_txt)) return false;
        sandhi_.init(&sets_, &jieba_, &pinyin_);
        erhua_.init(&sets_);
        return true;
    }

    // ChineseG2P.g2p on already-normalized text.
    std::pair<std::vector<std::string>, std::vector<int>> g2p(const std::string& text) const {
        std::vector<std::string> all_phones;
        std::vector<int> all_w2ph;
        for (const std::string& seg0 : split_sentences(text)) {
            std::string seg = remove_eng(seg0);
            auto seg_cut = sandhi_.pre_merge_for_modify(jieba_.posseg_cut(seg));
            std::vector<std::string> pinyins = g2pm_(seg);
            for (auto& p : pinyins) p = detail::replace_all(p, "u:", "v");
            std::vector<std::string> initials, finals;
            size_t pre = 0;
            jieba::detail::UStr su(seg);
            for (auto& pr : seg_cut) {
                jieba::detail::UStr wu(pr.word);
                size_t now = pre + wu.size();
                if (pr.pos == "eng") {
                    pre = now;
                    continue;
                }
                std::vector<std::string> word_pinyins;
                for (size_t i = pre; i < now && i < pinyins.size(); i++) word_pinyins.push_back(pinyins[i]);
                word_pinyins = correct_pronunciation(pr.word, word_pinyins);
                std::vector<std::string> sub_i, sub_f;
                for (auto& py : word_pinyins) {
                    auto cps = wptok::utf8_decode(py);
                    if (!cps.empty() && detail::isalpha_cp(cps[0])) {
                        sub_i.push_back(pinyin_.to_initials(py));
                        sub_f.push_back(pinyin_.to_finals_tone3(py, true));
                    } else {
                        sub_i.push_back(py);
                        sub_f.push_back(py);
                    }
                }
                pre = now;
                sub_f = sandhi_.modified_tone(pr.word, pr.pos, sub_f);
                auto merged = erhua_.merge_erhua(sub_i, sub_f, pr.word, pr.pos);
                initials.insert(initials.end(), merged.first.begin(), merged.first.end());
                finals.insert(finals.end(), merged.second.begin(), merged.second.end());
            }
            for (size_t i = 0; i < initials.size() && i < finals.size(); i++) {
                if (initials[i] == finals[i]) {
                    all_phones.push_back(initials[i]);
                    all_w2ph.push_back(1);
                } else {
                    auto pair = opencpop_phones(initials[i], finals[i]);
                    if (pair) {
                        all_phones.push_back(pair->first);
                        all_phones.push_back(pair->second);
                        all_w2ph.push_back(2);
                    }
                }
            }
        }
        return {all_phones, all_w2ph};
    }

    const std::unordered_map<std::string, int>& symbol_ids() const { return symbol_id_; }

  private:
    static std::vector<std::string> split_sentences(const std::string& text) {
        // pattern_split: (?<=[!?\…,\.-])\s* ; keep pieces
        // whose strip() is non-empty.
        static const std::unordered_set<uint32_t> punct = {'!', '?', 0x2026, ',', '.', '-'};
        jieba::detail::UStr u(text);
        std::vector<std::string> out;
        size_t start = 0, i = 0, N = u.size();
        auto push = [&](size_t a, size_t b) {
            std::string piece = u.slice(a, b);
            std::string t = piece;
            size_t s = 0, e = t.size();
            auto cps = wptok::utf8_decode(t);
            size_t cs = 0, ce = cps.size();
            while (cs < ce && wptok::is_space(cps[cs])) cs++;
            while (ce > cs && wptok::is_space(cps[ce - 1])) ce--;
            (void)s;
            (void)e;
            if (cs < ce) out.push_back(piece);
        };
        while (i < N) {
            if (punct.count(u.cp[i])) {
                size_t j = i + 1;
                while (j < N && wptok::is_space(u.cp[j])) j++;
                push(start, i + 1);
                start = j;
                i = j;
            } else {
                i++;
            }
        }
        push(start, N);
        return out;
    }
    static std::string remove_eng(const std::string& seg) {
        std::string out;
        for (uint32_t c : wptok::utf8_decode(seg)) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) continue;
            wptok::utf8_encode(c, out);
        }
        return out;
    }
    std::vector<std::string> correct_pronunciation(const std::string& word,
                                                   std::vector<std::string> word_pinyins) const {
        auto it = pp_.find(word);
        if (it != pp_.end()) return it->second;
        jieba::detail::UStr u(word);
        for (size_t idx = 0; idx < u.size(); idx++) {
            if (idx >= word_pinyins.size()) break;
            auto ci = pp_.find(u.slice(idx, idx + 1));
            if (ci != pp_.end() && !ci->second.empty()) word_pinyins[idx] = ci->second[0];
        }
        return word_pinyins;
    }
    std::optional<std::pair<std::string, std::string>> opencpop_phones(const std::string& c,
                                                                       const std::string& v) const {
        if (v.empty()) return std::nullopt;  // Python raises; skip
        std::string tone = detail::last_cp_str(v);
        std::string vwt = v.substr(0, v.size() - tone.size());
        std::string key;
        if (!c.empty()) {
            static const std::unordered_map<std::string, std::string> v_rep = {
                {"uei", "ui"}, {"iou", "iu"}, {"uen", "un"}};
            auto it = v_rep.find(vwt);
            key = c + (it != v_rep.end() ? it->second : vwt);
        } else {
            static const std::unordered_map<std::string, std::string> rep = {
                {"ing", "ying"}, {"i", "yi"}, {"in", "yin"}, {"u", "wu"}};
            static const std::unordered_map<std::string, std::string> single = {
                {"v", "yu"}, {"e", "e"}, {"i", "y"}, {"u", "w"}};
            auto it = rep.find(vwt);
            if (it != rep.end()) {
                key = it->second;
            } else if (!vwt.empty()) {
                auto cps0 = wptok::utf8_decode(vwt);
                std::string first;
                wptok::utf8_encode(cps0[0], first);
                auto si = single.find(first);
                if (si != single.end()) {
                    auto cps = wptok::utf8_decode(vwt);
                    std::string rest;
                    for (size_t i = 1; i < cps.size(); i++) wptok::utf8_encode(cps[i], rest);
                    key = si->second + rest;
                } else {
                    key = vwt;
                }
            } else {
                key = vwt;
            }
        }
        auto it = opencpop_.find(key);
        if (it == opencpop_.end()) return std::nullopt;
        const std::string& ps = it->second;
        size_t sp = ps.find(' ');
        if (sp == std::string::npos) return std::nullopt;
        return std::make_pair(ps.substr(0, sp), ps.substr(sp + 1) + tone);
    }
    bool load_polyphonic(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        std::vector<uint8_t> buf;
        uint8_t chunk[65536];
        size_t n;
        while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
        fclose(f);
        if (buf.size() < 4 || std::memcmp(buf.data(), "PPD1", 4) != 0) return false;
        size_t p = 4;
        auto u32 = [&]() { uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
        auto str = [&]() {
            uint16_t len;
            std::memcpy(&len, buf.data() + p, 2);
            p += 2;
            std::string s((const char*)buf.data() + p, len);
            p += len;
            return s;
        };
        uint32_t cnt = u32();
        for (uint32_t i = 0; i < cnt; i++) {
            std::string w = str();
            uint8_t np = buf[p++];
            auto& v = pp_[w];
            for (uint8_t k = 0; k < np; k++) v.push_back(str());
        }
        return p == buf.size();
    }
    bool load_opencpop(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        std::string all;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
        fclose(f);
        size_t pos = 0;
        while (pos < all.size()) {
            size_t eol = all.find('\n', pos);
            if (eol == std::string::npos) eol = all.size();
            std::string line = all.substr(pos, eol - pos);
            pos = eol + 1;
            size_t b = line.find_first_not_of(" \t\r");
            size_t e = line.find_last_not_of(" \t\r");
            if (b == std::string::npos) continue;
            line = line.substr(b, e - b + 1);
            size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string rest = line.substr(tab + 1);
            if (rest.find(' ') == std::string::npos && rest.find('\t') == std::string::npos) continue;
            opencpop_[line.substr(0, tab)] = rest;
        }
        return !opencpop_.empty();
    }
    bool load_symbols(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        std::string all;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
        fclose(f);
        size_t pos = 0;
        int id = 0;
        while (pos < all.size()) {
            size_t eol = all.find('\n', pos);
            if (eol == std::string::npos) eol = all.size();
            std::string line = all.substr(pos, eol - pos);
            pos = eol + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            symbol_id_[line] = id++;
        }
        return id > 0;
    }

    g2pm::G2pM g2pm_;
    jieba::Jieba jieba_;
    pypin::PinyinMini pinyin_;
    WordSets sets_;
    ToneSandhi sandhi_;
    Erhua erhua_;
    std::unordered_map<std::string, std::vector<std::string>> pp_;
    std::unordered_map<std::string, std::string> opencpop_;
    std::unordered_map<std::string, int> symbol_id_;
};

}  // namespace chng2p
