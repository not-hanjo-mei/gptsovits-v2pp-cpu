// C++ port of the jieba_fast segmentation Genie's Chinese
// G2P uses: the dictionary DAG + route DP (Tokenizer.get_DAG
// / calc), posseg's __cut_DAG with its BMES+tag viterbi for
// unknown words, and the main cut / cut_for_search path
// (with finalseg for unknown buffers) that ToneSandhi's
// _split_word relies on. Mirrors the Python sources
// (jieba_fast/__init__.py, posseg/__init__.py,
// posseg/viterbi.py, finalseg/__init__.py) including their
// tie-breaking: Python max() over (value, key) tuples picks
// the larger key on exact value ties.
//
// Assets: dict.txt (jieba_fast's own, plain text) and the
// exported jieba_posseg.bin / finalseg.bin tables.
// Validated by jieba_check against jieba_fast oracles.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "wordpiece.h"  // utf8 helpers

namespace jieba {

struct Pair {
    std::string word, pos;
};

namespace detail {
// A UTF-8 string viewed as codepoints with byte offsets.
struct UStr {
    std::string s;
    std::vector<uint32_t> cp;
    std::vector<size_t> off;  // size cp+1
    UStr() = default;
    explicit UStr(const std::string& str) : s(str) {
        cp = wptok::utf8_decode(str);
        off.assign(cp.size() + 1, 0);
        size_t p = 0;
        for (size_t i = 0; i < cp.size(); i++) {
            off[i] = p;
            p += cp[i] < 0x80 ? 1 : cp[i] < 0x800 ? 2 : cp[i] < 0x10000 ? 3 : 4;
        }
        off[cp.size()] = s.size();
    }
    size_t size() const { return cp.size(); }
    std::string slice(size_t a, size_t b) const {  // codepoint range [a, b)
        if (b > cp.size()) b = cp.size();
        return s.substr(off[a], off[b] - off[a]);
    }
};

inline bool in_han_detail(uint32_t c) { return c >= 0x4E00 && c <= 0x9FD5; }
inline bool in_han_internal(uint32_t c) {
    return in_han_detail(c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '+' || c == '#' || c == '&' || c == '.' || c == '_';
}
inline bool in_han_default(uint32_t c) { return in_han_internal(c) || c == '%'; }
inline bool is_num_char(uint32_t c) { return c == '.' || (c >= '0' && c <= '9'); }
inline bool is_eng_char(uint32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
}  // namespace detail

class Jieba {
  public:
    // dict.txt: "word freq tag" lines. Builds FREQ (real
    // words), the prefix set, total, and the word->tag table.
    bool load_dict(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        std::string all;
        char buf[1 << 16];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
        fclose(f);
        size_t pos = 0;
        while (pos < all.size()) {
            size_t eol = all.find('\n', pos);
            if (eol == std::string::npos) eol = all.size();
            std::string line = all.substr(pos, eol - pos);
            pos = eol + 1;
            while (!line.empty() && (line.back() == ' ' || line.back() == '\r')) line.pop_back();
            if (line.empty()) continue;
            size_t s1 = line.find(' ');
            if (s1 == std::string::npos) continue;
            size_t s2 = line.find(' ', s1 + 1);
            std::string word = line.substr(0, s1);
            int freq = std::stoi(line.substr(s1 + 1, s2 == std::string::npos ? s2 : s2 - s1 - 1));
            freq_[word] = freq;
            total_ += freq;
            if (s2 != std::string::npos) tag_[word] = line.substr(s2 + 1);
            detail::UStr u(word);
            for (size_t i = 1; i <= u.size(); i++) prefixes_.insert(u.slice(0, i));
        }
        logtotal_ = std::log((double)total_);
        return true;
    }

    bool load_posseg(const std::string& path) {
        std::vector<uint8_t> buf = read_file(path);
        if (buf.empty()) return false;
        size_t p = 0;
        auto u32 = [&]() { uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
        auto u8 = [&]() { return buf[p++]; };
        auto f64 = [&]() { double v; std::memcpy(&v, buf.data() + p, 8); p += 8; return v; };
        auto str = [&]() {
            uint16_t len;
            std::memcpy(&len, buf.data() + p, 2);
            p += 2;
            std::string s((const char*)buf.data() + p, len);
            p += len;
            return s;
        };
        if (buf.size() < 4 || std::memcmp(buf.data(), "JPS1", 4) != 0) return false;
        p = 4;
        uint32_t ns = u32();
        states_.resize(ns);
        for (uint32_t i = 0; i < ns; i++) {
            states_[i].pos = (char)u8();
            states_[i].tag = str();
        }
        uint32_t nc = u32();
        for (uint32_t i = 0; i < nc; i++) {
            uint32_t cp = u32();
            uint16_t cnt;
            std::memcpy(&cnt, buf.data() + p, 2);
            p += 2;
            auto& v = tab_[cp];
            for (uint16_t k = 0; k < cnt; k++) v.push_back((int)u32());
        }
        uint32_t nstart = u32();
        start_.assign(nstart, 0.0);
        start_has_.assign(nstart, 0);
        for (uint32_t i = 0; i < nstart; i++) {
            start_has_[i] = u8();
            start_[i] = f64();
        }
        uint32_t ntr = u32();
        trans_.resize(ntr);
        trans_map_.resize(ntr);
        for (uint32_t i = 0; i < ntr; i++) {
            uint32_t cnt = u32();
            for (uint32_t k = 0; k < cnt; k++) {
                int to = (int)u32();
                double v = f64();
                trans_[i].push_back({to, v});
                trans_map_[i][to] = v;
            }
        }
        uint32_t nkeys = u32();
        for (uint32_t i = 0; i < nkeys; i++) trans_keys_.push_back((int)u32());
        uint32_t nem = u32();
        emit_.resize(nem);
        for (uint32_t i = 0; i < nem; i++) {
            uint32_t cnt = u32();
            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t cp = u32();
                emit_[i][cp] = f64();
            }
        }
        return p == buf.size();
    }

    bool load_finalseg(const std::string& path) {
        std::vector<uint8_t> buf = read_file(path);
        if (buf.size() < 4 || std::memcmp(buf.data(), "FSG1", 4) != 0) return false;
        size_t p = 4;
        auto f64 = [&]() { double v; std::memcpy(&v, buf.data() + p, 8); p += 8; return v; };
        auto u32 = [&]() { uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
        for (int i = 0; i < 4; i++) {
            fs_start_has_[i] = buf[p++];
            fs_start_[i] = f64();
        }
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) {
                fs_trans_has_[a][b] = buf[p++];
                fs_trans_[a][b] = f64();
            }
        for (int i = 0; i < 4; i++) {
            uint32_t cnt = u32();
            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t cp = u32();
                fs_emit_[i][cp] = f64();
            }
        }
        return p == buf.size();
    }

    // FREQ.get(word): >0 for real words, 0 when absent or a
    // mere prefix (jieba stores prefixes with freq 0).
    int freq_of(const std::string& w) const {
        auto it = freq_.find(w);
        return it == freq_.end() ? 0 : it->second;
    }
    std::string tag_of(const std::string& w) const {
        auto it = tag_.find(w);
        return it == tag_.end() ? "x" : it->second;
    }

    // ---- posseg (jieba_fast.posseg.lcut) ----
    std::vector<Pair> posseg_cut(const std::string& sentence) const {
        detail::UStr u(sentence);
        std::vector<Pair> out;
        size_t i = 0, N = u.size();
        while (i < N) {
            size_t j = i;
            bool han = detail::in_han_internal(u.cp[i]);
            while (j < N && detail::in_han_internal(u.cp[j]) == han) j++;
            if (han) {
                auto pairs = cut_dag_posseg(u, i, j);
                out.insert(out.end(), pairs.begin(), pairs.end());
            } else {
                cut_skip_internal(u, i, j, out);
            }
            i = j;
        }
        return out;
    }

    // ---- main cut + cut_for_search ----
    std::vector<std::string> cut_for_search(const std::string& sentence) const {
        std::vector<std::string> words = cut_main(sentence);
        std::vector<std::string> out;
        for (const std::string& w : words) {
            detail::UStr u(w);
            if (u.size() > 2)
                for (size_t i = 0; i + 1 < u.size(); i++) {
                    std::string g = u.slice(i, i + 2);
                    if (freq_of(g) > 0) out.push_back(g);
                }
            if (u.size() > 3)
                for (size_t i = 0; i + 2 < u.size(); i++) {
                    std::string g = u.slice(i, i + 3);
                    if (freq_of(g) > 0) out.push_back(g);
                }
            out.push_back(w);
        }
        return out;
    }

    std::vector<std::string> cut_main(const std::string& sentence) const {
        detail::UStr u(sentence);
        std::vector<std::string> out;
        size_t i = 0, N = u.size();
        while (i < N) {
            size_t j = i;
            bool han = detail::in_han_default(u.cp[i]);
            while (j < N && detail::in_han_default(u.cp[j]) == han) j++;
            if (han) {
                auto ws = cut_dag_main(u, i, j);
                out.insert(out.end(), ws.begin(), ws.end());
            } else {
                // re_skip_default split: \r\n or single \s are
                // yielded whole, other spans char by char.
                size_t k = i;
                while (k < j) {
                    if (u.cp[k] == '\r' && k + 1 < j && u.cp[k + 1] == '\n') {
                        out.push_back(u.slice(k, k + 2));
                        k += 2;
                    } else if (wptok::is_space(u.cp[k])) {
                        out.push_back(u.slice(k, k + 1));
                        k++;
                    } else {
                        size_t m = k;
                        while (m < j && !wptok::is_space(u.cp[m])) m++;
                        for (size_t t = k; t < m; t++) out.push_back(u.slice(t, t + 1));
                        k = m;
                    }
                }
            }
            i = j;
        }
        return out;
    }

  private:
    static std::vector<uint8_t> read_file(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return {};
        std::vector<uint8_t> buf;
        uint8_t chunk[65536];
        size_t n;
        while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
        fclose(f);
        return buf;
    }

    // Tokenizer.get_DAG over codepoints [b, e) of u.
    std::vector<std::vector<int>> get_dag(const detail::UStr& u, size_t b, size_t e) const {
        size_t N = e - b;
        std::vector<std::vector<int>> dag(N);
        for (size_t k = 0; k < N; k++) {
            std::vector<int> tmplist;
            size_t i = k;
            while (i < N) {
                std::string frag = u.slice(b + k, b + i + 1);
                if (prefixes_.find(frag) == prefixes_.end() && freq_.find(frag) == freq_.end()) break;
                if (freq_of(frag) > 0) tmplist.push_back((int)i);
                i++;
            }
            if (tmplist.empty()) tmplist.push_back((int)k);
            dag[k] = tmplist;
        }
        return dag;
    }

    // Tokenizer.calc: route[i] = best end index (tuple max:
    // value first, larger end index wins ties).
    std::vector<int> calc(const detail::UStr& u, size_t b, size_t e,
                          const std::vector<std::vector<int>>& dag) const {
        size_t N = e - b;
        std::vector<double> val(N + 1, 0.0);
        std::vector<int> route(N + 1, 0);
        for (size_t idx = N; idx-- > 0;) {
            double best = -std::numeric_limits<double>::infinity();
            int bestx = (int)idx;
            for (int x : dag[idx]) {
                int f = freq_of(u.slice(b + idx, b + x + 1));
                if (f == 0) f = 1;
                double v = std::log((double)f) - logtotal_ + val[x + 1];
                if (v > best || (v == best && x > bestx)) {
                    best = v;
                    bestx = x;
                }
            }
            val[idx] = best;
            route[idx] = bestx;
        }
        return route;
    }

    // posseg __cut: BMES+tag viterbi segmentation of a Han run.
    std::vector<Pair> cut_viterbi_posseg(const detail::UStr& u, size_t b, size_t e) const {
        std::vector<int> route = viterbi_posseg(u, b, e);
        std::vector<Pair> out;
        size_t begin = 0, nexti = 0, N = e - b;
        for (size_t i = 0; i < N; i++) {
            char pos = states_[route[i]].pos;
            if (pos == 'B') {
                begin = i;
            } else if (pos == 'E') {
                out.push_back({u.slice(b + begin, b + i + 1), states_[route[i]].tag});
                nexti = i + 1;
            } else if (pos == 'S') {
                out.push_back({u.slice(b + i, b + i + 1), states_[route[i]].tag});
                nexti = i + 1;
            }
        }
        if (nexti < N) out.push_back({u.slice(b + nexti, b + N), states_[route[nexti]].tag});
        return out;
    }

    bool state_less(int a, int b) const {
        if (states_[a].pos != states_[b].pos) return states_[a].pos < states_[b].pos;
        return states_[a].tag < states_[b].tag;
    }

    std::vector<int> viterbi_posseg(const detail::UStr& u, size_t b, size_t e) const {
        const double MIN_FLOAT = -3.14e100;
        const double MIN_INF = -std::numeric_limits<double>::infinity();
        size_t N = e - b;
        std::vector<int> route(N, -1);
        if (N == 0) return route;
        auto states_of = [&](uint32_t cp) -> const std::vector<int>& {
            auto it = tab_.find(cp);
            return it != tab_.end() ? it->second : trans_keys_;
        };
        auto emit_get = [&](int st, uint32_t cp) {
            auto it = emit_[st].find(cp);
            return it != emit_[st].end() ? it->second : MIN_FLOAT;
        };
        // V[t]: state -> prob, with insertion order kept.
        std::vector<std::vector<int>> ord(N);
        std::vector<std::unordered_map<int, double>> V(N);
        std::vector<std::unordered_map<int, int>> mem(N);
        for (int y : states_of(u.cp[b])) {
            V[0][y] = (start_has_[y] ? start_[y] : MIN_FLOAT) + emit_get(y, u.cp[b]);
            mem[0][y] = -1;
            ord[0].push_back(y);
        }
        for (size_t t = 1; t < N; t++) {
            std::vector<int> prev_states;
            for (int x : ord[t - 1])
                if (!trans_[x].empty()) prev_states.push_back(x);
            std::unordered_set<int> expect;
            for (int x : prev_states)
                for (auto& kv : trans_[x]) expect.insert(kv.first);
            std::vector<int> obs_states;
            {
                const std::vector<int>& cand = states_of(u.cp[b + t]);
                std::unordered_set<int> candset(cand.begin(), cand.end());
                for (int y : cand)
                    if (expect.count(y)) obs_states.push_back(y);
                if (obs_states.empty()) {
                    if (!expect.empty())
                        for (int y : expect) obs_states.push_back(y);
                    else
                        obs_states = trans_keys_;
                }
                (void)candset;
            }
            for (int y : obs_states) {
                double em = emit_get(y, u.cp[b + t]);
                double best = MIN_INF;
                int besty0 = -1;
                for (int y0 : prev_states) {
                    auto it = trans_map_[y0].find(y);
                    double tr = it != trans_map_[y0].end() ? it->second : MIN_INF;
                    auto vit = V[t - 1].find(y0);
                    if (vit == V[t - 1].end()) continue;
                    double v = vit->second + tr + em;
                    if (besty0 < 0 || v > best || (v == best && state_less(besty0, y))) {
                        best = v;
                        besty0 = y0;
                    }
                }
                V[t][y] = best;
                mem[t][y] = besty0;
                ord[t].push_back(y);
            }
        }
        // Final: max over (V[-1][y], y).
        double best = 0;
        int state = -1;
        for (int y : ord[N - 1]) {
            double v = V[N - 1][y];
            if (state < 0 || v > best || (v == best && state_less(state, y))) {
                best = v;
                state = y;
            }
        }
        for (size_t i = N; i-- > 0;) {
            route[i] = state;
            state = mem[i][state];
        }
        return route;
    }

    // posseg __cut_detail for an unrecognized buffer.
    void cut_detail(const detail::UStr& u, size_t b, size_t e, std::vector<Pair>& out) const {
        size_t i = b;
        while (i < e) {
            size_t j = i;
            bool han = detail::in_han_detail(u.cp[i]);
            while (j < e && detail::in_han_detail(u.cp[j]) == han) j++;
            if (han) {
                auto pairs = cut_viterbi_posseg(u, i, j);
                out.insert(out.end(), pairs.begin(), pairs.end());
            } else {
                // re_skip_detail split: runs of [.0-9] or
                // [a-zA-Z0-9] alternate with gap text.
                size_t k = i;
                while (k < j) {
                    size_t m = k;
                    if (detail::is_num_char(u.cp[k])) {
                        while (m < j && detail::is_num_char(u.cp[m])) m++;
                        out.push_back({u.slice(k, m), "m"});
                    } else if (detail::is_eng_char(u.cp[k])) {
                        while (m < j && detail::is_eng_char(u.cp[m])) m++;
                        out.push_back({u.slice(k, m), "eng"});
                    } else {
                        while (m < j && !detail::is_num_char(u.cp[m]) && !detail::is_eng_char(u.cp[m])) m++;
                        out.push_back({u.slice(k, m), "x"});
                    }
                    k = m;
                }
            }
            i = j;
        }
    }

    // posseg __cut_DAG over [b, e).
    std::vector<Pair> cut_dag_posseg(const detail::UStr& u, size_t b, size_t e) const {
        auto dag = get_dag(u, b, e);
        auto route = calc(u, b, e, dag);
        std::vector<Pair> out;
        std::string buf;
        size_t buflen = 0;
        auto flush = [&]() {
            if (buf.empty()) return;
            if (buflen == 1) {
                out.push_back({buf, tag_of(buf)});
            } else if (freq_of(buf) == 0) {
                detail::UStr bu(buf);
                cut_detail(bu, 0, bu.size(), out);
            } else {
                detail::UStr bu(buf);
                for (size_t i = 0; i < bu.size(); i++) {
                    std::string ch = bu.slice(i, i + 1);
                    out.push_back({ch, tag_of(ch)});
                }
            }
            buf.clear();
            buflen = 0;
        };
        size_t N = e - b, x = 0;
        while (x < N) {
            size_t y = (size_t)route[x] + 1;
            std::string w = u.slice(b + x, b + y);
            if (y - x == 1) {
                buf += w;
                buflen++;
            } else {
                flush();
                out.push_back({w, tag_of(w)});
            }
            x = y;
        }
        flush();
        return out;
    }

    // posseg skip blocks: (\r\n|\s) pieces get 'x'; other
    // spans go char by char with m/eng/x by span's first char.
    void cut_skip_internal(const detail::UStr& u, size_t b, size_t e, std::vector<Pair>& out) const {
        size_t k = b;
        while (k < e) {
            if (u.cp[k] == '\r' && k + 1 < e && u.cp[k + 1] == '\n') {
                out.push_back({u.slice(k, k + 2), "x"});
                k += 2;
            } else if (wptok::is_space(u.cp[k])) {
                out.push_back({u.slice(k, k + 1), "x"});
                k++;
            } else {
                size_t m = k;
                while (m < e && !wptok::is_space(u.cp[m])) m++;
                bool eng = detail::is_eng_char(u.cp[k]);
                for (size_t t = k; t < m; t++) {
                    std::string ch = u.slice(t, t + 1);
                    if (detail::is_num_char(u.cp[t]))
                        out.push_back({ch, "m"});
                    else if (eng)
                        out.push_back({ch, "eng"});
                    else
                        out.push_back({ch, "x"});
                }
                k = m;
            }
        }
    }

    // finalseg viterbi over BMES (state indices 0..3 = B,M,E,S).
    std::vector<int> viterbi_finalseg(const detail::UStr& u, size_t b, size_t e) const {
        const double MIN_FLOAT = -3.14e100;
        size_t N = e - b;
        std::vector<int> route(N, 0);
        if (N == 0) return route;
        // PrevStatus: B<-{E,S}, M<-{M,B}, E<-{B,M}, S<-{S,E}
        static const int prevs[4][2] = {{2, 3}, {1, 0}, {0, 1}, {3, 2}};
        double V[4], Vn[4];
        std::vector<std::array<int, 4>> parent(N);
        auto emit_get = [&](int st, uint32_t cp) {
            auto it = fs_emit_[st].find(cp);
            return it != fs_emit_[st].end() ? it->second : MIN_FLOAT;
        };
        for (int y = 0; y < 4; y++) {
            V[y] = (fs_start_has_[y] ? fs_start_[y] : MIN_FLOAT) + emit_get(y, u.cp[b]);
            parent[0][y] = -1;
        }
        for (size_t t = 1; t < N; t++) {
            for (int y = 0; y < 4; y++) {
                double em = emit_get(y, u.cp[b + t]);
                double best = -std::numeric_limits<double>::infinity();
                int besty0 = prevs[y][0];
                for (int pi = 0; pi < 2; pi++) {
                    int y0 = prevs[y][pi];
                    double tr = fs_trans_has_[y0][y] ? fs_trans_[y0][y] : MIN_FLOAT;
                    double v = V[y0] + tr + em;
                    if (v > best || (v == best && y0 > besty0)) {
                        best = v;
                        besty0 = y0;
                    }
                }
                Vn[y] = best;
                parent[t][y] = besty0;
            }
            for (int y = 0; y < 4; y++) V[y] = Vn[y];
        }
        // max over 'ES' with tuple compare (prob, state char):
        int state = (V[3] > V[2] || (V[3] == V[2] && 'S' > 'E')) ? 3 : 2;
        for (size_t i = N; i-- > 0;) {
            route[i] = state;
            state = parent[i][state];
        }
        return route;
    }

    std::vector<std::string> finalseg_cut(const detail::UStr& u, size_t b, size_t e) const {
        std::vector<std::string> out;
        size_t i = b;
        while (i < e) {
            size_t j = i;
            bool han = detail::in_han_detail(u.cp[i]);
            while (j < e && detail::in_han_detail(u.cp[j]) == han) j++;
            if (han) {
                auto route = viterbi_finalseg(u, i, j);
                size_t begin = 0, nexti = 0, N = j - i;
                for (size_t t = 0; t < N; t++) {
                    int pos = route[t];
                    if (pos == 0) {
                        begin = t;
                    } else if (pos == 2) {
                        out.push_back(u.slice(i + begin, i + t + 1));
                        nexti = t + 1;
                    } else if (pos == 3) {
                        out.push_back(u.slice(i + t, i + t + 1));
                        nexti = t + 1;
                    }
                }
                if (nexti < N) out.push_back(u.slice(i + nexti, i + N));
            } else {
                // re_skip: ([a-zA-Z0-9]+(?:\.\d+)?%?) pieces
                // and gaps, yielded as-is.
                size_t k = i;
                while (k < j) {
                    size_t m = k;
                    if (detail::is_eng_char(u.cp[k])) {
                        while (m < j && detail::is_eng_char(u.cp[m])) m++;
                        if (m < j && u.cp[m] == '.' && m + 1 < j && u.cp[m + 1] >= '0' && u.cp[m + 1] <= '9') {
                            m++;
                            while (m < j && u.cp[m] >= '0' && u.cp[m] <= '9') m++;
                        }
                        if (m < j && u.cp[m] == '%') m++;
                        out.push_back(u.slice(k, m));
                    } else {
                        while (m < j && !detail::is_eng_char(u.cp[m])) m++;
                        out.push_back(u.slice(k, m));
                    }
                    k = m;
                }
            }
            i = j;
        }
        return out;
    }

    // Main tokenizer __cut_DAG over [b, e).
    std::vector<std::string> cut_dag_main(const detail::UStr& u, size_t b, size_t e) const {
        auto dag = get_dag(u, b, e);
        auto route = calc(u, b, e, dag);
        std::vector<std::string> out;
        std::string buf;
        size_t buflen = 0;
        auto flush = [&]() {
            if (buf.empty()) return;
            if (buflen == 1) {
                out.push_back(buf);
            } else if (freq_of(buf) == 0) {
                detail::UStr bu(buf);
                auto ws = finalseg_cut(bu, 0, bu.size());
                out.insert(out.end(), ws.begin(), ws.end());
            } else {
                detail::UStr bu(buf);
                for (size_t i = 0; i < bu.size(); i++) out.push_back(bu.slice(i, i + 1));
            }
            buf.clear();
            buflen = 0;
        };
        size_t N = e - b, x = 0;
        while (x < N) {
            size_t y = (size_t)route[x] + 1;
            std::string w = u.slice(b + x, b + y);
            if (y - x == 1) {
                buf += w;
                buflen++;
            } else {
                flush();
                out.push_back(w);
            }
            x = y;
        }
        flush();
        return out;
    }

    std::unordered_map<std::string, int> freq_;
    std::unordered_set<std::string> prefixes_;
    std::unordered_map<std::string, std::string> tag_;
    long long total_ = 0;
    double logtotal_ = 0;

    struct StateInfo {
        char pos = 0;
        std::string tag;
    };
    std::vector<StateInfo> states_;
    std::unordered_map<uint32_t, std::vector<int>> tab_;
    std::vector<double> start_;
    std::vector<char> start_has_;
    std::vector<std::vector<std::pair<int, double>>> trans_;
    std::vector<std::unordered_map<int, double>> trans_map_;
    std::vector<int> trans_keys_;
    std::vector<std::unordered_map<uint32_t, double>> emit_;

    double fs_start_[4] = {0, 0, 0, 0};
    char fs_start_has_[4] = {0, 0, 0, 0};
    double fs_trans_[4][4] = {};
    char fs_trans_has_[4][4] = {};
    std::unordered_map<uint32_t, double> fs_emit_[4];
};

}  // namespace jieba
