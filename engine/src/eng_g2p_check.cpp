// eng_g2p_check ENGDIR SYMBOLS.txt ORACLE.jsonl -- verify
// the English G2P port stage by stage (normalize, token
// alignment, POS tags, phones, ids) against Genie oracles.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "eng_g2p.h"

static std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '\n') { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
static std::vector<std::string> jarr_str(const Json& j) {
    std::vector<std::string> o;
    for (auto& v : j.arr) o.push_back(v.as_str());
    return o;
}
static std::vector<std::wstring> jarr_w(const Json& j) {
    std::vector<std::wstring> o;
    for (auto& v : j.arr) o.push_back(eng2p::u8_to_w(v.as_str()));
    return o;
}
static std::string show(const std::vector<std::string>& v) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) { if (i) r += " "; r += v[i]; }
    return r;
}
static std::string show_w(const std::vector<std::wstring>& v) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) { if (i) r += " "; r += eng2p::w_to_u8(v[i]); }
    return r;
}

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: eng_g2p_check ENGDIR SYMBOLS.txt ORACLE.jsonl\n"; return 2; }
    std::vector<std::string> symbols;
    {
        std::ifstream f(argv[2]);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            symbols.push_back(line);
        }
        while (!symbols.empty() && symbols.back().empty()) symbols.pop_back();
    }
    eng2p::EngG2P g;
    std::string err;
    if (!g.load(argv[1], symbols, err)) { std::cerr << "load: " << err << "\n"; return 1; }
    std::ifstream of(argv[3]);
    std::stringstream ss;
    ss << of.rdbuf();
    int n = 0, n_norm = 0, n_corr = 0, n_tags = 0, n_phones = 0, n_ids = 0;
    for (auto& line : split_lines(ss.str())) {
        if (line.empty()) continue;
        Json row = JsonParser(line).parse();
        std::wstring text = eng2p::u8_to_w(row.find("text")->as_str());
        n++;
        std::wstring norm = eng2p::eng_text_normalize(text);
        bool ok_norm = norm == eng2p::u8_to_w(row.find("normalized")->as_str());
        auto corr = g.corrected_tokens(text);
        bool ok_corr = corr == jarr_w(*row.find("corrected"));
        auto tags = g.tagger.tag(corr);
        bool ok_tags = tags == jarr_str(*row.find("tags"));
        auto phones = g.g2p(text);
        bool ok_phones = phones == jarr_str(*row.find("phones"));
        auto ids = g.phones_to_ids(text);
        std::vector<int64_t> want_ids;
        for (auto& v : row.find("ids")->arr) want_ids.push_back((int64_t)v.as_num());
        bool ok_ids = ids == want_ids;
        n_norm += ok_norm; n_corr += ok_corr; n_tags += ok_tags; n_phones += ok_phones; n_ids += ok_ids;
        if (!(ok_norm && ok_corr && ok_tags && ok_phones && ok_ids)) {
            std::cout << "CASE " << n << ": " << row.find("text")->as_str() << "\n";
            if (!ok_norm) std::cout << "  norm got [" << eng2p::w_to_u8(norm) << "] want [" << row.find("normalized")->as_str() << "]\n";
            if (!ok_corr) std::cout << "  corr got [" << show_w(corr) << "] want [" << show(jarr_str(*row.find("corrected"))) << "]\n";
            if (!ok_tags) std::cout << "  tags got [" << show(tags) << "] want [" << show(jarr_str(*row.find("tags"))) << "]\n";
            if (!ok_phones) std::cout << "  phones got [" << show(phones) << "]\n         want [" << show(jarr_str(*row.find("phones"))) << "]\n";
            if (!ok_ids) std::cout << "  ids differ\n";
        }
    }
    std::printf("eng oracle: %d cases | norm %d corr %d tags %d phones %d ids %d\n",
                n, n_norm, n_corr, n_tags, n_phones, n_ids);
    return 0;
}
