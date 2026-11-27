// M3-3d stage (b) check: the assembled C++ Chinese g2p
// (jieba + g2pM + pypinyin + sandhi/erhua + opencpop) vs
// Genie's g2p() oracle on normalized texts, plus the
// TextNormalizer port vs the normalize oracle, plus the
// full process() pipeline vs the full oracle. Crash rows
// in the oracles (inputs where Genie raises) are reported
// and skipped -- the port must not crash on them, but its
// output there is out of contract.
//
// Usage: chn_g2p_check ASSETDIR REFDIR
// ASSETDIR: g2pm.bin jieba_posseg.bin finalseg.bin
//   pypinyin.bin sandhi.bin polyphonic.bin symbols_v2.txt
//   opencpop-strict.txt dict.txt t2s.bin
#include <cstdio>
#include <string>
#include <vector>

#include "chn_g2p.h"
#include "json_min.h"

static std::string read_all(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

static std::vector<std::string> lines(const std::string& data) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < data.size()) {
        size_t eol = data.find('\n', pos);
        if (eol == std::string::npos) eol = data.size();
        if (eol > pos) out.push_back(data.substr(pos, eol - pos));
        pos = eol + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: chn_g2p_check ASSETDIR REFDIR\n");
        return 2;
    }
    std::string a = argv[1];
    chng2p::ChineseG2P g;
    if (!g.load(a + "/g2pm.bin", a + "/dict.txt", a + "/jieba_posseg.bin", a + "/finalseg.bin",
                a + "/pypinyin.bin", a + "/sandhi.bin", a + "/polyphonic.bin", a + "/opencpop-strict.txt",
                a + "/symbols_v2.txt", a + "/t2s.bin")) {
        std::fprintf(stderr, "asset load failed\n");
        return 1;
    }
    std::string refdir = argv[2];
    int total = 0, bad = 0, crashes = 0;
    for (const std::string& line : lines(read_all(refdir + "/g2p_oracle.jsonl"))) {
        Json j = JsonParser(line).parse();
        const Json* jt = j.find("text");
        if (!jt) continue;
        auto mine = g.g2p(jt->as_str());
        if (j.find("crash")) {
            crashes++;
            std::printf("genie-crash row (port produced %zu phones): %s\n", mine.first.size(),
                        jt->as_str().c_str());
            continue;
        }
        const Json* jp = j.find("phones");
        const Json* jw = j.find("word2ph");
        total++;
        bool ok = mine.first.size() == jp->arr.size() && mine.second.size() == jw->arr.size();
        if (ok)
            for (size_t i = 0; i < mine.first.size(); i++)
                if (mine.first[i] != jp->arr[i].as_str()) ok = false;
        if (ok)
            for (size_t i = 0; i < mine.second.size(); i++)
                if (mine.second[i] != (int)jw->arr[i].as_num()) ok = false;
        if (!ok) {
            bad++;
            if (bad <= 5) {
                std::printf("G2P MISMATCH text=%s\n  mine:", jt->as_str().c_str());
                for (auto& s : mine.first) std::printf(" %s", s.c_str());
                std::printf(" |");
                for (int v : mine.second) std::printf(" %d", v);
                std::printf("\n  ref :");
                for (auto& v : jp->arr) std::printf(" %s", v.as_str().c_str());
                std::printf(" |");
                for (auto& v : jw->arr) std::printf(" %d", (int)v.as_num());
                std::printf("\n");
            }
        }
    }
    std::printf("chn_g2p_check: %d/%d exact (%d genie-crash rows skipped)\n", total - bad, total, crashes);

    // (A) normalize_text vs norm oracle.
    int ntot = 0, nbad = 0;
    for (const std::string& line : lines(read_all(refdir + "/norm_oracle.jsonl"))) {
        Json j = JsonParser(line).parse();
        const Json* jt = j.find("text");
        const Json* jn = j.find("normalized");
        if (!jt || !jn) continue;
        ntot++;
        std::string mine = g.normalize_text(jt->as_str());
        if (mine != jn->as_str()) {
            nbad++;
            if (nbad <= 5)
                std::printf("NORM MISMATCH\n  text: %s\n  mine: %s\n  ref : %s\n", jt->as_str().c_str(),
                            mine.c_str(), jn->as_str().c_str());
        }
    }
    std::printf("norm_check: %d/%d exact\n", ntot - nbad, ntot);

    // (C) process() vs full oracle.
    int ftot = 0, fbad = 0, fcrash = 0;
    for (const std::string& line : lines(read_all(refdir + "/full_oracle.jsonl"))) {
        Json j = JsonParser(line).parse();
        const Json* jt = j.find("text");
        if (!jt) continue;
        auto mine = g.process(jt->as_str());
        if (j.find("crash")) {
            fcrash++;
            continue;
        }
        const Json* jc = j.find("clean");
        const Json* jp = j.find("phones");
        const Json* ji = j.find("ids");
        const Json* jw = j.find("word2ph");
        ftot++;
        bool ok = mine.clean == jc->as_str() && mine.phones.size() == jp->arr.size() &&
                  mine.ids.size() == ji->arr.size() && mine.word2ph.size() == jw->arr.size();
        if (ok)
            for (size_t i = 0; i < mine.phones.size(); i++)
                if (mine.phones[i] != jp->arr[i].as_str()) ok = false;
        if (ok)
            for (size_t i = 0; i < mine.ids.size(); i++)
                if (mine.ids[i] != (int64_t)ji->arr[i].as_num()) ok = false;
        if (ok)
            for (size_t i = 0; i < mine.word2ph.size(); i++)
                if (mine.word2ph[i] != (int)jw->arr[i].as_num()) ok = false;
        if (!ok) {
            fbad++;
            if (fbad <= 5) {
                std::printf("FULL MISMATCH text=%s\n  clean mine: %s\n  clean ref : %s\n", jt->as_str().c_str(),
                            mine.clean.c_str(), jc->as_str().c_str());
                std::printf("  phones mine(%zu):", mine.phones.size());
                for (auto& s : mine.phones) std::printf(" %s", s.c_str());
                std::printf("\n  phones ref (%zu):", jp->arr.size());
                for (auto& v : jp->arr) std::printf(" %s", v.as_str().c_str());
                std::printf("\n");
            }
        }
    }
    std::printf("full_check: %d/%d exact (%d genie-crash rows skipped)\n", ftot - fbad, ftot, fcrash);
    return (bad || nbad || fbad) ? 1 : 0;
}
