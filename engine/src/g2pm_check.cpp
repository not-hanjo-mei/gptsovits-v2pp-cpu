// M3-3d stage (b1) check: the C++ g2pM port vs the original
// g2pM on the oracle corpus (JSON lines {"text", "pinyins"}).
//
// Usage: g2pm_check G2PM.bin ORACLE.jsonl
#include <cstdio>
#include <string>
#include <vector>

#include "g2pm.h"
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

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: g2pm_check G2PM.bin ORACLE.jsonl\n");
        return 2;
    }
    g2pm::G2pM g;
    if (!g.load(argv[1])) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    std::string data = read_all(argv[2]);
    int total = 0, bad = 0, badchars = 0, totchars = 0;
    size_t pos = 0;
    while (pos < data.size()) {
        size_t eol = data.find('\n', pos);
        if (eol == std::string::npos) eol = data.size();
        std::string line = data.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.empty()) continue;
        Json j = JsonParser(line).parse();
        const Json* jt = j.find("text");
        const Json* jp = j.find("pinyins");
        if (!jt || !jp) continue;
        std::vector<std::string> mine = g(jt->as_str());
        total++;
        totchars += (int)jp->arr.size();
        bool ok = mine.size() == jp->arr.size();
        if (ok)
            for (size_t i = 0; i < mine.size(); i++)
                if (mine[i] != jp->arr[i].as_str()) { ok = false; badchars++; }
        if (!ok) {
            bad++;
            std::printf("MISMATCH text=%s\n  mine:", jt->as_str().c_str());
            for (auto& s : mine) std::printf(" %s", s.c_str());
            std::printf("\n  ref :");
            for (auto& v : jp->arr) std::printf(" %s", v.as_str().c_str());
            std::printf("\n");
        }
    }
    std::printf("g2pm_check: %d/%d texts exact, %d char mismatches of %d\n",
                total - bad, total, badchars, totchars);
    return bad ? 1 : 0;
}
