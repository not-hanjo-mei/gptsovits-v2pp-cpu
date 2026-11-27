// M3-3d stage (b3) check, pypinyin part: to_initials /
// to_finals_tone3 over the whole pinyin universe, and
// lazy_pinyin(FINALS_TONE3) over the segmented word list.
//
// Usage: pypinyin_check PYPINYIN.bin REFDIR
#include <cstdio>
#include <string>
#include <vector>

#include "json_min.h"
#include "pypinyin_mini.h"

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
        std::fprintf(stderr, "usage: pypinyin_check PYPINYIN.bin REFDIR\n");
        return 2;
    }
    pypin::PinyinMini pm;
    if (!pm.load(argv[1])) {
        std::fprintf(stderr, "asset load failed\n");
        return 1;
    }
    std::string refdir = argv[2];
    int total = 0, bad = 0;
    for (const std::string& line : lines(read_all(refdir + "/toneconv_oracle.jsonl"))) {
        Json j = JsonParser(line).parse();
        const Json* jp = j.find("pinyin");
        if (!jp) continue;
        std::string mine_i = pm.to_initials(jp->as_str());
        std::string mine_f = pm.to_finals_tone3(jp->as_str(), true);
        total++;
        if (mine_i != j.find("initials")->as_str() || mine_f != j.find("finals")->as_str()) {
            bad++;
            if (bad <= 10)
                std::printf("TC MISMATCH %s: mine (%s, %s) ref (%s, %s)\n", jp->as_str().c_str(),
                            mine_i.c_str(), mine_f.c_str(), j.find("initials")->as_str().c_str(),
                            j.find("finals")->as_str().c_str());
        }
    }
    std::printf("pypinyin_check toneconv: %d/%d exact\n", total - bad, total);
    int total2 = 0, bad2 = 0;
    for (const std::string& line : lines(read_all(refdir + "/lazy_oracle.jsonl"))) {
        Json j = JsonParser(line).parse();
        const Json* jw = j.find("word");
        if (!jw) continue;
        auto mine = pm.lazy_finals_tone3(jw->as_str());
        const Json* jf = j.find("finals");
        total2++;
        bool ok = mine.size() == jf->arr.size();
        if (ok)
            for (size_t i = 0; i < mine.size(); i++)
                if (mine[i] != jf->arr[i].as_str()) ok = false;
        if (!ok) {
            bad2++;
            if (bad2 <= 10) {
                std::printf("LAZY MISMATCH %s: mine [", jw->as_str().c_str());
                for (auto& s : mine) std::printf("%s ", s.c_str());
                std::printf("] ref [");
                for (auto& v : jf->arr) std::printf("%s ", v.as_str().c_str());
                std::printf("]\n");
            }
        }
    }
    std::printf("pypinyin_check lazy: %d/%d exact\n", total2 - bad2, total2);
    return (bad || bad2) ? 1 : 0;
}
