// M3-3d stage (b2) check: the C++ jieba port vs jieba_fast
// oracles -- posseg.lcut on the normalized corpus and
// cut_for_search on the segmented word list.
//
// Usage: jieba_check DICT.txt POSSEG.bin FINALSEG.bin REFDIR
// REFDIR holds psg_oracle.jsonl and cfs_oracle.jsonl.
#include <cstdio>
#include <string>
#include <vector>

#include "jieba.h"
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
    if (argc != 5) {
        std::fprintf(stderr, "usage: jieba_check DICT POSSEG FINALSEG REFDIR\n");
        return 2;
    }
    jieba::Jieba jb;
    if (!jb.load_dict(argv[1]) || !jb.load_posseg(argv[2]) || !jb.load_finalseg(argv[3])) {
        std::fprintf(stderr, "asset load failed\n");
        return 1;
    }
    std::string refdir = argv[4];
    int total = 0, bad = 0;
    for (const std::string& line : lines(read_all(refdir + "/psg_oracle.jsonl"))) {
        Json j = JsonParser(line).parse();
        const Json* jt = j.find("text");
        const Json* jp = j.find("pairs");
        if (!jt || !jp) continue;
        auto mine = jb.posseg_cut(jt->as_str());
        total++;
        bool ok = mine.size() == jp->arr.size();
        if (ok)
            for (size_t i = 0; i < mine.size(); i++)
                if (mine[i].word != jp->arr[i].arr[0].as_str() || mine[i].pos != jp->arr[i].arr[1].as_str()) ok = false;
        if (!ok) {
            bad++;
            if (bad <= 5) {
                std::printf("PSG MISMATCH text=%s\n  mine:", jt->as_str().c_str());
                for (auto& pr : mine) std::printf(" %s/%s", pr.word.c_str(), pr.pos.c_str());
                std::printf("\n  ref :");
                for (auto& v : jp->arr) std::printf(" %s/%s", v.arr[0].as_str().c_str(), v.arr[1].as_str().c_str());
                std::printf("\n");
            }
        }
    }
    std::printf("jieba_check psg: %d/%d exact\n", total - bad, total);
    int total2 = 0, bad2 = 0;
    for (const std::string& line : lines(read_all(refdir + "/cfs_oracle.jsonl"))) {
        Json j = JsonParser(line).parse();
        const Json* jt = j.find("word");
        const Json* jc = j.find("cut");
        if (!jt || !jc) continue;
        auto mine = jb.cut_for_search(jt->as_str());
        total2++;
        bool ok = mine.size() == jc->arr.size();
        if (ok)
            for (size_t i = 0; i < mine.size(); i++)
                if (mine[i] != jc->arr[i].as_str()) ok = false;
        if (!ok) {
            bad2++;
            if (bad2 <= 5) {
                std::printf("CFS MISMATCH word=%s\n  mine:", jt->as_str().c_str());
                for (auto& w : mine) std::printf(" %s", w.c_str());
                std::printf("\n  ref :");
                for (auto& v : jc->arr) std::printf(" %s", v.as_str().c_str());
                std::printf("\n");
            }
        }
    }
    std::printf("jieba_check cfs: %d/%d exact\n", total2 - bad2, total2);
    return (bad || bad2) ? 1 : 0;
}
