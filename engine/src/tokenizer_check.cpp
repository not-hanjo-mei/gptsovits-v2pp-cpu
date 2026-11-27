// M3-3d stage (c) check: the C++ WordPiece tokenizer vs the
// tokenizers library. The oracle file is JSON lines
// {"text": ..., "ids": [...]} produced by
// tools/gen_tokenizer_oracle.py (not shipped; the corpus is
// the 69 reference texts plus edge cases).
//
// Usage: tokenizer_check VOCAB.txt ORACLE.jsonl
#include <cstdio>
#include <string>
#include <vector>

#include "json_min.h"
#include "wordpiece.h"

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
        std::fprintf(stderr, "usage: tokenizer_check VOCAB.txt ORACLE.jsonl\n");
        return 2;
    }
    wptok::WordPieceTokenizer tk;
    if (!tk.load_vocab(argv[1])) {
        std::fprintf(stderr, "cannot load vocab %s\n", argv[1]);
        return 1;
    }
    std::string data = read_all(argv[2]);
    int total = 0, bad = 0;
    size_t pos = 0;
    while (pos < data.size()) {
        size_t eol = data.find('\n', pos);
        if (eol == std::string::npos) eol = data.size();
        std::string line = data.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.empty()) continue;
        Json j = JsonParser(line).parse();
        const Json* jt = j.find("text");
        const Json* ji = j.find("ids");
        if (!jt || !ji) continue;
        std::vector<int64_t> ids, mask;
        tk.encode(jt->as_str(), ids, mask);
        total++;
        bool ok = ids.size() == ji->arr.size();
        if (ok)
            for (size_t i = 0; i < ids.size(); i++)
                if (ids[i] != (int64_t)ji->arr[i].as_num()) { ok = false; break; }
        if (!ok) {
            bad++;
            std::printf("MISMATCH text=%s\n  mine:", jt->as_str().c_str());
            for (auto v : ids) std::printf(" %lld", (long long)v);
            std::printf("\n  ref :");
            for (auto& v : ji->arr) std::printf(" %lld", (long long)v.as_num());
            std::printf("\n");
        }
    }
    std::printf("tokenizer_check: %d/%d exact\n", total - bad, total);
    return bad ? 1 : 0;
}
