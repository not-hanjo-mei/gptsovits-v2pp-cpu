// M3-3d stage (d) check: runtime reference extraction vs
// the offline pack. Runs RefExtractor on a wav + prompt
// text and compares all five tensors against the .rpk the
// offline pipeline produced for the same reference.
//
// Usage: refextract_check ASSETDIR WAV PROMPT_TEXT RPK
#include <cmath>
#include <cstdio>
#include <string>

#include "refextract.h"

struct Stats {
    double corr = 0, maxerr = 0;
};
static Stats compare(const float* a, const float* b, size_t n) {
    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0, me = 0;
    for (size_t i = 0; i < n; i++) {
        double x = a[i], y = b[i];
        sa += x;
        sb += y;
        sab += x * y;
        saa += x * x;
        sbb += y * y;
        double d = std::fabs(x - y);
        if (d > me) me = d;
    }
    Stats s;
    double num = sab - sa * sb / (double)n;
    double den = std::sqrt((saa - sa * sa / (double)n) * (sbb - sb * sb / (double)n));
    s.corr = den > 0 ? num / den : 0;
    s.maxerr = me;
    return s;
}
static bool shape_eq(const RpkTensor& a, const RpkTensor& b) { return a.shape == b.shape; }
static std::string shape_str(const RpkTensor& t) {
    std::string s = "[";
    for (size_t i = 0; i < t.shape.size(); i++) {
        if (i) s += ",";
        s += std::to_string(t.shape[i]);
    }
    return s + "]";
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: refextract_check ASSETDIR WAV PROMPT_TEXT RPK [LANG]\n");
        return 2;
    }
    RefExtractor ex;
    std::string err;
    if (!ex.init(argv[1], err)) {
        std::fprintf(stderr, "init failed: %s\n", err.c_str());
        return 1;
    }
    auto mine = ex.extract(argv[2], argv[3], argc == 6 ? argv[5] : "zh");
    auto ref = load_rpk(argv[4]);
    int bad = 0;
    // ref_seq: exact integer equality.
    {
        const auto& a = mine.at("ref_seq");
        const auto& b = ref.at("ref_seq");
        bool ok = shape_eq(a, b);
        if (ok)
            for (int64_t i = 0; i < a.numel(); i++)
                if (a.i64()[i] != b.i64()[i]) ok = false;
        std::printf("ref_seq     shape %s vs %s : %s\n", shape_str(a).c_str(), shape_str(b).c_str(),
                    ok ? "EXACT" : "MISMATCH");
        if (!ok) bad++;
    }
    for (const char* name : {"ref_bert", "ssl_content", "ge", "ge_advanced"}) {
        const auto& a = mine.at(name);
        const auto& b = ref.at(name);
        if (!shape_eq(a, b)) {
            std::printf("%-11s shape %s vs %s : SHAPE MISMATCH\n", name, shape_str(a).c_str(), shape_str(b).c_str());
            bad++;
            continue;
        }
        Stats s = compare(a.f32(), b.f32(), (size_t)a.numel());
        std::printf("%-11s shape %s : corr %.6f maxerr %.6g\n", name, shape_str(a).c_str(), s.corr, s.maxerr);
    }
    std::printf("refextract_check: %s\n", bad ? "FAIL" : "DONE");
    return bad ? 1 : 0;
}
