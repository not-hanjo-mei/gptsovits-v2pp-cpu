// BERT WordPiece tokenizer matching the tokenizers-library
// configuration shipped for chinese-roberta-wwm-ext-large
// (GenieData/RoBERTa/roberta_tokenizer/tokenizer.json):
// BertNormalizer (clean_text, handle_chinese_chars,
// lowercase) -> BertPreTokenizer -> WordPiece (## prefix,
// [UNK], max 100 chars per word) -> [CLS] ... [SEP].
// Validated by tokenizer_check against the tokenizers
// library on the reference-text corpus.
//
// In production the input is Genie's text_clean, which only
// contains CJK chars and the punctuation set ! ? ... , . -
// so most normalizer edge cases never trigger; the general
// behavior is implemented anyway for the ASCII/CJK domain.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace wptok {

// ---- UTF-8 helpers ----
inline std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        uint8_t c = (uint8_t)s[i];
        if (c < 0x80) { out.push_back(c); i++; continue; }
        int len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        if (len == 1 || i + (size_t)len > s.size()) { out.push_back(0xFFFD); i++; continue; }
        uint32_t cp = c & (0x7F >> len);
        bool ok = true;
        for (int k = 1; k < len; k++) {
            uint8_t cc = (uint8_t)s[i + k];
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        out.push_back(ok ? cp : 0xFFFD);
        i += ok ? (size_t)len : 1;
    }
    return out;
}

inline void utf8_encode(uint32_t cp, std::string& out) {
    if (cp < 0x80) out.push_back((char)cp);
    else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

inline bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

inline bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0x85 || cp == 0xA0 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Unicode punctuation (category P) approximated by the ASCII
// set plus the blocks that occur in Chinese text.
inline bool is_punct(uint32_t cp) {
    if (cp >= 0x21 && cp <= 0x2F) return true;
    if (cp >= 0x3A && cp <= 0x40) return true;
    if (cp >= 0x5B && cp <= 0x5E) return true;
    if (cp == 0x60) return true;
    if (cp >= 0x7B && cp <= 0x7E) return true;
    if (cp == 0xAB || cp == 0xBB) return true;
    if (cp >= 0x2000 && cp <= 0x206F) return true;
    if (cp >= 0x3000 && cp <= 0x303F) return true;
    if (cp >= 0xFF01 && cp <= 0xFF0F) return true;
    if (cp >= 0xFF1A && cp <= 0xFF20) return true;
    if (cp >= 0xFF3B && cp <= 0xFF40) return true;
    if (cp >= 0xFF5B && cp <= 0xFF65) return true;
    return false;
}

// Latin-1 Supplement + Latin Extended-A case and accent
// tables (generated from Unicode NFD data): BertNormalizer
// lowercases and strips accents; full Unicode case folding
// is out of domain for this tokenizer's inputs, but the
// Latin ranges cover what can survive upstream filtering.
static const uint32_t kLowerLatin[][2] = {
    {0xC0, 0xE0}, {0xC1, 0xE1}, {0xC2, 0xE2}, {0xC3, 0xE3}, {0xC4, 0xE4}, {0xC5, 0xE5},
    {0xC6, 0xE6}, {0xC7, 0xE7}, {0xC8, 0xE8}, {0xC9, 0xE9}, {0xCA, 0xEA}, {0xCB, 0xEB},
    {0xCC, 0xEC}, {0xCD, 0xED}, {0xCE, 0xEE}, {0xCF, 0xEF}, {0xD0, 0xF0}, {0xD1, 0xF1},
    {0xD2, 0xF2}, {0xD3, 0xF3}, {0xD4, 0xF4}, {0xD5, 0xF5}, {0xD6, 0xF6}, {0xD8, 0xF8},
    {0xD9, 0xF9}, {0xDA, 0xFA}, {0xDB, 0xFB}, {0xDC, 0xFC}, {0xDD, 0xFD}, {0xDE, 0xFE},
    {0x100, 0x101}, {0x102, 0x103}, {0x104, 0x105}, {0x106, 0x107}, {0x108, 0x109}, {0x10A, 0x10B},
    {0x10C, 0x10D}, {0x10E, 0x10F}, {0x110, 0x111}, {0x112, 0x113}, {0x114, 0x115}, {0x116, 0x117},
    {0x118, 0x119}, {0x11A, 0x11B}, {0x11C, 0x11D}, {0x11E, 0x11F}, {0x120, 0x121}, {0x122, 0x123},
    {0x124, 0x125}, {0x126, 0x127}, {0x128, 0x129}, {0x12A, 0x12B}, {0x12C, 0x12D}, {0x12E, 0x12F},
    {0x132, 0x133}, {0x134, 0x135}, {0x136, 0x137}, {0x139, 0x13A}, {0x13B, 0x13C}, {0x13D, 0x13E},
    {0x13F, 0x140}, {0x141, 0x142}, {0x143, 0x144}, {0x145, 0x146}, {0x147, 0x148}, {0x14A, 0x14B},
    {0x14C, 0x14D}, {0x14E, 0x14F}, {0x150, 0x151}, {0x152, 0x153}, {0x154, 0x155}, {0x156, 0x157},
    {0x158, 0x159}, {0x15A, 0x15B}, {0x15C, 0x15D}, {0x15E, 0x15F}, {0x160, 0x161}, {0x162, 0x163},
    {0x164, 0x165}, {0x166, 0x167}, {0x168, 0x169}, {0x16A, 0x16B}, {0x16C, 0x16D}, {0x16E, 0x16F},
    {0x170, 0x171}, {0x172, 0x173}, {0x174, 0x175}, {0x176, 0x177}, {0x178, 0xFF}, {0x179, 0x17A},
    {0x17B, 0x17C}, {0x17D, 0x17E},
};
static const uint32_t kStripLatin[][2] = {
    {0xC0, 0x41}, {0xC1, 0x41}, {0xC2, 0x41}, {0xC3, 0x41}, {0xC4, 0x41}, {0xC5, 0x41},
    {0xC7, 0x43}, {0xC8, 0x45}, {0xC9, 0x45}, {0xCA, 0x45}, {0xCB, 0x45}, {0xCC, 0x49},
    {0xCD, 0x49}, {0xCE, 0x49}, {0xCF, 0x49}, {0xD1, 0x4E}, {0xD2, 0x4F}, {0xD3, 0x4F},
    {0xD4, 0x4F}, {0xD5, 0x4F}, {0xD6, 0x4F}, {0xD9, 0x55}, {0xDA, 0x55}, {0xDB, 0x55},
    {0xDC, 0x55}, {0xDD, 0x59}, {0xE0, 0x61}, {0xE1, 0x61}, {0xE2, 0x61}, {0xE3, 0x61},
    {0xE4, 0x61}, {0xE5, 0x61}, {0xE7, 0x63}, {0xE8, 0x65}, {0xE9, 0x65}, {0xEA, 0x65},
    {0xEB, 0x65}, {0xEC, 0x69}, {0xED, 0x69}, {0xEE, 0x69}, {0xEF, 0x69}, {0xF1, 0x6E},
    {0xF2, 0x6F}, {0xF3, 0x6F}, {0xF4, 0x6F}, {0xF5, 0x6F}, {0xF6, 0x6F}, {0xF9, 0x75},
    {0xFA, 0x75}, {0xFB, 0x75}, {0xFC, 0x75}, {0xFD, 0x79}, {0xFF, 0x79}, {0x100, 0x41},
    {0x101, 0x61}, {0x102, 0x41}, {0x103, 0x61}, {0x104, 0x41}, {0x105, 0x61}, {0x106, 0x43},
    {0x107, 0x63}, {0x108, 0x43}, {0x109, 0x63}, {0x10A, 0x43}, {0x10B, 0x63}, {0x10C, 0x43},
    {0x10D, 0x63}, {0x10E, 0x44}, {0x10F, 0x64}, {0x112, 0x45}, {0x113, 0x65}, {0x114, 0x45},
    {0x115, 0x65}, {0x116, 0x45}, {0x117, 0x65}, {0x118, 0x45}, {0x119, 0x65}, {0x11A, 0x45},
    {0x11B, 0x65}, {0x11C, 0x47}, {0x11D, 0x67}, {0x11E, 0x47}, {0x11F, 0x67}, {0x120, 0x47},
    {0x121, 0x67}, {0x122, 0x47}, {0x123, 0x67}, {0x124, 0x48}, {0x125, 0x68}, {0x128, 0x49},
    {0x129, 0x69}, {0x12A, 0x49}, {0x12B, 0x69}, {0x12C, 0x49}, {0x12D, 0x69}, {0x12E, 0x49},
    {0x12F, 0x69}, {0x130, 0x49}, {0x134, 0x4A}, {0x135, 0x6A}, {0x136, 0x4B}, {0x137, 0x6B},
    {0x139, 0x4C}, {0x13A, 0x6C}, {0x13B, 0x4C}, {0x13C, 0x6C}, {0x13D, 0x4C}, {0x13E, 0x6C},
    {0x143, 0x4E}, {0x144, 0x6E}, {0x145, 0x4E}, {0x146, 0x6E}, {0x147, 0x4E}, {0x148, 0x6E},
    {0x14C, 0x4F}, {0x14D, 0x6F}, {0x14E, 0x4F}, {0x14F, 0x6F}, {0x150, 0x4F}, {0x151, 0x6F},
    {0x154, 0x52}, {0x155, 0x72}, {0x156, 0x52}, {0x157, 0x72}, {0x158, 0x52}, {0x159, 0x72},
    {0x15A, 0x53}, {0x15B, 0x73}, {0x15C, 0x53}, {0x15D, 0x73}, {0x15E, 0x53}, {0x15F, 0x73},
    {0x160, 0x53}, {0x161, 0x73}, {0x162, 0x54}, {0x163, 0x74}, {0x164, 0x54}, {0x165, 0x74},
    {0x168, 0x55}, {0x169, 0x75}, {0x16A, 0x55}, {0x16B, 0x75}, {0x16C, 0x55}, {0x16D, 0x75},
    {0x16E, 0x55}, {0x16F, 0x75}, {0x170, 0x55}, {0x171, 0x75}, {0x172, 0x55}, {0x173, 0x75},
    {0x174, 0x57}, {0x175, 0x77}, {0x176, 0x59}, {0x177, 0x79}, {0x178, 0x59}, {0x179, 0x5A},
    {0x17A, 0x7A}, {0x17B, 0x5A}, {0x17C, 0x7A}, {0x17D, 0x5A}, {0x17E, 0x7A},
};

inline uint32_t table_lookup(const uint32_t tab[][2], size_t n, uint32_t cp) {
    for (size_t i = 0; i < n; i++)
        if (tab[i][0] == cp) return tab[i][1];
    return cp;
}

class WordPieceTokenizer {
  public:
    // vocab.txt: one token per line, id = line number.
    bool load_vocab(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        std::string all;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
        fclose(f);
        vocab_.clear();
        id2tok_.clear();
        size_t pos = 0;
        while (pos < all.size()) {
            size_t eol = all.find('\n', pos);
            if (eol == std::string::npos) eol = all.size();
            std::string tok = all.substr(pos, eol - pos);
            if (!tok.empty() && tok.back() == '\r') tok.pop_back();
            vocab_[tok] = (int64_t)id2tok_.size();
            id2tok_.push_back(tok);
            pos = eol + 1;
        }
        auto it = vocab_.find("[UNK]");
        unk_id_ = it != vocab_.end() ? it->second : 100;
        cls_id_ = vocab_.count("[CLS]") ? vocab_["[CLS]"] : 101;
        sep_id_ = vocab_.count("[SEP]") ? vocab_["[SEP]"] : 102;
        return true;
    }

    // BertNormalizer: clean (drop control except whitespace,
    // whitespace -> space), spaces around CJK chars, strip
    // accents and lowercase (ASCII + Latin table ranges).
    static std::string normalize(const std::string& text) {
        std::string out;
        for (uint32_t cp : utf8_decode(text)) {
            if (cp == 0xFFFD) continue;
            if (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F)) {
                if (cp == '\t' || cp == '\n' || cp == '\r') out.push_back(' ');
                continue;
            }
            if (is_space(cp)) { utf8_encode(' ', out); continue; }
            if (cp >= 0x300 && cp <= 0x36F) continue;  // combining marks: stripped
            cp = table_lookup(kStripLatin, sizeof(kStripLatin) / sizeof(kStripLatin[0]), cp);
            if (cp >= 'A' && cp <= 'Z') cp += 'a' - 'A';
            else cp = table_lookup(kLowerLatin, sizeof(kLowerLatin) / sizeof(kLowerLatin[0]), cp);
            if (is_cjk(cp)) {
                out.push_back(' ');
                utf8_encode(cp, out);
                out.push_back(' ');
            } else {
                utf8_encode(cp, out);
            }
        }
        return out;
    }

    // BertPreTokenizer + WordPiece + special tokens.
    void encode(const std::string& text, std::vector<int64_t>& ids,
                std::vector<int64_t>& mask) const {
        ids.clear();
        mask.clear();
        ids.push_back(cls_id_);
        std::string norm = normalize(text);
        // Pre-tokenize on codepoints: whitespace separates,
        // punctuation is its own token.
        std::vector<uint32_t> cps = utf8_decode(norm);
        std::vector<uint32_t> cur;
        auto flush = [&]() {
            if (cur.empty()) return;
            wordpiece(cur, ids);
            cur.clear();
        };
        for (uint32_t cp : cps) {
            if (cp == ' ') { flush(); continue; }
            if (is_punct(cp)) {
                flush();
                std::vector<uint32_t> one{cp};
                wordpiece(one, ids);
                continue;
            }
            cur.push_back(cp);
        }
        flush();
        ids.push_back(sep_id_);
        mask.assign(ids.size(), 1);
    }

  private:
    void wordpiece(const std::vector<uint32_t>& word, std::vector<int64_t>& ids) const {
        if (word.size() > 100) { ids.push_back(unk_id_); return; }
        std::vector<int64_t> out;
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            int64_t found = -1;
            size_t found_end = start;
            while (end > start) {
                std::string sub;
                if (start > 0) sub = "##";
                for (size_t i = start; i < end; i++) utf8_encode(word[i], sub);
                auto it = vocab_.find(sub);
                if (it != vocab_.end()) { found = it->second; found_end = end; break; }
                end--;
            }
            if (found < 0) { ids.push_back(unk_id_); return; }
            out.push_back(found);
            start = found_end;
        }
        ids.insert(ids.end(), out.begin(), out.end());
    }

    std::unordered_map<std::string, int64_t> vocab_;
    std::vector<std::string> id2tok_;
    int64_t unk_id_ = 100, cls_id_ = 101, sep_id_ = 102;
};

}  // namespace wptok
