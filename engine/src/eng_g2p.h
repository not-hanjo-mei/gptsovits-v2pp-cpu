// English G2P for runtime reference extraction: a port of
// genie_tts's English path (G2P/English/{EnglishG2P,
// Normalization, WordSegment}.py + nltk TweetTokenizer +
// the averaged-perceptron pos_tag), mirroring upstream
// behavior including its quirks (e.g. _ordinal_custom
// glues the suffix onto the number words). Verified
// stage-by-stage against Genie oracles (eng_g2p_check).
//
// Assets (host/refextract/eng/, see tools/export_eng_g2p.py
// and tools/REFEXTRACT_ASSETS.md):
//   eng.bin          merged CMU dict + name dict + GRU weights
//   tagger.bin       perceptron tagger, flattened from its JSONs
//   wordsegment/{unigrams,bigrams,words}.txt (verbatim)
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json_min.h"

namespace eng2p {

inline std::wstring u8_to_w(const std::string& s) {
    std::wstring o;
    for (size_t i = 0; i < s.size();) {
        uint8_t c = (uint8_t)s[i];
        if (c < 0x80) { o += (wchar_t)c; i += 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) { o += (wchar_t)(((c & 0x1F) << 6) | ((uint8_t)s[i + 1] & 0x3F)); i += 2; }
        else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) { o += (wchar_t)(((c & 0x0F) << 12) | (((uint8_t)s[i + 1] & 0x3F) << 6) | ((uint8_t)s[i + 2] & 0x3F)); i += 3; }
        else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            uint32_t cp = ((c & 0x07) << 18) | (((uint8_t)s[i + 1] & 0x3F) << 12) | (((uint8_t)s[i + 2] & 0x3F) << 6) | ((uint8_t)s[i + 3] & 0x3F);
            o += (wchar_t)cp; i += 4;
        } else { o += (wchar_t)0xFFFD; i += 1; }
    }
    return o;
}

inline std::string w_to_u8(const std::wstring& w) {
    std::string o;
    for (wchar_t wc : w) {
        uint32_t c = (uint32_t)wc;
        if (c < 0x80) o += (char)c;
        else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
        else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    }
    return o;
}

inline std::string read_file_str(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline bool is_word_cp(uint32_t c) {  // Python re \w, approximated for this pipeline
    if (c < 0x80) return std::isalnum((int)c) || c == '_';
    if (c >= 0x3400 && c <= 0x9FFF) return true;
    return (c >= 0xC0 && c <= 0x24F) || (c >= 0x1E00 && c <= 0x1EFF);
}
inline bool is_letter_cp(uint32_t c) {  // [^\W\d_] for the tokenizer
    if (c < 0x80) return std::isalpha((int)c) != 0;
    if (c >= 0x3400 && c <= 0x9FFF) return true;
    return (c >= 0xC0 && c <= 0x24F) || (c >= 0x1E00 && c <= 0x1EFF);
}
inline bool is_digit_cp(uint32_t c) { return c >= '0' && c <= '9'; }
inline bool is_space_cp(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' || c == 0xA0 || c == 0x3000;
}
inline wchar_t lower_cp(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return (wchar_t)(c + 32);
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return (wchar_t)(c + 32);
    if (c >= 0x100 && c <= 0x12E && !(c & 1)) return (wchar_t)(c + 1);
    if (c >= 0x132 && c <= 0x136 && (c & 1)) return (wchar_t)(c + 1);
    return (wchar_t)c;
}
inline std::wstring lower_w(const std::wstring& s) {
    std::wstring o = s;
    for (auto& c : o) c = lower_cp((uint32_t)c);
    return o;
}

// ---- NFD accent strip (Normalization.normalize's first step) ----
// Canonical decompositions whose base is an ASCII letter and
// whose remaining parts are combining marks, generated from
// Python's unicodedata; the marks themselves are dropped.
inline const std::pair<uint32_t, uint32_t> kNfdBase[] = {
    {0x00C0,0x41}, {0x00C1,0x41}, {0x00C2,0x41}, {0x00C3,0x41}, {0x00C4,0x41}, {0x00C5,0x41}, {0x00C7,0x43}, {0x00C8,0x45},
    {0x00C9,0x45}, {0x00CA,0x45}, {0x00CB,0x45}, {0x00CC,0x49}, {0x00CD,0x49}, {0x00CE,0x49}, {0x00CF,0x49}, {0x00D1,0x4E},
    {0x00D2,0x4F}, {0x00D3,0x4F}, {0x00D4,0x4F}, {0x00D5,0x4F}, {0x00D6,0x4F}, {0x00D9,0x55}, {0x00DA,0x55}, {0x00DB,0x55},
    {0x00DC,0x55}, {0x00DD,0x59}, {0x00E0,0x61}, {0x00E1,0x61}, {0x00E2,0x61}, {0x00E3,0x61}, {0x00E4,0x61}, {0x00E5,0x61},
    {0x00E7,0x63}, {0x00E8,0x65}, {0x00E9,0x65}, {0x00EA,0x65}, {0x00EB,0x65}, {0x00EC,0x69}, {0x00ED,0x69}, {0x00EE,0x69},
    {0x00EF,0x69}, {0x00F1,0x6E}, {0x00F2,0x6F}, {0x00F3,0x6F}, {0x00F4,0x6F}, {0x00F5,0x6F}, {0x00F6,0x6F}, {0x00F9,0x75},
    {0x00FA,0x75}, {0x00FB,0x75}, {0x00FC,0x75}, {0x00FD,0x79}, {0x00FF,0x79}, {0x0100,0x41}, {0x0101,0x61}, {0x0102,0x41},
    {0x0103,0x61}, {0x0104,0x41}, {0x0105,0x61}, {0x0106,0x43}, {0x0107,0x63}, {0x0108,0x43}, {0x0109,0x63}, {0x010A,0x43},
    {0x010B,0x63}, {0x010C,0x43}, {0x010D,0x63}, {0x010E,0x44}, {0x010F,0x64}, {0x0112,0x45}, {0x0113,0x65}, {0x0114,0x45},
    {0x0115,0x65}, {0x0116,0x45}, {0x0117,0x65}, {0x0118,0x45}, {0x0119,0x65}, {0x011A,0x45}, {0x011B,0x65}, {0x011C,0x47},
    {0x011D,0x67}, {0x011E,0x47}, {0x011F,0x67}, {0x0120,0x47}, {0x0121,0x67}, {0x0122,0x47}, {0x0123,0x67}, {0x0124,0x48},
    {0x0125,0x68}, {0x0128,0x49}, {0x0129,0x69}, {0x012A,0x49}, {0x012B,0x69}, {0x012C,0x49}, {0x012D,0x69}, {0x012E,0x49},
    {0x012F,0x69}, {0x0130,0x49}, {0x0134,0x4A}, {0x0135,0x6A}, {0x0136,0x4B}, {0x0137,0x6B}, {0x0139,0x4C}, {0x013A,0x6C},
    {0x013B,0x4C}, {0x013C,0x6C}, {0x013D,0x4C}, {0x013E,0x6C}, {0x0143,0x4E}, {0x0144,0x6E}, {0x0145,0x4E}, {0x0146,0x6E},
    {0x0147,0x4E}, {0x0148,0x6E}, {0x014C,0x4F}, {0x014D,0x6F}, {0x014E,0x4F}, {0x014F,0x6F}, {0x0150,0x4F}, {0x0151,0x6F},
    {0x0154,0x52}, {0x0155,0x72}, {0x0156,0x52}, {0x0157,0x72}, {0x0158,0x52}, {0x0159,0x72}, {0x015A,0x53}, {0x015B,0x73},
    {0x015C,0x53}, {0x015D,0x73}, {0x015E,0x53}, {0x015F,0x73}, {0x0160,0x53}, {0x0161,0x73}, {0x0162,0x54}, {0x0163,0x74},
    {0x0164,0x54}, {0x0165,0x74}, {0x0168,0x55}, {0x0169,0x75}, {0x016A,0x55}, {0x016B,0x75}, {0x016C,0x55}, {0x016D,0x75},
    {0x016E,0x55}, {0x016F,0x75}, {0x0170,0x55}, {0x0171,0x75}, {0x0172,0x55}, {0x0173,0x75}, {0x0174,0x57}, {0x0175,0x77},
    {0x0176,0x59}, {0x0177,0x79}, {0x0178,0x59}, {0x0179,0x5A}, {0x017A,0x7A}, {0x017B,0x5A}, {0x017C,0x7A}, {0x017D,0x5A},
    {0x017E,0x7A}, {0x01A0,0x4F}, {0x01A1,0x6F}, {0x01AF,0x55}, {0x01B0,0x75}, {0x01CD,0x41}, {0x01CE,0x61}, {0x01CF,0x49},
    {0x01D0,0x69}, {0x01D1,0x4F}, {0x01D2,0x6F}, {0x01D3,0x55}, {0x01D4,0x75}, {0x01D5,0x55}, {0x01D6,0x75}, {0x01D7,0x55},
    {0x01D8,0x75}, {0x01D9,0x55}, {0x01DA,0x75}, {0x01DB,0x55}, {0x01DC,0x75}, {0x01DE,0x41}, {0x01DF,0x61}, {0x01E0,0x41},
    {0x01E1,0x61}, {0x01E6,0x47}, {0x01E7,0x67}, {0x01E8,0x4B}, {0x01E9,0x6B}, {0x01EA,0x4F}, {0x01EB,0x6F}, {0x01EC,0x4F},
    {0x01ED,0x6F}, {0x01F0,0x6A}, {0x01F4,0x47}, {0x01F5,0x67}, {0x01F8,0x4E}, {0x01F9,0x6E}, {0x01FA,0x41}, {0x01FB,0x61},
    {0x0200,0x41}, {0x0201,0x61}, {0x0202,0x41}, {0x0203,0x61}, {0x0204,0x45}, {0x0205,0x65}, {0x0206,0x45}, {0x0207,0x65},
    {0x0208,0x49}, {0x0209,0x69}, {0x020A,0x49}, {0x020B,0x69}, {0x020C,0x4F}, {0x020D,0x6F}, {0x020E,0x4F}, {0x020F,0x6F},
    {0x0210,0x52}, {0x0211,0x72}, {0x0212,0x52}, {0x0213,0x72}, {0x0214,0x55}, {0x0215,0x75}, {0x0216,0x55}, {0x0217,0x75},
    {0x0218,0x53}, {0x0219,0x73}, {0x021A,0x54}, {0x021B,0x74}, {0x021E,0x48}, {0x021F,0x68}, {0x0226,0x41}, {0x0227,0x61},
    {0x0228,0x45}, {0x0229,0x65}, {0x022A,0x4F}, {0x022B,0x6F}, {0x022C,0x4F}, {0x022D,0x6F}, {0x022E,0x4F}, {0x022F,0x6F},
    {0x0230,0x4F}, {0x0231,0x6F}, {0x0232,0x59}, {0x0233,0x79}, {0x1E00,0x41}, {0x1E01,0x61}, {0x1E02,0x42}, {0x1E03,0x62},
    {0x1E04,0x42}, {0x1E05,0x62}, {0x1E06,0x42}, {0x1E07,0x62}, {0x1E08,0x43}, {0x1E09,0x63}, {0x1E0A,0x44}, {0x1E0B,0x64},
    {0x1E0C,0x44}, {0x1E0D,0x64}, {0x1E0E,0x44}, {0x1E0F,0x64}, {0x1E10,0x44}, {0x1E11,0x64}, {0x1E12,0x44}, {0x1E13,0x64},
    {0x1E14,0x45}, {0x1E15,0x65}, {0x1E16,0x45}, {0x1E17,0x65}, {0x1E18,0x45}, {0x1E19,0x65}, {0x1E1A,0x45}, {0x1E1B,0x65},
    {0x1E1C,0x45}, {0x1E1D,0x65}, {0x1E1E,0x46}, {0x1E1F,0x66}, {0x1E20,0x47}, {0x1E21,0x67}, {0x1E22,0x48}, {0x1E23,0x68},
    {0x1E24,0x48}, {0x1E25,0x68}, {0x1E26,0x48}, {0x1E27,0x68}, {0x1E28,0x48}, {0x1E29,0x68}, {0x1E2A,0x48}, {0x1E2B,0x68},
    {0x1E2C,0x49}, {0x1E2D,0x69}, {0x1E2E,0x49}, {0x1E2F,0x69}, {0x1E30,0x4B}, {0x1E31,0x6B}, {0x1E32,0x4B}, {0x1E33,0x6B},
    {0x1E34,0x4B}, {0x1E35,0x6B}, {0x1E36,0x4C}, {0x1E37,0x6C}, {0x1E38,0x4C}, {0x1E39,0x6C}, {0x1E3A,0x4C}, {0x1E3B,0x6C},
    {0x1E3C,0x4C}, {0x1E3D,0x6C}, {0x1E3E,0x4D}, {0x1E3F,0x6D}, {0x1E40,0x4D}, {0x1E41,0x6D}, {0x1E42,0x4D}, {0x1E43,0x6D},
    {0x1E44,0x4E}, {0x1E45,0x6E}, {0x1E46,0x4E}, {0x1E47,0x6E}, {0x1E48,0x4E}, {0x1E49,0x6E}, {0x1E4A,0x4E}, {0x1E4B,0x6E},
    {0x1E4C,0x4F}, {0x1E4D,0x6F}, {0x1E4E,0x4F}, {0x1E4F,0x6F}, {0x1E50,0x4F}, {0x1E51,0x6F}, {0x1E52,0x4F}, {0x1E53,0x6F},
    {0x1E54,0x50}, {0x1E55,0x70}, {0x1E56,0x50}, {0x1E57,0x70}, {0x1E58,0x52}, {0x1E59,0x72}, {0x1E5A,0x52}, {0x1E5B,0x72},
    {0x1E5C,0x52}, {0x1E5D,0x72}, {0x1E5E,0x52}, {0x1E5F,0x72}, {0x1E60,0x53}, {0x1E61,0x73}, {0x1E62,0x53}, {0x1E63,0x73},
    {0x1E64,0x53}, {0x1E65,0x73}, {0x1E66,0x53}, {0x1E67,0x73}, {0x1E68,0x53}, {0x1E69,0x73}, {0x1E6A,0x54}, {0x1E6B,0x74},
    {0x1E6C,0x54}, {0x1E6D,0x74}, {0x1E6E,0x54}, {0x1E6F,0x74}, {0x1E70,0x54}, {0x1E71,0x74}, {0x1E72,0x55}, {0x1E73,0x75},
    {0x1E74,0x55}, {0x1E75,0x75}, {0x1E76,0x55}, {0x1E77,0x75}, {0x1E78,0x55}, {0x1E79,0x75}, {0x1E7A,0x55}, {0x1E7B,0x75},
    {0x1E7C,0x56}, {0x1E7D,0x76}, {0x1E7E,0x56}, {0x1E7F,0x76}, {0x1E80,0x57}, {0x1E81,0x77}, {0x1E82,0x57}, {0x1E83,0x77},
    {0x1E84,0x57}, {0x1E85,0x77}, {0x1E86,0x57}, {0x1E87,0x77}, {0x1E88,0x57}, {0x1E89,0x77}, {0x1E8A,0x58}, {0x1E8B,0x78},
    {0x1E8C,0x58}, {0x1E8D,0x78}, {0x1E8E,0x59}, {0x1E8F,0x79}, {0x1E90,0x5A}, {0x1E91,0x7A}, {0x1E92,0x5A}, {0x1E93,0x7A},
    {0x1E94,0x5A}, {0x1E95,0x7A}, {0x1E96,0x68}, {0x1E97,0x74}, {0x1E98,0x77}, {0x1E99,0x79}, {0x1EA0,0x41}, {0x1EA1,0x61},
    {0x1EA2,0x41}, {0x1EA3,0x61}, {0x1EA4,0x41}, {0x1EA5,0x61}, {0x1EA6,0x41}, {0x1EA7,0x61}, {0x1EA8,0x41}, {0x1EA9,0x61},
    {0x1EAA,0x41}, {0x1EAB,0x61}, {0x1EAC,0x41}, {0x1EAD,0x61}, {0x1EAE,0x41}, {0x1EAF,0x61}, {0x1EB0,0x41}, {0x1EB1,0x61},
    {0x1EB2,0x41}, {0x1EB3,0x61}, {0x1EB4,0x41}, {0x1EB5,0x61}, {0x1EB6,0x41}, {0x1EB7,0x61}, {0x1EB8,0x45}, {0x1EB9,0x65},
    {0x1EBA,0x45}, {0x1EBB,0x65}, {0x1EBC,0x45}, {0x1EBD,0x65}, {0x1EBE,0x45}, {0x1EBF,0x65}, {0x1EC0,0x45}, {0x1EC1,0x65},
    {0x1EC2,0x45}, {0x1EC3,0x65}, {0x1EC4,0x45}, {0x1EC5,0x65}, {0x1EC6,0x45}, {0x1EC7,0x65}, {0x1EC8,0x49}, {0x1EC9,0x69},
    {0x1ECA,0x49}, {0x1ECB,0x69}, {0x1ECC,0x4F}, {0x1ECD,0x6F}, {0x1ECE,0x4F}, {0x1ECF,0x6F}, {0x1ED0,0x4F}, {0x1ED1,0x6F},
    {0x1ED2,0x4F}, {0x1ED3,0x6F}, {0x1ED4,0x4F}, {0x1ED5,0x6F}, {0x1ED6,0x4F}, {0x1ED7,0x6F}, {0x1ED8,0x4F}, {0x1ED9,0x6F},
    {0x1EDA,0x4F}, {0x1EDB,0x6F}, {0x1EDC,0x4F}, {0x1EDD,0x6F}, {0x1EDE,0x4F}, {0x1EDF,0x6F}, {0x1EE0,0x4F}, {0x1EE1,0x6F},
    {0x1EE2,0x4F}, {0x1EE3,0x6F}, {0x1EE4,0x55}, {0x1EE5,0x75}, {0x1EE6,0x55}, {0x1EE7,0x75}, {0x1EE8,0x55}, {0x1EE9,0x75},
    {0x1EEA,0x55}, {0x1EEB,0x75}, {0x1EEC,0x55}, {0x1EED,0x75}, {0x1EEE,0x55}, {0x1EEF,0x75}, {0x1EF0,0x55}, {0x1EF1,0x75},
    {0x1EF2,0x59}, {0x1EF3,0x79}, {0x1EF4,0x59}, {0x1EF5,0x79}, {0x1EF6,0x59}, {0x1EF7,0x79}, {0x1EF8,0x59}, {0x1EF9,0x79},
    {0x212B,0x41},
};
inline bool is_mn_cp(uint32_t c) {
    return (c >= 0x300 && c <= 0x36F) || (c >= 0x483 && c <= 0x489) || (c >= 0x591 && c <= 0x5BD) ||
           (c >= 0x610 && c <= 0x61A) || (c >= 0x64B && c <= 0x65F) || c == 0x670 ||
           (c >= 0x6D6 && c <= 0x6DC) || (c >= 0x900 && c <= 0x902) || (c >= 0x93A && c <= 0x94F) ||
           (c >= 0x1AB0 && c <= 0x1AFF) || (c >= 0x1DC0 && c <= 0x1DFF) || (c >= 0x20D0 && c <= 0x20F0) ||
           (c >= 0x3099 && c <= 0x309A) || (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFE20 && c <= 0xFE2F);
}
inline std::wstring nfd_strip(const std::wstring& s) {
    static const std::unordered_map<uint32_t, uint32_t> tab = [] {
        std::unordered_map<uint32_t, uint32_t> m;
        for (auto& p : kNfdBase) m[p.first] = p.second;
        return m;
    }();
    std::wstring o;
    for (wchar_t wc : s) {
        uint32_t c = (uint32_t)wc;
        auto it = tab.find(c);
        if (it != tab.end()) o += (wchar_t)it->second;
        else if (!is_mn_cp(c)) o += wc;
    }
    return o;
}

// ---- number words (Normalization._number_to_words_custom etc.) ----
inline bool all_digits(const std::wstring& s) {
    if (s.empty()) return false;
    for (wchar_t c : s) if (!is_digit_cp((uint32_t)c)) return false;
    return true;
}
inline std::wstring number_to_words(const std::wstring& num_str_in) {
    std::wstring num_str = num_str_in;
    size_t b = num_str.find_first_not_of(L" \t\n\r");
    size_t e = num_str.find_last_not_of(L" \t\n\r");
    num_str = (b == std::wstring::npos) ? L"" : num_str.substr(b, e - b + 1);
    if (!all_digits(num_str)) return num_str;
    long long num = 0;
    for (wchar_t c : num_str) {
        num = num * 10 + (c - '0');
        if (num > (long long)4e15) throw std::runtime_error("eng norm: number too large");
    }
    if (num == 0) return L"zero";
    static const wchar_t* units[] = {L"", L"one", L"two", L"three", L"four", L"five", L"six", L"seven", L"eight", L"nine"};
    static const wchar_t* teens[] = {L"ten", L"eleven", L"twelve", L"thirteen", L"fourteen", L"fifteen", L"sixteen", L"seventeen", L"eighteen", L"nineteen"};
    static const wchar_t* tens[] = {L"", L"", L"twenty", L"thirty", L"forty", L"fifty", L"sixty", L"seventy", L"eighty", L"ninety"};
    static const wchar_t* thousands[] = {L"", L"thousand", L"million", L"billion", L"trillion"};
    std::function<std::wstring(long long)> lt1000 = [&](long long n) -> std::wstring {
        if (n == 0) return L"";
        if (n < 10) return units[n];
        if (n < 20) return teens[n - 10];
        if (n < 100) return std::wstring(tens[n / 10]) + (n % 10 ? L" " + std::wstring(units[n % 10]) : L"");
        std::wstring r = std::wstring(units[n / 100]) + L" hundred";
        if (n % 100) r += L" " + lt1000(n % 100);
        return r;
    };
    std::vector<std::wstring> words;
    long long n = num;
    int i = 0;
    while (n > 0) {
        if (n % 1000 != 0) {
            if (i > 4) throw std::runtime_error("eng norm: number too large");
            std::wstring w = lt1000(n % 1000) + L" " + thousands[i];
            words.insert(words.begin(), w);
        }
        n /= 1000;
        i++;
    }
    std::wstring r;
    for (size_t k = 0; k < words.size(); k++) { if (k) r += L" "; r += words[k]; }
    size_t b2 = r.find_first_not_of(L' '), e2 = r.find_last_not_of(L' ');
    return (b2 == std::wstring::npos) ? L"" : r.substr(b2, e2 - b2 + 1);
}
inline std::wstring ordinal_custom(const std::wstring& num_str) {
    long long num = 0;
    for (wchar_t c : num_str) num = num * 10 + (c - '0');
    std::wstring suffix;
    if (num % 100 >= 10 && num % 100 <= 20) suffix = L"th";
    else if (num % 10 == 1) suffix = L"st";
    else if (num % 10 == 2) suffix = L"nd";
    else if (num % 10 == 3) suffix = L"rd";
    else suffix = L"th";
    return number_to_words(num_str) + suffix;  // upstream glues the suffix on
}
inline std::wstring expand_number_positive(const std::wstring& num_str) {
    long long num = 0;
    for (wchar_t c : num_str) num = num * 10 + (c - '0');
    if (num >= 2000 && num < 2010)
        return L"two thousand and " + number_to_words(std::to_wstring(num % 100));
    if (num >= 1100 && num < 2100 && num % 100 != 0)
        return number_to_words(std::to_wstring(num / 100)) + L" " + number_to_words(std::to_wstring(num % 100));
    return number_to_words(num_str);
}

// ---- re.sub machinery (Python semantics: left-to-right,
// replacement not rescanned; `accept` plays the role of
// lookarounds std::regex cannot express) ----
using SubFn = std::function<std::wstring(const std::wsmatch&)>;
using AcceptFn = std::function<bool(const std::wstring&, size_t, size_t)>;
inline std::wstring re_sub(const std::wstring& text, const std::wregex& re, const SubFn& fn,
                           const AcceptFn& accept = nullptr) {
    std::wstring out;
    size_t pos = 0;
    while (pos <= text.size()) {
        auto first = text.begin() + (long)pos;
        std::wsmatch m;
        if (!std::regex_search(first, text.end(), m, re)) { out += text.substr(pos); break; }
        size_t start = pos + (size_t)m.position();
        size_t len = (size_t)m.length();
        if (accept && !accept(text, start, start + len)) {
            out += text.substr(pos, start - pos + 1);
            pos = start + 1;
            continue;
        }
        out += text.substr(pos, start - pos);
        out += fn(m);
        pos = start + len;
        if (len == 0) { if (pos < text.size()) out += text[pos]; pos++; }
    }
    return out;
}
inline std::wregex wre(const wchar_t* pat, bool icase = false) {
    auto flags = std::regex_constants::ECMAScript;
    if (icase) flags |= std::regex_constants::icase;
    return std::wregex(pat, flags);
}
inline std::wstring group_or(const std::wsmatch& m, size_t i) {
    return i < m.size() && m[i].matched ? m[i].str() : L"";
}
inline bool word_before(const std::wstring& t, size_t i) { return i > 0 && is_word_cp((uint32_t)t[i - 1]); }
inline bool word_at(const std::wstring& t, size_t i) { return i < t.size() && is_word_cp((uint32_t)t[i]); }

inline std::wstring expand_fraction_str(const std::wstring& ns, const std::wstring& ds, const std::wstring& whole) {
    long long n = 0, d = 0;
    for (wchar_t c : ns) n = n * 10 + (c - '0');
    for (wchar_t c : ds) d = d * 10 + (c - '0');
    if (d == 0) return whole;
    if (n == 1 && d == 2) return L"one half";
    if (n == 1 && d == 4) return L"one quarter";
    if (n == 3 && d == 4) return L"three quarters";
    return number_to_words(ns) + L" over " + number_to_words(ds);
}

// ---- Normalization.normalize ----
inline std::wstring eng_normalize(std::wstring text) {
    text = nfd_strip(text);
    // "@" -> " at "
    {
        std::wstring o;
        for (wchar_t c : text) { if (c == L'@') o += L" at "; else o += c; }
        text = o;
    }
    static const std::pair<const wchar_t*, const wchar_t*> abbrs[] = {
        {L"Mr", L"Mister"}, {L"Mrs", L"Missus"}, {L"Dr", L"Doctor"}, {L"Prof", L"Professor"},
        {L"St", L"Street"}, {L"Co", L"Company"}, {L"Ltd", L"Limited"},
        {L"e\\.g", L"for example"}, {L"i\\.e", L"that is"},
    };
    for (auto& ab : abbrs) {
        std::wstring pat = std::wstring(L"\\b") + ab.first + L"\\.(?=[\\s,.]|$)";
        std::wstring rep = ab.second;
        text = re_sub(text, wre(pat.c_str(), true), [&](const std::wsmatch&) { return rep; });
    }
    text = re_sub(text, wre(L"([£$€])([\\d,.]*\\d)\\s*(million|billion|thousand)\\b", true),
        [](const std::wsmatch& m) {
            std::wstring amt = m[2].str();
            amt.erase(std::remove(amt.begin(), amt.end(), L','), amt.end());
            std::wstring cur = m[1].str() == L"$" ? L"dollars" : m[1].str() == L"£" ? L"pounds" : L"euros";
            return number_to_words(amt) + L" " + m[3].str() + L" " + cur;
        });
    text = re_sub(text, wre(L"(\\+?\\d{1,3}-)?\\b(\\d{3})-(?:(\\d{3})-)?(\\d{4})\\b"),
        [](const std::wsmatch& m) {
            std::vector<std::wstring> parts;
            auto per_digit = [](const std::wstring& s) {
                std::wstring r;
                for (wchar_t c : s) { if (!r.empty()) r += L" "; r += number_to_words(std::wstring(1, c)); }
                return r;
            };
            if (m[1].matched) {
                std::wstring country = m[1].str();
                std::wstring cw;
                if (!country.empty() && country[0] == L'+') cw = L"plus";
                std::wstring digits;
                for (wchar_t c : country) if (is_digit_cp((uint32_t)c)) digits += c;
                if (!digits.empty()) { if (!cw.empty()) cw += L" "; cw += per_digit(digits); }
                parts.push_back(cw);
            }
            parts.push_back(per_digit(m[2].str()));
            if (m[3].matched) parts.push_back(per_digit(m[3].str()));
            parts.push_back(per_digit(m[4].str()));
            std::wstring r;
            for (size_t i = 0; i < parts.size(); i++) { if (i) r += L", "; r += parts[i]; }
            return r;
        });
    text = re_sub(text, wre(L"\\b(\\d+(?:\\.\\d+)?)\\s*x\\s*(\\d+(?:\\.\\d+)?)(?:\\s*x\\s*(\\d+(?:\\.\\d+)?))?\\b"),
        [](const std::wsmatch& m) {
            std::wstring r;
            for (size_t i = 1; i <= 3; i++) {
                if (i < m.size() && m[i].matched) { if (!r.empty()) r += L" by "; r += number_to_words(m[i].str()); }
            }
            return r;
        });
    static const std::map<std::wstring, std::wstring> roman_map = {
        {L"ii", L"two"}, {L"iii", L"three"}, {L"iv", L"four"}, {L"v", L"five"}, {L"vi", L"six"},
        {L"vii", L"seven"}, {L"viii", L"eight"}, {L"ix", L"nine"}, {L"x", L"ten"}, {L"xi", L"eleven"},
        {L"xii", L"twelve"}, {L"xiii", L"thirteen"}, {L"xiv", L"fourteen"}, {L"xv", L"fifteen"},
        {L"xvi", L"sixteen"}, {L"xvii", L"seventeen"}, {L"xviii", L"eighteen"}, {L"xix", L"nineteen"}};
    text = re_sub(text, wre(L"\\b(XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II)\\b", true),
        [](const std::wsmatch& m) {
            auto it = roman_map.find(lower_w(m[1].str()));
            return it == roman_map.end() ? m[1].str() : it->second;
        });
    text = re_sub(text, wre(L"\\b((?:1[89]|20)\\d0)s\\b"),
        [](const std::wsmatch& m) {
            std::wstring yw = expand_number_positive(m[1].str());
            if (yw.size() >= 2 && yw.substr(yw.size() - 2) == L"ty") return yw.substr(0, yw.size() - 1) + L"ies";
            return yw + L"s";
        });
    text = re_sub(text, wre(L"\\b(\\d{1,2})-(\\d{1,2})\\b"),
        [](const std::wsmatch& m) { return number_to_words(m[1].str()) + L" to " + number_to_words(m[2].str()); });
    static const wchar_t* month_names[] = {L"", L"January", L"February", L"March", L"April", L"May", L"June",
                                           L"July", L"August", L"September", L"October", L"November", L"December"};
    text = re_sub(text, wre(L"\\b(0?[1-9]|1[0-2])/([0-2]?\\d|3[01])/(\\d{2,4})\\b"),
        [](const std::wsmatch& m) {
            int mon = 0;
            for (wchar_t c : m[1].str()) mon = mon * 10 + (c - '0');
            std::wstring ys = m[3].str();
            long long ynum = 0;
            for (wchar_t c : ys) ynum = ynum * 10 + (c - '0');
            if (ys.size() == 2) ynum += (ynum < 50 ? 2000 : 1900);
            return std::wstring(month_names[mon]) + L" " + ordinal_custom(m[2].str()) + L", " +
                   expand_number_positive(std::to_wstring(ynum));
        });
    text = re_sub(text, wre(L"\\b([01]?\\d|2[0-3]):([0-5]\\d)(?::([0-5]\\d))?(\\s*(?:a\\.?m\\.?|p\\.?m\\.?))?\\b", true),
        [](const std::wsmatch& m) {
            long long h = 0, mi = 0;
            for (wchar_t c : m[1].str()) h = h * 10 + (c - '0');
            for (wchar_t c : m[2].str()) mi = mi * 10 + (c - '0');
            std::wstring ampm = group_or(m, 4);
            bool has_ampm = m[4].matched;
            std::wstring hw = number_to_words(std::to_wstring((h <= 12 || !has_ampm) ? h : h - 12));
            if (h == 0 && has_ampm) hw = L"twelve";
            std::wstring mw;
            if (mi > 0) mw = (mi < 10 ? L" oh " : L" ") + number_to_words(std::to_wstring(mi));
            std::wstring r = hw + mw;
            size_t b = r.find_first_not_of(L' ');
            r = (b == std::wstring::npos) ? L"" : r.substr(b);
            if (m[3].matched) {
                long long s = 0;
                for (wchar_t c : m[3].str()) s = s * 10 + (c - '0');
                r += L" and " + number_to_words(std::to_wstring(s)) + L" seconds";
            }
            if (has_ampm) r += (lower_w(ampm).find(L'p') != std::wstring::npos ? L" pm" : L" am");
            return r;
        });
    text = re_sub(text, wre(L"\\b(\\d+)\\. "),
        [](const std::wsmatch& m) { return ordinal_custom(m[1].str()) + L", "; });
    text = re_sub(text, wre(L"(\\d[\\d,]+\\d)"),
        [](const std::wsmatch& m) {
            std::wstring s = m[1].str();
            s.erase(std::remove(s.begin(), s.end(), L','), s.end());
            return s;
        });
    text = re_sub(text, wre(L"([£$€])(\\d*\\.?\\d+)|(\\d*\\.?\\d+)\\s*([£$€])"),
        [](const std::wsmatch& m) {
            std::wstring sym = m[1].matched ? m[1].str() : m[4].str();
            std::wstring amt = m[1].matched ? m[2].str() : m[3].str();
            amt.erase(std::remove(amt.begin(), amt.end(), L','), amt.end());
            if (!amt.empty() && amt[0] == L'.') amt = L"0" + amt;
            std::wstring major_s, major_p, minor_s, minor_p;
            if (sym == L"$") { major_s = L"dollar"; major_p = L"dollars"; minor_s = L"cent"; minor_p = L"cents"; }
            else if (sym == L"£") { major_s = L"pound"; major_p = L"pounds"; minor_s = L"penny"; minor_p = L"pence"; }
            else { major_s = L"euro"; major_p = L"euros"; minor_s = L"cent"; minor_p = L"cents"; }
            std::wstring ip = amt, fp;
            size_t dot = amt.find(L'.');
            if (dot != std::wstring::npos) { ip = amt.substr(0, dot); fp = amt.substr(dot + 1); }
            long long major_v = 0;
            for (wchar_t c : ip) major_v = major_v * 10 + (c - '0');
            long long minor_v = 0;
            if (dot != std::wstring::npos && !fp.empty()) {
                while (fp.size() < 2) fp += L'0';
                for (wchar_t c : fp) minor_v = minor_v * 10 + (c - '0');
            }
            std::vector<std::wstring> parts;
            if (major_v > 0) parts.push_back(number_to_words(std::to_wstring(major_v)) + L" " + (major_v == 1 ? major_s : major_p));
            if (minor_v > 0) parts.push_back(number_to_words(std::to_wstring(minor_v)) + L" " + (minor_v == 1 ? minor_s : minor_p));
            if (parts.empty()) return L"zero " + major_p;
            std::wstring r = parts[0];
            if (parts.size() > 1) r += L" and " + parts[1];
            return r;
        });
    static const std::map<std::wstring, std::pair<std::wstring, std::wstring>> measure_map = {
        {L"km/h", {L"kilometer per hour", L"kilometers per hour"}}, {L"mph", {L"mile per hour", L"miles per hour"}},
        {L"°C", {L"degree celsius", L"degrees celsius"}}, {L"°F", {L"degree fahrenheit", L"degrees fahrenheit"}},
        {L"tbsp", {L"tablespoon", L"tablespoons"}}, {L"tsp", {L"teaspoon", L"teaspoons"}},
        {L"km", {L"kilometer", L"kilometers"}}, {L"kg", {L"kilogram", L"kilograms"}},
        {L"min", {L"minute", L"minutes"}}, {L"ft", {L"foot", L"feet"}}, {L"cm", {L"centimeter", L"centimeters"}},
        {L"m", {L"meter", L"meters"}}, {L"L", {L"liter", L"liters"}}, {L"h", {L"hour", L"hours"}},
        {L"s", {L"second", L"seconds"}}};
    text = re_sub(text, wre(L"(-?(?:\\d+/\\d+|\\d+(?:\\.\\d+)?))\\s*(km/h|tbsp|mph|tsp|min|°C|°F|km|kg|ft|cm|m|L|h|s)\\b"),
        [](const std::wsmatch& m) {
            std::wstring num = m[1].str();
            bool neg = !num.empty() && num[0] == L'-';
            if (neg) num = num.substr(1);
            std::wstring num_word;
            bool plural;
            size_t slash = num.find(L'/');
            if (slash != std::wstring::npos) {
                num_word = expand_fraction_str(num.substr(0, slash), num.substr(slash + 1), num);
                plural = true;
            } else {
                num_word = number_to_words(num);
                plural = std::stod(w_to_u8(num)) != 1.0;
            }
            auto it = measure_map.find(m[2].str());
            std::wstring r = num_word + L" " + (plural ? it->second.second : it->second.first);
            return neg ? L"minus " + r : r;
        },
        [](const std::wstring& t, size_t s, size_t) { return !word_before(t, s); });
    text = re_sub(text, wre(L"\\b(\\d+)/(\\d+)\\b"),
        [](const std::wsmatch& m) { return expand_fraction_str(m[1].str(), m[2].str(), m[0].str()); });
    text = re_sub(text, wre(L"(\\d+\\.\\d+)"),
        [](const std::wsmatch& m) {
            std::wstring s = m[1].str();
            size_t dot = s.find(L'.');
            std::wstring r = number_to_words(s.substr(0, dot)) + L" point";
            for (wchar_t c : s.substr(dot + 1)) r += L" " + number_to_words(std::wstring(1, c));
            return r;
        });
    text = re_sub(text, wre(L"\\b(\\d+)(st|nd|rd|th)\\b"),
        [](const std::wsmatch& m) { return ordinal_custom(m[1].str()); });
    text = re_sub(text, wre(L"\\b([a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+)\\b"),
        [](const std::wsmatch& m) {
            std::wstring s = m[0].str();
            std::vector<std::wstring> parts;
            for (size_t i = 0; i < s.size();) {
                size_t j = i;
                if (std::isalpha((int)s[i])) { while (j < s.size() && std::isalpha((int)s[j])) j++; }
                else { while (j < s.size() && is_digit_cp((uint32_t)s[j])) j++; }
                std::wstring run = s.substr(i, j - i);
                if (std::isalpha((int)run[0])) {
                    std::wstring r;
                    for (wchar_t c : run) { if (!r.empty()) r += L" "; r += c; }
                    parts.push_back(r);
                } else {
                    std::wstring r;
                    for (wchar_t c : run) { if (!r.empty()) r += L" "; r += number_to_words(std::wstring(1, c)); }
                    parts.push_back(r);
                }
                i = j;
            }
            std::wstring r;
            for (size_t i = 0; i < parts.size(); i++) { if (i) r += L" "; r += parts[i]; }
            return r;
        });
    text = re_sub(text, wre(L"\\b[A-Z]{2,}\\b"),
        [](const std::wsmatch& m) {
            std::wstring r;
            for (wchar_t c : m[0].str()) { if (!r.empty()) r += L" "; r += c; }
            return r;
        });
    text = re_sub(text, wre(L"-?\\d+"),
        [](const std::wsmatch& m) {
            std::wstring s = m[0].str();
            if (!s.empty() && s[0] == L'-') return L"minus " + expand_number_positive(s.substr(1));
            return expand_number_positive(s);
        },
        [](const std::wstring& t, size_t s, size_t e) { return !word_before(t, s) && !word_at(t, e); });
    text = lower_w(text);
    {
        std::wstring o;
        for (wchar_t c : text) { if (c == L'%') o += L" percent"; else o += c; }
        text = o;
    }
    for (int pass = 0; pass < 16; pass++) {
        std::wstring next = re_sub(text, wre(L"\\b([a-z0-9-]+)\\.([a-z]{2,})\\b"),
            [](const std::wsmatch& m) { return m[1].str() + L" dot " + m[2].str(); });
        if (next == text) break;
        text = next;
    }
    {
        std::wstring o;
        for (wchar_t c : text) {
            bool keep = (c >= L'a' && c <= L'z') || is_digit_cp((uint32_t)c) || c == L'\'' || c == L'.' ||
                        c == L',' || c == L'?' || c == L'!' || c == L':' || c == L';' || c == L'-';
            o += keep ? c : L' ';
        }
        text = o;
    }
    text = re_sub(text, wre(L"\\s+"), [](const std::wsmatch&) { return std::wstring(L" "); });
    size_t b = text.find_first_not_of(L' '), e = text.find_last_not_of(L' ');
    return (b == std::wstring::npos) ? L"" : text.substr(b, e - b + 1);
}

// EnglishG2P.text_normalize: rep-map, normalize, collapse
// consecutive punctuation. NOTE: upstream builds its
// rep-map pattern with re.escape() over keys that were
// meant as character classes, so the first two entries
// match only the literal bracketed strings, not individual
// punctuation -- mirrored faithfully (verified against
// Genie: "a;b" keeps its semicolon).
inline std::wstring eng_text_normalize(const std::wstring& in) {
    std::wstring t;
    const std::wstring lit1 = L"[;:\uFF1A\uFF0C\uFF1B]";
    const std::wstring lit2 = L"[\"\u2019]";
    for (size_t i = 0; i < in.size();) {
        if (in.compare(i, lit1.size(), lit1) == 0) { t += L','; i += lit1.size(); continue; }
        if (in.compare(i, lit2.size(), lit2) == 0) { t += L'\''; i += lit2.size(); continue; }
        uint32_t c = (uint32_t)in[i];
        if (c == 0x3002) t += L'.';
        else if (c == 0xFF01) t += L'!';
        else if (c == 0xFF1F) t += L'?';
        else t += in[i];
        i++;
    }
    t = eng_normalize(t);
    t = re_sub(t, wre(L"([!?…,\\.\\s])([!?…,\\.])+"),
               [](const std::wsmatch& m) { return m[1].str(); });
    return t;
}

// ---- nltk TweetTokenizer (casual.py, phone variant) ----
// Ordered alternation tried at each scan position, mirroring
// WORD_RE.findall after the html-entity and HANG_RE passes.
inline std::wstring replace_html_entities(const std::wstring& in) {
    static const std::map<std::wstring, uint32_t> named = {
        {L"amp", '&'}, {L"lt", '<'}, {L"gt", '>'}, {L"quot", '"'}, {L"apos", '\''},
        {L"nbsp", 0xA0}, {L"pound", 0xA3}, {L"euro", 0x20AC}, {L"copy", 0xA9}, {L"reg", 0xAE},
        {L"trade", 0x2122}, {L"hellip", 0x2026}, {L"mdash", 0x2014}, {L"ndash", 0x2013},
        {L"lsquo", 0x2018}, {L"rsquo", 0x2019}, {L"ldquo", 0x201C}, {L"rdquo", 0x201D},
        {L"laquo", 0xAB}, {L"raquo", 0xBB}, {L"deg", 0xB0}, {L"plusmn", 0xB1}, {L"middot", 0xB7},
        {L"cent", 0xA2}, {L"yen", 0xA5}, {L"sect", 0xA7}, {L"uml", 0xA8}, {L"ordf", 0xAA},
        {L"ordm", 0xBA}, {L"para", 0xB6}, {L"micro", 0xB5}, {L"sup1", 0xB9}, {L"frac12", 0xBD},
        {L"times", 0xD7}, {L"divide", 0xF7}, {L"szlig", 0xDF}, {L"agrave", 0xE0}, {L"eacute", 0xE9},
    };
    std::wstring out;
    for (size_t i = 0; i < in.size();) {
        if (in[i] == L'&') {
            size_t semi = in.find(L';', i + 1);
            if (semi != std::wstring::npos && semi - i <= 12 && semi > i + 1) {
                std::wstring body = in.substr(i + 1, semi - i - 1);
                bool ok = true;
                uint32_t cp = 0;
                if (body[0] == L'#') {
                    try {
                        if (body.size() > 2 && (body[1] == L'x' || body[1] == L'X'))
                            cp = (uint32_t)std::stoul(w_to_u8(body.substr(2)), nullptr, 16);
                        else cp = (uint32_t)std::stoul(w_to_u8(body.substr(1)));
                    } catch (...) { ok = false; }
                } else {
                    auto it = named.find(body);
                    if (it != named.end()) cp = it->second; else ok = false;
                }
                if (ok && cp != 0 && cp < 0x110000) out += (wchar_t)cp;
                // unknown/illegal entities are dropped (remove_illegal)
                i = semi + 1;
                continue;
            }
        }
        out += in[i++];
    }
    return out;
}
inline std::wstring hang_shorten(const std::wstring& in) {
    std::wstring out;
    for (size_t i = 0; i < in.size();) {
        size_t j = i + 1;
        while (j < in.size() && in[j] == in[i]) j++;
        uint32_t c = (uint32_t)in[i];
        bool alnum = is_letter_cp(c) || is_digit_cp(c);
        size_t run = j - i;
        size_t keep = (!alnum && run >= 4) ? 3 : run;
        out.append(keep, in[i]);
        i = j;
    }
    return out;
}
struct TweetTok {
    static bool url_body_cp(uint32_t c) {
        if (is_space_cp(c)) return false;
        switch (c) { case '(': case ')': case '<': case '>': case '{': case '}': case '[': case ']': return false; }
        return true;
    }
    static bool url_end_bad_cp(uint32_t c) {
        if (is_space_cp(c)) return true;
        switch (c) {
            case '`': case '!': case '(': case ')': case '[': case ']': case '{': case '}': case ';':
            case ':': case '\'': case '"': case '.': case ',': case '<': case '>': case '?':
            case 0xAB: case 0xBB: case 0x201C: case 0x201D: case 0x2018: case 0x2019: return true;
        }
        return false;
    }
    // Consume a balanced (...) group starting at s[j]=='(';
    // returns the end offset or npos.
    static size_t paren_group(const std::wstring& s, size_t j) {
        int depth = 0;
        for (size_t k = j; k < s.size(); k++) {
            uint32_t c = (uint32_t)s[k];
            if (is_space_cp(c)) return std::wstring::npos;
            if (c == '(') { if (++depth > 2) return std::wstring::npos; }
            else if (c == ')') { if (--depth == 0) return k + 1; }
        }
        return std::wstring::npos;
    }
    static bool match_url(const std::wstring& s, size_t i, size_t& len) {
        size_t n = s.size();
        size_t body = std::wstring::npos;
        std::wstring low5 = lower_w(s.substr(i, std::min<size_t>(6, n - i)));
        if (low5.rfind(L"http:", 0) == 0 || low5.rfind(L"https:", 0) == 0) {
            size_t colon = s.find(L':', i);
            size_t p = colon + 1;
            if (p < n && s[p] == L'/') {
                size_t q = p;
                while (q < n && s[q] == L'/' && q - p < 3) q++;
                body = q;
            } else if (p < n && (is_digit_cp((uint32_t)s[p]) || is_letter_cp((uint32_t)s[p]) || s[p] == L'%')) {
                body = p;
            }
        } else {
            // domain/ form: [a-z0-9.-]{1,255} '.' [a-z]{2,13} '/'
            size_t q = i;
            while (q < n && q - i < 255 && (is_letter_cp((uint32_t)s[q]) || is_digit_cp((uint32_t)s[q]) || s[q] == L'.' || s[q] == L'-')) q++;
            if (q < n && s[q] == L'/' && q > i) {
                size_t dot = s.rfind(L'.', q - 1);
                if (dot != std::wstring::npos && dot >= i) {
                    size_t tl = q - dot - 1;
                    bool allt = tl >= 2 && tl <= 13;
                    for (size_t k = dot + 1; k < q && allt; k++) allt = is_letter_cp((uint32_t)s[k]);
                    if (allt) body = q + 1;
                }
            }
        }
        if (body != std::wstring::npos) {
            size_t pos = body, last = body;
            while (pos < n) {
                if (s[pos] == L'(') {
                    size_t e = paren_group(s, pos);
                    if (e == std::wstring::npos) break;
                    pos = e; last = e;
                } else if (url_body_cp((uint32_t)s[pos])) {
                    while (pos < n && s[pos] != L'(' && url_body_cp((uint32_t)s[pos])) pos++;
                    last = pos;
                } else break;
            }
            size_t e = last;
            while (e > body && url_end_bad_cp((uint32_t)s[e - 1])) e--;
            // an unmatched ')' trimmed by the loop above is fine;
            // require at least the end char beyond the prefix
            if (e > body) { len = e - i; return true; }
        }
        // naked domain: labels '.' TLD, not after '@', not before '@'
        if (i > 0 && s[i - 1] == L'@') return false;
        {
            size_t pos = i;
            size_t last_dot = std::wstring::npos;
            bool ok = true;
            while (pos < n) {
                size_t ls = pos;
                while (pos < n && (is_letter_cp((uint32_t)s[pos]) || is_digit_cp((uint32_t)s[pos]))) pos++;
                if (pos == ls) { ok = false; break; }
                if (pos < n && (s[pos] == L'.' || s[pos] == L'-')) {
                    if (s[pos] == L'.') last_dot = pos;
                    pos++;
                    continue;
                }
                break;
            }
            if (ok && last_dot != std::wstring::npos && last_dot > i) {
                size_t ts = last_dot + 1, te = ts;
                while (te < n && is_letter_cp((uint32_t)s[te])) te++;
                size_t tl = te - ts;
                if (tl >= 2 && tl <= 13 && (te >= n || !is_word_cp((uint32_t)s[te])) && (te >= n || s[te] != L'@')) {
                    size_t e = te;
                    if (e < n && s[e] == L'/') e++;
                    len = e - i;
                    return true;
                }
            }
        }
        return false;
    }
    static bool match_phone(const std::wstring& s, size_t i, size_t& len) {
        size_t n = s.size();
        auto digits = [&](size_t p, size_t k) {
            if (p + k > n) return false;
            for (size_t t = 0; t < k; t++) if (!is_digit_cp((uint32_t)s[p + t])) return false;
            return true;
        };
        auto seps = [&](size_t p) {
            while (p < n && (s[p] == L' ' || s[p] == L'*' || s[p] == L'-' || s[p] == L'.' || s[p] == L')')) p++;
            return p;
        };
        for (int att = 0; att < 4; att++) {
            size_t j = i;
            if (att < 2) {  // with intl prefix
                size_t q = j;
                if (q < n && s[q] == L'+') q++;
                if (q < n && (s[q] == L'0' || s[q] == L'1')) j = seps(q + 1);
                else continue;
            }
            if (att % 2 == 0) {  // with area code
                size_t q = j;
                if (q < n && s[q] == L'(') q++;
                if (digits(q, 3)) j = seps(q + 3);
            }
            if (!digits(j, 3)) continue;
            j = seps(j + 3);
            if (!digits(j, 4)) continue;
            len = j + 4 - i;
            return true;
        }
        return false;
    }
    static bool match_emoticon(const std::wstring& s, size_t i, size_t& len) {
        size_t n = s.size();
        auto eyes = [](uint32_t c) { return c == ':' || c == ';' || c == '=' || c == '8'; };
        auto nose = [](uint32_t c) { return c == '-' || c == 'o' || c == '*' || c == '\''; };
        auto mouth = [](uint32_t c) {
            switch (c) { case ')': case ']': case '(': case '[': case 'd': case 'D': case 'p': case 'P':
                         case '/': case ':': case '}': case '{': case '@': case '|': case '\\': return true; }
            return false;
        };
        size_t j = i;
        if (j < n && (s[j] == L'<' || s[j] == L'>')) j++;
        if (j < n && eyes((uint32_t)s[j])) {
            size_t k = j + 1;
            if (k < n && nose((uint32_t)s[k])) k++;
            if (k < n && mouth((uint32_t)s[k])) { len = k + 1 - i; return true; }
        }
        j = i;
        if (mouth((uint32_t)s[j])) {
            size_t k = j + 1;
            if (k < n && nose((uint32_t)s[k])) k++;
            if (k < n && eyes((uint32_t)s[k])) {
                k++;
                if (k < n && (s[k] == L'<' || s[k] == L'>')) k++;
                len = k - i;
                return true;
            }
        }
        return false;
    }
    static bool match_at_pos(const std::wstring& s, size_t i, size_t& len) {
        size_t n = s.size();
        uint32_t c = (uint32_t)s[i];
        if (match_url(s, i, len)) return true;
        if (match_phone(s, i, len)) return true;
        if (match_emoticon(s, i, len)) return true;
        if (c == '<') {  // html tag
            size_t j = i + 1;
            while (j < n && s[j] != L'>' && !is_space_cp((uint32_t)s[j])) j++;
            if (j < n && s[j] == L'>' && j > i + 1) { len = j + 1 - i; return true; }
        }
        if (c == '-' || c == '<') {  // arrows
            size_t j = i;
            if (c == '-') { while (j < n && s[j] == L'-') j++; if (j < n && s[j] == L'>' && j > i) { len = j + 1 - i; return true; } }
            else { j = i + 1; size_t k = j; while (k < n && s[k] == L'-') k++; if (k > j) { len = k - i; return true; } }
        }
        if (c == '@') {
            size_t j = i + 1;
            while (j < n && is_word_cp((uint32_t)s[j])) j++;
            if (j > i + 1) { len = j - i; return true; }
        }
        if (c == '#') {
            size_t j = i;
            while (j < n && s[j] == L'#') j++;
            size_t start = j;
            size_t e = j;
            while (e < n && (is_word_cp((uint32_t)s[e]) || s[e] == L'\'' || s[e] == L'-')) e++;
            while (e > start && !is_word_cp((uint32_t)s[e - 1])) e--;
            int words = 0;
            for (size_t k = start; k < e; k++) if (is_word_cp((uint32_t)s[k])) words++;
            if (words >= 2 && is_word_cp((uint32_t)s[start])) { len = e - i; return true; }
        }
        if (is_word_cp(c)) {  // email
            size_t j = i;
            while (j < n && j - i < 64 && (is_word_cp((uint32_t)s[j]) || s[j] == L'.' || s[j] == L'+' || s[j] == L'-')) j++;
            if (j < n && s[j] == L'@' && j > i) {
                size_t d = j + 1, ds = d;
                while (d < n && d - ds < 63 && (is_word_cp((uint32_t)s[d]) || s[d] == L'-')) d++;
                if (d < n && s[d] == L'.' && d > ds) {
                    size_t e = d + 1;
                    while (e < n && (is_word_cp((uint32_t)s[e]) || s[e] == L'-' || s[e] == L'.')) e++;
                    while (e > d + 1 && !is_word_cp((uint32_t)s[e - 1]) && s[e - 1] != L'-') e--;
                    if (e > d + 1) { len = e - i; return true; }
                }
            }
        }
        if (c >= 0x1F1E6 && c <= 0x1F1FF) {  // flag pair
            if (i + 1 < n && (uint32_t)s[i + 1] >= 0x1F1E6 && (uint32_t)s[i + 1] <= 0x1F1FF) { len = 2; return true; }
        }
        if (c == 0x1F3F4) {  // tag-sequence flags (eng/sct/wls)
            size_t need = 7;
            if (i + need <= n && (uint32_t)s[i + need - 1] == 0xE007F) { len = need; return true; }
        }
        // remaining word types
        if (is_letter_cp(c)) {  // words with apostrophes or dashes
            size_t e = i;
            while (e < n && (is_letter_cp((uint32_t)s[e]) || s[e] == L'\'' || s[e] == L'-' || s[e] == L'_')) e++;
            size_t t = e;
            while (t > i && !is_letter_cp((uint32_t)s[t - 1])) t--;
            if (t - i >= 3) { len = t - i; return true; }
        }
        {  // numbers with separators
            size_t j = i;
            if (s[j] == L'+' || s[j] == L'-') j++;
            size_t ds = j;
            while (j < n && is_digit_cp((uint32_t)s[j])) j++;
            if (j > ds && j < n && (s[j] == L',' || s[j] == L'/' || s[j] == L'.' || s[j] == L':' || s[j] == L'-')) {
                size_t k = j + 1, de = k;
                while (k < n && is_digit_cp((uint32_t)s[k])) k++;
                if (k > de) {
                    if (k < n && (s[k] == L'+' || s[k] == L'-')) k++;
                    len = k - i;
                    return true;
                }
            }
        }
        if (is_word_cp(c)) {
            size_t j = i;
            while (j < n && is_word_cp((uint32_t)s[j])) j++;
            len = j - i;
            return true;
        }
        if (c == '.') {  // ellipsis
            size_t j = i + 1;
            int extra = 0;
            while (true) {
                size_t k = j;
                while (k < n && (s[k] == L' ' || s[k] == L'\t')) k++;
                if (k < n && s[k] == L'.') { j = k + 1; extra++; } else break;
            }
            if (extra >= 1) { len = j - i; return true; }
        }
        if (!is_space_cp(c)) { len = 1; return true; }
        return false;
    }
    static std::vector<std::wstring> tokenize(const std::wstring& input) {
        std::wstring s = hang_shorten(replace_html_entities(input));
        std::vector<std::wstring> out;
        size_t i = 0, n = s.size();
        while (i < n) {
            size_t len = 0;
            if (match_at_pos(s, i, len) && len > 0) { out.push_back(s.substr(i, len)); i += len; }
            else i++;
        }
        return out;
    }
};

// ---- nltk averaged perceptron tagger ----
struct PerceptronTagger {
    std::vector<std::string> classes;
    std::map<std::string, std::string> tagdict;
    std::map<std::string, std::map<std::string, double>> weights;
    // tagger.bin: the JSON model flattened by
    // tools/export_eng_g2p.py (the 5.5 MB weights JSON as a
    // Json tree costs more memory than the tagger itself).
    bool load(const std::string& path, std::string& err) {
        try {
            std::string buf = read_file_str(path);
            size_t p = 0;
            auto need = [&](size_t n) { if (p + n > buf.size()) throw std::runtime_error("tagger.bin truncated"); };
            auto u32 = [&] { need(4); uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
            auto u16 = [&] { need(2); uint16_t v; std::memcpy(&v, buf.data() + p, 2); p += 2; return v; };
            auto str = [&] { uint16_t n = u16(); need(n); std::string s = buf.substr(p, n); p += n; return s; };
            if (buf.size() < 4 || buf.compare(0, 4, "PTG1") != 0) throw std::runtime_error("tagger.bin magic");
            p = 4;
            uint32_t nc = u32();
            for (uint32_t i = 0; i < nc; i++) classes.push_back(str());
            uint32_t nt = u32();
            for (uint32_t i = 0; i < nt; i++) { std::string w = str(); tagdict[w] = str(); }
            uint32_t nf = u32();
            for (uint32_t i = 0; i < nf; i++) {
                std::string feat = str();
                uint32_t nw = u32();
                auto& dst = weights[feat];
                for (uint32_t k = 0; k < nw; k++) {
                    uint16_t ci = u16();
                    need(8);
                    double v;
                    std::memcpy(&v, buf.data() + p, 8);
                    p += 8;
                    dst[classes[ci]] = v;
                }
            }
        } catch (const std::exception& e) { err = e.what(); return false; }
        return true;
    }
    static std::string normalize(const std::string& word) {
        if (word.find('-') != std::string::npos && !word.empty() && word[0] != '-') return "!HYPHEN";
        bool digits = !word.empty();
        for (char ch : word) if (!std::isdigit((uint8_t)ch)) digits = false;
        if (digits && word.size() == 4) return "!YEAR";
        if (!word.empty() && std::isdigit((uint8_t)word[0])) return "!DIGITS";
        std::string o = word;
        for (auto& ch : o) ch = (char)std::tolower((uint8_t)ch);
        return o;
    }
    std::string predict(const std::map<std::string, int>& feats) const {
        std::map<std::string, double> scores;
        for (auto& cls : classes) scores[cls] = 0.0;
        for (auto& f : feats) {
            auto it = weights.find(f.first);
            if (it == weights.end()) continue;
            for (auto& tw : it->second) scores[tw.first] += tw.second * f.second;
        }
        std::string best;
        double best_score = 0;
        bool first = true;
        for (auto& cls : classes) {
            double sc = scores[cls];
            if (first || sc > best_score || (sc == best_score && cls > best)) { best = cls; best_score = sc; first = false; }
        }
        return best;
    }
    std::vector<std::string> tag(const std::vector<std::wstring>& tokens_w) const {
        std::vector<std::string> tokens;
        for (auto& t : tokens_w) tokens.push_back(w_to_u8(t));
        std::vector<std::string> context = {"-START-", "-START2-"};
        for (auto& t : tokens) context.push_back(normalize(t));
        context.push_back("-END-");
        context.push_back("-END2-");
        std::vector<std::string> out;
        std::string prev = "-START-", prev2 = "-START2-";
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string tag;
            auto itd = tagdict.find(tokens[i]);
            if (itd != tagdict.end()) tag = itd->second;
            else {
                const std::string& word = tokens[i];
                size_t ci = i + 2;
                std::map<std::string, int> feats;
                auto add = [&](const std::string& name, std::initializer_list<std::string> args) {
                    std::string f = name;
                    for (auto& a : args) { f += " "; f += a; }
                    feats[f]++;
                };
                auto suf = [](const std::string& w) { return w.size() > 3 ? w.substr(w.size() - 3) : w; };
                add("bias", {});
                add("i suffix", {suf(word)});
                add("i pref1", {word.empty() ? "" : word.substr(0, 1)});
                add("i-1 tag", {prev});
                add("i-2 tag", {prev2});
                add("i tag+i-2 tag", {prev, prev2});
                add("i word", {context[ci]});
                add("i-1 tag+i word", {prev, context[ci]});
                add("i-1 word", {context[ci - 1]});
                add("i-1 suffix", {suf(context[ci - 1])});
                add("i-2 word", {context[ci - 2]});
                add("i+1 word", {context[ci + 1]});
                add("i+1 suffix", {suf(context[ci + 1])});
                add("i+2 word", {context[ci + 2]});
                tag = predict(feats);
            }
            out.push_back(tag);
            prev2 = prev;
            prev = tag;
        }
        return out;
    }
};

// ---- wordsegment (WordSegment.py) ----
struct WordSegmenter {
    std::unordered_map<std::string, double> unigrams, bigrams;
    static constexpr double kTotal = 1024908267229.0;
    bool load(const std::string& dir, std::string& err) {
        try {
            for (auto which : {0, 1}) {
                std::ifstream f(dir + (which == 0 ? "/unigrams.txt" : "/bigrams.txt"));
                if (!f) throw std::runtime_error("wordsegment data missing in " + dir);
                auto& dst = which == 0 ? unigrams : bigrams;
                std::string line;
                while (std::getline(f, line)) {
                    size_t tab = line.find('\t');
                    if (tab == std::string::npos) continue;
                    std::string w = line.substr(0, tab), num = line.substr(tab + 1);
                    if (!w.empty() && !num.empty()) dst[w] = std::stod(num);
                }
            }
        } catch (const std::exception& e) { err = e.what(); return false; }
        return true;
    }
    double score(const std::string& word, const std::string* prev) const {
        if (!prev) {
            auto it = unigrams.find(word);
            if (it != unigrams.end()) return it->second / kTotal;
            double d = 1.0;
            for (size_t i = 0; i < word.size(); i++) d *= 10.0;
            return 10.0 / (kTotal * d);
        }
        auto ib = bigrams.find(*prev + " " + word);
        auto iu = unigrams.find(*prev);
        if (ib != bigrams.end() && iu != unigrams.end())
            return ib->second / kTotal / score(*prev, nullptr);
        return score(word, nullptr);
    }
    using Result = std::pair<double, std::vector<std::string>>;
    Result search(const std::string& text, const std::string& prev,
                  std::map<std::pair<std::string, std::string>, Result>& memo) const {
        if (text.empty()) return {0.0, {}};
        Result best;
        bool have = false;
        size_t lim = std::min<size_t>(text.size(), 24);
        for (size_t pos = 1; pos <= lim; pos++) {
            std::string prefix = text.substr(0, pos), suffix = text.substr(pos);
            double ps = std::log10(score(prefix, &prev));
            auto key = std::make_pair(suffix, prefix);
            auto it = memo.find(key);
            Result sub;
            if (it != memo.end()) sub = it->second;
            else { sub = search(suffix, prefix, memo); memo[key] = sub; }
            Result cand{ps + sub.first, {}};
            cand.second.push_back(prefix);
            cand.second.insert(cand.second.end(), sub.second.begin(), sub.second.end());
            // Python max() over (score, words) tuples: largest
            // score, ties broken by the larger word list.
            if (!have || cand.first > best.first ||
                (cand.first == best.first && cand.second > best.second)) {
                best = cand;
                have = true;
            }
        }
        return best;
    }
    std::vector<std::string> segment(const std::string& text) const {
        std::string clean;
        for (char ch : text) {
            char l = (char)std::tolower((uint8_t)ch);
            if ((l >= 'a' && l <= 'z') || (l >= '0' && l <= '9')) clean += l;
        }
        if (clean.size() > 250) clean = clean.substr(0, 250);  // upstream chunks; words never reach this
        std::map<std::pair<std::string, std::string>, Result> memo;
        return search(clean, "<s>", memo).second;
    }
};

// ---- the G2P itself ----
struct EngG2P {
    std::map<std::string, std::vector<std::string>> cmu, namedict;
    PerceptronTagger tagger;
    WordSegmenter segmenter;
    // GRU weights (float32, from checkpoint20.npz via eng.bin)
    std::map<std::string, std::vector<float>> wts;
    std::vector<std::string> symbols;
    std::map<std::string, int> symbol_id;
    static const std::vector<std::string>& graphemes() {
        static const std::vector<std::string> g = [] {
            std::vector<std::string> v = {"<pad>", "<unk>", "</s>"};
            for (char c = 'a'; c <= 'z'; c++) v.push_back(std::string(1, c));
            return v;
        }();
        return g;
    }
    static const std::vector<std::string>& phonemes() {
        static const std::vector<std::string> p = {
            "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1",
            "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
            "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
            "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1",
            "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW",
            "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"};
        return p;
    }

    bool load(const std::string& dir, const std::vector<std::string>& symbols_v2, std::string& err) {
        symbols = symbols_v2;
        for (size_t i = 0; i < symbols.size(); i++) symbol_id[symbols[i]] = (int)i;
        try {
            std::string buf = read_file_str(dir + "/eng.bin");
            size_t p = 0;
            auto need = [&](size_t n) { if (p + n > buf.size()) throw std::runtime_error("eng.bin truncated"); };
            auto u32 = [&] { need(4); uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
            auto u16 = [&] { need(2); uint16_t v; std::memcpy(&v, buf.data() + p, 2); p += 2; return v; };
            auto u8 = [&] { need(1); return (uint8_t)buf[p++]; };
            auto str = [&] { uint16_t n = u16(); need(n); std::string s = buf.substr(p, n); p += n; return s; };
            if (buf.size() < 4 || buf.compare(0, 4, "ENG1") != 0) throw std::runtime_error("eng.bin magic");
            p = 4;
            for (auto which : {0, 1}) {
                auto& dst = which == 0 ? cmu : namedict;
                uint32_t n = u32();
                for (uint32_t i = 0; i < n; i++) {
                    std::string word = str();
                    uint8_t np = u8();
                    std::vector<std::string> pron;
                    for (int k = 0; k < np; k++) pron.push_back(str());
                    dst[word] = pron;
                }
            }
            uint32_t nw = u32();
            for (uint32_t i = 0; i < nw; i++) {
                std::string name = str();
                uint8_t dt = u8(), nd = u8();
                std::vector<uint32_t> dims;
                size_t cnt = 1;
                for (int k = 0; k < nd; k++) { dims.push_back(u32()); cnt *= dims.back(); }
                std::vector<float> v(cnt);
                if (dt == 0) { need(cnt * 4); std::memcpy(v.data(), buf.data() + p, cnt * 4); p += cnt * 4; }
                else {
                    need(cnt * 8);
                    for (size_t k = 0; k < cnt; k++) { double d; std::memcpy(&d, buf.data() + p, 8); p += 8; v[k] = (float)d; }
                }
                wts[name] = std::move(v);
            }
        } catch (const std::exception& e) { err = e.what(); return false; }
        if (!tagger.load(dir + "/tagger.bin", err)) return false;
        if (!segmenter.load(dir + "/wordsegment", err)) return false;
        return true;
    }

    // GRU cell, PyTorch layout as in EnglishG2P._grucell:
    // gates r,z from the first 2/3, n from the last 1/3.
    std::vector<float> grucell(const std::vector<float>& x, const std::vector<float>& h,
                               const std::string& prefix) const {
        const float* wih = wts.at(prefix + "_w_ih").data();
        const float* whh = wts.at(prefix + "_w_hh").data();
        const float* bih = wts.at(prefix + "_b_ih").data();
        const float* bhh = wts.at(prefix + "_b_hh").data();
        const int H = 256, I = 256;
        std::vector<double> gih(3 * H), ghh(3 * H);
        for (int k = 0; k < 3 * H; k++) {
            double a = bih[k], b = bhh[k];
            for (int j = 0; j < I; j++) a += (double)wih[k * I + j] * x[j];
            for (int j = 0; j < H; j++) b += (double)whh[k * H + j] * h[j];
            gih[k] = a; ghh[k] = b;
        }
        std::vector<float> out(H);
        for (int k = 0; k < H; k++) {
            double r = 1.0 / (1.0 + std::exp(-(gih[k] + ghh[k])));
            double z = 1.0 / (1.0 + std::exp(-(gih[H + k] + ghh[H + k])));
            double nn = std::tanh(gih[2 * H + k] + r * ghh[2 * H + k]);
            out[k] = (float)((1.0 - z) * nn + z * h[k]);
        }
        return out;
    }
    std::vector<std::string> predict(const std::string& o_word) const {
        std::map<std::string, int> g2idx, p2idx;
        for (size_t i = 0; i < graphemes().size(); i++) g2idx[graphemes()[i]] = (int)i;
        for (size_t i = 0; i < phonemes().size(); i++) p2idx[phonemes()[i]] = (int)i;
        std::string word = w_to_u8(lower_w(u8_to_w(o_word)));
        const float* enc_emb = wts.at("enc_emb").data();
        std::vector<float> h(256, 0.0f);
        auto step_in = [&](const std::vector<float>& x) { return grucell(x, h, "enc"); };
        for (char ch : word) {
            int id = 1;
            auto it = g2idx.find(std::string(1, ch));
            if (it != g2idx.end()) id = it->second;
            std::vector<float> x(enc_emb + id * 256, enc_emb + id * 256 + 256);
            h = step_in(x);
        }
        {
            int id = g2idx["</s>"];
            std::vector<float> x(enc_emb + id * 256, enc_emb + id * 256 + 256);
            h = step_in(x);
        }
        const float* dec_emb = wts.at("dec_emb").data();
        const float* fcw = wts.at("fc_w").data();
        const float* fcb = wts.at("fc_b").data();
        std::vector<float> dec(dec_emb + p2idx["<s>"] * 256, dec_emb + p2idx["<s>"] * 256 + 256);
        std::vector<std::string> preds;
        for (int step = 0; step < 20; step++) {
            h = grucell(dec, h, "dec");
            int best = 0;
            double bestv = -1e300;
            for (int k = 0; k < 74; k++) {
                double v = fcb[k];
                for (int j = 0; j < 256; j++) v += (double)fcw[k * 256 + j] * h[j];
                if (v > bestv) { bestv = v; best = k; }
            }
            if (best == p2idx["</s>"]) break;
            preds.push_back(phonemes()[best]);
            dec.assign(dec_emb + best * 256, dec_emb + best * 256 + 256);
        }
        return preds;
    }

    static bool istitle(const std::string& w) {
        // Python str.istitle over the token (ASCII-faithful here)
        bool any_cased = false, prev_cased = false, ok = true;
        for (size_t i = 0; i < w.size(); i++) {
            uint8_t c = (uint8_t)w[i];
            bool upper = std::isupper(c) != 0, lower = std::islower(c) != 0;
            if (upper || lower) {
                any_cased = true;
                if (prev_cased && upper) ok = false;
                if (!prev_cased && lower) ok = false;
                prev_cased = true;
            } else prev_cased = false;
        }
        return any_cased && ok;
    }

    std::vector<std::string> query_word(const std::string& o_word) const {
        std::string word = w_to_u8(lower_w(u8_to_w(o_word)));
        auto it = cmu.find(word);
        if (it != cmu.end()) {
            if (o_word == "A") return {"AH0"};
            return it->second;
        }
        if (istitle(o_word)) {
            auto in = namedict.find(word);
            if (in != namedict.end()) return in->second;
        }
        if (word.size() > 2 && word.compare(word.size() - 2, 2, "'s") == 0) {
            auto base = query_word(word.substr(0, word.size() - 2));
            if (!base.empty()) {
                const std::string& last = base.back();
                static const std::set<std::string> s1 = {"S", "Z", "SH", "ZH", "CH", "JH"};
                static const std::set<std::string> s2 = {"P", "T", "K", "F", "TH"};
                if (s1.count(last)) { base.push_back("AH0"); base.push_back("Z"); }
                else if (s2.count(last)) base.push_back("S");
                else base.push_back("Z");
                return base;
            }
        }
        if (word.find('-') != std::string::npos && word.size() > 1) {
            std::vector<std::string> parts;
            for (size_t i = 0; i <= word.size();) {
                size_t j = word.find('-', i);
                std::string part = word.substr(i, j == std::string::npos ? j : j - i);
                if (!part.empty()) parts.push_back(part);
                if (j == std::string::npos) break;
                i = j + 1;
            }
            if (parts.size() > 1) {
                std::vector<std::string> result;
                for (auto& part : parts) {
                    auto r = query_word(part);
                    result.insert(result.end(), r.begin(), r.end());
                }
                if (!result.empty()) return result;
            }
        }
        {
            auto segs = segmenter.segment(word);
            std::string joined;
            for (auto& sg : segs) joined += sg;
            if (segs.size() > 1 && joined == word) {
                std::vector<std::string> result;
                for (auto& sg : segs) {
                    auto r = query_word(sg);
                    result.insert(result.end(), r.begin(), r.end());
                }
                if (!result.empty()) return result;
            }
        }
        return predict(o_word);
    }

    // CleanG2p.__call__'s alignment walk, exposed for checks.
    std::vector<std::wstring> corrected_tokens(const std::wstring& text) const {
        auto orig = TweetTok::tokenize(text);
        auto norm = TweetTok::tokenize(eng_text_normalize(text));
        std::vector<std::wstring> corrected;
        size_t oi = 0, ni = 0;
        while (oi < orig.size() && ni < norm.size()) {
            if (orig[oi] == L"I" && ni + 1 < norm.size() + 1 &&
                (ni + 2 <= norm.size() && (norm[ni] + (ni + 1 < norm.size() ? L" " + norm[ni + 1] : L"")) == L"the first")) {
                corrected.push_back(L"I");
                oi += 1;
                ni += 2;
            } else {
                corrected.push_back(norm[ni]);
                oi += 1;
                ni += 1;
            }
        }
        while (ni < norm.size()) corrected.push_back(norm[ni++]);
        return corrected;
    }

    std::vector<std::string> g2p(const std::wstring& text) const {
        auto corrected = corrected_tokens(text);
        if (corrected.empty()) return {};
        auto tags = tagger.tag(corrected);
        static const std::map<std::string, std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>>
            homographs = {
                {"read", {{"R", "EH1", "D"}, {"R", "IY1", "D"}, "VBD"}},
                {"complex", {{"K", "AH0", "M", "P", "L", "EH1", "K", "S"}, {"K", "AA1", "M", "P", "L", "EH0", "K", "S"}, "JJ"}},
                {"lead", {{"L", "IY1", "D"}, {"L", "EH1", "D"}, "NN"}},
                {"presents", {{"P", "R", "IY0", "Z", "EH1", "N", "T", "S"}, {"P", "R", "EH1", "Z", "AH0", "N", "T", "S"}, "VBZ"}},
            };
        std::vector<std::string> prons;
        for (size_t i = 0; i < corrected.size(); i++) {
            std::string o_word = w_to_u8(corrected[i]);
            std::string word = w_to_u8(lower_w(corrected[i]));
            std::vector<std::string> pron;
            bool has_az = word.find_first_of("abcdefghijklmnopqrstuvwxyz") != std::string::npos;
            if (!has_az) pron = {word};
            else {
                auto ih = homographs.find(word);
                if (ih != homographs.end()) {
                    const auto& [p1, p2, pos1] = ih->second;
                    pron = (tags[i].rfind(pos1, 0) == 0) ? p1 : p2;
                } else pron = query_word(o_word);
            }
            prons.insert(prons.end(), pron.begin(), pron.end());
            prons.push_back(" ");
        }
        if (!prons.empty()) prons.pop_back();
        std::vector<std::string> phones;
        for (auto& ph : prons) {
            if (ph == " " || ph == "<pad>" || ph == "UW" || ph == "</s>" || ph == "<s>") continue;
            phones.push_back(ph == "<unk>" ? "UNK" : ph);
        }
        std::vector<std::string> out;
        for (auto& ph : phones) {
            if (symbol_id.count(ph)) out.push_back(ph);
            else if (ph == "'") out.push_back("-");
        }
        return out;
    }
    std::vector<int64_t> phones_to_ids(const std::wstring& text) const {
        std::vector<int64_t> ids;
        for (auto& ph : g2p(text)) ids.push_back(symbol_id.at(ph));
        return ids;
    }
};

}  // namespace eng2p
