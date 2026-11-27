// TextNormalizer port (M3-3d stage b4): genie_tts
// G2P/Chinese/Normalization (text_normlization.py + num.py +
// chronology.py + quantifier.py + phonecode.py + char_convert
// t2s). Turns raw text into normalized Chinese sentences:
// traditional->simplified, fullwidth->halfwidth, then a fixed
// cascade of regex substitutions verbalizing dates, times,
// ranges, temperatures, measures, arithmetic, fractions,
// percentages, phone numbers, versions and bare numbers.
//
// Implementation notes:
// - Everything runs on std::wstring (UTF-32 codepoints) with
//   std::wregex: the Python patterns use non-ASCII character
//   classes, which are byte-unsafe in narrow regex.
// - Python lookarounds ((?<!\d), (?![...])) do not exist in
//   ECMAScript regex; RE_MOBILE_PHONE / RE_TELEPHONE /
//   RE_RANGE use re_sub_bound with manual boundary checks.
//   One known corner divergence: Python can backtrack
//   RE_RANGE's optional decimal away to satisfy the lookahead
//   (e.g. "1-2.5-3"); the port rejects the whole candidate.
// - replace_temperature mirrors the source's group indexing
//   exactly: it reads group(3) (the decimal group) as the
//   unit, so the emitted unit is always the plain degree
//   word. replace_time's range branch likewise tests the
//   FIRST minute for the half-hour word. Both are upstream
//   bugs, kept for oracle equality.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "wordpiece.h"  // utf8_decode / utf8_encode

namespace textnorm {

inline std::wstring to_wide(const std::string& s) {
    std::wstring w;
    for (int32_t cp : wptok::utf8_decode(s)) w.push_back((wchar_t)cp);
    return w;
}
inline std::string from_wide(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) wptok::utf8_encode((int32_t)c, s);
    return s;
}
inline std::string narrow_ascii(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) s.push_back(c < 128 ? (char)c : '?');
    return s;
}
inline std::wstring replace_all_w(std::wstring s, const std::wstring& from, const std::wstring& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::wstring::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

using SubFn = std::function<std::wstring(const std::wsmatch&)>;

// Python re.sub with a callable replacement.
inline std::wstring re_sub(const std::wstring& s, const std::wregex& re, const SubFn& fn) {
    std::wstring out;
    size_t pos = 0;
    while (pos <= s.size()) {
        std::wsmatch m;
        if (!std::regex_search(s.begin() + (long)pos, s.end(), m, re)) break;
        size_t ms = pos + (size_t)m.position();
        size_t ml = (size_t)m.length();
        out.append(s, pos, ms - pos);
        out += fn(m);
        if (ml == 0) {
            if (ms < s.size()) {
                out += s[ms];
                pos = ms + 1;
            } else {
                pos = ms;
                break;
            }
        } else {
            pos = ms + ml;
        }
    }
    out.append(s, pos, std::wstring::npos);
    return out;
}

// re.sub for patterns guarded by a one-char lookbehind /
// lookahead pair: ok(s, begin, end) decides acceptance.
inline std::wstring re_sub_bound(const std::wstring& s, const std::wregex& re,
                                 const std::function<bool(const std::wstring&, size_t, size_t)>& ok,
                                 const SubFn& fn) {
    std::wstring out;
    size_t copied = 0, pos = 0;
    while (pos <= s.size()) {
        std::wsmatch m;
        bool found = false;
        size_t ms = 0, ml = 0;
        std::wsmatch keep;
        size_t search = pos;
        while (search <= s.size()) {
            if (!std::regex_search(s.begin() + (long)search, s.end(), m, re)) break;
            ms = search + (size_t)m.position();
            ml = (size_t)m.length();
            if (ok(s, ms, ms + ml)) {
                found = true;
                keep = m;
                break;
            }
            search = ms + 1;
        }
        if (!found) break;
        out.append(s, copied, ms - copied);
        out += fn(keep);
        copied = pos = ms + ml;
        if (ml == 0) pos = ms + 1;
    }
    out.append(s, copied, std::wstring::npos);
    return out;
}

// ---- num.py (narrow ASCII in, UTF-8 Chinese out) ----

inline std::string digit_word(int d) {
    static const char* kDigits[10] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
    return kDigits[d];
}
inline std::string lstrip0(const std::string& s) {
    size_t i = s.find_first_not_of('0');
    return i == std::string::npos ? "" : s.substr(i);
}
inline std::vector<std::string> get_value(const std::string& vs, bool use_zero = true) {
    std::string stripped = lstrip0(vs);
    if (stripped.empty()) return {};
    if (stripped.size() == 1) {
        int d = stripped[0] - '0';
        if (use_zero && stripped.size() < vs.size()) return {digit_word(0), digit_word(d)};
        return {digit_word(d)};
    }
    static const int kPow[5] = {8, 4, 3, 2, 1};
    static const char* kUnit[9] = {"", "十", "百", "千", "万", "", "", "", "亿"};
    int largest = 1;
    for (int p : kPow)
        if (p < (int)stripped.size()) {
            largest = p;
            break;
        }
    std::string first = vs.substr(0, vs.size() - largest);
    std::string second = vs.substr(vs.size() - largest);
    auto out = get_value(first);
    out.push_back(kUnit[largest]);
    auto tail = get_value(second);
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}
inline std::string verbalize_cardinal(const std::string& vs_in) {
    if (vs_in.empty()) return "";
    std::string vs = lstrip0(vs_in);
    if (vs.empty()) return digit_word(0);
    auto syms = get_value(vs);
    if (syms.size() >= 2 && syms[0] == digit_word(1) && syms[1] == "十") syms.erase(syms.begin());
    std::string out;
    for (auto& s : syms) out += s;
    return out;
}
inline std::string verbalize_digit(const std::string& vs, bool alt_one = false) {
    std::string out;
    for (char c : vs)
        if (c >= '0' && c <= '9') out += digit_word(c - '0');
    if (alt_one) {
        std::string r;
        size_t pos = 0;
        while (pos < out.size()) {
            if (out.compare(pos, 3, "一") == 0) {
                r += "幺";
                pos += 3;
            } else {
                r += out[pos];
                pos += 1;
            }
        }
        out = r;
    }
    return out;
}
inline std::string num2str(const std::string& vs) {
    size_t dot = vs.find('.');
    std::string integer = dot == std::string::npos ? vs : vs.substr(0, dot);
    std::string decimal = dot == std::string::npos ? "" : vs.substr(dot + 1);
    std::string result = verbalize_cardinal(integer);
    if (decimal.size() && decimal.back() == '0') {
        size_t last = decimal.find_last_not_of('0');
        decimal = (last == std::string::npos ? "" : decimal.substr(0, last + 1)) + "0";
    } else {
        size_t last = decimal.find_last_not_of('0');
        decimal = last == std::string::npos ? "" : decimal.substr(0, last + 1);
    }
    if (!decimal.empty()) {
        if (result.empty()) result = digit_word(0);
        result += "点" + verbalize_digit(decimal);
    }
    return result;
}
inline std::string time_num2str(const std::string& ns) {
    std::string r = num2str(lstrip0(ns));
    if (!ns.empty() && ns[0] == '0') r = digit_word(0) + r;
    return r;
}

inline std::wstring W(const std::string& utf8) { return to_wide(utf8); }
inline std::string grp(const std::wsmatch& m, size_t i) { return narrow_ascii(m.str(i)); }
inline bool grpm(const std::wsmatch& m, size_t i) { return m[i].matched; }

}  // namespace textnorm

namespace textnorm {

// COM_QUANTIFIERS from num.py, verbatim.
inline const std::wstring& com_quantifiers() {
static const std::wstring kQ =
L"(处|台|架|枚|趟|幅|平|方|堵|间|床|株|批|项|例|列|篇|栋|注|亩|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)";
return kQ;
}

struct Patterns {
std::wregex splitor, split_rm, post_rm;
std::wregex date, date2, time, time_range, to_range, temperature;
std::wregex asmd, power, frac, percentage;
std::wregex mobile, telephone, national, range;
std::wregex integer, version, decimal, pos_quant, default_num, number;
Patterns()
: splitor(L"([：、，；。？！,;?!][”’]?)"),
split_rm(L"[——《》<>{}()（）#&@“”^_|\\\\]"),
post_rm(L"[-——《》<=>{}()（）#&@“”^_|\\\\]"),
date(L"(\\d{4}|\\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?"),
date2(L"(\\d{4})([- /.])(0[1-9]|1[012])\\2(0[1-9]|[12][0-9]|3[01])"),
time(L"([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?"),
time_range(L"([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?(~|-)([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?"),
to_range(L"((-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)[~]((-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)"),
temperature(L"(-?)(\\d+(\\.\\d+)?)(°C|℃|度|摄氏度)"),
asmd(L"((-?)((\\d+)(\\.\\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\\.\\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))([+\\-×÷=])((-?)((\\d+)(\\.\\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\\.\\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))"),
power(L"[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]+"),
frac(L"(-?)(\\d+)/(\\d+)"),
percentage(L"(-?)(\\d+(\\.\\d+)?)%"),
mobile(L"((\\+?86?)?1([38]\\d|5[0-35-9]|7[678]|9[89])\\d{8})"),
telephone(L"((0(10|2[1-3]|[3-9]\\d{2})-?)?[1-9]\\d{6,7})"),
national(L"(400)(-)?\\d{3}(-)?\\d{4}"),
range(L"((-?)((\\d+)(\\.\\d+)?))[-~]((-?)((\\d+)(\\.\\d+)?))"),
integer(L"(-)(\\d+)"),
version(L"((\\d+)(\\.\\d+)(\\.\\d+)?(\\.\\d+)+)"),
decimal(L"(-?)((\\d+)(\\.\\d+))|(\\.(\\d+))"),
pos_quant(std::wstring(L"(\\d+)([多余几+])?") + com_quantifiers()),
default_num(L"\\d{3}\\d*"),
number(L"(-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+))") {}
};

inline const Patterns& pats() {
static const Patterns p;
return p;
}

inline std::wstring repl_date(const std::wsmatch& m) {
std::wstring r;
if (grpm(m, 1)) r += W(verbalize_digit(grp(m, 1))) + L"年";
if (grpm(m, 3)) r += W(verbalize_cardinal(grp(m, 3))) + L"月";
if (grpm(m, 5)) r += W(verbalize_cardinal(grp(m, 5))) + m.str(9);
return r;
}
inline std::wstring repl_date2(const std::wsmatch& m) {
return W(verbalize_digit(grp(m, 1))) + L"年" + W(verbalize_cardinal(grp(m, 3))) + L"月" +
W(verbalize_cardinal(grp(m, 4))) + L"日";
}
inline std::wstring repl_time_impl(const std::wsmatch& m, bool is_range) {
std::string hour = grp(m, 1), minute = grp(m, 2), second = grpm(m, 4)? grp(m, 4): "";
std::wstring r = W(num2str(hour)) + L"点";
if (!lstrip0(minute).empty()) {
if (std::atoi(minute.c_str()) == 30)
r += L"半";
else
r += W(time_num2str(minute)) + L"分";
}
if (grpm(m, 4) &&!lstrip0(second).empty()) r += W(time_num2str(second)) + L"秒";
if (is_range) {
std::string h2 = grp(m, 6), m2 = grp(m, 7), s2 = grpm(m, 9)? grp(m, 9): "";
r += L"至" + W(num2str(h2)) + L"点";
if (!lstrip0(m2).empty()) {
if (std::atoi(minute.c_str()) == 30) // source tests the first minute
r += L"半";
else
r += W(time_num2str(m2)) + L"分";
}
if (grpm(m, 9) &&!lstrip0(s2).empty()) r += W(time_num2str(s2)) + L"秒";
}
return r;
}
inline std::wstring repl_to_range(const std::wsmatch& m) {
return replace_all_w(m.str(0), L"~", L"至");
}
inline std::wstring repl_temperature(const std::wsmatch& m) {
std::wstring r;
if (grpm(m, 1) && m.str(1).size()) r += L"零下";
r += W(num2str(grp(m, 2)));
r += L"度"; // source reads group(3) as the unit: always plain degree
return r;
}
inline std::wstring replace_measure(std::wstring s) {
static const std::pair<const wchar_t*, const wchar_t*> kM[] = {
{L"cm2", L"平方厘米"}, {L"cm²", L"平方厘米"}, {L"cm3", L"立方厘米"}, {L"cm³", L"立方厘米"},
{L"cm", L"厘米"}, {L"db", L"分贝"}, {L"ds", L"毫秒"}, {L"kg", L"千克"},
{L"km", L"千米"}, {L"m2", L"平方米"}, {L"m²", L"平方米"}, {L"m³", L"立方米"},
{L"m3", L"立方米"}, {L"ml", L"毫升"}, {L"m", L"米"}, {L"mm", L"毫米"},
{L"s", L"秒"},
};
for (auto& kv: kM)
if (s.find(kv.first)!= std::wstring::npos) s = replace_all_w(s, kv.first, kv.second);
return s;
}
inline std::wstring repl_asmd(const std::wsmatch& m) {
std::wstring op;
switch (m.str(8)[0]) {
case L'+': op = L"加"; break;
case L'-': op = L"减"; break;
case L'×': op = L"乘"; break;
case L'÷': op = L"除"; break;
default: op = L"等于"; break;
}
return m.str(1) + op + m.str(9);
}
inline std::wstring repl_power(const std::wsmatch& m) {
std::wstring digits;
for (wchar_t c: m.str(0)) {
switch (c) {
case L'⁰': digits += L'0'; break;
case L'¹': digits += L'1'; break;
case L'²': digits += L'2'; break;
case L'³': digits += L'3'; break;
case L'⁴': digits += L'4'; break;
case L'⁵': digits += L'5'; break;
case L'⁶': digits += L'6'; break;
case L'⁷': digits += L'7'; break;
case L'⁸': digits += L'8'; break;
case L'⁹': digits += L'9'; break;
case L'ˣ': digits += L'x'; break;
case L'ʸ': digits += L'y'; break;
case L'ⁿ': digits += L'n'; break;
default: break;
}
}
return L"的" + digits + L"次方";
}
inline std::wstring repl_frac(const std::wsmatch& m) {
std::wstring r;
if (m.str(1).size()) r += L"负";
r += W(num2str(grp(m, 3))) + L"分之" + W(num2str(grp(m, 2)));
return r;
}
inline std::wstring repl_percentage(const std::wsmatch& m) {
std::wstring r;
if (m.str(1).size()) r += L"负";
r += L"百分之" + W(num2str(grp(m, 2)));
return r;
}
inline std::wstring phone2str(const std::wstring& phone, bool mobile) {
std::vector<std::wstring> parts;
if (mobile) {
size_t b = phone.find_first_not_of(L'+');
size_t e = phone.find_last_not_of(L'+');
std::wstring t = b == std::wstring::npos? L"": phone.substr(b, e - b + 1);
size_t pos = 0;
while (pos < t.size()) {
size_t sp = t.find_first_of(L" \t\n\r\v\f", pos);
if (sp == std::wstring::npos) sp = t.size();
if (sp > pos) parts.push_back(t.substr(pos, sp - pos));
pos = sp + 1;
}
} else {
size_t pos = 0;
while (true) {
size_t d = phone.find(L'-', pos);
if (d == std::wstring::npos) {
parts.push_back(phone.substr(pos));
break;
}
parts.push_back(phone.substr(pos, d - pos));
pos = d + 1;
}
}
std::wstring r;
for (size_t i = 0; i < parts.size(); i++) {
if (i) r += L"，";
r += W(verbalize_digit(narrow_ascii(parts[i]), true));
}
return r;
}
inline std::wstring repl_number(const std::wsmatch& m) {
if (grpm(m, 5)) return W(num2str(grp(m, 5)));
std::wstring r;
if (m.str(1).size()) r += L"负";
r += W(num2str(grp(m, 2)));
return r;
}
inline std::wstring repl_range(const std::wsmatch& m) {
std::wstring first = re_sub(m.str(1), pats().number, repl_number);
std::wstring second = re_sub(m.str(6), pats().number, repl_number);
return first + L"到" + second;
}
inline std::wstring repl_negative(const std::wsmatch& m) { return L"负" + W(num2str(grp(m, 2)));}
inline std::wstring repl_version(const std::wsmatch& m) {
std::wstring r;
for (wchar_t c: m.str(1)) {
if (c == L'.')
r += L"点";
else
r += W(num2str(std::string(1, (char)c)));
}
return r;
}
inline std::wstring repl_pos_quant(const std::wsmatch& m) {
std::string number = num2str(grp(m, 1));
if (number == "二") number = "两";
std::wstring mid;
if (grpm(m, 2)) mid = m.str(2) == L"+"? std::wstring(L"多"): m.str(2);
return W(number) + mid + m.str(3);
}
inline std::wstring repl_default_num(const std::wsmatch& m) {
return W(verbalize_digit(grp(m, 0), true));
}

inline std::wstring post_replace(std::wstring s) {
static const std::pair<const wchar_t*, const wchar_t*> kGreek[] = {
{L"α", L"阿尔法"}, {L"β", L"贝塔"}, {L"γ", L"伽马"}, {L"δ", L"德尔塔"},
{L"ε", L"艾普西龙"}, {L"ζ", L"捷塔"}, {L"η", L"依塔"}, {L"θ", L"西塔"},
{L"ι", L"艾欧塔"}, {L"κ", L"喀帕"}, {L"λ", L"拉姆达"}, {L"μ", L"缪"},
{L"ν", L"拗"}, {L"ξ", L"克西"}, {L"ο", L"欧米克伦"}, {L"π", L"派"},
{L"ρ", L"肉"}, {L"σ", L"西格马"}, {L"τ", L"套"}, {L"υ", L"宇普西龙"},
{L"φ", L"服艾"}, {L"χ", L"器"}, {L"ψ", L"普赛"}, {L"ω", L"欧米伽"},
};
for (auto& kv: kGreek) s = replace_all_w(s, kv.first, kv.second);
s = replace_all_w(s, L"+", L"加");
s = replace_all_w(s, L"-", L"减");
s = replace_all_w(s, L"×", L"乘");
s = replace_all_w(s, L"÷", L"除");
s = replace_all_w(s, L"=", L"等");
s = re_sub(s, pats().post_rm, [](const std::wsmatch&) { return std::wstring();});
return s;
}

inline bool is_ws_cp(wchar_t c) {
return c == L' ' || c == L'\t' || c == L'\n' || c == L'\r' || c == L'\v' || c == L'\f' ||
c == 0x85 || c == 0xA0 || c == 0x3000 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
c == 0x2029;
}
inline std::wstring strip_w(const std::wstring& s) {
size_t b = 0, e = s.size();
while (b < e && is_ws_cp(s[b])) b++;
while (e > b && is_ws_cp(s[e - 1])) e--;
return s.substr(b, e - b);
}

class TextNormalizer {
public:
bool load_t2s(const std::string& path) {
FILE* f = fopen(path.c_str(), "rb");
if (!f) return false;
auto rd = [&](void* p, size_t n) { return fread(p, 1, n, f) == n;};
char magic[4];
uint32_t n = 0;
bool ok = rd(magic, 4) && memcmp(magic, "T2S1", 4) == 0 && rd(&n, 4);
for (uint32_t i = 0; ok && i < n; i++) {
uint32_t pair[2];
ok = rd(pair, 8);
if (ok) t2s_[(int32_t)pair[0]] = (int32_t)pair[1];
}
fclose(f);
return ok;
}

// TextNormalizer.normalize: split, then normalize each.
std::vector<std::string> normalize(const std::string& text) const {
std::vector<std::string> out;
for (auto& s: split_sentences(text)) out.push_back(normalize_sentence(s));
return out;
}

std::vector<std::string> split_sentences(const std::string& text) const {
std::wstring w = to_wide(text);
w = replace_all_w(w, L" ", L"");
w = re_sub(w, pats().split_rm, [](const std::wsmatch&) { return std::wstring();});
w = re_sub(w, pats().splitor,
[](const std::wsmatch& m) { return m.str(1) + L"\n";});
w = strip_w(w);
std::vector<std::string> out;
size_t pos = 0;
while (true) {
size_t nl = w.find(L'\n', pos);
std::wstring piece = nl == std::wstring::npos? w.substr(pos): w.substr(pos, nl - pos);
out.push_back(from_wide(strip_w(piece)));
if (nl == std::wstring::npos) break;
pos = nl;
while (pos < w.size() && w[pos] == L'\n') pos++;
if (pos >= w.size()) {
// re.split keeps a trailing empty piece only if
// the string ends with the separator; after the
// outer strip it cannot. Nothing more.
break;
}
}
if (w.empty()) out = {""};
return out;
}

std::string normalize_sentence(const std::string& s_in) const {
std::wstring w = to_wide(s_in);
// traditional -> simplified
for (auto& c: w) {
auto it = t2s_.find((int32_t)c);
if (it!= t2s_.end()) c = (wchar_t)it->second;
}
// fullwidth letters/digits/space -> halfwidth
for (auto& c: w) {
if (c >= 0xFF21 && c <= 0xFF3A)
c = c - 0xFF21 + L'A';
else if (c >= 0xFF41 && c <= 0xFF5A)
c = c - 0xFF41 + L'a';
else if (c >= 0xFF10 && c <= 0xFF19)
c = c - 0xFF10 + L'0';
else if (c == 0x3000)
c = L' ';
}
const Patterns& P = pats();
w = re_sub(w, P.date, repl_date);
w = re_sub(w, P.date2, repl_date2);
w = re_sub(w, P.time_range, [](const std::wsmatch& m) { return repl_time_impl(m, true);});
w = re_sub(w, P.time, [](const std::wsmatch& m) { return repl_time_impl(m, false);});
w = re_sub(w, P.to_range, repl_to_range);
w = re_sub(w, P.temperature, repl_temperature);
w = replace_measure(w);
while (std::regex_search(w, P.asmd)) w = re_sub(w, P.asmd, repl_asmd);
w = re_sub(w, P.power, repl_power);
w = re_sub(w, P.frac, repl_frac);
w = re_sub(w, P.percentage, repl_percentage);
auto not_digit = [](const std::wstring& s, size_t b, size_t e) {
if (b > 0 && s[b - 1] >= L'0' && s[b - 1] <= L'9') return false;
if (e < s.size() && s[e] >= L'0' && s[e] <= L'9') return false;
return true;
};
w = re_sub_bound(w, P.mobile, not_digit,
[](const std::wsmatch& m) { return phone2str(m.str(0), true);});
w = re_sub_bound(w, P.telephone, not_digit,
[](const std::wsmatch& m) { return phone2str(m.str(0), false);});
w = re_sub(w, P.national, [](const std::wsmatch& m) { return phone2str(m.str(0), false);});
auto range_ok = [](const std::wstring& s, size_t b, size_t e) {
auto bad = [](wchar_t c) {
return (c >= L'0' && c <= L'9') || c == L'+' || c == L'-' || c == L'×' || c == L'÷' ||
c == L'=';
};
if (b > 0 && bad(s[b - 1])) return false;
if (e < s.size() && bad(s[e])) return false;
return true;
};
w = re_sub_bound(w, P.range, range_ok, repl_range);
w = re_sub(w, P.integer, repl_negative);
w = re_sub(w, P.version, repl_version);
w = re_sub(w, P.decimal, repl_number);
w = re_sub(w, P.pos_quant, repl_pos_quant);
w = re_sub(w, P.default_num, repl_default_num);
w = re_sub(w, P.number, repl_number);
w = post_replace(w);
return from_wide(w);
}

private:
std::map<int32_t, int32_t> t2s_;
};

} // namespace textnorm
