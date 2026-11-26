#include "avx2_kernels.h"

#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__x86_64__)
// The AVX-512 group kernels below are target-attributed; clang
// needs the intrinsic declarations visible even when the TU is
// compiled without AVX2 flags (GCC declares them either way).
#include <immintrin.h>
#endif

namespace avx2k {

void saxpy(float a, const float* x, float* y, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 va = _mm256_set1_ps(a);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        y0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), y0);
        y1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8), y1);
        _mm256_storeu_ps(y + i, y0);
        _mm256_storeu_ps(y + i + 8, y1);
    }
    for (; i < n; i++) y[i] += a * x[i];
#else
    for (int64_t i = 0; i < n; i++) y[i] += a * x[i];
#endif
}

#if defined(__AVX2__) && defined(__FMA__)
namespace {
// Unpack 32 packed bytes into 8 float vectors of 8 values each, in
// nibble order (high nibble of byte b first). out[i] covers values
// [8i, 8i+8).
inline void w4_unpack64(const uint8_t* prow, __m256* out) {
    const __m256i v = _mm256_loadu_si256((const __m256i*)prow);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
    const __m256i lo = _mm256_and_si256(v, m0f);
    for (int lane = 0; lane < 2; lane++) {
        const __m128i h = lane ? _mm256_extracti128_si256(hi, 1) : _mm256_castsi256_si128(hi);
        const __m128i l = lane ? _mm256_extracti128_si256(lo, 1) : _mm256_castsi256_si128(lo);
        const __m128i u0 = _mm_unpacklo_epi8(h, l);  // values 0..15 of the lane
        const __m128i u1 = _mm_unpackhi_epi8(h, l);  // values 16..31 of the lane
        __m256* dst = out + lane * 4;
        dst[0] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u0));
        dst[1] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u0, 8)));
        dst[2] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u1));
        dst[3] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u1, 8)));
    }
}
}  // namespace
#endif

void w4a_saxpy64_x2(float f0, const uint8_t* prow0, float f1, const uint8_t* prow1,
                    float* acc) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a[8], w[8];
    for (int i = 0; i < 8; i++) a[i] = _mm256_loadu_ps(acc + 8 * i);
    w4_unpack64(prow0, w);
    const __m256 v0 = _mm256_set1_ps(f0);
    for (int i = 0; i < 8; i++) a[i] = _mm256_fmadd_ps(v0, w[i], a[i]);
    w4_unpack64(prow1, w);
    const __m256 v1 = _mm256_set1_ps(f1);
    for (int i = 0; i < 8; i++) a[i] = _mm256_fmadd_ps(v1, w[i], a[i]);
    for (int i = 0; i < 8; i++) _mm256_storeu_ps(acc + 8 * i, a[i]);
#else
    w4a_saxpy64(f0, prow0, acc);
    w4a_saxpy64(f1, prow1, acc);
#endif
}

void w4a_saxpy64(float f, const uint8_t* prow, float* acc) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 w[8];
    w4_unpack64(prow, w);
    const __m256 vf = _mm256_set1_ps(f);
    for (int i = 0; i < 8; i++) {
        __m256 a = _mm256_loadu_ps(acc + 8 * i);
        _mm256_storeu_ps(acc + 8 * i, _mm256_fmadd_ps(vf, w[i], a));
    }
#else
    for (int b = 0; b < 32; b++) {
        uint8_t byte = prow[b];
        acc[2 * b] += f * (float)(byte >> 4);
        acc[2 * b + 1] += f * (float)(byte & 0xF);
    }
#endif
}

float w4b_dot64(const uint8_t* prow, float sc, float ze, const float* x) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 w[8];
    w4_unpack64(prow, w);
    const __m256 vs = _mm256_set1_ps(sc);
    const __m256 vz = _mm256_set1_ps(ze);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 8; i++) {
        __m256 d = _mm256_mul_ps(_mm256_sub_ps(w[i], vz), vs);
        acc = _mm256_fmadd_ps(d, _mm256_loadu_ps(x + 8 * i), acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
#else
    float a = 0.0f;
    for (int b = 0; b < 32; b++) {
        uint8_t byte = prow[b];
        a += ((float)(byte >> 4) - ze) * sc * x[2 * b];
        a += ((float)(byte & 0xF) - ze) * sc * x[2 * b + 1];
    }
    return a;
#endif
}

float dot_f32(const float* a, const float* b, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    __m256 s = _mm256_add_ps(s0, s1);
    __m128 sh = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
    sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
    float acc = _mm_cvtss_f32(sh);
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
#else
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += (double)a[i] * b[i];
    return (float)s;
#endif
}

void w4_dequant64(const uint8_t* prow, float sc, float ze, float* out) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 w[8];
    w4_unpack64(prow, w);
    const __m256 vs = _mm256_set1_ps(sc), vz = _mm256_set1_ps(ze);
    for (int q = 0; q < 8; q++)
        _mm256_storeu_ps(out + q * 8, _mm256_mul_ps(_mm256_sub_ps(w[q], vz), vs));
#else
    for (int i = 0; i < 64; i++) {
        uint8_t byte = prow[i / 2];
        int nib = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
        out[i] = ((float)nib - ze) * sc;
    }
#endif
}

void pv_n32(const float* a, const float* B, int64_t S, int64_t K, float* y) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(),
           acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    for (int64_t k = 0; k < K; k++) {
        const __m256 bc = _mm256_set1_ps(a[k]);
        const float* row = B + k * S;
        acc0 = _mm256_fmadd_ps(bc, _mm256_loadu_ps(row), acc0);
        acc1 = _mm256_fmadd_ps(bc, _mm256_loadu_ps(row + 8), acc1);
        acc2 = _mm256_fmadd_ps(bc, _mm256_loadu_ps(row + 16), acc2);
        acc3 = _mm256_fmadd_ps(bc, _mm256_loadu_ps(row + 24), acc3);
    }
    _mm256_storeu_ps(y, acc0);
    _mm256_storeu_ps(y + 8, acc1);
    _mm256_storeu_ps(y + 16, acc2);
    _mm256_storeu_ps(y + 24, acc3);
#else
    for (int j = 0; j < 32; j++) y[j] = 0.0f;
    for (int64_t k = 0; k < K; k++) {
        float av = a[k];
        const float* row = B + k * S;
        for (int j = 0; j < 32; j++) y[j] += av * row[j];
    }
#endif
}

int32_t w4b_idot64(const uint8_t* pg, const int8_t* xq) {
#if defined(__AVX2__)
    // Two 16-byte halves; per half, nibble-unpack to 32 uint8 in the
    // [hi, lo] interleave the packed layout uses, maddubs against
    // the int8 activations (pairwise i16), madd to i32.
    auto half = [](const uint8_t* p, const int8_t* x) -> __m128i {
        __m128i pv = _mm_loadu_si128((const __m128i*)p);
        __m128i lo = _mm_and_si128(pv, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(pv, 4), _mm_set1_epi8(0x0F));
        __m128i w0 = _mm_unpacklo_epi8(hi, lo);
        __m128i w1 = _mm_unpackhi_epi8(hi, lo);
        __m128i x0 = _mm_loadu_si128((const __m128i*)x);
        __m128i x1 = _mm_loadu_si128((const __m128i*)(x + 16));
        __m128i s0 = _mm_maddubs_epi16(w0, x0);
        __m128i s1 = _mm_maddubs_epi16(w1, x1);
        __m128i one = _mm_set1_epi16(1);
        return _mm_add_epi32(_mm_madd_epi16(s0, one), _mm_madd_epi16(s1, one));
    };
    __m128i acc = _mm_add_epi32(half(pg, xq), half(pg + 16, xq + 32));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return _mm_cvtsi128_si32(acc);
#else
    int32_t s = 0;
    for (int b = 0; b < 32; b++)
        s += (int32_t)xq[2 * b] * (pg[b] >> 4) + (int32_t)xq[2 * b + 1] * (pg[b] & 0xF);
    return s;
#endif
}

float w4b_dotraw64(const uint8_t* prow, const float* x) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 w[8];
    w4_unpack64(prow, w);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 8; i++) acc = _mm256_fmadd_ps(w[i], _mm256_loadu_ps(x + 8 * i), acc);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
#else
    float a = 0.0f;
    for (int b = 0; b < 32; b++) {
        uint8_t byte = prow[b];
        a += (float)(byte >> 4) * x[2 * b];
        a += (float)(byte & 0xF) * x[2 * b + 1];
    }
    return a;
#endif
}

#if defined(__x86_64__)
namespace {
// AVX-512 unpack of the same 32 packed bytes into 4 zmm, values in
// the same nibble order as w4_unpack64: out[lane*2] covers the
// lane's values 0..15, out[lane*2+1] its values 16..31.
__attribute__((target("avx512f"))) inline void w4_unpack64_512(const uint8_t* prow,
                                                               __m512* out) {
    const __m256i v = _mm256_loadu_si256((const __m256i*)prow);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
    const __m256i lo = _mm256_and_si256(v, m0f);
    for (int lane = 0; lane < 2; lane++) {
        const __m128i h = lane ? _mm256_extracti128_si256(hi, 1) : _mm256_castsi256_si128(hi);
        const __m128i l = lane ? _mm256_extracti128_si256(lo, 1) : _mm256_castsi256_si128(lo);
        const __m128i u0 = _mm_unpacklo_epi8(h, l);
        const __m128i u1 = _mm_unpackhi_epi8(h, l);
        out[lane * 2] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u0));
        out[lane * 2 + 1] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u1));
    }
}
}  // namespace

__attribute__((target("avx512f"))) void w4a_saxpy64_512(float f, const uint8_t* prow,
                                                        float* acc) {
    __m512 w[4];
    w4_unpack64_512(prow, w);
    const __m512 vf = _mm512_set1_ps(f);
    for (int i = 0; i < 4; i++)
        _mm512_storeu_ps(acc + 16 * i,
                         _mm512_fmadd_ps(vf, w[i], _mm512_loadu_ps(acc + 16 * i)));
}

__attribute__((target("avx512f"))) float w4b_dot64_512(const uint8_t* prow, float sc,
                                                       float ze, const float* x) {
    __m512 w[4];
    w4_unpack64_512(prow, w);
    const __m512 vs = _mm512_set1_ps(sc);
    const __m512 vz = _mm512_set1_ps(ze);
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < 4; i++) {
        __m512 d = _mm512_mul_ps(_mm512_sub_ps(w[i], vz), vs);
        acc = _mm512_fmadd_ps(d, _mm512_loadu_ps(x + 16 * i), acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#endif

namespace {
// Elementwise op codes, mirroring free_engine's BinOp / UnOp order.
enum { EB_ADD, EB_SUB, EB_MUL, EB_DIV, EB_POW, EB_MAX, EB_LESS, EB_GREATER, EB_EQUAL, EB_OR, EB_AND };
enum { EU_NEG, EU_SQRT, EU_SIN, EU_COS, EU_EXP, EU_TANH, EU_SIGMOID, EU_RELU, EU_ABS, EU_LEAKY };

#if defined(__AVX2__) && defined(__FMA__)
template <int OP>
inline __m256 ebinop_vec(__m256 a, __m256 b) {
    switch (OP) {
        case EB_ADD: return _mm256_add_ps(a, b);
        case EB_SUB: return _mm256_sub_ps(a, b);
        case EB_MUL: return _mm256_mul_ps(a, b);
        case EB_DIV: return _mm256_div_ps(a, b);
        case EB_LESS: return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_LT_OQ), _mm256_set1_ps(1.0f));
        case EB_GREATER: return _mm256_and_ps(_mm256_cmp_ps(b, a, _CMP_LT_OQ), _mm256_set1_ps(1.0f));
        case EB_EQUAL: return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_EQ_OQ), _mm256_set1_ps(1.0f));
        case EB_OR:
            return _mm256_and_ps(
                _mm256_or_ps(_mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_NEQ_OQ),
                             _mm256_cmp_ps(b, _mm256_setzero_ps(), _CMP_NEQ_OQ)),
                _mm256_set1_ps(1.0f));
        case EB_AND:
            return _mm256_and_ps(
                _mm256_and_ps(_mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_NEQ_OQ),
                              _mm256_cmp_ps(b, _mm256_setzero_ps(), _CMP_NEQ_OQ)),
                _mm256_set1_ps(1.0f));
    }
    return a;
}
template <int OP>
inline __m256 eunop_vec(__m256 a, __m256 alpha) {
    switch (OP) {
        case EU_NEG: return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
        case EU_SQRT: return _mm256_sqrt_ps(a);
        case EU_RELU: return _mm256_max_ps(a, _mm256_setzero_ps());
        case EU_ABS: return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        case EU_LEAKY: {
            __m256 m = _mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_GT_OQ);
            return _mm256_blendv_ps(_mm256_mul_ps(alpha, a), a, m);
        }
    }
    return a;
}
#endif

template <int OP>
inline float ebinop_scalar(float a, float b) {
    switch (OP) {
        case EB_ADD: return a + b;
        case EB_SUB: return a - b;
        case EB_MUL: return a * b;
        case EB_DIV: return a / b;
        case EB_LESS: return a < b ? 1.0f : 0.0f;
        case EB_GREATER: return a > b ? 1.0f : 0.0f;
        case EB_EQUAL: return a == b ? 1.0f : 0.0f;
        case EB_OR: return (a != 0 || b != 0) ? 1.0f : 0.0f;
        case EB_AND: return (a != 0 && b != 0) ? 1.0f : 0.0f;
    }
    return a;
}
template <int OP>
inline float eunop_scalar(float a, float alpha) {
    switch (OP) {
        case EU_NEG: return -a;
        case EU_SQRT: return std::sqrt(a);
        case EU_RELU: return a > 0 ? a : 0.0f;
        case EU_ABS: return std::fabs(a);
        case EU_LEAKY: return a > 0 ? a : alpha * a;
    }
    return a;
}

template <int OP>
void ebinop_run(const float* a, const float* b, float* y, int64_t n, bool as, bool bs) {
    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 av = as ? _mm256_set1_ps(a[0]) : _mm256_setzero_ps();
    const __m256 bv = bs ? _mm256_set1_ps(b[0]) : _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = as ? av : _mm256_loadu_ps(a + i);
        __m256 vb = bs ? bv : _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(y + i, ebinop_vec<OP>(va, vb));
    }
#endif
    for (; i < n; i++) y[i] = ebinop_scalar<OP>(a[as ? 0 : i], b[bs ? 0 : i]);
}
template <int OP>
void eunop_run(const float* a, float* y, int64_t n, float alpha) {
    int64_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 al = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, eunop_vec<OP>(_mm256_loadu_ps(a + i), al));
#endif
    for (; i < n; i++) y[i] = eunop_scalar<OP>(a[i], alpha);
}
}  // namespace

bool binop_f32(int op, const float* a, const float* b, float* y, int64_t n, bool a_scalar,
               bool b_scalar) {
    switch (op) {
        case EB_ADD: ebinop_run<EB_ADD>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_SUB: ebinop_run<EB_SUB>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_MUL: ebinop_run<EB_MUL>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_DIV: ebinop_run<EB_DIV>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_LESS: ebinop_run<EB_LESS>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_GREATER: ebinop_run<EB_GREATER>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_EQUAL: ebinop_run<EB_EQUAL>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_OR: ebinop_run<EB_OR>(a, b, y, n, a_scalar, b_scalar); return true;
        case EB_AND: ebinop_run<EB_AND>(a, b, y, n, a_scalar, b_scalar); return true;
        default: return false;
    }
}
bool unop_f32(int op, const float* a, float* y, int64_t n, float alpha) {
    switch (op) {
        case EU_NEG: eunop_run<EU_NEG>(a, y, n, alpha); return true;
        case EU_SQRT: eunop_run<EU_SQRT>(a, y, n, alpha); return true;
        case EU_RELU: eunop_run<EU_RELU>(a, y, n, alpha); return true;
        case EU_ABS: eunop_run<EU_ABS>(a, y, n, alpha); return true;
        case EU_LEAKY: eunop_run<EU_LEAKY>(a, y, n, alpha); return true;
        default: return false;
    }
}

namespace {
inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal half: normalize.
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
}  // namespace

void saxpy_f16(float a, const uint16_t* x, float* y, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    const __m256 va = _mm256_set1_ps(a);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + i)));
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, xv, _mm256_loadu_ps(y + i)));
    }
    if (i < n) {
        uint16_t tmp[8] = {};
        std::memcpy(tmp, x + i, (size_t)(n - i) * 2);
        __m256 xv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)tmp));
        float ty[8];
        _mm256_storeu_ps(ty, _mm256_fmadd_ps(va, xv, _mm256_loadu_ps(y + i)));
        std::memcpy(y + i, ty, (size_t)(n - i) * 4);
    }
#else
    for (int64_t i = 0; i < n; i++) y[i] += a * f16_bits_to_f32(x[i]);
#endif
}

float dot_f16(const float* a, const uint16_t* b, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 bv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), bv, acc);
    }
    if (i < n) {
        uint16_t tmp[8] = {};
        std::memcpy(tmp, b + i, (size_t)(n - i) * 2);
        __m256 bv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)tmp));
        float ta[8] = {};
        std::memcpy(ta, a + i, (size_t)(n - i) * 4);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ta), bv, acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
#else
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += a[i] * f16_bits_to_f32(b[i]);
    return s;
#endif
}

void f32_to_f16(const float* src, uint16_t* dst, int64_t n) {
#if defined(__AVX2__) && defined(__F16C__)
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), h);
    }
    if (i < n) {
        float ts[8] = {};
        std::memcpy(ts, src + i, (size_t)(n - i) * 4);
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(ts), _MM_FROUND_TO_NEAREST_INT);
        uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h);
        std::memcpy(dst + i, tmp, (size_t)(n - i) * 2);
    }
#else
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits;
        std::memcpy(&bits, src + i, 4);
        uint32_t sign = (bits >> 16) & 0x8000;
        int exp = (int)((bits >> 23) & 0xFF) - 112;
        uint32_t mant = bits & 0x7FFFFF;
        uint16_t h;
        if (exp <= 0) h = (uint16_t)sign;  // underflow to zero (denormal halves ignored)
        else if (exp >= 31) h = (uint16_t)(sign | 0x7C00);
        else h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
        dst[i] = h;
    }
#endif
}

}  // namespace avx2k
