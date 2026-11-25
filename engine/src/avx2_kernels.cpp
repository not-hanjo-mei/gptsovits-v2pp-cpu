#include "avx2_kernels.h"

#if defined(__AVX2__) && defined(__FMA__)
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

}  // namespace avx2k
