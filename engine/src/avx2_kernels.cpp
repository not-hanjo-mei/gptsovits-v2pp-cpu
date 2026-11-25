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

}  // namespace avx2k
