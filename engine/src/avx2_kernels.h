// AVX2/FMA kernels for the generic (standard-precision) engine path.
//
// The translation unit is compiled with -mavx2 -mfma on x86-64 (see
// CMakeLists.txt); on any other target, or if the flags are absent,
// the functions below compile to their scalar forms. Callers gate on
// cpu::features() (runtime detection), so the vector path only runs
// where the CPU and OS support it, and the scalar engine code remains
// the floor everywhere. Numerics: the vector forms use FMA, so results
// differ from the scalar mul+add loops at the last-ulp level; the
// correctness gates are correlation/argmax based and absorb that.
#pragma once

#include <cstdint>

namespace avx2k {

// y[i] += a * x[i] for i in [0, n). x and y are contiguous, possibly
// unaligned.
void saxpy(float a, const float* x, float* y, int64_t n);

}  // namespace avx2k
