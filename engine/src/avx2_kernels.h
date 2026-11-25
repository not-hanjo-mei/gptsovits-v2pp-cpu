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

// W4 packed GEMV group kernels (64 weights = 32 packed bytes; nibble
// order in both pack modes is high nibble first: value 2b is the high
// nibble of byte b, value 2b+1 the low nibble).
//
// acc[j] += f * nib(j) for j in [0, 64). Mode A inner loop: f already
// carries the group scale (f = x_i * sc), the zero-point correction
// is the caller's scalar term, exactly as in gemv_a_w4_scalar.
void w4a_saxpy64(float f, const uint8_t* prow, float* acc);

// Returns sum over k in [0, 64) of (nib(k) - ze) * sc * x[k]. Mode B
// inner loop: one k-group of one output row.
float w4b_dot64(const uint8_t* prow, float sc, float ze, const float* x);

}  // namespace avx2k
