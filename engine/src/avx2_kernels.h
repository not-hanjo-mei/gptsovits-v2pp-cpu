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
// Plain fp32 dot product, AVX2 FMA with a horizontal reduction. The
// scalar fallback accumulates in double, matching the attention
// fast path's historical numerics on machines without AVX2.
float dot_f32(const float* a, const float* b, int64_t n);
// Decode probs*V for one head: y[32] = sum_k a[k] * B[k*S .. k*S+32),
// accumulators held in registers across the K loop. One FMA per
// element in k-ascending order, exactly as the SAXPY form.
void pv_n32(const float* a, const float* B, int64_t S, int64_t K, float* y);
// Dequantize one 64-element group of a mode-B row (32 packed bytes)
// into fp32 with the group's scale and zero point.
void w4_dequant64(const uint8_t* prow, float sc, float ze, float* out);
// Raw nibble dot: sum of unpacked weights times x, with no scale or
// zero-point applied. Callers fold those in algebraically --
// sum((w - ze) * sc * x) == sc * dotraw - sc * ze * sum(x) -- which
// drops the per-vector sub+mul from the inner loop; the group sums
// of x are computed once per GEMV, not once per row. The result
// differs from w4b_dot64's accumulation at the ulp level.
float w4b_dotraw64(const uint8_t* prow, const float* x);

#if defined(__x86_64__)
// AVX-512F versions of the two group kernels (target-attributed in
// the .cpp; callers gate on cpu::features().avx512). The SAXPY is
// elementwise and bit-identical to w4a_saxpy64; the dot reduces
// over 16-lane partials instead of 8-lane, so its last-ulp rounding
// differs from w4b_dot64.
__attribute__((target("avx512f"))) void w4a_saxpy64_512(float f, const uint8_t* prow,
                                                         float* acc);
__attribute__((target("avx512f"))) float w4b_dot64_512(const uint8_t* prow, float sc,
                                                       float ze, const float* x);
#endif

// Elementwise f32 kernels over contiguous ranges. op codes mirror
// free_engine's enum orderings (BinOp: ADD SUB MUL DIV POW MAXIMUM
// LESS GREATER EQUAL OR AND; UnOp: NEG SQRT SIN COS EXP TANH
// SIGMOID RELU ABS LEAKY). Only ops whose vector form is exactly
// the IEEE scalar result are handled (ADD/SUB/MUL/DIV and the
// comparisons; NEG/SQRT/RELU/ABS/LEAKY); the functions return false
// for the rest and the caller keeps its scalar path (POW, MAXIMUM
// with its NaN corner, and the libm transcendentals).
// a_scalar / b_scalar broadcast element 0 across the range.
bool binop_f32(int op, const float* a, const float* b, float* y, int64_t n, bool a_scalar,
               bool b_scalar);
bool unop_f32(int op, const float* a, float* y, int64_t n, float alpha);

// fp16-stored weight kernels (M2): the values are converted on the
// fly with F16C; compute stays fp32.
//
// y[i] += a * (float)x16[i] for i in [0, n).
void saxpy_f16(float a, const uint16_t* x, float* y, int64_t n);
// Returns sum over i in [0, n) of a[i] * (float)b16[i].
float dot_f16(const float* a, const uint16_t* b, int64_t n);
// Bulk conversion for plan compile time.
void f32_to_f16(const float* src, uint16_t* dst, int64_t n);

}  // namespace avx2k
