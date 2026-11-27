// ORT-free static-plan engine: plan compiler + executor + kernels.
// Design notes live in ORT-FREE.md. This file includes packed_core.h
// (split out of packed.h: the thread pool, the fused-dequant kernels
// packed::gemm_* and the BlobStore -- none of them reference ORT, and
// binaries built on this engine link no onnxruntime library; verify
// with ldd). packed.h re-includes packed_core.h and adds the ORT glue.
#include "free_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <random>
#include <stdexcept>

#include "avx2_kernels.h"  // M2: AVX2/FMA kernels for the generic fp32 path
#include "cpu.h"
#include "packed_core.h"
#include "vits_kernels.h"  // phase 2: packed conv/gather kernels (ORT-free header)

namespace freeeng {

struct KvCache {
    struct Slot {
        // Uninitialized allocation: capacity is headroom, and
        // zero-filling it at copy-in cost real bind time.
        std::unique_ptr<float[]> buf;
        std::vector<int64_t> shape;      // shape template from copy-in
        int64_t len = -1;                // rows held, -1 = empty
        int64_t cap_rows = 0;
        void reserve(int64_t rows, int64_t rowf) {
            if (rows <= cap_rows) return;
            // new float[n] without parentheses: default-initialized
            // (unfilled). make_unique<float[]>(n) would value-
            // initialize, i.e. zero the whole capacity.
            std::unique_ptr<float[]> nb(new float[(size_t)rows * (size_t)rowf]);
            if (buf && len > 0)
                std::memcpy(nb.get(), buf.get(), (size_t)len * (size_t)rowf * sizeof(float));
            buf = std::move(nb);
            cap_rows = rows;
        }
    };
    std::vector<Slot> slots;             // indexed by pair order (step order)
};

std::shared_ptr<KvCache> make_kv_cache() { return std::make_shared<KvCache>(); }

namespace {

// ------------------------------------------------------------------ RT --
// Internal resolved tensor: explicit strides (elements), always set.
struct RT {
    int dtype = 0;
    uint8_t* p = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    bool contig() const {
        int64_t exp = 1;
        for (int i = (int)shape.size() - 1; i >= 0; i--) {
            if (shape[i] == 0) return true;
            if (shape[i] > 1) {
                if (strides[i] != exp) return false;
                exp *= shape[i];
            }
        }
        return true;
    }
};

std::vector<int64_t> rowmajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size(), 1);
    int64_t acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("free_engine: " + m); }

// Read/write one element as double (slow generic path only).
double read_d(const RT& t, int64_t off) {
    switch (t.dtype) {
        case 1: { float v; memcpy(&v, t.p + off * 4, 4); return v; }
        case 11: { double v; memcpy(&v, t.p + off * 8, 8); return v; }
        case 6: { int32_t v; memcpy(&v, t.p + off * 4, 4); return v; }
        case 7: { int64_t v; memcpy(&v, t.p + off * 8, 8); return (double)v; }
        case 3: { int8_t v; memcpy(&v, t.p + off, 1); return v; }
        default: return t.p[off];
    }
}
void write_d(const RT& t, int64_t off, double x) {
    switch (t.dtype) {
        case 1: { float v = (float)x; memcpy(t.p + off * 4, &v, 4); break; }
        case 11: { double v = x; memcpy(t.p + off * 8, &v, 8); break; }
        case 6: { int32_t v = (int32_t)x; memcpy(t.p + off * 4, &v, 4); break; }
        case 7: { int64_t v = (int64_t)x; memcpy(t.p + off * 8, &v, 8); break; }
        case 3: { int8_t v = (int8_t)x; memcpy(t.p + off, &v, 1); break; }
        default: t.p[off] = (uint8_t)(x != 0); break;
    }
}

// Odometer over `shape`, invoking fn(flat_index) while per-operand
// offsets are maintained by the caller through coord tracking. Here we
// expose coords; callers compute offsets from strides.
template <typename F>
void odometer(const std::vector<int64_t>& shape, F&& fn) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    if (n == 0) return;
    std::vector<int64_t> coord(shape.size(), 0);
    for (int64_t i = 0; i < n; i++) {
        fn(coord);
        for (int d = (int)shape.size() - 1; d >= 0; d--) {
            if (++coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

int64_t off_of(const RT& t, const std::vector<int64_t>& coord, const std::vector<int64_t>& align) {
    // align[d] = dim of t corresponding to output dim d, or -1 (broadcast
    // from a leading position / scalar).
    int64_t off = 0;
    for (size_t d = 0; d < coord.size(); d++) {
        int td = align[d];
        if (td < 0) continue;
        int64_t c = coord[d];
        if (t.shape[td] == 1) c = 0;
        off += c * t.strides[td];
    }
    return off;
}

// Align an operand's dims to the output rank (right-aligned broadcast).
std::vector<int64_t> align_of(const RT& t, size_t out_rank) {
    std::vector<int64_t> a(out_rank, -1);
    int64_t shift = (int64_t)out_rank - (int64_t)t.shape.size();
    for (size_t i = 0; i < t.shape.size(); i++) a[shift + i] = (int64_t)i;
    return a;
}

void copy_tensor(const RT& dst, const RT& src);  // fwd

// Materialize src into fresh contiguous storage (scratch or dst buffer).
RT materialize(const RT& src, uint8_t* buf) {
    RT d;
    d.dtype = src.dtype;
    d.p = buf;
    d.shape = src.shape;
    d.strides = rowmajor(src.shape);
    copy_tensor(d, src);
    return d;
}

void copy_tensor(const RT& dst, const RT& src) {
    if (dst.numel() == 0) return;
    size_t es = onnxm::elem_size(dst.dtype);
    if (dst.contig() && src.contig() && dst.shape == src.shape) {
        memcpy(dst.p, src.p, (size_t)dst.numel() * es);
        return;
    }
    // Strided walk with incremental offsets (dst rank == src rank here
    // by construction of callers; src dims may broadcast from size 1).
    if (dst.shape.size() <= 8 && src.shape.size() == dst.shape.size()) {
        int r = (int)dst.shape.size();
        int64_t ss[8], ds[8];
        for (int d = 0; d < r; d++) {
            ss[d] = src.shape[d] == 1 ? 0 : src.strides[d];
            ds[d] = dst.strides[d];
        }
        int64_t coord[8] = {0};
        int64_t so = 0, dof = 0;
        int64_t n = dst.numel();
        for (int64_t i = 0; i < n; i++) {
            memcpy(dst.p + dof * (int64_t)es, src.p + so * (int64_t)es, es);
            for (int d = r - 1; d >= 0; d--) {
                so += ss[d]; dof += ds[d];
                if (++coord[d] < dst.shape[d]) break;
                coord[d] = 0;
                so -= ss[d] * dst.shape[d]; dof -= ds[d] * dst.shape[d];
            }
        }
        return;
    }
    auto sa = align_of(src, dst.shape.size());
    auto da = align_of(dst, dst.shape.size());
    odometer(dst.shape, [&](const std::vector<int64_t>& c) {
        int64_t so = off_of(src, c, sa);
        int64_t dof = off_of(dst, c, da);
        memcpy(dst.p + dof * (int64_t)es, src.p + so * (int64_t)es, es);
    });
}

// Logical-order copy (reshape semantics): element i of dst in row-major
// order takes element i of src in its own row-major logical order.
void flatten_copy(const RT& dst, const RT& src) {
    size_t es = onnxm::elem_size(dst.dtype);
    if (src.contig() && dst.contig()) {
        memcpy(dst.p, src.p, (size_t)dst.numel() * es);
        return;
    }
    int64_t idx = 0;
    odometer(src.shape, [&](const std::vector<int64_t>& c) {
        int64_t so = 0;
        for (size_t d = 0; d < c.size(); d++) so += c[d] * src.strides[d];
        memcpy(dst.p + idx * (int64_t)es, src.p + so * (int64_t)es, es);
        idx++;
    });
}

// Scratch bump allocator for per-step temporaries (contiguity fixes).
// Chunked: blocks never move once allocated, because a step may hold
// several materialized tensors at once (a growing vector would leave
// the earlier pointers dangling -- that was a real segfault source).
struct Scratch {
    std::vector<std::vector<uint8_t>> blocks;
    size_t bi = 0, top = 0;
    struct Mark { size_t bi, top; };
    Mark mark() const { return {bi, top}; }
    void reset(Mark m) { bi = m.bi; top = m.top; }
    uint8_t* alloc(size_t bytes) {
        if (blocks.empty()) blocks.emplace_back(1 << 20);
        size_t at = (top + 63) & ~(size_t)63;
        if (at + bytes > blocks[bi].size()) {
            size_t want = std::max<size_t>(bytes + 4096, 1 << 20);
            if (bi + 1 < blocks.size() && blocks[bi + 1].size() >= want) {
                bi++;
            } else if (bi + 1 < blocks.size()) {
                blocks[bi + 1].resize(want);
                bi++;
            } else {
                blocks.emplace_back(want);
                bi = blocks.size() - 1;
            }
            at = 0;
        }
        top = at + bytes;
        return blocks[bi].data() + at;
    }
    RT contig_of(const RT& t) {
        if (t.contig()) return t;
        return materialize(t, alloc((size_t)t.numel() * onnxm::elem_size(t.dtype)));
    }
};

// ------------------------------------------------------------- kernels --


// One SAXPY step of k_matmul over a contiguous B row, in fp32 or
// fp16 storage (bh != nullptr selects the f16 copy at the same
// element offset; avx2k carries the scalar fallbacks).
inline void mm_saxpy(bool use_avx2, const uint16_t* bh, float av, const float* browp, float* crow,
                     int64_t n) {
    if (bh) { avx2k::saxpy_f16(av, bh, crow, n); return; }
    if (use_avx2) { avx2k::saxpy(av, browp, crow, n); return; }
    for (int64_t j = 0; j < n; j++) crow[j] += av * browp[j];
}

void k_matmul(const RT& A0, const RT& B0, RT& C, Scratch& sc, const uint16_t* Bf16 = nullptr) {
    // FE_MM_DEBUG=1: dump shapes/strides of the first 400 calls.
    if (std::getenv("FE_MM_DEBUG")) {
        static int mm_dbg = 0;
        if (mm_dbg++ < 400) {
            auto fmt = [](const RT& t) {
                std::string s = "[";
                for (size_t i = 0; i < t.shape.size(); i++) { if (i) s += ","; s += std::to_string(t.shape[i]); }
                s += "]<";
                for (size_t i = 0; i < t.strides.size(); i++) { if (i) s += ","; s += std::to_string(t.strides[i]); }
                return s + ">";
            };
            std::fprintf(stderr, "[mm] A%s B%s C%s\n", fmt(A0).c_str(), fmt(B0).c_str(), fmt(C).c_str());
        }
    }
    // Fast path: batched GEMV against a transposed view of a contiguous
    // matrix (attention scores Q.K^T, one row per head). B0 (..., K, N)
    // with last-two strides (1, K) means the underlying (..., N, K) rows
    // are contiguous: y[b][j] = sum_k a[b][k] * B0.p[b][j*K + k]. The
    // general path below computes the same product as a stride-K SAXPY
    // sweep, re-reading the whole K cache once per k (stride-K gathers);
    // the dot form reads it exactly once, sequentially. Accumulation is
    // double, in k order, as in the original single-batch fast path.
    // The wide variant (row stride S > K, AVX2 only) covers the
    // position-major KV layout and any M, with a float dot kernel.
    if (Bf16 == nullptr && A0.dtype == 1 && B0.dtype == 1 && C.contig()) {
        int ra0 = (int)A0.shape.size(), rb0 = (int)B0.shape.size();
        if (ra0 >= 2 && rb0 >= 2) {
            int64_t M = A0.shape[ra0 - 2], K = A0.shape[ra0 - 1], N = B0.shape[rb0 - 1];
            // B's rows are K contiguous floats at stride S. S == K is
            // the packed case (the original fast path, double
            // accumulation, any M == 1). S > K is the position-major
            // KV layout of the GPT attention scores (rows live 512
            // floats apart in a [L, heads*dim] buffer): the general
            // path reads those as stride-S scalar gathers, so on
            // AVX2 machines a float dot kernel takes any M here.
            const int64_t S = B0.strides[rb0 - 1];
            const bool classic = M == 1 && B0.strides[rb0 - 2] == 1 && S == K;
            const bool wide = cpu::features().avx2 && cpu::features().fma &&
                              B0.strides[rb0 - 2] == 1 && S >= K;
            if ((classic || wide) && A0.strides[ra0 - 1] == 1) {
                int batch = (int)(C.numel() / (M * N));
                auto bst = [](const RT& t, const std::vector<int64_t>& cshape) {
                    std::vector<int64_t> st;
                    int rank = (int)t.shape.size();
                    int b = (int)cshape.size() - 2;
                    for (int i = 0; i < b; i++) {
                        int ti = i - (b - (rank - 2));
                        if (ti < 0 || (t.shape[ti] == 1 && cshape[i] != 1)) st.push_back(0);
                        else st.push_back(t.strides[ti]);
                    }
                    return st;
                };
                std::vector<int64_t> as = bst(A0, C.shape), bs = bst(B0, C.shape);
                std::vector<int64_t> bdims(C.shape.begin(), C.shape.end() - 2);
                std::vector<int64_t> aoff(batch), boff(batch);
                for (int b = 0; b < batch; b++) {
                    int64_t rem = b, ao = 0, bo = 0;
                    for (int i = (int)bdims.size() - 1; i >= 0; i--) {
                        int64_t c = bdims[i] ? rem % bdims[i] : 0;
                        rem /= bdims[i] ? bdims[i] : 1;
                        ao += c * as[i];
                        bo += c * bs[i];
                    }
                    aoff[b] = ao;
                    boff[b] = bo;
                }
                const float* ap0 = (const float*)A0.p;
                const float* bp0 = (const float*)B0.p;
                float* cp0 = (float*)C.p;
                const int64_t arow = A0.strides[ra0 - 2];
                int64_t work = (int64_t)batch * M * N;
                auto body = [&](int idx, int nt) {
                    packed::chunk(idx, nt, work, [&](int64_t w0, int64_t w1) {
                        for (int64_t w = w0; w < w1; w++) {
                            int b = (int)(w / (M * N));
                            int64_t rem = w % (M * N);
                            int64_t m = rem / N, j = rem % N;
                            const float* a = ap0 + aoff[b] + m * arow;
                            const float* row = bp0 + boff[b] + j * S;
                            if (wide) {
                                cp0[w] = avx2k::dot_f32(a, row, K);
                                continue;
                            }
                            double s = 0;
                            for (int64_t k = 0; k < K; k++) s += (double)a[k] * row[k];
                            cp0[w] = (float)s;
                        }
                    });
                };
                // Small GEMVs serially: a pool rendezvous costs more
                // than the whole product at this size.
                if (work * K >= (1 << 20)) packed::kernel_pool().run(body);
                else body(0, 1);
                return;
            }
        }
    }
    // No contiguity forcing: every access below goes through strides,
    // and materializing a transposed view (attention V) costs more
    // than the strided reads it saves.
    const bool use_avx2 = cpu::features().avx2 && cpu::features().fma;
    RT A = A0, B = B0;
    // Shapes: A [..., M, K], B [..., K, N] with broadcast batch dims.
    int ra = (int)A.shape.size(), rb = (int)B.shape.size();
    int64_t M = A.shape[ra - 2], K = A.shape[ra - 1], N = B.shape[rb - 1];
    int batch = (int)C.numel() / (int)(M * N);
    // Batch strides (elements) after broadcast expansion.
    auto batch_strides = [](const RT& t, const std::vector<int64_t>& cshape, int mat_dims) {
        std::vector<int64_t> st;
        int rank = (int)t.shape.size();
        int b = (int)cshape.size() - mat_dims;
        for (int i = 0; i < b; i++) {
            int ti = i - (b - (rank - mat_dims));
            if (ti < 0 || t.shape[ti] == 1 && cshape[i] != 1) st.push_back(0);
            else st.push_back(t.strides[ti]);
        }
        return st;
    };
    std::vector<int64_t> as = batch_strides(A, C.shape, 2), bs = batch_strides(B, C.shape, 2);
    std::vector<int64_t> bdims(C.shape.begin(), C.shape.end() - 2);
    const float* ap0 = (const float*)A.p;
    const float* bp0 = (const float*)B.p;
    float* cp0 = (float*)C.p;
    int64_t arow = A.strides[ra - 2], acol = A.strides[ra - 1];
    int64_t brow = B.strides[rb - 2], bcol = B.strides[rb - 1];
    // Per-batch operand offsets.
    std::vector<int64_t> aoff(batch), boff(batch);
    for (int b = 0; b < batch; b++) {
        int64_t rem = b, ao = 0, bo = 0;
        for (int i = (int)bdims.size() - 1; i >= 0; i--) {
            int64_t c = bdims[i] ? rem % bdims[i] : 0;
            rem /= bdims[i] ? bdims[i] : 1;
            ao += c * as[i];
            bo += c * bs[i];
        }
        aoff[b] = ao;
        boff[b] = bo;
    }
    // fp16-stored B in transposed layout (brow == 1, bcol == K: the
    // stored rows are the output columns), decode shape M == 1: one
    // dot product per output, the f16 row read exactly once.
    if (Bf16 && M == 1 && brow == 1 && bcol == K && acol == 1) {
        auto dot_body = [&](int idx, int nt) {
            packed::chunk(idx, nt, (int64_t)batch * N, [&](int64_t w0, int64_t w1) {
                for (int64_t w = w0; w < w1; w++) {
                    int b = (int)(w / N);
                    int64_t j = w % N;
                    // avx2k::dot_f16 carries its own scalar
                    // fallback; Bf16 is only ever set on machines
                    // whose features passed the compile-time check.
                    cp0[w] = avx2k::dot_f16(ap0 + aoff[b], Bf16 + boff[b] + j * K, K);
                }
            });
        };
        if ((int64_t)batch * N * K >= (1 << 20)) packed::kernel_pool().run(dot_body);
        else dot_body(0, 1);
        return;
    }
    // Decode probs*V: one query row per head against a
    // position-major V cache (row stride 512), N == 32 outputs.
    // The tiled path below keeps the 32 accumulators in memory;
    // this path keeps them in registers across the whole K loop.
    // Same FMA order as mm_saxpy (k ascending, one FMA per
    // element), so results are bit-identical.
    if (!Bf16 && use_avx2 && M == 1 && N == 32 && acol == 1 && bcol == 1 && brow >= 32 &&
        C.contig()) {
        auto pv_body = [&](int idx, int nt) {
            packed::chunk(idx, nt, batch, [&](int64_t b0, int64_t b1) {
                for (int64_t b = b0; b < b1; b++)
                    avx2k::pv_n32(ap0 + aoff[b], bp0 + boff[b], brow, K, cp0 + b * 32);
            });
        };
        if ((int64_t)batch * N * K >= (1 << 20)) packed::kernel_pool().run(pv_body);
        else pv_body(0, 1);
        return;
    }
    int64_t rows = (int64_t)batch * M;
    if (rows >= 4) {
        // One pool run over all (batch, row) pairs: per-batch pool
        // synchronization would dominate multi-head attention. Rows are
        // processed in tiles of 16 with k as the outer loop, so each B
        // row segment is read once per tile instead of once per row
        // (prefill M~200 against a 16 MB B otherwise re-reads B ~200x).
        const int64_t RT = 16;
        int64_t ntiles = (rows + RT - 1) / RT;
        auto tile_body = [&](int idx, int nt) {
            packed::chunk(idx, nt, ntiles, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; t++) {
                    int64_t r0 = t * RT, r1 = std::min(rows, r0 + RT);
                    for (int64_t r = r0; r < r1; r++) {
                        float* crow = cp0 + r * N;
                        for (int64_t j = 0; j < N; j++) crow[j] = 0.0f;
                    }
                    for (int64_t k = 0; k < K; k++) {
                        for (int64_t r = r0; r < r1; r++) {
                            int b = (int)(r / M);
                            int64_t m = r % M;
                            const float* a = ap0 + aoff[b] + m * arow;
                            const float* bb = bp0 + boff[b];
                            float av = a[k * acol];
                            const float* browp = bb + k * brow;
                            float* crow = cp0 + r * N;
                            if (bcol == 1) {
                                mm_saxpy(use_avx2, Bf16 ? Bf16 + (browp - bp0) : nullptr, av, browp,
                                         crow, N);
                            } else {
                                for (int64_t j = 0; j < N; j++) crow[j] += av * browp[j * bcol];
                            }
                        }
                    }
                }
            });
        };
        // Decode attention products are ~1e5 MAC: serial wins over a
        // pool rendezvous. Prefill stays pooled.
        if ((int64_t)batch * M * N * K >= (1 << 20)) packed::kernel_pool().run(tile_body);
        else tile_body(0, 1);
        return;
    }
    for (int b = 0; b < batch; b++) {
        const float* a = ap0 + aoff[b];
        const float* bb = bp0 + boff[b];
        float* c = cp0 + (int64_t)b * M * N;
        if (M == 1) {
            // GEMV: y[j] = sum_k a[k] * B[k][j]; parallel over j blocks.
            auto gemv_body = [&](int idx, int nt) {
                packed::chunk(idx, nt, N, [&](int64_t j0, int64_t j1) {
                    for (int64_t j = j0; j < j1; j++) c[j] = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        float av = a[k * acol];
                        const float* browp = bb + k * brow + j0 * bcol;
                        if (bcol == 1) {
                            mm_saxpy(use_avx2, Bf16 ? Bf16 + (browp - bp0) : nullptr, av, browp,
                                     c + j0, j1 - j0);
                        } else {
                            for (int64_t j = j0; j < j1; j++) c[j] += av * browp[(j - j0) * bcol];
                        }
                    }
                });
            };
            if (M * N * K >= (1 << 20)) packed::kernel_pool().run(gemv_body);
            else gemv_body(0, 1);
        } else {
            auto gemm_body = [&](int idx, int nt) {
                packed::chunk(idx, nt, M, [&](int64_t m0, int64_t m1) {
                    for (int64_t m = m0; m < m1; m++) {
                        float* crow = c + m * N;
                        for (int64_t j = 0; j < N; j++) crow[j] = 0.0f;
                        for (int64_t k = 0; k < K; k++) {
                            float av = a[m * arow + k * acol];
                            const float* browp = bb + k * brow;
                            if (bcol == 1) {
                                mm_saxpy(use_avx2, Bf16 ? Bf16 + (browp - bp0) : nullptr, av, browp,
                                         crow, N);
                            } else {
                                for (int64_t j = 0; j < N; j++) crow[j] += av * browp[j * bcol];
                            }
                        }
                    }
                });
            };
            if (M * N * K >= (1 << 20)) packed::kernel_pool().run(gemm_body);
            else gemm_body(0, 1);
        }
    }
}

void k_layernorm(const RT& X0, const RT* W, const RT* B, RT& Y, int64_t axis, float eps, Scratch& sc) {
    RT X = sc.contig_of(X0);
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = 1;
    for (int i = (int)axis; i < rank; i++) D *= X.shape[i];
    int64_t rows = X.numel() / D;
    const float* x = (const float*)X.p;
    float* y = (float*)Y.p;
    const float* w = W ? (const float*)W->p : nullptr;
    const float* b = B ? (const float*)B->p : nullptr;
    // W/B may be strided views; normalize through contig copies if needed.
    RT wc, bc;
    if (W && !W->contig()) wc = sc.contig_of(*W), w = (const float*)wc.p;
    if (B && !B->contig()) bc = sc.contig_of(*B), b = (const float*)bc.p;
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; r++) {
                const float* xr = x + r * D;
                float* yr = y + r * D;
                double mean = 0;
                for (int64_t i = 0; i < D; i++) mean += xr[i];
                mean /= D;
                double var = 0;
                for (int64_t i = 0; i < D; i++) { double d = xr[i] - mean; var += d * d; }
                var /= D;
                double inv = 1.0 / std::sqrt(var + eps);
                for (int64_t i = 0; i < D; i++) {
                    float v = (float)((xr[i] - mean) * inv);
                    if (w) v *= w[i];
                    if (b) v += b[i];
                    yr[i] = v;
                }
            }
        });
    };
    // Small work serially: the pool barrier costs more than the math.
    if (X.numel() < 8192) body(0, 1);
    else packed::kernel_pool().run(body);
}

// InstanceNormalization: per (n, c) mean/var over every remaining
// dim; scale/bias are [C]. On a contiguous tensor each (n, c)
// slice is one contiguous block of S elements.
void k_instancenorm(const RT& X0, const RT* W, const RT* B, RT& Y, float eps, Scratch& sc) {
    RT X = sc.contig_of(X0);
    int64_t N = X.shape.size() > 0 ? X.shape[0] : 1;
    int64_t C = X.shape.size() > 1 ? X.shape[1] : 1;
    int64_t S = X.numel() / (N * C);
    const float* x = (const float*)X.p;
    float* y = (float*)Y.p;
    const float* w = W ? (const float*)W->p : nullptr;
    const float* b = B ? (const float*)B->p : nullptr;
    RT wc, bc;
    if (W && !W->contig()) wc = sc.contig_of(*W), w = (const float*)wc.p;
    if (B && !B->contig()) bc = sc.contig_of(*B), b = (const float*)bc.p;
    for (int64_t i = 0; i < N * C; i++) {
        const float* xr = x + i * S;
        float* yr = y + i * S;
        double mean = 0;
        for (int64_t j = 0; j < S; j++) mean += xr[j];
        mean /= S;
        double var = 0;
        for (int64_t j = 0; j < S; j++) { double d = xr[j] - mean; var += d * d; }
        var /= S;
        double inv = 1.0 / std::sqrt(var + eps);
        int64_t c = i % C;
        for (int64_t j = 0; j < S; j++) {
            float v = (float)((xr[j] - mean) * inv);
            if (w) v *= w[c];
            if (b) v += b[c];
            yr[j] = v;
        }
    }
}

// Fused attention (fuse_one_attn): Q [B,H,M,32], K as the transposed
// view [B,H,32,L] (position stride = strides[3]), V [B,H,L,32].
// Scores use the same float dot as k_matmul's wide path (the
// double-accumulation classic path when the position stride is 32,
// mirroring its S == K condition), the softmax replicates
// k_softmax's row math exactly, and probs*V goes through pv_n32 --
// the three-step sequence is bit-identical.
void k_attn(const RT& Q, const RT& K, const RT& V, RT& Y, Scratch& sc, float score_scale) {
    if (Q.shape.size() != 4 || K.shape.size() != 4 || V.shape.size() != 4)
        fail("attn: rank");
    const int64_t B = Q.shape[0], H = Q.shape[1], M = Q.shape[2], D = Q.shape[3];
    const int64_t L = K.shape[3];
    if (D != 32 || K.shape[0] != B || K.shape[1] != H || K.shape[2] != 32 ||
        V.shape[0] != B || V.shape[1] != H || V.shape[2] != L || V.shape[3] != 32 ||
        Q.strides[3] != 1 || K.strides[2] != 1 || V.strides[3] != 1 || !Y.contig() ||
        Y.shape != Q.shape)
        fail("attn: layout");
    float* scores = (float*)sc.alloc((size_t)L * sizeof(float));
    const bool classic = K.strides[3] == 32;
    const bool use_avx2 = cpu::features().avx2 && cpu::features().fma;
    for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < H; h++) {
            const float* kb = (const float*)K.p + b * K.strides[0] + h * K.strides[1];
            const float* vb = (const float*)V.p + b * V.strides[0] + h * V.strides[1];
            for (int64_t m = 0; m < M; m++) {
                const float* q = (const float*)Q.p + b * Q.strides[0] + h * Q.strides[1] +
                                 m * Q.strides[2];
                for (int64_t l = 0; l < L; l++) {
                    const float* krow = kb + l * K.strides[3];
                    if (classic) {
                        double s = 0;
                        for (int64_t d = 0; d < 32; d++) s += (double)q[d] * krow[d];
                        scores[l] = (float)s;
                    } else if (use_avx2) {
                        scores[l] = avx2k::dot_f32(q, krow, 32);
                    } else {
                        float s = 0;
                        for (int64_t d = 0; d < 32; d++) s += q[d] * krow[d];
                        scores[l] = s;
                    }
                }
                if (score_scale != 1.0f)
                    for (int64_t l = 0; l < L; l++)
                        scores[l] = (float)((double)scores[l] * (double)score_scale);
                float mx = -INFINITY;
                for (int64_t l = 0; l < L; l++) mx = std::max(mx, scores[l]);
                double sum = 0;
                for (int64_t l = 0; l < L; l++) {
                    scores[l] = std::exp(scores[l] - mx);
                    sum += scores[l];
                }
                float inv = (float)(1.0 / sum);
                for (int64_t l = 0; l < L; l++) scores[l] *= inv;
                float* y = (float*)Y.p + ((b * H + h) * M + m) * 32;
                if (use_avx2) {
                    avx2k::pv_n32(scores, vb, V.strides[2], L, y);
                } else {
                    for (int64_t d = 0; d < 32; d++) y[d] = 0.0f;
                    for (int64_t l = 0; l < L; l++) {
                        float av = scores[l];
                        const float* vrow = vb + l * V.strides[2];
                        for (int64_t d = 0; d < 32; d++) y[d] += av * vrow[d];
                    }
                }
            }
        }
}

void k_softmax(const RT& X0, RT& Y, int64_t axis, Scratch& sc) {
    RT X = sc.contig_of(X0);
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (size_t i = (size_t)axis + 1; i < X.shape.size(); i++) inner *= X.shape[i];
    const float* x = (const float*)X.p;
    float* y = (float*)Y.p;
    int64_t lines = outer * inner;
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, lines, [&](int64_t l0, int64_t l1) {
            for (int64_t l = l0; l < l1; l++) {
                int64_t o = l / inner, in = l % inner;
                const float* xp = x + o * D * inner + in;
                float* yp = y + o * D * inner + in;
                float mx = -INFINITY;
                for (int64_t i = 0; i < D; i++) mx = std::max(mx, xp[i * inner]);
                double sum = 0;
                for (int64_t i = 0; i < D; i++) { yp[i * inner] = std::exp(xp[i * inner] - mx); sum += yp[i * inner]; }
                float inv = (float)(1.0 / sum);
                for (int64_t i = 0; i < D; i++) yp[i * inner] *= inv;
            }
        });
    };
    if (X.numel() < 8192) body(0, 1);
    else packed::kernel_pool().run(body);
}


// Elementwise binary op codes, shared by compiler + kernels.
enum class BinOp { ADD, SUB, MUL, DIV, POW, MAXIMUM, LESS, GREATER, EQUAL, OR, AND };
enum class UnOp { NEG, SQRT, SIN, COS, EXP, TANH, SIGMOID, RELU, ABS, LEAKY, ERF, GELU, GELUT, SOFTPLUS, LOG };

double binop_apply(BinOp op, double a, double b) {
    switch (op) {
        case BinOp::ADD: return a + b;
        case BinOp::SUB: return a - b;
        case BinOp::MUL: return a * b;
        case BinOp::DIV: return a / b;
        case BinOp::POW: return std::pow(a, b);
        case BinOp::MAXIMUM: return std::max(a, b);
        case BinOp::LESS: return a < b ? 1 : 0;
        case BinOp::GREATER: return a > b ? 1 : 0;
        case BinOp::EQUAL: return a == b ? 1 : 0;
        case BinOp::OR: return (a != 0 || b != 0) ? 1 : 0;
        case BinOp::AND: return (a != 0 && b != 0) ? 1 : 0;
    }
    return 0;
}
double unop_apply(UnOp op, double a, double alpha = 0.0) {
    switch (op) {
        case UnOp::NEG: return -a;
        case UnOp::SQRT: return std::sqrt(a);
        case UnOp::SIN: return std::sin(a);
        case UnOp::COS: return std::cos(a);
        case UnOp::EXP: return std::exp(a);
        case UnOp::TANH: return std::tanh(a);
        case UnOp::SIGMOID: return 1.0 / (1.0 + std::exp(-a));
        case UnOp::RELU: return a > 0 ? a : 0;
        case UnOp::ABS: return std::fabs(a);
        case UnOp::LEAKY: return a > 0 ? a : alpha * a;
        case UnOp::ERF: return std::erf(a);
        case UnOp::GELU: return 0.5 * a * (1.0 + std::erf(a * 0.70710678118654752440));
        case UnOp::GELUT: return 0.5 * a * (1.0 + std::tanh(0.79788456080286535588 * (a + 0.044715 * a * a * a)));
        case UnOp::SOFTPLUS: {
            // alpha carries beta (default 1); the 20 threshold
            // is the stable form ORT and PyTorch both use.
            double bx = alpha * a;
            return bx > 20.0 ? a : std::log1p(std::exp(bx)) / alpha;
        }
        case UnOp::LOG: return std::log(a);
    }
    return 0;
}

float binop_apply_f(BinOp op, float a, float b);  // fwd, defined by k_binop
float unop_apply_f(UnOp op, float a, double alpha);  // fwd

// f32 flat fast path for binary ops when both operands are contiguous
// and either same-shape or scalar.
bool binop_f32_fast(BinOp op, const RT& A, const RT& B, RT& Y) {
    if (A.dtype != 1 || B.dtype != 1 || Y.dtype != 1) return false;
    if (!A.contig() || !B.contig() || !Y.contig()) return false;
    const float* a = (const float*)A.p;
    const float* b = (const float*)B.p;
    float* y = (float*)Y.p;
    int64_t n = Y.numel();
    bool as = A.numel() == n, bs = B.numel() == n;
    if (!as && A.numel() != 1) return false;
    if (!bs && B.numel() != 1) return false;
    auto range = [&](int64_t i0, int64_t i1) {
        if (cpu::features().avx2 && cpu::features().fma &&
            avx2k::binop_f32((int)op, as ? a + i0 : a, bs ? b + i0 : b, y + i0, i1 - i0, !as, !bs))
            return;
        for (int64_t i = i0; i < i1; i++) y[i] = binop_apply_f(op, a[as ? i : 0], b[bs ? i : 0]);
    };
    if (n >= (1 << 17)) {
        packed::kernel_pool().run([&](int idx, int nt) {
            packed::chunk(idx, nt, n, [&](int64_t i0, int64_t i1) { range(i0, i1); });
        });
    } else {
        range(0, n);
    }
    return true;
}

// Aligned strides of an operand against the output shape; broadcast
// dims get stride 0. False when rank exceeds the walker limit.
bool aligned_strides(const RT& t, const std::vector<int64_t>& oshape, int64_t* out) {
    int r = (int)oshape.size();
    if (r > 8) return false;
    int shift = r - (int)t.shape.size();
    for (int d = 0; d < r; d++) {
        int td = d - shift;
        out[d] = td < 0 ? 0 : (t.shape[td] == 1 ? 0 : t.strides[td]);
    }
    return true;
}

// Drop size-1 dims from an aligned shape/stride set: their coordinate
// is always 0, so the compacted walk visits exactly the same elements
// in the same order. This turns layouts like [C, L, 1] (a plain
// [C, L] tensor with a trailing degenerate dim, the common vits
// per-channel scale/bias shape) back into rows the fast path takes.
static int compact_dims(const std::vector<int64_t>& oshape, const int64_t* as, const int64_t* bs,
                        const int64_t* ys, int64_t* ca, int64_t* cb, int64_t* cy, int64_t* cs) {
    int r = 0;
    for (size_t d = 0; d < oshape.size(); d++) {
        if (oshape[d] == 1 && oshape.size() > 1) continue;
        cs[r] = oshape[d];
        ca[r] = as[d];
        if (cb) cb[r] = bs[d];
        cy[r] = ys[d];
        r++;
    }
    return r;
}

// Pool-parallel strided binary walk (f32): each thread decomposes its
// start index once, then walks incrementally.
void binop_walk_par(BinOp op, const RT& A, const RT& B, RT& Y, const int64_t* as, const int64_t* bs, const int64_t* ys,
                    int r, const int64_t* oshape) {
    int64_t suf[8];
    suf[r - 1] = 1;
    for (int d = r - 2; d >= 0; d--) suf[d] = suf[d + 1] * oshape[d + 1];
    int64_t n = Y.numel();
    const float* a = (const float*)A.p;
    const float* b = (const float*)B.p;
    float* y = (float*)Y.p;
    packed::pool_run(n * 16, [&](int idx, int nt) {
        packed::chunk(idx, nt, n, [&](int64_t i0, int64_t i1) {
            int64_t coord[8] = {0};
            int64_t ao = 0, bo = 0, yo = 0;
            int64_t rem = i0;
            for (int d = 0; d < r; d++) {
                coord[d] = rem / suf[d];
                rem -= coord[d] * suf[d];
                ao += coord[d] * as[d]; bo += coord[d] * bs[d]; yo += coord[d] * ys[d];
            }
            for (int64_t i = i0; i < i1; i++) {
                y[yo] = binop_apply_f(op, a[ao], b[bo]);
                for (int d = r - 1; d >= 0; d--) {
                    ao += as[d]; bo += bs[d]; yo += ys[d];
                    if (++coord[d] < oshape[d]) break;
                    coord[d] = 0;
                    ao -= as[d] * oshape[d]; bo -= bs[d] * oshape[d]; yo -= ys[d] * oshape[d];
                }
            }
        });
    });
}

// Float form of binop_apply for the ops whose float evaluation is the
// correctly-rounded one (ORT evaluates these in float too). POW and
// anything unlisted fall back to the double form.
float binop_apply_f(BinOp op, float a, float b) {
    switch (op) {
        case BinOp::ADD: return a + b;
        case BinOp::SUB: return a - b;
        case BinOp::MUL: return a * b;
        case BinOp::DIV: return a / b;
        case BinOp::MAXIMUM: return std::max(a, b);
        case BinOp::LESS: return a < b ? 1.0f : 0.0f;
        case BinOp::GREATER: return a > b ? 1.0f : 0.0f;
        case BinOp::EQUAL: return a == b ? 1.0f : 0.0f;
        case BinOp::OR: return (a != 0 || b != 0) ? 1.0f : 0.0f;
        case BinOp::AND: return (a != 0 && b != 0) ? 1.0f : 0.0f;
        default: return (float)binop_apply(op, a, b);
    }
}
float unop_apply_f(UnOp op, float a, double alpha) {
    switch (op) {
        case UnOp::NEG: return -a;
        case UnOp::SQRT: return std::sqrt(a);
        case UnOp::SIN: return std::sin(a);
        case UnOp::COS: return std::cos(a);
        case UnOp::EXP: return std::exp(a);
        case UnOp::TANH: return std::tanh(a);
        case UnOp::SIGMOID: return 1.0f / (1.0f + std::exp(-a));
        case UnOp::RELU: return a > 0 ? a : 0.0f;
        case UnOp::ABS: return std::fabs(a);
        case UnOp::LEAKY: return a > 0 ? a : (float)(alpha * a);
        case UnOp::ERF: return std::erf(a);
        case UnOp::GELU: return 0.5f * a * (1.0f + std::erf(a * 0.70710678118654752440f));
        case UnOp::GELUT: return 0.5f * a * (1.0f + std::tanh(0.79788456080286535588f * (a + 0.044715f * a * a * a)));
        case UnOp::SOFTPLUS: {
            double bx = alpha * (double)a;
            return bx > 20.0 ? a : (float)(std::log1p(std::exp(bx)) / alpha);
        }
        case UnOp::LOG: return std::log(a);
    }
    return 0;
}

// Row-organized elementwise walk (f32): when the output's last dim is
// contiguous and each operand's last-dim stride is 0 (broadcast) or 1,
// each output row is one flat loop the compiler can vectorize, instead
// of a per-element odometer walk. Outer offsets are decomposed once
// per row. Returns false when the layout does not qualify.
bool binop_rows(BinOp op, const RT& A, const RT& B, RT& Y, const int64_t* as, const int64_t* bs, const int64_t* ys,
                int r, const int64_t* oshape) {
    if (r < 1 || ys[r - 1] != 1) return false;
    if ((as[r - 1] != 0 && as[r - 1] != 1) || (bs[r - 1] != 0 && bs[r - 1] != 1)) return false;
    int64_t D = oshape[r - 1];
    int64_t rows = Y.numel() / D;
    int64_t suf[8];
    if (r >= 2) {
        suf[r - 2] = 1;
        for (int d = r - 3; d >= 0; d--) suf[d] = suf[d + 1] * oshape[d + 1];
    }
    const float* a = (const float*)A.p;
    const float* b = (const float*)B.p;
    float* y = (float*)Y.p;
    int64_t asl = as[r - 1], bsl = bs[r - 1];
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            for (int64_t rr = r0; rr < r1; rr++) {
                int64_t ao = 0, bo = 0, yo = 0, rem = rr;
                for (int d = 0; d < r - 1; d++) {
                    int64_t c = rem / suf[d];
                    rem -= c * suf[d];
                    ao += c * as[d]; bo += c * bs[d]; yo += c * ys[d];
                }
                const float* ap = a + ao;
                const float* bp = b + bo;
                float* yp = y + yo;
                if (asl == 1 && bsl == 1) {
                    if (!(cpu::features().avx2 && cpu::features().fma &&
                          avx2k::binop_f32((int)op, ap, bp, yp, D, false, false)))
                        for (int64_t i = 0; i < D; i++) yp[i] = binop_apply_f(op, ap[i], bp[i]);
                } else if (asl == 1) {
                    if (!(cpu::features().avx2 && cpu::features().fma &&
                          avx2k::binop_f32((int)op, ap, bp, yp, D, false, true))) {
                        float bv = bp[0];
                        for (int64_t i = 0; i < D; i++) yp[i] = binop_apply_f(op, ap[i], bv);
                    }
                } else if (bsl == 1) {
                    if (!(cpu::features().avx2 && cpu::features().fma &&
                          avx2k::binop_f32((int)op, ap, bp, yp, D, true, false))) {
                        float av = ap[0];
                        for (int64_t i = 0; i < D; i++) yp[i] = binop_apply_f(op, av, bp[i]);
                    }
                } else {
                    float v = binop_apply_f(op, ap[0], bp[0]);
                    for (int64_t i = 0; i < D; i++) yp[i] = v;
                }
            }
        });
    };
    if (Y.numel() >= (1 << 17)) packed::kernel_pool().run(body);
    else body(0, 1);
    return true;
}
bool unop_rows(UnOp op, const RT& A, RT& Y, double alpha, const int64_t* as, const int64_t* ys,
               int r, const int64_t* oshape) {
    if (r < 1 || ys[r - 1] != 1) return false;
    if (as[r - 1] != 0 && as[r - 1] != 1) return false;
    int64_t D = oshape[r - 1];
    int64_t rows = Y.numel() / D;
    int64_t suf[8];
    if (r >= 2) {
        suf[r - 2] = 1;
        for (int d = r - 3; d >= 0; d--) suf[d] = suf[d + 1] * oshape[d + 1];
    }
    const float* a = (const float*)A.p;
    float* y = (float*)Y.p;
    int64_t asl = as[r - 1];
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            for (int64_t rr = r0; rr < r1; rr++) {
                int64_t ao = 0, yo = 0, rem = rr;
                for (int d = 0; d < r - 1; d++) {
                    int64_t c = rem / suf[d];
                    rem -= c * suf[d];
                    ao += c * as[d]; yo += c * ys[d];
                }
                const float* ap = a + ao;
                float* yp = y + yo;
                if (asl == 1) {
                    if (!(cpu::features().avx2 && cpu::features().fma &&
                          avx2k::unop_f32((int)op, ap, yp, D, (float)alpha)))
                        for (int64_t i = 0; i < D; i++) yp[i] = unop_apply_f(op, ap[i], alpha);
                } else {
                    float v = unop_apply_f(op, ap[0], alpha);
                    for (int64_t i = 0; i < D; i++) yp[i] = v;
                }
            }
        });
    };
    if (Y.numel() >= (1 << 17)) packed::kernel_pool().run(body);
    else body(0, 1);
    return true;
}

// BINOP path histogram (diagnostics under FE_PROFILE): which k_binop
// path serves each call, with time, elements and the slowest single
// call per path. Printed by prof_print alongside the per-kind table.
// Kept from the 2026-11-10 knife-5 diagnosis (ORT-FREE.md section
// 12): walk_par is where decode's elementwise time lives.
struct BinopPathStat { long n = 0; double us = 0; long long elems = 0; double max_us = 0; };
static BinopPathStat g_binop_paths[6];
static const char* g_binop_path_names[6] = {"f32_fast", "rows", "walk_par", "serial_odom", "generic_odom", "axis_rows"};
static bool fe_profile_on() { static const bool on = std::getenv("FE_PROFILE") != nullptr; return on; }

// Generalized rows for the case binop_rows misses only because the
// flat axis is not the last one: walk all but one axis as an outer
// odometer, run the AVX2 kernel on the chosen axis. Selection is
// restricted to the vectorizable case (see the note inside).
bool binop_axis_rows(BinOp op, const RT& A, const RT& B, RT& Y, const int64_t* as, const int64_t* bs,
                     const int64_t* ys, int r, const int64_t* oshape) {
    // Only the fully vectorizable case is taken: unit output stride
    // and unit-or-broadcast operands on the chosen axis, so the
    // inner loop delegates to the AVX2 kernels. Strided forms were
    // measured SLOWER than the linear odometer walk in an
    // interleaved A/B on the attention pattern (axis 362-378 ms vs
    // walk 291-300 ms per request): the walk's Y-linear order keeps
    // reusing the operand's cache window, scattered access in
    // either direction loses to it.
    int d = -1;
    for (int i = 0; i < r; i++) {
        if (ys[i] != 1 || oshape[i] < 2) continue;
        if (!((as[i] == 0 || as[i] == 1) && (bs[i] == 0 || bs[i] == 1))) continue;
        if (as[i] != 1 && bs[i] != 1) continue;
        if (d < 0 || oshape[i] > oshape[d]) d = i;
    }
    if (d < 0) return false;
    const int64_t D = oshape[d];
    const int64_t outer = Y.numel() / D;
    int64_t suf[8];
    {
        int64_t acc = 1;
        for (int i = r - 1; i >= 0; i--) {
            if (i == d) continue;
            suf[i] = acc;
            acc *= oshape[i];
        }
    }
    const float* a = (const float*)A.p;
    const float* b = (const float*)B.p;
    float* y = (float*)Y.p;
    const int64_t sa = as[d], sb = bs[d], sy = ys[d];
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, outer, [&](int64_t o0, int64_t o1) {
            for (int64_t oo = o0; oo < o1; oo++) {
                int64_t ao = 0, bo = 0, yo = 0, rem = oo;
                for (int i = 0; i < r; i++) {
                    if (i == d) continue;
                    int64_t c = rem / suf[i];
                    rem -= c * suf[i];
                    ao += c * as[i]; bo += c * bs[i]; yo += c * ys[i];
                }
                const float* ap = a + ao;
                const float* bp = b + bo;
                float* yp = y + yo;
                if (sy == 1 && (sa == 0 || sa == 1) && (sb == 0 || sb == 1) &&
                    cpu::features().avx2 && cpu::features().fma &&
                    avx2k::binop_f32((int)op, ap, bp, yp, D, sa == 0, sb == 0))
                    continue;
                for (int64_t i = 0; i < D; i++) yp[i * sy] = binop_apply_f(op, ap[i * sa], bp[i * sb]);
            }
        });
    };
    if (Y.numel() >= (1 << 17)) packed::kernel_pool().run(body);
    else body(0, 1);
    return true;
}

void k_binop(BinOp op, const RT& A, const RT& B, RT& Y) {
    const bool prof = fe_profile_on();
    const auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto note = [&](int path) {
        if (!prof) return;
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        BinopPathStat& s = g_binop_paths[path];
        s.n++;
        s.us += us;
        s.elems += Y.numel();
        if (us > s.max_us) s.max_us = us;
    };
    if (binop_f32_fast(op, A, B, Y)) { note(0); return; }
    if (A.dtype == 1 && B.dtype == 1 && Y.dtype == 1) {
        int64_t as[8], bs[8], ys[8];
        if (aligned_strides(A, Y.shape, as) && aligned_strides(B, Y.shape, bs) && aligned_strides(Y, Y.shape, ys)) {
            int64_t ca[8], cb[8], cy[8], cs[8];
            int cr = compact_dims(Y.shape, as, bs, ys, ca, cb, cy, cs);
            if (binop_rows(op, A, B, Y, ca, cb, cy, cr, cs)) { note(1); return; }
            if (binop_axis_rows(op, A, B, Y, ca, cb, cy, cr, cs)) { note(5); return; }
            if (Y.numel() >= 8192) {
                if (std::getenv("FE_WALK_DEBUG")) {
                    int r = (int)Y.shape.size();
                    std::fprintf(stderr, "[WALK] op=%d rank=%d shape=", (int)op, r);
                    for (int d = 0; d < r; d++) std::fprintf(stderr, "%s%lld", d ? "x" : "", (long long)Y.shape[d]);
                    std::fprintf(stderr, " as=[");
                    for (int d = 0; d < r; d++) std::fprintf(stderr, "%s%lld", d ? "," : "", (long long)as[d]);
                    std::fprintf(stderr, "] bs=[");
                    for (int d = 0; d < r; d++) std::fprintf(stderr, "%s%lld", d ? "," : "", (long long)bs[d]);
                    std::fprintf(stderr, "] ys=[");
                    for (int d = 0; d < r; d++) std::fprintf(stderr, "%s%lld", d ? "," : "", (long long)ys[d]);
                    std::fprintf(stderr, "] n=%lld\n", (long long)Y.numel());
                }
                binop_walk_par(op, A, B, Y, ca, cb, cy, cr, cs); note(2); return;
            }
            int r = (int)Y.shape.size();
            int64_t coord[8] = {0};
            int64_t ao = 0, bo = 0, yo = 0;
            const float* a = (const float*)A.p;
            const float* b = (const float*)B.p;
            float* y = (float*)Y.p;
            int64_t n = Y.numel();
            for (int64_t i = 0; i < n; i++) {
                y[yo] = (float)binop_apply(op, a[ao], b[bo]);
                for (int d = r - 1; d >= 0; d--) {
                    ao += as[d]; bo += bs[d]; yo += ys[d];
                    if (++coord[d] < Y.shape[d]) break;
                    coord[d] = 0;
                    ao -= as[d] * Y.shape[d]; bo -= bs[d] * Y.shape[d]; yo -= ys[d] * Y.shape[d];
                }
            }
            note(3);
            return;
        }
    }
    auto aa = align_of(A, Y.shape.size());
    auto ab = align_of(B, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        double va = read_d(A, off_of(A, c, aa));
        double vb = read_d(B, off_of(B, c, ab));
        write_d(Y, off_of(Y, c, ay), binop_apply(op, va, vb));
    });
    note(4);
}

void k_unop(UnOp op, const RT& A, RT& Y, double alpha = 0.0) {
    if (A.dtype == 1 && A.contig() && Y.contig() && A.numel() == Y.numel()) {
        const float* a = (const float*)A.p;
        float* y = (float*)Y.p;
        int64_t n = Y.numel();
        auto range = [&](int64_t i0, int64_t i1) {
            if (cpu::features().avx2 && cpu::features().fma &&
                avx2k::unop_f32((int)op, a + i0, y + i0, i1 - i0, (float)alpha))
                return;
            for (int64_t i = i0; i < i1; i++) y[i] = unop_apply_f(op, a[i], alpha);
        };
        if (n >= (1 << 17)) {
            packed::kernel_pool().run([&](int idx, int nt) {
                packed::chunk(idx, nt, n, [&](int64_t i0, int64_t i1) { range(i0, i1); });
            });
        } else {
            range(0, n);
        }
        return;
    }
    if (A.dtype == 1 && Y.dtype == 1) {
        int64_t as[8], ys[8];
        if (aligned_strides(A, Y.shape, as) && aligned_strides(Y, Y.shape, ys)) {
            int64_t ca[8], cy[8], cs[8];
            int cr = compact_dims(Y.shape, as, nullptr, ys, ca, nullptr, cy, cs);
            if (unop_rows(op, A, Y, alpha, ca, cy, cr, cs)) return;
            int r = (int)Y.shape.size();
            int64_t coord[8] = {0};
            int64_t ao = 0, yo = 0;
            const float* a = (const float*)A.p;
            float* y = (float*)Y.p;
            int64_t n = Y.numel();
            for (int64_t i = 0; i < n; i++) {
                y[yo] = (float)unop_apply(op, a[ao], alpha);
                for (int d = r - 1; d >= 0; d--) {
                    ao += as[d]; yo += ys[d];
                    if (++coord[d] < Y.shape[d]) break;
                    coord[d] = 0;
                    ao -= as[d] * Y.shape[d]; yo -= ys[d] * Y.shape[d];
                }
            }
            return;
        }
    }
    auto aa = align_of(A, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        write_d(Y, off_of(Y, c, ay), unop_apply(op, read_d(A, off_of(A, c, aa)), alpha));
    });
}

template <typename InT, typename OutT>
void cast_loop(const RT& A, RT& Y) {
    auto aa = align_of(A, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        InT v;
        memcpy(&v, A.p + off_of(A, c, aa) * (int64_t)sizeof(InT), sizeof(InT));
        OutT o = (OutT)v;
        memcpy(Y.p + off_of(Y, c, ay) * (int64_t)sizeof(OutT), &o, sizeof(OutT));
    });
}
void k_cast(const RT& A, RT& Y) {
    if (A.dtype == Y.dtype) { copy_tensor(Y, A); return; }
    auto run = [&](auto in_tag, auto out_tag) {
        cast_loop<decltype(in_tag), decltype(out_tag)>(A, Y);
    };
    // Dispatch through read_d/write_d for the long tail of type pairs;
    // sizes here are small (metadata casts, position tensors).
    (void)run;
    auto aa = align_of(A, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        write_d(Y, off_of(Y, c, ay), read_d(A, off_of(A, c, aa)));
    });
}

void k_where(const RT& C, const RT& A, const RT& B, RT& Y) {
    auto ac = align_of(C, Y.shape.size());
    auto aa = align_of(A, Y.shape.size());
    auto ab = align_of(B, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& cd) {
        bool sel = read_d(C, off_of(C, cd, ac)) != 0;
        write_d(Y, off_of(Y, cd, ay), sel ? read_d(A, off_of(A, cd, aa)) : read_d(B, off_of(B, cd, ab)));
    });
}

void k_reducesum(const RT& X, RT& Y, const std::vector<int64_t>& axes, bool keepdims) {
    int rank = (int)X.shape.size();
    std::vector<char> red(rank, 0);
    for (auto a : axes) red[a] = 1;
    // Map each X dim to its Y dim (keepdims: same index; else: index
    // among the non-reduced dims), or -1 when reduced away.
    std::vector<int64_t> ydim(rank, -1);
    int64_t yn = 0;
    for (int d = 0; d < rank; d++) {
        if (keepdims) ydim[d] = d;
        else if (!red[d]) ydim[d] = yn++;
    }
    memset(Y.p, 0, (size_t)Y.numel() * onnxm::elem_size(Y.dtype));
    odometer(X.shape, [&](const std::vector<int64_t>& c) {
        int64_t xo = 0, yo = 0;
        for (int d = 0; d < rank; d++) {
            xo += c[d] * X.strides[d];
            if (!red[d]) yo += c[d] * Y.strides[ydim[d]];
            else if (keepdims) yo += 0;  // coord contributes 0 on size-1 dim
        }
        write_d(Y, yo, read_d(Y, yo) + read_d(X, xo));
    });
}

void k_reducel2(const RT& X, RT& Y, const std::vector<int64_t>& axes, bool keepdims) {
    int rank = (int)X.shape.size();
    std::vector<char> red(rank, 0);
    for (auto a : axes) red[a] = 1;
    std::vector<int64_t> ydim(rank, -1);
    int64_t yn = 0;
    for (int d = 0; d < rank; d++) {
        if (keepdims) ydim[d] = d;
        else if (!red[d]) ydim[d] = yn++;
    }
    memset(Y.p, 0, (size_t)Y.numel() * onnxm::elem_size(Y.dtype));
    odometer(X.shape, [&](const std::vector<int64_t>& c) {
        int64_t xo = 0, yo = 0;
        for (int d = 0; d < rank; d++) {
            xo += c[d] * X.strides[d];
            if (!red[d]) yo += c[d] * Y.strides[ydim[d]];
        }
        double v = read_d(X, xo);
        write_d(Y, yo, read_d(Y, yo) + v * v);
    });
    int64_t n = Y.numel();
    for (int64_t i = 0; i < n; i++) write_d(Y, i, std::sqrt(read_d(Y, i)));
}

void k_argmax(const RT& X, RT& Y, int64_t axis, bool keepdims, bool last) {
    (void)keepdims;
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= X.shape[i];
    int64_t s_axis = X.strides[axis];
    int64_t s_outer = axis > 0 ? X.strides[axis - 1] : X.numel();
    (void)s_outer;
    // Lines are addressed via (outer, inner) decomposition over strides.
    std::vector<int64_t> oc_shape = X.shape;
    oc_shape[axis] = 1;
    int64_t lines = outer * inner;
    for (int64_t l = 0; l < lines; l++) {
        int64_t o = l / inner, in = l % inner;
        // base offset: walk outer dims and inner dims through strides
        int64_t base = 0;
        {
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
        }
        double best = -INFINITY;
        int64_t bi = 0;
        for (int64_t i = 0; i < D; i++) {
            double v = read_d(X, base + i * s_axis);
            if (v > best || (last && v == best)) { best = v; bi = i; }
        }
        int64_t yo = 0;
        {
            int64_t rem = o;
            std::vector<int64_t> yshape = Y.shape;
            // Y rank == X rank (keepdims) or rank-1 less; map via oc walk:
            std::vector<int64_t> coord(rank, 0);
            for (int d = (int)axis - 1; d >= 0; d--) {
                coord[d] = X.shape[d] ? (rem % X.shape[d]) : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                coord[d] = X.shape[d] ? (rem % X.shape[d]) : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
            }
            coord[axis] = 0;
            // Convert coord in X-rank space to Y offset.
            if ((int)Y.shape.size() == rank) {
                for (int d = 0; d < rank; d++) yo += coord[d] * Y.strides[d];
            } else {
                for (int d = 0; d < rank; d++) {
                    if (d == (int)axis) continue;
                    int yd = d < (int)axis ? d : d - 1;
                    yo += coord[d] * Y.strides[yd];
                }
            }
            (void)yshape;
        }
        int64_t v = bi;
        memcpy(Y.p + yo * 8, &v, 8);
    }
}

void k_topk(const RT& X, RT& Yv, RT& Yi, int64_t axis, int64_t k, bool largest, bool sorted) {
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= X.shape[i];
    int64_t lines = outer * inner;
    std::vector<std::pair<double, int64_t>> buf(D);
    for (int64_t l = 0; l < lines; l++) {
        int64_t o = l / inner, in = l % inner;
        int64_t base = 0;
        {
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
        }
        for (int64_t i = 0; i < D; i++) buf[i] = {read_d(X, base + i * X.strides[axis]), i};
        std::partial_sort(buf.begin(), buf.begin() + k, buf.end(),
                          [&](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                              return largest ? a.first > b.first : a.first < b.first;
                          });
        if (!sorted) std::sort(buf.begin(), buf.begin() + k, [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) { return a.second < b.second; });
        // Output line offset: same (o, in) decomposition on Y shapes.
        auto line_off = [&](const RT& T) {
            int64_t off = 0;
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                off += c * T.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                off += c * T.strides[d];
            }
            return off;
        };
        int64_t vo = line_off(Yv), io = line_off(Yi);
        for (int64_t i = 0; i < k; i++) {
            write_d(Yv, vo + i * Yv.strides[axis], buf[i].first);
            int64_t id = buf[i].second;
            memcpy(Yi.p + (io + i * Yi.strides[axis]) * 8, &id, 8);
        }
    }
}

void k_cumsum(const RT& X, RT& Y, int64_t axis, bool exclusive, bool reverse) {
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= X.shape[i];
    int64_t lines = outer * inner;
    for (int64_t l = 0; l < lines; l++) {
        int64_t o = l / inner, in = l % inner;
        int64_t base = 0;
        {
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
        }
        double acc = 0;
        for (int64_t ii = 0; ii < D; ii++) {
            int64_t i = reverse ? D - 1 - ii : ii;
            double v = read_d(X, base + i * X.strides[axis]);
            if (exclusive) {
                write_d(Y, base + i * Y.strides[axis], acc);
                acc += v;
            } else {
                acc += v;
                write_d(Y, base + i * Y.strides[axis], acc);
            }
        }
    }
}

void k_gather(const RT& D, const RT& I, RT& Y, int64_t axis) {
    int rank = (int)D.shape.size();
    if (axis < 0) axis += rank;
    size_t es = onnxm::elem_size(D.dtype);
    int64_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= D.shape[i];
    int64_t inner = 1;
    for (int i = axis + 1; i < rank; i++) inner *= D.shape[i];
    int64_t icount = I.numel();
    // Fast path: data contiguous, output contiguous, inner block memcpy.
    if (D.contig() && Y.contig()) {
        int64_t d_axis_stride = inner;
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t ii = 0; ii < icount; ii++) {
                int64_t idx = (int64_t)read_d(I, ii);
                if (idx < 0) idx += D.shape[axis];
                const uint8_t* s = D.p + ((o * D.shape[axis] + idx) * inner) * (int64_t)es;
                uint8_t* d = Y.p + ((o * icount + ii) * inner) * (int64_t)es;
                memcpy(d, s, (size_t)inner * es);
            }
        }
        return;
    }
    auto ya = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        // Y coords: [data outer dims..., indices dims..., data inner dims...]
        std::vector<int64_t> dc(D.shape.size(), 0), ic(I.shape.size(), 0);
        for (int i = 0; i < axis; i++) dc[i] = c[i];
        for (size_t i = 0; i < I.shape.size(); i++) ic[i] = c[axis + i];
        for (int i = axis + 1; i < rank; i++) dc[i] = c[axis + (int64_t)I.shape.size() + (i - axis - 1)];
        int64_t io = 0;
        for (size_t i = 0; i < I.shape.size(); i++) io += ic[i] * I.strides[i];
        int64_t idx = (int64_t)read_d(I, io);
        if (idx < 0) idx += D.shape[axis];
        dc[axis] = idx;
        int64_t so = 0;
        for (int i = 0; i < rank; i++) so += dc[i] * D.strides[i];
        int64_t dof = off_of(Y, c, ya);
        memcpy(Y.p + dof * (int64_t)es, D.p + so * (int64_t)es, es);
    });
}

void k_gather_elements(const RT& D, const RT& I, RT& Y, int64_t axis) {
    int rank = (int)D.shape.size();
    if (axis < 0) axis += rank;
    size_t es = onnxm::elem_size(D.dtype);
    auto ia = align_of(I, Y.shape.size());
    auto ya = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        int64_t idx = (int64_t)read_d(I, off_of(I, c, ia));
        if (idx < 0) idx += D.shape[axis];
        int64_t so = 0;
        for (int i = 0; i < rank; i++) so += (i == (int)axis ? idx : c[i]) * D.strides[i];
        memcpy(Y.p + off_of(Y, c, ya) * (int64_t)es, D.p + so * (int64_t)es, es);
    });
}

void k_scatter_elements(const RT& D, const RT& I, const RT& U, RT& Y, int64_t axis, int reduction) {
    copy_tensor(Y, D);
    int rank = (int)D.shape.size();
    if (axis < 0) axis += rank;
    auto ia = align_of(I, I.shape.size());
    auto ua = align_of(U, I.shape.size());
    odometer(I.shape, [&](const std::vector<int64_t>& c) {
        int64_t idx = (int64_t)read_d(I, off_of(I, c, ia));
        if (idx < 0) idx += D.shape[axis];
        int64_t yo = 0;
        for (int i = 0; i < rank; i++) yo += (i == (int)axis ? idx : c[i]) * Y.strides[i];
        double v = read_d(U, off_of(U, c, ua));
        if (reduction == 1) write_d(Y, yo, read_d(Y, yo) + v);        // add
        else if (reduction == 2) write_d(Y, yo, read_d(Y, yo) * v);   // mul
        else write_d(Y, yo, v);                                       // none
    });
}

void k_concat(const std::vector<const RT*>& ins, RT& Y, int64_t axis, Scratch& sc) {
    int rank = (int)Y.shape.size();
    if (axis < 0) axis += rank;
    int64_t off = 0;
    for (auto* ip : ins) {
        RT sub;
        sub.dtype = Y.dtype;
        sub.p = Y.p;
        sub.shape = ip->shape;
        sub.strides = Y.strides;
        // Offset along axis in elements of Y's layout.
        sub.p = Y.p + off * Y.strides[axis] * (int64_t)onnxm::elem_size(Y.dtype);
        RT in = sc.contig_of(*ip);
        copy_tensor(sub, in);
        off += ip->shape[axis];
    }
}

void k_conv_g1(const RT& X0, const RT& W0, const RT* B, RT& Y, const std::vector<int64_t>& strides_a,
            const std::vector<int64_t>& pads, const std::vector<int64_t>& dil, Scratch& sc) {
    RT X = sc.contig_of(X0), W = sc.contig_of(W0);
    RT Bc;
    const float* bias = nullptr;
    if (B) { Bc = sc.contig_of(*B); bias = (const float*)Bc.p; }
    int sp = (int)X.shape.size() - 2;
    if (sp < 1 || sp > 2) fail("Conv: spatial rank not 1..2");
    int64_t N = X.shape[0], C = X.shape[1], O = W.shape[0];
    const float* x = (const float*)X.p;
    const float* w = (const float*)W.p;
    float* y = (float*)Y.p;
    if (sp == 1) {
        // Direct conv1d, SAXPY-over-l form (same loop shape as the
        // packed conv1d_fast): one output line (n, o) per task; for
        // each (c, k) the valid l range is clamped once and the inner
        // loop is a contiguous (stride 1) or regularly strided SAXPY.
        int64_t L = X.shape[2], K = W.shape[2], Lout = Y.shape[2];
        int64_t stride = strides_a[0], dilv = dil[0], pl = pads[0];
        const bool use_avx2 = cpu::features().avx2 && cpu::features().fma;
        const bool use_avx512 = use_avx2 && packed::conv_tile_has_avx512();
        if (stride == 1 && (K - 1) * dilv <= 64 && use_avx2) {
            // Register-blocked tile (4 output channels x 32 positions,
            // shared with the packed path): the SAXPY form below
            // re-reads and re-writes every output row once per
            // (input channel, tap); the tile keeps outputs in
            // registers over the whole contraction.
            const int64_t OB4 = 4, LT = 1024;
            const int64_t nblocks = (O + OB4 - 1) / OB4;
            const int64_t ntiles = (Lout + LT - 1) / LT;
            // Dynamic task fetch, same reasoning as conv1d_fast: on
            // shared vCPUs a static partition makes the node as slow
            // as its most stolen worker (measured +83% on the EPYC
            // box for this branch's first, static, version).
            std::atomic<int64_t> next_task{0};
            const int64_t total_tasks = N * ntiles * nblocks;
            packed::pool_run(N * O * Lout * C * K, [&](int, int) {
                while (true) {
                    int64_t task = next_task.fetch_add(1, std::memory_order_relaxed);
                    if (task >= total_tasks) break;
                    int64_t blk = task % nblocks;
                    int64_t tile = (task / nblocks) % ntiles;
                    int64_t n = task / (nblocks * ntiles);
                    int64_t o0 = blk * OB4, ob = std::min(OB4, O - o0);
                    int64_t lt0 = tile * LT, lt_len = std::min(LT, Lout - lt0);
                    if (use_avx512)
                        packed::conv_tile_avx512(x + (size_t)n * C * L, L, w, C * K, C, K, pl, dilv,
                                                 bias, y + (size_t)n * O * Lout, Lout, lt0, lt_len,
                                                 o0, ob);
                    else
                        packed::conv_tile_avx2(x + (size_t)n * C * L, L, w, C * K, C, K, pl, dilv,
                                               bias, y + (size_t)n * O * Lout, Lout, lt0, lt_len,
                                               o0, ob);
                }
            });
            return;
        }
        packed::pool_run(N * O * Lout * C * K, [&](int idx, int nt) {
            packed::chunk(idx, nt, N * O, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; t++) {
                    int64_t n = t / O, o = t % O;
                    float* yl = y + (size_t)(n * O + o) * Lout;
                    float bv = bias ? bias[o] : 0.0f;
                    for (int64_t l = 0; l < Lout; l++) yl[l] = bv;
                    for (int64_t c = 0; c < C; c++) {
                        const float* xr = x + (size_t)(n * C + c) * L;
                        const float* wr = w + (size_t)(o * C + c) * K;
                        for (int64_t k = 0; k < K; k++) {
                            float wv = wr[k];
                            if (wv == 0.0f) continue;
                            int64_t base = k * dilv - pl;
                            int64_t l_lo = 0, l_hi = Lout;
                            if (base < 0) l_lo = (-base + stride - 1) / stride;
                            if (base > L - 1) continue;
                            l_hi = std::min(Lout, (L - 1 - base) / stride + 1);
                            if (l_lo >= l_hi) continue;
                            if (stride == 1) {
                                const float* xs = xr + l_lo + base;
                                if (use_avx2) avx2k::saxpy(wv, xs, yl + l_lo, l_hi - l_lo);
                                else for (int64_t l = l_lo; l < l_hi; l++) yl[l] += wv * xs[l - l_lo];
                            } else {
                                for (int64_t l = l_lo; l < l_hi; l++) yl[l] += wv * xr[l * stride + base];
                            }
                        }
                    }
                }
            });
        });
        return;
    }
    // sp == 2: direct dot-product form, one output line (n, o) per task.
    int64_t H = X.shape[2], Wd = X.shape[3], KH = W.shape[2], KW = W.shape[3];
    int64_t Hout = Y.shape[2], Wout = Y.shape[3];
    packed::pool_run(N * O * Hout * Wout * C * KH * KW, [&](int idx, int nt) {
        packed::chunk(idx, nt, N * O, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; t++) {
                int64_t n = t / O, o = t % O;
                float* yl = y + (size_t)(n * O + o) * Hout * Wout;
                for (int64_t h = 0; h < Hout; h++) {
                    for (int64_t wd = 0; wd < Wout; wd++) {
                        double acc = bias ? bias[o] : 0.0;
                        for (int64_t c = 0; c < C; c++) {
                            const float* xr = x + (size_t)(n * C + c) * H * Wd;
                            const float* wr = w + (size_t)(o * C + c) * KH * KW;
                            for (int64_t kh = 0; kh < KH; kh++) {
                                int64_t ph = h * strides_a[0] + kh * dil[0] - pads[0];
                                if (ph < 0 || ph >= H) continue;
                                for (int64_t kw = 0; kw < KW; kw++) {
                                    int64_t pw = wd * strides_a[1] + kw * dil[1] - pads[1];
                                    if (pw < 0 || pw >= Wd) continue;
                                    acc += (double)xr[ph * Wd + pw] * wr[kh * KW + kw];
                                }
                            }
                        }
                        yl[h * Wout + wd] = (float)acc;
                    }
                }
            }
        });
    });
}

// Grouped Conv: split into per-group group-1 convs. X and Y
// channel slices are strided views over the contiguous bases
// (contig_of inside k_conv_g1 materializes the X slice; the Y
// slice is written through a scratch output plus a strided
// copy). W slices are contiguous blocks by construction (the
// weight's dim 1 is already C/group).
void k_conv(const RT& X0, const RT& W0, const RT* B, RT& Y, const std::vector<int64_t>& strides_a,
            const std::vector<int64_t>& pads, const std::vector<int64_t>& dil, int64_t group, Scratch& sc) {
    if (group == 1) { k_conv_g1(X0, W0, B, Y, strides_a, pads, dil, sc); return; }
    RT X = sc.contig_of(X0), W = sc.contig_of(W0);
    int64_t N = X.shape[0], C = X.shape[1], O = W.shape[0];
    if (C % group != 0 || O % group != 0) fail("Conv: channels not divisible by group");
    int64_t Cg = C / group, Og = O / group;
    int64_t Sin = X.numel() / (N * C), Sout = Y.numel() / (N * O);
    int es = (int)onnxm::elem_size(X.dtype);
    int64_t Ksp = 1;
    for (size_t i = 1; i < W.shape.size(); i++) Ksp *= W.shape[i];
    Ksp /= Cg;
    for (int64_t g = 0; g < group; g++) {
        auto mark = sc.mark();
        RT xg; xg.dtype = X.dtype; xg.shape = X.shape; xg.shape[1] = Cg; xg.strides = X.strides;
        xg.p = X.p + g * Cg * Sin * es;
        RT wg; wg.dtype = W.dtype; wg.shape = W.shape; wg.shape[0] = Og; wg.strides = W.strides;
        wg.p = W.p + g * Og * Cg * Ksp * es;
        RT bg; const RT* bgp = nullptr;
        if (B) { bg.dtype = B->dtype; bg.shape = {Og}; bg.strides = {1}; bg.p = B->p + g * Og * es; bgp = &bg; }
        RT yg; yg.dtype = Y.dtype; yg.shape = Y.shape; yg.shape[1] = Og;
        yg.p = sc.alloc((size_t)N * Og * Sout * es);
        // Contiguous strides for the temp output [N, Og, spatial...].
        yg.strides.assign(Y.shape.size(), 1);
        for (int i = (int)Y.shape.size() - 2; i >= 0; i--) yg.strides[i] = yg.strides[i + 1] * yg.shape[i + 1];
        k_conv_g1(xg, wg, bgp, yg, strides_a, pads, dil, sc);
        RT yv; yv.dtype = Y.dtype; yv.shape = Y.shape; yv.shape[1] = Og; yv.strides = Y.strides;
        yv.p = Y.p + g * Og * Sout * es;
        copy_tensor(yv, yg);
        sc.reset(mark);
    }
}

// fp32 ConvTranspose (phase 2; the baked vits graph's 5 nodes are all
// sp=1, group=1). Gather form: each output element accumulates over
// (ci, k) with li = (lo + pad_begin - k*dil) / stride where divisible.
// Output lines (n, o) are independent, so they fan out over the pool.
void k_convtranspose(const RT& X0, const RT& W0, const RT* B, RT& Y,
                     const std::vector<int64_t>& strides_a, const std::vector<int64_t>& pads_b,
                     const std::vector<int64_t>& dil, Scratch& sc) {
    RT X = sc.contig_of(X0), W = sc.contig_of(W0);
    int sp = (int)X.shape.size() - 2;
    if (sp < 1 || sp > 2) fail("ConvTranspose: spatial rank not 1..2");
    if (X.dtype != 1 || W.dtype != 1 || Y.dtype != 1) fail("ConvTranspose: f32 only");
    int64_t N = X.shape[0], Ci = X.shape[1], O = W.shape[1];
    int64_t xsp = 1, ksp = 1, osp = 1;
    for (int i = 0; i < sp; i++) {
        xsp *= X.shape[2 + i];
        ksp *= W.shape[2 + i];
        osp *= Y.shape[2 + i];
    }
    const float* x = (const float*)X.p;
    const float* w = (const float*)W.p;
    float* y = (float*)Y.p;
    if (sp == 1 && dil[0] == 1) {
        // Phase decomposition (same structure as the packed
        // conv_transpose1d_fast): output positions lo = m*stride + p
        // draw on taps kk with (p + pad_b - kk) % stride == 0; per
        // phase the result is a plain stride-1 conv1d with a gathered
        // phase kernel, written back interleaved. The gather form
        // below costs an integer division per (output, tap) candidate
        // and is ~100x slower on the vits decoder upsamples.
        RT Bc;
        const float* bias = nullptr;
        if (B) { Bc = sc.contig_of(*B); bias = (const float*)Bc.p; }
        int64_t L = X.shape[2], K = W.shape[2], Tout = Y.shape[2];
        int64_t stride = strides_a[0], pl = pads_b[0];
        for (int64_t p = 0; p < stride; p++) {
            if (p >= Tout) break;
            int64_t kks[64];
            int64_t Kp = 0;
            bool overflow = false;
            for (int64_t kk = K - 1; kk >= 0; kk--) {
                int64_t rem = (p + pl - kk) % stride;
                if (rem < 0) rem += stride;
                if (rem == 0) {
                    if (Kp >= 64) { overflow = true; break; }
                    kks[Kp++] = kk;
                }
            }
            if (overflow) break;  // falls through to the gather form, which rewrites all of Y
            if (Kp == 0) {
                // No tap lands on this phase: those outputs are bias only.
                int64_t Lm0 = (Tout - 1 - p) / stride + 1;
                for (int64_t n = 0; n < N; n++)
                    for (int64_t o = 0; o < O; o++) {
                        float* dst = y + ((size_t)n * O + o) * Tout + p;
                        float bv = bias ? bias[o] : 0.0f;
                        for (int64_t m = 0; m < Lm0; m++) dst[(size_t)m * stride] = bv;
                    }
                continue;
            }
            int64_t shift0 = (p + pl - kks[0]) / stride;
            int64_t Lm = (Tout - 1 - p) / stride + 1;
            // prow[(o*Ci + ci)*Kp + j] = W[ci][o][kks[j]]
            std::vector<float> prow((size_t)O * Ci * Kp);
            for (int64_t o = 0; o < O; o++)
                for (int64_t ci = 0; ci < Ci; ci++) {
                    const float* wrow = w + ((size_t)ci * O + o) * K;
                    float* dst = prow.data() + ((size_t)o * Ci + ci) * Kp;
                    for (int64_t j = 0; j < Kp; j++) dst[j] = wrow[kks[j]];
                }
            std::vector<float> tmp((size_t)N * O * Lm);
            // Phase conv: out[m] = bias + sum_{ci,j} X[ci][m + shift0 + j] * prow[o][ci][j]
            // (kks descending makes the implied input shift increase by
            // exactly 1 per tap: a stride-1 conv with pad_l = -shift0,
            // the same form conv_transpose1d_fast uses).
            packed::pool_run(N * O * Lm * Ci * Kp, [&](int idx, int nt) {
                packed::chunk(idx, nt, N * O, [&](int64_t t0, int64_t t1) {
                    for (int64_t t = t0; t < t1; t++) {
                        int64_t n = t / O, o = t % O;
                        float* yl = tmp.data() + (size_t)t * Lm;
                        float bv = bias ? bias[o] : 0.0f;
                        for (int64_t m = 0; m < Lm; m++) yl[m] = bv;
                        for (int64_t ci = 0; ci < Ci; ci++) {
                            const float* xr = x + ((size_t)n * Ci + ci) * L;
                            const float* wr = prow.data() + ((size_t)o * Ci + ci) * Kp;
                            for (int64_t j = 0; j < Kp; j++) {
                                float wv = wr[j];
                                if (wv == 0.0f) continue;
                                int64_t base = shift0 + j;
                                int64_t m_lo = base >= 0 ? 0 : -base;
                                int64_t m_hi = std::min(Lm, L - base);
                                if (m_lo >= m_hi) continue;
                                const float* xs = xr + m_lo + base;
                                for (int64_t m = m_lo; m < m_hi; m++) yl[m] += wv * xs[m - m_lo];
                            }
                        }
                    }
                });
            });
            for (int64_t n = 0; n < N; n++)
                for (int64_t o = 0; o < O; o++) {
                    const float* src = tmp.data() + ((size_t)n * O + o) * Lm;
                    float* dst = y + ((size_t)n * O + o) * Tout + p;
                    for (int64_t m = 0; m < Lm; m++) dst[(size_t)m * stride] = src[m];
                }
            if (p == stride - 1) return;
            continue;
        }
        // overflow (Kp > 64) falls through to the gather form.
    }
    packed::pool_run(N * O * osp * Ci * ksp, [&](int idx, int nt) {
        packed::chunk(idx, nt, N * O, [&](int64_t l0, int64_t l1) {
            std::vector<int64_t> oc(sp), kc(sp), li(sp);
            for (int64_t ln = l0; ln < l1; ln++) {
                int64_t n = ln / O, o = ln % O;
                float bv = B ? ((const float*)B->p)[o] : 0.0f;
                float* yrow = y + (size_t)ln * osp;
                std::fill(oc.begin(), oc.end(), 0);
                for (int64_t oi = 0; oi < osp; oi++) {
                    double acc = bv;
                    for (int64_t ci = 0; ci < Ci; ci++) {
                        const float* xrow = x + ((size_t)n * Ci + ci) * xsp;
                        const float* wrow = w + ((size_t)ci * O + o) * ksp;
                        std::fill(kc.begin(), kc.end(), 0);
                        for (int64_t ki = 0; ki < ksp; ki++) {
                            int64_t xoff = 0;
                            bool ok = true;
                            for (int d = 0; d < sp; d++) {
                                int64_t num = oc[d] + pads_b[d] - kc[d] * dil[d];
                                if (num < 0 || num % strides_a[d] != 0) { ok = false; break; }
                                li[d] = num / strides_a[d];
                                if (li[d] >= X.shape[2 + d]) { ok = false; break; }
                                xoff = xoff * X.shape[2 + d] + li[d];
                            }
                            if (ok) acc += (double)xrow[xoff] * (double)wrow[ki];
                            for (int d = sp - 1; d >= 0; d--) {
                                if (++kc[d] < W.shape[2 + d]) break;
                                kc[d] = 0;
                            }
                        }
                    }
                    yrow[oi] = (float)acc;
                    for (int d = sp - 1; d >= 0; d--) {
                        if (++oc[d] < Y.shape[2 + d]) break;
                        oc[d] = 0;
                    }
                }
            }
        });
    });
}

// Pad (phase 2): mode 0 constant (fill), 1 reflect (mirror, edge not
// repeated), 2 edge (clamp). pads has 2*rank entries [begin..., end...].
// Fast path (f32, both contiguous): rows along the last dim. Outer
// coords are mapped once per row; the last dim is segment copies
// (memcpy for the interior, short reversed runs for reflect). The
// generic path below walks every element through the odometer and was
// the vits glue hotspot (96 calls on multi-M-element tensors).
bool k_pad_rows(const RT& X, RT& Y, const std::vector<int64_t>& pads, int mode, double fill) {
    if (X.dtype != 1 || Y.dtype != 1 || !X.contig() || !Y.contig()) return false;
    int r = (int)Y.shape.size();
    if (r < 1 || (int)X.shape.size() != r) return false;
    int64_t Wout = Y.shape[r - 1], Win = X.shape[r - 1];
    int64_t pb = pads[r - 1];
    int64_t rows = Y.numel() / Wout;
    std::vector<int64_t> oshape(Y.shape.begin(), Y.shape.end() - 1);
    const float* x = (const float*)X.p;
    float* y = (float*)Y.p;
    float fv = (float)fill;
    auto mapidx = [&](int64_t v, int64_t n) -> int64_t {
        if (mode == 2) return std::max<int64_t>(0, std::min(v, n - 1));
        if (mode == 1) {
            if (n == 1) return 0;
            while (v < 0 || v >= n) {
                if (v < 0) v = -v;
                if (v >= n) v = 2 * n - 2 - v;
            }
            return v;
        }
        return v;  // mode 0: caller range-checks
    };
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            std::vector<int64_t> oc(oshape.size(), 0);
            // Seek oc to r0.
            {
                int64_t rem = r0;
                for (int d = (int)oshape.size() - 1; d >= 0; d--) {
                    oc[d] = oshape[d] ? rem % oshape[d] : 0;
                    rem /= oshape[d] ? oshape[d] : 1;
                }
            }
            for (int64_t rr = r0; rr < r1; rr++) {
                // Map outer coords to a source row (or none for mode 0).
                int64_t srow = 0;
                bool outside = false;
                for (int d = 0; d < (int)oshape.size(); d++) {
                    int64_t n = X.shape[d];
                    int64_t v = oc[d] - pads[d];
                    if (mode == 0 && (v < 0 || v >= n)) { outside = true; break; }
                    srow = srow * n + mapidx(v, n);
                }
                float* yp = y + rr * Wout;
                if (outside) {
                    for (int64_t j = 0; j < Wout; j++) yp[j] = fv;
                } else {
                    const float* xp = x + srow * Win;
                    int64_t lo = std::max<int64_t>(0, pb);           // first j with src >= 0
                    int64_t hi = std::min<int64_t>(Wout, pb + Win);  // first j with src >= Win
                    if (mode == 0) {
                        for (int64_t j = 0; j < lo; j++) yp[j] = fv;
                        if (hi > lo) memcpy(yp + lo, xp + (lo - pb), (size_t)(hi - lo) * 4);
                        for (int64_t j = hi; j < Wout; j++) yp[j] = fv;
                    } else if (mode == 2) {
                        for (int64_t j = 0; j < lo; j++) yp[j] = xp[0];
                        if (hi > lo) memcpy(yp + lo, xp + (lo - pb), (size_t)(hi - lo) * 4);
                        for (int64_t j = hi; j < Wout; j++) yp[j] = xp[Win - 1];
                    } else {
                        for (int64_t j = 0; j < lo; j++) yp[j] = xp[mapidx(j - pb, Win)];
                        if (hi > lo) memcpy(yp + lo, xp + (lo - pb), (size_t)(hi - lo) * 4);
                        for (int64_t j = hi; j < Wout; j++) yp[j] = xp[mapidx(j - pb, Win)];
                    }
                }
                for (int d = (int)oshape.size() - 1; d >= 0; d--) {
                    if (++oc[d] < oshape[d]) break;
                    oc[d] = 0;
                }
            }
        });
    };
    if (Y.numel() >= 8192) packed::kernel_pool().run(body);
    else body(0, 1);
    return true;
}

void k_pad(const RT& X, RT& Y, const std::vector<int64_t>& pads, int mode, double fill) {
    if (k_pad_rows(X, Y, pads, mode, fill)) return;
    int r = (int)Y.shape.size();
    if (mode == 0) {
        // Fill everything, then copy the interior overlap.
        odometer(Y.shape, [&](const std::vector<int64_t>& c) {
            int64_t yo = 0;
            for (int d = 0; d < r; d++) yo += c[d] * Y.strides[d];
            write_d(Y, yo, fill);
        });
    }
    auto xa = align_of(X, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        std::vector<int64_t> xc = c;
        for (int d = 0; d < r; d++) {
            int td = xa[d];
            if (td < 0) continue;
            int64_t n = X.shape[td];
            int64_t v = c[d] - pads[td];
            if (mode == 0) {
                if (v < 0 || v >= n) return;  // outside: keep fill
            } else if (mode == 2) {
                v = std::max<int64_t>(0, std::min(v, n - 1));
            } else {  // reflect
                if (n == 1) v = 0;
                else {
                    while (v < 0 || v >= n) {
                        if (v < 0) v = -v;
                        if (v >= n) v = 2 * n - 2 - v;
                    }
                }
            }
            xc[d] = v;
        }
        int64_t yo = 0;
        for (int d = 0; d < r; d++) yo += c[d] * Y.strides[d];
        write_d(Y, yo, read_d(X, off_of(X, xc, xa)));
    });
}

// Expand (phase 2): broadcast copy to the (compile-time computed)
// output shape; off_of already treats size-1 source dims as broadcast.
void k_expand(const RT& X, RT& Y) {
    auto xa = align_of(X, Y.shape.size());
    auto ya = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        write_d(Y, off_of(Y, c, ya), read_d(X, off_of(X, c, xa)));
    });
}


// -------------------------------------------------------------- Expr -----
// Shape dimensions and symbolic int elements are expression trees over
// runtime dim symbols. Everything evaluates numerically at bind time;
// compile time only needs construction, constant folding, and equality.
struct Expr {
    enum K { C, S, ADD, SUB, MUL, DIV, SLEN, MAXK, MINK, EQK, ITE } k = C;
    int64_t c = 0;
    int sym = -1;
    std::shared_ptr<Expr> a, b, d;  // ITE: a = cond, b = then, d = else
    int64_t step = 1;  // SLEN: slice length from (a=start, b=end, step, d handled via a/b only)
};
using EP = std::shared_ptr<Expr>;
EP ec(int64_t v) { auto e = std::make_shared<Expr>(); e->k = Expr::C; e->c = v; return e; }
EP esym(int s) { auto e = std::make_shared<Expr>(); e->k = Expr::S; e->sym = s; return e; }
bool econst(const EP& e, int64_t& v) {
    if (!e) { v = 0; return true; }
    if (e->k == Expr::C) { v = e->c; return true; }
    return false;
}
int64_t eeval(const EP& e, const std::vector<int64_t>& sv) {
    switch (e->k) {
        case Expr::C: return e->c;
        case Expr::S: return sv[e->sym];
        case Expr::ADD: return eeval(e->a, sv) + eeval(e->b, sv);
        case Expr::SUB: return eeval(e->a, sv) - eeval(e->b, sv);
        case Expr::MUL: return eeval(e->a, sv) * eeval(e->b, sv);
        case Expr::DIV: {
            int64_t d = eeval(e->b, sv);
            if (d == 0) fail("expr div by zero");
            int64_t num = eeval(e->a, sv);
            if (num == INT64_MIN && d == -1) return INT64_MAX;  // saturate, never trap
            return num / d;
        }
        case Expr::SLEN: return e->c;  // resolved to a constant at bind-precompute; see compiler note
        case Expr::MAXK: return std::max(eeval(e->a, sv), eeval(e->b, sv));
        case Expr::MINK: return std::min(eeval(e->a, sv), eeval(e->b, sv));
        case Expr::EQK: return eeval(e->a, sv) == eeval(e->b, sv) ? 1 : 0;
        case Expr::ITE: return eeval(e->a, sv) != 0 ? eeval(e->b, sv) : eeval(e->d, sv);
    }
    return 0;
}
bool eseq(const EP& x, const EP& y) {
    if (x == y) return true;
    if (!x || !y || x->k != y->k) return false;
    if (x->k == Expr::C) return x->c == y->c;
    if (x->k == Expr::S) return x->sym == y->sym;
    return eseq(x->a, y->a) && eseq(x->b, y->b) && eseq(x->d, y->d);
}
EP ebin(Expr::K k, const EP& a, const EP& b) {
    int64_t ca, cb;
    bool ka = econst(a, ca), kb = econst(b, cb);
    if (ka && kb) {
        // Fold through __int128 with saturation: slice-bound sentinel
        // arithmetic (e.g. end = -2^63+1 with a negative step) must not
        // trap on INT64_MIN / -1 or wrap (phase 2).
        switch (k) {
            case Expr::ADD: case Expr::SUB: case Expr::MUL: {
                __int128 r = k == Expr::ADD ? (__int128)ca + cb
                             : k == Expr::SUB ? (__int128)ca - cb
                                              : (__int128)ca * cb;
                if (r > INT64_MAX) r = INT64_MAX;
                if (r < INT64_MIN) r = INT64_MIN;
                return ec((int64_t)r);
            }
            case Expr::DIV:
                if (cb == 0) return ec(0);
                if (ca == INT64_MIN && cb == -1) return ec(INT64_MAX);
                return ec(ca / cb);
            case Expr::MAXK: return ec(std::max(ca, cb));
            case Expr::MINK: return ec(std::min(ca, cb));
            default: break;
        }
    }
    if ((k == Expr::MAXK || k == Expr::MINK) && eseq(a, b)) return a;
    if (k == Expr::ADD && ka && ca == 0) return b;
    if (k == Expr::ADD && kb && cb == 0) return a;
    if (k == Expr::SUB && kb && cb == 0) return a;
    if (k == Expr::MUL && ka && ca == 1) return b;
    if (k == Expr::MUL && kb && cb == 1) return a;
    if (k == Expr::MUL && ((ka && ca == 0) || (kb && cb == 0))) return ec(0);
    if (k == Expr::DIV && kb && cb == 1) return a;
    auto e = std::make_shared<Expr>();
    e->k = k;
    e->a = a;
    e->b = b;
    return e;
}
EP eeq(const EP& a, const EP& b) {
    int64_t ca, cb;
    if (econst(a, ca) && econst(b, cb)) return ec(ca == cb ? 1 : 0);
    if (eseq(a, b)) return ec(1);
    auto e = std::make_shared<Expr>();
    e->k = Expr::EQK;
    e->a = a;
    e->b = b;
    return e;
}
EP eite(const EP& c, const EP& x, const EP& y) {
    int64_t cc;
    if (econst(c, cc)) return cc != 0 ? x : y;
    if (eseq(x, y)) return x;
    auto e = std::make_shared<Expr>();
    e->k = Expr::ITE;
    e->a = c;
    e->b = x;
    e->d = y;
    return e;
}
EP eadd(const EP& a, const EP& b) { return ebin(Expr::ADD, a, b); }
EP esub(const EP& a, const EP& b) { return ebin(Expr::SUB, a, b); }
EP emul(const EP& a, const EP& b) { return ebin(Expr::MUL, a, b); }
EP ediv(const EP& a, const EP& b) { return ebin(Expr::DIV, a, b); }
EP enumel(const std::vector<EP>& shape) {
    EP n = ec(1);
    for (auto& d : shape) n = emul(n, d);
    return n;
}

// ------------------------------------------------------- values/steps --
enum class SK { MATMUL, GEMM, PACKED, CONV, LN, SOFTMAX, REDUCESUM, ARGMAX, TOPK, CUMSUM,
                GATHER, GATHER_EL, SCATTER_EL, CONCAT, CONSTSHAPE, RANDNORM,
                BINOP, UNOP, CAST, WHERE, COPY,
                // Phase 2 (vits): packed conv family, fp32 ConvTranspose,
                // Pad, Expand. PCONV also carries PackedConvTranspose
                // (pc_transpose), PGATHER is PackedGather.
                PCONV, PGATHER, CONVT, PAD, EXPAND, ATTN, INSTNORM,
                SEQ_EMPTY, SEQ_SPLIT, SEQ_AT, SEQ_INSERT, SEQ_CONCAT, LOOP,
                IF, SLICE_RT, RANGE, STFT, TILE, PRELU, DFT, BNORM, CLIP, SHAPE };
struct Step {
    SK k = SK::COPY;
    std::vector<int> in, out;
    BinOp bop = BinOp::ADD;
    UnOp uop = UnOp::NEG;
    int64_t axis = 0;
    bool keepdims = true, exclusive = false, reverse = false, largest = true, sorted = true;
    int64_t topk = 0;
    int reduction = 0;
    float alpha = 1, beta = 1, eps = 0;
    // Fused epilogue (set by fuse_one_leaky): 1 = LeakyReLU with
    // epi_alpha, applied to this step's output before the step ends.
    int epi_act = 0;
    float epi_alpha = 0;
    int transA = 0, transB = 0;
    std::vector<int64_t> axes;
    std::vector<int64_t> c_strides{1}, c_pads{0}, c_dil{1};
    std::vector<int64_t> c_pads_end{0};  // ConvTranspose: end pads (begin in c_pads)
    int64_t c_group = 1;
    bool flat_copy = false;  // COPY step: logical-order (reshape) copy
    int pm_mode = 0;
    int64_t pm_rows = 0, pm_cols = 0;
    bool pm_zero = false, pm_bias = false;
    std::string pm_base;
    // PackedConv / PackedConvTranspose geometry (attr-carried).
    int64_t pc_k = 1, pc_stride = 1, pc_pad_l = 0, pc_pad_r = 0, pc_dil = 1, pc_out_pad = 0;
    bool pc_transpose = false;
    int64_t ct_out_pad = 0;   // fp32 ConvTranspose output_padding
    int pad_mode = 0;         // PAD: 0 constant, 1 reflect, 2 edge
    bool mean = false;        // REDUCESUM: ReduceMean (scale by 1/count)
    bool l2 = false;          // REDUCESUM: ReduceL2 (sqrt of sum of squares)
    int b_f16 = -1;           // MATMUL/GEMM: index into Impl::f16_weights, or -1
    const float* b_f16_src = nullptr;  // fp32 base pointer the runtime B must match
    float out_scale = 1;      // MATMUL: epilogue scale folded from a downstream scalar Mul
    int kv_slot = -1;         // CONCAT: stateful KV pair index, or -1
    onnxm::Tensor fill;
    int out_dtype = 0;
    bool keep = false;        // fusion/DCE passes must not rewrite or erase
    int uid = -1;             // stable id across pass rewrites (add_step)
    // LOOP: body step range by uid (fixed up to indices after the
    // passes), body input slot value ids [iter, cond, carried...],
    // body output value ids [cond_out, carried_out...], and the
    // value-id range the body compilation created.
    int lp_u0 = -1, lp_u1 = -1, lp_b0 = -1, lp_b1 = -1;
    std::vector<int> lp_slots, lp_body_outs;
    int lp_v0 = -1, lp_v1 = -1;
    // IF: then/else body step ranges by uid (fixed up like LOOP),
    // branch output value ids, and the value-id range both branch
    // compilations created.
    int if_u0 = -1, if_u1 = -1, if_eu0 = -1, if_eu1 = -1;
    int if_b0 = -1, if_b1 = -1, if_eb0 = -1, if_eb1 = -1;
    std::vector<int> if_t_outs, if_e_outs;
    int if_v0 = -1, if_v1 = -1;
};

struct Value {
    enum Kind { CONSTV, SYMVAL, RT } kind = RT;
    onnxm::Tensor ct;                    // CONSTV
    int sdt = 7;                         // SYMVAL materialization dtype
    std::vector<int64_t> sshape;         // SYMVAL
    std::vector<EP> selems;              // SYMVAL
    int dtype = 0;                       // RT
    std::vector<EP> rshape;              // RT
    bool is_seq = false;                 // RT: a sequence of tensors (bseqs at run)
    bool dyn = false;                    // RT: shape/buffer resolved at execution
    std::vector<EP> eshape;              // is_seq: element shape exprs
    int view_kind = 0;                   // RT: 0 materialized/input, 1 transpose, 2 reshape,
                                         // 3 unsqueeze, 4 squeeze, 5 slice, 6 identity
    int base = -1;
    std::vector<int64_t> vperm, vaxes;
    int meta0 = -1, meta1 = -1, meta2 = -1, meta3 = -1;  // slice: starts/ends/axes/steps value ids; reshape: meta0 = shape value id
    int def_step = -1, last_use = -1;
    bool is_input = false, is_output = false;
};

struct Impl {
    std::string label;
    // Stateful KV feedback pairs, in step order; the pair index
    // is the slot in a shared KvCache. Detection is structural and
    // runs on every plan; behavior changes only when a cache is
    // attached (Plan::set_kv_cache).
    struct KvPair { int in_vid, out_vid, in_idx; };
    std::vector<KvPair> kv_pairs;
    std::shared_ptr<KvCache> kv;
    std::vector<Value> vals;
    // eff_dyn memo, parallel to vals: -1 unknown, 0/1 result,
    // 2 = computation in progress (cycle guard). A value's
    // effective dyn-ness is fixed once the value and its
    // producing step exist, and eff_dyn is only queried on
    // already-lowered values, so results are stable for the
    // rest of lowering.
    std::vector<int8_t> eff_memo;
    std::vector<Step> steps;
    std::map<std::string, int> by_name;
    std::vector<std::string> in_names, out_names;
    std::vector<int> in_vals, out_vals;
    int nsyms = 0;
    std::vector<std::pair<int, int>> sym_origin;  // sym -> (input value id, dim)
    // Per-run state (single-threaded use, like a session).
    std::vector<uint8_t> arena, out_arena;
    std::vector<std::vector<uint8_t>> view_blocks;  // reshape materializations (never moved)
    std::vector<std::vector<uint8_t>> sym_bufs;
    uint64_t rng_ctr = 0;
    // Bind workspace, reused across runs: capacities persist, so a
    // decode step's bind allocates almost nothing once warm. The
    // planning algorithm and its results are unchanged.
    struct BAlloc { int vid; int64_t size; int def, use; bool out; };
    struct BBlock { size_t off, size; int use; };
    std::vector<int64_t> bsv;
    std::vector<RT> bres;
    std::vector<char> bdone;
    std::vector<std::vector<int64_t>> bcshape;
    std::vector<BAlloc> ballocs, bwork, bouta;
    std::vector<BBlock> bblocks;
    std::vector<size_t> boff_w, boff_o;
    std::vector<RT> bsin, bsout;
    Scratch bsc;
    PackedLookup lookup;
    int next_uid = 0;
    // Sequence values at run time, parallel to bres.
    std::vector<std::vector<RT>> bseqs;
    // Execution-allocated buffers for dyn values (loop-carried and
    // data-shaped outputs). Chunked like Scratch so pointers never
    // move; reset at the start of each run. Buffers handed to an
    // accumulator sequence must outlive later iterations, so this
    // arena is append-only within a run.
    std::vector<std::vector<uint8_t>> dyn_blocks;
    size_t dyn_bi = 0, dyn_top = 0;
    uint8_t* dyn_alloc(size_t bytes) {
        if (dyn_blocks.empty()) dyn_blocks.emplace_back(1 << 20);
        size_t at = (dyn_top + 63) & ~(size_t)63;
        if (at + bytes > dyn_blocks[dyn_bi].size()) {
            size_t want = std::max<size_t>(bytes + 4096, 1 << 20);
            if (dyn_bi + 1 < dyn_blocks.size() && dyn_blocks[dyn_bi + 1].size() >= want) {
                dyn_bi++;
            } else if (dyn_bi + 1 < dyn_blocks.size()) {
                dyn_blocks[dyn_bi + 1].resize(want);
                dyn_bi++;
            } else {
                dyn_blocks.emplace_back(want);
                dyn_bi = dyn_blocks.size() - 1;
            }
            at = 0;
        }
        dyn_top = at + bytes;
        return dyn_blocks[dyn_bi].data() + at;
    }
    void dyn_reset() { dyn_bi = 0; dyn_top = 0; }
    // fp16 weight storage (M2): converted copies of initializer
    // tensors used as MatMul/Gemm B operands, deduped by value id.
    bool want_f16 = false;
    std::vector<std::vector<uint16_t>> f16_weights;
    std::map<int, int> f16_by_val;
};

// ------------------------------------------------------- shape helpers -
std::vector<EP> shape_exprs(Impl& im, const Value& v) {
    if (v.kind == Value::CONSTV) {
        std::vector<EP> r;
        for (auto d : v.ct.dims) r.push_back(ec(d));
        return r;
    }
    if (v.kind == Value::SYMVAL) {
        std::vector<EP> r;
        for (auto d : v.sshape) r.push_back(ec(d));
        return r;
    }
    return v.rshape;
}
int64_t numel_of(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (auto d : s) n *= d;
    return n;
}
// Read a CONST int tensor's elements.
std::vector<int64_t> const_ints(const onnxm::Tensor& t) {
    std::vector<int64_t> r;
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; i++) {
        if (t.data_type == 7) { int64_t v; memcpy(&v, t.data.data() + i * 8, 8); r.push_back(v); }
        else if (t.data_type == 6) { int32_t v; memcpy(&v, t.data.data() + i * 4, 4); r.push_back(v); }
        else fail("const_ints: dtype not int");
    }
    return r;
}
// If a value is CONST or a SYMVAL of all-constant exprs, get its ints.
bool value_const_ints(Impl& im, int vid, std::vector<int64_t>& out) {
    Value& v = im.vals[vid];
    if (v.kind == Value::CONSTV) {
        if (v.ct.data_type != 7 && v.ct.data_type != 6) return false;
        out = const_ints(v.ct);
        return true;
    }
    if (v.kind == Value::SYMVAL) {
        out.clear();
        for (auto& e : v.selems) {
            int64_t c;
            if (!econst(e, c)) return false;
            out.push_back(c);
        }
        return true;
    }
    return false;
}
// Symbolic element list of a value used as int metadata (may be symbolic).
bool value_sym_elems(Impl& im, int vid, std::vector<EP>& out) {
    Value& v = im.vals[vid];
    if (v.kind == Value::SYMVAL) { out = v.selems; return true; }
    if (v.kind == Value::CONSTV) {
        if (v.ct.data_type != 7 && v.ct.data_type != 6) return false;
        out.clear();
        for (auto i : const_ints(v.ct)) out.push_back(ec(i));
        return true;
    }
    return false;
}

std::vector<EP> broadcast_shape(const std::vector<EP>& A, const std::vector<EP>& B) {
    size_t r = std::max(A.size(), B.size());
    std::vector<EP> out(r);
    for (size_t i = 0; i < r; i++) {
        EP a = i < r - A.size() ? ec(1) : A[i - (r - A.size())];
        EP b = i < r - B.size() ? ec(1) : B[i - (r - B.size())];
        int64_t ca, cb;
        bool ka = econst(a, ca), kb = econst(b, cb);
        if (eseq(a, b)) out[i] = a;
        else if (ka && ca == 1) out[i] = b;
        else if (kb && cb == 1) out[i] = a;
        else if (ka && kb) {
            if (ca != cb) fail("broadcast: incompatible const dims");
            out[i] = a;
        } else if (!ka) out[i] = a;
        else out[i] = b;
    }
    return out;
}

int64_t norm_axis(int64_t a, size_t rank) {
    if (a < 0) a += (int64_t)rank;
    return a;
}

// ONNX Slice element count + first selected index for one axis, with
// the spec's clamping (negative bounds count from the end, then clip:
// step > 0 clips both to [0, D]; step < 0 clips start to [0, D-1] and
// end to [-1, D-1]). Shared by the compile-time shape computation, the
// const-data slice path, the symbolic slice, and bind-time view
// resolution so the four can never drift. Sentinel bounds (INT64_MIN
// class) are safe: they stay far inside int64 after += D.
int64_t slice_count(int64_t D, int64_t s, int64_t e, int64_t step, int64_t& first) {
    if (D <= 0 || step == 0) { first = 0; return 0; }
    if (step > 0) {
        if (s < 0) s += D;
        if (e < 0) e += D;
        s = std::max<int64_t>(0, std::min(s, D));
        e = std::max<int64_t>(0, std::min(e, D));
        first = s;
        return e > s ? (e - s + step - 1) / step : 0;
    }
    if (s < 0) s += D;
    if (e < 0) e += D;
    s = std::max<int64_t>(0, std::min(s, D - 1));
    e = std::max<int64_t>(-1, std::min(e, D - 1));
    first = s;
    int64_t m = -step;
    return s > e ? (s - e - 1) / m + 1 : 0;
}

// ------------------------------------------- symbolic value operations -
// Elementwise combine of two SYMVAL/CONST int/float element lists.
EP sym_bin(Expr::K k, const EP& a, const EP& b) { return ebin(k, a, b); }

// Try to compute an elementwise binary op symbolically. Operands are
// value ids known to be CONSTV or SYMVAL. Returns new value id or -1.
int sym_binop(Impl& im, const std::string& op, int va, int vb) {
    if (op == "Pow") {
        // Symbolic Pow only as repeated multiplication by a small
        // non-negative constant exponent (phase 2: vits attention
        // reshapes use Pow(dim, 2) in their target chains).
        std::vector<int64_t> ex;
        if (!value_const_ints(im, vb, ex) || ex.size() != 1 || ex[0] < 0 || ex[0] > 8) return -1;
        std::vector<EP> ea;
        if (!value_sym_elems(im, va, ea)) return -1;
        Value& A = im.vals[va];
        Value v;
        v.kind = Value::SYMVAL;
        v.sshape = A.kind == Value::SYMVAL ? A.sshape : A.ct.dims;
        for (auto& e : ea) {
            EP r = ec(1);
            for (int64_t i = 0; i < ex[0]; i++) r = ebin(Expr::MUL, r, e);
            v.selems.push_back(r);
        }
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    }
    std::vector<EP> ea, eb;
    if (!value_sym_elems(im, va, ea) || !value_sym_elems(im, vb, eb)) return -1;
    std::vector<int64_t> sa, sb;
    {
        Value& A = im.vals[va];
        Value& B = im.vals[vb];
        sa = A.kind == Value::SYMVAL ? A.sshape : A.ct.dims;
        sb = B.kind == Value::SYMVAL ? B.sshape : B.ct.dims;
    }
    // broadcast shapes (concrete small shapes)
    size_t r = std::max(sa.size(), sb.size());
    std::vector<int64_t> so(r);
    for (size_t i = 0; i < r; i++) {
        int64_t da = i < r - sa.size() ? 1 : sa[i - (r - sa.size())];
        int64_t db = i < r - sb.size() ? 1 : sb[i - (r - sb.size())];
        if (da == db) so[i] = da;
        else if (da == 1) so[i] = db;
        else if (db == 1) so[i] = da;
        else return -1;
    }
    auto at = [](const std::vector<EP>& e, const std::vector<int64_t>& s, const std::vector<int64_t>& coord) {
        int64_t off = 0, stride = 1;
        for (int d = (int)s.size() - 1; d >= 0; d--) {
            int64_t c = s[d] == 1 ? 0 : coord[d + (coord.size() - s.size())];
            off += c * stride;
            stride *= s[d];
        }
        return e[off];
    };
    Expr::K k = op == "Add" ? Expr::ADD : op == "Sub" ? Expr::SUB : op == "Mul" ? Expr::MUL
                : op == "Max" ? Expr::MAXK : op == "Min" ? Expr::MINK : Expr::DIV;
    Value v;
    v.kind = Value::SYMVAL;
    v.sshape = so;
    int64_t n = numel_of(so);
    v.selems.resize(n);
    std::vector<int64_t> coord(r, 0);
    for (int64_t i = 0; i < n; i++) {
        v.selems[i] = sym_bin(k, at(ea, sa, coord), at(eb, sb, coord));
        for (int d = (int)r - 1; d >= 0; d--) {
            if (++coord[d] < so[d]) break;
            coord[d] = 0;
        }
    }
    im.vals.push_back(std::move(v));
    return (int)im.vals.size() - 1;
}

// If all elements are constant, collapse a SYMVAL into a CONST int tensor.
int maybe_const_collapse(Impl& im, int vid) {
    Value& v = im.vals[vid];
    if (v.kind != Value::SYMVAL || v.sdt != 7) return vid;
    std::vector<int64_t> ints;
    for (auto& e : v.selems) {
        int64_t c;
        if (!econst(e, c)) return vid;
        ints.push_back(c);
    }
    Value nv;
    nv.kind = Value::CONSTV;
    nv.ct.data_type = 7;
    nv.ct.dims = v.sshape;
    nv.ct.data.resize(ints.size() * 8);
    memcpy(nv.ct.data.data(), ints.data(), ints.size() * 8);
    im.vals.push_back(std::move(nv));
    return (int)im.vals.size() - 1;
}


// ------------------------------------------------------ step executor --
// Shared by the runtime and by compile-time constant folding.
void exec_step(Impl& im, const Step& st, const std::vector<RT>& ins, std::vector<RT>& outs, Scratch& sc);

RT tensor_of_const(const onnxm::Tensor& t) {
    RT r;
    r.dtype = t.data_type;
    r.p = const_cast<uint8_t*>(t.data.data());
    r.shape = t.dims;
    r.strides = rowmajor(t.dims);
    return r;
}

// ---------------------------------------------------------- compiler ---
// fp16-stored graphs (Genie's RoBERTa export keeps every weight
// fp16 with fp32 compute, the same distribution trick as its
// HuBERT bin): decode fp16 constant tensors to fp32 when they
// enter the compiler, so every kernel sees the fp32 layout it
// already handles. Exact conversion (round-to-nearest values in
// fp16 are all representable in fp32).
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3ff;
            bits = sign | ((uint32_t)(e + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | (0xffu << 23) | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static onnxm::Tensor decode_f16_const(onnxm::Tensor t) {
    if (t.data_type != 10) return t;
    onnxm::Tensor r;
    r.data_type = 1;
    r.dims = t.dims;
    int64_t n = t.numel();
    r.data.resize((size_t)n * 4);
    const uint16_t* h = (const uint16_t*)t.data.data();
    float* f = (float*)r.data.data();
    for (int64_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
    return r;
}

struct Compiler {
    Impl& im;
    const onnxm::Graph& g;
    std::string label;
    explicit Compiler(Impl& i, const onnxm::Graph& gr, std::string l) : im(i), g(gr), label(std::move(l)) { im.label = label; }

    int vid(const std::string& name) const {
        auto it = im.by_name.find(name);
        if (it == im.by_name.end()) fail("unknown value: " + name + " in " + label);
        return it->second;
    }
    void reg(const std::string& name, int v) {
        if (!name.empty()) im.by_name[name] = v;
    }
    int new_rt(int dtype, std::vector<EP> shape) {
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype;
        v.rshape = std::move(shape);
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    }
    int new_const(onnxm::Tensor t) {
        Value v;
        v.kind = Value::CONSTV;
        v.ct = decode_f16_const(std::move(t));
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    }
    void add_step(Step st, const std::vector<int>& outs) {
        int idx = (int)im.steps.size();
        for (int o : outs) im.vals[o].def_step = idx;
        st.out = outs;
        st.uid = im.next_uid++;
        im.steps.push_back(std::move(st));
    }
    bool all_const(const std::vector<int>& ins) const {
        for (int v : ins)
            if (v >= 0 && im.vals[v].kind != Value::CONSTV) return false;
        return true;
    }
    bool all_sym_or_const(const std::vector<int>& ins) const {
        for (int v : ins)
            if (v >= 0 && im.vals[v].kind == Value::RT) return false;
        return true;
    }
};

// Execute one node immediately over constant inputs; returns CONST value
// ids. Used for folding. Supports the compute step set plus layout ops
// via ad-hoc copies. Throws on unsupported ops (caller falls back).
std::vector<int> fold_node(Compiler& C, const onnxm::Node& n, const std::vector<int>& ins, Step st,
                           const std::vector<std::pair<int, std::vector<EP>>>& out_specs,
                           const std::vector<int64_t>& symvals_dummy) {
    (void)symvals_dummy;
    std::vector<RT> inT, outT;
    for (int v : ins) inT.push_back(tensor_of_const(C.im.vals[v].ct));
    std::vector<onnxm::Tensor> bufs(out_specs.size());
    for (size_t i = 0; i < out_specs.size(); i++) {
        bufs[i].data_type = out_specs[i].first;
        int64_t nel = 1;
        for (auto& e : out_specs[i].second) {
            int64_t c;
            if (!econst(e, c)) fail("fold: symbolic output shape");
            bufs[i].dims.push_back(c);
            nel *= c;
        }
        bufs[i].data.resize((size_t)nel * onnxm::elem_size(bufs[i].data_type));
        RT r;
        r.dtype = bufs[i].data_type;
        r.p = bufs[i].data.data();
        r.shape = bufs[i].dims;
        r.strides = rowmajor(bufs[i].dims);
        outT.push_back(r);
    }
    Scratch sc;
    exec_step(C.im, st, inT, outT, sc);
    std::vector<int> r;
    for (auto& b : bufs) r.push_back(C.new_const(std::move(b)));
    return r;
}

// Symbolic structural ops on SYMVAL element lists (small int tensors).
// Broadcast two concrete shapes (numpy rules); false if they
// cannot broadcast.
static bool sym_bcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                            std::vector<int64_t>& so) {
    size_t r = std::max(a.size(), b.size());
    so.assign(r, 1);
    for (size_t i = 0; i < r; i++) {
        int64_t da = i < r - a.size() ? 1 : a[i - (r - a.size())];
        int64_t db = i < r - b.size() ? 1 : b[i - (r - b.size())];
        if (da == db) so[i] = da;
        else if (da == 1) so[i] = db;
        else if (db == 1) so[i] = da;
        else return false;
    }
    return true;
}

// Expand a symbolic element list from shape s to the (already
// broadcast-compatible) concrete target shape so.
static void sym_align(const std::vector<EP>& e, const std::vector<int64_t>& s,
                      const std::vector<int64_t>& so, std::vector<EP>& out) {
    int64_t n = numel_of(so);
    out.assign((size_t)n, ec(0));
    int r = (int)so.size();
    std::vector<int64_t> coord(r, 0);
    for (int64_t i = 0; i < n; i++) {
        int64_t off = 0, stride = 1;
        for (int d = (int)s.size() - 1; d >= 0; d--) {
            int64_t c = s[d] == 1 ? 0 : coord[d + (r - (int)s.size())];
            off += c * stride;
            stride *= s[d];
        }
        out[(size_t)i] = e[(size_t)off];
        for (int d = r - 1; d >= 0; d--) {
            if (++coord[d] < so[d]) break;
            coord[d] = 0;
        }
    }
}

static std::vector<int64_t> sym_shape_of(Impl& im, int vid) {
    Value& v = im.vals[vid];
    return v.kind == Value::SYMVAL ? v.sshape : v.ct.dims;
}

// Symbolic Equal over sym-or-const operands (RoBERTa's reshape
// spec resolution compares a shape vector against a constant).
int sym_equal(Impl& im, int va, int vb) {
    std::vector<EP> ea, eb;
    if (!value_sym_elems(im, va, ea) || !value_sym_elems(im, vb, eb)) return -1;
    std::vector<int64_t> so;
    if (!sym_bcast_shape(sym_shape_of(im, va), sym_shape_of(im, vb), so)) return -1;
    std::vector<EP> aa, ab;
    sym_align(ea, sym_shape_of(im, va), so, aa);
    sym_align(eb, sym_shape_of(im, vb), so, ab);
    Value v;
    v.kind = Value::SYMVAL;
    v.sshape = so;
    v.sdt = 9;
    for (size_t i = 0; i < aa.size(); i++) v.selems.push_back(eeq(aa[i], ab[i]));
    im.vals.push_back(std::move(v));
    return (int)im.vals.size() - 1;
}

// Symbolic Where over sym-or-const operands: elements become
// ITE expressions, evaluated at bind like every other EP.
int sym_where(Impl& im, int vc, int vx, int vy) {
    std::vector<EP> ec_, ex, ey;
    if (!value_sym_elems(im, vc, ec_) || !value_sym_elems(im, vx, ex) ||
        !value_sym_elems(im, vy, ey))
        return -1;
    std::vector<int64_t> s1, so;
    if (!sym_bcast_shape(sym_shape_of(im, vc), sym_shape_of(im, vx), s1)) return -1;
    if (!sym_bcast_shape(s1, sym_shape_of(im, vy), so)) return -1;
    std::vector<EP> ac, ax, ay;
    sym_align(ec_, sym_shape_of(im, vc), so, ac);
    sym_align(ex, sym_shape_of(im, vx), so, ax);
    sym_align(ey, sym_shape_of(im, vy), so, ay);
    Value& X = im.vals[vx];
    Value v;
    v.kind = Value::SYMVAL;
    v.sshape = so;
    v.sdt = X.kind == Value::SYMVAL ? X.sdt : X.ct.data_type;
    for (size_t i = 0; i < ac.size(); i++) v.selems.push_back(eite(ac[i], ax[i], ay[i]));
    im.vals.push_back(std::move(v));
    return (int)im.vals.size() - 1;
}

int sym_structural(Compiler& C, const onnxm::Node& n, const std::vector<int>& ins) {
    Impl& im = C.im;
    const std::string& op = n.op;
    Value& A = im.vals[ins[0]];
    if (std::getenv("FE_DEBUG") && n.name.find("attn_layers.0") != std::string::npos)
        std::fprintf(stderr, "[sym] %s %s in0kind=%d\n", op.c_str(), n.name.c_str(), (int)A.kind);
    if (op == "Concat") {
        // Concat is handled before the SYMVAL-first-input check below:
        // its inputs may lead with a CONSTV (phase 2: vits pads chains
        // concat unsqueezed constants ahead of the symbolic elements).
        auto mkc = [&](std::vector<int64_t> shape, std::vector<EP> elems, int sdt) {
            Value v;
            v.kind = Value::SYMVAL;
            v.sshape = std::move(shape);
            v.selems = std::move(elems);
            v.sdt = sdt;
            im.vals.push_back(std::move(v));
            return (int)im.vals.size() - 1;
        };
        int64_t axis = n.attr_i("axis", 0);
        std::vector<int64_t> nshape;
        std::vector<EP> elems;
        std::vector<std::vector<EP>> per_in;
        for (int v : ins) {
            Value& B = im.vals[v];
            std::vector<EP> be;
            if (!value_sym_elems(im, v, be)) return -1;
            std::vector<int64_t> bshape = B.kind == Value::SYMVAL ? B.sshape : B.ct.dims;
            if (nshape.empty()) nshape = bshape;
            else {
                if (bshape.size() != nshape.size()) return -1;
                int64_t ax = norm_axis(axis, (int64_t)nshape.size());
                nshape[ax] += bshape[ax];
            }
            per_in.push_back(std::move(be));
        }
        // Only axis-concat of equal-rank small tensors; do it by blocks
        // when axis == 0 (the observed case: shape vectors), else bail.
        int64_t ax = norm_axis(axis, (int64_t)nshape.size());
        if (ax != 0) return -1;
        for (auto& be : per_in) elems.insert(elems.end(), be.begin(), be.end());
        int rsdt = A.kind == Value::SYMVAL ? A.sdt : A.ct.data_type;
        return mkc(nshape, elems, rsdt);
    }
    if (A.kind != Value::SYMVAL) return -1;
    auto mk = [&](std::vector<int64_t> shape, std::vector<EP> elems, int sdt) {
        Value v;
        v.kind = Value::SYMVAL;
        v.sshape = std::move(shape);
        v.selems = std::move(elems);
        v.sdt = sdt;
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    };
    if (op == "Unsqueeze" || op == "Squeeze") {
        std::vector<int64_t> axes;
        if (ins.size() > 1 && ins[1] >= 0) {
            if (!value_const_ints(im, ins[1], axes)) return -1;
        } else if (auto* at = n.attr("axes")) axes = at->ints;
        int64_t rank = (int64_t)A.sshape.size();
        std::vector<int64_t> nshape;
        std::vector<char> drop(rank, 0), add;
        if (op == "Unsqueeze") {
            int64_t nr = rank + (int64_t)axes.size();
            std::vector<char> ins_at(nr, 0);
            for (auto a : axes) ins_at[norm_axis(a, nr)] = 1;
            size_t src = 0;
            for (int64_t i = 0; i < nr; i++) nshape.push_back(ins_at[i] ? 1 : A.sshape[src++]);
            return mk(nshape, A.selems, A.sdt);
        }
        if (axes.empty()) {
            for (int64_t i = 0; i < rank; i++)
                if (A.sshape[i] == 1) drop[i] = 1;
        } else
            for (auto a : axes) drop[norm_axis(a, rank)] = 1;
        for (int64_t i = 0; i < rank; i++)
            if (!drop[i]) nshape.push_back(A.sshape[i]);
        // Elements: squeezing only removes size-1 dims; but our SYMVAL
        // shapes with dropped dims must have those dims == 1 (checked at
        // materialization time in the runtime path; here trust + verify).
        for (int64_t i = 0; i < rank; i++)
            if (drop[i] && A.sshape[i] != 1) return -1;
        return mk(nshape, A.selems, A.sdt);
    }
    if (op == "Reshape") {
        std::vector<int64_t> target;
        if (!value_const_ints(im, ins[1], target)) return -1;
        int64_t total = numel_of(A.sshape), prod = 1;
        int infer = -1;
        for (size_t i = 0; i < target.size(); i++) {
            if (target[i] == -1) infer = (int)i;
            else if (target[i] == 0) target[i] = A.sshape[i];
            else prod *= target[i];
        }
        if (infer >= 0) target[infer] = total / prod;
        return mk(target, A.selems, A.sdt);
    }
    if (op == "Slice") {
        std::vector<int64_t> starts, ends, axes, steps;
        if (!value_const_ints(im, ins[1], starts) || !value_const_ints(im, ins[2], ends)) return -1;
        if (ins.size() > 3 && ins[3] >= 0 && !value_const_ints(im, ins[3], axes)) return -1;
        if (ins.size() > 4 && ins[4] >= 0 && !value_const_ints(im, ins[4], steps)) return -1;
        int64_t rank = (int64_t)A.sshape.size();
        if (axes.empty()) { axes.resize(rank); for (int64_t i = 0; i < rank; i++) axes[i] = i; }
        if (steps.empty()) steps.assign(axes.size(), 1);
        std::vector<int64_t> nshape = A.sshape;
        std::vector<int64_t> off(rank, 0), stp(rank, 1);
        for (size_t i = 0; i < axes.size(); i++) {
            int64_t ax = norm_axis(axes[i], rank);
            int64_t first;
            nshape[ax] = slice_count(A.sshape[ax], starts[i], ends[i], steps[i], first);
            off[ax] = first;
            stp[ax] = steps[i];
        }
        int64_t n = numel_of(nshape);
        std::vector<EP> elems(n);
        std::vector<int64_t> coord(nshape.size(), 0);
        auto astr = rowmajor(A.sshape);
        for (int64_t i = 0; i < n; i++) {
            int64_t so = 0;
            for (size_t d = 0; d < coord.size(); d++) so += (off[d] + coord[d] * stp[d]) * astr[d];
            elems[i] = A.selems[so];
            for (int d = (int)coord.size() - 1; d >= 0; d--) {
                if (++coord[d] < nshape[d]) break;
                coord[d] = 0;
            }
        }
        return mk(nshape, elems, A.sdt);
    }
    if (op == "Gather") {
        std::vector<int64_t> idx;
        if (!value_const_ints(im, ins[1], idx)) return -1;
        int64_t axis = norm_axis(n.attr_i("axis", 0), (int64_t)A.sshape.size());
        // Output shape = A.shape[:axis] + indices.shape + A.shape[axis+1:]
        // (a 0-dim scalar index REMOVES the axis -- the vits shape chains
        // rely on this; the phase-1 code assumed 1-D indices throughout).
        Value& B = im.vals[ins[1]];
        std::vector<int64_t> ishape = B.kind == Value::SYMVAL ? B.sshape : B.ct.dims;
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < axis; i++) outer *= A.sshape[i];
        for (size_t i = (size_t)axis + 1; i < A.sshape.size(); i++) inner *= A.sshape[i];
        std::vector<int64_t> nshape;
        for (int64_t i = 0; i < axis; i++) nshape.push_back(A.sshape[i]);
        for (auto d : ishape) nshape.push_back(d);
        for (size_t i = (size_t)axis + 1; i < A.sshape.size(); i++) nshape.push_back(A.sshape[i]);
        std::vector<EP> elems((size_t)outer * idx.size() * inner);
        for (int64_t o = 0; o < outer; o++)
            for (size_t ii = 0; ii < idx.size(); ii++) {
                int64_t s = idx[ii] < 0 ? idx[ii] + A.sshape[axis] : idx[ii];
                for (int64_t k = 0; k < inner; k++)
                    elems[((o * (int64_t)idx.size()) + (int64_t)ii) * inner + k] =
                        A.selems[((o * A.sshape[axis]) + s) * inner + k];
            }
        return mk(nshape, elems, A.sdt);
    }
    if (op == "Cast") {
        int64_t to = n.attr_i("to", 0);
        return mk(A.sshape, A.selems, (int)to);
    }
    if (op == "Transpose") {
        std::vector<int64_t> perm;
        if (auto* at = n.attr("perm")) perm = at->ints;
        int64_t rank = (int64_t)A.sshape.size();
        if (perm.empty()) { perm.resize(rank); for (int64_t i = 0; i < rank; i++) perm[i] = rank - 1 - i; }
        std::vector<int64_t> nshape(rank);
        for (int64_t i = 0; i < rank; i++) nshape[i] = A.sshape[perm[i]];
        int64_t n = numel_of(nshape);
        std::vector<EP> elems(n);
        std::vector<int64_t> coord(nshape.size(), 0);
        auto astr = rowmajor(A.sshape);
        for (int64_t i = 0; i < n; i++) {
            int64_t so = 0;
            for (int64_t d = 0; d < rank; d++) so += coord[d] * astr[perm[d]];
            elems[i] = A.selems[so];
            for (int d = (int)coord.size() - 1; d >= 0; d--) {
                if (++coord[d] < nshape[d]) break;
                coord[d] = 0;
            }
        }
        return mk(nshape, elems, A.sdt);
    }
    // Concat is handled at the top of this function (its first input
    // may be a CONSTV).
    return -1;
}


// --------------------------------------------------------- lowering ----
int dtype_of(Impl& im, int vid) {
    Value& v = im.vals[vid];
    if (v.kind == Value::CONSTV) return v.ct.data_type;
    if (v.kind == Value::SYMVAL) return v.sdt;
    return v.dtype;
}
int promoted_dtype(int a, int b) {
    if (a == b) return a;
    if (a == 11 || b == 11) return 11;
    if (a == 1 || b == 1) return 1;
    if (a == 7 || b == 7) return 7;
    if (a == 6 || b == 6) return 6;
    return a;
}

std::vector<EP> matmul_shape(const std::vector<EP>& A, const std::vector<EP>& B) {
    std::vector<EP> a = A, b = B;
    bool a1 = a.size() == 1, b1 = b.size() == 1;
    if (a1) a = {ec(1), a[0]};
    if (b1) b = {b[0], ec(1)};
    size_t r = std::max(a.size(), b.size());
    std::vector<EP> out;
    for (size_t i = 0; i + 2 < r; i++) {
        EP da = i < r - a.size() ? ec(1) : a[i - (r - a.size())];
        EP db = i < r - b.size() ? ec(1) : b[i - (r - b.size())];
        out.push_back(eseq(da, db) || [&] { int64_t c; return econst(da, c) && c == 1; }() ? db : da);
    }
    out.push_back(a[a.size() - 2]);
    out.push_back(b[b.size() - 1]);
    if (a1) out.erase(out.begin() + (int)out.size() - 2);
    if (b1) out.pop_back();
    return out;
}

// Generic permute copy for constant folding of Transpose.
onnxm::Tensor permute_copy(const onnxm::Tensor& t, const std::vector<int64_t>& perm) {
    onnxm::Tensor o;
    o.data_type = t.data_type;
    o.name = t.name;
    for (auto p : perm) o.dims.push_back(t.dims[p]);
    o.data.resize(t.data.size());
    size_t es = onnxm::elem_size(t.data_type);
    std::vector<int64_t> coord(o.dims.size(), 0);
    auto istr = rowmajor(t.dims);
    int64_t n = numel_of(o.dims);
    for (int64_t i = 0; i < n; i++) {
        int64_t so = 0;
        for (size_t d = 0; d < coord.size(); d++) so += coord[d] * istr[perm[d]];
        int64_t dof = 0;
        auto ostr = rowmajor(o.dims);
        for (size_t d = 0; d < coord.size(); d++) dof += coord[d] * ostr[d];
        memcpy(o.data.data() + dof * (int64_t)es, t.data.data() + so * (int64_t)es, es);
        for (int d = (int)coord.size() - 1; d >= 0; d--) {
            if (++coord[d] < o.dims[d]) break;
            coord[d] = 0;
        }
    }
    return o;
}

// fp16 weight storage marking (M2): if a MatMul/Gemm step's B input
// resolves through views to a graph initializer (fp32, at least 4096
// elements), convert it once into Impl::f16_weights and record the
// binding on the step. The executor re-checks the runtime pointer
// against b_f16_src, so any view whose data does not start exactly
// at the tensor base (a slice with an offset, say) falls back to
// the fp32 operand.
void mark_b_f16(Compiler& C, Step& st) {
    Impl& im = C.im;
    if (!im.want_f16 || st.in.size() < 2 || st.in[1] < 0) return;
    int v = st.in[1];
    while (true) {
        const Value& val = im.vals[v];
        if (val.kind == Value::CONSTV) break;
        if (val.kind == Value::RT && val.base >= 0 && val.view_kind != 0) {
            v = val.base;
            continue;
        }
        return;
    }
    const onnxm::Tensor& ct = im.vals[v].ct;
    if (ct.data_type != 1 || ct.numel() < 4096 || ct.data.empty()) return;
    auto it = im.f16_by_val.find(v);
    int idx;
    if (it != im.f16_by_val.end()) {
        idx = it->second;
    } else {
        idx = (int)im.f16_weights.size();
        im.f16_weights.emplace_back((size_t)ct.numel());
        avx2k::f32_to_f16((const float*)ct.data.data(), im.f16_weights.back().data(), ct.numel());
        im.f16_by_val[v] = idx;
    }
    st.b_f16 = idx;
    st.b_f16_src = (const float*)ct.data.data();
}

void lower_node(Compiler& C, const onnxm::Node& n);

// Loop lowering: the body compiles into the same Impl, appended
// after the values/steps that exist so far. Body inputs become
// fresh slot values ([iter, cond, carried...]); outer names stay
// visible as captures through the shared by_name map, which is
// snapshot-restored afterwards so body-local names do not leak.
// The LOOP step records the body step range by uid (pass rewrites
// shift indices; a fixup after the passes resolves uids back to
// indices) and the body value range (bind skips its views; the
// runner resolves them per iteration). Scan outputs (body
// outputs beyond cond + carried) are not supported.
void lower_if(Compiler& C, const onnxm::Node& n, const std::vector<int>& ins) {
    Impl& im = C.im;
    auto* ta = n.attr("then_branch");
    auto* ea = n.attr("else_branch");
    if (!ta || !ta->g || !ea || !ea->g) fail("If (" + n.name + "): missing branch graph");
    if (ins.size() != 1 || ins[0] < 0) fail("If (" + n.name + "): needs one cond input");
    auto saved_names = im.by_name;
    int v_begin = (int)im.vals.size();
    auto compile_branch = [&](const onnxm::Graph& br, const char* tag,
                              std::vector<int>& outs, int& u0, int& u1) {
        Compiler BC(im, br, C.label + tag);
        for (auto& t : br.initializers) {
            Value v;
            v.kind = Value::CONSTV;
            v.ct = decode_f16_const(t);
            im.vals.push_back(std::move(v));
            BC.reg(t.name, (int)im.vals.size() - 1);
        }
        u0 = im.next_uid;
        for (auto& bn : br.nodes) lower_node(BC, bn);
        u1 = im.next_uid;
        for (auto& vo : br.outputs) outs.push_back(BC.vid(vo.name));
        im.by_name = saved_names;
    };
    std::vector<int> t_outs, e_outs;
    int tu0 = 0, tu1 = 0, eu0 = 0, eu1 = 0;
    compile_branch(*ta->g, ":then", t_outs, tu0, tu1);
    compile_branch(*ea->g, ":else", e_outs, eu0, eu1);
    if (t_outs.size() != e_outs.size() || t_outs.size() != n.outputs.size())
        fail("If (" + n.name + "): branch output count mismatch");
    int v_end = (int)im.vals.size();
    auto info_of = [](const Value& v, int& dt, std::vector<EP>& rs) {
        if (v.kind == Value::CONSTV) {
            dt = v.ct.data_type;
            rs.clear();
            for (auto d : v.ct.dims) rs.push_back(ec(d));
        } else if (v.kind == Value::SYMVAL) {
            dt = v.sdt;
            rs.clear();
            for (auto d : v.sshape) rs.push_back(ec(d));
        } else {
            dt = v.dtype;
            rs = v.rshape;
        }
    };
    std::vector<int> outs;
    for (size_t i = 0; i < t_outs.size(); i++) {
        const Value& tv = im.vals[t_outs[i]];
        int tdt, edt;
        std::vector<EP> trs, ers;
        info_of(tv, tdt, trs);
        info_of(im.vals[e_outs[i]], edt, ers);
        Value v;
        v.kind = Value::RT;
        v.dtype = tdt;
        v.is_seq = tv.is_seq;
        v.eshape = tv.eshape;
        v.rshape = trs;
        if (!tv.is_seq) {
            bool same = trs.size() == ers.size();
            for (size_t d = 0; same && d < trs.size(); d++)
                if (!eseq(trs[d], ers[d])) same = false;
            if (!same) v.dyn = true;
        }
        im.vals.push_back(std::move(v));
        outs.push_back((int)im.vals.size() - 1);
    }
    Step st;
    st.k = SK::IF;
    st.keep = true;
    st.in = ins;
    st.if_u0 = tu0;
    st.if_u1 = tu1;
    st.if_eu0 = eu0;
    st.if_eu1 = eu1;
    st.if_t_outs = t_outs;
    st.if_e_outs = e_outs;
    st.if_v0 = v_begin;
    st.if_v1 = v_end;
    C.add_step(st, outs);
    for (auto& s : im.steps)
        if ((s.uid >= tu0 && s.uid < tu1) || (s.uid >= eu0 && s.uid < eu1)) s.keep = true;
    for (size_t i = 0; i < outs.size(); i++) C.reg(n.outputs[i], outs[i]);
}

void lower_loop(Compiler& C, const onnxm::Node& n, const std::vector<int>& ins) {
    Impl& im = C.im;
    auto* body_attr = n.attr("body");
    if (!body_attr || !body_attr->g) fail("Loop (" + n.name + "): no body graph");
    const onnxm::Graph& body = *body_attr->g;
    if (ins.size() < 2) fail("Loop (" + n.name + "): fewer than 2 inputs");
    int ncarried = (int)ins.size() - 2;
    if ((int)body.inputs.size() != ncarried + 2)
        fail("Loop (" + n.name + "): body input count mismatch");
    auto saved_names = im.by_name;
    Compiler BC(im, body, C.label + ":loop");
    int v_begin = (int)im.vals.size();
    for (auto& t : body.initializers) {
        Value v;
        v.kind = Value::CONSTV;
        v.ct = decode_f16_const(t);
        im.vals.push_back(std::move(v));
        BC.reg(t.name, (int)im.vals.size() - 1);
    }
    std::vector<int> slots;
    for (size_t i = 0; i < body.inputs.size(); i++) {
        Value v;
        v.kind = Value::RT;
        if (i == 0) v.dtype = 7;       // iteration count (i64 scalar)
        else if (i == 1) v.dtype = 9;  // condition (bool scalar)
        else {
            int src = ins[i];
            if (src < 0) fail("Loop (" + n.name + "): carried input omitted");
            const Value& sv = im.vals[src];
            v.dtype = sv.dtype;
            v.is_seq = sv.is_seq;
            v.eshape = sv.eshape;
            v.rshape = sv.rshape;
            v.dyn = sv.dyn;
        }
        im.vals.push_back(std::move(v));
        slots.push_back((int)im.vals.size() - 1);
        BC.reg(body.inputs[i].name, slots.back());
    }
    int u_begin = im.next_uid;
    for (auto& bn : body.nodes) lower_node(BC, bn);
    std::vector<int> body_outs;
    for (auto& vo : body.outputs) body_outs.push_back(BC.vid(vo.name));
    if ((int)body_outs.size() < 1 + ncarried)
        fail("Loop (" + n.name + "): body outputs fewer than cond + carried");
    if ((int)body_outs.size() > 1 + ncarried)
        fail("Loop (" + n.name + "): scan outputs not supported");
    int v_end = (int)im.vals.size();
    std::vector<int> outs;
    for (int j = 0; j < ncarried; j++) {
        const Value& bv = im.vals[body_outs[1 + j]];
        Value v;
        v.kind = Value::RT;
        v.dtype = bv.dtype;
        v.is_seq = bv.is_seq;
        v.eshape = bv.eshape;
        v.rshape = bv.rshape;
        v.dyn = bv.dyn;
        im.vals.push_back(std::move(v));
        outs.push_back((int)im.vals.size() - 1);
    }
    Step st;
    st.k = SK::LOOP;
    st.keep = true;
    st.in = ins;
    st.lp_u0 = u_begin;
    st.lp_u1 = im.next_uid;
    st.lp_slots = slots;
    st.lp_body_outs = body_outs;
    st.lp_v0 = v_begin;
    st.lp_v1 = v_end;
    C.add_step(st, outs);
    for (auto& s : im.steps)
        if (s.uid >= st.lp_u0 && s.uid < st.lp_u1) s.keep = true;
    im.by_name = saved_names;
    for (size_t i = 0; i < outs.size(); i++) C.reg(n.outputs[i], outs[i]);
}

// Effective dyn-ness, valid during lowering (before the
// Plan::compile propagation pass runs): a value counts as dyn
// when it is flagged dyn, when it is a view of an effectively
// dyn base, or when its producing step reads an effectively
// dyn input. Producer steps always precede their consumers in
// the steps array, so the recursion terminates -- but without
// memoization it re-walks shared ancestors on every query and
// blows up exponentially on fan-out-heavy graphs (a GPT stage
// plan's compile spent >99% of its time here, perf-verified).
// Results are memoized on the Impl; see the eff_memo comment.
static bool eff_dyn(Impl& im, int vid) {
    if ((size_t)vid >= im.eff_memo.size()) im.eff_memo.resize(im.vals.size(), -1);
    int8_t& memo = im.eff_memo[vid];
    if (memo >= 0) return memo == 1;
    memo = 2;  // in progress; a re-entrant query sees "not dyn"
    bool r = false;
    Value& v = im.vals[vid];
    if (v.dyn) r = true;
    else if (v.view_kind != 0 && v.base >= 0) r = eff_dyn(im, v.base);
    else if (v.kind == Value::RT && v.def_step >= 0) {
        const Step& s = im.steps[v.def_step];
        if (s.k == SK::IF) {
            for (size_t j = 0; j < s.out.size() && !r; j++) {
                if (s.out[j] != vid) continue;
                if (j < s.if_t_outs.size() && eff_dyn(im, s.if_t_outs[j])) r = true;
                if (j < s.if_e_outs.size() && eff_dyn(im, s.if_e_outs[j])) r = true;
            }
        } else {
            if (s.k == SK::LOOP) {
                for (size_t j = 0; j < s.out.size() && !r; j++)
                    if (s.out[j] == vid && j + 1 < s.lp_body_outs.size() &&
                        eff_dyn(im, s.lp_body_outs[j + 1]))
                        r = true;
            }
            if (!r)
                for (int in : s.in)
                    if (in >= 0 && im.vals[in].kind == Value::RT && !im.vals[in].is_seq &&
                        eff_dyn(im, in)) {
                        r = true;
                        break;
                    }
        }
    }
    // im.vals may have grown (and moved) during the recursion;
    // write through a fresh index, not the cached reference.
    im.eff_memo[vid] = r ? 1 : 0;
    return r;
}

void lower_node(Compiler& C, const onnxm::Node& n) {
    Impl& im = C.im;
    const std::string& op = n.op;
    if (std::getenv("FE_DEBUG")) std::fprintf(stderr, "[lower] %s %s\n", op.c_str(), n.name.c_str());
    std::vector<int> ins;
    if (op == "PackedMatMul") {
        // Packed slots (__pk/__sc/__ze) are not plan values; only the
        // data input and the optional bias resolve through by_name.
        bool hz = n.attr_i("has_zero", 0) != 0;
        bool hb = n.attr_i("has_bias", 0) != 0;
        ins.push_back(C.vid(n.inputs[0]));
        ins.push_back(-2);
        ins.push_back(-2);
        if (hz) ins.push_back(-2);
        if (hb) ins.push_back(C.vid(n.inputs[3 + (hz ? 1 : 0)]));
    } else if (op == "PackedConv" || op == "PackedConvTranspose") {
        // Fixed 5 slots [X, base__pk, base__sc, base__ze-or-"", bias-or-""];
        // only X and the bias (slot 4) are plan values.
        bool hb = n.attr_i("has_bias", 0) != 0;
        ins.push_back(C.vid(n.inputs[0]));
        ins.push_back(-2);
        ins.push_back(-2);
        ins.push_back(-2);
        ins.push_back(hb ? C.vid(n.inputs[4]) : -1);
    } else if (op == "PackedGather") {
        // Slots [base__pk, base__sc, base__ze-or-"", indices]; only the
        // indices (slot 3) are a plan value.
        ins.push_back(C.vid(n.inputs[3]));
    } else {
        for (auto& name : n.inputs) ins.push_back(name.empty() ? -1 : C.vid(name));
    }
    auto reg_out = [&](int idx, int v) { C.reg(n.outputs[idx], v); };
    auto fail_op = [&](const std::string& m) { fail("op " + op + " (" + n.name + "): " + m); };

    // ---- Constant / Shape / Identity ----
    if (op == "Constant") {
        auto* t = n.attr("value");
        if (!t || !t->t) fail_op("no value attr");
        reg_out(0, C.new_const(*t->t));
        return;
    }
    if (op == "Shape") {
        // A dyn input whose symbolic rank is unknown (empty
        // rshape, e.g. a SLICE_RT output) has no compile-time
        // dim list: emit a runtime SHAPE step that reads the
        // true dims at execution. Collapsing its empty SYMVAL
        // to a fake const was the prompt_encoder fold crash.
        if (im.vals[ins[0]].kind == Value::RT && im.vals[ins[0]].rshape.empty() &&
            eff_dyn(im, ins[0])) {
            Step st;
            st.k = SK::SHAPE;
            st.in = {ins[0]};
            int o = C.new_rt(7, {});
            im.vals[o].dyn = true;
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        // Opset 15+: optional start/end attrs select a slice
        // of the dim list (negatives count from the end).
        Value v;
        v.kind = Value::SYMVAL;
        v.sdt = 7;
        auto sh = shape_exprs(im, im.vals[ins[0]]);
        int64_t rank = (int64_t)sh.size();
        int64_t start = n.attr_i("start", 0);
        int64_t end = n.attr_i("end", rank);
        if (start < 0) start += rank;
        if (end < 0) end += rank;
        start = std::max<int64_t>(0, std::min(start, rank));
        end = std::max<int64_t>(0, std::min(end, rank));
        if (end < start) end = start;
        v.selems.assign(sh.begin() + start, sh.begin() + end);
        v.sshape = {(int64_t)v.selems.size()};
        im.vals.push_back(std::move(v));
        reg_out(0, maybe_const_collapse(im, (int)im.vals.size() - 1));
        return;
    }
    if (op == "Identity") {
        C.reg(n.outputs[0], ins[0]);
        return;
    }

    // ---- PackedMatMul (potato domain) ----
    if (op == "PackedMatMul") {
        Step st;
        st.k = SK::PACKED;
        st.pm_mode = (int)n.attr_i("mode", 0);
        st.pm_rows = n.attr_i("rows", 0);
        st.pm_cols = n.attr_i("cols", 0);
        st.pm_zero = n.attr_i("has_zero", 0) != 0;
        st.pm_bias = n.attr_i("has_bias", 0) != 0;
        // Input names: [x, base__pk, base__sc, (base__ze), (bias)].
        const std::string& pkn = n.inputs[1];
        size_t suf = pkn.rfind("__pk");
        st.pm_base = suf == std::string::npos ? pkn : pkn.substr(0, suf);
        st.in.push_back(ins[0]);
        size_t bi = 3 + (st.pm_zero ? 1 : 0);
        if (st.pm_bias) st.in.push_back(ins[bi]);
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        xs.back() = ec((st.pm_mode == 0 || st.pm_mode == 2) ? st.pm_cols : st.pm_rows);
        int o = C.new_rt(1, xs);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- PackedConv / PackedConvTranspose / PackedGather (potato) ----
    // Glue only: the step resolves a PackedBinding through the same
    // lookup as PackedMatMul and calls the vits_kernels.cpp kernels --
    // the exact functions the ORT custom ops call (see vc_compute_impl
    // in vits_packed.h). Output-shape formulas mirror that glue.
    if (op == "PackedConv" || op == "PackedConvTranspose") {
        Step st;
        st.k = SK::PCONV;
        st.pc_transpose = op == "PackedConvTranspose";
        st.pm_mode = (int)n.attr_i("mode", 0);
        st.pm_rows = n.attr_i("w_rows", 0);
        st.pm_cols = n.attr_i("w_cols", 0);
        st.pm_zero = n.attr_i("has_zero", 0) != 0;
        st.pm_bias = n.attr_i("has_bias", 0) != 0;
        st.pc_k = n.attr_i("k", 1);
        st.pc_stride = n.attr_i("stride", 1);
        st.pc_pad_l = n.attr_i("pad_l", 0);
        st.pc_pad_r = n.attr_i("pad_r", 0);
        st.pc_dil = n.attr_i("dilation", 1);
        st.pc_out_pad = n.attr_i("out_pad", 0);
        const std::string& pkn = n.inputs[1];
        size_t suf = pkn.rfind("__pk");
        st.pm_base = suf == std::string::npos ? pkn : pkn.substr(0, suf);
        st.in.push_back(ins[0]);
        if (st.pm_bias) st.in.push_back(ins[4]);
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        if (xs.size() != 3) fail_op("PackedConv: X must be (N, C, L)");
        EP och, olen;
        if (!st.pc_transpose) {
            och = ec(st.pm_rows);
            EP num = eadd(xs[2], ec(st.pc_pad_l + st.pc_pad_r - 1));
            num = esub(num, emul(ec(st.pc_dil), ec(st.pc_k - 1)));
            olen = eadd(ediv(num, ec(st.pc_stride)), ec(1));
        } else {
            och = ec(st.pm_cols / st.pc_k);
            EP t = emul(esub(xs[2], ec(1)), ec(st.pc_stride));
            t = esub(t, ec(st.pc_pad_l + st.pc_pad_r));
            t = eadd(t, emul(ec(st.pc_dil), ec(st.pc_k - 1)));
            olen = eadd(t, ec(st.pc_out_pad + 1));
        }
        int o = C.new_rt(1, {xs[0], och, olen});
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "PackedGather") {
        Step st;
        st.k = SK::PGATHER;
        st.pm_mode = (int)n.attr_i("mode", 0);
        st.pm_rows = n.attr_i("w_rows", 0);
        st.pm_cols = n.attr_i("w_cols", 0);
        st.pm_zero = n.attr_i("has_zero", 0) != 0;
        const std::string& pkn = n.inputs[0];
        size_t suf = pkn.rfind("__pk");
        st.pm_base = suf == std::string::npos ? pkn : pkn.substr(0, suf);
        st.in.push_back(ins[0]);  // indices
        auto os = shape_exprs(im, im.vals[ins[0]]);
        os.push_back(ec(st.pm_cols));
        int o = C.new_rt(1, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- elementwise binary ----
    static const std::map<std::string, BinOp> binops = {
        {"Add", BinOp::ADD}, {"Sub", BinOp::SUB}, {"Mul", BinOp::MUL}, {"Div", BinOp::DIV},
        {"Pow", BinOp::POW}, {"Max", BinOp::MAXIMUM}, {"Less", BinOp::LESS},
        {"Greater", BinOp::GREATER}, {"Equal", BinOp::EQUAL}, {"Or", BinOp::OR}, {"And", BinOp::AND}};
    if (binops.count(op)) {
        BinOp bop = binops.at(op);
        bool cmp = op == "Less" || op == "Greater" || op == "Equal" || op == "Or" || op == "And";
        if (!cmp && (bop == BinOp::ADD || bop == BinOp::SUB || bop == BinOp::MUL || bop == BinOp::DIV ||
                     bop == BinOp::MAXIMUM || bop == BinOp::POW) &&
            C.all_sym_or_const(ins)) {
            int r = sym_binop(im, op, ins[0], ins[1]);
            if (r >= 0) {
                reg_out(0, maybe_const_collapse(im, r));
                return;
            }
        }
        if (op == "Equal" && C.all_sym_or_const(ins)) {
            int r = sym_equal(im, ins[0], ins[1]);
            if (r >= 0) {
                reg_out(0, maybe_const_collapse(im, r));
                return;
            }
        }
        auto os = broadcast_shape(shape_exprs(im, im.vals[ins[0]]), shape_exprs(im, im.vals[ins[1]]));
        int odt = cmp ? 9 : promoted_dtype(dtype_of(im, ins[0]), dtype_of(im, ins[1]));
        Step st;
        st.k = SK::BINOP;
        st.bop = bop;
        st.in = ins;
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- elementwise unary ----
    static const std::map<std::string, UnOp> unops = {
        {"Neg", UnOp::NEG}, {"Sqrt", UnOp::SQRT}, {"Sin", UnOp::SIN}, {"Cos", UnOp::COS},
        {"Exp", UnOp::EXP}, {"Tanh", UnOp::TANH}, {"Sigmoid", UnOp::SIGMOID}, {"Relu", UnOp::RELU},
        {"Abs", UnOp::ABS}, {"LeakyRelu", UnOp::LEAKY}, {"Erf", UnOp::ERF},
        {"Softplus", UnOp::SOFTPLUS}, {"Log", UnOp::LOG}};
    if (unops.count(op)) {
        if (op == "Neg" && im.vals[ins[0]].kind == Value::SYMVAL) {
            Value v = im.vals[ins[0]];
            for (auto& e : v.selems) e = esub(ec(0), e);
            im.vals.push_back(std::move(v));
            reg_out(0, maybe_const_collapse(im, (int)im.vals.size() - 1));
            return;
        }
        Step st;
        st.k = SK::UNOP;
        st.uop = unops.at(op);
        if (op == "LeakyRelu") st.alpha = n.attr_f("alpha", 0.01f);
        if (op == "Softplus") st.alpha = n.attr_f("beta", 1.0f);
        st.in = ins;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "Gelu") {
        Step st;
        st.k = SK::UNOP;
        std::string approx = "none";
        if (auto* at = n.attr("approximate")) approx = at->s;
        st.uop = approx == "tanh" ? UnOp::GELUT : UnOp::GELU;
        st.in = ins;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "Not") {
        // Not(x) == Equal(x, false)
        onnxm::Tensor f;
        f.data_type = 9;
        f.dims = {1};
        f.data = {0};
        int fc = C.new_const(std::move(f));
        Step st;
        st.k = SK::BINOP;
        st.bop = BinOp::EQUAL;
        st.in = {ins[0], fc};
        auto os = shape_exprs(im, im.vals[ins[0]]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, st.in, st, {{9, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(9, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Cast ----
    if (op == "Cast") {
        int64_t to = n.attr_i("to", 0);
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) {
                reg_out(0, r);
                return;
            }
        }
        Step st;
        st.k = SK::CAST;
        st.in = ins;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{(int)to, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt((int)to, os);
        if (eff_dyn(im, ins[0])) im.vals[o].dyn = true;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "CastLike") {
        // Cast to the dtype of the second input (a type witness;
        // only the data input becomes a step input).
        int64_t to = dtype_of(im, ins[1]);
        Step st;
        st.k = SK::CAST;
        st.in = {ins[0]};
        auto os = shape_exprs(im, im.vals[ins[0]]);
        if (C.all_const({ins[0]})) {
            auto outs = fold_node(C, n, {ins[0]}, st, {{(int)to, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt((int)to, os);
        if (eff_dyn(im, ins[0])) im.vals[o].dyn = true;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- MatMul / Gemm ----
    if (op == "MatMul" || op == "Gemm") {
        Step st;
        st.in = ins;
        std::vector<EP> os;
        int odt = dtype_of(im, ins[0]);
        // A dyn operand's rshape may be rank-deficient; the
        // output shape then comes from dyn_shape_for at exec.
        bool dyn_in = eff_dyn(im, ins[0]) || (ins.size() > 1 && ins[1] >= 0 && eff_dyn(im, ins[1]));
        if (op == "MatMul") {
            st.k = SK::MATMUL;
            st.in = {ins[0], ins[1]};
            if (!dyn_in)
                os = matmul_shape(shape_exprs(im, im.vals[ins[0]]), shape_exprs(im, im.vals[ins[1]]));
        } else {
            st.k = SK::GEMM;
            st.alpha = n.attr_f("alpha", 1.0f);
            st.beta = n.attr_f("beta", 1.0f);
            st.transA = (int)n.attr_i("transA", 0);
            st.transB = (int)n.attr_i("transB", 0);
            if (!dyn_in) {
                auto as = shape_exprs(im, im.vals[ins[0]]);
                auto bs = shape_exprs(im, im.vals[ins[1]]);
                if (as.size() != 2 || bs.size() != 2) fail_op("Gemm rank != 2");
                os = {st.transA ? as[1] : as[0], st.transB ? bs[0] : bs[1]};
            }
            st.in = {ins[0], ins[1]};
            if (ins.size() > 2 && ins[2] >= 0) st.in.push_back(ins[2]);
        }
        if (!dyn_in && C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        if (dyn_in) im.vals[o].dyn = true;
        mark_b_f16(C, st);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Conv ----
    if (op == "Conv") {
        Step st;
        st.k = SK::CONV;
        st.in = ins;
        auto* at = n.attr("strides");
        if (at) st.c_strides = at->ints;
        at = n.attr("dilations");
        if (at) st.c_dil = at->ints;
        at = n.attr("pads");
        if (at) {
            // ONNX pads: [begin..., end...]; kernel takes begin per dim.
            size_t sp = at->ints.size() / 2;
            st.c_pads.assign(at->ints.begin(), at->ints.begin() + sp);
        }
        st.c_group = n.attr_i("group", 1);
        if (eff_dyn(im, ins[0])) {
            // Attr-derived geometry vectors keep the true
            // spatial count (from the pads/strides attrs);
            // dyn_shape_for sizes the output at exec.
            int o = C.new_rt(dtype_of(im, ins[0]), {});
            im.vals[o].dyn = true;
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        auto ws = shape_exprs(im, im.vals[ins[1]]);
        int sp = (int)xs.size() - 2;
        if ((int)st.c_strides.size() != sp) st.c_strides.assign(sp, 1);
        if ((int)st.c_dil.size() != sp) st.c_dil.assign(sp, 1);
        if ((int)st.c_pads.size() != sp) st.c_pads.assign(sp, 0);
        std::vector<EP> os = {xs[0], ws[0]};
        for (int i = 0; i < sp; i++) {
            // out = floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1
            EP num = eadd(xs[2 + i], ec(2 * st.c_pads[i] - 1));
            num = esub(num, emul(ec(st.c_dil[i]), esub(ws[2 + i], ec(1))));
            os.push_back(eadd(ediv(num, ec(st.c_strides[i])), ec(1)));
        }
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- ConvTranspose (fp32; phase 2 for the baked vits graph) ----
    if (op == "ConvTranspose") {
        Step st;
        st.k = SK::CONVT;
        st.in = ins;
        st.c_group = n.attr_i("group", 1);
        if (st.c_group != 1) fail_op("ConvTranspose: group != 1 not implemented");
        auto* at = n.attr("strides");
        if (at) st.c_strides = at->ints;
        at = n.attr("dilations");
        if (at) st.c_dil = at->ints;
        at = n.attr("pads");
        if (at) {
            size_t spn = at->ints.size() / 2;
            st.c_pads.assign(at->ints.begin(), at->ints.begin() + spn);
            st.c_pads_end.assign(at->ints.begin() + spn, at->ints.end());
        }
        at = n.attr("output_padding");
        if (at && !at->ints.empty()) st.ct_out_pad = at->ints[0];
        if (eff_dyn(im, ins[0])) {
            int o = C.new_rt(dtype_of(im, ins[0]), {});
            im.vals[o].dyn = true;
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        auto ws = shape_exprs(im, im.vals[ins[1]]);
        int sp = (int)xs.size() - 2;
        if (sp < 1) fail_op("ConvTranspose: rank < 3");
        if ((int)st.c_strides.size() != sp) st.c_strides.assign(sp, 1);
        if ((int)st.c_dil.size() != sp) st.c_dil.assign(sp, 1);
        if ((int)st.c_pads.size() != sp) st.c_pads.assign(sp, 0);
        if ((int)st.c_pads_end.size() != sp) st.c_pads_end.assign(sp, 0);
        // out = (in-1)*stride - pad_b - pad_e + dil*(k-1) + out_pad + 1;
        // weight shape is [Ci, O, K...], so out channels = ws[1].
        std::vector<EP> os = {xs[0], ws[1]};
        for (int i = 0; i < sp; i++) {
            EP t = emul(esub(xs[2 + i], ec(1)), ec(st.c_strides[i]));
            t = eadd(t, emul(ec(st.c_dil[i]), esub(ws[2 + i], ec(1))));
            t = eadd(t, ec(st.ct_out_pad + 1 - st.c_pads[i] - st.c_pads_end[i]));
            os.push_back(t);
        }
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- LayerNormalization / Softmax ----
    if (op == "LayerNormalization") {
        Step st;
        st.k = SK::LN;
        st.in = ins;
        st.axis = n.attr_i("axis", -1);
        st.eps = n.attr_f("epsilon", 1e-5f);
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        for (size_t i = 1; i < n.outputs.size(); i++) {
            // Mean / InvStd outputs: not consumed in these graphs; give
            // them a degenerate value so name lookup succeeds.
            std::vector<EP> ds = os;
            int64_t ax = norm_axis(st.axis, ds.size());
            for (size_t d = (size_t)ax; d < ds.size(); d++) ds[d] = ec(1);
            reg_out((int)i, C.new_rt(odt, ds));
            im.vals.back().def_step = -2;  // mark: no producer (never materialized unless used)
        }
        return;
    }
    if (op == "InstanceNormalization") {
        Step st;
        st.k = SK::INSTNORM;
        st.in = ins;
        st.eps = n.attr_f("epsilon", 1e-5f);
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "Softmax") {
        Step st;
        st.k = SK::SOFTMAX;
        st.in = ins;
        st.axis = n.attr_i("axis", -1);
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- ReduceSum / ReduceMean / ArgMax / TopK / CumSum ----
    // ReduceMean (phase 2) shares the ReduceSum path with a mean flag;
    // its noop_with_empty_axes form is an identity copy.
    if (op == "ReduceMean" && n.attr_i("noop_with_empty_axes", 0) != 0) {
        std::vector<int64_t> axes;
        bool have_axes = false;
        if (ins.size() > 1 && ins[1] >= 0) have_axes = value_const_ints(im, ins[1], axes);
        else if (auto* at = n.attr("axes")) { axes = at->ints; have_axes = true; }
        if (have_axes && axes.empty()) {
            Step st;
            st.k = SK::COPY;
            st.in = {ins[0]};
            auto os = shape_exprs(im, im.vals[ins[0]]);
            int odt = dtype_of(im, ins[0]);
            if (C.all_const(st.in)) {
                auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
                reg_out(0, outs[0]);
                return;
            }
            int o = C.new_rt(odt, os);
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
    }
    if (op == "ReduceSum" || op == "ReduceMean" || op == "ReduceL2") {
        Step st;
        st.k = SK::REDUCESUM;
        st.mean = op == "ReduceMean";
        st.l2 = op == "ReduceL2";
        st.in = {ins[0]};
        st.keepdims = n.attr_i("keepdims", 1) != 0;
        std::vector<int64_t> axes;
        bool have_axes = false;
        if (ins.size() > 1 && ins[1] >= 0) {
            have_axes = value_const_ints(im, ins[1], axes);
            if (!have_axes) fail_op("Reduce axes not const");
        } else if (auto* at = n.attr("axes")) {
            axes = at->ints;
            have_axes = true;
        }
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        if (!have_axes || axes.empty()) {
            axes.clear();
            for (size_t i = 0; i < xs.size(); i++) axes.push_back((int64_t)i);
        }
        for (auto& a : axes) a = norm_axis(a, xs.size());
        st.axes = axes;
        // A dyn input's symbolic shape may be rank-deficient
        // (its true rank only exists at execution); the axes
        // stay as lowered and the output becomes dyn instead
        // of indexing past the shape vector.
        bool shape_bad = eff_dyn(im, ins[0]);
        for (auto a : axes)
            if (a < 0 || a >= (int64_t)xs.size()) shape_bad = true;
        std::vector<EP> os;
        if (!shape_bad) {
            std::vector<char> red(xs.size(), 0);
            for (auto a : axes) red[a] = 1;
            for (size_t i = 0; i < xs.size(); i++) {
                if (red[i]) {
                    if (st.keepdims) os.push_back(ec(1));
                } else
                    os.push_back(xs[i]);
            }
        }
        int odt = dtype_of(im, ins[0]);
        if (!shape_bad && C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        if (shape_bad) im.vals[o].dyn = true;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "ArgMax") {
        Step st;
        st.k = SK::ARGMAX;
        st.in = {ins[0]};
        st.axis = n.attr_i("axis", 0);
        st.keepdims = n.attr_i("keepdims", 1) != 0;
        st.exclusive = n.attr_i("select_last_index", 0) != 0;  // field reuse
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(st.axis, xs.size());
        std::vector<EP> os;
        for (size_t i = 0; i < xs.size(); i++) {
            if ((int64_t)i == ax) {
                if (st.keepdims) os.push_back(ec(1));
            } else
                os.push_back(xs[i]);
        }
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{7, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(7, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "TopK") {
        std::vector<int64_t> kv;
        if (!value_const_ints(im, ins[1], kv) || kv.empty()) fail_op("TopK k not const");
        Step st;
        st.k = SK::TOPK;
        st.in = {ins[0]};
        st.topk = kv[0];
        st.axis = n.attr_i("axis", -1);
        st.largest = n.attr_i("largest", 1) != 0;
        st.sorted = n.attr_i("sorted", 1) != 0;
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(st.axis, xs.size());
        std::vector<EP> os = xs;
        os[ax] = ec(st.topk);
        int odt = dtype_of(im, ins[0]);
        int o1 = C.new_rt(odt, os);
        int o2 = C.new_rt(7, os);
        C.add_step(st, {o1, o2});
        reg_out(0, o1);
        reg_out(1, o2);
        return;
    }
    if (op == "CumSum") {
        std::vector<int64_t> ax;
        if (!value_const_ints(im, ins[1], ax) || ax.empty()) fail_op("CumSum axis not const");
        Step st;
        st.k = SK::CUMSUM;
        st.in = {ins[0]};
        st.axis = ax[0];
        st.exclusive = n.attr_i("exclusive", 0) != 0;
        st.reverse = n.attr_i("reverse", 0) != 0;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Gather family ----
    if (op == "Gather") {
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) {
                reg_out(0, r);
                return;
            }
        }
        Step st;
        st.k = SK::GATHER;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        auto ds = shape_exprs(im, im.vals[ins[0]]);
        auto is = shape_exprs(im, im.vals[ins[1]]);
        int64_t ax = norm_axis(st.axis, ds.size());
        std::vector<EP> os;
        for (int64_t i = 0; i < ax; i++) os.push_back(ds[i]);
        for (auto& e : is) os.push_back(e);
        for (size_t i = (size_t)ax + 1; i < ds.size(); i++) os.push_back(ds[i]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "GatherElements") {
        Step st;
        st.k = SK::GATHER_EL;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        auto os = shape_exprs(im, im.vals[ins[1]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "ScatterElements") {
        Step st;
        st.k = SK::SCATTER_EL;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        std::string red = n.attr("reduction") ? n.attr("reduction")->s : "none";
        st.reduction = red == "add" ? 1 : red == "mul" ? 2 : 0;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Concat ----
    if (op == "Concat") {
        if (C.all_sym_or_const(ins)) {
            int r = sym_structural(C, n, ins);
            if (std::getenv("FE_DEBUG") && n.name.find("attn_layers.0") != std::string::npos)
                std::fprintf(stderr, "[concat] %s sym=%d kinds=%d,%d,%d,%d\n", n.name.c_str(), r,
                             (int)im.vals[ins[0]].kind, (int)im.vals[ins[1]].kind,
                             ins.size() > 2 ? (int)im.vals[ins[2]].kind : -9,
                             ins.size() > 3 ? (int)im.vals[ins[3]].kind : -9);
            if (r >= 0) {
                reg_out(0, r);
                return;
            }
        }
        Step st;
        st.k = SK::CONCAT;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        // Template shape: the first input with a known rank (a
        // dyn input, e.g. a ConcatFromSequence result whose
        // element shape is unknown, may carry an empty shape).
        // When every input's shape is unknown the output is
        // dyn with no shape at all; the exec dyn branch derives
        // it from the true descriptors and normalizes the raw
        // axis there.
        std::vector<EP> os;
        for (int v : ins) {
            auto s = shape_exprs(im, im.vals[v]);
            if (!s.empty()) { os = s; break; }
        }
        bool any_dyn = os.empty();
        for (int v : ins)
            if (eff_dyn(im, v)) any_dyn = true;
        if (!os.empty()) {
            int64_t ax = norm_axis(st.axis, (int64_t)os.size());
            EP sum = ec(0);
            for (int v : ins) {
                auto s = shape_exprs(im, im.vals[v]);
                if ((int64_t)s.size() <= ax) { any_dyn = true; continue; }
                sum = eadd(sum, s[ax]);
            }
            if (!any_dyn) os[ax] = sum;
            else os.clear();
        }
        int odt = dtype_of(im, ins[0]);
        if (!any_dyn && C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        if (any_dyn) im.vals[o].dyn = true;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- ConstantOfShape / RandomNormalLike ----
    if (op == "ConstantOfShape") {
        std::vector<EP> elems;
        bool sym = value_sym_elems(im, ins[0], elems);
        Step st;
        st.k = SK::CONSTSHAPE;
        st.in = {ins[0]};
        auto* at = n.attr("value");
        if (at && at->t) st.fill = *at->t;
        else {
            st.fill.data_type = 1;
            st.fill.dims = {1};
            st.fill.data.assign(4, 0);
        }
        if (!sym) {
            // Runtime shape spec (a tensor produced at
            // execution, e.g. by a SHAPE step over a dyn
            // tensor): output dims exist only at execution;
            // dyn_shape_for reads them from the input there.
            int o = C.new_rt(st.fill.data_type, {});
            im.vals[o].dyn = true;
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        // Constant spec -> constant tensor: fold it (phase 2; the vits
        // attention pads chains route through a ConstantOfShape whose
        // spec is a fixed [0], and the RT step would poison the
        // symbolic Concat that follows). Symbolic specs keep the step.
        {
            std::vector<int64_t> dims;
            bool allc = true;
            for (auto& e : elems) {
                int64_t c;
                if (!econst(e, c) || c < 0) { allc = false; break; }
                dims.push_back(c);
            }
            int64_t nel = 1;
            for (auto d : dims) nel *= d;
            if (allc && nel <= (1 << 20)) {
                onnxm::Tensor t;
                t.data_type = st.fill.data_type;
                t.dims = dims;
                size_t es = onnxm::elem_size(t.data_type);
                t.data.resize((size_t)nel * es);
                size_t fbytes = st.fill.data.size();
                if (fbytes != es) fail_op("ConstantOfShape fill size mismatch");
                for (int64_t i = 0; i < nel; i++) memcpy(t.data.data() + (size_t)i * es, st.fill.data.data(), es);
                reg_out(0, C.new_const(std::move(t)));
                return;
            }
        }
        int odt = st.fill.data_type;
        int o = C.new_rt(odt, elems);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "RandomNormalLike") {
        Step st;
        st.k = SK::RANDNORM;
        st.out_dtype = (int)n.attr_i("dtype", dtype_of(im, ins[0]));
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int o = C.new_rt(st.out_dtype, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Where ----
    if (op == "Where") {
        if (C.all_sym_or_const(ins)) {
            int r = sym_where(im, ins[0], ins[1], ins[2]);
            if (r >= 0) {
                reg_out(0, maybe_const_collapse(im, r));
                return;
            }
        }
        auto os = broadcast_shape(shape_exprs(im, im.vals[ins[1]]), shape_exprs(im, im.vals[ins[2]]));
        os = broadcast_shape(shape_exprs(im, im.vals[ins[0]]), os);
        Step st;
        st.k = SK::WHERE;
        st.in = ins;
        int odt = dtype_of(im, ins[1]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- layout ops (views) ----
    if (op == "Transpose") {
        std::vector<int64_t> perm;
        if (auto* at = n.attr("perm")) perm = at->ints;
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            auto is = shape_exprs(im, im.vals[ins[0]]);
            if (perm.empty()) { perm.resize(is.size()); for (size_t i = 0; i < is.size(); i++) perm[i] = (int64_t)is.size() - 1 - i; }
            reg_out(0, C.new_const(permute_copy(im.vals[ins[0]].ct, perm)));
            return;
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        if (eff_dyn(im, ins[0])) {
            // Keep the perm raw (empty = reverse-all); the
            // true rank only exists at execution and
            // resolve_view normalizes against it.
            Value v;
            v.kind = Value::RT;
            v.dtype = dtype_of(im, ins[0]);
            v.dyn = true;
            v.view_kind = 1;
            v.base = ins[0];
            v.vperm = perm;
            im.vals.push_back(std::move(v));
            reg_out(0, (int)im.vals.size() - 1);
            return;
        }
        if (perm.empty()) { perm.resize(is.size()); for (size_t i = 0; i < is.size(); i++) perm[i] = (int64_t)is.size() - 1 - i; }
        std::vector<EP> os;
        for (auto p : perm) os.push_back(is[p]);
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = 1;
        v.base = ins[0];
        v.vperm = perm;
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }
    if (op == "Reshape") {
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        std::vector<EP> target;
        if (!value_sym_elems(im, ins[1], target)) {
            // Runtime target tensor (produced by a step, e.g. an
            // If branch in the SV index loop): a dyn view;
            // resolve_view recomputes the shape from the
            // target's values whenever it re-resolves.
            if (im.vals[ins[1]].kind != Value::RT) fail_op("Reshape target not symbolic/const");
            Value v;
            v.kind = Value::RT;
            v.dtype = dtype_of(im, ins[0]);
            v.dyn = true;
            v.view_kind = 2;
            v.base = ins[0];
            v.meta0 = ins[1];
            im.vals.push_back(std::move(v));
            reg_out(0, (int)im.vals.size() - 1);
            return;
        }
        if (eff_dyn(im, ins[0])) {
            // Symbolic target but a dyn base: the 0-copy and
            // -1-infer entries would index a rank-deficient
            // shape vector. The dyn view path recomputes the
            // shape from the target's values at exec; the
            // target value (const or symbolic) is materialized
            // in res like any meta tensor.
            Value v;
            v.kind = Value::RT;
            v.dtype = dtype_of(im, ins[0]);
            v.dyn = true;
            v.view_kind = 2;
            v.base = ins[0];
            v.meta0 = ins[1];
            im.vals.push_back(std::move(v));
            reg_out(0, (int)im.vals.size() - 1);
            return;
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        std::vector<EP> os;
        int infer = -1;
        for (size_t i = 0; i < target.size(); i++) {
            int64_t c;
            if (econst(target[i], c) && c == -1) {
                infer = (int)i;
                os.push_back(ec(-1));  // placeholder, replaced below
            } else if (econst(target[i], c) && c == 0) {
                os.push_back(is[i]);
            } else
                os.push_back(target[i]);
        }
        if (infer >= 0) {
            EP prod = ec(1);
            for (size_t i = 0; i < os.size(); i++)
                if ((int)i != infer) prod = emul(prod, os[i]);
            os[infer] = ediv(enumel(is), prod);
        }
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            std::vector<int64_t> dims;
            for (auto& e : os) {
                int64_t c;
                if (!econst(e, c)) fail_op("Reshape of const with symbolic shape");
                dims.push_back(c);
            }
            onnxm::Tensor t = im.vals[ins[0]].ct;
            t.dims = dims;
            reg_out(0, C.new_const(std::move(t)));
            return;
        }
        // A reshape is a pure view only when its base is a materialized
        // (row-major contiguous) tensor. On a view or an input the data
        // would have to be copied, and copying cannot happen at bind
        // time (the producer step has not run yet) -- emit a COPY step
        // with reshape (logical-order) semantics instead.
        Value& bv = im.vals[ins[0]];
        bool view_ok = bv.kind == Value::RT && bv.view_kind == 0 && !bv.is_input;
        if (!view_ok) {
            Step st;
            st.k = SK::COPY;
            st.flat_copy = true;
            st.in = {ins[0]};
            int o = C.new_rt(dtype_of(im, ins[0]), os);
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = 2;
        v.base = ins[0];
        v.meta0 = ins[1];
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }
    if (op == "Unsqueeze" || op == "Squeeze") {
        std::vector<int64_t> axes;
        bool have_axes = false;
        if (ins.size() > 1 && ins[1] >= 0) have_axes = value_const_ints(im, ins[1], axes);
        else if (auto* at = n.attr("axes")) { axes = at->ints; have_axes = true; }
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        std::vector<EP> os;
        std::vector<int64_t> naxes;
        bool in_dyn = eff_dyn(im, ins[0]);
        if (op == "Unsqueeze") {
            if (!have_axes) fail_op("Unsqueeze axes unknown");
            if (in_dyn) {
                // The true rank only exists at execution; keep
                // the axes raw (negatives included) and let
                // resolve_view normalize them against the
                // base's true rank. The shape stays unknown.
                naxes = axes;
            } else {
                int64_t nr = (int64_t)is.size() + (int64_t)axes.size();
                std::vector<char> at(nr, 0);
                for (auto a : axes) {
                    int64_t na = norm_axis(a, nr);
                    at[na] = 1;
                    naxes.push_back(na);
                }
                size_t src = 0;
                for (int64_t i = 0; i < nr; i++) os.push_back(at[i] ? ec(1) : is[src++]);
            }
        } else if (in_dyn) {
            // Squeeze of a dyn tensor: with axes, keep them raw
            // for exec-time normalization; without axes, the
            // {-1} sentinel tells resolve_view to drop every
            // size-1 dim of the true shape.
            naxes = have_axes ? axes : std::vector<int64_t>{-1};
        } else {
            std::vector<char> drop(is.size(), 0);
            if (have_axes) {
                for (auto a : axes) {
                    int64_t na = norm_axis(a, (int64_t)is.size());
                    drop[na] = 1;
                    naxes.push_back(na);
                }
            } else {
                for (size_t i = 0; i < is.size(); i++) {
                    int64_t c;
                    if (econst(is[i], c) && c == 1) { drop[i] = 1; naxes.push_back((int64_t)i); }
                }
            }
            for (size_t i = 0; i < is.size(); i++)
                if (!drop[i]) os.push_back(is[i]);
        }
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            onnxm::Tensor t = im.vals[ins[0]].ct;
            t.dims.clear();
            for (auto& e : os) {
                int64_t c;
                if (!econst(e, c)) fail_op("layout of const with symbolic shape");
                t.dims.push_back(c);
            }
            reg_out(0, C.new_const(std::move(t)));
            return;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = op == "Unsqueeze" ? 3 : 4;
        v.base = ins[0];
        v.vaxes = naxes;
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }
    if (op == "Slice") {
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        // Resolve axes at compile time (must be const); starts/ends/steps
        // may stay symbolic for the view, but shape exprs need them.
        std::vector<int64_t> axes;
        if (ins.size() > 3 && ins[3] >= 0) {
            if (!value_const_ints(im, ins[3], axes)) fail_op("Slice axes not const");
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        int64_t rank = (int64_t)is.size();
        std::vector<EP> sev, eev, pev;
        bool sym_ok = value_sym_elems(im, ins[1], sev) && value_sym_elems(im, ins[2], eev);
        if (sym_ok && ins.size() > 4 && ins[4] >= 0)
            sym_ok = value_sym_elems(im, ins[4], pev);
        if (!sym_ok || eff_dyn(im, ins[0])) {
            // A dyn input's symbolic shape may be rank-deficient;
            // SLICE_RT resolves against the true descriptor and
            // normalizes the raw axes at execution.
            Value v;
            v.kind = Value::RT;
            v.dtype = dtype_of(im, ins[0]);
            v.dyn = true;
            im.vals.push_back(std::move(v));
            int o = (int)im.vals.size() - 1;
            Step st;
            st.k = SK::SLICE_RT;
            st.in = ins;
            st.axes = axes;
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        if (axes.empty()) { axes.resize(rank); for (int64_t i = 0; i < rank; i++) axes[i] = i; }
        for (auto& a : axes) a = norm_axis(a, rank);
        if (pev.empty()) pev.assign(axes.size(), ec(1));
        std::vector<EP> os = is;
        for (size_t i = 0; i < axes.size(); i++) {
            int64_t ax = axes[i];
            EP start = sev[i], end = eev[i];
            int64_t cs, ce, cp;
            bool ks = econst(start, cs), ke = econst(end, ce);
            int64_t step = econst(pev[i], cp) ? cp : 1;
            if (step == 0) fail_op("Slice step 0");
            int64_t dc;
            if (ks && ke && econst(is[ax], dc)) {
                int64_t first;
                os[ax] = ec(slice_count(dc, cs, ce, step, first));
                continue;
            }
            // Symbolic dim: build the length from clamped bound exprs.
            // Bounds beyond +-2^29 are sentinels (real dims in these
            // graphs never reach 2^29; the phase-1 code used the same
            // threshold for large positive ends) and act as infinities;
            // ordinary negative bounds count from the end (D + v), the
            // same normalization slice_count applies numerically.
            const int64_t INF = (int64_t)1 << 29;
            EP sE = start, eE = end;
            if (ks) {
                if (cs <= -INF) sE = step > 0 ? ec(0) : esub(is[ax], ec(1));
                else if (cs >= INF) sE = step > 0 ? is[ax] : esub(is[ax], ec(1));
                else if (cs < 0) sE = eadd(is[ax], ec(cs));
                else sE = ec(cs);
            }
            if (ke) {
                if (ce <= -INF) eE = step > 0 ? ec(0) : ec(-1);
                else if (ce >= INF) eE = step > 0 ? is[ax] : esub(is[ax], ec(1));
                else if (ce < 0) eE = eadd(is[ax], ec(ce));
                else eE = ec(ce);
            }
            if (step > 0) os[ax] = ediv(eadd(esub(eE, sE), ec(step - 1)), ec(step));
            else os[ax] = ediv(eadd(esub(sE, eE), ec(-step - 1)), ec(-step));
        }
        // A const base with all-const bounds folds at compile time.
        // A const base with a symbolic bound (RoBERTa slices a
        // constant by a runtime length) falls through to the view
        // path: consts are materialized as RTs at bind, so the
        // view resolves against the const buffer like any base.
        bool bounds_const = true;
        for (auto& e : sev) { int64_t c; if (!econst(e, c)) bounds_const = false; }
        for (auto& e : eev) { int64_t c; if (!econst(e, c)) bounds_const = false; }
        for (auto& e : pev) { int64_t c; if (!econst(e, c)) bounds_const = false; }
        if (im.vals[ins[0]].kind == Value::CONSTV && bounds_const) {
            // Materialize via a view resolution on the const buffer.
            RT base = tensor_of_const(im.vals[ins[0]].ct);
            std::vector<int64_t> s_ints, e_ints, p_ints(axes.size(), 1);
            for (auto& e : sev) { int64_t c; if (!econst(e, c)) fail_op("const slice symbolic start"); s_ints.push_back(c); }
            for (auto& e : eev) { int64_t c; if (!econst(e, c)) fail_op("const slice symbolic end"); e_ints.push_back(c); }
            for (size_t i = 0; i < pev.size(); i++) { int64_t c; if (!econst(pev[i], c)) fail_op("const slice symbolic step"); p_ints[i] = c; }
            RT view = base;
            int64_t off = 0;
            for (size_t i = 0; i < axes.size(); i++) {
                int64_t ax = axes[i];
                int64_t first;
                view.shape[ax] = slice_count(base.shape[ax], s_ints[i], e_ints[i], p_ints[i], first);
                off += first * base.strides[ax];
                view.strides[ax] *= p_ints[i];
            }
            view.p = base.p + off * (int64_t)onnxm::elem_size(base.dtype);
            onnxm::Tensor t;
            t.data_type = base.dtype;
            t.dims = view.shape;
            t.data.resize((size_t)view.numel() * onnxm::elem_size(base.dtype));
            RT dst;
            dst.dtype = base.dtype;
            dst.p = t.data.data();
            dst.shape = t.dims;
            dst.strides = rowmajor(t.dims);
            copy_tensor(dst, view);
            reg_out(0, C.new_const(std::move(t)));
            return;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = 5;
        v.base = ins[0];
        v.meta0 = ins[1];
        v.meta1 = ins[2];
        v.meta2 = ins.size() > 3 ? ins[3] : -1;
        v.meta3 = ins.size() > 4 ? ins[4] : -1;
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }

    // ---- Pad (phase 2) ----
    if (op == "Pad") {
        std::string mode = "constant";
        if (auto* at = n.attr("mode")) mode = at->s;
        Step st;
        st.k = SK::PAD;
        if (mode == "constant") st.pad_mode = 0;
        else if (mode == "reflect") st.pad_mode = 1;
        else if (mode == "edge") st.pad_mode = 2;
        else fail_op("Pad mode not implemented: " + mode);
        int pads_vid = -1;
        if (ins.size() > 1 && ins[1] >= 0) pads_vid = ins[1];
        else if (auto* at = n.attr("pads")) {
            // Pre-opset-11 form: pads as an attribute.
            onnxm::Tensor t;
            t.data_type = 7;
            t.dims = {(int64_t)at->ints.size()};
            t.data.resize(at->ints.size() * 8);
            memcpy(t.data.data(), at->ints.data(), at->ints.size() * 8);
            pads_vid = C.new_const(std::move(t));
        }
        if (pads_vid < 0) fail_op("Pad: no pads input or attr");
        std::vector<EP> pads_e;
        if (!value_sym_elems(im, pads_vid, pads_e)) fail_op("Pad pads not symbolic/const");
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        st.in = {ins[0], pads_vid};
        if (ins.size() > 2 && ins[2] >= 0) st.in.push_back(ins[2]);
        if (eff_dyn(im, ins[0])) {
            // Rank is only known at execution; dyn_shape_for
            // sizes the output from the true input shape and
            // the pads descriptor.
            Value v;
            v.kind = Value::RT;
            v.dtype = odt;
            v.dyn = true;
            im.vals.push_back(std::move(v));
            int o = (int)im.vals.size() - 1;
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        int64_t r = (int64_t)xs.size();
        if ((int64_t)pads_e.size() != 2 * r) fail_op("Pad: pads length != 2*rank");
        std::vector<EP> os = xs;
        for (int64_t d = 0; d < r; d++) os[d] = eadd(eadd(xs[d], pads_e[d]), pads_e[r + d]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Split (phase 2): outputs are views into the input ----
    if (op == "Split") {
        std::vector<int64_t> sizes;
        if (ins.size() > 1 && ins[1] >= 0) {
            if (!value_const_ints(im, ins[1], sizes)) fail_op("Split sizes not const");
        } else if (auto* at = n.attr("split")) sizes = at->ints;
        if (eff_dyn(im, ins[0])) {
            // Dyn base: the views carry their split geometry in
            // meta fields and resolve_view computes the window
            // from the base's true shape. meta3 = 1: const
            // sizes (meta1 = offset, meta2 = length). meta3 =
            // 2: equal split (meta1 = output index, meta2 =
            // output count). meta0 keeps the raw axis.
            int odt = dtype_of(im, ins[0]);
            int64_t raw_ax = n.attr_i("axis", 0);
            if (!sizes.empty()) {
                if (sizes.size() != n.outputs.size()) fail_op("Split: sizes/outputs mismatch");
                int64_t off = 0;
                for (size_t i = 0; i < sizes.size(); i++) {
                    Value v;
                    v.kind = Value::RT;
                    v.dtype = odt;
                    v.dyn = true;
                    v.view_kind = 7;
                    v.base = ins[0];
                    v.meta0 = (int)raw_ax;
                    v.meta1 = (int)off;
                    v.meta2 = (int)sizes[i];
                    v.meta3 = 1;
                    im.vals.push_back(std::move(v));
                    reg_out((int)i, (int)im.vals.size() - 1);
                    off += sizes[i];
                }
            } else {
                int64_t cnt = (int64_t)n.outputs.size();
                if (cnt == 0) fail_op("Split: no outputs");
                for (int64_t i = 0; i < cnt; i++) {
                    Value v;
                    v.kind = Value::RT;
                    v.dtype = odt;
                    v.dyn = true;
                    v.view_kind = 7;
                    v.base = ins[0];
                    v.meta0 = (int)raw_ax;
                    v.meta1 = (int)i;
                    v.meta2 = (int)cnt;
                    v.meta3 = 2;
                    im.vals.push_back(std::move(v));
                    reg_out((int)i, (int)im.vals.size() - 1);
                }
            }
            return;
        }
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(n.attr_i("axis", 0), xs.size());
        if (sizes.empty()) {
            // Equal split: input dim must be a compile-time constant.
            int64_t dim;
            if (!econst(xs[ax], dim)) fail_op("Split: equal split of symbolic dim");
            int64_t cnt = (int64_t)n.outputs.size();
            if (cnt == 0 || dim % cnt != 0) fail_op("Split: dim not divisible");
            sizes.assign(cnt, dim / cnt);
        }
        if (sizes.size() != n.outputs.size()) fail_op("Split: sizes/outputs mismatch");
        int odt = dtype_of(im, ins[0]);
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            RT base = tensor_of_const(im.vals[ins[0]].ct);
            size_t es = onnxm::elem_size(base.dtype);
            int64_t off = 0;
            for (size_t i = 0; i < sizes.size(); i++) {
                RT view = base;
                view.shape[ax] = sizes[i];
                view.p = base.p + (size_t)off * base.strides[ax] * es;
                onnxm::Tensor t;
                t.data_type = base.dtype;
                t.dims = view.shape;
                t.data.resize((size_t)view.numel() * es);
                RT dst;
                dst.dtype = base.dtype;
                dst.p = t.data.data();
                dst.shape = t.dims;
                dst.strides = rowmajor(t.dims);
                copy_tensor(dst, view);
                reg_out((int)i, C.new_const(std::move(t)));
                off += sizes[i];
            }
            return;
        }
        int64_t off = 0;
        for (size_t i = 0; i < sizes.size(); i++) {
            std::vector<EP> os = xs;
            os[ax] = ec(sizes[i]);
            Value v;
            v.kind = Value::RT;
            v.dtype = odt;
            v.rshape = os;
            v.view_kind = 7;  // split view
            v.base = ins[0];
            v.meta0 = (int)ax;
            v.meta1 = (int)off;
            im.vals.push_back(std::move(v));
            reg_out((int)i, (int)im.vals.size() - 1);
            off += sizes[i];
        }
        return;
    }

    // ---- Expand (phase 2): broadcast copy to the target shape ----
    if (op == "Expand") {
        std::vector<EP> target;
        if (!value_sym_elems(im, ins[1], target)) {
            // Runtime target (a Loop body computing its expand
            // shape from a SequenceAt value): the output is dyn;
            // its rank is the target tensor's static numel, and
            // exec sizes the buffer from the target's values.
            const Value& tv = im.vals[ins[1]];
            int64_t rank = 1;
            bool ok = tv.kind == Value::RT;
            if (ok)
                for (auto& e : tv.rshape) {
                    int64_t c;
                    if (!econst(e, c)) { ok = false; break; }
                    rank *= c;
                }
            if (!ok) fail_op("Expand target not symbolic/const");
            Value v;
            v.kind = Value::RT;
            v.dtype = dtype_of(im, ins[0]);
            v.dyn = true;
            for (int64_t i = 0; i < rank; i++) {
                v.rshape.push_back(esym(im.nsyms++));
                im.sym_origin.push_back({-1, -1});
            }
            im.vals.push_back(std::move(v));
            int o = (int)im.vals.size() - 1;
            Step st;
            st.k = SK::EXPAND;
            st.in = {ins[0], ins[1]};
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        auto os = broadcast_shape(shape_exprs(im, im.vals[ins[0]]), target);
        Step st;
        st.k = SK::EXPAND;
        st.in = {ins[0]};
        // A fully-constant target is also stored on the step:
        // when the data is dyn-shaped at runtime (its shape
        // exprs may contain unbindable placeholder symbols
        // from a runtime-target Expand upstream), the exec
        // case recomputes the output shape by broadcasting
        // the data's true shape with these dims.
        {
            bool allc = true;
            for (auto& e : target) {
                int64_t c;
                if (!econst(e, c)) { allc = false; break; }
                st.axes.push_back(c);
            }
            if (!allc) st.axes.clear();
        }
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- sequences and Loop (runtime reference extraction) ----
    if (op == "SequenceEmpty") {
        Value v;
        v.kind = Value::RT;
        v.is_seq = true;
        v.dtype = (int)n.attr_i("dtype", 1);
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::SEQ_EMPTY;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "SplitToSequence") {
        if (ins.size() < 2 || ins[1] < 0) fail_op("SplitToSequence without split input");
        auto is = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(n.attr_i("axis", 0), (int64_t)is.size());
        bool keepdims = n.attr_i("keepdims", 1) != 0;
        Value v;
        v.kind = Value::RT;
        v.is_seq = true;
        v.dtype = dtype_of(im, ins[0]);
        v.eshape = is;
        // Splits built by a ConstantOfShape of 1s (the
        // split-into-singles idiom in the BERT tail and the SV
        // frame batching) give every element an exact axis length
        // of 1, keeping downstream shapes static. Any other split
        // source leaves the axis dim a placeholder that evaluates
        // to 0; tensors read back out are dyn and get their true
        // shapes at execution.
        bool ones = false;
        {
            auto elem0 = [](const onnxm::Tensor& t) {
                double d = 0;
                if (t.data_type == 7) { int64_t x; memcpy(&x, t.data.data(), 8); d = (double)x; }
                else if (t.data_type == 6) { int32_t x; memcpy(&x, t.data.data(), 4); d = (double)x; }
                else if (t.data_type == 1) { float x; memcpy(&x, t.data.data(), 4); d = (double)x; }
                else if (t.data_type == 11) { memcpy(&d, t.data.data(), 8); }
                return d;
            };
            const Value& sv = im.vals[ins[1]];
            if (sv.kind == Value::CONSTV && sv.ct.numel() > 0) {
                ones = true;
                for (int64_t i = 0; i < sv.ct.numel(); i++) {
                    onnxm::Tensor one;
                    one.data_type = sv.ct.data_type;
                    one.data.assign(sv.ct.data.begin() + i * (int64_t)onnxm::elem_size(sv.ct.data_type),
                                    sv.ct.data.begin() + (i + 1) * (int64_t)onnxm::elem_size(sv.ct.data_type));
                    if (elem0(one) != 1.0) { ones = false; break; }
                }
            } else if (sv.kind == Value::RT && sv.def_step >= 0) {
                const Step& ds = im.steps[sv.def_step];
                if (ds.k == SK::CONSTSHAPE && ds.fill.numel() == 1 && elem0(ds.fill) == 1.0) ones = true;
            }
        }
        if (ones) v.eshape[ax] = ec(1);
        else {
            v.eshape[ax] = esym(im.nsyms++);
            im.sym_origin.push_back({-1, -1});
        }
        if (!keepdims) v.eshape.erase(v.eshape.begin() + ax);
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::SEQ_SPLIT;
        st.axis = ax;
        st.in = ins;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "SequenceAt") {
        Value& S = im.vals[ins[0]];
        if (!S.is_seq) fail_op("SequenceAt on a non-sequence");
        Value v;
        v.kind = Value::RT;
        v.dtype = S.dtype;
        v.dyn = true;
        v.rshape = S.eshape;
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::SEQ_AT;
        st.in = ins;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "SequenceInsert") {
        Value& S = im.vals[ins[0]];
        if (!S.is_seq) fail_op("SequenceInsert on a non-sequence");
        Value v;
        v.kind = Value::RT;
        v.is_seq = true;
        v.dtype = S.dtype;
        v.eshape = shape_exprs(im, im.vals[ins[1]]);
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::SEQ_INSERT;
        st.in = ins;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "ConcatFromSequence") {
        Value& S = im.vals[ins[0]];
        if (!S.is_seq) fail_op("ConcatFromSequence on a non-sequence");
        Value v;
        v.kind = Value::RT;
        v.dtype = S.dtype;
        v.dyn = true;
        v.rshape = S.eshape;
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::SEQ_CONCAT;
        st.axis = n.attr_i("axis", 0);
        st.in = ins;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "If") {
        lower_if(C, n, ins);
        return;
    }
    if (op == "Range") {
        // start / limit / delta are scalars, const or runtime; the
        // element count is only known at exec time, so the output
        // is a dyn tensor filled by a RANGE step.
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.dyn = true;
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::RANGE;
        st.in = ins;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "STFT") {
        // ORT's input order, pinned by a bare-graph probe against
        // numpy: signal, frame_step, window, frame_length. (The
        // ONNX spec lists frame_length second; ORT reads the
        // fourth input as frame_length and rejects the graph
        // when it differs from the window length. Genie's
        // export follows ORT.)
        std::vector<int64_t> flv, fsv;
        if (ins.size() > 3 && ins[3] >= 0) {
            if (!value_const_ints(im, ins[3], flv) || flv.size() != 1) fail_op("STFT frame_length not const");
        }
        if (ins.size() > 1 && ins[1] >= 0) {
            if (!value_const_ints(im, ins[1], fsv) || fsv.size() != 1) fail_op("STFT frame_step not const");
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        if (is.size() < 2) fail_op("STFT signal rank < 2");
        int64_t fl = flv.empty() ? -1 : flv[0];
        if (fl < 0 && ins.size() > 2 && ins[2] >= 0) {
            auto ws = shape_exprs(im, im.vals[ins[2]]);
            int64_t c;
            if (ws.size() == 1 && econst(ws[0], c)) fl = c;
        }
        if (fl < 0) fail_op("STFT frame_length unknown");
        int64_t fs = fsv.empty() ? fl : fsv[0];
        int64_t onesided = n.attr_i("onesided", 1);
        int64_t bins = onesided ? fl / 2 + 1 : fl;
        EP frames = eadd(ediv(esub(is[1], ec(fl)), ec(fs)), ec(1));
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = {is[0], frames, ec(bins), ec(2)};
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::STFT;
        st.in = ins;
        st.axis = onesided;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "Tile") {
        auto is = shape_exprs(im, im.vals[ins[0]]);
        std::vector<int64_t> reps;
        bool reps_const = ins.size() > 1 && ins[1] >= 0 &&
                          value_const_ints(im, ins[1], reps) && reps.size() == is.size();
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = is;
        if (reps_const) {
            for (size_t i = 0; i < is.size(); i++) v.rshape[i] = emul(is[i], ec(reps[i]));
        } else {
            v.dyn = true;
        }
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::TILE;
        st.in = ins;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "PRelu") {
        Step st;
        st.k = SK::PRELU;
        st.in = ins;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "DFT") {
        // axis in st.axis, onesided in st.keepdims, inverse in
        // st.exclusive. Output's last dim (components) is
        // always 2; the axis length halves (onesided).
        auto is = shape_exprs(im, im.vals[ins[0]]);
        // The exec case normalizes the raw axis against the
        // true rank; the symbolic shape is only trusted when
        // the input is not dyn (a dyn input's rshape may be
        // rank-deficient, and indexing it would go out of
        // bounds).
        int64_t ax = n.attr_i("axis", 1);
        int64_t onesided = n.attr_i("onesided", 0);
        std::vector<EP> os;
        if (!eff_dyn(im, ins[0])) {
            int64_t nax = norm_axis(ax, (int64_t)is.size());
            os = is;
            if (onesided) os[nax] = eadd(ediv(is[nax], ec(2)), ec(1));
            if (!os.empty()) os.back() = ec(2);
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        if (eff_dyn(im, ins[0])) v.dyn = true;
        im.vals.push_back(std::move(v));
        int o = (int)im.vals.size() - 1;
        Step st;
        st.k = SK::DFT;
        st.in = ins;
        st.axis = ax;
        st.keepdims = onesided != 0;
        st.exclusive = n.attr_i("inverse", 0) != 0;
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "BatchNormalization") {
        if (n.attr_i("training_mode", 0) != 0) fail_op("BatchNormalization training mode");
        if (n.outputs.size() != 1) fail_op("BatchNormalization extra outputs");
        Step st;
        st.k = SK::BNORM;
        st.in = ins;
        st.eps = n.attr_f("epsilon", 1e-5f);
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "Clip") {
        // Bounds from attrs (old opsets) or scalar inputs; input
        // bounds are re-read at exec, attr bounds ride st.alpha /
        // st.beta with infinities for "unbounded".
        Step st;
        st.k = SK::CLIP;
        st.in = ins;
        st.alpha = n.attr_f("min", -std::numeric_limits<float>::infinity());
        st.beta = n.attr_f("max", std::numeric_limits<float>::infinity());
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "Loop") {
        lower_loop(C, n, ins);
        return;
    }

    fail_op("unsupported op in free engine (phase-1 GPT set)");
}



// ------------------------------------------------------ step execute ---
// Fused LeakyReLU epilogue: applied in place over a step's
// just-written output buffer, split over the pool for large
// tensors. The values are the same stored fp32 the standalone
// UNOP step would have read, so the result is bit-identical.
static void apply_epi_leaky(float* y, int64_t n, float alpha) {
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, n, [&](int64_t i0, int64_t i1) {
            float* p = y + i0;
            int64_t m = i1 - i0;
            if (!(cpu::features().avx2 && cpu::features().fma &&
                  avx2k::unop_f32((int)UnOp::LEAKY, p, p, m, alpha))) {
                for (int64_t i = 0; i < m; i++) p[i] = p[i] > 0 ? p[i] : alpha * p[i];
            }
        });
    };
    packed::pool_run(n, body);
}

// Exact integer elements of a small runtime tensor (slice
// bounds, range scalars), in logical order, following strides.
static std::vector<int64_t> rt_ints(const RT& t) {
    std::vector<int64_t> r;
    int64_t n = t.numel();
    r.reserve((size_t)n);
    int rank = (int)t.shape.size();
    std::vector<int64_t> str = t.strides.empty() ? rowmajor(t.shape) : t.strides;
    std::vector<int64_t> coord(rank, 0);
    int64_t off = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = 0;
        switch (t.dtype) {
            case 7: memcpy(&v, t.p + off * 8, 8); break;
            case 6: { int32_t x; memcpy(&x, t.p + off * 4, 4); v = x; break; }
            case 9: v = t.p[off] != 0 ? 1 : 0; break;
            default: v = (int64_t)read_d(t, off); break;
        }
        r.push_back(v);
        for (int d = rank - 1; d >= 0; d--) {
            off += str[d];
            if (++coord[d] < t.shape[d]) break;
            coord[d] = 0;
            off -= str[d] * t.shape[d];
        }
    }
    return r;
}

// Iterative radix-2 FFT (forward, double precision) for the
// STFT step. a.size() must be a power of two.
static void fft_pow2(std::vector<std::complex<double>>& a) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double PI = 3.14159265358979323846;
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * PI / len;
        std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w = 1.0;
            for (int k = 0; k < len / 2; k++) {
                std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

// Step kinds whose exec case sizes a dyn output itself (from
// runtime values or by copying a descriptor); all other kinds
// get their dyn output descriptor from dyn_shape_for.
static bool step_self_alloc(SK k) {
    switch (k) {
        case SK::SEQ_EMPTY: case SK::SEQ_SPLIT: case SK::SEQ_AT: case SK::SEQ_INSERT:
        case SK::SEQ_CONCAT: case SK::IF: case SK::LOOP: case SK::SLICE_RT:
        case SK::RANGE: case SK::TILE: case SK::EXPAND: case SK::CAST:
        case SK::CONCAT: case SK::DFT: case SK::SHAPE:
            return true;
        default: return false;
    }
}

static std::vector<int64_t> bcast_shape_c(const std::vector<int64_t>& A, const std::vector<int64_t>& B) {
    size_t r = std::max(A.size(), B.size());
    std::vector<int64_t> out(r);
    for (size_t i = 0; i < r; i++) {
        int64_t da = i < r - A.size() ? 1 : A[i - (r - A.size())];
        int64_t db = i < r - B.size() ? 1 : B[i - (r - B.size())];
        out[i] = da == db ? da : (da == 1 ? db : da);
    }
    return out;
}

static std::vector<int64_t> matmul_shape_c(std::vector<int64_t> a, std::vector<int64_t> b) {
    bool a1 = a.size() == 1, b1 = b.size() == 1;
    if (a1) a = {1, a[0]};
    if (b1) b = {b[0], 1};
    std::vector<int64_t> out = bcast_shape_c(std::vector<int64_t>(a.begin(), a.end() - 2),
                                             std::vector<int64_t>(b.begin(), b.end() - 2));
    out.push_back(a[a.size() - 2]);
    out.push_back(b[b.size() - 1]);
    if (a1) out.erase(out.begin() + (int)out.size() - 2);
    if (b1) out.pop_back();
    return out;
}

// Concrete output descriptor for a dyn step output, computed
// from the input descriptors at execution time.
static RT dyn_shape_for(Impl& im, const Step& st, const std::vector<RT>& sin, int ov) {
    std::vector<int64_t> sh;
    auto A = [&](int i) -> const RT& { return sin[(size_t)i]; };
    switch (st.k) {
        case SK::BINOP: sh = bcast_shape_c(A(0).shape, A(1).shape); break;
        case SK::WHERE: sh = bcast_shape_c(bcast_shape_c(A(0).shape, A(1).shape), A(2).shape); break;
        case SK::UNOP: case SK::PRELU: case SK::CLIP: case SK::BNORM:
        case SK::SOFTMAX: case SK::LN: case SK::INSTNORM: case SK::CUMSUM:
        case SK::COPY:
            sh = A(0).shape;
            break;
        case SK::GATHER_EL: sh = A(1).shape; break;
        case SK::SCATTER_EL: sh = A(0).shape; break;
        case SK::MATMUL: sh = matmul_shape_c(A(0).shape, A(1).shape); break;
        case SK::GEMM: {
            std::vector<int64_t> a = A(0).shape, b = A(1).shape;
            if (st.transA && a.size() >= 2) std::swap(a[a.size() - 2], a[a.size() - 1]);
            if (st.transB && b.size() >= 2) std::swap(b[b.size() - 2], b[b.size() - 1]);
            sh = matmul_shape_c(a, b);
            break;
        }
        case SK::PACKED: {
            sh = A(0).shape;
            if (!sh.empty()) sh[sh.size() - 1] = st.pm_cols;
            break;
        }
        case SK::CONV: {
            const auto& xs = A(0).shape;
            const auto& ws = A(1).shape;
            sh = xs;
            sh[1] = ws[0];
            // Geometry vectors may be shorter than the true
            // spatial rank (attr defaults); missing entries
            // take the ONNX defaults.
            auto gv = [](const std::vector<int64_t>& v, size_t i, int64_t dflt) {
                return i < v.size() ? v[i] : dflt;
            };
            for (size_t d = 2; d < xs.size(); d++)
                sh[d] = (xs[d] + 2 * gv(st.c_pads, d - 2, 0) - gv(st.c_dil, d - 2, 1) * (ws[d] - 1) - 1) /
                            gv(st.c_strides, d - 2, 1) + 1;
            break;
        }
        case SK::CONVT: {
            const auto& xs = A(0).shape;
            const auto& ws = A(1).shape;
            sh = xs;
            sh[1] = ws[1] * st.c_group;
            auto gv = [](const std::vector<int64_t>& v, size_t i, int64_t dflt) {
                return i < v.size() ? v[i] : dflt;
            };
            for (size_t d = 2; d < xs.size(); d++)
                sh[d] = (xs[d] - 1) * gv(st.c_strides, d - 2, 1) - gv(st.c_pads, d - 2, 0) -
                        gv(st.c_pads_end, d - 2, 0) + gv(st.c_dil, d - 2, 1) * (ws[d] - 1) + st.ct_out_pad + 1;
            break;
        }
        case SK::REDUCESUM: {
            sh = A(0).shape;
            std::vector<int64_t> axes = st.axes;
            for (auto& a : axes) a = norm_axis(a, sh.size());
            std::sort(axes.begin(), axes.end(), std::greater<int64_t>());
            for (auto a : axes) {
                if (a < 0 || a >= (int64_t)sh.size()) continue;
                if (st.keepdims) sh[(size_t)a] = 1;
                else sh.erase(sh.begin() + (size_t)a);
            }
            break;
        }
        case SK::ARGMAX: {
            sh = A(0).shape;
            int64_t a = norm_axis(st.axis, sh.size());
            if (st.keepdims) sh[(size_t)a] = 1;
            else sh.erase(sh.begin() + (size_t)a);
            break;
        }
        case SK::TOPK: {
            sh = A(0).shape;
            int64_t a = norm_axis(st.axis, sh.size());
            sh[(size_t)a] = st.topk;
            break;
        }
        case SK::GATHER: {
            int64_t a = norm_axis(st.axis, A(0).shape.size());
            sh.clear();
            for (int64_t i = 0; i < a; i++) sh.push_back(A(0).shape[(size_t)i]);
            for (auto d : A(1).shape) sh.push_back(d);
            for (size_t i = (size_t)a + 1; i < A(0).shape.size(); i++) sh.push_back(A(0).shape[i]);
            break;
        }
        case SK::PAD: {
            sh = A(0).shape;
            std::vector<int64_t> pads = rt_ints(A(1));
            // ONNX pads layout: all begins, then all ends.
            for (size_t d = 0; d < sh.size() && d + sh.size() < pads.size(); d++)
                sh[d] += pads[d] + pads[d + sh.size()];
            break;
        }
        case SK::CONSTSHAPE: sh = rt_ints(A(0)); break;
        case SK::RANDNORM: sh = A(0).shape; break;
        case SK::STFT: {
            // Mirror the exec geometry: [N, F, bins, 2] with
            // F = (L - fl) / fs + 1, bins from the onesided flag
            // (carried in st.axis at lowering).
            int64_t fs = rt_ints(A(1)).at(0);
            int64_t fl = (st.in.size() > 3 && st.in[3] >= 0) ? rt_ints(A(3)).at(0)
                         : (st.in.size() > 2 && st.in[2] >= 0 ? A(2).numel() : A(0).shape[1]);
            int64_t N = A(0).shape[0], L = A(0).shape[1];
            int64_t bins = st.axis ? fl / 2 + 1 : fl;
            int64_t F = L >= fl ? (L - fl) / fs + 1 : 0;
            sh = {N, F, bins, 2};
            break;
        }
        default: fail("dyn output shape not implemented for this step kind");
    }
    RT Y;
    Y.dtype = im.vals[ov].dtype;
    Y.shape = sh;
    Y.strides = rowmajor(sh);
    Y.p = im.dyn_alloc((size_t)Y.numel() * onnxm::elem_size(Y.dtype));
    return Y;
}

void exec_step(Impl& im, const Step& st, const std::vector<RT>& ins, std::vector<RT>& outs, Scratch& sc) {
    switch (st.k) {
        case SK::MATMUL: {
            const uint16_t* bf = nullptr;
            if (st.b_f16 >= 0 && (const float*)ins[1].p == st.b_f16_src)
                bf = im.f16_weights[st.b_f16].data();
            k_matmul(ins[0], ins[1], outs[0], sc, bf);
            if (st.out_scale != 1.0f) {
                // Folded scalar Mul: scale in place, in the output's
                // own layout. The double round-trip is exact for an
                // fp32 value times an fp32 scalar, so this is
                // bit-identical to the separate Mul step.
                auto ya = align_of(outs[0], outs[0].shape.size());
                odometer(outs[0].shape, [&](const std::vector<int64_t>& c) {
                    int64_t yo = off_of(outs[0], c, ya);
                    write_d(outs[0], yo, read_d(outs[0], yo) * (double)st.out_scale);
                });
            }
            break;
        }
        case SK::GEMM: {
            RT A = ins[0], B = ins[1];
            if (st.transA) {
                std::swap(A.shape[A.shape.size() - 2], A.shape[A.shape.size() - 1]);
                std::swap(A.strides[A.strides.size() - 2], A.strides[A.strides.size() - 1]);
            }
            if (st.transB) {
                std::swap(B.shape[B.shape.size() - 2], B.shape[B.shape.size() - 1]);
                std::swap(B.strides[B.strides.size() - 2], B.strides[B.strides.size() - 1]);
            }
            const uint16_t* bf = nullptr;
            if (st.b_f16 >= 0 && (const float*)B.p == st.b_f16_src)
                bf = im.f16_weights[st.b_f16].data();
            k_matmul(A, B, outs[0], sc, bf);
            bool have_c = ins.size() > 2;
            if (st.alpha != 1.0f || have_c) {
                const RT* Cp = have_c ? &ins[2] : nullptr;
                auto ca = Cp ? align_of(*Cp, outs[0].shape.size()) : std::vector<int64_t>{};
                auto ya = align_of(outs[0], outs[0].shape.size());
                odometer(outs[0].shape, [&](const std::vector<int64_t>& c) {
                    int64_t yo = off_of(outs[0], c, ya);
                    double v = read_d(outs[0], yo) * st.alpha;
                    if (Cp) v += st.beta * read_d(*Cp, off_of(*Cp, c, ca));
                    write_d(outs[0], yo, v);
                });
            }
            break;
        }
        case SK::PACKED: {
            RT X = sc.contig_of(ins[0]);
            int64_t D = (st.pm_mode == 0 || st.pm_mode == 2) ? st.pm_cols : st.pm_rows;
            int64_t K = X.shape.back();
            int64_t M = X.numel() / K;
            if (!im.lookup) fail("PackedMatMul: no packed lookup set");
            PackedBinding pb;
            if (!im.lookup(st.pm_base, pb)) fail("PackedMatMul: unknown weight " + st.pm_base);
            const float* bias = nullptr;
            RT bc;
            if (st.pm_bias) {
                bc = sc.contig_of(ins[1]);
                bias = (const float*)bc.p;
            }
            float* Y = (float*)outs[0].p;
            switch (st.pm_mode) {
                case 0: packed::gemm_a_w4((const float*)X.p, M, K, D, pb.pk, pb.sc, pb.ze, bias, Y); break;
                case 1: packed::gemm_b_w4((const float*)X.p, M, K, D, pb.pk, pb.sc, pb.ze, bias, Y); break;
                case 2: packed::gemm_a_w8((const float*)X.p, M, K, D, pb.pk, pb.sc, bias, Y); break;
                case 3: packed::gemm_b_w8((const float*)X.p, M, K, D, pb.pk, pb.sc, bias, Y); break;
                default: fail("PackedMatMul: bad mode");
            }
            break;
        }
        case SK::CONV: k_conv(ins[0], ins[1], ins.size() > 2 ? &ins[2] : nullptr, outs[0], st.c_strides, st.c_pads, st.c_dil, st.c_group, sc); if (st.epi_act == 1) apply_epi_leaky((float*)outs[0].p, outs[0].numel(), st.epi_alpha); break;
        case SK::CONVT: k_convtranspose(ins[0], ins[1], ins.size() > 2 ? &ins[2] : nullptr, outs[0], st.c_strides, st.c_pads, st.c_dil, sc); if (st.epi_act == 1) apply_epi_leaky((float*)outs[0].p, outs[0].numel(), st.epi_alpha); break;
        case SK::PCONV: {
            RT X = sc.contig_of(ins[0]);
            if (X.shape.size() != 3) fail("PackedConv: X must be (N, C, L)");
            if (X.dtype != 1 || outs[0].dtype != 1) fail("PackedConv: f32 only");
            if (!im.lookup) fail("PackedConv: no packed lookup set");
            PackedBinding pb;
            if (!im.lookup(st.pm_base, pb)) fail("PackedConv: unknown weight " + st.pm_base);
            const float* bias = nullptr;
            RT bc;
            if (st.pm_bias) {
                bc = sc.contig_of(ins[1]);
                bias = (const float*)bc.p;
            }
            int64_t N = X.shape[0], Ci = X.shape[1], L = X.shape[2];
            const uint8_t* ze = st.pm_zero ? pb.ze : nullptr;
            float* Y = (float*)outs[0].p;
            if (!st.pc_transpose) {
                if (Ci * st.pc_k != st.pm_cols) fail("PackedConv: Ci*K != w_cols");
                packed::conv1d_fast((const float*)X.p, N, Ci, L, pb.pk, pb.sc, ze, st.pm_mode,
                                    st.pm_rows, st.pc_k, st.pc_stride, st.pc_pad_l, st.pc_dil,
                                    outs[0].shape[2], bias, Y);
                if (st.epi_act == 1) apply_epi_leaky(Y, outs[0].numel(), st.epi_alpha);
            } else {
                if (Ci != st.pm_rows) fail("PackedConvTranspose: Ci != w_rows");
                int64_t O = st.pm_cols / st.pc_k;
                packed::conv_transpose1d_fast((const float*)X.p, N, Ci, L, pb.pk, pb.sc, ze,
                                              st.pm_mode, O, st.pc_k, st.pc_stride, st.pc_pad_l,
                                              st.pc_dil, outs[0].shape[2], bias, Y);
                if (st.epi_act == 1) apply_epi_leaky(Y, outs[0].numel(), st.epi_alpha);
            }
            break;
        }
        case SK::PGATHER: {
            RT idx = sc.contig_of(ins[0]);
            if (idx.dtype != 7) fail("PackedGather: indices must be int64");
            if (!im.lookup) fail("PackedGather: no packed lookup set");
            PackedBinding pb;
            if (!im.lookup(st.pm_base, pb)) fail("PackedGather: unknown weight " + st.pm_base);
            packed::gather_packed(pb.pk, pb.sc, st.pm_zero ? pb.ze : nullptr, st.pm_mode,
                                  st.pm_rows, st.pm_cols, (const int64_t*)idx.p, idx.numel(),
                                  (float*)outs[0].p);
            break;
        }
        case SK::PAD: {
            // Exact integer read of the pads tensor (small values, but
            // keep them out of the double path on principle).
            const RT& pt = ins[1];
            std::vector<int64_t> pads;
            for (int64_t i = 0; i < pt.numel(); i++) {
                int64_t v = 0;
                if (pt.dtype == 7) memcpy(&v, pt.p + i * 8, 8);
                else if (pt.dtype == 6) { int32_t x; memcpy(&x, pt.p + i * 4, 4); v = x; }
                else v = (int64_t)read_d(pt, i);
                pads.push_back(v);
            }
            double fillv = ins.size() > 2 ? read_d(ins[2], 0) : 0.0;
            k_pad(ins[0], outs[0], pads, st.pad_mode, fillv);
            break;
        }
        case SK::SEQ_EMPTY: im.bseqs[st.out[0]].clear(); break;
        case SK::SEQ_SPLIT: {
            const RT& X = ins[0];
            const RT& sp = ins[1];
            std::vector<int64_t> splits;
            for (int64_t i = 0; i < sp.numel(); i++) {
                int64_t v = 0;
                if (sp.dtype == 7) memcpy(&v, sp.p + i * 8, 8);
                else if (sp.dtype == 6) { int32_t x; memcpy(&x, sp.p + i * 4, 4); v = x; }
                else v = (int64_t)read_d(sp, i);
                splits.push_back(v);
            }
            int64_t ax = norm_axis(st.axis, (int64_t)X.shape.size());
            auto& seq = im.bseqs[st.out[0]];
            seq.clear();
            int64_t cum = 0;
            for (int64_t s : splits) {
                RT e = X;
                e.shape[ax] = s;
                e.p = X.p + cum * X.strides[ax] * (int64_t)onnxm::elem_size(X.dtype);
                seq.push_back(e);
                cum += s;
            }
            break;
        }
        case SK::SEQ_AT: {
            const RT& ix = ins[1];
            int64_t idx = 0;
            if (ix.dtype == 7) memcpy(&idx, ix.p, 8);
            else if (ix.dtype == 6) { int32_t x; memcpy(&x, ix.p, 4); idx = x; }
            else idx = (int64_t)read_d(ix, 0);
            const auto& seq = im.bseqs[st.in[0]];
            if (idx < 0) idx += (int64_t)seq.size();
            if (idx < 0 || idx >= (int64_t)seq.size()) fail("SequenceAt: index out of range");
            outs[0] = seq[(size_t)idx];
            break;
        }
        case SK::SEQ_INSERT: {
            im.bseqs[st.out[0]] = im.bseqs[st.in[0]];
            im.bseqs[st.out[0]].push_back(ins[1]);
            break;
        }
        case SK::SEQ_CONCAT: {
            const auto& seq = im.bseqs[st.in[0]];
            RT Y;
            // outs[0] is the unbound placeholder for a dyn value
            // (dtype 0, no buffer); the true dtype lives on the
            // value / the sequence elements.
            Y.dtype = seq.empty() ? im.vals[st.out[0]].dtype : seq[0].dtype;
            if (seq.empty()) {
                Y.shape = outs[0].shape;
                if (!Y.shape.empty()) Y.shape[norm_axis(st.axis, (int64_t)Y.shape.size())] = 0;
                Y.strides = rowmajor(Y.shape);
                Y.p = im.dyn_alloc(0);
                outs[0] = Y;
                break;
            }
            int64_t ax = norm_axis(st.axis, (int64_t)seq[0].shape.size());
            Y.shape = seq[0].shape;
            int64_t total = 0;
            for (auto& e : seq) total += e.shape[ax];
            Y.shape[ax] = total;
            Y.strides = rowmajor(Y.shape);
            Y.p = im.dyn_alloc((size_t)Y.numel() * onnxm::elem_size(Y.dtype));
            int64_t cum = 0;
            for (auto& e : seq) {
                RT sub = Y;
                sub.shape = e.shape;
                sub.p = Y.p + cum * Y.strides[ax] * (int64_t)onnxm::elem_size(Y.dtype);
                copy_tensor(sub, e);
                cum += e.shape[ax];
            }
            outs[0] = Y;
            break;
        }
        case SK::SLICE_RT: {
            // Runtime bounds: build the strided view descriptor
            // from the bound tensors' current values; materialize
            // into the dyn arena when the view is not contiguous
            // (flat readers downstream assume one or the other).
            const RT& X = ins[0];
            std::vector<int64_t> starts = rt_ints(ins[1]);
            std::vector<int64_t> ends = rt_ints(ins[2]);
            std::vector<int64_t> stepsv;
            if (st.in.size() > 4 && st.in[4] >= 0) stepsv = rt_ints(ins[4]);
            RT Y = X;
            int64_t off = 0;
            std::vector<int64_t> axes = st.axes;
            if (axes.empty()) {
                axes.resize(X.shape.size());
                for (size_t a = 0; a < axes.size(); a++) axes[a] = (int64_t)a;
            }
            for (size_t i = 0; i < axes.size(); i++) {
                int64_t ax = norm_axis(axes[i], X.shape.size());
                int64_t step = stepsv.empty() ? 1 : stepsv[i];
                if (step == 0) fail("Slice: step 0");
                int64_t first;
                Y.shape[ax] = slice_count(X.shape[ax], starts[i], ends[i], step, first);
                off += first * X.strides[ax];
                Y.strides[ax] = X.strides[ax] * step;
            }
            Y.p = X.p + off * (int64_t)onnxm::elem_size(X.dtype);
            if (!Y.contig()) {
                RT M;
                M.dtype = Y.dtype;
                M.shape = Y.shape;
                M.strides = rowmajor(M.shape);
                M.p = im.dyn_alloc((size_t)M.numel() * onnxm::elem_size(M.dtype));
                copy_tensor(M, Y);
                Y = M;
            }
            outs[0] = Y;
            break;
        }
        case SK::SHAPE: {
            // Runtime Shape: the output is the input's true
            // dim list as an i64 vector (rank known only here).
            RT Y;
            Y.dtype = 7;
            Y.shape = {(int64_t)ins[0].shape.size()};
            Y.strides = {1};
            Y.p = im.dyn_alloc((size_t)ins[0].shape.size() * 8);
            int64_t* d = (int64_t*)Y.p;
            for (size_t i = 0; i < ins[0].shape.size(); i++) d[i] = ins[0].shape[i];
            outs[0] = Y;
            break;
        }
        case SK::RANGE: {
            int odt = im.vals[st.out[0]].dtype;
            RT Y;
            Y.dtype = odt;
            int64_t n = 0;
            if (odt == 7 || odt == 6) {
                int64_t s = rt_ints(ins[0]).at(0), l = rt_ints(ins[1]).at(0), d = rt_ints(ins[2]).at(0);
                if (d == 0) fail("Range: delta 0");
                if (d > 0 && l > s) n = (l - s + d - 1) / d;
                if (d < 0 && l < s) n = (s - l - d - 1) / -d;
                Y.shape = {n};
                Y.strides = {1};
                Y.p = im.dyn_alloc((size_t)n * onnxm::elem_size(odt));
                for (int64_t i = 0; i < n; i++) write_d(Y, i, (double)(s + i * d));
            } else {
                double s = read_d(ins[0], 0), l = read_d(ins[1], 0), d = read_d(ins[2], 0);
                if (d == 0) fail("Range: delta 0");
                double cnt = (l - s) / d;
                if (cnt > 0) n = (int64_t)std::ceil(cnt - 1e-9);
                Y.shape = {n};
                Y.strides = {1};
                Y.p = im.dyn_alloc((size_t)n * onnxm::elem_size(odt));
                for (int64_t i = 0; i < n; i++) write_d(Y, i, s + i * d);
            }
            outs[0] = Y;
            break;
        }
        case SK::STFT: {
            // Input order as ORT reads it: signal, frame_step,
            // window, frame_length (see the lowering note).
            const RT& X = ins[0];
            int64_t fs = rt_ints(ins[1]).at(0);
            const RT* W = (st.in.size() > 2 && st.in[2] >= 0) ? &ins[2] : nullptr;
            int64_t fl = (st.in.size() > 3 && st.in[3] >= 0) ? rt_ints(ins[3]).at(0)
                         : (W ? W->numel() : X.shape[1]);
            int64_t N = X.shape[0], L = X.shape[1];
            int64_t bins = st.axis ? fl / 2 + 1 : fl;
            int64_t F = L >= fl ? (L - fl) / fs + 1 : 0;
            if (outs[0].shape.size() != 4 || outs[0].shape[1] != F || outs[0].shape[2] != bins)
                fail("STFT: bind shape disagrees with exec frame count");
            int64_t s0 = X.strides.empty() ? L : X.strides[0];
            int64_t s1 = X.strides.size() > 1 ? X.strides[1] : 1;
            float* outp = (float*)outs[0].p;
            std::vector<std::complex<double>> buf((size_t)fl);
            bool pow2 = fl > 0 && (fl & (fl - 1)) == 0;
            const double PI = 3.14159265358979323846;
            for (int64_t n = 0; n < N; n++) {
                for (int64_t f = 0; f < F; f++) {
                    int64_t base = n * s0 + f * fs * s1;
                    for (int64_t k = 0; k < fl; k++) {
                        double w = W ? read_d(*W, k) : 1.0;
                        buf[(size_t)k] = read_d(X, base + k * s1) * w;
                    }
                    float* dst = outp + ((n * F + f) * bins) * 2;
                    if (pow2) {
                        fft_pow2(buf);
                        for (int64_t b = 0; b < bins; b++) {
                            dst[b * 2] = (float)buf[(size_t)b].real();
                            dst[b * 2 + 1] = (float)buf[(size_t)b].imag();
                        }
                    } else {
                        for (int64_t b = 0; b < bins; b++) {
                            std::complex<double> wstep = std::polar(1.0, -2.0 * PI * (double)b / (double)fl);
                            std::complex<double> wk = 1.0, sum = 0.0;
                            for (int64_t k = 0; k < fl; k++) { sum += buf[(size_t)k] * wk; wk *= wstep; }
                            dst[b * 2] = (float)sum.real();
                            dst[b * 2 + 1] = (float)sum.imag();
                        }
                    }
                }
            }
            break;
        }
        case SK::TILE: {
            const RT& X = ins[0];
            std::vector<int64_t> reps = rt_ints(ins[1]);
            RT Y;
            Y.dtype = X.dtype;
            if (!st.out.empty() && im.vals[st.out[0]].dyn) {
                Y.shape = X.shape;
                for (size_t i = 0; i < Y.shape.size(); i++) Y.shape[i] *= reps[i];
                Y.strides = rowmajor(Y.shape);
                Y.p = im.dyn_alloc((size_t)Y.numel() * onnxm::elem_size(Y.dtype));
            } else {
                Y = outs[0];
            }
            std::vector<int64_t> xstr = X.strides.empty() ? rowmajor(X.shape) : X.strides;
            int rank = (int)Y.shape.size();
            std::vector<int64_t> coord(rank, 0);
            int64_t esz = (int64_t)onnxm::elem_size(Y.dtype);
            int64_t n = Y.numel();
            for (int64_t i = 0; i < n; i++) {
                int64_t ioff = 0, ooff = 0;
                for (int d = 0; d < rank; d++) {
                    ioff += (coord[d] % X.shape[d]) * xstr[d];
                    ooff += coord[d] * Y.strides[d];
                }
                memcpy(Y.p + ooff * esz, X.p + ioff * esz, (size_t)esz);
                for (int d = rank - 1; d >= 0; d--) {
                    if (++coord[d] < Y.shape[d]) break;
                    coord[d] = 0;
                }
            }
            outs[0] = Y;
            break;
        }
        case SK::PRELU: {
            const RT& X = ins[0];
            const RT& S = ins[1];
            RT& Y = outs[0];
            int rank = (int)X.shape.size();
            int srank = (int)S.shape.size();
            std::vector<int64_t> xstr = X.strides.empty() ? rowmajor(X.shape) : X.strides;
            std::vector<int64_t> sstr = S.strides.empty() ? rowmajor(S.shape) : S.strides;
            std::vector<int64_t> ystr = Y.strides.empty() ? rowmajor(Y.shape) : Y.strides;
            std::vector<int64_t> coord(rank, 0);
            int64_t n = X.numel();
            for (int64_t i = 0; i < n; i++) {
                int64_t xo = 0, yo = 0, so = 0;
                for (int d = 0; d < rank; d++) {
                    xo += coord[d] * xstr[d];
                    yo += coord[d] * ystr[d];
                    int sd = d - (rank - srank);
                    if (sd >= 0) so += (S.shape[sd] == 1 ? 0 : coord[d]) * sstr[sd];
                }
                double x = read_d(X, xo);
                write_d(Y, yo, x >= 0 ? x : read_d(S, so) * x);
                for (int d = rank - 1; d >= 0; d--) {
                    if (++coord[d] < X.shape[d]) break;
                    coord[d] = 0;
                }
            }
            break;
        }
        case SK::DFT: {
            const RT& X = ins[0];
            RT Y;
            if (!st.out.empty() && im.vals[st.out[0]].dyn) {
                Y.dtype = X.dtype;
                Y.shape = X.shape;
                int64_t ax0 = norm_axis(st.axis, (int64_t)Y.shape.size());
                if (st.keepdims) Y.shape[ax0] = Y.shape[ax0] / 2 + 1;
                if (!Y.shape.empty()) Y.shape.back() = 2;
                Y.strides = rowmajor(Y.shape);
                Y.p = im.dyn_alloc((size_t)Y.numel() * onnxm::elem_size(Y.dtype));
            } else {
                Y = outs[0];
            }
            int rank = (int)X.shape.size();
            int64_t ax = norm_axis(st.axis, rank);
            int64_t L = X.shape[ax];
            int64_t bins = st.keepdims ? L / 2 + 1 : L;
            int64_t lcomp = X.shape[rank - 1];
            std::vector<int64_t> xstr = X.strides.empty() ? rowmajor(X.shape) : X.strides;
            std::vector<int64_t> ystr = Y.strides.empty() ? rowmajor(Y.shape) : Y.strides;
            bool pow2 = L > 0 && (L & (L - 1)) == 0;
            double sgn = st.exclusive ? 1.0 : -1.0;
            double scale = st.exclusive ? 1.0 / (double)L : 1.0;
            const double PI = 3.14159265358979323846;
            std::vector<int64_t> coord(rank, 0);
            std::vector<std::complex<double>> buf((size_t)L);
            int64_t outer = 1;
            for (int d = 0; d < rank; d++)
                if (d != ax && d != rank - 1) outer *= X.shape[d];
            for (int64_t o = 0; o < outer; o++) {
                int64_t xb = 0, yb = 0;
                for (int d = 0; d < rank; d++) {
                    if (d == ax || d == rank - 1) continue;
                    xb += coord[d] * xstr[d];
                    yb += coord[d] * ystr[d];
                }
                for (int64_t k = 0; k < L; k++) {
                    int64_t off = xb + k * xstr[ax];
                    double re = read_d(X, off);
                    double imv = lcomp == 2 ? read_d(X, off + xstr[rank - 1]) : 0.0;
                    buf[(size_t)k] = {re, imv};
                }
                if (pow2) {
                    if (st.exclusive)
                        for (auto& c : buf) c = std::conj(c);
                    fft_pow2(buf);
                    if (st.exclusive)
                        for (auto& c : buf) c = std::conj(c) * scale;
                }
                for (int64_t b = 0; b < bins; b++) {
                    std::complex<double> sum;
                    if (pow2) {
                        sum = buf[(size_t)b];
                    } else {
                        std::complex<double> wstep = std::polar(1.0, sgn * 2.0 * PI * (double)b / (double)L);
                        std::complex<double> wk = 1.0;
                        sum = 0.0;
                        for (int64_t k = 0; k < L; k++) { sum += buf[(size_t)k] * wk; wk *= wstep; }
                        sum *= scale;
                    }
                    int64_t off = yb + b * ystr[ax];
                    write_d(Y, off, sum.real());
                    write_d(Y, off + ystr[rank - 1], sum.imag());
                }
                for (int d = rank - 1; d >= 0; d--) {
                    if (d == ax || d == rank - 1) continue;
                    if (++coord[d] < X.shape[d]) break;
                    coord[d] = 0;
                }
            }
            outs[0] = Y;
            break;
        }
        case SK::BNORM: {
            // Inference BN: per-channel affine from the given
            // mean/var (channel = dim 1, or dim 0 for rank 1).
            const RT& X = ins[0];
            RT& Y = outs[0];
            int rank = (int)X.shape.size();
            int cd = rank >= 2 ? 1 : 0;
            int64_t C = X.shape[cd];
            std::vector<double> inv((size_t)C), shift((size_t)C);
            for (int64_t c = 0; c < C; c++) {
                double mean = st.in.size() > 3 && st.in[3] >= 0 ? read_d(ins[3], c) : 0.0;
                double var = st.in.size() > 4 && st.in[4] >= 0 ? read_d(ins[4], c) : 1.0;
                double scale = st.in.size() > 1 && st.in[1] >= 0 ? read_d(ins[1], c) : 1.0;
                double bias = st.in.size() > 2 && st.in[2] >= 0 ? read_d(ins[2], c) : 0.0;
                inv[(size_t)c] = scale / std::sqrt(var + st.eps);
                shift[(size_t)c] = bias - mean * inv[(size_t)c];
            }
            std::vector<int64_t> xstr = X.strides.empty() ? rowmajor(X.shape) : X.strides;
            std::vector<int64_t> ystr = Y.strides.empty() ? rowmajor(Y.shape) : Y.strides;
            std::vector<int64_t> coord(rank, 0);
            int64_t n = X.numel();
            for (int64_t i = 0; i < n; i++) {
                int64_t xo = 0, yo = 0;
                for (int d = 0; d < rank; d++) {
                    xo += coord[d] * xstr[d];
                    yo += coord[d] * ystr[d];
                }
                int64_t c = coord[cd];
                write_d(Y, yo, read_d(X, xo) * inv[(size_t)c] + shift[(size_t)c]);
                for (int d = rank - 1; d >= 0; d--) {
                    if (++coord[d] < X.shape[d]) break;
                    coord[d] = 0;
                }
            }
            break;
        }
        case SK::CLIP: {
            double lo = st.alpha, hi = st.beta;
            if (st.in.size() > 1 && st.in[1] >= 0) lo = read_d(ins[1], 0);
            if (st.in.size() > 2 && st.in[2] >= 0) hi = read_d(ins[2], 0);
            const RT& X = ins[0];
            RT& Y = outs[0];
            std::vector<int64_t> xstr = X.strides.empty() ? rowmajor(X.shape) : X.strides;
            std::vector<int64_t> ystr = Y.strides.empty() ? rowmajor(Y.shape) : Y.strides;
            int rank = (int)X.shape.size();
            std::vector<int64_t> coord(rank, 0);
            int64_t n = X.numel();
            for (int64_t i = 0; i < n; i++) {
                int64_t xo = 0, yo = 0;
                for (int d = 0; d < rank; d++) {
                    xo += coord[d] * xstr[d];
                    yo += coord[d] * ystr[d];
                }
                double x = read_d(X, xo);
                write_d(Y, yo, x < lo ? lo : (x > hi ? hi : x));
                for (int d = rank - 1; d >= 0; d--) {
                    if (++coord[d] < X.shape[d]) break;
                    coord[d] = 0;
                }
            }
            break;
        }
        case SK::EXPAND: {
            if (!st.out.empty() && im.vals[st.out[0]].dyn && st.in.size() == 1) {
                // Symbolic-target step (the target was folded
                // into the output's shape exprs at lowering, so
                // the step carries no target input) whose output
                // was marked dyn by propagation because the data
                // is dyn. With a constant target (kept in
                // st.axes), broadcast the data's true shape with
                // it; otherwise evaluate the shape exprs.
                // Reading ins[1] here was an out-of-bounds
                // access on the input vector.
                std::vector<int64_t> os;
                if (!st.axes.empty()) os = bcast_shape_c(ins[0].shape, st.axes);
                else
                    for (auto& e : im.vals[st.out[0]].rshape) os.push_back(eeval(e, im.bsv));
                outs[0].shape = os;
                outs[0].strides = rowmajor(os);
                outs[0].dtype = ins[0].dtype;
                outs[0].p = im.dyn_alloc((size_t)outs[0].numel() * onnxm::elem_size(outs[0].dtype));
            }
            if (!st.out.empty() && im.vals[st.out[0]].dyn && st.in.size() > 1) {
                // Dyn output (loop body): the target shape is a
                // runtime tensor; size a fresh buffer from it on
                // every execution (the descriptor written back by
                // the previous iteration must not suppress this).
                const RT& tp = ins[1];
                std::vector<int64_t> tgt;
                for (int64_t i = 0; i < tp.numel(); i++) {
                    int64_t v = 0;
                    if (tp.dtype == 7) memcpy(&v, tp.p + i * 8, 8);
                    else if (tp.dtype == 6) { int32_t x; memcpy(&x, tp.p + i * 4, 4); v = x; }
                    else v = (int64_t)read_d(tp, i);
                    tgt.push_back(v);
                }
                const RT& X = ins[0];
                std::vector<int64_t> os = tgt;
                for (size_t i = 0; i < os.size(); i++) {
                    if (os[i] < 0) {
                        int64_t sd = (int64_t)i - ((int64_t)os.size() - (int64_t)X.shape.size());
                        os[i] = sd >= 0 ? X.shape[sd] : 1;
                    }
                }
                // ONNX Expand semantics: the target is a
                // minimum shape; the output is the broadcast
                // of the data shape with the target (a target
                // dim of 1 takes the data dim). Treating the
                // target as the literal output shape broke
                // prompt_encoder's mask expand ([1,1,212] to
                // target [1,212,1] must give [1,212,212]).
                os = bcast_shape_c(X.shape, os);
                outs[0].shape = os;
                outs[0].strides = rowmajor(os);
                outs[0].dtype = X.dtype;
                outs[0].p = im.dyn_alloc((size_t)outs[0].numel() * onnxm::elem_size(outs[0].dtype));
            }
            k_expand(ins[0], outs[0]);
            break;
        }
        case SK::LN: k_layernorm(ins[0], ins.size() > 1 ? &ins[1] : nullptr, ins.size() > 2 ? &ins[2] : nullptr, outs[0], st.axis, st.eps, sc); break;
        case SK::INSTNORM: k_instancenorm(ins[0], ins.size() > 1 ? &ins[1] : nullptr, ins.size() > 2 ? &ins[2] : nullptr, outs[0], st.eps, sc); break;
        case SK::SOFTMAX: k_softmax(ins[0], outs[0], st.axis, sc); break;
        case SK::ATTN: k_attn(ins[0], ins[1], ins[2], outs[0], sc, st.out_scale); break;
        case SK::REDUCESUM: {
            // Axes were normalized at lowering against the
            // symbolic shape; for a dyn input that shape may be
            // rank-deficient, so re-normalize against the true
            // rank here (kernels index red[axis] unchecked).
            std::vector<int64_t> axes = st.axes;
            for (auto& a : axes) a = norm_axis(a, ins[0].shape.size());
            if (st.l2) { k_reducel2(ins[0], outs[0], axes, st.keepdims); break; }
            k_reducesum(ins[0], outs[0], axes, st.keepdims);
            if (st.mean) {
                double f = (double)outs[0].numel() / (double)ins[0].numel();
                RT& Y = outs[0];
                if (Y.dtype == 1 && Y.contig()) {
                    float* p = (float*)Y.p;
                    int64_t n = Y.numel();
                    for (int64_t i = 0; i < n; i++) p[i] = (float)(p[i] * f);
                } else {
                    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
                        int64_t yo = 0;
                        for (size_t d = 0; d < c.size(); d++) yo += c[d] * Y.strides[d];
                        write_d(Y, yo, read_d(Y, yo) * f);
                    });
                }
            }
            break;
        }
        case SK::ARGMAX: k_argmax(ins[0], outs[0], st.axis, st.keepdims, st.exclusive); break;
        case SK::TOPK: k_topk(ins[0], outs[0], outs[1], st.axis, st.topk, st.largest, st.sorted); break;
        case SK::CUMSUM: k_cumsum(ins[0], outs[0], st.axis, st.exclusive, st.reverse); break;
        case SK::GATHER: k_gather(ins[0], ins[1], outs[0], st.axis); break;
        case SK::GATHER_EL: k_gather_elements(ins[0], ins[1], outs[0], st.axis); break;
        case SK::SCATTER_EL: k_scatter_elements(ins[0], ins[1], ins[2], outs[0], st.axis, st.reduction); break;
        case SK::CONCAT: {
            if (!st.out.empty() && im.vals[st.out[0]].dyn) {
                // An input's true length is data-dependent:
                // derive the output shape from the input
                // descriptors and allocate in the dyn arena.
                const RT* tmpl = &ins[0];
                for (auto& e : ins)
                    if (!e.shape.empty()) { tmpl = &e; break; }
                int64_t ax = norm_axis(st.axis, (int64_t)tmpl->shape.size());
                RT Y;
                Y.dtype = ins[0].dtype;
                Y.shape = tmpl->shape;
                int64_t total = 0;
                for (auto& e : ins) total += e.shape[ax];
                Y.shape[ax] = total;
                Y.strides = rowmajor(Y.shape);
                Y.p = im.dyn_alloc((size_t)Y.numel() * onnxm::elem_size(Y.dtype));
                int64_t cum = 0;
                for (auto& e : ins) {
                    RT sub = Y;
                    sub.shape = e.shape;
                    sub.p = Y.p + cum * Y.strides[ax] * (int64_t)onnxm::elem_size(Y.dtype);
                    copy_tensor(sub, e);
                    cum += e.shape[ax];
                }
                outs[0] = Y;
                break;
            }
            if (st.kv_slot >= 0 && im.kv) {
                // Stateful KV append: outs[0] views the cache
                // buffer; copy only the new slice at the slot's
                // current length and advance it.
                KvCache::Slot& slot = im.kv->slots[st.kv_slot];
                RT in = sc.contig_of(ins[1]);
                int64_t rows = in.shape.empty() ? 0 : in.shape[0];
                size_t rowf = rows > 0 ? (size_t)in.numel() / (size_t)rows : 0;
                std::memcpy(slot.buf.get() + (size_t)slot.len * rowf, in.p,
                            (size_t)rows * rowf * sizeof(float));
                slot.len += rows;
                break;
            }
            std::vector<const RT*> ptrs;
            for (auto& t : ins) ptrs.push_back(&t);
            k_concat(ptrs, outs[0], st.axis, sc);
            break;
        }
        case SK::CONSTSHAPE: {
            double v = read_d(tensor_of_const(st.fill), 0);
            RT& Y = outs[0];
            if (v == 0) {
                memset(Y.p, 0, (size_t)Y.numel() * onnxm::elem_size(Y.dtype));
            } else {
                for (int64_t i = 0; i < Y.numel(); i++) write_d(Y, i, v);
            }
            break;
        }
        case SK::RANDNORM: {
            RT& Y = outs[0];
            std::mt19937_64 rng(0x9E3779B97F4A7C15ULL + (++im.rng_ctr) * 0x2545F4914F6CDD1DULL);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (int64_t i = 0; i < Y.numel(); i++) write_d(Y, i, nd(rng));
            break;
        }
        case SK::BINOP: k_binop(st.bop, ins[0], ins[1], outs[0]); if (st.epi_act == 1) apply_epi_leaky((float*)outs[0].p, outs[0].numel(), st.epi_alpha); break;
        case SK::UNOP: k_unop(st.uop, ins[0], outs[0], st.alpha); break;
        case SK::CAST: {
            if (!st.out.empty() && im.vals[st.out[0]].dyn) {
                RT Y;
                Y.dtype = im.vals[st.out[0]].dtype;
                Y.shape = ins[0].shape;
                Y.strides = rowmajor(Y.shape);
                Y.p = im.dyn_alloc((size_t)Y.numel() * onnxm::elem_size(Y.dtype));
                k_cast(ins[0], Y);
                outs[0] = Y;
                break;
            }
            k_cast(ins[0], outs[0]);
            break;
        }
        case SK::WHERE: k_where(ins[0], ins[1], ins[2], outs[0]); break;
        case SK::COPY:
            if (st.flat_copy) flatten_copy(outs[0], ins[0]);
            else copy_tensor(outs[0], ins[0]);
            break;
    }
}

}  // namespace

// ---------------------------------------------------------- compile ----
// ------------------------------------------------- scalar-mul fold ---
// Fold "Mul(MatMul(..), scalar-const)" into the matmul: the scalar
// becomes the matmul's epilogue scale (GEMM: its alpha, when the
// Gemm has no C input) and the Mul step is erased, its consumers
// rewired to the matmul-side value. Motivated by the attention
// score scale in the GPT stage plans: as a standalone step it walks
// the score tensor in the consumer's layout (~200 ms per request);
// as an epilogue it rides the matmul's own write pass. Bit-identical:
// the epilogue multiplies each finished fp32 product by the same
// fp32 scalar the Mul would have used (the double round-trip in the
// epilogue is exact for an fp32 value times an fp32 scalar).
// Conservative by construction: the matmul output value, resolved
// through its view chain, must reach this Mul and nothing else.
static bool fold_one_scalar_mul(Impl& im) {
    auto& vals = im.vals;
    auto& steps = im.steps;
    std::vector<int> uses(vals.size(), 0), view_kids(vals.size(), 0);
    for (auto& s : steps)
        for (int v : s.in) if (v >= 0) uses[v]++;
    for (size_t i = 0; i < vals.size(); i++) {
        const Value& v = vals[i];
        if (v.kind == Value::RT && v.view_kind != 0 && v.base >= 0) {
            uses[v.base]++;
            view_kids[v.base]++;
        }
        for (int m : {v.meta0, v.meta1, v.meta2, v.meta3})
            if (m >= 0) uses[m]++;
    }
    for (int v : im.out_vals) uses[v]++;
    for (int s2i = 0; s2i < (int)steps.size(); s2i++) {
        Step& s2 = steps[s2i];
        if (s2.keep) continue;
        if (s2.k != SK::BINOP || s2.bop != BinOp::MUL || s2.in.size() != 2 ||
            s2.out.size() != 1)
            continue;
        int vx = -1;
        float scale = 0;
        for (int t = 0; t < 2; t++) {
            const Value& vc = vals[s2.in[t]];
            if (vc.kind == Value::CONSTV && vc.ct.data_type == 1 && vc.ct.numel() == 1 &&
                vc.ct.data.size() >= 4) {
                scale = *(const float*)vc.ct.data.data();
                vx = s2.in[1 - t];
            }
        }
        if (vx < 0) continue;
        int vy = s2.out[0];
        if (vals[vy].kind != Value::RT || vals[vy].view_kind != 0) continue;
        if (view_kids[vy] != 0) continue;
        bool meta_ref = false;
        for (auto& v : vals)
            for (int m : {v.meta0, v.meta1, v.meta2, v.meta3})
                if (m == vy) meta_ref = true;
        if (meta_ref) continue;
        // Resolve vx through views to its base value.
        int vb = vx;
        std::vector<int> chain;
        while (vals[vb].kind == Value::RT && vals[vb].view_kind != 0 && vals[vb].base >= 0) {
            chain.push_back(vb);
            vb = vals[vb].base;
        }
        if (vals[vb].kind != Value::RT || vals[vb].view_kind != 0) continue;
        if (vals[vb].is_output) continue;
        // Use-count discipline: every value on the path is consumed
        // exactly by the next link (and vb by nothing else); any
        // extra consumer, meta ref or output flag breaks the equalities.
        if (view_kids[vb] != (chain.empty() ? 0 : 1)) continue;
        if (uses[vb] != view_kids[vb] + (chain.empty() ? 1 : 0)) continue;
        bool ok = true;
        for (size_t ci = 0; ci < chain.size(); ci++) {
            int w = chain[ci];
            if (vals[w].is_output) ok = false;
            if (view_kids[w] != (ci == 0 ? 0 : 1)) ok = false;
            if (uses[w] != view_kids[w] + (ci == 0 ? 1 : 0)) ok = false;
        }
        if (!ok) continue;
        int s1i = vals[vb].def_step;
        if (s1i < 0 || s1i >= (int)steps.size()) continue;
        Step& s1 = steps[s1i];
        if (s1.out.size() != 1 || s1.out[0] != vb) continue;
        if (s1.k == SK::MATMUL) {
            s1.out_scale *= scale;
        } else if (s1.k == SK::GEMM && s1.in.size() == 2) {
            s1.alpha *= scale;
        } else {
            continue;
        }
        for (auto& s : steps)
            for (int& v : s.in)
                if (v == vy) v = vx;
        for (int& v : im.out_vals)
            if (v == vy) v = vx;
        if (vals[vy].is_output) vals[vx].is_output = true;
        vals[vy].def_step = -2;
        vals[vy].is_output = false;
        steps.erase(steps.begin() + s2i);
        for (auto& v : vals)
            if (v.def_step > s2i) v.def_step--;
        return true;
    }
    return false;
}

// Compile-time scalar resolution for small constant expressions:
// CONSTV leaves, Cast, Sqrt, Neg, Mul, Div over resolvable operands.
// Returns false when any part is runtime. The float arithmetic
// mirrors the runtime's fp32 ops (sqrt is correctly rounded, and
// double rounding through 53 bits is innocuous for sqrt at 24-bit
// precision), so a resolved value is bit-identical to the runtime
// one.
static bool resolve_scalar(Impl& im, int v, float* out, int depth = 0) {
    if (depth > 8 || v < 0) return false;
    const Value& val = im.vals[v];
    if (val.kind == Value::CONSTV) {
        if (val.ct.numel() != 1) return false;
        if (val.ct.data_type == 1 && val.ct.data.size() >= 4) {
            *out = *(const float*)val.ct.data.data();
            return true;
        }
        if (val.ct.data_type == 11 && val.ct.data.size() >= 8) {
            *out = (float)*(const double*)val.ct.data.data();
            return true;
        }
        if (val.ct.data_type == 7 && val.ct.data.size() >= 8) {
            *out = (float)*(const int64_t*)val.ct.data.data();
            return true;
        }
        if (val.ct.data_type == 6 && val.ct.data.size() >= 4) {
            *out = (float)*(const int32_t*)val.ct.data.data();
            return true;
        }
        return false;
    }
    // A 1-element SYMVAL whose element is a constant expression
    // (the attention scale bottoms out at Sqrt(Shape(..)[-1])).
    if (val.kind == Value::SYMVAL) {
        int64_t c;
        if (val.selems.size() == 1 && econst(val.selems[0], c)) {
            *out = (float)c;
            return true;
        }
        return false;
    }
    if (val.kind != Value::RT) return false;
    // Views of a scalar (Reshape/Unsqueeze/Transpose of a 1-element
    // value) carry the same scalar; resolving through them is safe
    // because the base only resolves when it is itself a numel-1
    // constant expression (a Slice out of a big tensor never does).
    if (val.view_kind != 0) {
        if (val.base < 0) return false;
        return resolve_scalar(im, val.base, out, depth + 1);
    }
    int si = val.def_step;
    if (si < 0 || si >= (int)im.steps.size()) return false;
    const Step& st = im.steps[si];
    if (st.out.size() != 1 || st.out[0] != v) return false;
    float a, b;
    if (st.k == SK::CAST && st.in.size() == 1)
        return resolve_scalar(im, st.in[0], out, depth + 1);
    if (st.k == SK::UNOP && st.in.size() == 1) {
        if (!resolve_scalar(im, st.in[0], &a, depth + 1)) return false;
        if (st.uop == UnOp::SQRT) {
            *out = std::sqrt(a);
            return true;
        }
        if (st.uop == UnOp::NEG) {
            *out = -a;
            return true;
        }
        return false;
    }
    if (st.k == SK::BINOP && st.in.size() == 2) {
        if (!resolve_scalar(im, st.in[0], &a, depth + 1) ||
            !resolve_scalar(im, st.in[1], &b, depth + 1))
            return false;
        if (st.bop == BinOp::DIV) {
            *out = a / b;
            return true;
        }
        if (st.bop == BinOp::MUL) {
            *out = a * b;
            return true;
        }
        if (st.bop == BinOp::ADD) {
            *out = a + b;
            return true;
        }
        if (st.bop == BinOp::SUB) {
            *out = a - b;
            return true;
        }
        return false;
    }
    return false;
}

// Mirror fold: "MatMul(Mul(x, scalar-const-expr), ..)" -- the
// scale moves into the matmul's epilogue (out_scale, or the Gemm
// alpha when it has no C input) and the Mul step is erased. This
// is the attention Q/K scaling shape in the GPT stage plans: the
// scalar arrives as Sqrt(Cast(Constant)), and the standalone Mul
// walks the [1, heads, L, dim] tensor in a gathered layout. The
// consumer path from the Mul output to the matmul input slot
// (through views) must be unique; the first view on that path is
// re-based onto the Mul's tensor input (same shape).
static bool fold_one_scalar_mul_pre(Impl& im) {
    auto& vals = im.vals;
    auto& steps = im.steps;
    std::vector<int> view_kids(vals.size(), 0);
    std::vector<std::vector<int>> kids(vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
        const Value& v = vals[i];
        if (v.kind == Value::RT && v.view_kind != 0 && v.base >= 0) {
            view_kids[v.base]++;
            kids[v.base].push_back((int)i);
        }
    }
    for (int s2i = 0; s2i < (int)steps.size(); s2i++) {
        Step& s2 = steps[s2i];
        if (s2.keep) continue;
        if (s2.k != SK::BINOP || s2.bop != BinOp::MUL || s2.in.size() != 2 ||
            s2.out.size() != 1)
            continue;
        int vx = -1;
        float scale = 0;
        for (int t = 0; t < 2; t++) {
            float sc;
            if (resolve_scalar(im, s2.in[t], &sc)) {
                scale = sc;
                vx = s2.in[1 - t];
            }
        }
        if (vx < 0) continue;
        if (vals[vx].kind != Value::RT) continue;
        int vy = s2.out[0];
        if (vals[vy].kind != Value::RT || vals[vy].view_kind != 0 || vals[vy].is_output)
            continue;
        bool meta_ref = false;
        for (auto& v : vals)
            for (int m : {v.meta0, v.meta1, v.meta2, v.meta3})
                if (m == vy) meta_ref = true;
        if (meta_ref) continue;
        // Walk the unique consumer path from vy: at each value,
        // exactly one continuation (one view child XOR one step
        // input slot) is allowed.
        int cur = vy, first_view = -1, cons_step = -1, cons_slot = -1;
        bool bad = false;
        for (int hops = 0; hops < 16 && cons_step < 0 && !bad; hops++) {
            int slot_hits = 0, fs = -1, fl = -1;
            for (int s = 0; s < (int)steps.size(); s++) {
                if (s == s2i) continue;
                for (int l = 0; l < (int)steps[s].in.size(); l++)
                    if (steps[s].in[l] == cur) {
                        slot_hits++;
                        fs = s;
                        fl = l;
                    }
            }
            int nk = view_kids[cur];
            if (nk == 1 && slot_hits == 0) {
                if (first_view < 0) first_view = kids[cur][0];
                cur = kids[cur][0];
            } else if (nk == 0 && slot_hits == 1) {
                cons_step = fs;
                cons_slot = fl;
            } else {
                bad = true;
            }
        }
        if (bad || cons_step < 0) continue;
        Step& s3 = steps[cons_step];
        if (s3.k == SK::MATMUL) {
            s3.out_scale *= scale;
        } else if (s3.k == SK::GEMM && s3.in.size() == 2) {
            s3.alpha *= scale;
        } else {
            continue;
        }
        if (first_view < 0) {
            s3.in[cons_slot] = vx;
        } else {
            vals[first_view].base = vx;
        }
        vals[vy].def_step = -2;
        steps.erase(steps.begin() + s2i);
        for (auto& v : vals)
            if (v.def_step > s2i) v.def_step--;
        return true;
    }
    return false;
}

// LeakyReLU epilogue fusion: a LeakyRelu whose input is the sole
// output of a Conv / ConvTranspose / packed-conv / residual-Add
// step, with no other consumer, no view children and no meta
// reference, moves into the producer as an epilogue (the ORT
// FusedConv pattern). The standalone step is erased with the same
// bookkeeping the scalar folds use.
static bool fuse_one_leaky(Impl& im) {
    auto& steps = im.steps;
    auto& vals = im.vals;
    std::vector<int> uses(vals.size(), 0), view_kids(vals.size(), 0);
    for (auto& s : steps)
        for (int v : s.in) if (v >= 0) uses[v]++;
    for (auto& v : vals) {
        if (v.kind == Value::RT && v.view_kind != 0 && v.base >= 0) view_kids[v.base]++;
        for (int m : {v.meta0, v.meta1, v.meta2, v.meta3})
            if (m >= 0) uses[m]++;
    }
    for (size_t ui = 0; ui < steps.size(); ui++) {
        Step& u = steps[ui];
        if (u.keep) continue;
        if (u.k != SK::UNOP || u.uop != UnOp::LEAKY || u.in.size() != 1 || u.out.size() != 1) continue;
        int vx = u.in[0], vy = u.out[0];
        if (vals[vx].kind != Value::RT || vals[vx].view_kind != 0 || vals[vx].is_output) continue;
        if (uses[vx] != 1 || view_kids[vx] != 0) continue;
        int pi = vals[vx].def_step;
        if (pi < 0 || pi >= (int)steps.size()) continue;
        Step& p = steps[pi];
        if (p.keep) continue;
        if (p.out.size() != 1 || p.out[0] != vx || p.epi_act != 0) continue;
        bool ok = p.k == SK::PCONV || p.k == SK::CONV || p.k == SK::CONVT ||
                  (p.k == SK::BINOP && p.bop == BinOp::ADD);
        if (!ok) continue;
        p.epi_act = 1;
        p.epi_alpha = u.alpha;
        for (auto& s : steps)
            for (int& v : s.in)
                if (v == vy) v = vx;
        for (int& v : im.out_vals)
            if (v == vy) v = vx;
        if (vals[vy].is_output) vals[vx].is_output = true;
        vals[vy].is_output = false;
        vals[vy].def_step = -2;
        steps.erase(steps.begin() + ui);
        for (auto& v : vals)
            if (v.def_step > (int)ui) v.def_step--;
        return true;
    }
    return false;
}

// Dead-step elimination, run after the scalar folds: the folds
// rewire the scale chains' consumers onto the pre-Mul values and
// leave the chains' own steps (Shape, Cast, Sqrt, Div, ...) behind
// with no live reader. A step is removed when its kind is pure (no
// RNG state, no side effects) and none of its outputs is used -- by
// another step, as the base of a used view, through a meta
// reference, or as a graph output. Iterated to a fixpoint so whole
// chains collapse. Removed values follow the folds' convention
// (def_step = -2) and later def_steps shift down.
static void dce_pure_steps(Impl& im) {
    auto& steps = im.steps;
    auto& vals = im.vals;
    auto pure = [](SK k) {
        switch (k) {
            case SK::UNOP: case SK::BINOP: case SK::CAST: case SK::CONSTSHAPE:
            case SK::COPY: case SK::EXPAND: case SK::WHERE: case SK::REDUCESUM:
            case SK::CUMSUM: case SK::GATHER: case SK::GATHER_EL: case SK::CONCAT:
            case SK::PAD:
                return true;
            default:
                return false;
        }
    };
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<char> used(vals.size(), 0);
        for (auto& s : steps)
            for (int v : s.in) if (v >= 0) used[v] = 1;
        for (int v : im.out_vals) used[v] = 1;
        // Branch/body outputs are consumed by their owning
        // IF/LOOP step's copy-out, not by a step input: mark
        // them used, or a pure producer whose only reader is
        // a branch-internal view (Pad -> Squeeze in
        // prompt_encoder) is erased and the view resolves
        // against a stale bind descriptor.
        for (auto& s : steps) {
            if (s.k == SK::IF) {
                for (int v : s.if_t_outs) if (v >= 0) used[v] = 1;
                for (int v : s.if_e_outs) if (v >= 0) used[v] = 1;
            }
            if (s.k == SK::LOOP)
                for (int v : s.lp_body_outs) if (v >= 0) used[v] = 1;
        }
        for (auto& v : vals)
            for (int m : {v.meta0, v.meta1, v.meta2, v.meta3})
                if (m >= 0) used[m] = 1;
        for (int i = (int)vals.size() - 1; i >= 0; i--)
            if (used[i] && vals[i].kind == Value::RT && vals[i].view_kind != 0 && vals[i].base >= 0)
                used[vals[i].base] = 1;
        for (int s = (int)steps.size() - 1; s >= 0; s--) {
            Step& st = steps[s];
            if (st.keep) continue;
            if (!pure(st.k)) continue;
            bool dead = true;
            for (int v : st.out)
                if (used[v] || vals[v].is_output) dead = false;
            if (!dead) continue;
            for (int v : st.out) vals[v].def_step = -2;
            steps.erase(steps.begin() + s);
            for (auto& v : vals)
                if (v.def_step > s) v.def_step--;
            changed = true;
            break;  // indices shifted; recompute use counts
        }
    }
}

static bool fuse_one_attn(Impl& im);  // defined below, with the folds

// Stateful KV feedback pairs: a Concat whose first input is a
// graph input, whose second input is the new slice, and whose
// output is a graph output (axis 0, fp32). The stage graphs have
// 48 (24 layers x K/V). Pair order is step order, identical for
// every plan compiled from the same graph, which is what lets
// two plans share one KvCache by slot index.
static void detect_kv_pairs(Impl& im) {
    for (size_t s = 0; s < im.steps.size(); s++) {
        Step& st = im.steps[s];
        if (st.k != SK::CONCAT || st.in.size() != 2 || st.out.size() != 1) continue;
        if (st.axis != 0) continue;
        int iv = st.in[0], ov = st.out[0];
        const Value& vin = im.vals[iv];
        if (!vin.is_input || vin.view_kind != 0 || vin.dtype != 1) continue;
        if (!im.vals[ov].is_output) continue;
        int in_idx = -1;
        for (size_t i = 0; i < im.in_vals.size(); i++)
            if (im.in_vals[i] == iv) { in_idx = (int)i; break; }
        if (in_idx < 0) continue;
        st.kv_slot = (int)im.kv_pairs.size();
        im.kv_pairs.push_back({iv, ov, in_idx});
    }
}

std::shared_ptr<Plan> Plan::compile(const onnxm::Graph& g, std::string label, bool fp16_weights) {
    auto plan = std::make_shared<Plan>();
    auto im = std::make_shared<Impl>();
    plan->impl_ = im;
    im->want_f16 = fp16_weights && cpu::features().avx2 && cpu::features().fma &&
                   cpu::features().f16c;
    Compiler C(*im, g, label);
    // Initializers first (they may also be listed as inputs; parse_graph
    // already filtered them out of g.inputs).
    for (auto& t : g.initializers) {
        Value v;
        v.kind = Value::CONSTV;
        v.ct = decode_f16_const(t);
        im->vals.push_back(std::move(v));
        C.reg(t.name, (int)im->vals.size() - 1);
    }
    // Runtime inputs; packed weight slots (__pk/__sc/__ze) are not
    // values: PackedMatMul steps bind them through the lookup.
    for (auto& vi : g.inputs) {
        if (vi.name.size() >= 4) {
            std::string suf = vi.name.substr(vi.name.size() - 4);
            if (suf == "__pk" || suf == "__sc" || suf == "__ze") continue;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = vi.elem_type;
        v.is_input = true;
        v.def_step = -1;
        for (size_t d = 0; d < vi.dims.size(); d++) {
            if (vi.dims[d] >= 0) v.rshape.push_back(ec(vi.dims[d]));
            else {
                int s = im->nsyms++;
                im->sym_origin.push_back({(int)im->vals.size(), (int)d});
                v.rshape.push_back(esym(s));
            }
        }
        im->vals.push_back(std::move(v));
        int id = (int)im->vals.size() - 1;
        C.reg(vi.name, id);
        im->in_names.push_back(vi.name);
        im->in_vals.push_back(id);
    }
    for (auto& n : g.nodes) {
        lower_node(C, n);
        if (std::getenv("FE_DEBUG") && !n.outputs.empty() &&
            n.name.find("attn_layers.0") != std::string::npos) {
            auto it = im->by_name.find(n.outputs[0]);
            if (it != im->by_name.end())
                std::fprintf(stderr, "[kind] %s %s -> kind=%d\n", n.op.c_str(), n.name.c_str(),
                             (int)im->vals[it->second].kind);
        }
    }
    for (auto& vo : g.outputs) {
        int id = C.vid(vo.name);
        im->vals[id].is_output = true;
        im->out_names.push_back(vo.name);
        im->out_vals.push_back(id);
    }
    // Dyn propagation through steps: a step reading a tensor
    // whose shape is only known at execution produces outputs
    // whose shapes are only known at execution; the runner
    // sizes them from the input descriptors (dyn_shape_for).
    // Steps are in lowering order (bodies precede their
    // Loop / If step), so one forward pass settles it.
    for (auto& s : im->steps) {
        if (s.k == SK::LOOP) {
            for (size_t j = 0; j < s.out.size() && j + 1 < s.lp_body_outs.size(); j++)
                if (im->vals[s.lp_body_outs[j + 1]].dyn) im->vals[s.out[j]].dyn = true;
            continue;
        }
        if (s.k == SK::IF) {
            for (size_t j = 0; j < s.out.size(); j++)
                if ((j < s.if_t_outs.size() && im->vals[s.if_t_outs[j]].dyn) ||
                    (j < s.if_e_outs.size() && im->vals[s.if_e_outs[j]].dyn))
                    im->vals[s.out[j]].dyn = true;
            continue;
        }
        bool in_dyn = false;
        for (int v : s.in)
            if (v >= 0 && im->vals[v].kind == Value::RT && !im->vals[v].is_seq && im->vals[v].dyn) in_dyn = true;
        if (!in_dyn) continue;
        for (int v : s.out)
            if (im->vals[v].kind == Value::RT && !im->vals[v].is_seq && im->vals[v].view_kind == 0)
                im->vals[v].dyn = true;
    }
    while (fold_one_scalar_mul(*im)) {
    }
    while (fold_one_scalar_mul_pre(*im)) {
    }
    while (fuse_one_leaky(*im)) {
    }
    while (fuse_one_attn(*im)) {
    }
    dce_pure_steps(*im);
    // Dyn propagation: a view of a dyn base is itself dyn (its
    // bind-time shape/buffer cannot be trusted; the loop runner
    // resolves it per iteration).
    for (auto& v : im->vals)
        if (v.kind == Value::RT && v.view_kind != 0 && v.base >= 0 && im->vals[v.base].dyn)
            v.dyn = true;
    // Resolve LOOP body ranges from uids to step indices (the
    // passes above may have erased or rewritten steps around the
    // bodies; body steps are keep-flagged and survive).
    {
        std::map<int, int> by_uid;
        for (int i = 0; i < (int)im->steps.size(); i++) by_uid[im->steps[i].uid] = i;
        auto resolve_range = [&](int u0, int u1, int self_uid, int& b0, int& b1, const char* what) {
            if (u0 == u1) {
                auto it = by_uid.find(self_uid);
                b0 = b1 = it == by_uid.end() ? 0 : it->second;
                return;
            }
            auto a = by_uid.find(u0), b = by_uid.find(u1 - 1);
            if (a == by_uid.end() || b == by_uid.end()) fail(what);
            b0 = a->second;
            b1 = b->second + 1;
        };
        for (auto& s : im->steps) {
            if (s.k == SK::LOOP)
                resolve_range(s.lp_u0, s.lp_u1, s.uid, s.lp_b0, s.lp_b1, "loop body steps lost in passes");
            if (s.k == SK::IF) {
                resolve_range(s.if_u0, s.if_u1, s.uid, s.if_b0, s.if_b1, "if then-branch steps lost in passes");
                resolve_range(s.if_eu0, s.if_eu1, s.uid, s.if_eb0, s.if_eb1, "if else-branch steps lost in passes");
            }
        }
    }
    detect_kv_pairs(*im);
    // Liveness: last use over steps, then propagate through views
    // (reverse creation order: views are created after their bases, and
    // a view's last use must extend its base's).
    int nsteps = (int)im->steps.size();
    for (int s = 0; s < nsteps; s++)
        for (int v : im->steps[s].in)
            if (v >= 0) im->vals[v].last_use = std::max(im->vals[v].last_use, s);
    for (auto& v : im->vals)
        if (v.is_output) v.last_use = nsteps + 1;
    // Loop slots and body outputs, and If branch outputs, are
    // read/written by the runner itself, outside the step list:
    // keep their arena blocks live until the LOOP / IF step. This
    // runs before the view propagation below so that a body or
    // branch output which is a view extends its base's life too.
    for (int si = 0; si < nsteps; si++) {
        const Step& s = im->steps[si];
        if (s.k == SK::LOOP) {
            for (int v : s.lp_slots) im->vals[v].last_use = std::max(im->vals[v].last_use, si);
            for (int v : s.lp_body_outs) im->vals[v].last_use = std::max(im->vals[v].last_use, si);
        }
        if (s.k == SK::IF) {
            for (int v : s.if_t_outs) im->vals[v].last_use = std::max(im->vals[v].last_use, si);
            for (int v : s.if_e_outs) im->vals[v].last_use = std::max(im->vals[v].last_use, si);
        }
    }
    for (int i = (int)im->vals.size() - 1; i >= 0; i--) {
        Value& v = im->vals[i];
        if (v.kind == Value::RT && v.view_kind != 0 && v.base >= 0)
            im->vals[v.base].last_use = std::max(im->vals[v.base].last_use, v.last_use);
    }
    plan->in_names_ = im->in_names;
    plan->out_names_ = im->out_names;
    plan->n_steps_ = im->steps.size();
    return plan;
}

// -------------------------------------------------------------- bind ---
// Read integer metadata (slice bounds, reshape target) from a resolved
// value at bind time. Exact integer reads: routing through double (as
// read_d does) corrupts sentinel values like INT64_MAX slice ends.
std::vector<int64_t> read_meta_ints(const RT& t) {
    std::vector<int64_t> r;
    odometer(t.shape, [&](const std::vector<int64_t>& c) {
        int64_t off = 0;
        for (size_t d = 0; d < c.size(); d++) off += c[d] * t.strides[d];
        int64_t v = 0;
        switch (t.dtype) {
            case 7: memcpy(&v, t.p + off * 8, 8); break;
            case 6: { int32_t x; memcpy(&x, t.p + off * 4, 4); v = x; break; }
            case 1: { float x; memcpy(&x, t.p + off * 4, 4); v = (int64_t)x; break; }
            default: v = (int64_t)read_d(t, off); break;
        }
        r.push_back(v);
    });
    return r;
}

// FE_PROFILE accumulators (see Plan::run). Kind order matches enum SK.
// Totals are kept per plan label so a decode step (stage plans) can be
// read separately from prefill (first) and the encoder.
struct ProfAcc { double us[44] = {}; long n[44] = {}; };
static std::map<std::string, ProfAcc> g_prof;
static const char* g_prof_names[44] = {"MATMUL", "GEMM", "PACKED", "CONV", "LN", "SOFTMAX",
                                       "REDUCESUM", "ARGMAX", "TOPK", "CUMSUM", "GATHER",
                                       "GATHER_EL", "SCATTER_EL", "CONCAT", "CONSTSHAPE",
                                       "RANDNORM", "BINOP", "UNOP", "CAST", "WHERE", "COPY",
                                       "PCONV", "PGATHER", "CONVT", "PAD", "EXPAND", "ATTN",
                                       "INSTNORM", "SEQ_EMPTY", "SEQ_SPLIT", "SEQ_AT",
                                       "SEQ_INSERT", "SEQ_CONCAT", "LOOP", "IF",
                                       "SLICE_RT", "RANGE", "STFT", "TILE", "PRELU", "DFT",
                                       "BNORM", "CLIP", "SHAPE"};
static std::map<std::string, double> g_prof_bind, g_prof_exec;
static std::map<std::string, long> g_prof_runs;
static void prof_print() {
    for (auto& kv : g_prof) {
        auto b = g_prof_bind.find(kv.first), e = g_prof_exec.find(kv.first);
        if (b != g_prof_bind.end())
            std::fprintf(stderr, "[FE_PROFILE] plan=%s runs=%ld bind %.1f ms exec %.1f ms\n", kv.first.c_str(),
                         g_prof_runs[kv.first], b->second / 1000.0, e->second / 1000.0);
        std::fprintf(stderr, "[FE_PROFILE] per-kind totals, plan=%s:\n", kv.first.c_str());
        for (int i = 0; i < 27; i++)
            if (kv.second.n[i]) std::fprintf(stderr, "  %-10s %8ld calls %12.1f ms\n", g_prof_names[i], kv.second.n[i], kv.second.us[i] / 1000.0);
    }
    std::fprintf(stderr, "[FE_PROFILE] binop paths (all plans):\n");
    for (int i = 0; i < 6; i++)
        if (g_binop_paths[i].n)
            std::fprintf(stderr, "  %-12s %8ld calls %12.1f ms  %14lld elems  max %.1f ms\n",
                         g_binop_path_names[i], g_binop_paths[i].n, g_binop_paths[i].us / 1000.0,
                         g_binop_paths[i].elems, g_binop_paths[i].max_us / 1000.0);
}
static void prof_add(const std::string& label, int kind, double us) {
    static bool reg = (std::atexit(prof_print), true);
    (void)reg;
    if (kind >= 0 && kind < 44) { ProfAcc& a = g_prof[label]; a.us[kind] += us; a.n[kind]++; }
}

void Plan::release_arena() {
    Impl& im = *std::static_pointer_cast<Impl>(impl_);
    std::vector<uint8_t>().swap(im.arena);
    std::vector<uint8_t>().swap(im.out_arena);
    arena_bytes_ = 0;
}

// Fuse MATMUL -> SOFTMAX(last axis) -> MATMUL into one ATTN step
// (the GPT attention block; 24 sites per stage plan, 24 in the
// encoder and first-stage plans). The intermediate scores/probs
// values lose their only consumers. Bit-identical by construction
// (see k_attn). Conservative: single-use intermediates, no view
// children, no graph outputs, no epilogue scales on either matmul.
static bool fuse_one_attn(Impl& im) {
    auto& steps = im.steps;
    auto& vals = im.vals;
    std::vector<int> uses(vals.size(), 0), view_kids(vals.size(), 0);
    for (auto& s : steps)
        for (int v : s.in) if (v >= 0) uses[v]++;
    for (auto& v : vals) {
        if (v.kind == Value::RT && v.view_kind != 0 && v.base >= 0) view_kids[v.base]++;
        for (int m : {v.meta0, v.meta1, v.meta2, v.meta3})
            if (m >= 0) uses[m]++;
    }
    for (size_t i = 0; i < steps.size(); i++) {
        Step& s0 = steps[i];
        if (s0.keep) continue;
        if (s0.k != SK::MATMUL || s0.in.size() != 2 || s0.out.size() != 1) continue;
        // s0.out_scale (the attention score scale, folded into the
        // scores matmul by the pre-fold) stays on the ATTN step;
        // k_attn applies it to the scores before the softmax with
        // the same double multiply the epilogue used.
        int vscores = s0.out[0];
        if (vals[vscores].is_output || uses[vscores] != 1 || view_kids[vscores] != 0) continue;
        int js = -1;
        for (size_t j = 0; j < steps.size(); j++)
            if (steps[j].k == SK::SOFTMAX && steps[j].in.size() == 1 &&
                steps[j].in[0] == vscores && steps[j].out.size() == 1) {
                js = (int)j;
                break;
            }
        if (js < 0) continue;
        if (steps[js].keep) continue;
        if (!(steps[js].axis == -1 || steps[js].axis == 3)) continue;
        int vprobs = steps[js].out[0];
        if (vals[vprobs].is_output || uses[vprobs] != 1 || view_kids[vprobs] != 0) continue;
        int ms = -1;
        for (size_t j = 0; j < steps.size(); j++)
            if (steps[j].k == SK::MATMUL && steps[j].in.size() == 2 &&
                steps[j].in[0] == vprobs && steps[j].out.size() == 1) {
                ms = (int)j;
                break;
            }
        if (ms < 0) continue;
        if (steps[ms].keep) continue;
        Step& s2 = steps[ms];
        if (s2.out_scale != 1.0f) continue;
        int vctx = s2.out[0];
        int vq = s0.in[0], vk = s0.in[1], vv = s2.in[1];
        // k_attn is specialized to head dim 32 (the GPT graphs).
        // Fuse only when all three operand shapes are compile-time
        // constants with that layout; anything else (HuBERT's 64,
        // symbolic dims) keeps the plain matmul/softmax steps.
        auto const_dim = [&](int vid, size_t d, int64_t& out) {
            const auto& rs = vals[vid].rshape;
            return rs.size() > d && econst(rs[d], out);
        };
        int64_t dq, dk, dv;
        if (!const_dim(vq, 3, dq) || dq != 32) continue;
        if (!const_dim(vk, 2, dk) || dk != 32) continue;
        if (!const_dim(vv, 3, dv) || dv != 32) continue;
        s0.k = SK::ATTN;
        s0.in = {vq, vk, vv};
        s0.out = {vctx};
        vals[vctx].def_step = (int)i;
        vals[vscores].def_step = -2;
        vals[vprobs].def_step = -2;
        int e1 = std::max(js, ms), e2 = std::min(js, ms);
        steps.erase(steps.begin() + e1);
        steps.erase(steps.begin() + e2);
        for (auto& v : vals) {
            if (v.def_step == e1 || v.def_step == e2) {
                v.def_step = -2;
            } else {
                int d = v.def_step;
                if (d > e2) d--;
                if (d > e1) d--;
                v.def_step = d;
            }
        }
        return true;
    }
    return false;
}

std::map<std::string, Tensor> Plan::run(const std::map<std::string, Tensor>& inputs) {
    std::vector<Tensor> ins(in_names_.size());
    for (size_t i = 0; i < in_names_.size(); i++) {
        auto it = inputs.find(in_names_[i]);
        if (it != inputs.end()) ins[i] = it->second;
    }
    std::vector<Tensor> outs = run_vec(ins);
    std::map<std::string, Tensor> out;
    for (size_t i = 0; i < out_names_.size(); i++) out[out_names_[i]] = outs[i];
    return out;
}

std::vector<Tensor> Plan::run_vec(const std::vector<Tensor>& inputs) {
    Impl& im = *std::static_pointer_cast<Impl>(impl_);
    if (inputs.size() != im.in_names.size()) fail("run_vec: input count mismatch");
    im.lookup = lookup_;
    static const bool no_kv = std::getenv("POTATO_NO_KV") != nullptr;
    im.kv = no_kv ? nullptr : kv_cache_;
    static const bool prof_run = std::getenv("FE_PROFILE") != nullptr;
    auto run_t0 = std::chrono::steady_clock::now();
    int nv = (int)im.vals.size();
    // Bind workspace (Impl members; capacities persist across runs).
    if ((int)im.bres.size() != nv) {
        im.bres.assign(nv, RT{});
        im.bdone.assign(nv, 0);
        im.bcshape.assign(nv, {});
        im.boff_w.assign(nv, 0);
        im.boff_o.assign(nv, 0);
        im.sym_bufs.assign(nv, {});
        im.bseqs.assign(nv, {});
    }
    for (auto& s : im.bseqs) s.clear();
    im.dyn_reset();
    // Symbol values from the caller's input shapes.
    std::vector<int64_t>& sv = im.bsv;
    sv.assign(im.nsyms, 0);
    std::vector<RT>& res = im.bres;
    std::vector<char>& done = im.bdone;
    std::fill(done.begin(), done.end(), 0);
    // Stateful KV pasts: pre-bind from the shared cache. A supplied
    // input is copied in when the slot is empty or its length
    // disagrees; a missing input binds the buffer view directly.
    // Capacity grows at bind time only, before any RT of this run
    // points into the buffer.
    std::vector<char> kv_out;
    if (im.kv && !im.kv_pairs.empty()) {
        if (im.kv->slots.size() < im.kv_pairs.size()) im.kv->slots.resize(im.kv_pairs.size());
        kv_out.assign(nv, 0);
        for (auto& pr : im.kv_pairs) kv_out[pr.out_vid] = 1;
        for (size_t pi = 0; pi < im.kv_pairs.size(); pi++) {
            const auto& pr = im.kv_pairs[pi];
            KvCache::Slot& slot = im.kv->slots[pi];
            const Tensor& t = inputs[pr.in_idx];
            if (t.data != nullptr) {
                if (t.dtype != 1) fail("kv: past input not fp32");
                int64_t rows = t.shape.empty() ? 0 : t.shape[0];
                int64_t rowf = rows > 0 ? t.numel() / rows : 0;
                if (slot.len < 0 || slot.len != rows) {
                    // POTATO_KV_INIT_CAP (rows) overrides the
                    // initial capacity; small values exercise the
                    // growth path in tests.
                    static const int64_t init_cap = [] {
                        const char* e = std::getenv("POTATO_KV_INIT_CAP");
                        return e ? std::atoll(e) : 1024;
                    }();
                    slot.reserve(std::max<int64_t>(rows * 2, init_cap), rowf);
                    // The source may be a strided view (the first
                    // plan's presents are row-strided slices of the
                    // QKV projection). Rows are contiguous blocks:
                    // copy row by row when the inner layout is
                    // packed, else fall back to the generic copy.
                    std::vector<int64_t> sstr = t.strides.empty() ? rowmajor(t.shape) : t.strides;
                    std::vector<int64_t> rm = rowmajor(t.shape);
                    bool rows_packed = sstr.size() == t.shape.size();
                    for (size_t d = 1; d < sstr.size() && rows_packed; d++)
                        rows_packed = sstr[d] == rm[d];
                    if (rows_packed) {
                        const float* sp = (const float*)t.data;
                        int64_t sstride = sstr.empty() ? rowf : sstr[0];
                        for (int64_t r = 0; r < rows; r++)
                            std::memcpy(slot.buf.get() + (size_t)r * (size_t)rowf,
                                        sp + (size_t)r * (size_t)sstride,
                                        (size_t)rowf * sizeof(float));
                    } else {
                        RT src;
                        src.dtype = t.dtype;
                        src.p = const_cast<uint8_t*>(t.data);
                        src.shape = t.shape;
                        src.strides = sstr;
                        RT dst;
                        dst.dtype = 1;
                        dst.p = (uint8_t*)slot.buf.get();
                        dst.shape = t.shape;
                        dst.strides = rm;
                        copy_tensor(dst, src);
                    }
                    slot.len = rows;
                    slot.shape = t.shape;
                }
            }
            if (slot.len < 0) fail("kv: past input missing and cache empty");
            int64_t rowf = 1;
            for (size_t d = 1; d < slot.shape.size(); d++) rowf *= slot.shape[d];
            if (slot.cap_rows < slot.len + 16)
                slot.reserve(std::max<int64_t>(slot.cap_rows * 2, slot.len + 1024), rowf);
            RT r;
            r.dtype = 1;
            r.p = (uint8_t*)slot.buf.get();
            r.shape = slot.shape;
            r.shape[0] = slot.len;
            r.strides = rowmajor(r.shape);
            res[pr.in_vid] = r;
            done[pr.in_vid] = 1;
        }
    }
    // Inputs (indexed parallel to in_names; null data = missing).
    for (size_t i = 0; i < im.in_vals.size(); i++) {
        int vid = im.in_vals[i];
        if (done[vid]) continue;  // kv pre-bound above
        const Tensor& t = inputs[i];
        if (t.data == nullptr) {
            if (im.vals[vid].last_use >= 0 || im.vals[vid].is_output)
                fail("missing input: " + im.in_names[i]);
            res[vid] = RT{};
            done[vid] = 1;
            continue;
        }
        RT r;
        r.dtype = t.dtype;
        r.p = const_cast<uint8_t*>(t.data);
        r.shape = t.shape;
        r.strides = t.strides.empty() ? rowmajor(r.shape) : t.strides;
        res[vid] = r;
        done[vid] = 1;
    }
    for (int s = 0; s < im.nsyms; s++) {
        auto [vid, dim] = im.sym_origin[s];
        // Placeholder symbols (origin -1: split sizes, dyn expand
        // dims) evaluate to 0; their true values only exist at
        // execution time inside loop bodies.
        sv[s] = (vid < 0 || dim < 0) ? 0
                : (dim < (int)res[vid].shape.size() ? res[vid].shape[dim] : 0);
    }
    // Consts and symbolic values. (sym_bufs is sized by the
    // workspace guard above; SYMVAL buffers are fully rewritten
    // below on every run.)
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind == Value::CONSTV) {
            res[i] = tensor_of_const(v.ct);
            done[i] = 1;
        } else if (v.kind == Value::SYMVAL) {
            auto& buf = im.sym_bufs[i];
            buf.resize((size_t)numel_of(v.sshape) * onnxm::elem_size(v.sdt));
            RT r;
            r.dtype = v.sdt;
            r.p = buf.data();
            r.shape = v.sshape;
            r.strides = rowmajor(v.sshape);
            for (size_t e = 0; e < v.selems.size(); e++) write_d(r, (int64_t)e, (double)eeval(v.selems[e], sv));
            res[i] = r;
            done[i] = 1;
        }
    }
    // Concrete shapes for RT values (from exprs).
    std::vector<std::vector<int64_t>>& cshape = im.bcshape;
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind != Value::RT || v.is_seq) continue;
        cshape[i].clear();
        for (auto& e : v.rshape) cshape[i].push_back(eeval(e, sv));
    }
    // Values created inside Loop bodies: their views are resolved
    // per iteration by the runner, not here.
    std::vector<char> in_body(nv, 0);
    for (auto& s : im.steps)
        if (s.k == SK::LOOP)
            for (int i = s.lp_v0; i < s.lp_v1 && i < nv; i++) in_body[i] = 1;
    // Arena assignment for materialized RT values (not views / inputs).
    using Alloc = Impl::BAlloc;
    std::vector<Alloc>& allocs = im.ballocs;
    allocs.clear();
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind != Value::RT || v.view_kind != 0 || v.is_input) continue;
        if (v.is_seq || v.dyn) continue;  // sequences live in bseqs; dyn buffers are execution-allocated
        if (v.def_step == -2) continue;  // degenerate LN side output
        if (!kv_out.empty() && kv_out[i]) continue;  // stateful KV present: cache-bound below
        int64_t sz = 1;
        for (auto d : cshape[i]) sz *= d;
        sz *= (int64_t)onnxm::elem_size(v.dtype);
        allocs.push_back({i, sz, std::max(v.def_step, 0), v.last_use, v.is_output});
    }
    auto plan_arena = [&](std::vector<Alloc>& as, std::vector<uint8_t>& buf,
                          std::vector<size_t>& offs) {
        std::stable_sort(as.begin(), as.end(), [](const Alloc& a, const Alloc& b) { return a.def < b.def; });
        using Block = Impl::BBlock;
        std::vector<Block>& blocks = im.bblocks;
        blocks.clear();
        size_t total = 0;
        for (auto& a : as) {
            size_t need = (size_t)((a.size + 63) & ~63LL);
            if (need == 0) need = 64;
            int found = -1;
            for (size_t bi = 0; bi < blocks.size(); bi++) {
                if (blocks[bi].size >= need && blocks[bi].use < a.def) {
                    found = (int)bi;
                    break;
                }
            }
            if (found >= 0) {
                offs[a.vid] = blocks[found].off;
                int old_use = blocks[found].use;
                blocks[found].use = a.use;
                if (blocks[found].size > need + 4096) {
                    // Split large leftovers back into the free pool; the
                    // leftover keeps the old block's (earlier) last use.
                    blocks.push_back({blocks[found].off + need, blocks[found].size - need, old_use});
                    blocks[found].size = need;
                }
            } else {
                offs[a.vid] = total;
                blocks.push_back({total, need, a.use});
                total += need;
            }
        }
        if (buf.size() < total) buf.resize(total);
        return total;
    };
    std::vector<Alloc>& work = im.bwork;
    std::vector<Alloc>& outa = im.bouta;
    work.clear();
    outa.clear();
    for (auto& a : allocs) (a.out ? outa : work).push_back(a);
    std::vector<size_t>& off_w = im.boff_w;
    std::vector<size_t>& off_o = im.boff_o;
    size_t arena_total = plan_arena(work, im.arena, off_w);
    size_t out_total = plan_arena(outa, im.out_arena, off_o);
#if defined(__linux__)
    // POTATO_HUGEPAGES=1: ask THP to back the big arenas with 2 MB
    // pages (advisory; the streaming kernels walk these end to end
    // every step). Experiment gate, default off.
    static const bool hugepages = std::getenv("POTATO_HUGEPAGES") != nullptr;
    if (hugepages) {
        if (im.arena.size() >= (2u << 20)) madvise(im.arena.data(), im.arena.size(), MADV_HUGEPAGE);
        if (im.out_arena.size() >= (2u << 20)) madvise(im.out_arena.data(), im.out_arena.size(), MADV_HUGEPAGE);
    }
#endif
    arena_bytes_ = arena_total + out_total;
    for (auto& a : allocs) {
        RT r;
        r.dtype = im.vals[a.vid].dtype;
        r.shape = cshape[a.vid];
        r.strides = rowmajor(r.shape);
        r.p = (a.out ? im.out_arena.data() + off_o[a.vid] : im.arena.data() + off_w[a.vid]);
        res[a.vid] = r;
        done[a.vid] = 1;
    }
    // Stateful KV presents: views into the cache buffers at the
    // new length (cshape already resolved it from the past view).
    if (im.kv && !im.kv_pairs.empty()) {
        for (size_t pi = 0; pi < im.kv_pairs.size(); pi++) {
            int ov = im.kv_pairs[pi].out_vid;
            KvCache::Slot& slot = im.kv->slots[pi];
            RT r;
            r.dtype = 1;
            r.p = (uint8_t*)slot.buf.get();
            r.shape = cshape[ov];
            r.strides = rowmajor(r.shape);
            res[ov] = r;
            done[ov] = 1;
        }
    }
    // Degenerate unproduced values (LN side outputs): zero-size views.
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind == Value::RT && v.def_step == -2 && !done[i]) {
            RT r;
            r.dtype = v.dtype;
            r.p = im.arena.data();
            r.shape = cshape[i];
            r.strides = rowmajor(r.shape);
            res[i] = r;
            done[i] = 1;
        }
    }
    // Views, in creation order (bases always precede their views).
    // Factored into a lambda: the loop runner re-resolves body
    // views per iteration through the same code.
    im.view_blocks.clear();
    auto resolve_view = [&](int i) {
        Value& v = im.vals[i];
        const RT& B = res[v.base];
        RT r;
        r.dtype = v.dtype;
        if (v.view_kind == 1) {  // transpose
            r.p = B.p;
            std::vector<int64_t> perm = v.vperm;
            if (perm.empty()) {
                perm.resize(B.shape.size());
                for (size_t i = 0; i < perm.size(); i++) perm[i] = (int64_t)perm.size() - 1 - (int64_t)i;
            }
            for (auto p : perm) {
                int64_t pa = norm_axis(p, B.shape.size());
                r.shape.push_back(B.shape[pa]);
                r.strides.push_back(B.strides[pa]);
            }
        } else if (v.view_kind == 2) {  // reshape
            RT src = B;
            if (!src.contig()) {
                size_t bytes = (size_t)src.numel() * onnxm::elem_size(src.dtype);
                im.view_blocks.emplace_back(bytes);
                src = materialize(B, im.view_blocks.back().data());
            }
            r.p = src.p;
            if (v.dyn && v.meta0 >= 0) {
                // Dyn reshape (loop body): the compile-time shape
                // exprs contain placeholders; recompute from the
                // target tensor's runtime values and the base's
                // true numel.
                std::vector<int64_t> tgt = read_meta_ints(res[v.meta0]);
                std::vector<int64_t> os;
                int64_t known = 1, infer = -1;
                for (size_t d = 0; d < tgt.size(); d++) {
                    if (tgt[d] == -1) { infer = (int64_t)d; os.push_back(-1); }
                    else if (tgt[d] == 0) { os.push_back(B.shape[d]); known *= B.shape[d]; }
                    else { os.push_back(tgt[d]); known *= tgt[d]; }
                }
                if (infer >= 0) os[(size_t)infer] = known > 0 ? B.numel() / known : 0;
                r.shape = os;
            } else {
                r.shape = cshape[i];
            }
            r.strides = rowmajor(r.shape);
        } else if (v.view_kind == 3) {  // unsqueeze
            r.p = B.p;
            std::vector<char> at(B.shape.size() + v.vaxes.size(), 0);
            for (auto a : v.vaxes) at[(size_t)norm_axis(a, at.size())] = 1;
            size_t src = 0;
            for (size_t d = 0; d < at.size(); d++) {
                if (at[d]) {
                    r.shape.push_back(1);
                    r.strides.push_back(d + 1 < at.size() ? 1 : 1);
                } else {
                    r.shape.push_back(B.shape[src]);
                    r.strides.push_back(B.strides[src]);
                    src++;
                }
            }
            // Stride of a size-1 dim is irrelevant; keep it simple.
        } else if (v.view_kind == 4) {  // squeeze
            r.p = B.p;
            std::vector<char> drop(B.shape.size(), 0);
            bool drop_all = v.vaxes.size() == 1 && v.vaxes[0] == -1;
            if (!drop_all)
                for (auto a : v.vaxes) drop[(size_t)norm_axis(a, B.shape.size())] = 1;
            for (size_t d = 0; d < B.shape.size(); d++) {
                bool dropped = drop_all ? B.shape[d] == 1 : drop[d] != 0;
                if (!dropped) {
                    r.shape.push_back(B.shape[d]);
                    r.strides.push_back(B.strides[d]);
                }
            }
        } else if (v.view_kind == 5) {  // slice
            std::vector<int64_t> starts = read_meta_ints(res[v.meta0]);
            std::vector<int64_t> ends = read_meta_ints(res[v.meta1]);
            std::vector<int64_t> axes;
            if (v.meta2 >= 0) axes = read_meta_ints(res[v.meta2]);
            if (axes.empty()) {
                axes.resize(B.shape.size());
                for (size_t a = 0; a < axes.size(); a++) axes[a] = (int64_t)a;
            }
            std::vector<int64_t> steps(axes.size(), 1);
            if (v.meta3 >= 0) steps = read_meta_ints(res[v.meta3]);
            r = B;
            int64_t off = 0;
            for (size_t ai = 0; ai < axes.size(); ai++) {
                int64_t ax = norm_axis(axes[ai], B.shape.size());
                int64_t first;
                r.shape[ax] = slice_count(B.shape[ax], starts[ai], ends[ai], steps[ai], first);
                off += first * B.strides[ax];
                r.strides[ax] = B.strides[ax] * steps[ai];
            }
            r.p = B.p + off * (int64_t)onnxm::elem_size(B.dtype);
        } else if (v.view_kind == 7) {  // split: contiguous-axis window of the base
            int64_t ax = norm_axis(v.meta0, B.shape.size());
            r = B;
            if (v.meta3 == 2) {
                // Equal split of a dyn base: the window length
                // is the true axis length over the output count.
                int64_t len = B.shape[ax] / v.meta2;
                r.shape[ax] = len;
                r.p = B.p + (int64_t)v.meta1 * len * B.strides[ax] * (int64_t)onnxm::elem_size(B.dtype);
            } else if (v.meta3 == 1) {
                // Const sizes over a dyn base.
                r.shape[ax] = v.meta2;
                r.p = B.p + (int64_t)v.meta1 * B.strides[ax] * (int64_t)onnxm::elem_size(B.dtype);
            } else {
                r.shape[ax] = cshape[i][ax];
                r.p = B.p + (int64_t)v.meta1 * B.strides[ax] * (int64_t)onnxm::elem_size(B.dtype);
            }
        }
        // Validate compile-time shape exprs against the resolved view.
        if (!v.dyn && r.shape != cshape[i]) fail("view shape mismatch at bind (slice/reshape exprs)");
        res[i] = r;
        done[i] = 1;
    };
    // A view is bind-resolvable only when everything it reads
    // (its base chain, and the meta tensors for reshape/slice
    // views) already has a real descriptor: consts, symbolic
    // values, inputs and non-dyn step outputs (arena) qualify;
    // a dyn step output does not — its descriptor only exists
    // after its step runs. Deferred views start stale and the
    // runner resolves them on first use.
    std::function<bool(int)> bind_ready = [&](int vid) -> bool {
        const Value& v = im.vals[vid];
        if (v.kind != Value::RT) return true;
        if (v.view_kind != 0) {
            if (v.base >= 0 && !bind_ready(v.base)) return false;
            if (v.view_kind == 2 && v.meta0 >= 0 && !bind_ready(v.meta0)) return false;
            if (v.view_kind == 5)
                for (int m : {v.meta0, v.meta1, v.meta2, v.meta3})
                    if (m >= 0 && !bind_ready(m)) return false;
            return true;
        }
        return !(v.dyn && v.def_step >= 0);
    };
    std::vector<int> deferred_views;
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind != Value::RT || v.view_kind == 0 || done[i]) continue;
        if (in_body[i]) continue;  // resolved per iteration by the runner
        if (!bind_ready(i)) {
            deferred_views.push_back(i);
            continue;
        }
        resolve_view(i);
    }
    // Execute steps. FE_PROFILE=1 accumulates wall time per step kind
    // across all plans in the process and prints a table at exit.
    static const bool profile = prof_run;
    if (profile) {
        g_prof_bind[im.label] +=
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - run_t0).count();
        g_prof_runs[im.label]++;
    }
    auto exec_t0 = std::chrono::steady_clock::now();
    Scratch& sc = im.bsc;
    std::vector<RT>& sin = im.bsin;
    std::vector<RT>& sout = im.bsout;
    // View staleness for loop bodies: bind resolved every view
    // outside a body; body views start stale and are re-resolved
    // on demand whenever their base is (re)written.
    std::vector<std::vector<int>> view_kids(nv);
    for (int i = 0; i < nv; i++)
        if (im.vals[i].kind == Value::RT && im.vals[i].view_kind != 0 && im.vals[i].base >= 0)
            view_kids[im.vals[i].base].push_back(i);
    std::vector<char> view_stale(nv, 0);
    for (int i = 0; i < nv; i++)
        if (in_body[i] && im.vals[i].view_kind != 0) view_stale[i] = 1;
    for (int i : deferred_views) view_stale[i] = 1;
    std::function<void(int)> resolve_stale = [&](int i) {
        if (!view_stale[i]) return;
        int b = im.vals[i].base;
        if (b >= 0 && im.vals[b].view_kind != 0) resolve_stale(b);
        resolve_view(i);
        view_stale[i] = 0;
    };
    auto mark_kids_stale = [&](int v) {
        for (int k : view_kids[v]) view_stale[k] = 1;
    };
    // Step ownership: a step inside a Loop body runs only when
    // that Loop dispatches its body range, never when an outer
    // range walks past it. Owner = array index of the innermost
    // LOOP step whose body range contains the step; -1 = top
    // level. (Body steps sit inline in the steps array, between
    // the steps of the nodes lowered before and after the Loop.)
    int nsteps = (int)im.steps.size();
    std::vector<int> step_owner(nsteps, -1), owner_span(nsteps, 0x7fffffff);
    auto claim_range = [&](int li, int b0, int b1) {
        if (b0 < 0 || b1 <= b0) return;
        for (int i = b0; i < b1 && i < nsteps; i++)
            if (b1 - b0 < owner_span[i]) { owner_span[i] = b1 - b0; step_owner[i] = li; }
    };
    for (int li = 0; li < nsteps; li++) {
        const Step& ls = im.steps[li];
        if (ls.k == SK::LOOP) claim_range(li, ls.lp_b0, ls.lp_b1);
        if (ls.k == SK::IF) {
            claim_range(li, ls.if_b0, ls.if_b1);
            claim_range(li, ls.if_eb0, ls.if_eb1);
        }
    }
    std::function<void(int, int, int)> run_steps = [&](int b0, int b1, int owner) {
        for (int si = b0; si < b1; si++) {
            if (step_owner[si] != owner) continue;
            const Step& st = im.steps[si];
            if (st.k == SK::IF) {
                auto if0 = std::chrono::steady_clock::now();
                // ---- If execution: run the taken branch, then
                // copy its outputs onto the If's out values ----
                bool cond = read_d(res[st.in[0]], 0) != 0;
                if (cond) run_steps(st.if_b0, st.if_b1, si);
                else run_steps(st.if_eb0, st.if_eb1, si);
                const std::vector<int>& bouts = cond ? st.if_t_outs : st.if_e_outs;
                for (size_t j = 0; j < st.out.size(); j++) {
                    int bo = bouts[j], ov = st.out[j];
                    // A branch output may be a deferred view
                    // (e.g. Squeeze of a dyn tensor) that no
                    // regular step ever consumed: resolve it
                    // before copying its descriptor, like the
                    // step-input path below does.
                    if (im.vals[bo].view_kind != 0 && view_stale[bo]) resolve_stale(bo);
                    if (im.vals[ov].is_seq) im.bseqs[ov] = im.bseqs[bo];
                    else if (im.vals[bo].kind == Value::CONSTV) res[ov] = tensor_of_const(im.vals[bo].ct);
                    else res[ov] = res[bo];
                    mark_kids_stale(ov);
                }
                if (profile)
                    prof_add(im.label, (int)SK::IF,
                             std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - if0).count());
                continue;
            }
            if (st.k == SK::LOOP) {
                auto lp0 = std::chrono::steady_clock::now();
                // ---- Loop execution ----
                int ncarried = (int)st.out.size();
                int64_t max_iter = INT64_MAX;
                if (st.in[0] >= 0) max_iter = (int64_t)read_d(res[st.in[0]], 0);
                bool cond = true;
                if (st.in[1] >= 0) cond = read_d(res[st.in[1]], 0) != 0;
                int iter_slot = st.lp_slots[0], cond_slot = st.lp_slots[1];
                for (int j = 0; j < ncarried; j++) {
                    int slot = st.lp_slots[2 + j], src = st.in[2 + j];
                    if (im.vals[slot].is_seq) im.bseqs[slot] = im.bseqs[src];
                    else res[slot] = res[src];
                }
                size_t vb_mark = im.view_blocks.size();
                for (int i = st.lp_v0; i < st.lp_v1; i++)
                    if (im.vals[i].view_kind != 0) view_stale[i] = 1;
                int64_t iter = 0;
                while (iter < max_iter && cond) {
                    if (iter >= 10000000) fail("Loop: iteration cap");
                    int64_t iv = iter;
                    memcpy(res[iter_slot].p, &iv, 8);
                    uint8_t cv = cond ? 1 : 0;
                    memcpy(res[cond_slot].p, &cv, 1);
                    run_steps(st.lp_b0, st.lp_b1, si);
                    cond = read_d(res[st.lp_body_outs[0]], 0) != 0;
                    for (int j = 0; j < ncarried; j++) {
                        int slot = st.lp_slots[2 + j], bo = st.lp_body_outs[1 + j];
                        if (im.vals[slot].is_seq) im.bseqs[slot] = im.bseqs[bo];
                        else res[slot] = res[bo];
                    }
                    for (int i = st.lp_v0; i < st.lp_v1; i++)
                        if (im.vals[i].view_kind != 0) view_stale[i] = 1;
                    if (im.view_blocks.size() > vb_mark) im.view_blocks.resize(vb_mark);
                    iter++;
                }
                for (int j = 0; j < ncarried; j++) {
                    int bo = st.lp_body_outs[1 + j], ov = st.out[j];
                    if (im.vals[ov].is_seq) im.bseqs[ov] = im.bseqs[bo];
                    else res[ov] = res[bo];
                    mark_kids_stale(ov);
                }
                if (profile)
                    prof_add(im.label, (int)SK::LOOP,
                             std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lp0).count());
                continue;
            }
            for (int v : st.in)
                if (v >= 0 && im.vals[v].view_kind != 0 && view_stale[v]) resolve_stale(v);
            sin.resize(st.in.size());
            sout.resize(st.out.size());
            for (size_t j = 0; j < st.in.size(); j++) sin[j] = st.in[j] >= 0 ? res[st.in[j]] : RT{};
            if (!step_self_alloc(st.k))
                for (size_t j = 0; j < st.out.size(); j++) {
                    int ov = st.out[j];
                    if (ov >= 0 && im.vals[ov].dyn && !im.vals[ov].is_seq && im.vals[ov].view_kind == 0)
                        res[ov] = dyn_shape_for(im, st, sin, ov);
                }
            for (size_t j = 0; j < st.out.size(); j++) sout[j] = res[st.out[j]];
            Scratch::Mark mk = sc.mark();
            if (profile) {
                auto p0 = std::chrono::steady_clock::now();
                exec_step(im, st, sin, sout, sc);
                double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - p0).count();
                prof_add(im.label, (int)st.k, us);
            } else {
                exec_step(im, st, sin, sout, sc);
            }
            sc.reset(mk);
            for (size_t j = 0; j < st.out.size(); j++) {
                res[st.out[j]] = sout[j];
                mark_kids_stale(st.out[j]);
            }
        }
    };
    run_steps(0, (int)im.steps.size(), -1);
    if (profile)
        g_prof_exec[im.label] +=
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - exec_t0).count();
    // An output that is itself a deferred view may never have
    // been consumed by a step; resolve it now that its base is
    // final.
    for (int v : im.out_vals)
        if (im.vals[v].view_kind != 0 && view_stale[v]) resolve_stale(v);
    // Outputs, in out_names order.
    std::vector<Tensor> out(im.out_vals.size());
    for (size_t i = 0; i < im.out_vals.size(); i++) {
        const RT& r = res[im.out_vals[i]];
        Tensor t;
        t.dtype = r.dtype;
        t.data = r.p;
        t.shape = r.shape;
        if (!r.contig()) t.strides = r.strides;
        out[i] = t;
    }
    return out;
}

}  // namespace freeeng
