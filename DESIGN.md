# Design

## Goal

A native C++ engine that runs GPT-SoVITS v2ProPlus inference on x86-64
CPUs with AVX/AVX2, fast enough to beat the upstream PyTorch CPU path on
real-time factor while using a fraction of its memory. No GPU, no Python
in the hot path, standard weights.

## Compatibility contract

- Input weights: an upstream v2ProPlus pair, `*.ckpt` (GPT / T2S) and
  `*.pth` (SoVITS / VITS), as produced by upstream fine-tuning.
- Conversion scripts in this repo turn that pair into the engine's
  runtime layout. The conversion is a local, offline step; converted
  weights are never committed here.
- Synthesis semantics follow upstream: text plus a reference voice
  produces speech with the same sampling controls (temperature, top_k,
  top_p, seed, repetition penalty).

## Pipeline

Upstream v2ProPlus inference has five stages:

1. **Text frontend.** Language-aware G2P to phones and tones. Chinese
   additionally consumes BERT features upstream; English does not.
2. **Reference encoding.** The reference clip is turned into semantic
   tokens (HuBERT features quantized by the SoVITS codebook path) and a
   prompt latent (the VITS prompt_encoder).
3. **GPT decode.** A transformer decoder generates semantic tokens
   autoregressively, one step per token, with sampling.
4. **VITS decode.** One forward pass: semantic tokens plus prompt latent
   through the flow and the upsampling decoder to a waveform.
5. **Output chain.** Trim, gain and soft clipping on the way out.

## Decision: stage 2 runs both ways, the user chooses

Settled 2026-11-25: the engine supports both reference modes, and the
person running it picks per voice.

- **Runtime reference.** A wav plus its transcript is given at run time;
  the engine runs HuBERT and the prompt_encoder on the spot. Drop-in
  compatible with upstream usage, any clip works immediately.
- **Precomputed pack.** Reference features computed earlier (by this
  engine's own runtime path, saved out, or by the offline tool) are
  loaded from a pack file. Startup is milliseconds and the HuBERT side
  never loads.

There is exactly one canonical reference-feature format: the pack.
Runtime mode computes the features and may write them back as a pack,
so a voice extracted once can be reused in pack mode forever. Downstream
of the features there is one code path; the two modes differ only in
where the extraction bill is paid.

Sequencing: the synthesis core lands first against packs (the shape
already proven in the PotatoAmika ARM engine), then runtime extraction
is added on the same feature format.

One related point is independent of this choice: BERT features for
Chinese *target* text belong to the frontend (stage 1). If Chinese is
in scope, both modes carry that cost; it has nothing to do with how
the reference is supplied.

## Acceleration plan

Measurements from the ARM port carry over, because they are properties
of the workload, not of the instruction set:

- **GPT decode is latency-bound.** One token per step, each step reads
  the full weight set and does little math. What pays: wide-vector fp32
  GEMV, KV cache, and thread discipline. On the ARM device, more than
  two threads made decode *slower*; expect a small fixed thread count
  with pinning here too, tuned by measurement, not by core count.
- **Weight traffic dominates decode.** Storing weights as fp16 and
  converting on the fly (F16C) halves bytes read per step at fp32
  compute. AVX2 has no VNNI, so int8 decode gains are limited; int8 is
  an experiment behind a gate, not the baseline plan.
- **Quantization is the second acceleration tier.** The weight-only
  W4 packs from the ARM line carry over unchanged (same format, same
  conversion scripts, same gates: GPT agreement 299/361, VITS
  correlation 0.9599). What does not carry over is the kernel tier:
  the packed GEMV kernels are NEON/SDOT on ARM and fall back to a
  scalar floor on x86 today. AVX2 packed kernels are part of M2, so
  the quantized path is a real speed tier here, not a compatibility
  leftover. The user of the engine picks the tier per deployment:
  standard precision for exactness, W4 for speed and memory.
- **VITS is many small ops.** The graph has hundreds of operators with
  small tensors; fixed per-op overhead is a large share of the total.
  What pays: op fusion, preallocated arenas, no per-op allocation, and
  AVX2 kernels for the conv and attention blocks that do carry math.
- **Runtime dispatch.** Baseline target is AVX + FMA (Haswell class and
  later). Wider paths (AVX-512, VNNI) may be added behind cpuid dispatch
  but the AVX2 path must stand on its own.

## The bar

Upstream GPT-SoVITS on PyTorch CPU is not slow: it rides oneDNN. In a
2026-11-09 x86 comparison against the quantized ARM-lineage engine,
stock was about 2x faster in RTF and used about 11x the memory
(5,120.9 MB peak vs 457.8 MB). This project exists to close the speed
gap while keeping the memory profile of a native engine. Claims get
measured on the same sentences, same seed, same machine, or they do
not ship.

## Correctness gates

Same discipline as the ARM engine:

- GPT stage: teacher-forced token agreement against the PyTorch
  reference on a fixed probe set.
- VITS stage: full-graph waveform correlation against the reference.
- End to end: fixed sentences, fixed seed, listened to by a human.
  Numbers qualify a build; ears ship it.

## Milestones

1. **M0.** This scaffold: scope, design, red lines in .gitignore.
2. **M1. Done 2026-11-25.** The existing free engine carried over and
   re-gated at standard precision: the loader learned external-data
   graphs, the engine learned the ops the raw fp32 exports use, both
   stages pass the correctness gates below, and the host gained a
   standard-precision load path (`--engine free --fp32-graphs`,
   graphs under `<root>/graphs-fp32/`, stage decoder file holding the
   logits variant). End-to-end smoke on the dev VM: boot 14 s, one
   3.13 s sentence synthesized through the full output chain, peak
   RSS about 2.5 GB, RTF about 10 at the scalar kernel floor on a
   1.5-CPU VM. That RTF is a pre-acceleration baseline, not a result.
3. **M2 (done).** AVX2 kernels for decode GEMV and the heavy VITS blocks,
   AVX2 packed kernels for the W4 tier, fp16 weight storage, thread
   pinning. Profiling the standard-precision host (FE_PROFILE, dev
   VM) put the work in order: VITS is 89% Conv (20.8 s of 23.3 s),
   GPT is about 75% MatMul/GEMM. First kernel landed: the conv1d
   SAXPY inner loop as an AVX2/FMA kernel in its own translation
   unit (only that file is compiled -mavx2; callers dispatch on
   runtime CPU detection, scalar loops unchanged elsewhere). Dev-VM
   result: Conv 20.8 s to 15.4 s, host total 32.5 s to 26.5 s for
   the gate sentence, all four correctness gates unchanged
   (361/361, 1.000000, 299/361, 0.959880). Cross-verified on a
   second machine (InstaBox, AMD EPYC 4 vCPU, AVX2+FMA+AVX-512),
   cloned from this repo and built clean: all four gates reproduce
   exactly there (the packed correlation is bit-stable at 0.959880,
   relrms identical), and the same sentence measures, standard
   precision vs the W4 tier: RTF 3.15 vs 2.02 at 2 threads, RTF
   2.10 vs 1.25 at 4 threads; peak RSS 2,586 MB vs about 255 MB.
   The W4 kernels on x86 are still the scalar floor at this point,
   so the W4 lead is weight traffic alone; the AVX2 packed tier is
   still to come.
   Second kernel: the same SAXPY form vectorized inside k_matmul
   (tiled, GEMV and small-GEMM paths), which also covers the GEMM
   step kind. Gates unchanged (the packed vits correlation prints
   0.959881 instead of 0.959880 because the fp32 reference side now
   runs FMA matmuls; the packed path itself is untouched and its
   internal packed-vs-baked check stays 1.000000). EPYC result:
   GPT decode time down 26 to 28 percent at both thread counts;
   the standard-precision sentence goes to RTF 2.83 at 2 threads
   and 1.84 at 4 threads. The W4 sentence is unchanged within the
   machine's run-to-run noise (about plus or minus 10 percent;
   four samples span 4.00 to 4.47 s at 4 threads), as expected:
   W4 decode runs the packed kernels, not k_matmul.
   Third kernel set: AVX2 tiers for the W4 packed GEMVs (mode A
   group SAXPY, mode B group dot, both over a shared nibble unpack;
   the block GEMM tile apply also takes the SAXPY form). The
   packed teacher-forced gate holds at exactly 299/361. EPYC:
   W4 GPT decode down 13 to 23 percent, the W4 sentence at RTF
   1.88 (2 threads) and 1.21 (4 threads). The packed VITS conv
   time did not move; it lives in the packed conv kernels, which
   are still scalar on x86, and is the next target.
   Fourth change: the plain fp32 Conv takes the same
   register-blocked AVX2 tile for stride-1 shapes (the packed and
   fp32 weight layouts are identical), under dynamic task fetch --
   a static partition of the tile work cost +83 percent on the
   shared-vCPU box, the same effect conv1d_fast's fetch loop
   exists for. EPYC, 4 threads: fp32 CONV 2,445 -> 1,616..1,891
   ms, the fp32 sentence at 4.72..5.11 s (RTF about 1.5), the
   best standard-precision result so far. The vits fp32 gate
   statistics are bit-identical to the SAXPY form's.
   Fifth change: fp16 weight storage, opt-in per plan
   (--fp16-weights on the host's standard-precision path, GPT
   plans only). Initializer operands of MatMul/Gemm are converted
   once at compile time and converted back on the fly with F16C
   inside the matmul kernels; both weight layouts are covered
   (row-contiguous SAXPY form and transposed dot form). The
   teacher-forced gate holds at exactly 361/361. EPYC, 4
   threads: GPT decode 2,011..2,131 -> 1,871 ms (about -10
   percent); the sentence total is inside run-to-run noise. The
   fp32 originals are retained alongside the fp16 copies for
   now, so RSS grows by about 440 MB; dropping single-use
   originals after conversion is a recorded follow-up, not
   done. Thread pinning was not pursued: the verification box
   is shared vCPUs, and the device precedent favoured unpinned.
   This closes M2.
4. **M3.** Hardening: streaming output, packaging, docs.
5. **Final gate.** A published speed comparison, run once everything
   above is done: stock GPT-SoVITS (PyTorch CPU) vs this engine at
   standard precision vs this engine on the W4 tier, same machine,
   same sentences, same seed, RTF and peak memory side by side. The
   project is not done until that table exists.

## M1 gate results (2026-11-25, x86-64, this engine, standard precision)

- GPT teacher-forced agreement against the fp32 reference trajectory:
  **361/361 (1.0)**. The same build running the W4 baked graphs
  reproduces the quantized line's recorded 299/361 exactly, so the
  move changed nothing for the old path.
- VITS full graph against ONNX Runtime on the identical graph and
  feed, sampling noise pinned to zero in both: **correlation
  1.000000, max error 4.0e-05** over 96,000 samples. The packed
  hybrid path still reproduces its recorded 0.959880 against the
  stage-1 reference waveform.
- Two facts worth knowing about the inherited assets: the fp32 graph
  copies under the old tree's `graphs/fp32` are dangling references
  (their external `.bin` files were never kept; only fp16 bins
  remain), and the complete self-contained fp32 set lives in
  `work/fp32_full`. Its stage decoder must be the `*_logits` variant:
  the plain stage graph has no logits output; the repetition-penalty
  head exists only in the logits variant.

## Final gate, partial results (2026-11-25, EPYC verification box)

The user added Genie-TTS as a fourth column. Same machine (4 vCPU
EPYC, shared), same sentence ("Hello, this is a test of the standard
precision path."), same reference (neutral1, Chinese transcript),
4 threads (Genie leaves ONNX Runtime at its default, which is all
4 cores). Genie has no seed control, so its output length varies
per run; RTF is the normalized figure.

| engine | precision | RTF | peak RSS |
|---|---|---|---|
| Genie-TTS 2.0.2 (ORT, Python) | fp16 | **0.55** (1.72 s / 3.08 s audio; 2.27 s / 4.16 s) | 5,485 MB (whole Python process) |
| this engine, W4 tier | W4 | 1.26–1.38 today (4.03–4.43 s / 3.20 s audio; best recorded 1.13) | 256 MB |
| this engine, standard | fp32 | ~1.5 | 2,586 MB |

One-time costs: Genie load-character + set-reference 10.6 s; this
engine's W4 boot 0.4–0.6 s. Genie's first synthesis after load ran
at RTF 1.24 (session warmup); the 0.55 figure is steady state.
Output sanity-checked numerically (32 kHz, RMS ~0.10).

Reading: on this box Genie's ORT fp16 stack is about 2.3× faster
than our W4 tier in RTF, and our process is about 21× lighter.
The remaining leg of the final gate is stock GPT-SoVITS (PyTorch
CPU) on the same machine; the table is final only when it lands.

## Post-gate optimization round (2026-11-26)

The AVX2 conv tile was reshaped from 4 channels x 32 positions to
4 x 16 with direct interior loads. The old shape held 16
accumulators plus 4 position vectors in a 16-register file and
spilled; the new one holds 11 live ymm. Micro-bench on the EPYC
box: 29-35% ahead on every production shape (42.5 vs 32.4 GMAC/s
at 192ch K=11), bit-identical output, all four gates unchanged.
Same-session E2E (W4, 4 threads): vits 2,175-2,278 -> 1,512-1,592
ms; sentence total 3,244-3,374 ms, **RTF 1.01-1.05** (was
1.26-1.38 in the Genie session, best-ever 1.13 before the change).
fp32 total improved to RTF 1.38 (4,336 ms). The W4 tier now sits
on the RTF 1.0 line on this shared box.

Next lever, profiled but not taken: GPT decode is 1,245 ms for 81
steps (15.4 ms/step), of which the stage plans' kernel kinds sum
to only ~600 ms (PACKED ~450, MATMUL ~85, SOFTMAX ~30). The rest
is executor and pool-sync overhead across 668-step plans invoked
per token. Closing that gap is an executor change, not a kernel
change.

## Decode pool-gate round (2026-11-26, later)

Taken. A sampling profile first cleared the executor itself
(Plan::run self-time 0.32%); the cost was the pool: every decode
GEMV, however small, paid a full condvar barrier (~8,000 per
request across the stage plans). All GEMV/GEMM and packed-conv
pool dispatches now go through a work-size gate
(POTATO_POOL_MIN_MAC, MACs): below it the body runs serially on
the calling thread, bit-identical by chunk construction (gates
unchanged: 361/361, 299/361, 1.000000, 0.959881). Interleaved
A/B on the EPYC box picked 2^19 as the default: decode medians
1,400 ms at 2^19 vs 1,530 at 2^18 and 1,841 at 2^20. Final
shipped-default samples (W4, 4 threads): 3,192-3,694 ms per
sentence, **RTF 1.00-1.15**, decode 1,189-1,486 ms, RSS 256 MB.
Across both optimization rounds the sentence went from
4,026-4,426 ms (RTF 1.26-1.38, Genie session) to 3,192-3,694 ms.

## Elementwise round (2026-11-26, later still)

AVX2 elementwise kernels (exactly-IEEE ops only: ADD/SUB/MUL/DIV,
comparisons, NEG/SQRT/RELU/ABS/LEAKY) wired into the binop/unop
fast paths: vits UNOP 97 -> 41 ms, BINOP 55 -> 39 ms per request.
Size-1 dim compaction before path selection returns [C, L, 1]
layouts to the rows path. Two measured dead ends, kept out: a
generalized strided axis path for the attention score scale
(227M elements) lost to the plain linear odometer walk in an
interleaved A/B (362-378 ms vs 291-300 ms; the walk's Y-linear
order reuses the operand cache window, scattered access in
either direction loses), and software prefetch on the strided
form was ~3% worse than none. The axis path ships restricted to
the vectorizable middle-axis case. FE_PROFILE also gained
per-plan bind vs exec lines (bind is ~1.2 ms per stage-plan
invocation, not the decode gap). Final shipped samples (W4,
4 threads): 2,871-3,400 ms per sentence, **RTF 0.90-1.06**,
decode 1,053-1,298 ms, stage PACKED 212 + 214 ms, PCONV
1,400 ms, RSS 256 MB.

## Genie study round (2026-11-26, evening)

Took Genie-TTS apart on the same box to find where its RTF 0.55
comes from. Three findings:

1. **AVX-512 is real but small.** perf sampling of steady-state
   Genie synthesis puts ~13% of CPU in MLAS's AVX512F SGEMM kernel
   (zmm vbroadcastss/vfmadd231ps, verified by disassembly). ORT
   1.22 has no switch to disable it (raw CPUID dispatch; glibc
   hwcaps masking verified ineffective — the kernel stays hot).
   On Zen 4 the 512-bit datapath is double-pumped, so the kernel's
   AVX2 form would cost only ~0-20% more: AVX-512 explains a
   single-digit percent of wall time, not the ~2x RTF gap.
2. **The graphs are the same graphs.** Genie's converted set is
   the same export family as ours (vits: 218 Conv, ~8k nodes with
   ~3k Constants the ORT optimizer folds at load). ORT's edge is
   kernels + graph fusions (FusedConv, FusedMatMul appear in its
   profile) + its thread pool, not a different model.
3. **The single biggest hot spot in Genie's profile (~25% of
   samples) is its thread-pool dispatch/spin code** — ORT workers
   spin before parking. Transferred that: `packed::Pool` is now
   spin-then-block on both sides (workers spin on the generation
   counter, the caller spins on the completion counter;
   POTATO_POOL_SPIN, default 5000, 0 = old condvar pool).
   Interleaved same-session A/B on the EPYC box (W4, 4 threads,
   medians of 3): total 3,318 -> 2,961 ms (**RTF ~0.92**, range
   0.87-0.99), decode 1,238 -> 1,077 ms, vits 1,557 -> 1,398 ms.
   Spin won every round on every component.

One bug worth remembering: the first spin version incremented the
generation counter outside the pool mutex and wedged a worker at
generation ~17,662 of the fp32 TF gate (lost wakeup against the
wait predicate; a temporary watchdog build caught the exact
state). The increment stays inside the mutex; the spin reads are
lock-free. All four gates reproduce exactly (299/361, 361/361,
0.959881, 1.000000).

## Genie conv dissection (2026-11-26, midday)

Per-node profile of Genie's vits (steady run, 2,007 nodes, 573 ms
total): Conv 234 + FusedConv 218 + ConvTranspose 78 = 530 ms of
conv; 94% of it is the decoder. Same 223 conv calls as our packed
path; our PCONV was 1,332 ms. The difference is per-call rate, and
the rate difference is instruction width: this EPYC's AVX-512
datapath is full-width (single-thread FMA peak measured 176.2
GMAC/s AVX-512 vs 88.0 AVX2), and ORT's MLAS conv hits 113.5
GMAC/s per thread on the 192x192 K11 leader shape -- above the
AVX2 ceiling. (This corrects the morning's read, which generalized
the GPT stage profile, where the AVX-512 kernel is ~13% of CPU, to
the whole engine; for vits conv it is the main term.)

Transferred as a runtime-dispatched AVX-512F conv tile tier
(AVX2 remains the baseline; POTATO_NO_AVX512=1 forces it off).
First port at 16 positions wide measured 41.9 GMAC/s per thread,
no better than the AVX2 tile's 40.7 -- four accumulator chains
cannot hide FMA latency. Widened to 32 positions (eight chains,
one weight broadcast per two FMAs): 101.4 GMAC/s per thread,
bit-identical (maxdiff 0). End to end on the EPYC box, interleaved
A/B medians: total 3,173 -> 2,432 ms (**RTF ~0.76**, range
0.75-0.84), vits 1,535 -> 938 ms, PCONV 1,332 -> 857 ms. All four
gates reproduce exactly. Remaining vits gap to Genie (~380 ms of
node time): production-vs-isolation rate loss, the transposed
convs, and ~45 ms of elementwise; the kernel-rate remainder is
MLAS's GEMM panel packing.

## Compiler check (2026-11-26, afternoon)

Builds use GCC (13.3 dev VM, 14.2 EPYC box), Release (-O3), no
global -march; ISA tiers are per-function target attributes.
Measured whether the compiler matters, same tile bench, one
thread, 192x192 K11 L=12000 (GMAC/s):

| tile | g++ 14.2 | clang 19.1 |
|---|---|---|
| AVX2 4x16 | 41.2 | 16.7 |
| AVX-512 4x32 (shipping) | 100.0 | 106.0 |
| AVX-512 8x32 | 21.3 (spills) | 121.9 |

The compiler swings individual kernels by up to 2.5x in either
direction: clang's allocator handles the 16-accumulator zmm tile
that makes GCC spill, and beats even ORT's 113.5 on that shape;
GCC is far better on the AVX2 baseline tile. ORT sidesteps the
whole question -- MLAS kernels are hand-written assembly. Verdict:
the unified build stays on GCC (the AVX2 tier is the product
baseline and must not regress); an optional per-TU clang build of
just the AVX-512 tile is recorded as a possible ~6% end-to-end
gain on AVX-512 machines, not built.

## AVX-512 GEMV kernels (2026-11-26, afternoon)

The W4 group kernels got zmm forms (w4a_saxpy64_512,
w4b_dot64_512 in avx2_kernels.cpp), dispatched inside the AVX2
GEMV tiers when cpu::features().avx512 and POTATO_NO_AVX512 is
unset. The SAXPY is elementwise and bit-identical; the dot
reduces over 16-lane partials, changing last-ulp rounding.
Micro-bench on the EPYC box (gemv_bench, 4 threads): mode B at
the decode shape 512x512 +30% (14.3 vs 11.0 GMAC/s), mode A
shapes +10 to +23%. End-to-end decode moved only ~3%, inside
the box's noise band -- the stage plans' PACKED time is spread
over many small GEMVs and non-GEMV ops. All four gates were run
on the box itself (gpv-root carries every gate asset): tf
packed holds at exactly 299/361 with the new reduction, tf
fp32 361/361, vits packed 0.959881, vits fp32 corr 1.000000
relrms 1.3e-05 (maxerr 4.46e-05 under the box's g++ 14.2 vs
4.34e-05 under the dev VM's 13.3 -- a compiler-version ulp
artifact in auto-vectorized scalar code, corr/relrms exact).

## Scope ruling (2026-11-26)

The product baseline is AVX/AVX2, as chartered. The AVX-512
tier (conv tile, GEMV kernels) stays in the tree as a runtime-
dispatched bonus for CPUs that report it -- it never executes
on AVX2-only machines and POTATO_NO_AVX512 disables it -- but
it is frozen: no further AVX-512 optimization work. Optimization
effort returns to the AVX2 tier, including a newer-clang check
and hand-written assembly where they measurably pay.

## Attention scale fold (2026-11-26, afternoon)

The GPT stage plans scale Q and K by d^(-1/4) in standalone Mul
steps (the scalar computed at runtime as
Sqrt(Cast(Div(1, Sqrt(Shape[..]))))). As steps, those Muls walk
the [1, 16, L, 32] tensors in a gathered layout: ~300 ms of
BINOP time per request. Two compile-time folds in
Plan::compile now move such scales into the consuming matmul's
epilogue (Step::out_scale; Gemm alpha when it has no C input):
a post-matmul fold for Mul(MatMul, const) and a pre-matmul fold
for MatMul(Mul(x, const-expr)) whose scalar resolver evaluates
CONSTV / Cast / Sqrt / Neg / Add / Sub / Mul / Div chains and
1-element SYMVALs (shape-derived dims). Both folds are
bit-identical (the epilogue's double round-trip is exact at
fp32, sqrt/div included) and conservative: the scaled value's
consumer path must be unique. All four gates reproduce exactly,
locally and on the box. FE_PROFILE on the box: stage-plan BINOP
calls drop by 1,968 + 1,920 per request, stage BINOP time
~313 ms -> ~14 ms. A wall-clock A/B the same afternoon was
unreadable -- the shared box's baseline swung 2.4-4.8 s between
runs -- so the profile accounting is the recorded evidence.
Known rejection: a scaled tensor feeding two matmuls (encoder)
stays unfolded; it runs once per request and is not the target.

## Compiler round 2 and the assembly tile (2026-11-26)

clang 21 (apt.llvm.org) was installed on the box and measured
against GCC on the tile bench: the AVX2 tile is still ~2.4x
slower under clang (17.3 vs ~41 GMAC/s per thread), so the
compiler picture is unchanged -- GCC builds the product; clang's
advantage stays confined to the frozen AVX-512 tier (112-117
GMAC/s there). One portability fix landed on the way:
avx2_kernels.cpp now includes immintrin.h on x86_64 generally,
which clang requires to see the AVX-512 intrinsics at all.

The AVX2 conv tile now has a hand-written assembly core
(engine/src/conv16_asm.S) for interior 4-channel chunks with
even Ci. It walks input channels in pairs, doubling the FMA
dependency distance over the intrinsics tile (whose per-tap
accumulator hits sit ~14 instructions apart against a 4-5 cycle
FMA latency -- the cap behind its ~47% of FMA peak). Isolated
bench: 50.4 vs 39.3 GMAC/s (+28%), maxdiff 7.6e-06 (the pair
interleave reorders summation at the 1e-5 level). All four
gates reproduce their canonical stats locally and on the box
(the box packed internal maxerr moved 2.29e-06 -> 2.32e-06,
corr/relrms unchanged). POTATO_NO_ASM=1 restores the intrinsics
tile; edge chunks, tails and odd Ci still use it.

## Mode-B GEMV dotraw (2026-11-26)

The AVX2 mode-B dot folded scale/zero out of the inner loop:
sum((w - ze) * sc * x) == sc * dotraw(w, x) - sc * ze * sum(x),
with the group sums of x computed once per GEMV instead of once
per row. The isolated gemv_bench cannot see it (13.8 GMAC/s
either way; that shape is overhead-bound), but the host profile
can: stage-plan PACKED 250 + 260 ms -> 212 + 208 ms per request
in AVX2-only mode, about -90 ms. Gates exact (299/361 holds;
the restructure flipped no tokens).

## PCONV tile width sweep (2026-11-26)

POTATO_PCONV_LT tunes the packed conv's position-tile width
(default 1024). On the EPYC box, LT=512 measured lowest in both
runs (PCONV ~1,283/1,360 ms vs 1,428/1,446 at 1024; 256, 2048
and 4096 in between) -- an ~8% spread, inside the box's session
noise, so the default stays 1024 and the knob stays for
per-machine tuning.

## Executor and attention round (2026-11-26, evening)

Three hypotheses were measured and rejected; the record is kept
so they are not retried blind.

Executor overhead: FE_PROFILE per-kind sums match plan exec
times to ~0.5 us per step in the stage plans. There is no
hidden dispatch tax to reclaim.

KV concat cache: the stage plans spend ~190 ms per request in
Concat, almost all of it copying the past KV. A persistent
per-step cache buffer was built (sampled-checksum resync,
append-only on a hit) -- and the hit counter showed 0 hits in
17,689 stores. The host ping-pongs between the two stage plans,
so each plan's incoming past is always one slice longer than
its own cache; worse, a row-major buffer cannot grow along a
middle axis in place at all (the layout itself depends on the
axis length), so a hit would have read stale offsets. The
feature was removed in full. A correct version needs a
capacity-padded layout and a strided presentation to every
consumer; that is a redesign, not a patch.

Parallel concat: splitting the ~600 KB KV copies over the pool
measured no gain on the verification box (CONCAT 89-98 ms
serial vs 100-111 ms parallel, same sessions). The copies are
bound by shared-vCPU bandwidth there, not by core count.

Wide attention dot: the stage score MatMuls read K from a
position-major [L, 512] buffer as stride-512 scalar gathers in
the general path, which looked like the obvious waste. An
AVX2 float dot fast path over the contiguous 32-float rows
measured stage MATMUL 174-180 ms before and 181-183 ms after
-- nothing. The gather pattern reuses each cache line across
the head-dim loop and the hardware prefetcher keeps up; the
remaining cost is small-op overhead, not memory order. The
change also moved the vits fp32 gate statistics (maxerr
4.34e-05 -> 6.25e-05) for zero gain, so it was reverted and
the canonical statistics reproduce again.

## Correction: stale-clone incident and the wide dot verdict (2026-11-26, night)

The verification box's clone carried an uncommitted edit to
avx2_kernels.cpp (the clang immintrin fix, applied box-side
during the clang-21 attempt) that made every `git pull --ff-only`
abort silently in scripts that discarded its output. Box builds
therefore lagged HEAD by a varying amount during the late
afternoon. Consequences, re-measured with HEAD-asserted builds:

- The wide attention dot path (8d711f6) is NOT a wash. With
  verified builds, stage-plan MATMUL is 105-109 ms against
  164-171 ms without it -- about -120 ms per request across both
  stage plans. It is re-landed. Its cost stands as recorded:
  the vits fp32 gate statistics move to corr 1.000000, relrms
  1.7e-05, maxerr 6.25253e-05 (from 1.3e-05 / 4.33773e-05), and
  the packed internal maxerr to 1.66893e-06; the token gates
  hold exactly (299/361, 361/361). These are the new baselines.
- Parallel concat stays reverted: with verified builds the
  CONCAT lines are 77-94 ms on both sides. That verdict now has
  a sound basis.
- The dead-step elimination (e86d653) is confirmed on the box:
  stage UNOP calls drop 4,018 -> 1,066 per request.
- The mode-B dotraw box figure (-90 ms of stage PACKED) was
  measured inside the stale window and is unverified; the
  change stays on its algebra and exact local gates.
- The hand-written asm tile's box E2E reading from the same
  window is likewise unverified; its isolated bench compiled
  the sources directly and stands, as do all local gates.

Box discipline from here: every verification script prints the
clone's HEAD and working-tree status before building, and a
failed pull fails the script.

## Brainstorm campaign (2026-11-26, night)

Four parallel research passes (kernels, graph rewrites, runtime
systems, outside techniques) produced a ranked pool; every item
was verified or measured before any code was written.

Verified away (no code): the fp32 vits weight-norm chains are
already constant-folded at compile (BINOP totals prove it); the
ConvTranspose family is 4.6% of PCONV compute, at parity with
ORT's 78 ms -- not a gap.

Shipped, each gated and interleaved-A/B'd on the box:
- Dead-step elimination after the scalar folds: stage UNOP calls
  4,018 -> 1,066 per request (e86d653).
- LeakyReLU epilogue fusion into conv / conv-transpose /
  packed-conv / residual-add steps (46 sites in the vits plan):
  vits UNOP 32-37 -> 13 ms (7c3d633). The other 31 Add cases have
  a genuine second consumer and stay steps.
- Bind workspace reuse (persistent res/cshape/planner/Scratch):
  stage bind 45 -> 35 ms per plan, about -22 ms per request,
  planner arithmetic unchanged (49d61df).
- gemm_b_w4 prefill block path (vector dequant + transpose +
  apply_tile, was scalar dequant + scalar dot): gpt_first PACKED
  326-337 -> 221-224 ms (12c25a0).

Measured and not adopted:
- PCONV block grouping (POTATO_PCONV_G): G=2/4/8 at LT 256-512
  all flat to worse than G=1/LT=1024 -- the X slices are served
  by the shared L3, not re-streamed from DRAM. Knob stays,
  default unchanged (fdfcd7b).
- pshufb LUT nibble unpack: bit-identical and 5% SLOWER in the
  mode-B dot harness (77.9 -> 73.7 GMAC/s); the widening tail
  dominates. Not shipped.
- Huge pages (POTATO_HUGEPAGES madvise on the arenas): engages
  (RSS +~5 MB of rounding) with no elapsed-time separation --
  the hypervisor appears to back guest RAM with large pages
  already. Gate stays, default off (39cdbcb).
- PGO build: gpt_first exec -12% but stage exec +9% in the same
  interleaved session; net zero inside the box's noise. Not
  adopted (two-pass build burden for no proven gain).

Known remaining pool, documented and deliberately not started
(multi-day class): panel-packed GEMM for the packed conv and
the fp32 first stage; load-time row-interleaved W4 repack;
fused single-pass decode attention; an AVX2 executor for the
existing i8 GEMV tier (numerics change -- needs gate counts and
a listening verdict before any default); the probs*V register
kernel; an indexed Plan::run API for the host decode loop.

End state: all four gates at their baselines (299/361, 361/361,
0.959881 with internal maxerr 1.66893e-06, fp32 corr 1.000000 /
relrms 1.7e-05 / maxerr 6.25253e-05). AVX2-only E2E on the box:
2,523-2,684 ms (RTF 0.79-0.84) against ~3,180 ms before the
campaign; the interleaved profile deltas account for about
270 ms of it directly (wide dot ~120, gemm_b ~105, bind ~22,
Leaky ~20), the rest is second-order step-count reduction.

## Continuation round (2026-11-26, late)

Shipped:
- probs*V register kernel (ad4a9bb): the decode attention
  product (M=1, N=32, position-major V) keeps its 32
  accumulators in registers across the K loop instead of
  round-tripping memory in the tiled path. Same FMA order --
  bit-identical, all gates exact. Stage exec -37 and -56 ms
  across two interleaved pairs (~-45 ms per request).
- Mode-A GEMV two-row SAXPY (0a3ece4): one accumulator
  load/store round-trip serves two input rows. Bit-identical.
  Stage exec -59 and -67 ms across two interleaved pairs;
  gemv_bench 512x1536 43.6 -> 45.1 GMAC/s.
- write_wav stages PCM and writes once (ee6b81c).

Evaluated, not promoted:
- i8 GEMV tier on AVX2 (4cbea93): a maddubs executor for mode B,
  bit-equal to the portable executor (both score 300/361 under
  POTATO_GEMV_I8, one token above the fp32 floor's 299). At host
  level on the box it does not beat the fp32 dotraw path (two
  interleaved pairs, both negative, mode-B-only included) --
  per-call prelude allocations are the suspected drag. It stays
  behind POTATO_GEMV_I8=1; the unset default on x86 is unchanged
  (the ARM defaults rest on listening verdicts, not this).
- Panel GEMM for the packed conv: a standalone im2col + blocked
  GEMM harness at the leader shape (192->192 K11 L=12000) runs
  21.9 GMAC/s against the direct tile's 42.1 -- the GEMM form
  loses on copy traffic and register pressure. The restructure
  project is closed; ORT's conv lead on this box is the AVX-512
  width, already covered by the 512 tile tier.

End state: gates at baselines. Box E2E at HEAD, standard
sentence: AVX2-only 2,176-2,390 ms (RTF 0.68-0.75, decode
779-868), all tiers 2,288-2,396 ms (RTF 0.71-0.75). Against the
~3,180 ms AVX2-only figure before the evening campaign, the
day's rounds removed roughly a quarter of the sentence time.
Remaining documented pool (multi-day, unstarted): row-interleaved
W4 repack, fused decode attention, an indexed Plan::run API.

## Multi-day pool, worked (2026-11-26, night)

- Indexed Plan::run (c888d0f): run_vec takes inputs in
  input_names() order and returns outputs in output_names()
  order; the map form is an adapter over it. The host decode
  loop keeps stage state as a tensor vector and permutes slots
  between the ping-pong plans with name-derived index tables
  built once -- no per-step maps or renamed key strings.
  Token sequences identical; decode -22 to -32 ms per request
  (interleaved medians) on the box.
- Row-interleaved W4 repack: evaluated at kernel level first.
  A 4-row interleaved mode-B kernel sharing activation loads
  runs 16.3 GMAC/s against the dotraw kernel's 14.6 at the
  512x512 decode shape (+12%) -- the per-row nibble unpack does
  not amortize on AVX2-float (the published wins are integer-dot
  architectures). Below the bar for a pack-format change; the
  project is closed with the number recorded.
- Fused attention step (15721b6, fec501b): compile-time fusion
  of MATMUL -> SOFTMAX(last axis) -> MATMUL into one ATTN step
  (24 sites per GPT plan). Scores use k_matmul's wide dot (or
  its double-accumulation classic form at position stride 32),
  the scores matmul's folded out_scale is applied to the scores
  with the epilogue's double multiply, the softmax replicates
  k_softmax's row math, and probs*V uses pv_n32 -- bit-identical,
  all gates exact. Verified firing on the box (984 ATTN calls
  per stage plan per request); stage exec -40 and -57 ms in two
  interleaved pairs, about -80 to -110 ms per request across
  both stage plans.

End state at fec501b: box E2E, standard sentence -- AVX2-only
2,118-2,190 ms (RTF 0.66-0.68, decode 705-751), all tiers
2,157-2,180 ms. Against the ~3,180 ms AVX2-only figure from
before the evening campaign, the day removed about a third of
the sentence time. The documented pool is now empty except the
cache-owning attention variant above.

## Stateful KV cache (2026-11-26, late)

The last pool item. Compile-time detection marks every
Concat(graph_input, new_slice) -> graph output pair at axis 0
(the stage graphs' 48 past/present pairs; the KV tensors are
[L, 1, 512] position-major rows, which is what makes this
tractable). With a shared KvCache attached (Plan::set_kv_cache,
one per request, both stage plans), bind views the past input
into the pair's padded buffer -- the first run's supplied
tensor is copied in (row-wise; the first plan's presents are
row-strided slices of the QKV projection), later runs pass the
pasts as missing -- and the concat step appends only the new
row and advances the slot length. Growth happens at bind time
only, into uninitialized allocations (both std::vector resize
and make_unique value-initialize, and zero-filling ~100 MB of
capacity cost ~50 ms per session in bind time before this was
caught). POTATO_NO_KV=1 restores the copy protocol end to end.
Token sequences are bit-identical to the copy protocol on both
the first and the second request of a process. Box accounting:
stage-plan CONCAT 144-162 ms -> ~6.5 ms per 84 runs, bind
within ~12 ms of the copy protocol per 84 runs after the
zero-fill fix; same-binary interleaved totals put the win at
roughly 90-130 ms per request. End state at 3df41cf, AVX2-only:
totals 1,962-2,184 ms across the evening's sessions (RTF
0.61-0.68) for the standard sentence, against ~3,180 ms before
the evening campaign -- about 40% of the sentence time removed
in a day. The documented pool is now empty.

## Final comparison and persistence (2026-11-26, closing)

Four-way table, same box, sentence, and seed as the README table:
Genie-TTS 0.49 (5,395 MB), this engine W4 all tiers 0.56, W4
AVX2-only 0.63, fp32 0.84 (2,621 MB), upstream GPT-SoVITS on
PyTorch 2.7 CPU 0.97-0.99 steady state (5,056 MB; first request
25.2 s including lazy init and reference extraction). The fp32
row improved from the ~1.4-1.5 recorded earlier in the campaign:
the fp32 path inherits the executor-level wins (fusion, pool,
indexed bind). The stock pipeline was installed from the
upstream repo at /data/home/stock on the box with a CPU-only
torch 2.7.1 venv (torchaudio 2.7 for its soundfile backend;
2.9+ routes audio loading through torchcodec, which needs
system FFmpeg libraries the box lacks).

Persistence: one host process served 60 consecutive requests
(mixed lengths, one 3,839-character text) with no crash, no
error growth, and no timing degradation (first-10 median
2,161 ms vs last-10 2,095 ms). RSS returned to ~471 MB after
the run; the transient 5 GB peak belongs to the long text's
first-stage arena, a pre-existing profile for extreme inputs.
Outputs vary between requests inside one process, but the
variation is a pre-existing, build-independent property of
the host's sampling history: the pre-KV build, the copy
protocol, and the KV cache produce identical token sequences
given identical request history (verified on a 6-request
controlled run and on the 501-token long request, fresh
process, cache on vs off). Five "empty semantic sequence"
errors on two-word texts mid-soak are the same history effect
(first-token EOS), not a cache fault. The KV growth path was
additionally verified with POTATO_KV_INIT_CAP=64 forcing
repeated doubling: token-identical to the copy protocol.

## Streaming output (2026-11-27, M3)

Request field `"stream": true` (packed free path) emits audio in
chunks: one JSON line per chunk on stdout and the raw chunk audio
(s16le, pre output-chain) appended to reqNNN.pcm as it lands. The
summary gains `streamed`, `first_audio_ms`, and `chunks`. Two
designs were tried and measured on the noiseless fp32 graph
(vits_slice_check):

- Slicing the semantic input is unsound. The vits front aligns
  text and semantic globally (the MRTE cross-attention), so a
  token slice changes the output everywhere: chunked vs one-shot
  corr 0.157, interiors included. Dead end, do not retry.
- The shipped design splits the graph at the vocoder decoder
  (tools/split_vits.py): the front runs once on the full input
  and produces the latent z (/vq_model/Mul_2_output_0,
  [1, 192, 2T], 640 samples per frame); the decoder
  (/vq_model/dec/, 254 packed nodes, inputs z + ge) is purely
  convolutional and exactly local. Decoder chunks of 25 tokens
  with 16 latent frames of context per side reproduce the
  one-shot audio bit-exactly (ctx 8 frames: max err 5.9e-04;
  ctx 0: boundary damage, corr 0.968). z is channel-major: a
  frame slice is a per-channel row copy. An early cut of this
  check sliced z as a flat buffer and reported global coupling;
  that was the check's own layout bug, found by a tail-zeroing
  probe once the indexing was fixed per channel.

Split mechanics: ancestors of z vs ancestors of audio; the one
Constant feeding both sides (gather indices) is duplicated into
the decoder graph; the boundary tensor is redeclared with fresh
shape symbols (the exporter's inferred symbol names do not
rebind as a graph input, and a shapeless declaration leaves
rank 0, which the first decoder Conv cannot lower). The packed
split graphs sit next to vits_wnfold.onnx (front 1.7 MB, dec
156 KB; weights stay in the pool) and compile lazily on the
first stream request; any failure falls back to one-shot with
the reason in the summary's notes.

Box numbers (EPYC, 4 threads, AVX2-only): standard sentence
first audio 1.50 s of a 2.75 s total (the front must finish
before any audio exists, so the win is the decoder tail); a
201-token text: first audio 2.87 s of 6.40 s. Streamed vs plain
final WAV: corr 1.0, max diff 1 LSB. Streaming costs extra
decoder work proportional to the context margins (about +0.6 s
of vits time on the standard sentence at the defaults).

## Packaging (2026-11-27, M3)

tools/make_release.sh builds the free-only host and packs a
tarball with the binary, the full source tree (git archive), a
RELEASE.txt, and SHA256SUMS. Portability is by construction:
only the AVX2 kernel TU carries -mavx2/-mfma flags (the product
baseline), AVX-512 is runtime-dispatched, and there is no
-march=native anywhere. Verified on the EPYC box from a clean
extract: checksums pass (38 files), the packaged binary runs
plain and streaming requests against an existing root, and a
rebuild from the packaged source reproduces the same output
duration on the standard sentence. No weights are included; a
runtime root is still produced by the conversion scripts.

## Runtime reference extraction: design and first probe (2026-11-27, M3)

Goal: build a refpack from a raw wav + prompt text at runtime,
in the host, on the free engine -- the second half of the
dual-mode design (packs remain the supported mode). The offline
producer (host/precompute_refs.py in the asset root) defines the
target exactly. A pack holds five tensors:

  ref_seq      i64[1, L]     prompt phones (the host's own g2p)
  ref_bert     f32[L, 1024]  RoBERTa over the prompt text
  ssl_content  f32[1, 768, T]  HuBERT over the wav at 16 kHz
  ge           f32[1, 1024, 1] prompt_encoder (below)
  ge_advanced  f32[1, 512, 1]  prompt_encoder (below)

prompt_encoder_fp32.onnx takes the raw 32 kHz waveform plus an
SV embedding and produces both ge tensors (mel inside the
graph). The SV embedding comes from the ERes2NetV2 speaker
encoder over the 16 kHz waveform. Host-side DSP still needed:
wav decode and resampling to 16/32 kHz; a WordPiece tokenizer
for RoBERTa (vocab ships in the asset root).

Compile probe (plan_probe, compile-only) against the graphs in
the asset root, 2026-11-27:

  HuBERT (chinese-hubert-base.onnx, 238 KB + external weights)
    Blocked at load: the graph references
    chinese-hubert-base_weights.bin; the asset ships the fp16
    weights as chinese-hubert-base_weights_fp16.bin. A rename at
    pack time fixes the path; the fp16 external initializers
    then need loader support (convert at load, or an fp16 plan
    path like the GPT one).
  RoBERTa (RoBERTa.onnx, 1,707 nodes; inputs input_ids,
    attention_mask, repeats; output phone_features)
    One blocker: a Slice whose end is symbolic ("const slice
    symbolic end"). The engine's symbolic shape system has the
    values; the Slice lowering must learn to defer its bounds
    to bind time.
  Speaker encoder (speaker_encoder.onnx, 806 nodes)
    SequenceEmpty is unsupported; the graph batches frames with
    sequence ops. Either grow the sequence op family or rewrite
    the graph's batching away offline.
  prompt_encoder (prompt_encoder_fp32.onnx, 281 nodes)
    If is unsupported. At runtime the branch is fixed (an SV
    embedding is always supplied), so an offline specialization
    to the taken branch is legitimate; general If support is
    the alternative.

Order of attack, each stage with its oracle from an existing
pack: HuBERT vs ssl_content; RoBERTa vs ref_bert;
prompt_encoder vs ge/ge_advanced (SV embedding from an ORT run
until the speaker encoder lands); speaker encoder last,
validated through prompt_encoder's outputs. Extraction runs
once per new reference, so per-stage speed is secondary to
agreement; the packs' corr baselines (0.96 class for quantized
paths, ~1.0 for fp32) are the acceptance shape.

## Runtime reference extraction: HuBERT stage (2026-11-27, M3)

The HuBERT graph now runs on the free engine end to end.
hubert_check feeds the exact 16 kHz waveform Genie's
ReferenceAudio produced for the neutral1 clip and compares
last_hidden_state against both the ORT run and the shipped
refpack tensor: corr 1.000000 against both, maxerr 2.7e-05
(ORT) and 2.6e-05 (pack). The two oracles agree with each
other to one fp16 ulp (maxerr 1.7e-06), so the pack values
are a valid target for the remaining stages too.

Engine additions, all general:
- InstanceNormalization (per-(n, c) statistics over the
  remaining dims), used once by the feature extractor.
- Erf as a unary op (the exact-GELU form).
- Grouped Conv: the wrapper splits a group>1 conv into
  per-group group-1 convs over strided X/Y views with a
  scratch output per group, reusing the existing tiles
  unchanged. Validated by the encoder's pos-conv (group 16)
  at the encoder-input checkpoint.
- The fused-attention matcher now requires compile-time
  constant head dim 32 on Q/K/V before fusing. It previously
  matched HuBERT's 64-dim attention by pattern alone and
  failed at bind; the GPT plans still fuse (FE_PROFILE:
  4,344 ATTN calls on the packed gate run, as before).

Genie's weight distribution for this model is a flat fp16
bin whose byte layout only matches the graph after an
in-memory fp32 conversion in their loader (the graph declares
fp32 external tensors at fp32 offsets). tools/
convert_fp16_external.py materializes the same conversion
on disk; the graph then loads unmodified.

Two tooling lessons from the bisect: plan outputs can be
strided views (a graph ending in Transpose returns one), and
hubert_check now flattens outputs in logical order before
comparing -- the first full-graph comparison read the view
as flat memory and reported corr 0 for a correct engine.
Checkpoints inside the graph (feature extractor, encoder
input, encoder layer 0) all agreed with ORT at corr 1.0
before the full run did.

## Runtime reference extraction: RoBERTa stage (2026-11-27, M3)

The RoBERTa graph now runs on the free engine end to end.
roberta_check feeds Genie's frontend inputs for the neutral1
prompt (input_ids, attention_mask, word2ph repeats) and
compares phone_features [39, 1024] against the ORT run and
the shipped refpack tensor: corr 1.000000 against both,
maxerr 0.0219 (ORT) and 0.0184 (pack). The two oracles
disagree with each other by up to 0.03125 -- ORT computes
this graph with fp16 activations -- so the engine's fp32
result sits inside the oracles' own spread. A truncated
graph bisect (new generic tool: graph_check, arbitrary
graph + .npy inputs + .npy refs) confirms the encoder
output, the per-word Slice and the counts Where each agree
with ORT at corr 1.0 before the tail runs.

What the graph needed, all general:

- fp16-stored weights: all 357 initializers are dtype 10
  with fp32 compute, the same distribution trick as
  HuBERT's external bin but inline. The compiler decodes
  fp16 constant tensors to fp32 as they enter (graph
  initializers, Constant nodes, Loop body initializers);
  every kernel then sees the fp32 layout it already
  handles. The conversion is exact.
- Symbolic Equal and Where in the shape layer, so the
  exporter's Expand-target chain (Where over Equal over
  Reshape/Concat/Shape/ConstantOfShape) resolves at
  compile time. A runtime fallback sizes dyn Expand
  outputs from the target tensor at exec time.
- Gelu, exact erf form and tanh approximation.
- Slice with a const base and symbolic bounds no longer
  demands const bounds at lowering; it falls through to
  the view path (consts are materialized at bind).
- Sequence and Loop machinery: SequenceEmpty,
  SplitToSequence, SequenceAt, SequenceInsert,
  ConcatFromSequence, and Loop with a recursively
  compiled body sharing the outer value space (subgraph
  attributes in the ONNX parser). Shapes that depend on
  Loop-carried data live in an append-only dyn arena and
  are re-resolved per iteration; body views go stale on
  every write to their base and re-resolve on demand.
- SplitToSequence element shapes stay exact when the
  split tensor is provably all ones; RoBERTa's split is a
  runtime ConstantOfShape with fill 1, which the lowering
  now recognizes, keeping downstream shapes static.

Three bugs worth recording:

- Loop body steps sit inline in the steps array, between
  the steps lowered before and after the Loop node. The
  top-level walk executed the body once standalone --
  with a zeroed iteration slot -- before the Loop step
  ran, corrupting the sequence accumulator. Steps now
  carry an owner (the innermost Loop whose body range
  contains them) and execute only under that owner.
- Dyn outputs bind as placeholder descriptors with dtype
  0. Expand and ConcatFromSequence sized their arena
  allocations from that dtype (elem_size 1 instead of 4)
  and under-allocated 4x, each write clobbering its
  neighbour. Both now take the dtype from the input /
  the sequence elements. The under-allocation was inside
  one arena block, so ASAN could not see it; the shape
  was right and the data was zeros, which is what gave
  it away.
- The first segfault of this stage was not in the tail
  at all: k_layernorm read a 512-float buffer as 1024
  floats because the fp16 LayerNorm weight was handed
  to it undecoded. ASAN found it in one run once the
  build was configured with the free-only flag.

The ASAN build runs roberta_check clean. The four gates
re-run at their baselines after these changes (packed tf
299/361, fp32 tf 361/361, vits packed corr 0.959881 with
packed-vs-baked maxerr 1.66893e-06, vits fp32 corr
1.000000), and the fused ATTN still fires (4,344 calls
on the packed gate run).

Next stages: prompt_encoder (If x2) against ge /
ge_advanced with the SV embedding from an ORT run, then
the speaker encoder (SequenceEmpty + Loop, the machinery
above was built for it), then host integration.

## Runtime reference extraction: prompt_encoder and speaker encoder (2026-11-27, M3)

Both remaining extraction models now run on the free
engine, so all four stages of runtime reference
extraction (HuBERT, RoBERTa, prompt_encoder, speaker
encoder) are proven against their oracles:

- prompt_encoder compiles to 66 steps and matches ORT
  with an ORT speaker embedding: ge corr 1.000000
  (maxerr 1.4e-05), ge_advanced corr 1.000000 (maxerr
  6.9e-05). With the engine's own speaker embedding in
  place of ORT's, the outputs match the shipped
  reference pack itself: corr 1.000000, maxerr 4.4e-05
  / 7.6e-05.
- speaker_encoder compiles to 785 steps and matches
  ORT: corr 1.000000, relrms 2e-06, maxerr 1.0e-05.
  It is slow on the throttled dev box (~9 minutes);
  extraction runs once per reference, so this is
  acceptable.

One oracle correction: prompt_encoder's ref_audio
input is Genie's prepared ReferenceAudio.audio_32k,
not the raw wav file (for neutral1, 136,000 samples
prepared from a 126,400-sample file; the shared prefix
is identical). Fed the true input, ORT reproduces the
pack's ge / ge_advanced bit-exactly; fed the raw file
they differ by ~2. Host integration must replicate
the preparation, not feed the file.

Engine additions for these two graphs:

- General If: both branches compile into the shared
  step array with owner ranges (the Loop owner
  mechanism generalized); the taken branch runs and
  its outputs are copied out, resolving deferred
  views first. The speaker encoder's Loop body
  contains an If whose else branch reads the
  Loop-carried sequence, so branch specialization
  was not an option.
- Runtime step kinds: STFT, DFT, Range, runtime
  Slice, runtime Shape, Tile, BatchNormalization,
  Clip, Prelu, CastLike, Log, Softplus. STFT's input
  order is the one ORT reads -- signal, frame_step,
  window, frame_length -- not the spec's order.
- Dyn-shape machinery: an eff_dyn predicate usable
  during lowering, a compile-time dyn propagation
  pass, per-kind output sizing at exec
  (dyn_shape_for), self-allocating step kinds, and
  deferred views at bind for views of dyn step
  outputs. Shape now honors the start / end attrs.

Bugs worth recording:

- DCE marked used values from step inputs, graph
  outputs, meta refs and view bases, but not If
  branch outputs or Loop body outputs. prompt_
  encoder's Pad step had exactly one reader -- a
  Squeeze view inside an If branch -- and was
  erased; the view then resolved against a stale
  descriptor pointing into an unrelated const
  buffer.
- Expand steps come in two shapes: runtime-target
  steps carry {data, target}; symbolic-target steps
  carry {data} only. The exec dyn branch read
  ins[1] unconditionally -- an out-of-bounds read on
  the runner's shared input vector that ASAN
  reported as a use-after-free inside the runner
  itself.
- Expand's target is a minimum shape, not the
  output shape: the output is broadcast(data,
  target). And a runtime-target Expand's output
  carries placeholder shape symbols nothing can
  bind, so a downstream symbolic Expand now keeps a
  constant target on the step and broadcasts the
  data's true shape at exec.
- Shape of a dyn value with an empty symbolic shape
  produced a symbolic value with zero elements,
  which the const collapse accepted vacuously,
  minting a fake const that a downstream fold read
  through a null pointer. It is a runtime SHAPE
  step now.
- The dyn Pad sizing used interleaved pad indexing;
  ONNX pads are [all begins, then all ends]. The
  4-element under-allocation made the pad copy
  overlap itself inside the arena.
- Optional ONNX inputs materialize as -1 entries in
  a step's input list; every pass that indexes the
  value table over step inputs must skip them (the
  uses counters, DCE and liveness each missed it
  once).
- eff_dyn, the lowering-time dyn predicate, was a
  plain recursion over the value DAG with no
  memoization. On fan-out-heavy graphs it re-walks
  shared ancestors on every query and blows up
  exponentially: a GPT stage plan's compile never
  finished (perf showed 99.9% of cycles inside
  eff_dyn, self-recursive), which presented as all
  four gates hanging on both machines while the
  narrow extraction graphs compiled fine. Memoized
  on the Impl, stage compiles dropped to tens of
  milliseconds.

The four gates re-run at their baselines after
these changes (packed tf 299/361, fp32 tf 361/361,
vits packed corr 0.959881, vits fp32 corr 1.000000;
verified on the EPYC box, with the packed tf gate
independently at 299/361 locally). The ASAN build
runs prompt_encoder clean.

Remaining M3 work at this point: host integration --
wav decode, resampling, a Chinese G2P and a WordPiece
tokenizer for the prompt side, and the four-plan chain
wired into the host (next entry).

## Host integration (2026-11-27, M3-3d)

The runtime reference path is wired end to end. Four pieces,
each verified against the offline Python pipeline (Genie /
genie_tts in the tts-engine venv) on the 69 reference texts
plus synthetic edge cases:

(a) Wav decode + resample (`resample.h`, `wavread.h`,
`resample_check`). RIFF reader (PCM 16/24/32, float,
extensible) with channel averaging; Kaiser-windowed sinc
resampler standing in for soxr 'hq'. For the neutral1
reference (32 kHz native): audio_32k bit-exact; audio_16k
corr 0.999918, with the error confined to soxr's gradual
7.3-8 kHz transition band (steeper/longer fits measured
worse). Downstream through HuBERT: ssl corr 0.999993 vs
the pack. Accepted on that basis.

(b) Chinese G2P (`chn_g2p.h`, `g2pm.h`, `jieba.h`,
`pypinyin_mini.h`, `textnorm.h`). A full port of Genie's
chinese_to_phones + TextNormalizer: g2pM BiLSTM over its
exported weights (76/76 texts exact), jieba_fast dict DAG +
posseg viterbi + finalseg (psg 93/93, cut_for_search
712/712), the pypinyin subset used by the pipeline
(to_initials / to_finals_tone3 / lazy finals: 1366/1366,
712/712), tone sandhi + erhua + polyphonic correction +
opencpop mapping, and the normalizer cascade (t2s,
fullwidth, date/time/range/temperature/measure/ASMD/
power/fraction/percent/phone/version/quantifier/number)
on std::wregex. Assembled: normalize 103/103, g2p 98/98,
full process (clean, phones, ids, word2ph) 98/98 exact.
Model/dict data ships as exported binary assets (export
scripts run the Python packages once, offline). Two
upstream behaviors are mirrored deliberately: the
normalizer's temperature unit reads the wrong regex group
(the emitted unit is always the plain degree word), and
the time-range half-hour test reads the first minute.
Five corpus texts crash Genie itself (an empty lazy final
in the three-tone merge, on interjection characters); the
port produces output there instead. Consistent with the
shipped packs holding 65 packs for 69 clips.

(c) WordPiece tokenizer (`wordpiece.h`, `tokenizer_check`):
the RoBERTa tokenizer.json stack (BertNormalizer,
BertPreTokenizer, WordPiece over vocab.txt). 87/87 exact.

(d) Host wiring (`refextract.h`, `refextract_check`,
main.cpp). A request with "ref_wav" + "ref_text" extracts
the five tensors in memory (HuBERT -> ssl_content,
speaker encoder -> prompt encoder -> ge / ge_advanced,
g2p + tokenizer + RoBERTa -> ref_seq / ref_bert), stores
them in the pack cache under an FNV-1a content key, and
synthesizes through the unchanged pack path. Extraction
plans and assets load lazily from `host/refextract/` on
the first such request (free engine only).

Acceptance on neutral1 vs the offline pack: ref_seq
exact; ref_bert corr 1.000000 (maxerr 0.0184, the free
engine's own RoBERTa delta from 3b); ssl_content corr
0.999993; ge corr 0.999510, ge_advanced corr 0.999835.
The ge delta is traced, not shrugged: the engine's
speaker embedding is exact on identical audio (3c), but
the embedding is sensitive to the resampler's transition
band -- engine SV on the port's audio_16k vs ORT SV on
soxr's gives corr 0.998910, relrms 0.046. Synthesis A/B
(pack vs runtime, same seed): token sequences share a
21-token prefix under sampling and 52 tokens under greedy
before a near-tie flip; the conditioning is functionally
equivalent, the divergence is autoregressive chaos on
fp-level deltas, not an extraction defect. Four gates
unchanged (299/361, 361/361, 0.959881, 1.000000).

## Saved runtime references (2026-11-28, M4)

A `ref_wav` request with `"ref_save": true` now persists its
extracted tensors: `write_rpk` (rpk.h, the mirror of the
loader) writes `host/refpacks/<hash>.rpk`, and a sidecar
`host/refpacks/rt_index.json` maps the `rt:<hash>` tag to
{file, audio, text} so the host never rewrites the user's
index.json. Tag lookup consults index.json first, then the
sidecar. Verified on neutral1: the saved pack loaded by tag
in a fresh process reproduced the runtime request's audio
bit-exactly (corr 1.0, maxdiff 0, tokens 25/25).

## SV resampler fit attempt (2026-11-28, M4 -- rejected)

Hypothesis: the runtime path's one soft tensor, ge (corr
0.999510 vs the pack), traces to the speaker embedding's
sensitivity to the 32k->16k transition band, so a FIR
fitted to soxr HQ's measured magnitude response (firwin2,
1023 taps, sparse probe points) should close it. In
isolation it did: SV embedding corr 0.998910 -> 0.999318,
relrms 0.0464 -> 0.0366. End to end it did not: with the
fitted filter feeding only the speaker encoder, ge fell
to corr 0.998960 (maxerr 1.74) from 0.999510 (1.23) --
the prompt encoder amplifies the fitted filter's residual
direction more than the Kaiser's. A denser 2047-tap fit
against an impulse-response measurement was far worse
everywhere (SV 0.930, ssl 0.974; truncated measurement,
rippled target). Conclusion: embedding corr is a proxy
that misleads one level up, exactly as waveform corr did
in M3-3d(a). The Kaiser resampler stays for both 16k
consumers; the fitted taps were removed unshipped.
