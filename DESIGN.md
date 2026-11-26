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
