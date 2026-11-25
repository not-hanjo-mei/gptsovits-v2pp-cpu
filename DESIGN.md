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
3. **M2.** AVX2 kernels for decode GEMV and the heavy VITS blocks,
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
