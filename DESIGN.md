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

## The one open decision: where stage 2 runs

Two shapes are possible, and they lead to different engines:

- **A. Full standard.** Reference audio is accepted at run time. HuBERT
  (and BERT for Chinese) run inside the engine. Drop-in compatible with
  upstream usage, any reference clip works immediately, but the engine
  carries two more models and their cost.
- **B. Precomputed reference packs.** Reference features are computed
  once by an offline tool and shipped to the engine as a pack. The
  runtime is stages 1, 3, 4, 5 only: smaller, faster to start, and the
  shape already proven in the PotatoAmika ARM engine. Changing the
  reference voice costs one offline step.

This decision gates the graph set, the runtime dependencies and the
conversion scripts. It must be settled before code.

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
2. **M1.** Weight conversion plus a straightforward fp32 engine that
   passes the correctness gates. No acceleration claims at M1.
3. **M2.** AVX2 kernels for decode GEMV and the heavy VITS blocks,
   fp16 weight storage, thread pinning. RTF comparison against stock
   on the project benchmark set.
4. **M3.** Hardening: streaming output, packaging, docs.
