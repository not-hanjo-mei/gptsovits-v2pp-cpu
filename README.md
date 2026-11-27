# gptsovits-v2pp-cpu

CPU-only accelerated inference for GPT-SoVITS **v2ProPlus** on x86-64.

A native C++ inference engine for the v2ProPlus model pair: it runs the
standard synthesis pipeline from converted graph packs, with no Python
in the hot path and no GPU. The baseline target is AVX/AVX2
(Haswell-class) with runtime dispatch; an AVX-512 tier ships as a
runtime-dispatched bonus on CPUs that have it.

## Status

M1 and M2 are complete, and several post-gate optimization rounds have
landed since. The engine synthesizes end to end in two precision tiers:

- **Standard precision** — raw fp32 graphs (`--fp32-graphs`), with an
  optional fp16 weight-storage mode (`--fp16-weights`) that halves GPT
  weight traffic.
- **W4 tier** — weight-only 4-bit packs (the GPT decoder and the VITS
  network), roughly 91 MB of weights, ~256 MB resident.

Correctness is gated against reference implementations on every
change, and all four gates currently reproduce exactly:

| Gate | Result |
|---|---|
| GPT teacher-forced, fp32 | 361/361 tokens |
| GPT teacher-forced, W4 | 299/361 tokens |
| VITS vs fp32 reference, fp32 | correlation 1.000000 |
| VITS vs fp32 reference, W4 | correlation 0.959881 |

The architecture, the settled design decisions, and the full
optimization log are in [DESIGN.md](DESIGN.md).

## Performance

Measured on a shared 4-vCPU AMD EPYC box (the project's verification
machine), 4 threads, one English sentence producing ~3.2 s of audio,
seed 42. RTF is wall time divided by audio duration; lower is better.

| Configuration | RTF | Peak RSS |
|---|---|---|
| Genie-TTS (ONNX Runtime, fp16) | 0.49 | 5,395 MB |
| This engine, W4 tier, all dispatch tiers | 0.56 | 256 MB |
| This engine, W4 tier, AVX2 only | 0.63 | 256 MB |
| This engine, fp32 | 0.84 | 2,621 MB |
| Upstream GPT-SoVITS (PyTorch 2.7 CPU, eager) | 0.97–0.99 | 5,056 MB |

Notes on these numbers:

- The box is shared, and its run-to-run swing is ±10–20%; ranges above
  are same-session samples, and the per-round evidence in DESIGN.md is
  profile-based where wall clock is not trustworthy.
- Genie-TTS is faster per sentence on this machine and ~21× heavier in
  memory. Its speed comes from ONNX Runtime's graph fusion and MLAS
  kernels on a full-width AVX-512 datapath; DESIGN.md records a
  dissection of where the remaining gap sits (mostly VITS convolution
  rate).
- The upstream row is the stock v2ProPlus pipeline (PyTorch eager
  mode, which includes oneDNN's own weight prepacking), 4 torch
  threads, steady state with its in-process prompt cache warm; its
  first request in a fresh process took 25 s including lazy model
  init and reference feature extraction. This engine instead reads
  pre-extracted reference packs, so its per-request cost never
  includes feature extraction.
- Same-session medians behind the table (2026-11-26): W4 all tiers
  1,805 ms, W4 AVX2-only 2,029 ms, fp32 2,709 ms per sentence.

Where the speed comes from:

- A register-blocked AVX2 convolution tile for VITS, with a
  hand-written assembly core for interior chunks (channels interleaved
  in pairs to lengthen FMA dependency chains), plus an AVX-512 tile
  where the CPU offers one.
- AVX2 W4 GEMV kernels for the GPT decode, with scale and zero-point
  folded out of the inner loops.
- A spin-then-block thread pool (the pattern ONNX Runtime uses) and
  size gates that keep small GEMVs off the pool entirely.
- Compile-time folding of the attention score scale into the matmul
  epilogues, and AVX2 elementwise kernels for the ops whose vector
  form is bit-exact against the scalar one.
- LeakyReLU fused into the convolution and residual-add epilogues,
  dead-step elimination after the compile-time folds, a blocked
  vector prefill path for the mode-B W4 GEMM, and a bind workspace
  whose buffers persist across decode steps.

## Build and run

```sh
cmake -S engine -B build -DPOTATO_FREE_ONLY=ON
cmake --build build -j 4        # produces build/tts_host
```

`tts_host` reads a root directory holding the converted graph packs
and speaks JSON lines on stdin/stdout: one request line
(`{"text": ..., "tag": ..., "seed": ...}`), one result line with the
output wav path and timings. Reference audio normally runs from
precomputed reference packs. A request may instead carry
`"ref_wav"` (a wav path) and `"ref_text"` (its transcript): the
host then extracts the five reference tensors at runtime — wav
decode + resample, Chinese G2P, WordPiece tokenization, and the
HuBERT / RoBERTa / speaker-encoder / prompt-encoder plans, all on
the free engine — and caches them by content hash for the life of
the process. Add `"ref_save": true` and the tensors are also
written to `host/refpacks/<hash>.rpk` and registered in the
sidecar `host/refpacks/rt_index.json`; later processes can then
use them as `{"tag": "rt:<hash>"}` like any offline pack.
Extraction assets live under `host/refextract/` in the root
(see `tools/REFEXTRACT_ASSETS.md` for how to build them).

Streaming: add `"stream": true` to a request (packed path) and the
host runs the vits front once, then emits the vocoder decoder's
output in chunks of 25 tokens — one JSON line per chunk plus the
raw s16le audio appended to `reqNNN.pcm` as each chunk lands. With
the default context the streamed audio is bit-exact against the
one-shot result; the final WAV is unchanged. The split graphs come
from `tools/split_vits.py` (see DESIGN.md, "Streaming output").

## What this is not

- No model weights are distributed here, ever. Bring your own.
- No training. Inference only.
- No GPU path. CPU only, by design.

## Background

This project grows out of the PotatoAmika low-resource port, where a
weight-only-quantized variant of this engine runs GPT-SoVITS v2ProPlus
on phone-class ARM hardware. This repository is the desktop x86-64
counterpart: normal weights, normal pipeline, built for speed on
AVX/AVX2 machines.

## License

MIT (see `LICENSE`). Model weights are not covered: none are
distributed here, and any weights you bring keep their own terms.
