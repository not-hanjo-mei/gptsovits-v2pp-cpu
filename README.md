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
| This engine, W4 tier, all dispatch tiers | 0.77–0.81 | 256 MB |
| This engine, W4 tier, AVX2 only | 0.79–0.84 | 256 MB |
| This engine, fp32 | ~1.4–1.5 | 2,586 MB |
| Genie-TTS (ONNX Runtime, same machine) | 0.55 | 5,485 MB |

Notes on these numbers:

- The box is shared, and its run-to-run swing is ±10–20%; ranges above
  are same-session samples, and the per-round evidence in DESIGN.md is
  profile-based where wall clock is not trustworthy.
- Genie-TTS is faster per sentence on this machine and ~21× heavier in
  memory. Its speed comes from ONNX Runtime's graph fusion and MLAS
  kernels; DESIGN.md records a dissection of where the remaining gap
  sits (mostly VITS convolution rate).
- The final comparison table also includes upstream GPT-SoVITS
  (PyTorch CPU) on the same machine; that leg is still pending, so the
  table above is partial.

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
output wav path and timings. Reference audio runs from precomputed
reference packs today; runtime extraction (wav in, features computed
on the spot, optionally written back as a pack) is the settled
dual-mode design in DESIGN.md and is not implemented yet.

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

Not yet chosen.
