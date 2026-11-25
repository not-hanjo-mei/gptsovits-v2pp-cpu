# gptsovits-v2pp-cpu

CPU-only accelerated inference for GPT-SoVITS **v2ProPlus** on x86-64.

The goal is a native inference engine that loads standard GPT-SoVITS
v2ProPlus weights (the `.ckpt` / `.pth` pair any upstream fine-tune ships),
runs the standard synthesis pipeline, and is fast on plain CPU:
AVX/AVX2 with runtime dispatch, no GPU, no Python in the hot path.

## Status

M1 complete: the engine is in `engine/`, passes its standard-precision
correctness gates (GPT teacher-forced 361/361, VITS correlation
1.000000 against ONNX Runtime), and the host synthesizes end to end
from raw fp32 graphs (`--engine free --fp32-graphs`). Next is M2, the
AVX2 acceleration work. The architecture and the settled design
decisions are in [DESIGN.md](DESIGN.md).

## What this is

- A from-scratch C++ inference engine for the v2ProPlus model pair,
  written against the model architecture, not a port of the PyTorch code.
- Weight-format compatible with upstream: users bring their own
  v2ProPlus fine-tune, converted by scripts shipped here.
- Accelerated where CPU inference actually spends its time: wide-vector
  GEMV for the autoregressive GPT decode, fused small ops in the VITS
  decoder, pinned threading, preallocated buffers.

## What this is not

- No model weights are distributed here, ever. Bring your own.
- No training. Inference only.
- No GPU path. CPU only, by design.

## Background

This project grows out of the PotatoAmika low-resource port, where a
weight-only-quantized variant of this engine runs GPT-SoVITS v2ProPlus
on phone-class ARM hardware. This repository is the desktop x86-64
counterpart at standard precision: normal weights, normal pipeline,
built for speed on AVX/AVX2 machines.

## License

To be decided before the first code drop.
