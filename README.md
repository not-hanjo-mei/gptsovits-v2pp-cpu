# gptsovits-v2pp-cpu

CPU-only inference for GPT-SoVITS **v2ProPlus** on x86-64, in native
C++. No Python in the hot path, no GPU. The baseline target is
AVX/AVX2; an AVX-512 tier is runtime-dispatched on CPUs that have it.

Two precision tiers: standard fp32 graphs (with optional fp16 weight
storage), and a weight-only 4-bit (W4) tier at roughly 91 MB of
weights and ~256 MB resident.

Correctness is gated against reference implementations: GPT
teacher-forced agreement is 361/361 tokens at fp32 and 299/361 on
the W4 tier; VITS waveform correlation against the fp32 reference
is 1.000000 (fp32) and 0.959881 (W4). Design notes are in
[DESIGN.md](DESIGN.md).

## Performance

One sentence producing ~3.2 s of audio, seed 42, 4 threads, on a
shared 4-vCPU AMD EPYC box (run-to-run swing ±10–20%). RTF is wall
time divided by audio duration; lower is better.

| Configuration | RTF | Peak RSS |
|---|---|---|
| Genie-TTS (ONNX Runtime, fp16) | 0.49 | 5,395 MB |
| This engine, W4, all dispatch tiers | 0.56 | 256 MB |
| This engine, W4, AVX2 only | 0.63 | 256 MB |
| This engine, fp32 | 0.84 | 2,621 MB |
| Upstream GPT-SoVITS (PyTorch 2.7 CPU) | 0.97–0.99 | 5,056 MB |

## Build and run

```sh
cmake -S engine -B build -DPOTATO_FREE_ONLY=ON
cmake --build build -j 4        # produces build/tts_host
```

`tts_host` reads a root directory holding the converted graph
packs and speaks JSON lines on stdin/stdout: one request line
(`{"text": ..., "tag": ..., "seed": ...}`), one result line with
the output wav path and timings.

Reference audio normally runs from precomputed reference packs.
A request can instead carry `"ref_wav"` and `"ref_text"` and the
host extracts the reference tensors at runtime, caching them by
content hash; `"ref_save": true` also writes them as a pack for
later use as `{"tag": "rt:<hash>"}`. `"ref_lang": "en"` selects
the English prompt path (default is Chinese). The extraction
assets live under `host/refextract/` in the root; see
`tools/REFEXTRACT_ASSETS.md` for how to build them.

With `"stream": true`, output comes out in chunks of 25 tokens
(one JSON line per chunk, audio appended to `reqNNN.pcm`); with
the default context the result is bit-exact against one-shot
synthesis.

No model weights are distributed here. Bring your own v2ProPlus
pair and convert it with the scripts in `tools/`. Inference only;
no training, no GPU path.

## License

MIT (see `LICENSE`). Model weights are not covered: none are
distributed here, and any weights you bring keep their own terms.
