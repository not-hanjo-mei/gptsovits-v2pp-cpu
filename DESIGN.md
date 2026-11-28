# Design

A native C++ engine that runs GPT-SoVITS v2ProPlus inference on
x86-64 CPUs with AVX/AVX2, faster than the upstream PyTorch CPU
path at a fraction of its memory. No GPU, no Python in the hot
path, standard weights.

Input is an upstream v2ProPlus pair (`*.ckpt` for GPT, `*.pth`
for SoVITS). Conversion scripts in `tools/` turn it into the
engine's runtime layout, offline; converted weights are never
committed here. Synthesis semantics follow upstream, including
the sampling controls (temperature, top_k, top_p, seed,
repetition penalty).

The pipeline is upstream's: text frontend (G2P, plus BERT
features for Chinese), reference encoding (HuBERT semantic
features and the VITS prompt encoder), autoregressive GPT
decode of semantic tokens, one VITS pass to a waveform, and a
trim/gain/soft-clip output chain.

## Reference modes

A reference voice can be supplied two ways, and the person
running the engine picks per voice. It can be precomputed into
a pack file, so startup is milliseconds and the extraction
models never load. Or a wav plus its transcript can be given
at runtime and the engine extracts the features on the spot.
Both modes produce the same five tensors — ref_seq, ref_bert,
ssl_content, ge, ge_advanced — and everything downstream of
the features is one code path. A runtime extraction can be
saved back as a pack (`"ref_save": true`, registered in a
sidecar index), so a voice extracted once can be reused as a
pack forever.

Runtime extraction runs HuBERT, RoBERTa, the speaker encoder
and the prompt encoder on the engine itself, plus ported
Chinese and English G2P frontends. Against the offline Python
pipeline it reproduces ref_seq exactly and the float tensors
at correlation 0.9995 or better; the one soft spot is ge,
whose delta traces to the speaker embedding's sensitivity to
the resampler's transition band. (Fitting the resampler to
soxr's response improves the embedding in isolation and makes
ge worse end to end, so the plain Kaiser resampler stays.)
English prompts take a zero ref_bert, as upstream does, so
the English path needs no RoBERTa.

## Where the speed comes from

GPT decode is latency-bound: one token per step, each step
reading the full weight set. What pays is wide-vector GEMV
kernels, a KV cache that appends instead of copying, folding
the attention scale into the matmul epilogues, and keeping
small jobs off the thread pool. VITS is hundreds of small ops;
what pays there is a register-blocked convolution tile (with a
hand-written assembly core), op fusion into epilogues, and
preallocated arenas. The W4 tier cuts weight traffic to 4 bits
per weight with the scales folded out of the inner loops.
Everything wider than AVX2 is runtime-dispatched; the AVX2
path stands on its own.

## Correctness gates

GPT is checked by teacher-forced token agreement against the
PyTorch reference, VITS by full-graph waveform correlation,
and end-to-end output by fixed sentences at a fixed seed.
Current baselines: 361/361 (fp32) and 299/361 (W4) tokens;
VITS correlation 1.000000 (fp32) and 0.959881 (W4). A change
that moves a gate does not ship unless the move is understood
and recorded.

## Streaming

With `"stream": true` the host runs the VITS front once, then
decodes the vocoder in chunks of 25 tokens. The split has to
be at the decoder: the front aligns text and semantics
globally, so slicing the semantic input is unsound, while the
decoder is purely convolutional and exactly local. With 16
latent frames of context per side the streamed audio is
bit-exact against the one-shot result.

## Packaging

`tools/make_release.sh` builds the free-only host and packs a
tarball with the binary, the full source tree, and checksums.
Only the AVX2 kernel file carries `-mavx2`/`-mfma` flags and
there is no `-march=native` anywhere, so the binary runs on
any AVX2 machine. No weights are included; a runtime root is
still produced by the conversion scripts.
