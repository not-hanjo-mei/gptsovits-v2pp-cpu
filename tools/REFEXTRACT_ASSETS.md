# Building the runtime-extraction assets

Runtime reference extraction (requests with `ref_wav` +
`ref_text`) loads its models and data lazily from
`<root>/host/refextract/`:

```
host/refextract/
  hubert/chinese-hubert-base.onnx      (+ its external weights bin)
  RoBERTa.onnx
  speaker_encoder.onnx
  prompt_encoder_fp32.onnx
  vocab.txt
  g2p/
    g2pm.bin  jieba_posseg.bin  finalseg.bin  pypinyin.bin
    sandhi.bin  polyphonic.bin  t2s.bin  symbols_v2.txt
    dict.txt  opencpop-strict.txt
  eng/                                  (only for ref_lang "en")
    eng.bin  tagger.bin
    wordsegment/{unigrams,bigrams,words}.txt
```

None of this is distributed with the repo. Build it from a
Genie-TTS installation: you need its data directory (`GENIE`
below) and a Python environment with `genie_tts`, `g2pM`,
`jieba_fast` and `pypinyin` installed (Genie's own venv has
all four); `SITE` is that env's site-packages.

## Model graphs

HuBERT ships as an fp32 graph with an fp16 external weights
bin; convert it once:

```
python3 tools/convert_fp16_external.py \
    GENIE/chinese-hubert-base/chinese-hubert-base.onnx \
    GENIE/chinese-hubert-base/chinese-hubert-base_weights_fp16.bin \
    host/refextract/hubert/
```

Copy `GENIE/RoBERTa/RoBERTa.onnx` as-is, with
`GENIE/RoBERTa/roberta_tokenizer/vocab.txt` next to it, and
copy `GENIE/speaker_encoder.onnx`.

The prompt encoder is an fp32 ONNX export of the voice
checkpoint's prompt-encoder submodule. Contract: inputs
`ref_audio` f32 [1, N] (32 kHz waveform) and `sv_emb`
f32 [1, 20480]; outputs `ge` [1, 1024, 1] and `ge_advanced`
[1, 512, 1]. Any export with that signature works.

## G2P data

```
python3 tools/export_g2pm.py SITE host/refextract/g2p
python3 tools/export_chn_g2p.py SITE GENIE host/refextract/g2p
cp SITE/jieba_fast/dict.txt host/refextract/g2p/
cp GENIE/G2P/ChineseG2P/opencpop-strict.txt host/refextract/g2p/

python3 tools/export_eng_g2p.py GENIE host/refextract/eng
cp -r GENIE/G2P/EnglishG2P/wordsegment host/refextract/eng/
```

The export scripts are the only place pickle is read; the
engine loads the plain formats they emit, documented in the
scripts and their C++ loaders.

## Verifying

`refextract_check ASSETDIR WAV PROMPT_TEXT PACK.rpk [LANG]`
compares a runtime extraction against an offline pack for
the same reference. `chn_g2p_check`, `g2pm_check`,
`jieba_check`, `pypinyin_check`, `tokenizer_check` and
`eng_g2p_check` check the individual stages against Python
oracles.
