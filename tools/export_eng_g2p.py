"""Export the English G2P data assets for the C++ port
(engine/src/eng_g2p.h): the merged CMU dictionary (cache
pickle updated by the hot dict, minus the six removals in
EnglishG2P.py), the name dictionary, and the OOV GRU
weights from checkpoint20.npz. Pickle/npz are read only
here, at build time; the engine loads the plain ENG1
binary. Only each word's first pronunciation is stored --
that is all EnglishG2P ever uses.

Also writes tagger.bin: the averaged-perceptron tagger's
classes/tagdict/weights flattened from its JSON files
(loading the 5.5 MB weights JSON in the engine costs more
memory than the model itself). The wordsegment tables are
already plain text: copy them verbatim (see
tools/REFEXTRACT_ASSETS.md).

Usage:
  export_eng_g2p.py GENIE_DATA_DIR OUTDIR

eng.bin layout (little-endian):
  magic 'ENG1'
  cmu:      u32 n; per entry u16 word_len + word,
            u8 n_ph, per phone u16 len + phone
  namedict: same again
  weights:  u32 n; per entry u16 name_len + name,
            u8 dtype (0=f32, 1=f64), u8 ndim,
            ndim * u32 dims, raw data

tagger.bin layout (little-endian):
  magic 'PTG1'
  classes:  u32 n; per entry u16 len + string
  tagdict:  u32 n; per entry u16 len + word, u16 len + tag
  weights:  u32 n_features; per feature u16 len + string,
            u32 n; per entry u16 class_index, f64 weight
"""
import os
import pickle
import re
import struct
import sys

import numpy as np

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)
D = os.path.join(sys.argv[1], "G2P", "EnglishG2P")
OUT = sys.argv[2]


def read_cmu_dict(path):
    g2p_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) < 2:
                continue
            word, pron_str = parts[0].lower(), parts[1]
            pron = pron_str.split(" ")
            word = re.sub(r"\(\d+\)$", "", word)
            if word not in g2p_dict:
                g2p_dict[word] = [pron]
    return g2p_dict


with open(os.path.join(D, "engdict_cache.pickle"), "rb") as f:
    cmu = pickle.load(f)
hot = read_cmu_dict(os.path.join(D, "engdict-hot.rep"))
if hot:
    cmu.update(hot)
for word in ["ae", "ai", "ar", "ios", "hud", "os"]:
    cmu.pop(word, None)
with open(os.path.join(D, "namedict_cache.pickle"), "rb") as f:
    namedict = pickle.load(f)
print("cmu:", len(cmu), "namedict:", len(namedict))

buf = bytearray(b"ENG1")


def w_dict(d):
    global buf
    buf += struct.pack("<I", len(d))
    for word, prons in d.items():
        pron = prons[0]
        b = word.encode("utf-8")
        buf += struct.pack("<H", len(b)) + b
        buf += struct.pack("<B", len(pron))
        for ph in pron:
            pb = ph.encode("utf-8")
            buf += struct.pack("<H", len(pb)) + pb


w_dict(cmu)
w_dict(namedict)

variables = np.load(os.path.join(D, "checkpoint20.npz"))
names = sorted(variables.files)
print("weights:", [(n, variables[n].shape, str(variables[n].dtype)) for n in names])
buf += struct.pack("<I", len(names))
for name in names:
    arr = np.ascontiguousarray(variables[name])
    dt = 0 if arr.dtype == np.float32 else 1 if arr.dtype == np.float64 else None
    assert dt is not None, (name, arr.dtype)
    b = name.encode("utf-8")
    buf += struct.pack("<H", len(b)) + b + struct.pack("<BB", dt, arr.ndim)
    for dim in arr.shape:
        buf += struct.pack("<I", dim)
    buf += arr.tobytes()
with open(os.path.join(OUT, "eng.bin"), "wb") as f:
    f.write(buf)
print("eng.bin bytes:", len(buf))

# ---------- perceptron tagger (JSON -> flat binary) ----------
import json

TD = os.path.join(D, "taggers", "averaged_perceptron_tagger_eng")
base = "averaged_perceptron_tagger_eng"
with open(os.path.join(TD, f"{base}.classes.json")) as f:
    classes = json.load(f)
with open(os.path.join(TD, f"{base}.tagdict.json")) as f:
    tagdict = json.load(f)
with open(os.path.join(TD, f"{base}.weights.json")) as f:
    weights = json.load(f)
print("tagger classes:", len(classes), "tagdict:", len(tagdict), "features:", len(weights))
cls_id = {c: i for i, c in enumerate(classes)}
buf = bytearray(b"PTG1")


def w_str2(b, s):
    e = s.encode("utf-8")
    b += struct.pack("<H", len(e)) + e
    return b


buf += struct.pack("<I", len(classes))
for c in classes:
    buf = w_str2(buf, c)
buf += struct.pack("<I", len(tagdict))
for w, t in tagdict.items():
    buf = w_str2(buf, w)
    buf = w_str2(buf, t)
buf += struct.pack("<I", len(weights))
for feat, tw in weights.items():
    buf = w_str2(buf, feat)
    buf += struct.pack("<I", len(tw))
    for t, v in tw.items():
        buf += struct.pack("<Hd", cls_id[t], v)
with open(os.path.join(OUT, "tagger.bin"), "wb") as f:
    f.write(buf)
print("tagger.bin bytes:", len(buf))
