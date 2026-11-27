"""Export g2pM's pickle data to the plain binary asset the
C++ Chinese G2P loads (engine/src/g2pm.h), plus an optional
pinyin oracle for the port check.

The g2pM package ships its model as pickles; unpickling is a
build-time-only step here -- the engine never sees pickle.

Usage:
  export_g2pm.py SITE_PACKAGES OUTDIR [--emotions EMOTIONS.json]

SITE_PACKAGES is the site-packages dir of a Python env with
g2pM installed (Genie-TTS's venv works). OUTDIR receives
g2pm.bin (and g2pm_oracle.jsonl when --emotions is given).

g2pm.bin layout (all little-endian):
  magic 'G2PM', u32 version=1
  cedict: u32 n; per entry: u32 codepoint, u8 n_prons,
          per pron: u16 len + utf8
  char2idx: u32 n; per entry: u16 key_len + key utf8, i32 idx
  idx2class: u32 n; per entry: u16 len + utf8 (index = position)
  weights: u32 n; per entry: u16 name_len + name, u8 dtype
          (0=f32, 1=f64), u8 ndim, ndim * u32 dims, raw data
"""
import json
import pickle
import struct
import sys

import numpy as np

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)
SITE = sys.argv[1]
OUT = sys.argv[2]
PKG = f"{SITE}/g2pM"
EMOTIONS = None
if "--emotions" in sys.argv:
    EMOTIONS = sys.argv[sys.argv.index("--emotions") + 1]


def load(name):
    with open(f"{PKG}/{name}", "rb") as f:
        return pickle.load(f)


cedict = load("digest_cedict.pkl")
char2idx = load("char2idx.pkl")
class2idx = load("class2idx.pkl")
ckpt = load("np_ckpt.pkl")

print("cedict chars:", len(cedict))
print("char2idx:", len(char2idx), "classes:", len(class2idx))
for k, v in ckpt.items():
    print("weight", k, v.shape, v.dtype)

idx2class = [None] * len(class2idx)
for pron, idx in class2idx.items():
    idx2class[idx] = pron

buf = bytearray()
buf += b"G2PM" + struct.pack("<I", 1)
buf += struct.pack("<I", len(cedict))
for ch, prons in cedict.items():
    assert len(ch) == 1
    buf += struct.pack("<IB", ord(ch), len(prons))
    for p in prons:
        b = p.encode("utf-8")
        buf += struct.pack("<H", len(b)) + b
buf += struct.pack("<I", len(char2idx))
for k, v in char2idx.items():
    b = k.encode("utf-8")
    buf += struct.pack("<H", len(b)) + b + struct.pack("<i", v)
buf += struct.pack("<I", len(idx2class))
for p in idx2class:
    b = p.encode("utf-8")
    buf += struct.pack("<H", len(b)) + b
buf += struct.pack("<I", len(ckpt))
for name, arr in ckpt.items():
    arr = np.ascontiguousarray(arr)
    dt = 0 if arr.dtype == np.float32 else 1 if arr.dtype == np.float64 else None
    assert dt is not None, (name, arr.dtype)
    b = name.encode("utf-8")
    buf += struct.pack("<H", len(b)) + b + struct.pack("<BB", dt, arr.ndim)
    for d in arr.shape:
        buf += struct.pack("<I", d)
    buf += arr.tobytes()
with open(f"{OUT}/g2pm.bin", "wb") as f:
    f.write(buf)
print("g2pm.bin bytes:", len(buf))

if EMOTIONS:
    sys.path.insert(0, SITE)
    from g2pM import G2pM

    g = G2pM()
    texts = []
    emo = json.load(open(EMOTIONS))
    for k, v in emo.items():
        if not k.startswith("_"):
            texts.append(v["text"])
    texts += [
        "欢迎来到梦神之城,我是这边的引路人,小樱.",
        "他还行了吗?银行行长不行了.",
        "重庆市长江大桥的重量很重要.",
        "音乐快乐,长大成长,长短长短.",
        "得到的东西,觉得很好,觉悟.",
        "我看著你,你看著我.",
        "嗯,啊,哦.",
    ]
    with open(f"{OUT}/g2pm_oracle.jsonl", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "pinyins": g(t, char_split=True)}, ensure_ascii=False) + "\n")
    print("oracle cases:", len(texts))
