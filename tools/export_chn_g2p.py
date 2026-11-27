"""Export the Chinese G2P data assets for the C++ port
(engine/src/chn_g2p.h and friends): jieba posseg/finalseg
HMM tables, the tone-sandhi/erhua word sets, the polyphonic
correction dict, the traditional->simplified map,
symbols_v2, and pypinyin's dictionaries. All outputs are
plain data files; pickle is read only here, at build time.

Usage:
  export_chn_g2p.py SITE_PACKAGES GENIE_DATA_DIR OUTDIR

SITE_PACKAGES is the site-packages dir of a Python env with
genie_tts, jieba_fast and pypinyin installed (Genie-TTS's
venv works). GENIE_DATA_DIR is Genie's data directory (it
holds G2P/ChineseG2P/polyphonic.pickle). Outputs:
  jieba_posseg.bin (JPS1), finalseg.bin (FSG1),
  sandhi.bin (SHD1), polyphonic.bin (PPD1), t2s.bin (T2S1),
  symbols_v2.txt, pypinyin.bin (PYP1)
Formats are documented by their loaders in engine/src/.
"""
import os
import pickle
import struct
import sys

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(2)
VENV = sys.argv[1]
GENIE_DATA = sys.argv[2]
OUT = sys.argv[3]
os.environ.setdefault("GENIE_DATA_DIR", GENIE_DATA)
sys.path.insert(0, VENV)


def w_str(buf, s):
    b = s.encode("utf-8")
    buf += struct.pack("<H", len(b)) + b
    return buf


# ---------- jieba posseg tables ----------
from jieba_fast.posseg.char_state_tab import P as char_state_tab
from jieba_fast.posseg.prob_start import P as start_p
from jieba_fast.posseg.prob_trans import P as trans_p
from jieba_fast.posseg.prob_emit import P as emit_p

states = []
state_id = {}


def sid(st):
    if st not in state_id:
        state_id[st] = len(states)
        states.append(st)
    return state_id[st]


for v in char_state_tab.values():
    for st in v:
        sid(st)
for st in start_p:
    sid(st)
for st, m in trans_p.items():
    sid(st)
    for t in m:
        sid(t)
for st in emit_p:
    sid(st)
print("posseg states:", len(states))

buf = bytearray(b"JPS1")
buf += struct.pack("<I", len(states))
for pos, tag in states:
    buf += struct.pack("<B", ord(pos))
    buf = w_str(buf, tag)
buf += struct.pack("<I", len(char_state_tab))
for ch, sts in char_state_tab.items():
    assert len(ch) == 1
    buf += struct.pack("<IH", ord(ch), len(sts))
    for st in sts:
        buf += struct.pack("<I", state_id[st])
buf += struct.pack("<I", len(states))
for st in states:
    if st in start_p:
        buf += struct.pack("<Bd", 1, start_p[st])
    else:
        buf += struct.pack("<Bd", 0, 0.0)
buf += struct.pack("<I", len(states))
for st in states:
    m = trans_p.get(st, {})
    buf += struct.pack("<I", len(m))
    for t, v in m.items():
        buf += struct.pack("<Id", state_id[t], v)
buf += struct.pack("<I", len(trans_p))
for st in trans_p:
    buf += struct.pack("<I", state_id[st])
buf += struct.pack("<I", len(states))
for st in states:
    m = emit_p.get(st, {})
    buf += struct.pack("<I", len(m))
    for ch, v in m.items():
        assert len(ch) == 1
        buf += struct.pack("<Id", ord(ch), v)
open(f"{OUT}/jieba_posseg.bin", "wb").write(buf)
print("jieba_posseg.bin:", len(buf))

# ---------- jieba finalseg tables (for cut_for_search) ----------
from jieba_fast.finalseg.prob_start import P as fs_start
from jieba_fast.finalseg.prob_trans import P as fs_trans
from jieba_fast.finalseg.prob_emit import P as fs_emit

BMES = "BMES"
buf = bytearray(b"FSG1")
for s in BMES:
    buf += struct.pack("<Bd", 1 if s in fs_start else 0, fs_start.get(s, 0.0))
for a in BMES:
    for b in BMES:
        v = fs_trans.get(a, {}).get(b)
        buf += struct.pack("<Bd", 1 if v is not None else 0, v if v is not None else 0.0)
for s in BMES:
    m = fs_emit.get(s, {})
    buf += struct.pack("<I", len(m))
    for ch, v in m.items():
        assert len(ch) == 1
        buf += struct.pack("<Id", ord(ch), v)
open(f"{OUT}/finalseg.bin", "wb").write(buf)
print("finalseg.bin:", len(buf))

# ---------- ToneSandhi / Erhua word sets ----------
from genie_tts.G2P.Chinese.ToneSandhi import ToneSandhi
from genie_tts.G2P.Chinese.Erhua import ErhuaProcessor

ts = ToneSandhi()
ep = ErhuaProcessor()
buf = bytearray(b"SHD1")
for s in (sorted(ts.must_neural_tone_words), sorted(ts.must_not_neural_tone_words),
          sorted(ep.must_erhua), sorted(ep.not_erhua)):
    buf += struct.pack("<I", len(s))
    for w in s:
        buf = w_str(buf, w)
open(f"{OUT}/sandhi.bin", "wb").write(buf)
print("sandhi.bin:", len(buf))

# ---------- polyphonic correction dict ----------
with open(f"{GENIE_DATA}/G2P/ChineseG2P/polyphonic.pickle", "rb") as f:
    pp = pickle.load(f)
print("polyphonic entries:", len(pp))
buf = bytearray(b"PPD1")
buf += struct.pack("<I", len(pp))
for word, prons in pp.items():
    buf = w_str(buf, word)
    buf += struct.pack("<B", len(prons))
    for p in prons:
        buf = w_str(buf, p)
open(f"{OUT}/polyphonic.bin", "wb").write(buf)
print("polyphonic.bin:", len(buf))

# ---------- traditional -> simplified ----------
import importlib.util

spec = importlib.util.spec_from_file_location(
    "char_convert", f"{VENV}/genie_tts/G2P/Chinese/Normalization/char_convert.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
simp = cc.simplified_charcters
trad = None
for name in dir(cc):
    if "traditional" in name:
        trad = getattr(cc, name)
assert trad is not None and len(trad) == len(simp)
buf = bytearray(b"T2S1")
buf += struct.pack("<I", len(simp))
for t, s in zip(trad, simp):
    buf += struct.pack("<II", ord(t), ord(s))
open(f"{OUT}/t2s.bin", "wb").write(buf)
print("t2s.bin:", len(buf))

# ---------- symbols_v2 ----------
from genie_tts.G2P.SymbolsV2 import symbols_v2

with open(f"{OUT}/symbols_v2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(symbols_v2) + "\n")
print("symbols_v2:", len(symbols_v2))

# ---------- pypinyin ----------
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
from pypinyin.phonetic_symbol import phonetic_symbol

print("pinyin_dict:", len(PINYIN_DICT), "phrases:", len(PHRASES_DICT),
      "phonetic symbols:", len(phonetic_symbol))
buf = bytearray(b"PYP1")
buf += struct.pack("<I", len(PINYIN_DICT))
for cp, v in PINYIN_DICT.items():
    prons = v.split(",")
    buf += struct.pack("<IB", cp, len(prons))
    for p in prons:
        buf = w_str(buf, p)
buf += struct.pack("<I", len(PHRASES_DICT))
for word, per_char in PHRASES_DICT.items():
    buf = w_str(buf, word)
    buf += struct.pack("<B", len(per_char))
    for prons in per_char:
        buf += struct.pack("<B", len(prons))
        for p in prons:
            buf = w_str(buf, p)
buf += struct.pack("<I", len(phonetic_symbol))
for sym, rep in phonetic_symbol.items():
    buf = w_str(buf, sym)
    buf = w_str(buf, rep)
open(f"{OUT}/pypinyin.bin", "wb").write(buf)
print("pypinyin.bin:", len(buf))
