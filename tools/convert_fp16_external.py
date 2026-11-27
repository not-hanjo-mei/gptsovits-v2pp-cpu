#!/usr/bin/env python3
"""Convert a Genie-style fp16 external weight bin to the fp32
external data file an ONNX graph declares.

Some Genie exports ship their weights as one flat fp16 bin while
the graph declares fp32 external tensors (fp32 offsets/lengths,
tensors laid out consecutively in graph order). Genie's own
loader converts the whole bin to fp32 in memory and patches the
tensors back in; this tool materializes the same conversion on
disk so a stock loader can read the graph unchanged:

    convert_fp16_external.py GRAPH.onnx FP16.bin OUTDIR

Copies GRAPH.onnx into OUTDIR and writes the converted fp32 bin
next to it under the location name the graph references. All
external tensors must share one location.
"""
import os
import shutil
import sys

import numpy as np
import onnx


def main():
    src, fp16_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    m = onnx.load(src, load_external_data=False)
    locations = set()
    total = 0
    for t in m.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        kv = {e.key: e.value for e in t.external_data}
        locations.add(kv["location"])
        total = max(total, int(kv["offset"]) + int(kv["length"]))
    if len(locations) != 1:
        sys.exit(f"expected exactly one external location, got {locations}")
    fp16 = np.fromfile(fp16_path, dtype=np.float16)
    if fp16.size * 4 != total:
        sys.exit(f"size mismatch: fp16 bin holds {fp16.size} elements "
                 f"({fp16.size * 4} fp32 bytes), graph expects {total}")
    os.makedirs(outdir, exist_ok=True)
    shutil.copy(src, os.path.join(outdir, os.path.basename(src)))
    out = os.path.join(outdir, locations.pop())
    fp16.astype(np.float32).tofile(out)
    print(f"{out}: {total} bytes fp32 from {fp16_path}")


if __name__ == "__main__":
    main()
