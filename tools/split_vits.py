#!/usr/bin/env python3
"""Split a GPT-SoVITS vits graph at the vocoder decoder boundary.

The decoder (node names under /vq_model/dec/, plus the small tail
that produces the `audio` output) is purely convolutional, so it
can run on slices of the latent z once the front of the graph
(everything that produces z) has run on the full input. Semantic
level slicing does NOT work: the front aligns text and semantic
globally, and its output changes everywhere when the semantic
input is cut. Decoder level slicing is exact in the interior and
needs only a few latent frames of context at the joints.

Usage:
    split_vits.py IN.onnx FRONT.onnx DEC.onnx

The split point is found from the node named
/vq_model/dec/conv_pre/Conv: its first input is z. The front graph
keeps the original inputs and outputs z; the decoder graph takes
(z, ge) and outputs audio. Any node that feeds both sides aborts
the split (initializers may be shared; computed tensors may not).
"""
import sys

import onnx
from onnx import helper


def ancestors(nodes, prod, targets):
    seen = set()
    stack = list(targets)
    while stack:
        t = stack.pop()
        i = prod.get(t)
        if i is None or i in seen:
            continue
        seen.add(i)
        stack.extend(nodes[i].input)
    return seen


def main():
    src, front_path, dec_path = sys.argv[1], sys.argv[2], sys.argv[3]
    m = onnx.load(src)
    g = m.graph
    nodes = list(g.node)
    prod = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            prod[o] = i
    conv_pre = next(n for n in nodes
                    if n.name in ("/vq_model/dec/conv_pre/Conv",
                                  "/vq_model/dec/conv_pre/Conv_packed"))
    z = conv_pre.input[0]
    audio = g.output[0].name
    front_set = ancestors(nodes, prod, [z])
    dec_set = ancestors(nodes, prod, [audio]) - front_set
    # Tensors the decoder side consumes but the front side produces:
    # only Constant producers may cross (they are stateless, so the
    # node is duplicated into the decoder graph). Anything else is a
    # real third boundary tensor and aborts the split.
    dec_consumed = {x for i in dec_set for x in nodes[i].input}
    front_produced = {o for i in front_set for o in nodes[i].output}
    for x in sorted((dec_consumed & front_produced) - {z}):
        p = prod[x]
        if nodes[p].op_type != "Constant":
            sys.exit(f"split impossible: {x} produced by "
                     f"{nodes[p].op_type} feeds both sides")
        dec_set.add(p)
        print(f"note: duplicating Constant producer of {x} into dec")
    inits = {i.name: i for i in g.initializer}

    def used_inits(idxs):
        names = []
        for i in sorted(idxs):
            for x in nodes[i].input:
                if x in inits and x not in names:
                    names.append(x)
        return [inits[x] for x in names]

    def value_info(name, like_names):
        for v in list(g.value_info) + list(g.input) + list(g.output):
            if v.name == name:
                return v
        return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)

    def side_inputs(idxs, extra):
        # Names this side consumes but does not produce and that are
        # not initializers: original graph inputs (in the packed
        # graphs the __pk/__sc/__ze pool tensors are graph inputs and
        # must be redeclared per side), plus any extra boundary
        # tensors (z for the decoder side).
        produced = set()
        consumed = []
        for i in sorted(idxs):
            produced.update(nodes[i].output)
            for x in nodes[i].input:
                if x not in consumed:
                    consumed.append(x)
        orig = {v.name: v for v in g.input}
        out = [value_info(x, None) for x in extra]
        for x in consumed:
            if x and x not in produced and x not in inits and x not in extra:
                out.append(orig.get(x) or value_info(x, None))
        return out

    front = helper.make_graph(
        [nodes[i] for i in sorted(front_set)],
        g.name + "_front",
        side_inputs(front_set, []),
        [value_info(z, None)],
        initializer=used_inits(front_set),
    )
    # The boundary tensor is declared with fresh symbol names: the
    # exporter's inferred shape uses symbol names the runtime's
    # shape system cannot rebind as a fresh graph input, and a
    # shapeless declaration leaves the rank at zero, which the
    # first decoder Conv cannot lower. Rank and the static channel
    # dim come from the source value_info when available.
    z_src = value_info(z, None)
    zdims = []
    if z_src is not None and z_src.type.HasField("tensor_type"):
        for d in z_src.type.tensor_type.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                zdims.append(d.dim_value)
            else:
                zdims.append(f"zsym{len(zdims)}")
    if not zdims:
        zdims = [None, 192, None]
    z_bare = helper.make_tensor_value_info(z, onnx.TensorProto.FLOAT, zdims)
    dec = helper.make_graph(
        [nodes[i] for i in sorted(dec_set)],
        g.name + "_dec",
        [z_bare] + side_inputs(dec_set, [z])[1:],
        list(g.output),
        initializer=used_inits(dec_set),
    )
    for sub, path in ((front, front_path), (dec, dec_path)):
        mm = helper.make_model(sub, producer_name="split_vits")
        mm.opset_import.extend(m.opset_import)
        mm.ir_version = m.ir_version
        onnx.save(mm, path)
        names = [i.name for i in sub.input]
        print(f"{path}: {len(sub.node)} nodes, {len(names)} inputs "
              f"(real: {[x for x in names if '__pk' not in x and '__sc' not in x and '__ze' not in x]}), "
              f"outputs {[o.name for o in sub.output]}")
    print(f"z tensor: {z}")


if __name__ == "__main__":
    main()
