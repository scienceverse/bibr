"""Rewrite the mask-to-box subgraph of an exported PP-DocLayoutV3 ONNX bundle.

Usage::

    python scripts/rewrite_onnx_layout_mask_boxes.py SRC_DIR DST_DIR

``SRC_DIR`` is the output of ``scripts/export_onnx_layout.py``. ``DST_DIR`` gets
``onnx/{model.onnx,bibr_onnx.json}``.

The exported graph turns each of the 300 query masks (logits ``[1, 300, 200, 200]``)
into a box and an "area > 0" flag, the decoder's initial reference boxes. It does so
with full-size operators over 12 million elements, about a fifth of the graph's CPU
time. A box only needs to know whether a column or a row holds a positive pixel, so
this reduces the logits to ``[1, 300, 200]`` twice and works on those. Every value
that reaches the rest of the graph is the float32 the original computed, including
``FLT_MAX`` and ``1.0`` for an empty mask.

The tensors are found by their exported names and checked first, so a graph from
another exporter or architecture is refused instead of rewritten wrongly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_PREFIX = "/inner/model/"
_REWRITE_ID = "layout-mask-boxes-v1"

# The tensors the rest of the graph reads from the old subgraph, and what makes them.
_OUT_XMIN = _PREFIX + "ReduceMin_output_0"
_OUT_YMIN = _PREFIX + "ReduceMin_1_output_0"
_OUT_XMAX1 = _PREFIX + "Add_40_output_0"
_OUT_YMAX1 = _PREFIX + "Add_41_output_0"
_OUT_NONEMPTY = _PREFIX + "Greater_3_output_0"
_REPLACED = {
    _OUT_XMIN: ("ReduceMin", _PREFIX + "ReduceMin"),
    _OUT_YMIN: ("ReduceMin", _PREFIX + "ReduceMin_1"),
    _OUT_XMAX1: ("Add", _PREFIX + "Add_40"),
    _OUT_YMAX1: ("Add", _PREFIX + "Add_41"),
    _OUT_NONEMPTY: ("Greater", _PREFIX + "Greater_3"),
}
_MASK = _PREFIX + "Reshape_33_output_0"  # float logits [1, Q, H, W]
_BOOL_MASK = _PREFIX + "Cast_15_output_0"
_XS, _YS = _PREFIX + "Range_8_output_0", _PREFIX + "Range_7_output_0"  # int64 [W], [H]
_SHAPE_PROBES = (_PREFIX + "Shape_112", _PREFIX + "Shape_113")


def _check_graph(graph: onnx.GraphProto) -> None:
    producers = {out: node for node in graph.node for out in node.output}
    for tensor, (op, name) in _REPLACED.items():
        node = producers.get(tensor)
        if node is None or node.op_type != op or node.name != name:
            raise ValueError(f"not the expected layout graph: {tensor} is not made by {name}")
    for tensor in (_MASK, _XS, _YS):
        if tensor not in producers:
            raise ValueError(f"not the expected layout graph: {tensor} is missing")
    by_name = {node.name: node for node in graph.node}
    for name in _SHAPE_PROBES:
        node = by_name.get(name)
        if node is None or list(node.input) != [_BOOL_MASK]:
            raise ValueError(f"not the expected layout graph: {name} does not read {_BOOL_MASK}")


def _topological(nodes: list, known: set[str]) -> list:
    ordered, pending, produced = [], list(nodes), set(known)
    while pending:
        ready = [n for n in pending if all(not t or t in produced for t in n.input)]
        if not ready:
            raise ValueError(f"{len(pending)} nodes cannot be ordered, e.g. {pending[0].name}")
        ready_ids = {id(n) for n in ready}
        ordered.extend(ready)
        for node in ready:
            produced.update(node.output)
        pending = [n for n in pending if id(n) not in ready_ids]
    return ordered


def _live(nodes: list, outputs: set[str]) -> list:
    producers = {out: node for node in nodes for out in node.output}
    live: set[int] = set()
    stack = list(outputs)
    while stack:
        node = producers.get(stack.pop())
        if node is None or id(node) in live:
            continue
        live.add(id(node))
        stack.extend(t for t in node.input if t)
    return [n for n in nodes if id(n) in live]


def rewrite_mask_boxes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace the mask-to-box subgraph of ``model`` in place and return it."""
    graph = model.graph
    _check_graph(graph)

    # The probes read the bool mask's shape; the logits have the same one.
    for node in graph.node:
        if node.name in _SHAPE_PROBES:
            node.input[0] = _MASK

    new_nodes = []

    def add(op: str, inputs: list[str], output: str, **attrs) -> str:
        name = _PREFIX + "mask_boxes/" + output.rsplit("/", 1)[-1]
        new_nodes.append(helper.make_node(op, inputs, [output], name=name, **attrs))
        return output

    def local(name: str) -> str:
        return _PREFIX + "mask_boxes/" + name

    def const(name: str, value: float) -> str:
        tensor = numpy_helper.from_array(np.float32(value), name=local(name))
        graph.initializer.append(tensor)
        return tensor.name

    zero, one = const("zero", 0.0), const("one", 1.0)
    flt_max = const("flt_max", float(np.finfo(np.float32).max))

    col_max = add("ReduceMax", [_MASK], local("col_max"), axes=[2], keepdims=0)  # [1, Q, W]
    row_max = add("ReduceMax", [_MASK], local("row_max"), axes=[3], keepdims=0)  # [1, Q, H]
    col_any = add("Greater", [col_max, zero], local("col_any"))
    row_any = add("Greater", [row_max, zero], local("row_any"))
    xs = add("Cast", [_XS], local("xs"), to=TensorProto.FLOAT)
    ys = add("Cast", [_YS], local("ys"), to=TensorProto.FLOAT)

    for any_pixel, grid, axis_name, out_min, out_max1 in (
        (col_any, xs, "x", _OUT_XMIN, _OUT_XMAX1),
        (row_any, ys, "y", _OUT_YMIN, _OUT_YMAX1),
    ):
        low = add("Where", [any_pixel, grid, flt_max], local(f"{axis_name}_or_flt_max"))
        add("ReduceMin", [low], out_min, axes=[-1], keepdims=0)
        high = add("Where", [any_pixel, grid, zero], local(f"{axis_name}_or_zero"))
        last = add("ReduceMax", [high], local(f"{axis_name}_last"), axes=[-1], keepdims=0)
        add("Add", [last, one], out_max1)

    query_max = add("ReduceMax", [col_max], local("query_max"), axes=[-1], keepdims=0)
    add("Greater", [query_max, zero], _OUT_NONEMPTY)

    replaced = {name for _, name in _REPLACED.values()}
    nodes = [n for n in graph.node if n.name not in replaced] + new_nodes
    known = {t.name for t in graph.initializer} | {t.name for t in graph.input}
    final = _live(_topological(nodes, known), {t.name for t in graph.output})

    del graph.node[:]
    graph.node.extend(final)
    used = {t for node in final for t in node.input}
    kept = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.value_info[:]  # shapes of removed tensors; onnxruntime infers its own
    onnx.checker.check_model(model, full_check=False)
    return model


def rewrite_bundle(src: Path, dst: Path) -> Path:
    """Write a rewritten copy of the bundle at ``src`` under ``dst`` and return its model path."""
    manifest = json.loads((src / "onnx" / "bibr_onnx.json").read_text(encoding="utf-8"))
    src_model = src / "onnx" / manifest.get("model_file", "model.onnx")
    if not src_model.is_file():
        raise FileNotFoundError(src_model)
    model = rewrite_mask_boxes(onnx.load(src_model))

    out = dst / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    dst_model = out / src_model.name
    onnx.save(model, dst_model)
    manifest["graph_rewrite"] = {
        "id": _REWRITE_ID,
        "source_sha256": hashlib.sha256(src_model.read_bytes()).hexdigest(),
    }
    (out / "bibr_onnx.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dst_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("src", type=Path, help="bundle directory holding onnx/model.onnx")
    parser.add_argument("dst", type=Path, help="directory to write the rewritten bundle to")
    args = parser.parse_args()
    print(rewrite_bundle(args.src, args.dst))


if __name__ == "__main__":
    main()
