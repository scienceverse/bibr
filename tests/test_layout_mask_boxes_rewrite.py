"""The layout mask-to-box rewrite returns exactly what the exported subgraph returns."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rewrite_onnx_layout_mask_boxes.py"
_spec = importlib.util.spec_from_file_location("rewrite_onnx_layout_mask_boxes", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

P = "/inner/model/"
_OUTPUTS = (
    P + "ReduceMin_output_0",
    P + "ReduceMin_1_output_0",
    P + "Add_40_output_0",
    P + "Add_41_output_0",
    P + "Greater_3_output_0",
    P + "Shape_112_output_0",
    P + "Shape_113_output_0",
)


def _exported_form(queries: int, height: int, width: int) -> onnx.ModelProto:
    """A small graph with the exported subgraph's tensor names and arithmetic."""
    nodes = []

    def node(op, inputs, name, **attrs):
        nodes.append(helper.make_node(op, inputs, [P + name + "_output_0"], name=P + name, **attrs))
        return P + name + "_output_0"

    def const(name, value, dtype=np.float32):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=P + name)

    inits = [
        const("zero_f", 0.0),
        const("one_f", 1.0),
        const("flt_max", np.finfo(np.float32).max),
        const("zero_i", 0, np.int64),
        const("one_i", 1, np.int64),
        const("height", height, np.int64),
        const("width", width, np.int64),
        const("flat", [1, queries, -1], np.int64),
        const("column", [height, 1], np.int64),
        const("spatial", [2, 3], np.int64),
    ]
    mask = node("Identity", ["mask"], "Reshape_33")
    positive = node("Greater", [mask, P + "zero_f"], "Greater_2")
    as_bool = node("Cast", [positive], "Cast_15", to=TensorProto.BOOL)
    node("Shape", [as_bool], "Shape_112")
    node("Shape", [as_bool], "Shape_113")
    ys_int = node("Range", [P + "zero_i", P + "height", P + "one_i"], "Range_7")
    xs_int = node("Range", [P + "zero_i", P + "width", P + "one_i"], "Range_8")
    xs = node("Cast", [xs_int], "Cast_18", to=TensorProto.FLOAT)
    ys_flat = node("Cast", [ys_int], "Cast_19", to=TensorProto.FLOAT)
    ys = node("Reshape", [ys_flat, P + "column"], "Reshape_34")
    as_float = node("Cast", [as_bool], "Cast_20", to=TensorProto.FLOAT)

    x_values = node("Mul", [xs, as_float], "Mul_59")
    x_flat = node("Reshape", [x_values, P + "flat"], "Reshape_38")
    x_max = node("ReduceMax", [x_flat], "ReduceMax_1", axes=[-1], keepdims=0)
    node("Add", [x_max, P + "one_f"], "Add_40")
    x_kept = node("Where", [as_bool, x_values, P + "flt_max"], "Where_48")
    x_kept_flat = node("Reshape", [x_kept, P + "flat"], "Reshape_39")
    node("ReduceMin", [x_kept_flat], "ReduceMin", axes=[-1], keepdims=0)

    y_values = node("Mul", [ys, as_float], "Mul_60")
    y_flat = node("Reshape", [y_values, P + "flat"], "Reshape_40")
    y_max = node("ReduceMax", [y_flat], "ReduceMax_2", axes=[-1], keepdims=0)
    node("Add", [y_max, P + "one_f"], "Add_41")
    y_kept = node("Where", [as_bool, y_values, P + "flt_max"], "Where_49")
    y_kept_flat = node("Reshape", [y_kept, P + "flat"], "Reshape_41")
    node("ReduceMin", [y_kept_flat], "ReduceMin_1", axes=[-1], keepdims=0)

    counts = node("Cast", [as_bool], "Cast_22", to=TensorProto.INT64)
    total = node("ReduceSum", [counts, P + "spatial"], "ReduceSum_1", keepdims=0)
    node("Greater", [total, P + "zero_i"], "Greater_3")

    per_query = [1, queries]
    outputs = [
        helper.make_tensor_value_info(_OUTPUTS[0], TensorProto.FLOAT, per_query),
        helper.make_tensor_value_info(_OUTPUTS[1], TensorProto.FLOAT, per_query),
        helper.make_tensor_value_info(_OUTPUTS[2], TensorProto.FLOAT, per_query),
        helper.make_tensor_value_info(_OUTPUTS[3], TensorProto.FLOAT, per_query),
        helper.make_tensor_value_info(_OUTPUTS[4], TensorProto.BOOL, per_query),
        helper.make_tensor_value_info(_OUTPUTS[5], TensorProto.INT64, [4]),
        helper.make_tensor_value_info(_OUTPUTS[6], TensorProto.INT64, [4]),
    ]
    mask_input = helper.make_tensor_value_info(
        "mask", TensorProto.FLOAT, [1, queries, height, width]
    )
    graph = helper.make_graph(nodes, "layout_mask_boxes", [mask_input], outputs, inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _run(model: onnx.ModelProto, mask: np.ndarray) -> list[np.ndarray]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(list(_OUTPUTS), {"mask": mask})


def test_rewrite_returns_the_same_bytes_for_every_kind_of_mask():
    queries, height, width = 7, 9, 13
    rng = np.random.default_rng(3)
    mask = rng.normal(size=(1, queries, height, width)).astype(np.float32)
    mask[0, 0] = -1.0  # empty: FLT_MAX and 1.0, not a box
    mask[0, 1] = -1.0
    mask[0, 1, 4, 6] = 0.5  # a single pixel
    mask[0, 2] = 1.0  # every pixel
    mask[0, 3] = 0.0  # exactly zero is not positive
    mask[0, 4] = -1.0
    mask[0, 4, 0, 0] = 2.0  # a pixel on x = 0 and y = 0, where x * mask is 0
    mask[0, 5] = -1.0
    mask[0, 5, height - 1, width - 1] = 2.0

    before = _run(_exported_form(queries, height, width), mask)
    after = _run(mod.rewrite_mask_boxes(_exported_form(queries, height, width)), mask)

    for name, old, new in zip(_OUTPUTS, before, after, strict=True):
        assert old.dtype == new.dtype, name
        assert old.tobytes() == new.tobytes(), name
    xmin, ymin, xmax1, ymax1, nonempty = before[:5]
    assert xmin[0, 0] == np.finfo(np.float32).max and xmax1[0, 0] == 1.0 and not nonempty[0, 0]
    assert (xmin[0, 1], ymin[0, 1], xmax1[0, 1], ymax1[0, 1]) == (6.0, 4.0, 7.0, 5.0)


def test_rewrite_removes_the_full_size_operators():
    model = mod.rewrite_mask_boxes(_exported_form(4, 8, 8))
    names = {node.name for node in model.graph.node}
    for gone in ("Where_48", "Where_49", "Mul_59", "Mul_60", "Cast_22", "ReduceSum_1"):
        assert P + gone not in names


def test_rewrite_refuses_a_graph_it_does_not_recognise():
    model = _exported_form(4, 8, 8)
    next(n for n in model.graph.node if n.name == P + "Add_40").name = P + "Add_99"
    with pytest.raises(ValueError, match="not the expected layout graph"):
        mod.rewrite_mask_boxes(model)


def test_rewrite_bundle_records_its_source(tmp_path):
    src = tmp_path / "src" / "onnx"
    src.mkdir(parents=True)
    onnx.save(_exported_form(4, 8, 8), src / "model.onnx")
    (src / "bibr_onnx.json").write_text(json.dumps({"model_file": "model.onnx"}), encoding="utf-8")

    written = mod.rewrite_bundle(tmp_path / "src", tmp_path / "dst")

    assert written == tmp_path / "dst" / "onnx" / "model.onnx"
    onnx.checker.check_model(onnx.load(written))
    manifest = json.loads((tmp_path / "dst" / "onnx" / "bibr_onnx.json").read_text("utf-8"))
    assert manifest["graph_rewrite"]["id"] == "layout-mask-boxes-v1"
    assert len(manifest["graph_rewrite"]["source_sha256"]) == 64
