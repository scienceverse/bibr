"""Generate the committed torch-free ONNX test bundles.

Writes deterministic tiny bundles under ``tests/fixtures/onnx/`` — one per
default runtime (section classifier, paper classifier, NER reference parser,
V3 layout detector) — built directly with the ``onnx`` helper API, so no
torch/transformers install is needed to regenerate them::

    PYTHONPATH=. python scripts/generate_onnx_test_bundles.py

The graphs are smoke-test stand-ins, not trained models: each output is a
small fixed affine function of the inputs (masked token-id mean for the text
models, image mean for layout), so inference is deterministic and
input-dependent. The torch-vs-ONNX *parity* tests keep exporting real tiny
models in-test (``tests/onnx_fixtures.py``); the committed bundles exist so
the core-only CI jobs — which have no torch — still execute the real
``Onnx*`` inference code (``tests/test_onnx_runtime_core.py``,
``tests/test_no_torch_on_http_path.py``). Manifests mirror the exporter
helpers exactly (same keys, ``schema_version`` 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIXTURE_ROOT = Path("tests/fixtures/onnx")

SECTION_LABELS = ["introduction", "method", "results", "discussion", "unknown"]
L1 = ["Natural Sciences", "Social Sciences"]
L2 = ["Physical Sciences", "Psychology and Cognitive Sciences", "Economics and Business"]
PAPER_TYPES = ["empirical", "review", "commentary"]
LAYOUT_QUERIES = 8
LAYOUT_CLASSES = 5
LAYOUT_SIZE = 64


def _np_to_initializer(name: str, array: np.ndarray):
    from onnx import TensorProto, helper

    return helper.make_tensor(
        name, TensorProto.FLOAT, list(array.shape), array.astype(np.float32).ravel().tolist()
    )


def _masked_mean_nodes():
    """Nodes computing the mask-weighted token-id mean, shape [B, 1]."""
    from onnx import TensorProto, helper

    # ReduceSum-13 input form (what torch.onnx.export emits): keeps the
    # committed bundles on opset 17 like the production bundles.
    axes = helper.make_tensor("reduce_axes", TensorProto.INT64, [1], [1])
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=1),  # FLOAT
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=1),
        helper.make_node("Mul", ["ids_f", "mask_f"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "reduce_axes"], ["num"], keepdims=1),
        helper.make_node("ReduceSum", ["mask_f", "reduce_axes"], ["den"], keepdims=1),
        helper.make_node("Clip", ["den", "clip_min"], ["den_c"]),
        helper.make_node("Div", ["num", "den_c"], ["mean"]),
    ]
    initializers = [
        axes,
        _np_to_initializer("clip_min", np.array(1.0, dtype=np.float32)),
    ]
    return nodes, initializers


def _text_inputs():
    from onnx import TensorProto, helper

    return [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence"]),
    ]


def _finish_graph(nodes, initializers, outputs, name: str):
    from onnx import helper

    graph = helper.make_graph(nodes, name, _text_inputs(), outputs, initializer=initializers)
    opset = helper.make_operatorsetid("", 17)
    # IR 10: loadable by every onnxruntime bibr supports (1.19+).
    return helper.make_model(graph, opset_imports=[opset], ir_version=10)


def _affine_head(mean_name: str, out_name: str, weight: np.ndarray, bias: np.ndarray):
    from onnx import helper

    w_name, b_name = f"{out_name}_W", f"{out_name}_b"
    node = helper.make_node("Gemm", [mean_name, w_name, b_name], [out_name])
    return node, [
        _np_to_initializer(w_name, weight),
        _np_to_initializer(b_name, bias),
    ]


def build_section_model():
    from onnx import TensorProto, helper

    nodes, initializers = _masked_mean_nodes()
    w_type = np.array([[0.3, -0.2, 0.5, -0.4, 0.1]], dtype=np.float32)
    b_type = np.array([0.1, -0.1, 0.2, -0.2, 0.0], dtype=np.float32)
    w_top = np.array([[0.7]], dtype=np.float32)
    b_top = np.array([-0.3], dtype=np.float32)
    for out, w, b in (("type_logits", w_type, b_type), ("top_level_logits", w_top, b_top)):
        node, inits = _affine_head("mean", out, w, b)
        nodes.append(node)
        initializers.extend(inits)
    outputs = [
        helper.make_tensor_value_info("type_logits", TensorProto.FLOAT, ["batch", 5]),
        helper.make_tensor_value_info("top_level_logits", TensorProto.FLOAT, ["batch", 1]),
    ]
    return _finish_graph(nodes, initializers, outputs, "tiny-section")


def build_paper_model():
    from onnx import TensorProto, helper

    nodes, initializers = _masked_mean_nodes()
    heads = {
        "l1_logits": (
            np.array([[0.4, -0.3]], dtype=np.float32),
            np.array([0.0, 0.1], dtype=np.float32),
        ),
        "l2_logits": (
            np.array([[0.2, -0.5, 0.6]], dtype=np.float32),
            np.array([0.1, 0.0, -0.1], dtype=np.float32),
        ),
        "paper_type_logits": (
            np.array([[-0.3, 0.5, 0.1]], dtype=np.float32),
            np.array([0.0, 0.0, 0.2], dtype=np.float32),
        ),
    }
    outputs = []
    for out, (w, b) in heads.items():
        node, inits = _affine_head("mean", out, w, b)
        nodes.append(node)
        initializers.extend(inits)
        outputs.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, ["batch", w.shape[1]]))
    return _finish_graph(nodes, initializers, outputs, "tiny-paper")


def build_ner_model(num_tags: int):
    from onnx import TensorProto, helper

    bias = (np.arange(num_tags, dtype=np.float32) - num_tags / 2) * 0.05
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=1),
        helper.make_node("Unsqueeze", ["ids_f", "unsq_axes"], ["ids_3d"]),
        helper.make_node("Tile", ["ids_3d", "tile_repeats"], ["tiled"]),
        helper.make_node("Add", ["tiled", "emit_bias"], ["emissions"]),
    ]
    initializers = [
        helper.make_tensor("unsq_axes", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tile_repeats", TensorProto.INT64, [3], [1, 1, num_tags]),
        _np_to_initializer("emit_bias", bias),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "emissions", TensorProto.FLOAT, ["batch", "sequence", num_tags]
        )
    ]
    graph = helper.make_graph(nodes, "tiny-ner", _text_inputs(), outputs, initializer=initializers)
    opset = helper.make_operatorsetid("", 17)
    return helper.make_model(graph, opset_imports=[opset], ir_version=10)


def build_layout_model():
    from onnx import TensorProto, helper

    q, c, size = LAYOUT_QUERIES, LAYOUT_CLASSES, LAYOUT_SIZE
    # Image mean via sum-then-divide: ReduceMean-18's input-form axes would
    # force the bundle onto opset 18; the sum form stays on opset 17.
    nodes = [
        helper.make_node("ReduceSum", ["pixel_values", "img_axes"], ["img_sum"], keepdims=1),
        helper.make_node("Div", ["img_sum", "img_count"], ["img_mean"]),
        helper.make_node("Flatten", ["img_mean"], ["flat"], axis=1),
    ]
    initializers: list = [
        helper.make_tensor("img_axes", TensorProto.INT64, [3], [1, 2, 3]),
        _np_to_initializer("img_count", np.array(3 * size * size, dtype=np.float32)),
    ]
    heads = {"logits": q * c, "pred_boxes": q * 4, "order_logits": q * q}
    for i, (final, width) in enumerate(heads.items()):
        if final == "pred_boxes":
            # Boxes must be probabilities in (0, 1): sigmoid the raw head.
            stem = "boxes_raw"
        else:
            stem = f"{final}_flat"
        w = (np.arange(width, dtype=np.float32).reshape(1, width) + 1 + i) * 0.01
        b = (np.arange(width, dtype=np.float32) - width / 2) * 0.02
        node, inits = _affine_head("flat", stem, w, b)
        nodes.append(node)
        initializers.extend(inits)
    nodes.append(helper.make_node("Sigmoid", ["boxes_raw"], ["boxes_sig"]))
    nodes.extend(
        [
            helper.make_node("Reshape", ["logits_flat", "shape_qc"], ["logits"]),
            helper.make_node("Reshape", ["boxes_sig", "shape_q4"], ["pred_boxes"]),
            helper.make_node("Reshape", ["order_logits_flat", "shape_qq"], ["order_logits"]),
        ]
    )
    initializers.extend(
        [
            helper.make_tensor("shape_qc", TensorProto.INT64, [3], [-1, q, c]),
            helper.make_tensor("shape_q4", TensorProto.INT64, [3], [-1, q, 4]),
            helper.make_tensor("shape_qq", TensorProto.INT64, [3], [-1, q, q]),
        ]
    )
    graph_inputs = [
        helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3, size, size])
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", q, c]),
        helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, ["batch", q, 4]),
        helper.make_tensor_value_info("order_logits", TensorProto.FLOAT, ["batch", q, q]),
    ]
    graph = helper.make_graph(nodes, "tiny-layout", graph_inputs, outputs, initializer=initializers)
    opset = helper.make_operatorsetid("", 17)
    return helper.make_model(graph, opset_imports=[opset], ir_version=10)


def _write_bundle(bundle: Path, model, manifest: dict, tokenizer=None) -> None:
    from onnx import checker

    checker.check_model(model)
    # Exporter layout (tests/onnx_fixtures.py): the loadable bundle is the
    # ``onnx/`` child; loaders accept the parent directory.
    bundle = bundle / "onnx"
    bundle.mkdir(parents=True, exist_ok=True)
    with open(bundle / "model.onnx", "wb") as fh:
        fh.write(model.SerializeToString())
    if tokenizer is not None:
        tokenizer.save(str(bundle / "tokenizer.json"))
    manifest = {"schema_version": 1, "model_file": "model.onnx", **manifest}
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE_ROOT)
    args = parser.parse_args()

    from bibr.ner.tags import BIO_TAGS
    from tests.onnx_fixtures import tiny_tokenizer

    root: Path = args.out
    _write_bundle(
        root / "section",
        build_section_model(),
        {
            "model": "section_classifier",
            "tokenizer_file": "tokenizer.json",
            "pad_token_id": 0,
            "pad_token": "[PAD]",
            "max_length": 32,
            "label_classes": SECTION_LABELS,
            "template_version": 3,
        },
        tokenizer=tiny_tokenizer(),
    )
    _write_bundle(
        root / "paper",
        build_paper_model(),
        {
            "model": "paper_classifier",
            "tokenizer_file": "tokenizer.json",
            "pad_token_id": 0,
            "pad_token": "[PAD]",
            "max_length": 32,
            "l1_classes": L1,
            "l2_classes": L2,
            "paper_type_classes": PAPER_TYPES,
            "paper_type_temperature": 1.5,
        },
        tokenizer=tiny_tokenizer(),
    )
    _write_bundle(
        root / "ner",
        build_ner_model(len(BIO_TAGS)),
        {
            "model": "ner_parser",
            "tokenizer_file": "tokenizer.json",
            "add_special_tokens": False,
            "pad_token_id": 0,
            "pad_token": "[PAD]",
            "max_length": 16,
            "bio_tags": list(BIO_TAGS),
            "crf": {
                "start_transitions": [0.0] * len(BIO_TAGS),
                "end_transitions": [0.0] * len(BIO_TAGS),
                "transitions": [[0.0] * len(BIO_TAGS) for _ in BIO_TAGS],
            },
        },
        tokenizer=tiny_tokenizer(),
    )
    _write_bundle(
        root / "layout",
        build_layout_model(),
        {
            "model": "layout",
            "preprocessing": {
                "size": {"height": LAYOUT_SIZE, "width": LAYOUT_SIZE},
                "rescale_factor": 1 / 255,
                "image_mean": [0, 0, 0],
                "image_std": [1, 1, 1],
            },
            "num_queries": LAYOUT_QUERIES,
            "num_labels": LAYOUT_CLASSES,
        },
    )
    for bundle in sorted(root.iterdir()):
        size = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
        print(f"{bundle.name}: {size} bytes")


if __name__ == "__main__":
    main()
