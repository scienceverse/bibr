"""Generate the committed torch-free ONNX test bundles.

Writes deterministic tiny bundles under ``tests/fixtures/onnx/`` — one per
default runtime (section classifier, paper classifier, NER reference parser,
V3 layout detector) — built directly with the ``onnx`` helper API, so no
torch/transformers install is needed to regenerate them::

    PYTHONPATH=. python scripts/generate_onnx_test_bundles.py

The graphs are smoke-test stand-ins, not trained models, but every output is
a genuine function of its inputs so the golden tests in
``tests/test_onnx_runtime_core.py`` catch preprocessing and decode
regressions:

- section/paper heads are parabolas (or a linear threshold) over the
  mask-weighted token-id mean and the sequence length, so zeroing,
  reversing or re-tokenizing the inputs moves the argmax;
- the NER emissions are a per-token-id lookup table with a structured
  nonzero CRF in the manifest, so the Viterbi path depends on both the
  ids and the transitions;
- the layout head reads the image mean, so rescaling or zeroing the
  pixels moves every box and score.

The torch-vs-ONNX *parity* tests keep exporting real tiny models in-test
(``tests/onnx_fixtures.py``); the committed bundles exist so the core-only
CI jobs — which have no torch — still execute the real ``Onnx*`` inference
code (``tests/test_onnx_runtime_core.py``,
``tests/test_no_torch_on_http_path.py``). Manifests mirror the exporter
helpers exactly (same keys, ``schema_version`` 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIXTURE_ROOT = Path("tests/fixtures/onnx")

# ``label_classes`` must be valid ``CanonicalSection`` values: the ONNX
# section loader maps each entry through that enum at predict time.
SECTION_LABELS = ["intro", "method", "results", "discussion", "unknown"]
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
    """Nodes computing the mask-weighted token-id mean and length, [B, 1] each.

    ``mean`` is the mask-weighted token-id mean; ``den_c`` doubles as the
    sequence-length feature (padding clipped to a minimum of one).
    """
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


def _parabola_head(
    nodes,
    initializers,
    out_name: str,
    class_lens,
    class_means,
    neg_len: float = -0.04,
    neg_mean: float = -2.0,
):
    """Append ``logits = -((len-CL)^2*|neg_len| + (mean-CM)^2*|neg_mean|)``.

    Class ``i`` peaks at ``(class_means[i], class_lens[i])``; every test
    input sits in a different basin, so re-tokenizing, zeroing or reversing
    the inputs moves at least one argmax. Broadcasts ``[B, 1]`` against the
    ``[C]`` center vectors — no per-class nodes.
    """
    from onnx import helper

    nodes.append(helper.make_node("Sub", ["den_c", f"{out_name}_CL"], [f"{out_name}_dl"]))
    nodes.append(helper.make_node("Mul", [f"{out_name}_dl", f"{out_name}_dl"], [f"{out_name}_ql"]))
    nodes.append(helper.make_node("Mul", [f"{out_name}_ql", f"{out_name}_NL"], [f"{out_name}_sl"]))
    nodes.append(helper.make_node("Sub", ["mean", f"{out_name}_CM"], [f"{out_name}_dm"]))
    nodes.append(helper.make_node("Mul", [f"{out_name}_dm", f"{out_name}_dm"], [f"{out_name}_qm"]))
    nodes.append(helper.make_node("Mul", [f"{out_name}_qm", f"{out_name}_NM"], [f"{out_name}_sm"]))
    nodes.append(helper.make_node("Add", [f"{out_name}_sl", f"{out_name}_sm"], [out_name]))
    initializers.extend(
        [
            _np_to_initializer(f"{out_name}_CL", np.asarray(class_lens, dtype=np.float32)),
            _np_to_initializer(f"{out_name}_CM", np.asarray(class_means, dtype=np.float32)),
            _np_to_initializer(f"{out_name}_NL", np.array(neg_len, dtype=np.float32)),
            _np_to_initializer(f"{out_name}_NM", np.array(neg_mean, dtype=np.float32)),
        ]
    )


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
    # Type head basins, one per fixed test context (lengths 32/23/19/19,
    # means ~23.1/21.9/27.6/21.6): Methods, Results, Ethics, Introduction.
    # Mean alone cannot separate Results from Introduction (21.9 vs 21.6);
    # their lengths (23 vs 19) can. ``discussion`` peaks far away and is
    # never predicted, like a rare class in a trained head.
    _parabola_head(
        nodes,
        initializers,
        "type_logits",
        [19.0, 32.0, 23.0, 50.0, 19.0],
        [21.58, 23.09, 21.87, 100.0, 27.63],
        neg_len=-0.125,
        neg_mean=-2.0,
    )
    # Top-level head reads the mean directly: sigmoid(mean - 22.5) is true
    # for the Methods and Ethics contexts, false for the other two.
    nodes.append(helper.make_node("Mul", ["mean", "mean"], ["_top_msq"]))
    nodes.append(helper.make_node("Concat", ["_top_msq", "mean"], ["_top_feat"], axis=1))
    node, inits = _affine_head(
        "_top_feat",
        "top_level_logits",
        np.array([[0.0], [1.0]], dtype=np.float32),
        np.array([-22.5], dtype=np.float32),
    )
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
    # L1 head: a linear length threshold (with a small mean tilt) isolating
    # the short "Sleep science" item from the other two.
    nodes.append(helper.make_node("Concat", ["den_c", "mean"], ["feat"], axis=1))
    node, inits = _affine_head(
        "feat",
        "l1_logits",
        np.array([[1.0, -1.0], [0.1, -0.1]], dtype=np.float32),
        np.array([-8.14, 8.14], dtype=np.float32),
    )
    nodes.append(node)
    initializers.extend(inits)
    # L2/type heads: 2-D parabolas over (length, mean), one basin per fixed
    # test item. The mean term keeps zeroed or reversed inputs (same mask,
    # hence same length) from scoring like the real ones.
    _parabola_head(nodes, initializers, "l2_logits", [4.0, 9.0, 14.0], [16.5, 16.78, 15.86])
    _parabola_head(
        nodes,
        initializers,
        "paper_type_logits",
        [4.0, 14.0, 9.0],
        [16.5, 15.86, 16.78],
    )
    outputs = [
        helper.make_tensor_value_info("l1_logits", TensorProto.FLOAT, ["batch", 2]),
        helper.make_tensor_value_info("l2_logits", TensorProto.FLOAT, ["batch", 3]),
        helper.make_tensor_value_info("paper_type_logits", TensorProto.FLOAT, ["batch", 3]),
    ]
    return _finish_graph(nodes, initializers, outputs, "tiny-paper")


def build_ner_model(num_tags: int):
    from onnx import TensorProto, helper

    # Emissions are a per-token-id lookup: token id ``r`` scores
    # ``table[r]`` on every tag, so re-tokenizing or zeroing the ids moves
    # the Viterbi path. Fixed seed keeps regeneration byte-identical.
    table = (np.random.RandomState(7).randn(50, num_tags) * 0.5).astype(np.float32)
    nodes = [helper.make_node("Gather", ["tag_table", "input_ids"], ["emissions"])]
    initializers = [_np_to_initializer("tag_table", table)]
    outputs = [
        helper.make_tensor_value_info(
            "emissions", TensorProto.FLOAT, ["batch", "sequence", num_tags]
        )
    ]
    graph = helper.make_graph(nodes, "tiny-ner", _text_inputs(), outputs, initializer=initializers)
    opset = helper.make_operatorsetid("", 17)
    return helper.make_model(graph, opset_imports=[opset], ir_version=10)


def build_crf():
    """Structured nonzero CRF: B-X leads into I-X, invalid jumps cost."""
    from bibr.ner.tags import BIO_TAGS, FIELD_TYPES

    own = {f: (BIO_TAGS.index(f"B-{f}"), BIO_TAGS.index(f"I-{f}")) for f in FIELD_TYPES}
    n = len(BIO_TAGS)
    oi = BIO_TAGS.index("O")
    transitions = np.zeros((n, n), dtype=np.float32)
    start = np.zeros(n, dtype=np.float32)
    end = np.zeros(n, dtype=np.float32)
    start[oi] = 0.2
    end[oi] = 0.1
    for field, (bi, ii) in own.items():
        start[bi] = 0.1
        start[ii] = -0.6
        end[ii] = -0.1
        transitions[oi, bi] = 0.15
        transitions[bi, ii] = 0.5
        transitions[ii, ii] = 0.25
        transitions[ii, oi] = 0.1
        transitions[bi, oi] = -0.1
        for other, (bj, ij) in own.items():
            if other != field:
                transitions[bi, ij] = -0.6
                transitions[ii, bj] = -0.4
                transitions[ii, ij] = -0.2
    return {
        "start_transitions": start.tolist(),
        "end_transitions": end.tolist(),
        "transitions": transitions.tolist(),
    }


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
            "crf": build_crf(),
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
