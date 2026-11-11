"""A tiny statement tagger bundle for the tests: a word-level tokenizer and a lookup graph.

The graph gives each token one logit per type (``E[token]``), keeps the highest over the
unmasked tokens of the (context, text) pair and applies a sigmoid, so a block is of a type when
it, or the start of the block before it, holds that type's keyword.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABELS = [
    "coi",
    "funding",
    "ethics",
    "data_availability",
    "author_contributions",
    "acknowledgment",
    "supplementary",
]
KEYWORDS = {
    "conflicts": "coi",
    "competing": "coi",
    "funded": "funding",
    "grant": "funding",
    "ethics": "ethics",
    "repository": "data_availability",
    "contributed": "author_contributions",
    "thank": "acknowledgment",
    "supplementary": "supplementary",
}
PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"
WORDS = [
    *KEYWORDS,
    "the",
    "we",
    "this",
    "work",
    "was",
    "by",
    "a",
    "results",
    "methods",
    "data",
]


def write_tokenizer(path: Path) -> list[str]:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    vocab = {PAD: 0, UNK: 1, CLS: 2, SEP: 3}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token=UNK))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single=f"{CLS} $A {SEP}",
        pair=f"{CLS} $A {SEP} $B:1 {SEP}:1",
        special_tokens=[(CLS, 2), (SEP, 3)],
    )
    tok.add_special_tokens([PAD, UNK, CLS, SEP])
    tok.save(str(path))
    return list(vocab)


def write_graph(path: Path, vocab: list[str], *, n_out: int | None = None) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n_out = len(LABELS) if n_out is None else n_out
    table = np.full((len(vocab), n_out), -6.0, np.float32)
    for word, label in KEYWORDS.items():
        if LABELS.index(label) < n_out:
            table[vocab.index(word), LABELS.index(label)] = 6.0
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["E", "input_ids"], ["emb"]),
            helper.make_node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["maskf", "axis"], ["mask3"]),
            helper.make_node("Sub", ["mask3", "one"], ["off"]),
            helper.make_node("Mul", ["off", "big"], ["penalty"]),
            helper.make_node("Add", ["emb", "penalty"], ["masked"]),
            helper.make_node("ReduceMax", ["masked", "axis1"], ["pooled"], keepdims=0),
            helper.make_node("Sigmoid", ["pooled"], ["probs"]),
        ],
        "tiny_statement_tagger",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "sequence"]
            ),
        ],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", n_out])],
        initializer=[
            numpy_helper.from_array(table, "E"),
            numpy_helper.from_array(np.asarray([-1], np.int64), "axis"),
            numpy_helper.from_array(np.asarray([1], np.int64), "axis1"),
            numpy_helper.from_array(np.asarray(1.0, np.float32), "one"),
            numpy_helper.from_array(np.asarray(100.0, np.float32), "big"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, str(path))


def manifest(**changes) -> dict:
    data = {
        "schema_version": 1,
        "model": "statement_tagger",
        "model_version": "test/1",
        "model_file": "model.onnx",
        "tokenizer_file": "tokenizer.json",
        "files": ["model.onnx", "tokenizer.json"],
        "labels": LABELS,
        "thresholds": dict.fromkeys(LABELS, 0.5),
        "input": {"rule": "pair:previous_block", "context_chars": 200, "max_length": 64},
        "pad_token_id": 0,
    }
    data.update(changes)
    return data


def write_bundle(root: Path, *, n_out: int | None = None, **changes) -> Path:
    """``root/onnx`` holding a working bundle (or one broken by ``changes``); returns root."""
    onnx_dir = root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    vocab = write_tokenizer(onnx_dir / "tokenizer.json")
    write_graph(onnx_dir / "model.onnx", vocab, n_out=n_out)
    (onnx_dir / "bibr_onnx.json").write_text(json.dumps(manifest(**changes)), encoding="utf-8")
    return root
