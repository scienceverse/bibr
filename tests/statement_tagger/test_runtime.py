"""The statement tagger runtime: blocks in, typed statements out, from a bundle on disk."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")

from bibr.statement_tagger import runtime  # noqa: E402
from bibr.statement_tagger.runtime import (  # noqa: E402
    StatementTagger,
    StatementTaggerError,
    TaggedStatement,
    join_runs,
    normalize_block_text,
)
from tests.statement_tagger.bundle import LABELS, write_bundle  # noqa: E402


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return write_bundle(tmp_path / "tagger")


def test_blocks_in_typed_statements_out(bundle):
    tagger = StatementTagger(bundle, threads=1)
    blocks = [
        "Methods",
        "We recruited a sample.",
        "This work was funded by a grant.",
        "We thank the reviewers.",
        "Results",
        "Competing interests: none.",
    ]
    got = tagger.tag(blocks)
    # The toy graph also reads each block's context (the start of the block
    # before), so a keyword carries one block on: the runs join per type.
    assert [(s.type, s.start, s.end) for s in got] == [
        ("funding", 2, 3),
        ("acknowledgment", 3, 4),
        ("coi", 5, 5),
    ]
    assert all(0.9 < s.score <= 1.0 for s in got)


def test_the_bundle_directory_or_its_onnx_subdirectory_loads(bundle):
    assert StatementTagger(bundle / "onnx").labels == tuple(LABELS)
    assert [p.name for p in StatementTagger(bundle).files] == [
        "bibr_onnx.json",
        "model.onnx",
        "tokenizer.json",
    ]


def test_each_block_is_read_with_the_start_of_the_one_before(bundle):
    tagger = StatementTagger(bundle)
    long = "x" * 300
    pairs = tagger.pairs([long, "  second­block\n text ", "third"])
    assert pairs[0] == ("", long)
    assert pairs[1] == ("x" * 200, "secondblock text")
    assert pairs[2] == ("secondblock text", "third")


def test_probabilities_keep_block_order_across_batches(bundle):
    blocks = ["the", "funded by a grant", "we", "thank", "results " * 20, "ethics"]
    one = StatementTagger(bundle, batch_size=1).probabilities(blocks)
    many = StatementTagger(bundle, batch_size=4).probabilities(blocks)
    np.testing.assert_allclose(one, many, atol=1e-6)
    assert one.shape == (len(blocks), len(LABELS))
    assert StatementTagger(bundle).probabilities([]).shape == (0, len(LABELS))


def test_normalisation_matches_training():
    assert normalize_block_text("ﬁnd​  the\n\tdata­ set﻿ ") == "find the data set"
    assert normalize_block_text(None) == ""


def test_runs_join_per_type_and_keep_the_best_score():
    probs = np.zeros((5, len(LABELS)), np.float32)
    probs[1:3, LABELS.index("funding")] = [0.6, 0.9]
    types = [set(), {"funding"}, {"funding", "acknowledgment"}, set(), {"funding"}]
    got = join_runs(types, probs, LABELS)
    assert got == [
        TaggedStatement("funding", 1, 2, pytest.approx(0.9)),
        TaggedStatement("acknowledgment", 2, 2, 0.0),
        TaggedStatement("funding", 4, 4, 0.0),
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model": "section_classifier"}, "model 'section_classifier'"),
        ({"schema_version": 2}, "schema_version 2"),
        ({"thresholds": {"coi": 0.5}}, "one number per label"),
        ({"thresholds": dict.fromkeys(LABELS, 1.5)}, "want a number in (0, 1]"),
        ({"input": {"rule": "single", "context_chars": 200, "max_length": 64}}, "input rule"),
        (
            {"input": {"rule": "pair:previous_block", "context_chars": 0, "max_length": 64}},
            "context_chars",
        ),
        ({"labels": ["coi", "coi"]}, "repeat"),
        ({"model_file": "../model.onnx"}, "leaves the bundle"),
    ],
)
def test_a_bundle_this_code_cannot_run_is_refused(tmp_path, changes, message):
    with pytest.raises(StatementTaggerError, match=message.replace("(", r"\(").replace(")", r"\)")):
        StatementTagger(write_bundle(tmp_path / "bad", **changes))


def test_a_missing_bundle_and_a_missing_file_are_refused(tmp_path, bundle):
    with pytest.raises(StatementTaggerError, match="not found"):
        StatementTagger(tmp_path / "nothing")
    (bundle / "onnx" / "tokenizer.json").unlink()
    with pytest.raises(StatementTaggerError, match="lacks"):
        StatementTagger(bundle)


def test_the_graph_is_run_once_at_load(tmp_path):
    # A graph with fewer outputs than the manifest has labels fails at load.
    with pytest.raises(StatementTaggerError, match="shape"):
        StatementTagger(write_bundle(tmp_path / "short", n_out=3))


def test_a_manifest_that_is_not_json_is_refused(bundle):
    (bundle / "onnx" / "bibr_onnx.json").write_text("{", encoding="utf-8")
    with pytest.raises(StatementTaggerError, match="unreadable"):
        StatementTagger(bundle)
    (bundle / "onnx" / "bibr_onnx.json").write_text(json.dumps([1]), encoding="utf-8")
    with pytest.raises(StatementTaggerError, match="JSON object"):
        StatementTagger(bundle)


def test_the_runtime_needs_nothing_from_bibr():
    """Another tool can call the same function on its own blocks."""
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported <= {
        "__future__",
        "json",
        "re",
        "threading",
        "unicodedata",
        "collections",
        "dataclasses",
        "pathlib",
        "typing",
        "numpy",
        "onnxruntime",
        "tokenizers",
    }
