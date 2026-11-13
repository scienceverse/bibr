"""A reference the tokenizer reduces to no tokens, batched with real references."""

from pathlib import Path

import numpy as np

from bibr.ner.crf_numpy import viterbi_decode

BUNDLE = Path(__file__).resolve().parents[1] / "fixtures" / "onnx" / "ner" / "onnx"


def test_all_off_row_decodes_empty_and_leaves_other_rows_alone():
    rng = np.random.default_rng(3)
    emissions = rng.normal(size=(2, 4, 3)).astype(np.float32)
    crf = {
        "start_transitions": rng.normal(size=3),
        "end_transitions": rng.normal(size=3),
        "transitions": rng.normal(size=(3, 3)),
    }
    mask = np.array([[True, True, True, False], [False, False, False, False]])
    alone = viterbi_decode(emissions[:1], mask[:1], **crf)
    assert viterbi_decode(emissions, mask, **crf) == [alone[0], []]


def test_zero_width_reference_in_a_batch_does_not_fail_the_batch():
    from bibr.ner.parser_onnx import OnnxRefParser

    # The fixture tokenizer has a BERT normalizer, which drops U+200B and U+FEFF.
    parser = OnnxRefParser(BUNDLE, device="cpu")
    ref = "smith j 2020 memory and consciousness"
    alone = parser.parse_batch([ref])[0]

    assert parser.parse_batch([ref, "​", "﻿​"]) == [alone, {}, {}]
