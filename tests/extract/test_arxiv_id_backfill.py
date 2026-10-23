"""A printed arXiv identifier reaches ``bib[].arxiv`` whatever the tagger tagged."""

from __future__ import annotations

import pytest

from bibr.extract.ref_extractor import _arxiv_id_from_segment, _finalize_reference_fields


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        (
            "Berk Atil et al. 2025. Non-determinism of “deterministic” LLM settings. "
            "Preprint, arXiv:2408.04667.",
            "2408.04667",
        ),
        (
            "Devlin, J. (2018). BERT: Pre-training of deep bidirectional transformers. "
            "arXiv preprint arXiv:1810.04805.",
            "1810.04805",
        ),
        ("Smith, A. (2023). A title. arXiv preprint arxiv:2301.07597, 2023.", "2301.07597"),
        (
            "Hinton, G. Distilling the knowledge, 2015. URL https://arxiv.org/abs/1503.02531.",
            "1503.02531",
        ),
        ("Doe, J. A survey, 2024. URL https: //arxiv.org/pdf/2407.06204v2.pdf", "2407.06204v2"),
        ("Roe, R. (2022). A model. https://doi.org/10.48550/arXiv.2211.06030", "2211.06030"),
        ("Witten, E. String theory. arXiv:hep-th/9503124 (1995).", "hep-th/9503124"),
        ("Lee, K. (2021). doi:10.1016/j.ipm.2022.103248, arXiv:2211.06030 [cs]", "2211.06030"),
        # No identifier, or two references run together: nothing to fill.
        ("Smith, A. (2020). Something about arXiv preprints. Journal, 3, 1-2.", None),
        ("A. arXiv:2101.00001. B. arXiv:2101.00002.", None),
        (None, None),
    ],
)
def test_arxiv_id_from_segment(segment, expected):
    assert _arxiv_id_from_segment(segment) == expected


def test_finalize_fills_an_empty_arxiv_and_keeps_a_parsed_one():
    segment = "Chen, M. (2021). Evaluating large language models trained on code. Preprint, arXiv:2107.03374."

    filled = _finalize_reference_fields({"title": "Evaluating", "arxiv": None}, segment)
    kept = _finalize_reference_fields({"title": "Evaluating", "arxiv": "arXiv:2107.03374"}, segment)

    assert filled["arxiv"] == "2107.03374"
    assert kept["arxiv"] == "arXiv:2107.03374"
