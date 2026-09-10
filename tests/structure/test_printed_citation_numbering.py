"""Numeric citations refer to printed labels, even after bibliography filtering."""

import pytest

from bibr.structure.citation_linker import detect_bib_xrefs_with_receipt
from tests.test_citation_linker import _ref, _sections_with_refs, _sent


def _bibliography(numbers, prefix="[{number}]"):
    references = []
    sentences = []
    for bib_id, number in enumerate(numbers, 1):
        references.append(_ref(bib_id, author=f"Author{bib_id}", text_id=10000 + bib_id))
        text = (
            f"{prefix.format(number=number)} Author{bib_id}. Study."
            if number is not None
            else "The authors declare no competing interests."
        )
        sentences.append(_sent(10000 + bib_id, text, section_id=2))
    return references, sentences


async def _link(body, numbers, prefix="[{number}]"):
    references, sources = _bibliography(numbers, prefix)
    sentences = [_sent(i, text) for i, text in enumerate(body, 1)] + sources
    return await detect_bib_xrefs_with_receipt(sentences, _sections_with_refs(), references)


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [([None, 1, 2, 3, 4, 5], [2, 4]), ([1, 3, 4, 5, 6], [1, 2]), ([1, 2, 3], [1, 3])],
)
async def test_citations_and_receipts_use_correct_internal_ids(numbers, expected):
    xrefs, receipt = await _link(["Prior work [1] and [3] agrees."], numbers)
    assert sorted(x.xref_id for x in xrefs) == expected
    assert sorted(i for c in receipt.candidates if c.accepted for i in c.bib_ids) == expected


@pytest.mark.parametrize(
    "prefix", ["[{number}]", "({number})", "{number}.", "{number} ", "^{{{number}}}"]
)
async def test_reference_label_formats(prefix):
    xrefs, _ = await _link(["Prior work [3] agrees."], [1, 3, 4, 5], prefix)
    assert [x.xref_id for x in xrefs] == [2]


async def test_dropped_label_does_not_resolve_by_position():
    xrefs, _ = await _link(["Prior work [2] agrees."], [1, 3, 4, 5, 6])
    assert xrefs == []


async def test_duplicate_label_does_not_resolve_by_position():
    xrefs, _ = await _link(["Prior work [1] and [3] agrees."], [1, 1, 2, 3, 4, 5])
    assert [x.xref_id for x in xrefs] == [4]


async def test_range_is_expanded_before_ids_are_mapped():
    xrefs, _ = await _link(["Prior work [3-5] agrees."], [1, 3, 4, 5, 6])
    assert [x.xref_id for x in xrefs] == [2, 3, 4]


async def test_unnumbered_bibliography_keeps_positional_behavior():
    xrefs, _ = await _link(["Prior work [1] and [3] agrees."], [None, None, None, None])
    assert [x.xref_id for x in xrefs] == [1, 3]


async def test_minority_of_numbered_rows_does_not_change_interpretation():
    xrefs, _ = await _link(["Prior work [2] agrees."], [None, 3, None, None])
    assert [x.xref_id for x in xrefs] == [2]


async def test_reference_without_a_text_row_keeps_identity_fallback():
    refs, sources = _bibliography([None, 1, 2, 3, 4, None])
    refs[-1] = refs[-1].model_copy(update={"text_id": None})
    xrefs, _ = await detect_bib_xrefs_with_receipt(
        [_sent(1, "Prior work [1] and [6] agrees."), *sources], _sections_with_refs(), refs
    )
    assert [x.xref_id for x in xrefs] == [2, 6]


async def test_author_year_matches_are_not_remapped():
    refs, sources = _bibliography([None, 1, 2, 3, 4])
    refs[1] = refs[1].model_copy(update={"authors": "Smith J"})
    xrefs, receipt = await detect_bib_xrefs_with_receipt(
        [_sent(1, "Prior work (Smith, 2020) agrees."), *sources], _sections_with_refs(), refs
    )
    assert [x.xref_id for x in xrefs] == [2]
    assert [c.bib_ids for c in receipt.candidates if c.accepted] == [(2,)]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            [
                "The first comparison (3) held.",
                "A replication (4) agreed.",
                "A final study (5) held.",
            ],
            {2, 3, 4},
        ),
        (
            [
                "Prior evidence.3,4 established this.",
                "Later reports4,5 confirmed it.",
                "A synthesis.5,6 agreed.",
            ],
            {2, 3, 4, 5},
        ),
    ],
)
async def test_fallback_styles_use_printed_numbers(body, expected):
    xrefs, _ = await _link(body, [1, 3, 4, 5, 6])
    assert {x.xref_id for x in xrefs} == expected


async def test_flattened_carrier_uses_the_mapped_reference():
    refs, sources = _bibliography([1, 3, 4, 5, 6])
    refs[1] = refs[1].model_copy(update={"title": "The Montreal Cognitive Assessment (MoCA)"})
    sentences = [
        _sent(1, "Prior evidence.3,4 established this."),
        _sent(2, "Later reports4,5 confirmed it."),
        _sent(3, "A synthesis.5,6 agreed."),
        _sent(4, "Further findings3,4 remained stable."),
        _sent(5, "Other analyses.4,5 replicated it."),
        _sent(6, "Final reports5,6 supported it."),
        _sent(7, "We used MoCA3 to assess cognition."),
        *sources,
    ]
    xrefs, receipt = await detect_bib_xrefs_with_receipt(sentences, _sections_with_refs(), refs)
    assert [x.xref_id for x in xrefs if x.text_id == 7] == [2]
    candidate = next(c for c in receipt.candidates if c.text_id == 7)
    assert "reference_carrier_grounded:3" in candidate.evidence
