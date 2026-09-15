"""List cleanup must not corrupt an already visible DOI or invent one."""

import pytest

from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates
from bibr.ocr.postprocess import clean_ocr_content
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence


@pytest.mark.parametrize(
    "text",
    [
        "10.1234/synthetic.2025.a",
        "10.123456789/Synthetic(2025)-A",
        "10.1234/synthetic.2025.a More publication information",
        "10.1234/synthetic.2025.a, with accompanying prose",
        "10.1234/synthetic.2025.a\nAnother header line",
    ],
)
def test_leading_doi_survives_list_cleanup_verbatim(text):
    assert clean_ocr_content(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10.item", "10. item"),
        ("10. item", "10. item"),
        ("10)Next item", "10) Next item"),
        ("(10)Next item", "(10) Next item"),
        ("A)First item", "A) First item"),
        ("10.Article with DOI: 10.1234/cited", "10. Article with DOI: 10.1234/cited"),
        ("10.10.1234/cited", "10. 10.1234/cited"),
        ("DOI: 10.1234/synthetic", "DOI: 10.1234/synthetic"),
        ("https://doi.org/10.1234/synthetic", "https://doi.org/10.1234/synthetic"),
        ("A reference to 10.1234/synthetic", "A reference to 10.1234/synthetic"),
    ],
)
def test_real_lists_and_nonleading_identifiers_keep_existing_behavior(text, expected):
    assert clean_ocr_content(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "10. 1234/synthetic",
        "10.123/no-short-registrant",
        "10.1234567890/no-long-registrant",
        "10.1234/",
        "10.1234/ missing-suffix",
        "10.1234 units measured",
    ],
)
def test_malformed_doi_text_is_never_joined_or_admitted(text):
    contents = PaperContents([], [], [], [], {}, detected_headers=[clean_ocr_content(text)])

    assert collect_doi_candidates(contents) == ()


def test_preserved_header_is_source_visible_without_expected_identity():
    raw = "10.1234/synthetic.2025.a"
    contents = PaperContents([], [], [], [], {}, detected_headers=[clean_ocr_content(raw)])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == raw
    assert selection.selected.raw == raw
    assert selection.selected.source_kind == "header"
    assert selection.selected.marker_kind == "bare"


def test_preserving_reference_doi_does_not_promote_it_to_article_identity():
    raw = "10.1234/cited"
    contents = PaperContents(
        [PaperSentence(1, clean_ocr_content(raw), 1, 1, 1)],
        [PaperSection(1, "References", 1, None, CanonicalSection.REFERENCES)],
        [],
        [],
        {},
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert len(selection.candidates) == 1
    assert selection.candidates[0].rejection_reason == "reference_candidate"


def test_competing_preserved_headers_remain_ambiguous():
    contents = PaperContents(
        [],
        [],
        [],
        [],
        {},
        detected_headers=[
            clean_ocr_content(value) for value in ("10.1234/first", "10.5678/second")
        ],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert len(selection.candidates) == 2
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]
