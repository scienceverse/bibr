"""The DOI candidate pool with the PDF's own evidence: text layer, links, metadata."""

from __future__ import annotations

import pytest

from bibr.extract.doi_identity import (
    AGREEMENT_ONLY,
    LINE_JOIN_OVERRUN,
    collect_doi_candidates,
    select_doi_candidates,
)
from bibr.extract.pdf_doi_evidence import LinkDoi, MetadataDoi, PdfDoiEvidence, TextLayerLine
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    RegionSummary,
)

# A point in the right page margin, outside every layout region below.
_MARGIN = (970.0, 500.0)


def _contents(
    rows: list[tuple[CanonicalSection, str, int]],
    *,
    headers: list[str] | None = None,
    footers: list[str] | None = None,
    regions: list[RegionSummary] | None = None,
) -> PaperContents:
    sections = [PaperSection(0, "Root", 0, None, CanonicalSection.TITLE)]
    sentences = []
    for index, (section_type, text, page) in enumerate(rows, start=1):
        sections.append(PaperSection(index, section_type.value, 1, 0, section_type))
        sentences.append(
            PaperSentence(
                text_id=index,
                text=text,
                section_id=index,
                paragraph_id=index,
                page_number=page,
                region_meta={"region_type": "text", "region_page": page, "region_index": index},
            )
        )
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
        detected_headers=list(headers or []),
        detected_footers=list(footers or []),
        region_summaries=list(regions or []),
    )


def _line(page: int, text: str, *, at=_MARGIN, next_text: str = "") -> TextLayerLine:
    return TextLayerLine(page, text, tuple(at for _ in text), next_text)


def _evidence(lines=(), links=(), metadata=()) -> PdfDoiEvidence:
    return PdfDoiEvidence((1, 2), tuple(lines), tuple(links), tuple(metadata))


def _xmp(doi: str) -> MetadataDoi:
    return MetadataDoi("pdf_xmp", "prism:doi", doi)


def _select(contents, evidence=None):
    candidates = collect_doi_candidates(contents, evidence)
    return candidates, select_doi_candidates(candidates)


_TITLE_ONLY = [(CanonicalSection.TITLE, "A study of examples", 1)]


@pytest.mark.parametrize(
    "banner",
    [
        "Example J: first published as 10.1234/banner.7 on 1 May 1999. Downloaded from "
        "https://example.org/",
        # The text layer may give the banner's phrases in reverse order.
        ".by guest on 1 May 2026 https://example.org/ Downloaded from 1 May 1999. "
        "10.1234/banner.7 on Example J: first published as",
    ],
)
def test_a_margin_banner_names_the_paper_the_parse_never_read(banner):
    candidates, selection = _select(_contents(_TITLE_ONLY), _evidence([_line(1, banner)]))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/banner.7"
    assert selection.selected.source_kind == "text_layer"
    assert selection.selected.marker_kind == "first_published_as"
    assert selection.selected.selection_tier == 3
    assert (selection.selected.page, selection.selected.region_index) == (1, None)
    assert selection.issues == ()


@pytest.mark.parametrize(
    ("label", "section_type", "rejection"),
    [
        ("reference_content", None, "reference_candidate"),
        ("table", None, "component_candidate"),
        ("image", None, "component_candidate"),
        ("text", CanonicalSection.REFERENCES, "reference_candidate"),
        ("text", CanonicalSection.TITLE, None),
    ],
)
def test_a_text_layer_doi_takes_the_context_of_its_layout_region(label, section_type, rejection):
    rows = [(CanonicalSection.TITLE, "A study of examples", 1)]
    if section_type is not None:
        rows.append((section_type, "Region text", 1))
    region = RegionSummary(
        page=1,
        index=4,
        label=label,
        bbox=(100.0, 400.0, 900.0, 600.0),
        section_id=2 if section_type is not None else None,
    )
    contents = _contents(rows, regions=[region])
    line = _line(1, "Prior work. https://doi.org/10.1234/inside.3", at=(500.0, 500.0))

    candidates = collect_doi_candidates(contents, _evidence([line]))

    [candidate] = [c for c in candidates if c.source_kind == "text_layer"]
    assert (candidate.page, candidate.region_index, candidate.region_type) == (1, 4, label)
    assert candidate.rejection_reason == rejection


def test_a_text_layer_doi_the_parse_already_holds_is_not_added_again():
    contents = _contents(
        [(CanonicalSection.TITLE, "https://doi.org/10.1234/abc.5", 1)],
        footers=["Example Journal | https://doi.org/10.1234/foot.6"],
    )
    lines = [
        _line(1, "https://doi.org/10.1234/abc.5"),
        # The pageless furniture holds this one for every page.
        _line(2, "Example Journal | https://doi.org/10.1234/foot.6"),
    ]

    candidates = collect_doi_candidates(contents, _evidence(lines))

    assert not [c for c in candidates if c.source_kind == "text_layer"]


def test_a_doi_printed_on_a_front_page_joins_even_when_a_later_page_repeats_it():
    contents = _contents(
        [
            (CanonicalSection.TITLE, "A study of examples", 1),
            (CanonicalSection.DISCUSSION, "Data at https://doi.org/10.1234/own.9 as well.", 9),
        ]
    )
    without = select_doi_candidates(collect_doi_candidates(contents))

    _candidates, selection = _select(contents, _evidence([_line(1, "doi.org/10.1234/own.9")]))

    assert without.selected is None
    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/own.9"
    assert selection.selected.source_kind == "text_layer"


def test_the_first_part_of_a_wrapped_doi_is_not_a_candidate():
    contents = _contents(
        [(CanonicalSection.TITLE, "Example 1: e0279. https://doi.org/10.1234/journal.pone.0279", 1)]
    )
    line = _line(1, "Example 1: e0279. https://doi.org/10.1234/journal.", next_text="pone.0279")

    candidates, selection = _select(contents, _evidence([line]))

    assert not [c for c in candidates if c.source_kind == "text_layer"]
    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/journal.pone.0279"


def test_a_partial_text_layer_reading_is_not_added():
    # The text layer lost the DOI's end mid-line; the parse read all of it.
    contents = _contents([(CanonicalSection.TITLE, "DOI: 10.1234/own.2020.004", 1)])
    line = _line(1, "DOI: 10.1234/own.2020\ufffd004 received", next_text="Accepted")

    candidates, selection = _select(contents, _evidence([line]))

    assert not [c for c in candidates if c.source_kind == "text_layer"]
    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/own.2020.004"


def test_the_parse_reading_of_a_layout_region_wins_over_the_text_layer():
    # A scan's hidden text layer can garble what the OCR of the region read.
    region = RegionSummary(page=1, index=1, label="text", bbox=(100.0, 400.0, 900.0, 600.0))
    contents = _contents(
        [(CanonicalSection.TITLE, "DOI: 10.1234/own.2020.004", 1)], regions=[region]
    )
    line = _line(1, "DOI: 10.1234/own.2O20.OO4", at=(500.0, 500.0))

    candidates, selection = _select(contents, _evidence([line]))

    assert not [c for c in candidates if c.source_kind == "text_layer"]
    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/own.2020.004"


@pytest.mark.parametrize(
    ("footer", "metadata", "overrun"),
    [
        # The parse glued the ISSN line under the DOI onto it.
        ("https://doi.org/10.1234/jex.2026.04.0061234-5678© 2026 The Authors.", (), True),
        # Only the line-end reading agrees with the document metadata.
        ("https://doi.org/10.1234/jex.2026.04.0062468", (_xmp("10.1234/jex.2026.04.006"),), True),
        # Otherwise a digit run continued on the next line is a wrapped DOI.
        ("https://doi.org/10.1234/jex.2026.04.0062468", (), False),
    ],
)
def test_a_parsed_doi_that_ran_into_the_next_line_is_rejected(footer, metadata, overrun):
    contents = _contents(_TITLE_ONLY, footers=[footer])
    line = _line(
        1,
        "https://doi.org/10.1234/jex.2026.04.006",
        next_text="1234-5678© 2026 The Authors." if overrun and not metadata else "2468",
    )

    candidates, selection = _select(contents, _evidence([line], metadata=metadata))

    [parsed] = [c for c in candidates if c.source_kind == "footer"]
    if overrun:
        assert parsed.rejection_reason == LINE_JOIN_OVERRUN
        assert selection.selected is not None
        assert selection.selected.normalized == "10.1234/jex.2026.04.006"
        assert selection.selected.source_kind == "text_layer"
    else:
        assert parsed.rejection_reason is None
        assert not [c for c in candidates if c.source_kind == "text_layer"]
        assert selection.selected is not None
        assert selection.selected.normalized == parsed.normalized


def test_agreement_breaks_a_tie_between_printed_rivals():
    contents = _contents(
        [
            (CanonicalSection.TITLE, "See https://doi.org/10.1234/first.1 for the data.", 1),
            (CanonicalSection.TITLE, "https://doi.org/10.1234/second.2", 1),
        ]
    )
    _candidates, alone = _select(contents)

    _candidates, selection = _select(contents, _evidence(metadata=[_xmp("10.1234/second.2")]))

    assert alone.selected is None
    assert [issue.code for issue in alone.issues] == ["VAL_DOI_AMBIGUOUS"]
    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/second.2"
    assert selection.selected.source_kind == "sentence"
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]


@pytest.mark.parametrize(
    ("printed_text", "resolved"),
    [
        # A link over the DOI's own print repeats it; it does not agree.
        ("https://doi.org/10.1234/second.2", False),
        # A link on the journal's citation line names the paper's DOI.
        ("Example Journal 11 (2026) 1-12", True),
    ],
)
def test_only_a_link_whose_text_is_not_the_doi_agrees(printed_text, resolved):
    contents = _contents(
        [
            (CanonicalSection.TITLE, "See https://doi.org/10.1234/first.1 for the data.", 1),
            (CanonicalSection.TITLE, "https://doi.org/10.1234/second.2", 1),
        ]
    )
    link = LinkDoi(
        1, "10.1234/second.2", "https://doi.org/10.1234/second.2", (0, 0, 1, 1), printed_text
    )

    _candidates, selection = _select(contents, _evidence(links=[link]))

    assert (selection.selected is not None) is resolved
    if resolved:
        assert selection.selected.normalized == "10.1234/second.2"


def test_link_and_metadata_dois_are_never_selected_alone():
    links = [
        LinkDoi(1, "10.1234/abc.5", "https://doi.org/10.1234/abc.5", (0, 0, 1, 1), "Example J 1"),
        LinkDoi(
            1,
            "10.1234/abc.5",
            "https://doi.org/10.1234/abc.5",
            (0, 0, 1, 1),
            "doi.org/10.1234/abc.5",
        ),
        LinkDoi(2, "10.1234/abc.5", "https://doi.org/10.1234/abc.5", (0, 0, 1, 1), "Example J 1"),
    ]
    metadata = [
        MetadataDoi("pdf_info", "Subject", "Example J 1 (2020) 1-2. doi:10.1234/abc.5"),
        MetadataDoi("pdf_info", "doi", "10.1234/abc.5"),
        _xmp("10.1234/abc.5"),
    ]

    candidates, selection = _select(
        _contents(_TITLE_ONLY), _evidence(links=links, metadata=metadata)
    )

    assert selection.selected is None
    assert {c.rejection_reason for c in candidates} == {AGREEMENT_ONLY}
    assert [(c.source_kind, c.page, c.semantic_context, c.marker_kind) for c in candidates] == [
        ("link_annotation", 1, "link_target", "link_uri"),
        ("link_annotation", 2, "link_target", "link_uri"),
        ("pdf_info", None, "pdf_metadata", "Subject"),
        ("pdf_xmp", None, "pdf_metadata", "prism:doi"),
    ]


def test_a_link_over_the_printed_doi_is_marked_printed():
    link = LinkDoi(
        1,
        "10.1234/abc.5",
        "https://doi.org/10.1234/abc.5",
        (0, 0, 1, 1),
        "https://doi.org/10.1234/\nabc.5",
    )

    candidates = collect_doi_candidates(_contents(_TITLE_ONLY), _evidence(links=[link]))

    assert [c.semantic_context for c in candidates] == ["printed_link"]


def test_agreement_confirms_a_printed_body_doi():
    contents = _contents(
        [
            (CanonicalSection.TITLE, "A study of examples", 1),
            (CanonicalSection.INTRODUCTION, "As published at 10.1234/own.9 in full.", 5),
        ]
    )
    _candidates, alone = _select(contents)

    candidates, selection = _select(contents, _evidence(metadata=[_xmp("10.1234/own.9")]))

    assert alone.selected is None
    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/own.9"
    assert selection.selected.selection_tier == 1


@pytest.mark.parametrize(
    "text",
    [
        "Supplementary materials: https://doi.org/10.1234/own.9.supp",
        "Data are available at https://doi.org/10.5281/zenodo.1234",
        "[3] Smith J. Prior work. https://doi.org/10.1234/cited.3",
    ],
)
def test_agreement_never_revives_a_rejected_candidate(text):
    contents = _contents([(CanonicalSection.TITLE, text, 1)])
    [doi] = [c.normalized for c in collect_doi_candidates(contents)]

    candidates, selection = _select(contents, _evidence(metadata=[_xmp(doi)]))

    assert selection.selected is None
    assert candidates[0].rejection_reason is not None


def test_a_complete_text_layer_reading_beats_a_parse_that_lost_the_end():
    contents = _contents([(CanonicalSection.TITLE, "Published 2024. DOI: 10.1234/ABC/25.202.3", 1)])
    line = _line(1, "Published: 01.09.2024 DOI: 10.1234/ABC/25.202.33")

    _candidates, selection = _select(contents, _evidence([line]))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/abc/25.202.33"
    assert selection.selected.source_kind == "text_layer"


def test_without_pdf_evidence_the_pool_is_unchanged():
    contents = _contents(
        [(CanonicalSection.TITLE, "https://doi.org/10.1234/abc.5", 1)],
        headers=["Example J 1 (2020) doi:10.1234/abc.5"],
    )

    assert collect_doi_candidates(contents) == collect_doi_candidates(contents, None)
    assert collect_doi_candidates(contents, _evidence()) == collect_doi_candidates(contents)
