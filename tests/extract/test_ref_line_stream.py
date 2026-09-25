"""Reference segmentation over one cleaned line stream (bibr.extract.ref_line_stream)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bibr.extract.ref_extractor import (
    ReferenceExtractor,
    _link_dois_for_segments,
    _strip_enum_markers,
)
from bibr.extract.ref_line_stream import (
    LineStream,
    StreamLine,
    build_line_stream,
    segment_line_stream,
    segmentation_quality,
)
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

# ---------------------------------------------------------------------------
# Fixture builders: a page is 1000 x 1000 in both the layout frame (y down) and
# PDF points (y up), so a line's layout box and point geometry agree.
# ---------------------------------------------------------------------------


def _line(text: str, page: int, x0: float, top: float, height: float = 10.0) -> dict:
    width = 8.0 * len(text)
    return {
        "text": text,
        "page": page,
        "x0": x0,
        "y_top": 1000.0 - top,
        "x1": x0 + width,
        "y_bottom": 1000.0 - top - height,
        "font_size": height,
        "bbox": [x0, top, x0 + width, top + height],
    }


def _contents(regions: list[tuple[int, tuple[float, float, float, float], str, str]], lines):
    """Contents with one reference row per ``(page, bbox, label, row_text)`` region."""
    sentences = []
    summaries = []
    for index, (page, bbox, label, text) in enumerate(regions):
        sentences.append(
            PaperSentence(
                text_id=index,
                text=text,
                section_id=1,
                paragraph_id=index,
                page_number=page,
                provenance=[Provenance(page_no=page, bbox=bbox)],
                region_meta={"region_type": label, "region_page": page, "region_index": index},
            )
        )
        summaries.append(RegionSummary(page=page, index=index, label=label, bbox=bbox))
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
        region_summaries=summaries,
        ref_page_lines=lines,
    )


def _ref_df(contents: PaperContents):
    df = contents.sentences_df
    return df[df["section_id"] == 1]


# Two pages of a hanging-indent list. The running head repeats at the top of
# both pages with a different page number, so it is furniture; on page 2 it
# sits inside the reference box, in the middle of the Brown entry. The
# standalone "12" at the foot of page 1 is the page number.
_HEAD_1 = _line("Journal of Tests 12 (2020) 1-20", 1, 50, 20)
_HEAD_2 = _line("Journal of Tests 13 (2020) 1-20", 2, 50, 20)
_P1 = [
    _HEAD_1,
    _line("Smith, J. (2019). A study of things.", 1, 50, 800),
    _line("Journal of Stuff, 12, 1-10.", 1, 70, 815),
    _line("Brown, K. (2020). Another study of", 1, 50, 830),
    _line("12", 1, 480, 960),
]
_P2 = [
    _HEAD_2,
    _line("matters. Journal of Things, 3, 4-5.", 2, 70, 100),
    _line("Clark, L. (2021). Third study.", 2, 50, 115),
    _line("Journal of Stuff, 1, 2-3.", 2, 70, 130),
    _line("Page 2 closing line", 2, 50, 700),
]
_REGIONS = [
    (
        1,
        (45.0, 795.0, 560.0, 975.0),
        "reference_content",
        "Smith, J. (2019). A study of things.\r\nJournal of Stuff, 12, 1-10.\r\n"
        "Brown, K. (2020). Another study of\r\n12",
    ),
    (
        2,
        (45.0, 15.0, 560.0, 145.0),
        "reference_content",
        "Journal of Tests 13 (2020) 1-20\r\nmatters. Journal of Things, 3, 4-5.\r\n"
        "Clark, L. (2021). Third study.\r\nJournal of Stuff, 1, 2-3.",
    ),
]


def _two_page_contents() -> PaperContents:
    return _contents(_REGIONS, [*_P1, *_P2])


def test_stream_reads_region_lines_across_pages_without_furniture():
    contents = _two_page_contents()
    stream = build_line_stream(contents, _ref_df(contents))

    assert stream is not None
    assert [line.text for line in stream.lines] == [
        "Smith, J. (2019). A study of things.",
        "Journal of Stuff, 12, 1-10.",
        "Brown, K. (2020). Another study of",
        "matters. Journal of Things, 3, 4-5.",
        "Clark, L. (2021). Third study.",
        "Journal of Stuff, 1, 2-3.",
    ]
    # The page-2 closing line is outside every reference box.
    assert stream.text_layer_regions == 2
    # Running head inside the page-2 box and the page-1 page number.
    assert stream.furniture_removed == 2
    assert all(line.geometry is not None for line in stream.lines)


def test_stream_joins_an_entry_across_the_page_break():
    contents = _two_page_contents()
    stream = build_line_stream(contents, _ref_df(contents))
    segmentation = segment_line_stream(stream)

    assert segmentation is not None
    assert list(segmentation.entries) == [
        "Smith, J. (2019). A study of things. Journal of Stuff, 12, 1-10.",
        "Brown, K. (2020). Another study of matters. Journal of Things, 3, 4-5.",
        "Clark, L. (2021). Third study. Journal of Stuff, 1, 2-3.",
    ]
    assert "furniture_removed" in segmentation.reason_flags
    for entry, (start, end) in zip(segmentation.entries, segmentation.spans, strict=True):
        assert segmentation.text[start:end] == entry


def test_region_without_text_layer_lines_is_read_from_its_row():
    regions = [
        (1, (45.0, 100.0, 560.0, 140.0), "reference_content", "Adams, P. (2001). Scanned.\nJ, 1."),
        (1, (45.0, 150.0, 560.0, 190.0), "reference_content", "Baker, Q. (2002). Also scanned."),
    ]
    contents = _contents(regions, [])
    stream = build_line_stream(contents, _ref_df(contents))

    assert stream is not None
    assert stream.region_text_regions == 2
    assert [line.text for line in stream.lines] == [
        "Adams, P. (2001). Scanned.",
        "J, 1.",
        "Baker, Q. (2002). Also scanned.",
    ]
    assert list(segment_line_stream(stream).entries) == [
        "Adams, P. (2001). Scanned. J, 1.",
        "Baker, Q. (2002). Also scanned.",
    ]


def test_text_layer_lines_the_rows_do_not_hold_are_left_out():
    # The box also covers the last sentence of the previous section.
    regions = [
        (
            1,
            (45.0, 100.0, 560.0, 200.0),
            "text",
            "Smith, J. (2019). A study of things.\r\nJournal of Stuff, 12, 1-10.",
        )
    ]
    lines = [
        _line("We thank the reviewers for comments.", 1, 50, 105),
        _line("Smith, J. (2019). A study of things.", 1, 50, 120),
        _line("Journal of Stuff, 12, 1-10.", 1, 70, 135),
    ]
    contents = _contents(regions, lines)
    stream = build_line_stream(contents, _ref_df(contents))

    assert [line.text for line in stream.lines] == [
        "Smith, J. (2019). A study of things.",
        "Journal of Stuff, 12, 1-10.",
    ]


def test_manuscript_line_numbers_are_trimmed_from_text_layer_lines():
    regions = [(1, (100.0, 100.0, 900.0, 140.0), "text", "Evans, J. (2011). Eyewitness memory.")]
    number_first = _line("582 Evans, J. (2011). Eyewitness memory.", 1, 30, 105)
    contents = _contents(regions, [number_first])
    # The line starts in the left margin, outside the box, where "582" sits;
    # its centre is still inside the box.
    number_first["bbox"][2] = 700.0
    stream = build_line_stream(contents, _ref_df(contents))

    assert [line.text for line in stream.lines] == ["Evans, J. (2011). Eyewitness memory."]


def test_end_of_list_heading_cuts_the_stream():
    regions = [
        (
            1,
            (45.0, 100.0, 560.0, 400.0),
            "text",
            "Adams, P. (2001). One.\nBaker, Q. (2002). Two.\nAcknowledgements\n"
            "We thank the funder.",
        )
    ]
    contents = _contents(regions, [])
    stream = build_line_stream(contents, _ref_df(contents))
    segmentation = segment_line_stream(stream)

    assert [line.text for line in stream.lines] == [
        "Adams, P. (2001). One.",
        "Baker, Q. (2002). Two.",
    ]
    assert stream.tail_text == "Acknowledgements We thank the funder."
    assert "end_of_list_cut" in segmentation.reason_flags
    assert list(segmentation.entries) == ["Adams, P. (2001). One.", "Baker, Q. (2002). Two."]


# ---------------------------------------------------------------------------
# Start votes and per-entry repair on hand-built streams
# ---------------------------------------------------------------------------


def _stream(texts: list[str], *, label: str = "text", region_starts=()) -> LineStream:
    lines = [
        StreamLine(
            text=text,
            page=1,
            bbox=(0.0, float(i), 10.0, float(i) + 1),
            region=0,
            region_label=label,
            region_first=i in region_starts,
        )
        for i, text in enumerate(texts)
    ]
    return LineStream(lines=lines)


def test_numbering_sequence_decides_the_boundaries():
    stream = _stream(
        [
            "[1] G. Caetano, Los retos de una nueva institucionalidad, 2004.",
            "[2] CEFIR: Centro de Formacion para la Integracion Regional",
            "[3] Inwent: Internationale Weiterbildung und Entwicklung",
            "12 Smith et al. running title",
            "[4] GIZ: Deutsche Gesellschaft fur Internationale",
            "Zusammenarbeit GmbH",
        ]
    )
    segmentation = segment_line_stream(stream)

    assert segmentation.numbered_style
    assert [entry[:4] for entry in segmentation.entries] == ["[1] ", "[2] ", "[3] ", "[4] "]
    # "12 Smith et al." does not continue the sequence, so it opens nothing.
    assert "12 Smith et al." in segmentation.entries[2]
    assert segmentation.numbered == (True, True, True, True)


def test_numbered_list_takes_an_entry_out_of_order_and_ends_at_its_last_box():
    texts = [
        ("1. Baars AJ (1992) Lead intoxication in cattle. Food Add 9:357-364.", 0, "reference"),
        ("2. Lund LJ (1989) Lead poisoning from feed. Vet Rec 125;536.", 1, "reference"),
        ("3. Report of the Chief Veterinary Officer 1989.", 2, "reference"),
        ("5. Sharma RP (1982) Accumulation of lead in milk.", 3, "reference"),
        ("4. MAFF News Releases November 1989 - February 1990.", 4, "reference"),
        ("6. Hathaway SC (1993) Risk assessment. Food Control 4;189-201.", 5, "reference"),
        ("Consent Decree Entered in Animal Drug GMP Case", 6, "text"),
        ("On October 20, 1998, the U.S. District Court entered a decree.", 6, "text"),
    ]
    lines = [
        StreamLine(
            text=text,
            page=1,
            bbox=(0.0, float(i), 10.0, float(i) + 1),
            region=region,
            region_label=label,
            region_first=i == 0 or texts[i - 1][1] != region,
        )
        for i, (text, region, label) in enumerate(texts)
    ]
    segmentation = segment_line_stream(LineStream(lines=lines))

    assert [entry[:2] for entry in segmentation.entries] == ["1.", "2.", "3.", "5.", "4.", "6."]
    assert segmentation.entries[-1].endswith("Food Control 4;189-201.")
    assert "end_of_list_cut" in segmentation.reason_flags
    assert segmentation.text.endswith("the U.S. District Court entered a decree.")


def test_roman_numbering_counts_as_a_sequence():
    stream = _stream(
        [
            "I. Akira Kuramori, Evaluation of Effects, 2004.",
            "II. Byung-Chan Chang, A Study of Classification,",
            "Proc. SICE Annual Conference, 2007.",
            "III. Erez Dagan, Forward Collision Warning, 2004.",
        ]
    )
    segmentation = segment_line_stream(stream)

    assert segmentation.numbered_style
    assert len(segmentation.entries) == 3
    assert segmentation.entries[1].endswith("Proc. SICE Annual Conference, 2007.")


def test_dash_led_same_author_entries_open_entries():
    stream = _stream(
        [
            "Darlington, C. D. 1929. Chromosome behaviour. Jour. Genet. 21: 207-86.",
            "—1930a. Chromosome studies in Fritillaria III. Cytologia 2: 37-55.",
            "- 1931b. The cytological theory of inheritance. Jour. Genet. 24: 405-74.",
            "Mather, K. 1933. The relation between chiasmata. Amer. Nat. 67: 476-9.",
        ],
        region_starts={0, 1, 2, 3},
        label="reference_content",
    )
    segmentation = segment_line_stream(stream)

    assert len(segmentation.entries) == 4


def test_lowercase_continuation_without_a_date_rejoins_the_previous_entry():
    stream = _stream(
        [
            "Norlinda, “Pengaruh Modal Kerja Terhadap Pendapatan Nelayan,” 2019.",
            "sungai utara, Kindai Vol 18, Nomor 1.",
            "Imaniah, I. M. (2016). Peran Modal Sosial. Koperasi, 2, 1-9.",
        ],
        region_starts={0, 1, 2},
        label="reference_content",
    )
    segmentation = segment_line_stream(stream)

    assert len(segmentation.entries) == 2
    assert segmentation.entries[0].endswith("Kindai Vol 18, Nomor 1.")


def test_entry_holding_two_dois_is_split_after_the_first():
    stream = _stream(
        [
            "Evans, J. (2011). Eyewitness memory. Applied Cognitive",
            "Psychology, 25, 501-508. https://doi.org/10.1002/acp.1722",
            "Eysenck M. W. 1979 Anxiety learning and memory Journal of",
            "Research in Personality 13 363-385 https://doi.org/10.1016/0092-6566(79)90001-1",
        ]
    )
    segmentation = segment_line_stream(stream)

    assert len(segmentation.entries) == 2
    assert segmentation.entries[1].startswith("Eysenck")


def test_doi_link_annotation_fills_an_entry_that_prints_none():
    stream = _stream(
        [
            "Smith, J. (2019). A study of things. Journal, 1, 2. [CrossRef]",
            "Brown, K. (2020). Another study. Journal, 3, 4. [CrossRef]",
        ]
    )
    stream.lines[0].link_dois.append("10.1000/one")
    stream.lines[1].link_dois.append("10.1000/two")
    segmentation = segment_line_stream(stream)

    assert segmentation.link_dois == ("10.1000/one", "10.1000/two")
    assert "doi_from_link_annotation" in segmentation.reason_flags


def test_link_dois_map_onto_the_cascade_segment_holding_the_entry():
    entries = [
        "Smith, J. (2019). A study of things. Journal, 1, 2. [CrossRef]",
        "Brown, K. (2020). Another study. Journal, 3, 4. https://doi.org/10.1000/printed",
    ]
    segments = [
        "Smith, J. (2019). A study of things. Journal, 1, 2. [CrossRef]",
        "Brown, K. (2020). Another study. Journal, 3, 4. https://doi.org/10.1000/printed",
    ]
    mapped = _link_dois_for_segments(segments, entries, ["10.1000/one", "10.1000/two"])

    # The second segment prints its own DOI and keeps it.
    assert mapped == ["10.1000/one", None]


# ---------------------------------------------------------------------------
# Quality and selection
# ---------------------------------------------------------------------------


def test_quality_prefers_single_references_over_merges_and_fragments():
    section = "".join(
        ch
        for ch in (
            "Smith, J. (2019). A study of things. Journal, 1, 2."
            "Brown, K. (2020). Another study. Journal, 3, 4."
            "Clark, L. (2021). Third study. Journal, 5, 6."
        ).lower()
        if ch.isalnum()
    )
    good = [
        "Smith, J. (2019). A study of things. Journal, 1, 2.",
        "Brown, K. (2020). Another study. Journal, 3, 4.",
        "Clark, L. (2021). Third study. Journal, 5, 6.",
    ]
    merged = [" ".join(good)]
    fragments = ["Smith, J. (2019).", "A study of things. Journal, 1, 2.", *good[1:]]

    assert segmentation_quality(good, section) == pytest.approx(1.0)
    assert segmentation_quality(merged, section) == 0.0
    assert segmentation_quality(fragments, section) < 0.8


def _fake_parser(texts: list[str]) -> list[dict]:
    return [{"title": text, "authors": "A"} for text in texts]


async def _extract_with_geom_spans(contents, spans, confidence=0.99):
    """Run ReferenceExtractor.extract with a fake geom segmenter and NER parser."""
    ref_df = _ref_df(contents)
    contents.ref_line_geometry = [{"text": "x"}]
    extractor = ReferenceExtractor(
        contents, file_hash="h", llm_client=MagicMock(), seg_strategy="geom", parse_strategy="ner"
    )
    segmenter = MagicMock()
    segmenter.segment_spans.side_effect = lambda ref_text, _lines: (
        spans(ref_text),
        confidence,
        len(spans(ref_text)),
        len(spans(ref_text)),
    )
    segmenter.line_start_probabilities.side_effect = lambda records: [0.5] * len(records)
    parser = MagicMock()
    parser.parse_batch.side_effect = _fake_parser
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=segmenter),
        patch("bibr.extract.ref_extractor._get_ner_parser", return_value=parser),
    ):
        refs = await extractor.extract(ref_df)
    return refs, contents.reference_yield_receipt


def _entry_spans(ref_text: str) -> list[tuple[int, int]]:
    starts = sorted(ref_text.find(name) for name in ("Smith, J.", "Brown, K.", "Clark, L."))
    ends = [*starts[1:], len(ref_text)]
    return [(start, len(ref_text[:end].rstrip())) for start, end in zip(starts, ends, strict=True)]


async def test_confident_geom_result_is_kept_when_the_stream_is_not_clearly_better():
    contents = _two_page_contents()
    refs, receipt = await _extract_with_geom_spans(contents, _entry_spans)

    strategies = {attempt.strategy: attempt for attempt in receipt.attempts}
    assert strategies["geom"].selected
    assert not strategies["line_stream"].selected
    assert "quality_margin" in strategies["line_stream"].reason_flags
    # The cascade's own segments were parsed, running head and all.
    assert len(refs) == 3
    assert "Journal of Tests 13" in refs[1].title


async def test_confident_but_merged_geom_result_gives_way_to_the_stream():
    contents = _two_page_contents()

    def one_span(ref_text: str) -> list[tuple[int, int]]:
        return [(0, len(ref_text))]

    refs, receipt = await _extract_with_geom_spans(contents, one_span)

    strategies = {attempt.strategy: attempt for attempt in receipt.attempts}
    assert not strategies["geom"].selected
    assert "superseded_by_line_stream" in strategies["geom"].reason_flags
    assert strategies["line_stream"].selected
    assert [ref.title for ref in refs] == [
        "Smith, J. (2019). A study of things. Journal of Stuff, 12, 1-10.",
        "Brown, K. (2020). Another study of matters. Journal of Things, 3, 4-5.",
        "Clark, L. (2021). Third study. Journal of Stuff, 1, 2-3.",
    ]


async def test_stream_replaces_a_region_recovery_result_unless_clearly_worse():
    contents = _two_page_contents()
    ref_df = _ref_df(contents)
    extractor = ReferenceExtractor(
        contents, file_hash="h", llm_client=MagicMock(), seg_strategy="geom", parse_strategy="ner"
    )

    async def region_recovery(ref_text: str, _strategy: str) -> list[str]:
        # The region tier found two anchors and merged Brown into Smith.
        cut = ref_text.find("Clark, L.")
        segments = [ref_text[:cut].strip(), ref_text[cut:].strip()]
        extractor._record_segmentation_attempt("region", ref_text, segments=segments, selected=True)
        return segments

    parser = MagicMock()
    parser.parse_batch.side_effect = _fake_parser
    with (
        patch.object(extractor, "_segment_references", side_effect=region_recovery),
        patch("bibr.extract.ref_extractor._get_ner_parser", return_value=parser),
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=None),
    ):
        refs = await extractor.extract(ref_df)

    strategies = {
        attempt.strategy: attempt for attempt in contents.reference_yield_receipt.attempts
    }
    assert strategies["line_stream"].selected
    assert "cascade_fallback" in strategies["line_stream"].reason_flags
    assert "superseded_by_line_stream" in strategies["region"].reason_flags
    assert len(refs) == 3


def test_split_reference_section_is_joined_in_front_of_the_located_rows():
    regions = [
        (1, (45.0, 800.0, 560.0, 840.0), "text", "Adams, P. (2001). One. J, 1, 2."),
        (2, (45.0, 100.0, 560.0, 140.0), "reference", "Baker, Q. (2002). Two. J, 3, 4."),
    ]
    contents = _contents(regions, [])
    contents.sections[1].header = "Referencias"
    contents.sections.append(
        PaperSection(
            section_id=2,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
            header_is_synthetic=True,
        )
    )
    contents.sentences[1].section_id = 2
    contents.invalidate_text_caches()
    df = contents.sentences_df
    located = df[df["section_id"] == 2]

    stream = build_line_stream(contents, located)

    assert stream.split_section
    assert [line.text for line in stream.lines] == [
        "Adams, P. (2001). One. J, 1, 2.",
        "Baker, Q. (2002). Two. J, 3, 4.",
    ]
    assert "split_section_joined" in segment_line_stream(stream).reason_flags


# ---------------------------------------------------------------------------
# Parser input
# ---------------------------------------------------------------------------


def test_roman_list_numbers_are_stripped_for_the_parser_only_in_a_roman_list():
    roman = ["I. Kuramori, A. 2004.", "II. Chang, B. 2007.", "III. Dagan, E. 2004."]
    assert _strip_enum_markers(roman) == [
        "Kuramori, A. 2004.",
        "Chang, B. 2007.",
        "Dagan, E. 2004.",
    ]
    initials = ["V. Lal, S. 2003.", "Smith J. 2004.", "Jones K. 2005."]
    assert _strip_enum_markers(initials) == initials
