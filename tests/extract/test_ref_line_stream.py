"""Reference segmentation over one cleaned line stream (bibr.extract.ref_line_stream)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bibr.extract.ref_extractor import ReferenceExtractor, _distinct_count, _strip_enum_markers
from bibr.extract.ref_line_stream import (
    LineStream,
    StreamLine,
    _is_dash_start,
    build_line_stream,
    is_merged_entry,
    link_dois_for_segments,
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
# standalone "12" at the foot of page 1 is the page number: page 2 carries
# "13" at its foot, one page on.
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
    _line("13", 2, 480, 960),
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


def test_locator_lines_and_lone_years_at_page_edges_are_not_furniture():
    # Two pages whose edge lines, digits masked, look alike: DOI lines of one
    # publisher, and a lone year. Neither is page furniture.
    page_1 = [
        ("Adams, P. (2001). One study of things. Psychological", 50, 700),
        ("Science, 12(3), 1-10. https://doi.org/10.1037/xge0000011", 70, 715),
        ("Baker, Q. (2017). Two study of things. Psychological", 50, 730),
        ("Science, 28(7), 1002-1014.", 70, 745),
        ("https://doi.org/10.1177/0956797617693326", 70, 760),
    ]
    page_2 = [
        ("https://doi.org/10.1177/0956797619881234", 70, 60),
        ("2015", 70, 75),
        ("Dunn, S. (2004). Four. Journal of Things, 7, 8-9.", 50, 90),
        ("Evans, T. (2005). Five. Journal of Things, 9, 10-11.", 50, 105),
        ("Fox, U. (2006). Six. Journal of Things, 11, 12-13.", 50, 120),
    ]
    lines = [_line(text, 1, x, top) for text, x, top in page_1]
    lines += [_line(text, 2, x, top) for text, x, top in page_2]
    regions = [
        (1, (45.0, 695.0, 900.0, 775.0), "reference_content", "\r\n".join(t for t, _, _ in page_1)),
        (2, (45.0, 55.0, 900.0, 135.0), "reference_content", "\r\n".join(t for t, _, _ in page_2)),
    ]
    contents = _contents(regions, lines)
    stream = build_line_stream(contents, _ref_df(contents))

    assert stream.furniture_removed == 0
    assert "https://doi.org/10.1177/0956797617693326" in [line.text for line in stream.lines]
    assert "2015" in [line.text for line in stream.lines]


def test_heading_word_inside_an_entry_does_not_end_the_list():
    regions = [
        (
            1,
            (45.0, 100.0, 560.0, 400.0),
            "text",
            "Adams, P. (2001). One. J, 1, 2.\nBaker, Q. (2002). Two. J, 3, 4.\n"
            "Carter, R. (2003). Survey instrument, reported in\nAppendix B.\n"
            "Journal of Methods, 5, 6-7.\nDunn, S. (2004). Four. J, 7, 8.",
        )
    ]
    contents = _contents(regions, [])
    stream = build_line_stream(contents, _ref_df(contents))

    assert stream.tail_text == ""
    assert len(segment_line_stream(stream).entries) == 4


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


def test_continuation_lines_opening_on_a_number_do_not_join_the_sequence():
    stream = _stream(
        [
            "1. Adams P. One study. J 2001;1:2-3.",
            "2. Baker Q. Two study. Berlin: Springer;",
            "3. Aufl. 2002.",
            "3. Carter R. Three. J 2003;5:6-7.",
            "4. Dunn S. Four. J 2004;7:8-9.",
            "5. Ivers R, Wang H, Zhou Q,",
            "X. Li, Z. Zhao. Five. J 2005;9:10-11.",
            "6. Jones K. Six. J 2006;11:12-13.",
            "0. Line that numbers nothing",
        ]
    )
    segmentation = segment_line_stream(stream)

    assert [entry[:2] for entry in segmentation.entries] == ["1.", "2.", "3.", "4.", "5.", "6."]
    assert segmentation.entries[1].endswith("3. Aufl. 2002.")
    assert "X. Li, Z. Zhao." in segmentation.entries[4]


def test_a_second_list_numbered_from_one_again_keeps_its_entries():
    main = [f"{n}. Author{n} A. Main study {n}. J 2001;{n}:1-2." for n in range(1, 5)]
    supplement = [f"{n}. Writer{n} B. Extra study {n}. J 2002;{n}:3-4." for n in range(1, 13)]
    segmentation = segment_line_stream(_stream([*main, *supplement]))

    assert len(segmentation.entries) == 16


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


def test_link_doi_goes_to_the_cascade_segment_holding_the_linked_line():
    stream = _stream(
        [
            "Andersson, B., Carlsson, D., & Eriksson, F. (2015). A long study of things.",
            "Journal of Stuff, 12(3), 100-120.",
            "Berg, K. (2016). Short. J Things, 3, 4-5. [CrossRef]",
            "Clark, L. (2021). Third. J, 5, 6. https://doi.org/10.1000/printed [CrossRef]",
        ]
    )
    stream.lines[2].link_dois.append("10.1000/berg")
    stream.lines[3].link_dois.append("10.1000/clark")
    segments = [
        "Andersson, B., Carlsson, D., & Eriksson, F. (2015). A long study of things. "
        "Journal of Stuff, 12(3), 100-120.",
        "Berg, K. (2016). Short. J Things, 3, 4-5. [CrossRef]",
        "Clark, L. (2021). Third. J, 5, 6. https://doi.org/10.1000/printed [CrossRef]",
    ]

    # Clark prints its own DOI and keeps it; Andersson has no link over its lines.
    assert link_dois_for_segments(segments, stream.lines) == [None, "10.1000/berg", None]


def test_link_doi_stays_off_a_segment_holding_two_linked_entries():
    stream = _stream(
        [
            "Berg, K. (2016). Short. J Things, 3, 4-5. [CrossRef]",
            "Clark, L. (2021). Third. J, 5, 6. [CrossRef]",
        ]
    )
    stream.lines[0].link_dois.append("10.1000/berg")
    stream.lines[1].link_dois.append("10.1000/clark")
    merged = [
        "Berg, K. (2016). Short. J Things, 3, 4-5. [CrossRef] "
        "Clark, L. (2021). Third. J, 5, 6. [CrossRef]"
    ]

    assert link_dois_for_segments(merged, stream.lines) == [None]


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


def test_numbered_undated_entries_score_alike_whoever_segmented_them():
    entries = [
        "[1] Python Software Foundation. Python Language Reference. https://www.python.org",
        "[2] Adams P. One study. J 2001;1:2-3.",
        "[3] ISO 9001 Quality management systems. Geneva: ISO.",
        "[4] Baker Q. Two study. J 2002;3:4-5.",
    ]
    key = "".join(ch for ch in " ".join(entries).lower() if ch.isalnum())

    assert segmentation_quality(entries, key) == pytest.approx(1.0)


def test_two_short_dated_references_in_one_entry_count_as_a_merge():
    entries = ["Adams, P. (2001). One. J, 1, 2. Baker, Q. (2002). Two. J, 3, 4."]
    key = "".join(ch for ch in entries[0].lower() if ch.isalnum())

    assert segmentation_quality(entries, key) == 0.0


@pytest.mark.parametrize(
    ("text", "merged"),
    [
        ("Smith, J. (2019). Title. Journal, 12(3), 1765-1770.", False),
        (
            "UFAC. Projeto Curricular. Acre, 2015. Disponível em: <https://x.org/y>. "
            "Acesso em: 13 abr. 2025.",
            False,
        ),
        ("Smith, A. (2002). The theory. London: Press. (Original work published 1759)", False),
        ("Lee, K. Title one. J 2001;1:2-3. Kim, H. Title two. J 2004;5:6-7.", True),
        ("Adams P. One. Journal of X. 1998. Baker Q. Two. Journal of Y. 2003.", True),
    ],
)
def test_merge_check_counts_publication_years_only(text, merged):
    assert is_merged_entry(text, typical_length=200.0) is merged


def test_dash_led_starts_need_a_break_after_a_single_hyphen():
    assert _is_dash_start("-1931. Chiasmas in flowering plants.")
    assert _is_dash_start("—. Ueber den Charakter.")
    assert _is_dash_start("- and Fischer, R. 1932.")
    assert not _is_dash_start("-Parfor Equidade. Edital 23/2023.")


def test_near_repeats_count_once_unless_their_years_differ():
    first = "Ridley, A. M. (2003). The effect of anxiety on eyewitness testimony. Thesis."
    reread = "Ridley, A. M. (2003). The efect of anxiety on eyewitness testimony. Thesis."
    edition = "Ridley, A. M. (2004). The effect of anxiety on eyewitness testimony. Thesis."

    assert _distinct_count([first, reread]) == 1
    assert _distinct_count([first, edition]) == 2


async def test_stream_with_fewer_entries_never_replaces_a_fallback_result():
    contents = _two_page_contents()
    ref_df = _ref_df(contents)
    extractor = ReferenceExtractor(
        contents, file_hash="h", llm_client=MagicMock(), seg_strategy="geom", parse_strategy="ner"
    )

    async def region_recovery(ref_text: str, _strategy: str) -> list[str]:
        # Four segments: one more than the three references the stream finds.
        cuts = [ref_text.find(name) for name in ("Smith, J.", "Brown, K.", "Clark, L.")]
        cuts.insert(2, ref_text.find("Journal of Tests"))
        bounds = [*cuts[1:], len(ref_text)]
        segments = [ref_text[a:b].strip() for a, b in zip(cuts, bounds, strict=True)]
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
    assert not strategies["line_stream"].selected
    assert "fewer_entries" in strategies["line_stream"].reason_flags
    assert strategies["region"].selected
    assert len(refs) == 4


async def test_stream_does_not_replace_a_result_on_a_rotated_page():
    contents = _two_page_contents()
    for line in contents.ref_page_lines:
        line["rotation"] = 90

    def one_span(ref_text: str) -> list[tuple[int, int]]:
        return [(0, len(ref_text))]

    refs, receipt = await _extract_with_geom_spans(contents, one_span)

    strategies = {attempt.strategy: attempt for attempt in receipt.attempts}
    assert strategies["geom"].selected
    assert not strategies["line_stream"].selected
    assert "rotated_page" in strategies["line_stream"].reason_flags


async def test_an_error_in_the_stream_decision_keeps_the_cascade_result():
    contents = _two_page_contents()
    with patch("bibr.extract.ref_extractor.segmentation_quality", side_effect=RuntimeError("boom")):
        refs, receipt = await _extract_with_geom_spans(contents, _entry_spans)

    strategies = {attempt.strategy: attempt for attempt in receipt.attempts}
    assert strategies["geom"].selected
    assert "line_stream" not in strategies
    assert len(refs) == 3


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
