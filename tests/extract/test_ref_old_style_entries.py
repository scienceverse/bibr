"""Old-style bibliographies that print several works under one byline.

From dev-set scans (texts shortened from the real entries):

* scans_eval80 W2084009407 (1930s author-year list): "Ehrke, G., 1931, Arch.
  wissensch. Bot., 13, 221; 1932, 17, 650." holds two works; the second has no
  byline of its own, so the parser needs the byline in front of it.
* scans_eval80 W1981667543: entries that open with the "same author" dash
  ("-1931b. The cytological theory …") run on after the entry above, on the
  same line or on a new line of the same segment, in both the cascade's and
  the line stream's segments.
* scans_eval80 W1512170573 (numbered physics list): one number holds several
  works, each with its own initials-first byline after the previous work's
  pages ("[18] J.D. Bekenstein, …, 2333-2346. J.D. Bekenstein, …"; "[30] J.F.
  Plebanski, …, 2511; J. Samuel, …"), and entries reach the parser cut into a
  numbered head and unnumbered fragments ("Phys. Rev., D9 (1974), 3292-3300.").
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bibr.extract.ref_extractor import (
    ReferenceExtractor,
    _split_inline_dash_entries,
    _split_numbered_entry_works,
    _split_same_byline_works,
    _strip_enum_markers,
)
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

_OLD_STYLE_LIST = [
    "Allen, E. J., 1914, J. Marine Biol. Assn., 10, 417.",
    "Ehrke, G., 1931, Arch. wissensch. Bot., 13, 221; 1932, 17, 650.",
    "Harder, R., 1915, Jahrb. wissensch. Bot., 56, 254. 1923, Z. Bot., 15, 305.",
    "Spoehr, H. A., 1926, Photosynthesis, New York, The Chemical Catalog Co.",
    "Warburg, O., 1919, Biochem. Z., Berlin, 100, 230; 1924, 152, 51; 1925, 166, 386.",
]


def test_later_works_of_one_byline_are_split_out_with_the_byline():
    strings, dois, added = _split_same_byline_works(_OLD_STYLE_LIST, [None] * 5)

    assert strings == [
        "Allen, E. J., 1914, J. Marine Biol. Assn., 10, 417.",
        "Ehrke, G., 1931, Arch. wissensch. Bot., 13, 221",
        "Ehrke, G., 1932, 17, 650.",
        "Harder, R., 1915, Jahrb. wissensch. Bot., 56, 254",
        "Harder, R., 1923, Z. Bot., 15, 305.",
        "Spoehr, H. A., 1926, Photosynthesis, New York, The Chemical Catalog Co.",
        "Warburg, O., 1919, Biochem. Z., Berlin, 100, 230",
        "Warburg, O., 1924, 152, 51",
        "Warburg, O., 1925, 166, 386.",
    ]
    assert dois == [None] * 9
    assert added == 4


def test_no_link_dois_stay_absent():
    strings, dois, added = _split_same_byline_works(_OLD_STYLE_LIST, None)

    assert len(strings) == 9
    assert dois is None
    assert added == 4


@pytest.mark.parametrize(
    "entry",
    [
        # Years going back: not an author's works in date order.
        "Ehrke, G., 1931, Arch. wissensch. Bot., 13, 221; 1912, 17, 650.",
        # A DOI or link belongs to the entry as printed.
        "Ehrke, G., 1931, Arch. wissensch. Bot., 13, 221; 1932, 17, 650. doi:10.1000/x",
        # A year after a journal abbreviation, not after a page.
        "Ehrke, G., 1931, Ber. d. bot. Ges. 1932, 17, 650.",
        # A year with nothing printed after it.
        "Ehrke, G., 1931, Arch. wissensch. Bot., 13, 221; 1932, n.p.",
    ],
)
def test_entries_that_are_one_work_stay_whole(entry):
    strings, _, added = _split_same_byline_works(
        [*_OLD_STYLE_LIST[:1], entry, *_OLD_STYLE_LIST[3:]], None
    )

    assert entry in strings
    assert added == 2  # Warburg only


def test_lists_in_another_style_are_left_alone():
    modern = [
        "Smith, J. (2019). A study. Journal, 1, 2.",
        "Brown, K. (2020). Another study. Journal, 3, 4.",
        "Ehrke, G., 1931, Arch. wissensch. Bot., 13, 221; 1932, 17, 650.",
    ]

    assert _split_same_byline_works(modern, None) == (modern, None, 0)


# --- "same author" dash entries run on inside a segment ------------------------

# Segments as the 948ffec capture's cascade and line stream give them (texts
# from the scan's OCR, some entries shortened).
_DASH_SEGMENTS = [
    "Beadle, G. W. 1932. The relation of crossing-over to chromosome association."
    " Genetics 17: 481-501.",
    "Belling, J. 1928. Nodes and chiasmas in the bivalents of Lilium. Biol. Bull. 54:"
    " 465-70.\n-1931. Chiasmas in flowering plants. Univ. Cal. Pub. Bot. 16: 311-38\n"
    "-1933. Crossing-over and gene rearrangements in flowering plants. Genetics 18: 388-112.",
    "-1931a. Meiosis in diploid and tetraploid Primula sinensis . Jour. Genet. 24: 65-96."
    " -1931b. The cytological theory of inheritance in Oenothera . Jour. Genet. 24: 405-74.",
    "-1932c. The origin and behaviour of chiasmata VI. Hyacinthus amethystinus. Biol. Bull."
    " 63: 368-71.- 1933. The origin and behaviour of chiasmata VIII . Secale cereale."
    " Cytologia 4: 444-52.",
    "- and Dark, S. O. S. 1932. Origin and behaviour of chiasmata II. Stenobothrus"
    " parallelus. Cytologia 3: 169-85. - and Janaki-Ammal, E. K. 1932. Origin and behaviour"
    " of chiasmata I. Diploid and tetraploid Tulipa. Bot. Gaz. 18: 296-312.",
    "Dobzhansky, T. 1933. Studies in chromosome conjugation. II. Z.I.A.V. 64: 269-309.",
]


def test_dash_entries_run_on_inside_a_segment_are_split_out():
    strings, dois, added = _split_inline_dash_entries(_DASH_SEGMENTS, [None] * 6)

    assert strings == [
        _DASH_SEGMENTS[0],
        "Belling, J. 1928. Nodes and chiasmas in the bivalents of Lilium. Biol. Bull. 54: 465-70.",
        "-1931. Chiasmas in flowering plants. Univ. Cal. Pub. Bot. 16: 311-38",
        "-1933. Crossing-over and gene rearrangements in flowering plants. Genetics 18: 388-112.",
        "-1931a. Meiosis in diploid and tetraploid Primula sinensis . Jour. Genet. 24: 65-96.",
        "-1931b. The cytological theory of inheritance in Oenothera . Jour. Genet. 24: 405-74.",
        "-1932c. The origin and behaviour of chiasmata VI. Hyacinthus amethystinus. Biol."
        " Bull. 63: 368-71.",
        "- 1933. The origin and behaviour of chiasmata VIII . Secale cereale. Cytologia 4: 444-52.",
        "- and Dark, S. O. S. 1932. Origin and behaviour of chiasmata II. Stenobothrus"
        " parallelus. Cytologia 3: 169-85.",
        "- and Janaki-Ammal, E. K. 1932. Origin and behaviour of chiasmata I. Diploid and"
        " tetraploid Tulipa. Bot. Gaz. 18: 296-312.",
        _DASH_SEGMENTS[5],
    ]
    assert dois == [None] * 11
    assert added == 5


def test_lists_without_dash_entry_lines_are_left_alone():
    # A GOST-style list separates its fields with dashes, some before a year.
    gost = [
        "1. Mitreikin, N. A. Reliability and testing of radio parts. – M., 1981. – 272 p.",
        "2. Bondarenko, I. B. Electrical radio elements. – SPb., 2015. – Part 1. – 2012. – 108 p.",
        "3. Karpenko, V. V. Studies of insulation systems. Fundamental problems. – 2015. Vol. 5.",
    ]

    assert _split_inline_dash_entries(gost, None) == (gost, None, 0)


@pytest.mark.parametrize(
    "entry",
    [
        # A DOI or link belongs to the entry as printed.
        "Sax, K. 1932. Crossing over. J. Arn. Arb. 13: 180-212. -1934. Interlocking."
        " Am. Nat. 68: 113-56. doi:10.1086/280576",
        # A piece too short to be an entry.
        "Sax, K. 1932. The cytological mechanism of crossing over. J. Arn. Arb. 13: 180-212."
        " -1934. A.",
    ],
)
def test_linked_or_fragment_entries_stay_whole(entry):
    strings, _, added = _split_inline_dash_entries([*_DASH_SEGMENTS, entry], None)

    assert strings[-1] == entry
    assert added == 5


# The printed rows behind those segments: the cascade ran the two dash rows
# after Belling's into his entry; the other dash entries run on inside a row.
_DASH_ROWS = [
    _DASH_SEGMENTS[0],
    "Belling, J. 1928. Nodes and chiasmas in the bivalents of Lilium. Biol. Bull. 54: 465-70.",
    "-1931. Chiasmas in flowering plants. Univ. Cal. Pub. Bot. 16: 311-38",
    "-1933. Crossing-over and gene rearrangements in flowering plants. Genetics 18: 388-112.",
    *_DASH_SEGMENTS[2:],
]
_CASCADE_GROUPS = [[0], [1, 2, 3], [4], [5], [6], [7]]


async def test_parser_gets_the_dash_entries_the_cascade_ran_on():
    """The selected segmentation's segments stay; the parser gets one entry per work."""
    sentences = []
    summaries = []
    for index, text in enumerate(_DASH_ROWS):
        bbox = (45.0, 100.0 + 100 * index, 560.0, 180.0 + 100 * index)
        sentences.append(
            PaperSentence(
                text_id=index,
                text=text,
                section_id=1,
                paragraph_id=index,
                page_number=1,
                provenance=[Provenance(page_no=1, bbox=bbox)],
                region_meta={"region_type": "reference_content"},
            )
        )
        summaries.append(RegionSummary(page=1, index=index, label="reference_content", bbox=bbox))
    contents = PaperContents(
        sentences=sentences,
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(1, "Literature Cited", 1, 0, CanonicalSection.REFERENCES),
        ],
        tables=[],
        links=[],
        sections_text={},
        region_summaries=summaries,
        ref_page_lines=[],
    )
    contents.ref_line_geometry = [{"text": "x"}]
    df = contents.sentences_df
    ref_df = df[df["section_id"] == 1]
    extractor = ReferenceExtractor(
        contents, file_hash="h", llm_client=MagicMock(), seg_strategy="geom", parse_strategy="ner"
    )

    def grouped_spans(ref_text: str) -> list[tuple[int, int]]:
        # ref_text joins the rows with "\n".
        starts, position = [], 0
        for row in _DASH_ROWS:
            starts.append(position)
            position += len(row) + 1
        return [
            (starts[group[0]], starts[group[-1]] + len(_DASH_ROWS[group[-1]]))
            for group in _CASCADE_GROUPS
        ]

    segmenter = MagicMock()
    segmenter.segment_spans.side_effect = lambda ref_text, _lines: (
        grouped_spans(ref_text),
        0.99,
        6,
        6,
    )
    parsed: list[str] = []

    def parse_batch(texts: list[str]) -> list[dict]:
        parsed.extend(texts)
        return [{"title": text, "authors": "A"} for text in texts]

    parser = MagicMock()
    parser.parse_batch.side_effect = parse_batch
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=segmenter),
        patch("bibr.extract.ref_extractor._get_ner_parser", return_value=parser),
        patch("bibr.extract.ref_extractor.build_line_stream", return_value=None),
    ):
        refs = await extractor.extract(ref_df)

    receipt = contents.reference_yield_receipt
    assert [a.strategy for a in receipt.attempts if a.selected] == ["geom"]
    assert "inline_dash_entries_split" in receipt.reason_flags
    assert parsed == _split_inline_dash_entries(_DASH_SEGMENTS, None)[0]
    assert len(refs) == 11


# --- numbered entries holding several works ------------------------------------

# Parser input from the 948ffec capture (the scan's OCR as printed).
_NUMBERED_SEGMENTS = [
    "[16] F. Barbero, From Euclidean to Lorentzian general relativity: the real way, Phys."
    " Rev., D54 (1996), 1492-1499; T. Thiemann, Reality conditions inducing transforms for"
    " quantum gauge theory and quantum gravity, Class, and Quant.",
    "Grav., 13 (1996), 1383-1404.",
    "[17] J.W. Bardeen, B. Carter, and S.W. Hawking, The four laws of black hole mechanics,"
    " Commun. Math. Phys., 31 (1973), 161-170.",
    "[18] J.D. Bekenstein, Black holes and entropy, Phys. Rev., D7 (1973), 2333-2346.\nJ.D."
    " Bekenstein, Generalized second law of thermodynamics in black hole physics,",
    "Phys. Rev., D9 (1974), 3292-3300.",
    "[19] R. Gambini, 0. Obregon, and J. Pullin, Yang-Mills analogs of the Immirzi"
    " ambiguity, Phys. Rev., D59 (1999), 047505.",
    "[20] J.F. Plebanski, J. Math. Phys., 18 (1977), 2511; J. Samuel, Pra-mana J.",
    "Phys., 28 (1987), L429; T. Jacobson and L. Smolin, Phys. Lett., B196 (1987), 39.",
    "[21] T. Regge and C. Teitelboim, Role ofsurf ace integrals in the Hamiltonian"
    " formulation of general relativity, Annals Phys., 88 (1974), 286.",
]


def test_numbered_entries_are_made_whole_then_split_into_their_works():
    strings, dois, joined, added = _split_numbered_entry_works(_NUMBERED_SEGMENTS, [None] * 9)

    assert strings == [
        "[16] F. Barbero, From Euclidean to Lorentzian general relativity: the real way, Phys."
        " Rev., D54 (1996), 1492-1499",
        "T. Thiemann, Reality conditions inducing transforms for quantum gauge theory and"
        " quantum gravity, Class, and Quant. Grav., 13 (1996), 1383-1404.",
        _NUMBERED_SEGMENTS[2],
        "[18] J.D. Bekenstein, Black holes and entropy, Phys. Rev., D7 (1973), 2333-2346.",
        "J.D. Bekenstein, Generalized second law of thermodynamics in black hole physics, Phys."
        " Rev., D9 (1974), 3292-3300.",
        _NUMBERED_SEGMENTS[5],
        "[20] J.F. Plebanski, J. Math. Phys., 18 (1977), 2511",
        "J. Samuel, Pra-mana J. Phys., 28 (1987), L429",
        "T. Jacobson and L. Smolin, Phys. Lett., B196 (1987), 39.",
        _NUMBERED_SEGMENTS[8],
    ]
    assert dois == [None] * 10
    assert (joined, added) == (3, 4)


@pytest.mark.parametrize(
    "entry",
    [
        # Initials inside one work: editors, an OCR "0." for "O.", a preprint.
        "[9] A. Ashtekar and K. Krasnov, Quantum geometry and black holes, in 'Black holes,"
        " gravitational radiation and the Universe', B. Bhawal and B.R. Iyer eds., Kluwer,"
        " Dordrecht, (1998), 149-170; available as gr-qc/9804039.",
        "[9] R. Gambini, 0. Obregon, and J. Pullin, Yang-Mills analogs of the Immirzi"
        " ambiguity, Phys. Rev., D59 (1999), 047505.",
        "[9] A. Ashtekar, C. Beetle, and S. Fairhurst, Mechanics of isolated horizons, Class."
        " Quantum Grav., 17 (2000), 253-298; gr-qc/9907068.",
        # An editor's byline after a series number.
        "[9] A. Ashtekar, Quantum geometry, in Lecture Notes in Physics 541. J. Kowalski-Glikman,"
        " ed., Springer, Berlin, 2000, 1-20.",
        # A DOI belongs to the entry as printed.
        "[9] A. Ashtekar, New variables, Phys. Rev. Lett., 57 (1986), 2244. A. Ashtekar, New"
        " Hamiltonian formulation, Phys. Rev., D36 (1987), 1587. doi:10.1103/x",
        # A later piece without a number is no work of its own.
        "[9] A. Ashtekar, New variables, Phys. Rev. Lett., 57 (1986), 2244. A. Ashtekar,"
        " private communication.",
    ],
)
def test_numbered_entries_that_are_one_work_stay_whole(entry):
    segments = [*_NUMBERED_SEGMENTS[2:3], entry, *_NUMBERED_SEGMENTS[5:6]]

    assert _split_numbered_entry_works(segments, None) == (segments, None, 0, 0)


_OPEN_ENTRY = (
    "[4] A. Ashtekar and A. Magnon, Asymptotically anti-de Sitter Space-times, Class. Quant."
)


@pytest.mark.parametrize(
    ("entry", "segment", "next_number"),
    [
        # The numbering skips: the segment may be the entry that lost its number.
        (_OPEN_ENTRY, "Phys. Rev., D9 (1974), 3292-3300.", "[6]"),
        # A segment with a byline of its own is a work, not a fragment.
        (
            _OPEN_ENTRY,
            "A. Ashtekar and J.D. Romano, Spatial infinity as a boundary of space-time, Class."
            " Quant. Grav., 9 (1992), 1069-1100.",
            "[5]",
        ),
        (
            _OPEN_ENTRY,
            "Romano, J.D., Spatial infinity, Class. Quant. Grav., 9 (1992), 1069-1100.",
            "[5]",
        ),
        # scans_ci20 W2133176560: the OCR read "[5]" as "[S]", and the list
        # prints [5] before [4].
        (
            "[3] DEMUROV (D. G.), VENEVTSEV (Yu. N.), K~ist~lgraphiya, 1971, 16, 168.",
            "[S] GALASSO (F. S.), Structure, Properties and Preparationof Perovskite-Type"
            " Compounds. Pergamon PressOxford, 1969.",
            "[4]",
        ),
        (
            "[3] DEMUROV (D. G.), VENEVTSEV (Yu. N.), Kristallographiya,",
            "[S] GALASSO (F. S.), Structure, Properties and Preparationof Perovskite-Type"
            " Compounds. Pergamon PressOxford, 1969.",
            "[4]",
        ),
        # audience_eval200 W4379468209: the entry above is whole; the fragment
        # ends the entry before it, which the reading order put earlier.
        (
            "[66] Y. Li, X. Huang, and G. Zhao, \u201cMicro-expression action unit detection with"
            " spatial and channel attention,\u201d Neurocomputing, vol. 436, pp. 221\u2013231,"
            " May 2021.",
            "Int. Conf. Multimedia, Oct. 2020, pp. 2237\u20132245.",
            "[67]",
        ),
    ],
)
def test_segments_that_may_be_entries_stay_apart(entry, segment, next_number):
    segments = [
        entry,
        segment,
        _NUMBERED_SEGMENTS[2].replace("[17]", next_number),
        _NUMBERED_SEGMENTS[5].replace("[19]", "[7]"),
    ]

    assert _split_numbered_entry_works(segments, None) == (segments, None, 0, 0)


def test_a_fragment_joins_an_open_entry():
    segments = [
        _OPEN_ENTRY,
        "Grav., 1 (1984), L39-L41.",
        _NUMBERED_SEGMENTS[2].replace("[17]", "[5]"),
        _NUMBERED_SEGMENTS[5].replace("[19]", "[6]"),
    ]

    strings, _, joined, added = _split_numbered_entry_works(segments, None)

    assert strings[0] == f"{_OPEN_ENTRY} Grav., 1 (1984), L39-L41."
    assert strings[1:] == segments[2:]
    assert (joined, added) == (1, 0)


def test_unnumbered_lists_are_left_alone():
    assert _split_numbered_entry_works(_DASH_SEGMENTS, None) == (_DASH_SEGMENTS, None, 0, 0)
    assert _split_numbered_entry_works(_OLD_STYLE_LIST, None) == (_OLD_STYLE_LIST, None, 0, 0)


async def test_parser_gets_the_works_of_numbered_entries():
    """The selected segmentation's segments stay; the parser gets one entry per work."""
    sentences = []
    summaries = []
    for index, text in enumerate(_NUMBERED_SEGMENTS):
        bbox = (45.0, 100.0 + 100 * index, 560.0, 180.0 + 100 * index)
        sentences.append(
            PaperSentence(
                text_id=index,
                text=text,
                section_id=1,
                paragraph_id=index,
                page_number=1,
                provenance=[Provenance(page_no=1, bbox=bbox)],
                region_meta={"region_type": "reference_content"},
            )
        )
        summaries.append(RegionSummary(page=1, index=index, label="reference_content", bbox=bbox))
    contents = PaperContents(
        sentences=sentences,
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(1, "References", 1, 0, CanonicalSection.REFERENCES),
        ],
        tables=[],
        links=[],
        sections_text={},
        region_summaries=summaries,
        ref_page_lines=[],
    )
    contents.ref_line_geometry = [{"text": "x"}]
    df = contents.sentences_df
    ref_df = df[df["section_id"] == 1]
    extractor = ReferenceExtractor(
        contents, file_hash="h", llm_client=MagicMock(), seg_strategy="geom", parse_strategy="ner"
    )

    def row_spans(ref_text: str) -> list[tuple[int, int]]:
        # ref_text joins the rows with "\n"; each row is one segment.
        spans, position = [], 0
        for row in _NUMBERED_SEGMENTS:
            spans.append((position, position + len(row)))
            position += len(row) + 1
        return spans

    segmenter = MagicMock()
    segmenter.segment_spans.side_effect = lambda ref_text, _lines: (
        row_spans(ref_text),
        0.99,
        9,
        9,
    )
    parsed: list[str] = []

    def parse_batch(texts: list[str]) -> list[dict]:
        parsed.extend(texts)
        return [{"title": text, "authors": "A"} for text in texts]

    parser = MagicMock()
    parser.parse_batch.side_effect = parse_batch
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=segmenter),
        patch("bibr.extract.ref_extractor._get_ner_parser", return_value=parser),
        patch("bibr.extract.ref_extractor.build_line_stream", return_value=None),
    ):
        refs = await extractor.extract(ref_df)

    receipt = contents.reference_yield_receipt
    assert [a.strategy for a in receipt.attempts if a.selected] == ["geom"]
    assert {"numbered_fragments_joined", "numbered_works_split"} <= set(receipt.reason_flags)
    # The parser reads each entry without its list number.
    works = _split_numbered_entry_works(_NUMBERED_SEGMENTS, None)[0]
    assert parsed == _strip_enum_markers(works)
    assert parsed[:2] == [
        "F. Barbero, From Euclidean to Lorentzian general relativity: the real way, Phys. Rev.,"
        " D54 (1996), 1492-1499",
        "T. Thiemann, Reality conditions inducing transforms for quantum gauge theory and"
        " quantum gravity, Class, and Quant. Grav., 13 (1996), 1383-1404.",
    ]
    assert len(refs) == 10
