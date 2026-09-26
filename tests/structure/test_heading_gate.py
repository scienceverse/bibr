"""Heading plausibility gate (chunk 2).

A ``paragraph_title`` region that is really a reference lead-in, a short label
fragment, OCR digit-garbage, a table/figure caption, or a full sentence must be
demoted to ordinary content instead of becoming a spurious section. Genuine
headings ("Conclusion.", "7 Conclusion", "Ethics Statement") stay sections.
"""

from __future__ import annotations

import pytest


def _split_one(text):
    sentences = []
    current = ""
    for ch in text:
        current += ch
        if ch == ".":
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)
    return sentences


def _region(index, label, content, bbox=None, **meta):
    return {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox or [0, 0, 100, 100],
        **meta,
    }


def _parse(json_result):
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(json_result)
    contents = parser.parse()
    texts = [text for text, _, _, needs_seg, _ in parser._deferred_texts if needs_seg]
    segments = [_split_one(t) for t in texts]
    parser.apply_segmentation(contents, segments)
    parser.create_content_sections(contents)
    return contents


def _headers_for(candidate: str) -> list[str]:
    json_result = [
        [
            _region(0, "doc_title", "Paper Title"),
            _region(1, "text", "Opening sentence."),
            _region(2, "paragraph_title", candidate),
            _region(3, "text", "Body sentence."),
        ]
    ]
    return [s.header for s in _parse(json_result).sections]


DEMOTED = [
    "Breiman [2001]:",
    "Best case.",
    "Data access.",
    "111115555557799991111",
    "Table 2 Caption",
    "Table 2: Some caption text here",
    "A very long sentence-like candidate that runs past twelve words and ends with a period.",
]

KEPT = [
    "Conclusion.",
    "7 Conclusion",
    "Ethics Statement",
    "Appendix A",
]


@pytest.mark.parametrize("candidate", DEMOTED)
def test_implausible_heading_demoted_to_content(candidate):
    assert candidate not in _headers_for(candidate)


@pytest.mark.parametrize("candidate", KEPT)
def test_plausible_heading_kept_as_section(candidate):
    assert candidate in _headers_for(candidate)


def test_demoted_heading_text_is_not_dropped():
    # The gate must route to content, never drop the text.
    json_result = [
        [
            _region(0, "doc_title", "Paper Title"),
            _region(1, "paragraph_title", "Best case."),
        ]
    ]
    contents = _parse(json_result)
    all_text = " ".join(s.text for s in contents.sentences)
    assert "Best case" in all_text


def test_implausible_heading_uses_terminal_body_emission(monkeypatch):
    """A rejected heading cannot re-enter content-heading promotion."""
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(
        [[_region(0, "doc_title", "Paper Title"), _region(1, "paragraph_title", "Best case.")]]
    )

    def fail_if_reclassified(*_args, **_kwargs):
        raise AssertionError("demoted heading re-entered heading classification")

    monkeypatch.setattr(parser, "_promotable_content_heading", fail_if_reclassified)
    contents = parser.parse()
    texts = [text for text, _, _, needs_seg, _ in parser._deferred_texts if needs_seg]
    parser.apply_segmentation(contents, [_split_one(text) for text in texts])

    assert [section.header for section in contents.sections].count("Best case.") == 0
    assert any(sentence.text == "Best case." for sentence in contents.sentences)


def test_trusted_content_alias_promotes_exactly_once():
    headers = _headers_for_text_region("Methods")
    assert headers.count("Methods") == 1


def _headers_for_text_region(candidate: str, **meta) -> list[str]:
    """Sections produced when *candidate* arrives as an OCR ``text`` region."""
    json_result = [
        [
            _region(0, "doc_title", "Paper Title"),
            _region(1, "text", "Opening sentence."),
            _region(2, "text", candidate, **meta),
            _region(3, "text", "Body sentence."),
        ]
    ]
    return [s.header for s in _parse(json_result).sections]


class TestBoldContentHeadingRescue:
    """A short bold standalone body row is a heading even when its text
    matches no alias — the rescue for novel or OCR-corrupted headings that
    the layout model mislabeled as ``text``."""

    def test_bold_short_row_promoted_without_alias(self):
        headers = _headers_for_text_region("The Current Research", _font_bold=True, _font_size=11.0)
        assert "The Current Research" in headers

    def test_bold_corrupted_header_fragment_promoted(self):
        # "Particip" — OCR-truncated "Participants": no alias hit, but the
        # bold font signal still identifies it as a heading.
        headers = _headers_for_text_region("Particip", _font_bold=True)
        assert "Particip" in headers

    def test_same_row_without_font_metadata_stays_content(self):
        headers = _headers_for_text_region("The Current Research")
        assert "The Current Research" not in headers

    def test_non_bold_row_stays_content(self):
        headers = _headers_for_text_region("The Current Research", _font_bold=False)
        assert "The Current Research" not in headers

    def test_bold_bullet_row_stays_content(self):
        headers = _headers_for_text_region("- First item of a list", _font_bold=True)
        assert "- First item of a list" not in headers

    def test_bold_sentence_stays_content(self):
        headers = _headers_for_text_region(
            "We recruited participants from nine different universities overall", _font_bold=True
        )
        assert "We recruited participants from nine different universities overall" not in headers

    def test_bold_caption_shaped_row_stays_content(self):
        # The plausibility gate would bounce a caption right back — the bold
        # branch must not promote what the gate would demote.
        headers = _headers_for_text_region("Table 2 Caption", _font_bold=True)
        assert "Table 2 Caption" not in headers


def _page_of(*regions):
    return list(regions)


def _title_region(content, bbox):
    return {"label": "doc_title", "content": content, "bbox_2d": bbox}


def test_split_front_page_title_joins_into_one_section():
    """Two adjacent doc_title regions form one title, not a stray section."""
    from bibr.structure.pdf_parser import PDFParser

    pages = [
        _page_of(
            _title_region("# A Large-Scale Registered Replication of", [100, 60, 900, 100]),
            _title_region("# the Stroop Interference Effect", [100, 105, 900, 140]),
            {"label": "paragraph_title", "content": "## Abstract", "bbox_2d": [100, 160, 900, 190]},
            {
                "label": "paragraph_title",
                "content": "## Introduction",
                "bbox_2d": [100, 300, 900, 330],
            },
        )
    ]
    parser = PDFParser(pages)
    parser.parse()

    assert parser._detected_title == (
        "A Large-Scale Registered Replication of the Stroop Interference Effect"
    )
    title_sections = [s for s in parser.sections if s.level == 1 and s.section_id != 0]
    assert len(title_sections) == 1
    assert title_sections[0].header == parser._detected_title
    intro = next(s for s in parser.sections if s.header == "Introduction")
    assert intro.parent_section_id == title_sections[0].section_id


def test_nonadjacent_second_doc_title_still_opens_a_section():
    """Guard: a far-away second doc_title is not a continuation."""
    from bibr.structure.pdf_parser import PDFParser

    pages = [
        _page_of(
            _title_region("# A Large-Scale Registered Replication of", [100, 60, 900, 100]),
            _title_region("# the Stroop Interference Effect", [100, 600, 900, 640]),
        )
    ]
    parser = PDFParser(pages)
    parser.parse()

    assert parser._detected_title == "A Large-Scale Registered Replication of"
    assert len([s for s in parser.sections if s.section_id != 0]) == 2


def test_text_between_titles_blocks_continuation():
    """Guard: body text emitted between two doc_titles blocks the join."""
    from bibr.structure.pdf_parser import PDFParser

    pages = [
        _page_of(
            _title_region("# A Large-Scale Registered Replication of", [100, 60, 900, 100]),
            {
                "label": "text",
                "content": "An author line sits between.",
                "bbox_2d": [100, 105, 900, 130],
            },
            _title_region("# the Stroop Interference Effect", [100, 135, 900, 170]),
        )
    ]
    parser = PDFParser(pages)
    parser.parse()

    assert parser._detected_title == "A Large-Scale Registered Replication of"
    assert len([s for s in parser.sections if s.section_id != 0]) == 2


def test_three_region_title_joins_on_the_growing_title_box():
    """Each continuation is compared with the merged title box, not the first line."""
    from bibr.structure.pdf_parser import PDFParser

    pages = [
        _page_of(
            _title_region("# A Large-Scale Registered", [100, 60, 900, 100]),
            _title_region("# Replication of the Stroop", [100, 250, 900, 290]),
            _title_region("# Interference Effect", [100, 440, 900, 480]),
        )
    ]
    parser = PDFParser(pages)
    parser.parse()

    assert parser._detected_title == (
        "A Large-Scale Registered Replication of the Stroop Interference Effect"
    )
    title_sections = [s for s in parser.sections if s.section_id != 0]
    assert len(title_sections) == 1
    assert len(title_sections[0].provenance) == 3


def test_masthead_doc_title_is_not_extended_with_the_real_title():
    """A journal masthead captured first stays apart from the real title."""
    from bibr.structure.pdf_parser import PDFParser

    pages = [
        _page_of(
            _title_region(
                "# INTERNATIONAL JOURNAL OF LAW, GOVERNMENT AND COMMUNICATION",
                [319, 137, 658, 223],
            ),
            _title_region("# NAVIGATING DIGITAL DIALOGUE", [115, 276, 883, 340]),
        )
    ]
    parser = PDFParser(pages)
    parser.parse()

    assert parser._detected_title == "INTERNATIONAL JOURNAL OF LAW, GOVERNMENT AND COMMUNICATION"
    assert [s.header for s in parser.sections if s.section_id != 0] == [
        "INTERNATIONAL JOURNAL OF LAW, GOVERNMENT AND COMMUNICATION",
        "NAVIGATING DIGITAL DIALOGUE",
    ]


def test_heading_between_title_regions_blocks_continuation():
    """Guard: once another section opened, a second doc_title is not the title."""
    from bibr.structure.pdf_parser import PDFParser

    pages = [
        _page_of(
            _title_region("# A Large-Scale Registered Replication of", [100, 60, 900, 100]),
            {"label": "paragraph_title", "content": "## Abstract", "bbox_2d": [100, 105, 900, 120]},
            _title_region("# the Stroop Interference Effect", [100, 125, 900, 160]),
        )
    ]
    parser = PDFParser(pages)
    parser.parse()

    assert parser._detected_title == "A Large-Scale Registered Replication of"
