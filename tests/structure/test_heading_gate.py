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
