"""Parser wiring for the PDF-outline heading-hierarchy signal.

When ``Settings.pipeline.outline_headings`` is on and an outline is threaded
into ``PDFParser``, confidently matched headings take the bookmark's
(compressed) level, overriding the numbering inference from
``_handle_heading``. The assigned levels are marked authoritative so the later
``assign_hierarchy_from_top_level`` pass leaves them intact. With the setting
off, or no outline, behavior is bit-identical to today.
"""

from __future__ import annotations

import pytest

from bibr.config import Settings
from bibr.input.pdf_outline import OutlineItem
from bibr.structure.pdf_parser import PDFParser
from bibr.structure.section_tree import assign_hierarchy_from_top_level


def _title(content: str, y: int = 80) -> dict:
    return {"label": "doc_title", "content": content, "bbox_2d": [50, y, 500, y + 30]}


def _heading(content: str, y: int = 200) -> dict:
    return {"label": "paragraph_title", "content": content, "bbox_2d": [50, y, 500, y + 30]}


def _text(content: str, y: int = 300) -> dict:
    return {"label": "text", "content": content, "bbox_2d": [50, y, 500, y + 50]}


def _pages() -> list[list[dict]]:
    return [
        [
            _title("My Paper Title"),
            _heading("1 Introduction", y=150),
            _text("Intro body sentence here."),
            _heading("Methods", y=400),
            _text("Methods body sentence here."),
        ],
        [
            _heading("2.1 Detailed Analysis", y=150),
            _text("Analysis body sentence here."),
        ],
    ]


def _outline() -> list[OutlineItem]:
    return [
        OutlineItem(title="Introduction", level=0, page_no=1),
        OutlineItem(title="Methods", level=0, page_no=1),
        OutlineItem(title="Detailed Analysis", level=1, page_no=2),
    ]


def _by_header(sections):
    return {s.header: s for s in sections}


def test_outline_overrides_numbering_and_is_protected(monkeypatch):
    monkeypatch.setattr(Settings.pipeline, "outline_headings", True)
    parser = PDFParser(json_result=_pages(), outline=_outline())
    contents = parser.parse()
    secs = _by_header(contents.sections)

    intro = secs["1 Introduction"]
    methods = secs["Methods"]
    analysis = secs["2.1 Detailed Analysis"]

    # Raw numbering inference would give 2 / 2 / 3; the outline compresses the
    # matched depths {0, 1} to contiguous {1, 2}.
    assert intro.level == 1
    assert methods.level == 1
    assert analysis.level == 2

    # Matched headings are flagged authoritative.
    assert intro.outline_level_authoritative is True
    assert methods.outline_level_authoritative is True
    assert analysis.outline_level_authoritative is True

    # The later hierarchy pass must not clobber the outline levels.
    assign_hierarchy_from_top_level(contents.sections)
    assert intro.level == 1
    assert methods.level == 1
    assert analysis.level == 2

    # Parent coherence: a level-2 section points at a lower-level section.
    by_id = {s.section_id: s for s in contents.sections}
    parent = by_id[analysis.parent_section_id]
    assert parent.level < analysis.level


def test_setting_off_is_bit_identical(monkeypatch):
    monkeypatch.setattr(Settings.pipeline, "outline_headings", False)
    # Even if an outline is threaded, the gate keeps it inert.
    parser = PDFParser(json_result=_pages(), outline=_outline())
    contents = parser.parse()
    secs = _by_header(contents.sections)

    # Numbering inference stands untouched.
    assert secs["1 Introduction"].level == 2
    assert secs["Methods"].level == 2
    assert secs["2.1 Detailed Analysis"].level == 3
    assert secs["1 Introduction"].outline_level_authoritative is False


def test_no_outline_is_bit_identical(monkeypatch):
    monkeypatch.setattr(Settings.pipeline, "outline_headings", True)
    parser = PDFParser(json_result=_pages(), outline=None)
    contents = parser.parse()
    secs = _by_header(contents.sections)
    assert secs["1 Introduction"].level == 2
    assert secs["Methods"].level == 2
    assert secs["2.1 Detailed Analysis"].level == 3


def test_unmatched_headings_keep_current_behavior(monkeypatch):
    monkeypatch.setattr(Settings.pipeline, "outline_headings", True)
    # Outline only covers Introduction; Methods/Analysis keep numbering levels.
    outline = [OutlineItem(title="Introduction", level=0, page_no=1)]
    parser = PDFParser(json_result=_pages(), outline=outline)
    contents = parser.parse()
    secs = _by_header(contents.sections)

    assert secs["1 Introduction"].level == 1  # matched, compressed to 1
    assert secs["1 Introduction"].outline_level_authoritative is True
    # Unmatched: original numbering-derived levels preserved.
    assert secs["Methods"].level == 2
    assert secs["2.1 Detailed Analysis"].level == 3
    assert secs["Methods"].outline_level_authoritative is False


@pytest.mark.parametrize("setting_on", [True, False])
def test_default_construction_still_works(monkeypatch, setting_on):
    # PDFParser(json_result=...) without the outline kwarg must behave exactly
    # as before regardless of the setting.
    monkeypatch.setattr(Settings.pipeline, "outline_headings", setting_on)
    parser = PDFParser(json_result=_pages())
    contents = parser.parse()
    secs = _by_header(contents.sections)
    assert secs["Methods"].level == 2
