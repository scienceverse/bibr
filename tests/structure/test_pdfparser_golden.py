"""Golden-output snapshot tests for PDFParser.

These run a full parse against synthetic OCR JSON fixtures that exercise
key parser invariants (carry-over, captions, footnotes) and assert the
resulting PaperContents shape is byte-identical across the refactor.

The first run generates snapshots; subsequent runs compare. Used as the
safety net for Plan 7 (PDFParser state extraction).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bibr.structure.pdf_parser import PDFParser

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _region(index, label, content, bbox=None, native_label=None):
    d = {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox or [0, 0, 100, 100],
    }
    if native_label is not None:
        d["native_label"] = native_label
    return d


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


def _parse_full(json_result):
    parser = PDFParser(json_result)
    contents = parser.parse()
    texts = [text for text, _, _, needs_seg, _ in parser._deferred_texts if needs_seg]
    segments = [_split_one(t) for t in texts]
    parser.apply_segmentation(contents, segments)
    parser.create_content_sections(contents)
    return contents


def _snapshot_contents(contents) -> dict:
    """Reduce a PaperContents to a comparable dict (deterministic)."""
    sections = [
        {
            "section_id": s.section_id,
            "header": s.header,
            "level": s.level,
            "parent_section_id": s.parent_section_id,
            "section_type": getattr(s.section_type, "value", s.section_type)
            if s.section_type is not None
            else None,
        }
        for s in contents.sections
    ]
    sentences = [
        {
            "text_id": s.text_id,
            "section_id": s.section_id,
            "text": s.text,
        }
        for s in (contents.sentences or [])
    ]
    tables = [
        {
            "table_id": t.table_id,
            "caption": t.caption,
            "section_id": t.section_id,
        }
        for t in contents.tables
    ]
    figures = [
        {
            "figure_id": f.figure_id,
            "caption": f.caption,
            "section_id": f.section_id,
        }
        for f in contents.figures
    ]
    return {
        "sections": sections,
        "sentences": sentences,
        "tables": tables,
        "figures": figures,
        "detected_title": contents.detected_title,
        "detected_headers": list(contents.detected_headers),
        "detected_footers": list(contents.detected_footers),
    }


# ---------------------------------------------------------------------------
# Fixtures — constructed inline so the test is hermetic.
# ---------------------------------------------------------------------------


def _fixture_basic():
    """A simple paper with title, two sections, and body text."""
    return [
        [
            _region(0, "doc_title", "My Paper"),
            _region(1, "paragraph_title", "Introduction"),
            _region(2, "text", "First sentence. Second sentence."),
            _region(3, "paragraph_title", "Methods"),
            _region(4, "text", "Method sentence one. Method sentence two."),
        ]
    ]


def _fixture_carry_over():
    """A paragraph that spans two pages — carry-over invariant."""
    return [
        [
            _region(0, "doc_title", "Carry Over Paper"),
            _region(1, "paragraph_title", "Body"),
            _region(2, "text", "This paragraph starts on page one and"),
        ],
        [
            _region(0, "text", "continues onto page two with more text."),
        ],
    ]


def _fixture_with_table_and_figure():
    """A paper with a table caption, a table, a figure caption, and a figure."""
    return [
        [
            _region(0, "doc_title", "Tabled Paper"),
            _region(1, "paragraph_title", "Results"),
            _region(2, "figure_title", "Table 1: Demographics", bbox=[0, 100, 200, 120]),
            _region(3, "table", "| A | B |\n| 1 | 2 |", bbox=[0, 130, 200, 200]),
            _region(4, "figure_title", "Figure 1: Plot", bbox=[0, 300, 200, 320]),
            _region(5, "image", "", bbox=[0, 330, 200, 400]),
        ]
    ]


def _fixture_with_footnote():
    """A paper with an inline footnote that should be relocated."""
    return [
        [
            _region(0, "doc_title", "Footnoted Paper"),
            _region(1, "paragraph_title", "Body"),
            _region(2, "text", "Body text containing a footnote reference."),
            _region(3, "footnote", "1. This is the footnote text."),
        ]
    ]


_FIXTURES = {
    "basic": _fixture_basic,
    "carry_over": _fixture_carry_over,
    "table_and_figure": _fixture_with_table_and_figure,
    "footnote": _fixture_with_footnote,
}


@pytest.mark.parametrize("name", list(_FIXTURES.keys()))
def test_pdfparser_output_unchanged(name):
    """PDFParser output must match its pre-refactor snapshot byte-for-byte."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_path = SNAPSHOT_DIR / f"{name}.snap.json"

    contents = _parse_full(_FIXTURES[name]())
    snap = _snapshot_contents(contents)

    if not snap_path.exists():
        snap_path.write_text(json.dumps(snap, indent=2, sort_keys=True))
        pytest.skip(f"snapshot baseline written: {snap_path.name} — re-run to compare")

    expected = json.loads(snap_path.read_text())
    assert snap == expected, (
        f"PDFParser output diverged from snapshot {snap_path.name}.\n"
        f"This means the refactor changed observable behavior."
    )
