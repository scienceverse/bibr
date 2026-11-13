"""Front-matter candidate collection on degenerate layouts. All text is invented."""

from __future__ import annotations

from bibr.extract import front_matter
from bibr.extract.front_matter import resolve_front_matter
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)


def _dense_page(count: int) -> PaperContents:
    """One page of *count* small text regions, each its own paragraph."""

    sentences = []
    summaries = []
    for index in range(count):
        bbox = (
            index * 0.5,
            float(index % 50) * 10.0,
            index * 0.5 + 4.0,
            float(index % 50) * 10.0 + 8.0,
        )
        sentences.append(
            PaperSentence(
                text_id=index,
                text=f"Cell{index} value",
                section_id=0,
                paragraph_id=index,
                page_number=1,
                provenance=[Provenance(page_no=1, bbox=bbox)],
                region_meta={"region_type": "text"},
            )
        )
        summaries.append(
            RegionSummary(
                page=1,
                index=index,
                label="text",
                bbox=bbox,
                section_id=0,
                font_size=float(index),
            )
        )
    root = PaperSection(
        section_id=0,
        header="Root",
        level=0,
        parent_section_id=None,
        section_type=CanonicalSection.UNKNOWN,
    )
    return PaperContents(
        sentences=sentences,
        sections=[root],
        tables=[],
        links=[],
        sections_text={0: ""},
        region_summaries=summaries,
    )


def test_region_lookup_does_not_compare_every_region_on_the_page(monkeypatch):
    # Each sentence looks its layout region up by geometry. Comparing it with
    # every region of its page made a page of a few thousand small regions
    # quadratic (2,000 regions took about 20 s, 4,000 over two minutes).
    count = 400
    calls = 0
    same_bbox = front_matter._same_bbox

    def counting(left, right):
        nonlocal calls
        calls += 1
        return same_bbox(left, right)

    monkeypatch.setattr(front_matter, "_same_bbox", counting)
    contents = _dense_page(count)

    resolution, _ = resolve_front_matter(contents)

    assert len(resolution.candidates) == count
    assert calls <= 10 * count


def test_region_lookup_keeps_matching_each_row_to_its_own_region():
    contents = _dense_page(60)

    candidates = front_matter.collect_front_matter_candidates(contents)

    # The font size is read from the matched region only.
    by_text = {candidate.raw_text: candidate for candidate in candidates}
    assert [by_text[f"Cell{index} value"].font_size for index in range(60)] == [
        float(index) for index in range(60)
    ]
