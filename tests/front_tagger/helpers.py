"""Build tagger inputs and hand-labelled tags for the assembler tests."""

from __future__ import annotations

from bibr.extract.front_role_features import FrontRegion
from bibr.front_tagger.contract import TaggerInput, build_tagger_input


def regions(*rows: tuple[str, str], page: int = 0) -> list[FrontRegion]:
    """One region per ``(layout_label, text)`` row, stacked down the page."""
    step = 1.0 / (len(rows) + 1)
    return [
        FrontRegion(
            page=page,
            index=i,
            label=label,
            text=text,
            x0=0.1,
            y0=i * step,
            x1=0.9,
            y1=i * step + step * 0.8,
        )
        for i, (label, text) in enumerate(rows)
    ]


def tagger_input(*rows: tuple[str, str]) -> TaggerInput:
    return build_tagger_input(regions(*rows))


def label(inp: TaggerInput, *spans: tuple[str, str]) -> list[str]:
    """BIO tags: each ``(field, phrase)`` labels the words of the phrase's next
    occurrence in the input (searching on from the previous span of any field)."""
    tags = ["O"] * len(inp.words)
    cursor = 0
    for field, phrase in spans:
        start = inp.text.index(phrase, cursor)
        end = start + len(phrase)
        inside = [i for i, w in enumerate(inp.words) if w.start >= start and w.end <= end]
        assert inside, (field, phrase)
        for n, i in enumerate(inside):
            tags[i] = f"{'B' if n == 0 else 'I'}-{field}"
        cursor = end
    return tags
