"""RefLocator accepts a front-role ``ref_header`` heading in any language."""

from __future__ import annotations

import pandas as pd

from bibr.extract.front_role import FrontRolePredictions, RoleScores
from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

REFS = [
    "Иванов И. И. (2019). Рост микроводорослей. Биология, 12(3), 45–52.",
    "Петров П. П. (2020). Температурные режимы. Экология, 4(1), 1–9.",
]


def _contents(predictions):
    sections = [
        PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN, []),
        PaperSection(1, "Введение", 1, None, CanonicalSection.INTRODUCTION, []),
        PaperSection(2, "Список литературы", 1, None, CanonicalSection.UNKNOWN, []),
    ]
    sentences = [
        PaperSentence(
            text_id=1,
            text="Текст введения.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
            provenance=[Provenance(page_no=1, bbox=(0.0, 0.0, 10.0, 10.0))],
        )
    ] + [
        PaperSentence(
            text_id=10 + i,
            text=ref,
            section_id=2,
            paragraph_id=10 + i,
            page_number=9,
            provenance=[Provenance(page_no=9, bbox=(0.0, 20.0 * i, 10.0, 20.0 * i + 10.0))],
        )
        for i, ref in enumerate(REFS)
    ]
    contents = PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={s.section_id: "" for s in sections},
        region_summaries=[
            RegionSummary(page=9, index=0, label="paragraph_title", bbox=None, section_id=2),
        ],
        front_role_predictions=predictions,
    )
    return contents


def _predictions(p_ref_header: float) -> FrontRolePredictions:
    probs = {"ref_header": p_ref_header, "heading": 1.0 - p_ref_header}
    top = max(probs, key=probs.get)
    return FrontRolePredictions(
        {(9, 0): RoleScores(probs=probs, top=top, confidence=probs[top])}, model_version="t"
    )


def test_model_ref_header_selects_the_reference_section():
    contents = _contents(_predictions(0.92))
    rows = RefLocator(contents).collect_reference_rows()
    assert isinstance(rows, pd.DataFrame)
    assert list(rows["text"]) == REFS
    assert "front_role_ref_header" in contents.reference_boundary_reason_flags


def test_low_confidence_model_header_does_not_fire():
    contents = _contents(_predictions(0.2))
    locator = RefLocator(contents)
    assert locator._model_reference_header() is None
