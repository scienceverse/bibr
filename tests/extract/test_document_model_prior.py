"""Local article anatomy can outrank a learned heading prior, only in documents."""

from dataclasses import replace

import pytest

from bibr.extract.document_records import refine_document_records
from bibr.extract.front_matter import MODEL_NON_TITLE_SEED_ROLE, group_front_matter_blocks
from tests.extract.test_document_records import _records
from tests.extract.test_document_scope import _resolution


def _denied_record(*, composite=False):
    resolution = _records(composite=composite)
    rows = list(resolution.candidates)
    index = 2 if composite else 3
    rows[index] = replace(
        rows[index],
        roles=rows[index].roles | {MODEL_NON_TITLE_SEED_ROLE},
        model_scores=(("heading", 0.98), ("title", 0.01), ("other", 0.01)),
    )
    return _resolution(rows), index


@pytest.mark.parametrize("composite", [False, True])
def test_own_local_anatomy_outranks_heading_prior_without_erasing_model_evidence(composite):
    original, index = _denied_record(composite=composite)
    assert len(group_front_matter_blocks(original.candidates)) == 1

    refined = refine_document_records(original)

    assert len(refined.blocks) == 2
    assert MODEL_NON_TITLE_SEED_ROLE not in refined.candidates[index].roles
    assert MODEL_NON_TITLE_SEED_ROLE in original.candidates[index].roles
    assert "document_local_anatomy_overrode_model_root_veto" in refined.reason_flags
    assert "document_contextual_byline" in refined.candidates[index + 1].roles
    for old, new in zip(original.candidates, refined.candidates, strict=True):
        assert (new.raw_text, new.bbox, new.text_ids, new.section_id, new.model_scores) == (
            old.raw_text,
            old.bbox,
            old.text_ids,
            old.section_id,
            old.model_scores,
        )
    # The legacy grouping function keeps its learned prior. Only the opt-in
    # refiner's proven source context permits the new record.
    assert len(group_front_matter_blocks(original.candidates)) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing-byline",
        "foreign-page",
        "foreign-column",
        "before-title",
        "nonperson-prefix",
        "unsafe-separator",
        "missing-prose",
        "foreign-prose",
        "body-heading",
        "probation-title",
        "probation-byline",
        "toc",
        "nonheading-title",
    ],
)
def test_unproven_article_cannot_override_model_root_veto(fault):
    original, index = _denied_record()
    rows = list(original.candidates)
    title, byline, prose = rows[index : index + 3]
    if fault == "missing-byline":
        rows[index + 1] = replace(byline, roles=frozenset())
    elif fault == "foreign-page":
        rows[index + 1] = replace(byline, page=2)
    elif fault == "foreign-column":
        rows[index + 1] = replace(byline, bbox=(500, 340, 900, 360))
    elif fault == "before-title":
        rows[index + 1] = replace(byline, bbox=(50, 250, 450, 280))
    elif fault == "nonperson-prefix":
        rows[index + 1] = replace(byline, raw_text="Observed water responses, Forest University")
    elif fault == "unsafe-separator":
        rows[index + 1] = replace(byline, raw_text="Mara Quill 2020: 4-8, Forest University")
    elif fault == "missing-prose":
        rows[index + 2] = replace(prose, raw_text="Short index description.")
    elif fault == "foreign-prose":
        rows[index + 2] = replace(prose, page=9)
    elif fault == "body-heading":
        rows[index] = replace(title, roles=title.roles | {"body_heading"})
    elif fault == "probation-title":
        rows[index] = replace(title, roles=title.roles | {"byline_probation"})
    elif fault == "probation-byline":
        rows[index + 1] = replace(byline, roles=byline.roles | {"byline_probation"})
    elif fault == "nonheading-title":
        rows[index] = replace(title, source_kind="paragraph", text_ids=(99,), paragraph_id=99)
    original = replace(
        original,
        candidates=tuple(rows),
        reason_flags=("toc_listing",) if fault == "toc" else original.reason_flags,
    )

    refined = refine_document_records(original)

    assert refined is original
    assert MODEL_NON_TITLE_SEED_ROLE in refined.candidates[index].roles
    assert "document_local_anatomy_overrode_model_root_veto" not in refined.reason_flags
