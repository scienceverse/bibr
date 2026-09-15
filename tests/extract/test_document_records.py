"""Document-only anatomy recovery for bylines glued to affiliations/prose."""

from dataclasses import replace

import pytest

from bibr.extract.document_records import refine_document_records
from tests.extract.test_document_scope import _candidate, _resolution

PROSE = (
    "We measured seedling growth in three independent field experiments. "
    "The study compared watering schedules across several seasons and recorded changes in leaf area. "
    "Our results show that the timing of water availability affects the observed response. "
    "These findings suggest that seasonal conditions should be considered when comparing different habitats."
)


def _records(name_row="Mara Quill, Forest University, Hill City, Country", *, composite=False):
    rows = []
    for index, title in enumerate(
        ["SHADE RESPONSES IN ALPINE SEEDLINGS", "WATER RESPONSES IN ALPINE SEEDLINGS"]
    ):
        sid = index + 1
        title_row = replace(
            _candidate(len(rows), title, "title", sid),
            bbox=(50, 100 + index * 200, 450, 130 + index * 200),
        )
        rows.append(title_row)
        byline = replace(
            _candidate(
                len(rows),
                name_row + (" " + PROSE if composite else ""),
                "affiliation",
                sid,
                text_id=sid * 2,
            ),
            roles=frozenset({"byline", "affiliation"} if index == 0 else {"affiliation"}),
            bbox=(50, 140 + index * 200, 450, 160 + index * 200),
        )
        rows.append(byline)
        if not composite:
            rows.append(
                replace(
                    _candidate(len(rows), PROSE, "prose", sid, text_id=sid * 2 + 1),
                    bbox=(50, 170 + index * 200, 450, 250 + index * 200),
                )
            )
    return _resolution(rows)


@pytest.mark.parametrize(
    "name_row",
    [
        "Mara Quill, Forest University, Hill City, Country",
        "Mara Quill1, and Elian Brook2, 1. Forest University, Hill City, Country",
        "SEEDLINGS Mara Quill1, Elian Brook2, and Talia Vale2, 1. Forest University, Hill City, Country",
    ],
)
@pytest.mark.parametrize("composite", [False, True])
def test_composite_local_byline_recovers_independent_article(name_row, composite):
    original = _records(name_row, composite=composite)
    assert len(original.blocks) == 1

    refined = refine_document_records(original)

    assert len(refined.blocks) == 2
    assert refined.selected_block_id is None
    assert "document_contextual_anatomy" in refined.reason_flags
    assert [row.raw_text for row in refined.candidates] == [
        row.raw_text for row in original.candidates
    ]
    assert [row.text_ids for row in refined.candidates] == [
        row.text_ids for row in original.candidates
    ]
    assert sum("document_contextual_byline" in row.roles for row in refined.candidates) == 1
    assert len(original.blocks) == 1


@pytest.mark.parametrize(
    "name_row",
    [
        "Forest University, Hill City, Country",
        "Observed differences in forest sites, Forest University, Hill City, Country",
        "DATA AND METHODS, Forest University, Hill City, Country",
        "Mara Quill 2020: 4-8, Forest University, Hill City, Country",
    ],
)
def test_nonperson_prefix_does_not_develop_an_article(name_row):
    resolution = _records(name_row)
    assert refine_document_records(resolution) is resolution


@pytest.mark.parametrize(
    "fault",
    [
        "foreign-page",
        "foreign-column",
        "before-title",
        "no-prose",
        "toc",
        "body-heading",
        "foreign-prose-page",
        "foreign-prose-column",
    ],
)
def test_unsupported_local_anatomy_does_not_split(fault):
    resolution = _records()
    candidates = list(resolution.candidates)
    title, byline = candidates[3:5]
    if fault == "foreign-page":
        candidates[4] = replace(byline, page=2)
    elif fault == "foreign-column":
        candidates[4] = replace(byline, bbox=(500, 340, 700, 360))
    elif fault == "before-title":
        candidates[4] = replace(byline, bbox=(50, 250, 450, 280))
    elif fault == "no-prose":
        candidates[5] = replace(candidates[5], raw_text="Short index summary.")
    elif fault == "toc":
        resolution = replace(resolution, reason_flags=("toc_listing",))
    elif fault == "body-heading":
        candidates[3] = replace(title, roles=title.roles | {"body_heading"})
    elif fault == "foreign-prose-page":
        candidates[5] = replace(candidates[5], page=9)
    elif fault == "foreign-prose-column":
        candidates[5] = replace(candidates[5], bbox=(500, 370, 700, 450))
    resolution = replace(resolution, candidates=tuple(candidates))

    assert refine_document_records(resolution) is resolution


@pytest.mark.parametrize("identity", ["repeated-title", "shared-doi", "conflicting-dois"])
def test_recovered_presentations_use_corroborated_article_identity(identity):
    resolution = _records()
    candidates = list(resolution.candidates)
    if identity != "shared-doi":
        candidates[3] = replace(
            candidates[3],
            raw_text=candidates[0].raw_text,
            normalized_text=candidates[0].normalized_text,
        )
    if identity != "repeated-title":
        for index, section in enumerate((1, 2)):
            doi = "10.9999/shared" if identity == "shared-doi" else f"10.9999/distinct-{index}"
            candidates.insert(
                index * 4 + 2,
                replace(
                    _candidate(100 + index, doi, "doi", section, text_id=100 + index),
                    bbox=(50, 165 + index * 200, 450, 168 + index * 200),
                ),
            )
    candidates = [replace(row, reading_order=index) for index, row in enumerate(candidates)]
    resolution = _resolution(candidates)

    refined = refine_document_records(resolution)

    assert len(refined.blocks) == (2 if identity == "conflicting-dois" else 1)
    if identity != "conflicting-dois":
        assert "shared_document_contextual_identity" in refined.blocks[0].merge_reasons
        assert len(refined.blocks[0].source_block_ids) == 2
    assert [row.raw_text for row in refined.candidates] == [row.raw_text for row in candidates]


@pytest.mark.parametrize("valid", [True, False])
def test_classifier_title_on_author_affiliation_heading_requires_local_anatomy(valid):
    original = _records()
    candidates = list(original.candidates)
    byline = candidates[4]
    candidates[4] = replace(
        byline,
        source_kind="heading",
        text_ids=(),
        paragraph_id=None,
        roles=byline.roles | {"heading", "title"},
        raw_text=byline.raw_text if valid else "Observed effects in forests, Forest University",
    )
    resolution = _resolution(candidates)

    refined = refine_document_records(resolution)

    if valid:
        assert len(refined.blocks) == 2
        assert "title" not in refined.candidates[4].roles
        assert "byline" in refined.candidates[4].roles
        assert "document_byline_title_role_repaired" in refined.reason_flags
        assert refined.candidates[4].raw_text == candidates[4].raw_text
    else:
        assert refined is resolution
        assert "title" in refined.candidates[4].roles
