"""Unified research-integrity statement ownership regressions."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from bibr.config import PipelineOptions
from bibr.models import PaperAuthor, PaperMetadata
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.validation import ValidationIssue

_SYNTHETIC_OUTPUT = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "integrity_statements_synthetic.json").read_text()
)


def _synthetic_contents(case: str) -> PaperContents:
    """Build invented output rows with explicit provenance for ownership tests."""
    snapshot = _SYNTHETIC_OUTPUT[case]
    sections = [
        PaperSection(
            section_id=row["section_id"],
            header=row["header"],
            level=row["level"],
            parent_section_id=row["parent_section_id"],
            section_type=CanonicalSection(row["section_type"]),
            classification_source=row["classification_source"],
            classification_score=row["classification_score"],
        )
        for row in snapshot["sections"]
    ]
    sentences = [
        PaperSentence(
            text_id=row["text_id"],
            text=row["text"],
            section_id=row["section_id"],
            paragraph_id=row["paragraph_id"],
            page_number=row["page_number"],
        )
        for row in snapshot["sentences"]
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
    )


def _resolver_module():
    # Keep collection green during the RED phase: the new module is itself part
    # of the contract under test, so its absence must be reported as a failed
    # test rather than aborting collection.
    return importlib.import_module("bibr.extract.integrity_statements")


def _contents(
    sections_spec: list[
        tuple[
            int,
            str,
            CanonicalSection,
            str | None,
            float,
            list[str | tuple[str, int]],
        ]
    ],
    *,
    native: bool = False,
) -> PaperContents:
    sections = [PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)]
    sentences: list[PaperSentence] = []
    text_id = 1
    for section_id, header, section_type, source, score, rows in sections_spec:
        sections.append(
            PaperSection(
                section_id=section_id,
                header=header,
                level=1,
                parent_section_id=0,
                section_type=section_type,
                classification_source=source,
                classification_score=score,
            )
        )
        for row in rows:
            text, paragraph_id = row if isinstance(row, tuple) else (row, text_id)
            sentences.append(
                PaperSentence(
                    text_id=text_id,
                    text=text,
                    section_id=section_id,
                    paragraph_id=paragraph_id,
                    page_number=section_id,
                )
            )
            text_id += 1
    contents = PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
    )
    if native:
        contents.preparsed_metadata = PaperMetadata(doi="", title="Native title")
    return contents


def _paper_authors(*names: tuple[str, str]) -> tuple[PaperAuthor, ...]:
    return tuple(
        PaperAuthor(
            author_id=index,
            given=given,
            family=family,
            affiliation="",
        )
        for index, (given, family) in enumerate(names, start=1)
    )


def _resolve_apply(
    contents: PaperContents,
    mode: str,
    *,
    authors: tuple[PaperAuthor, ...] = (),
) -> tuple[object, PaperMetadata]:
    module = _resolver_module()
    resolution = module.resolve_integrity_statements(
        contents,
        mode=mode,
        author_names=tuple((author.given, author.family) for author in authors),
    )
    metadata = PaperMetadata(doi="", title="T", authors=list(authors))
    module.apply_integrity_resolution(contents, metadata, resolution)
    return resolution, metadata


def test_public_types_and_default_shadow_mode_are_frozen():
    module = _resolver_module()
    assert module.IntegrityStatementCandidate.__dataclass_params__.frozen is True
    assert module.IntegrityStatementResolution.__dataclass_params__.frozen is True
    assert PipelineOptions().integrity_statement_mode == "shadow"


@pytest.mark.parametrize(
    ("heading", "section_type", "field"),
    [
        ("Funding", CanonicalSection.FUNDING, "funding_statement"),
        ("Competing interests", CanonicalSection.COI, "coi_statement"),
        ("Ethics approval", CanonicalSection.ETHICS, "ethics_statement"),
        ("Data availability", CanonicalSection.OPEN_DATA, "data_availability"),
    ],
)
def test_strong_exact_declaration_heading_accepts_compact_negative(
    heading: str, section_type: CanonicalSection, field: str
):
    contents = _contents([(1, heading, section_type, "exact_alias", 1.0, ["Not applicable."])])
    _resolution, metadata = _resolve_apply(contents, "active")
    assert getattr(metadata, field) == "Not applicable."


def test_topical_philosophy_sections_are_retrieval_only():
    contents = _synthetic_contents("topical_ethics")
    original_body = tuple(vars(sentence).copy() for sentence in contents.sentences)
    legacy_resolution, legacy = _resolve_apply(contents, "legacy")
    shadow_resolution, shadow = _resolve_apply(contents, "shadow")
    resolution, active = _resolve_apply(contents, "active")

    expected_legacy = "\n\n".join(
        " ".join(
            sentence.text
            for sentence in contents.sentences
            if sentence.section_id == section.section_id
        )
        for section in contents.sections
    )
    assert legacy.ethics_statement == expected_legacy
    assert shadow.ethics_statement == expected_legacy
    assert active.ethics_statement is None
    assert legacy_resolution.issues == ()
    issue = next(
        issue
        for issue in shadow_resolution.issues
        if issue.code == "VAL_STATEMENT_SUSPECT" and "ethics_statement" in issue.evidence_ids
    )
    assert issue.evidence_ids[:4] == (
        "ethics_statement",
        "section:1",
        "section:2",
        "section:3",
    )
    assert len(issue.evidence_ids) <= 20
    assert len(issue.evidence_ids) == len(set(issue.evidence_ids))
    assert tuple(vars(sentence).copy() for sentence in contents.sentences) == original_body
    section_candidates = {
        candidate.section_ids[0]: candidate
        for candidate in resolution.candidates
        if candidate.method == "classified_section"
    }
    assert section_candidates[1].text_ids == tuple(range(10, 22))
    assert section_candidates[2].text_ids == tuple(range(30, 42))
    assert section_candidates[3].text_ids == tuple(range(50, 62))
    assert section_candidates[1].paragraph_ids == (101, 102, 103, 104)
    assert section_candidates[1].pages == (1, 2)
    assert section_candidates[2].paragraph_ids == (111, 112, 113, 114)
    assert section_candidates[2].pages == (3, 4)
    assert section_candidates[3].paragraph_ids == (121, 122, 123, 124)
    assert section_candidates[3].pages == (5, 6)
    assert section_candidates[1].classification_score == 0.98
    assert section_candidates[2].classification_score == 0.97
    assert section_candidates[3].classification_score == 0.96


def test_model_classified_topical_funding_prose_needs_a_declaration_predicate():
    contents = _contents(
        [
            (
                1,
                "Funding and political influence",
                CanonicalSection.FUNDING,
                "model",
                0.997,
                ["Public funding shapes the priorities debated in science policy."],
            )
        ]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.funding_statement is None


def test_shadow_uses_exact_legacy_section_snapshot_but_selected_funding_is_bounded():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                [
                    ("This work was supported by NSF grant 123.", 7),
                    ("Ethics: Not applicable.", 7),
                ],
            )
        ]
    )

    _legacy_resolution, legacy = _resolve_apply(contents, "legacy")
    shadow_resolution, shadow = _resolve_apply(contents, "shadow")
    active_resolution, active = _resolve_apply(contents, "active")

    exact_legacy = "This work was supported by NSF grant 123. Ethics: Not applicable."
    assert legacy.funding_statement == exact_legacy
    assert shadow.funding_statement == exact_legacy
    assert active.funding_statement == "This work was supported by NSF grant 123."
    assert active.ethics_statement == "Ethics: Not applicable."
    assert (
        _resolver_module().render_selected_integrity_statement(
            contents, active_resolution, "funding_statement"
        )
        == "This work was supported by NSF grant 123."
    )
    funding_issues = [
        issue
        for issue in shadow_resolution.issues
        if issue.code == "VAL_STATEMENT_SUSPECT" and "funding_statement" in issue.evidence_ids
    ]
    assert len(funding_issues) == 1
    assert funding_issues[0].evidence_ids == (
        "funding_statement",
        "section:1",
        "text:1",
        "text:2",
    )


def test_legacy_snapshot_preserves_pre_finalize_bytes_while_selected_renders_final_clean():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                [
                    ("This work was supported by NSF grant $^{123}$ .", 7),
                    ("Ethics: Not applicable.", 7),
                ],
            )
        ]
    )
    module = _resolver_module()
    resolution = module.resolve_integrity_statements(contents, mode="shadow")
    contents.finalize_text()
    metadata = PaperMetadata(doi="", title="T")

    module.apply_integrity_resolution(contents, metadata, resolution)

    assert metadata.funding_statement == (
        "This work was supported by NSF grant $^{123}$ . Ethics: Not applicable."
    )
    assert (
        module.render_selected_integrity_statement(contents, resolution, "funding_statement")
        == "This work was supported by NSF grant 123 ."
    )


def test_legacy_section_snapshot_preserves_internal_sentence_whitespace():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                [
                    "  This work was supported by NSF grant 123.  ",
                    "  Ethics: Not applicable.  ",
                ],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "legacy")

    assert metadata.funding_statement == (
        "This work was supported by NSF grant 123.     Ethics: Not applicable."
    )


def test_formatting_only_final_clean_change_keeps_frozen_shadow_comparison_issue_free():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                ["This work was supported by NSF grant $^{123}$ ."],
            )
        ]
    )
    module = _resolver_module()
    resolution = module.resolve_integrity_statements(contents, mode="shadow")

    assert resolution.issues == ()
    assert isinstance(resolution.legacy_statement_snapshots, tuple)
    assert dict(resolution.legacy_statement_snapshots)["funding_statement"] == (
        "This work was supported by NSF grant $^{123}$ ."
    )

    contents.finalize_text()
    metadata = PaperMetadata(doi="", title="T")
    module.apply_integrity_resolution(contents, metadata, resolution)
    module.apply_integrity_resolution(contents, metadata, resolution)

    assert metadata.funding_statement == "This work was supported by NSF grant $^{123}$ ."
    assert (
        module.render_selected_integrity_statement(contents, resolution, "funding_statement")
        == "This work was supported by NSF grant 123 ."
    )
    assert resolution.issues == ()


def test_shadow_uses_exact_cross_paragraph_legacy_fallback_snapshot():
    contents = _contents(
        [
            (
                1,
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                "exact_alias",
                1.0,
                [
                    ("This work was supported by NSF grant 123.", 10),
                    ("The next paragraph describes unrelated analysis.", 11),
                    ("A final paragraph discusses the results.", 12),
                ],
            )
        ]
    )

    _legacy_resolution, legacy = _resolve_apply(contents, "legacy")
    shadow_resolution, shadow = _resolve_apply(contents, "shadow")
    _active_resolution, active = _resolve_apply(contents, "active")

    exact_legacy = (
        "This work was supported by NSF grant 123. "
        "The next paragraph describes unrelated analysis. "
        "A final paragraph discusses the results."
    )
    assert legacy.funding_statement == exact_legacy
    assert shadow.funding_statement == exact_legacy
    assert active.funding_statement == "This work was supported by NSF grant 123."
    assert any(
        issue.code == "VAL_STATEMENT_SUSPECT" and "funding_statement" in issue.evidence_ids
        for issue in shadow_resolution.issues
    )


@pytest.mark.parametrize("mode", ["legacy", "shadow"])
def test_legacy_lexical_fallback_warning_is_preserved_and_idempotent(mode: str):
    contents = _contents(
        [
            (
                1,
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                "exact_alias",
                1.0,
                ["This work was supported by NSF grant 123."],
            )
        ]
    )
    module = _resolver_module()
    resolution = module.resolve_integrity_statements(contents, mode=mode)
    metadata = PaperMetadata(doi="", title="T")

    module.apply_integrity_resolution(contents, metadata, resolution)
    module.apply_integrity_resolution(contents, metadata, resolution)

    assert contents.processing_warnings == ["STATEMENT_LEXICAL_FALLBACK: funding_statement"]


@pytest.mark.parametrize("canonical_rows", [[], ["   "]])
def test_empty_canonical_section_does_not_suppress_exact_legacy_lexical_fallback(
    canonical_rows: list[str],
):
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                canonical_rows,
            ),
            (
                2,
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                "exact_alias",
                1.0,
                ["This work was supported by NSF grant 123."],
            ),
        ]
    )

    _legacy_resolution, legacy = _resolve_apply(contents, "legacy")
    shadow_resolution, shadow = _resolve_apply(contents, "shadow")

    assert legacy.funding_statement == "This work was supported by NSF grant 123."
    assert shadow.funding_statement == legacy.funding_statement
    assert not any(
        issue.code == "VAL_STATEMENT_SUSPECT" and "funding_statement" in issue.evidence_ids
        for issue in shadow_resolution.issues
    )


@pytest.mark.parametrize(
    ("heading", "section_type", "text", "field"),
    [
        (
            "Ethics and public policy",
            CanonicalSection.ETHICS,
            "The government approved a carbon tax after public debate.",
            "ethics_statement",
        ),
        (
            "Funding and political influence",
            CanonicalSection.FUNDING,
            "The policy debate was funded by private actors in several countries.",
            "funding_statement",
        ),
    ],
)
def test_model_classified_topical_action_prose_is_not_a_declaration(
    heading: str,
    section_type: CanonicalSection,
    text: str,
    field: str,
):
    contents = _contents([(1, heading, section_type, "model", 0.999, [text])])
    _resolution, metadata = _resolve_apply(contents, "active")
    assert getattr(metadata, field) is None


@pytest.mark.parametrize(
    ("heading", "section_type", "text", "field"),
    [
        (
            "Ethics and public policy",
            CanonicalSection.ETHICS,
            "Animal welfare policy was approved by the government after public debate.",
            "ethics_statement",
        ),
        (
            "Funding and political influence",
            CanonicalSection.FUNDING,
            "The policy debate was funded by the National Science Foundation after public "
            "consultation.",
            "funding_statement",
        ),
        (
            "Data availability policy",
            CanonicalSection.OPEN_DATA,
            "University data are available to administrators under the new government policy.",
            "data_availability",
        ),
        (
            "Data availability policy",
            CanonicalSection.OPEN_DATA,
            "Datasets are increasingly available to researchers, creating new ethical questions.",
            "data_availability",
        ),
        (
            "Data availability policy",
            CanonicalSection.OPEN_DATA,
            "Datasets are publicly available to researchers across disciplines.",
            "data_availability",
        ),
        (
            "Data availability policy",
            CanonicalSection.OPEN_DATA,
            "Code is freely available on modern platforms for teaching.",
            "data_availability",
        ),
    ],
)
def test_model_classified_topical_subject_does_not_own_integrity_declaration(
    heading: str,
    section_type: CanonicalSection,
    text: str,
    field: str,
):
    contents = _contents([(1, heading, section_type, "model", 0.999, [text])])
    _resolution, metadata = _resolve_apply(contents, "active")
    assert getattr(metadata, field) is None


@pytest.mark.parametrize(
    ("heading", "section_type", "text", "field"),
    [
        (
            "Ethics approval",
            CanonicalSection.ETHICS,
            "This study was approved by the Institutional Review Board.",
            "ethics_statement",
        ),
        (
            "Funding statement",
            CanonicalSection.FUNDING,
            "This work was funded by the National Science Foundation.",
            "funding_statement",
        ),
        (
            "Data availability statement",
            CanonicalSection.OPEN_DATA,
            "The datasets generated during this study are available from the repository.",
            "data_availability",
        ),
    ],
)
def test_model_classified_study_or_work_subject_owns_real_declaration(
    heading: str,
    section_type: CanonicalSection,
    text: str,
    field: str,
):
    contents = _contents([(1, heading, section_type, "model", 0.999, [text])])
    _resolution, metadata = _resolve_apply(contents, "active")
    assert getattr(metadata, field) == text


@pytest.mark.parametrize(
    ("text", "authors"),
    [
        ("J.W. was supported by NSF grant 123.", _paper_authors(("Jane", "Werner"))),
        (
            "Jane Doe received funding from NIH grant R01-MH123.",
            _paper_authors(("Jane", "Doe")),
        ),
        (
            "A.B. and C.D. were supported by NIH grant R01-MH123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
        ),
        ("Jane W. Doe was supported by NSF grant 123.", _paper_authors(("Jane W.", "Doe"))),
        ("The present study was funded by NSF grant 123.", ()),
        ("Research reported in this publication was supported by NIH grant R01-MH123.", ()),
    ],
)
def test_author_specific_funding_declaration_remains_recoverable(
    text: str, authors: tuple[PaperAuthor, ...]
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "model", 0.999, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == text


def test_author_specific_funding_rejects_an_unlisted_name():
    text = "Jane Doe received funding from NIH grant R01-MH123."
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "model", 0.999, [text])]
    )

    _resolution, metadata = _resolve_apply(
        contents,
        "active",
        authors=_paper_authors(("Alice", "Roe")),
    )

    assert metadata.funding_statement is None


@pytest.mark.parametrize(
    ("text", "author"),
    [
        (
            "A.M.D.L.C. was supported by NSF grant 123.",
            _paper_authors(("Ana Maria", "de la Cruz")),
        ),
        (
            "A.D.L.C. was supported by NSF grant 123.",
            _paper_authors(("Ana Maria", "de la Cruz")),
        ),
        (
            "J.W.D.J. was supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe Jr.")),
        ),
        (
            "J.D.J. was supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe Jr.")),
        ),
    ],
)
def test_author_initial_aliases_include_particles_suffixes_and_short_given_form(
    text: str, author: tuple[PaperAuthor, ...]
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "model", 0.999, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=author)

    assert metadata.funding_statement == text


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        ((), None),
        (_paper_authors(("Alice", "Roe")), None),
        (
            _paper_authors(("Jane", "Doe")),
            "Funding: Jane Doe was funded by NSF grant 123.",
        ),
    ],
)
def test_labeled_named_funding_still_requires_a_grounded_author(
    authors: tuple[PaperAuthor, ...], expected: str | None
):
    text = "Funding: Jane Doe was funded by NSF grant 123."
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        ((), None),
        (_paper_authors(("Alice", "Roe")), None),
        (_paper_authors(("Jane", "Doe")), "funded by NSF grant 123."),
    ],
)
def test_category_clipping_cannot_erase_named_funding_grounding(
    authors: tuple[PaperAuthor, ...], expected: str | None
):
    text = "Ethics approval was obtained; Jane Doe was funded by NSF grant 123."
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors"),
    [
        (
            "Jane W. Doe, Jr. was supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe, Jr.")),
        ),
        (
            "Jane W. Doe, Jr. and Alice Brown were supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe, Jr."), ("Alice", "Brown")),
        ),
    ],
)
def test_grounded_comma_suffixes_are_not_mistaken_for_author_lists(
    text: str, authors: tuple[PaperAuthor, ...]
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "model", 0.999, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == text


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        (
            "Jane W. Doe, Jr. was supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe Jr.")),
            "Jane W. Doe, Jr. was supported by NSF grant 123.",
        ),
        (
            "Jane W. Doe Jr. was supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe, Jr.")),
            "Jane W. Doe Jr. was supported by NSF grant 123.",
        ),
        (
            "Jane W. Doe, Jr., Alice Brown were supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe, Jr."), ("Alice", "Brown")),
            "Jane W. Doe, Jr., Alice Brown were supported by NSF grant 123.",
        ),
        (
            "Jane W. Doe, Jr., Alice Brown were supported by NSF grant 123.",
            _paper_authors(("Jane W.", "Doe, Jr.")),
            None,
        ),
        (
            "i\u0307pek Doe was supported by NSF grant 123.",
            _paper_authors(("İpek", "Doe")),
            "i\u0307pek Doe was supported by NSF grant 123.",
        ),
    ],
)
def test_external_review_suffix_punctuation_and_list_partitioning(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        (
            "Jane and Doe were supported by NSF grant 123.",
            _paper_authors(("Jane", "Doe")),
            None,
        ),
        (
            "A. and B. were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown")),
            None,
        ),
        (
            "Jane Doe and Alice Brown were supported by NSF grant 123.",
            _paper_authors(("Jane", "Doe"), ("Alice", "Brown")),
            "Jane Doe and Alice Brown were supported by NSF grant 123.",
        ),
        (
            "Jane Doe, Alice Brown were supported by NSF grant 123.",
            _paper_authors(("Jane", "Doe"), ("Alice", "Brown")),
            "Jane Doe, Alice Brown were supported by NSF grant 123.",
        ),
        (
            "Jane Doe and Alice Brown were supported by NSF grant 123.",
            _paper_authors(("Jane", "Doe")),
            None,
        ),
    ],
)
def test_author_list_delimiters_cannot_be_merged_into_one_alias(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "model", 0.999, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        (
            "Abel was supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Edward", "Lee")),
            None,
        ),
        (
            "ABCD was supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            None,
        ),
        (
            "Alice BrownCarol Doe were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            None,
        ),
        (
            "Alice Brown, Carol Doe were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            "Alice Brown, Carol Doe were supported by NSF grant 123.",
        ),
        (
            "Alice Brown and Carol Doe were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            "Alice Brown and Carol Doe were supported by NSF grant 123.",
        ),
        (
            "Alice Brown & Carol Doe were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            "Alice Brown & Carol Doe were supported by NSF grant 123.",
        ),
        (
            "A.B., C.D. were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            "A.B., C.D. were supported by NSF grant 123.",
        ),
        (
            "AB & CD were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            "AB & CD were supported by NSF grant 123.",
        ),
        (
            "Alice Brown, Eve Fox were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            None,
        ),
    ],
)
def test_external_review_author_aliases_require_raw_list_delimiters(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        (
            "Funding: Jane Doe acknowledges financial support from NSF.",
            _paper_authors(("Alice", "Brown")),
            None,
        ),
        (
            "Funding: Jane Doe acknowledges financial support from NSF.",
            _paper_authors(("Jane", "Doe")),
            "Funding: Jane Doe acknowledges financial support from NSF.",
        ),
        (
            "Funding: The authors acknowledge financial support from NSF.",
            (),
            "Funding: The authors acknowledge financial support from NSF.",
        ),
    ],
)
def test_labeled_funding_acknowledgments_obey_subject_grounding(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Funding: NSF grant 123.", "Funding: NSF grant 123."),
        ("Funding: Jane Doe thanks NSF.", None),
    ],
)
def test_labeled_funder_only_fallback_is_compact_and_not_person_led(
    text: str, expected: str | None
):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        (_paper_authors(("Alice", "Roe")), None),
        (
            _paper_authors(("Jane", "Doe")),
            "Jane Doe — supported by NSF grant 123.",
        ),
    ],
)
def test_dash_cannot_detach_a_named_subject_from_its_funding_action(
    authors: tuple[PaperAuthor, ...], expected: str | None
):
    text = "Jane Doe — supported by NSF grant 123."
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        (
            "Jane Doe; supported by NSF grant 123.",
            _paper_authors(("Alice", "Roe")),
            None,
        ),
        (
            "Funding: Jane Doe; supported by NSF grant 123.",
            _paper_authors(("Alice", "Roe")),
            None,
        ),
        ("Public Policy; funded by NSF grant 123.", (), None),
        (
            "J.D.; A.B. were supported by NSF grant 123.",
            _paper_authors(("Alice", "Brown")),
            None,
        ),
        (
            "Jane Doe; Alice Brown were supported by NSF grant 123.",
            _paper_authors(("Jane", "Doe"), ("Alice", "Brown")),
            "Jane Doe; Alice Brown were supported by NSF grant 123.",
        ),
        (
            "Ethics approval was obtained; This work was funded by NSF grant 123.",
            (),
            "funded by NSF grant 123.",
        ),
    ],
)
def test_semicolon_cannot_turn_a_named_subject_into_a_subjectless_fragment(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        (
            "Alice Roe was supported by NSF grant 123; "
            "This work was funded by NIH grant R01-MH123.",
            _paper_authors(("Jane", "Doe")),
            None,
        ),
        (
            "Jane Doe was supported by NSF grant 123; Alice Roe was funded by NIH grant R01-MH123.",
            _paper_authors(("Jane", "Doe")),
            None,
        ),
        (
            "This work was funded by NSF grant 123; "
            "The study was supported by NIH grant R01-MH123.",
            _paper_authors(("Jane", "Doe")),
            "This work was funded by NSF grant 123; "
            "The study was supported by NIH grant R01-MH123.",
        ),
        (
            "Jane Doe was supported by NSF grant 123; Alice Roe was funded by NIH grant R01-MH123.",
            _paper_authors(("Jane", "Doe"), ("Alice", "Roe")),
            "Jane Doe was supported by NSF grant 123; Alice Roe was funded by NIH grant R01-MH123.",
        ),
        (
            "This work was funded by NSF grant 123; "
            "Competing interests: Alice Roe received funding from NIH.",
            _paper_authors(("Jane", "Doe")),
            "This work was funded by NSF grant 123;",
        ),
    ],
)
def test_external_review_all_funding_clauses_obey_named_grounding(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        ("Jane Doe was funded by NSF grant 123.", (), None),
        (
            "Jane Doe was funded by NSF grant 123.",
            _paper_authors(("Alice", "Roe")),
            None,
        ),
        (
            "Jane Doe was funded by NSF grant 123.",
            _paper_authors(("Jane", "Doe")),
            "Jane Doe was funded by NSF grant 123.",
        ),
        (
            "This work was funded by NSF grant 123.",
            (),
            "This work was funded by NSF grant 123.",
        ),
        ("NSF grant 123.", (), "NSF grant 123."),
    ],
)
def test_strong_funding_heading_cannot_override_failed_author_grounding(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("text", "authors", "expected"),
    [
        (
            "Funding: Jane Doe received no funding.",
            _paper_authors(("Alice", "Roe")),
            None,
        ),
        (
            "Funding: Jane Doe received no funding.",
            _paper_authors(("Jane", "Doe")),
            "Funding: Jane Doe received no funding.",
        ),
        (
            "Funding: Jane Doe had no external funding.",
            _paper_authors(("Alice", "Roe")),
            None,
        ),
        (
            "Funding: Jane Doe has no funding.",
            _paper_authors(("Jane", "Doe")),
            "Funding: Jane Doe has no funding.",
        ),
        (
            "Funding: This study received no funding.",
            (),
            "Funding: This study received no funding.",
        ),
        (
            "Funding: This study had no external funding.",
            (),
            "Funding: This study had no external funding.",
        ),
        (
            "Funding: No funding was received.",
            (),
            "Funding: No funding was received.",
        ),
    ],
)
def test_negative_named_funding_requires_grounding_but_generic_negatives_do_not(
    text: str,
    authors: tuple[PaperAuthor, ...],
    expected: str | None,
):
    contents = _contents(
        [(1, "Funding statement", CanonicalSection.FUNDING, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active", authors=authors)

    assert metadata.funding_statement == expected


@pytest.mark.parametrize(
    ("heading", "section_type", "source", "text", "field", "expected"),
    [
        (
            "Funding",
            CanonicalSection.FUNDING,
            "model",
            "Policy was funded by NSF after public consultation.",
            "funding_statement",
            None,
        ),
        (
            "Discussion",
            CanonicalSection.DISCUSSION,
            "exact_alias",
            "Public Policy was supported by the Wellcome Trust in several countries.",
            "funding_statement",
            None,
        ),
        (
            "Discussion",
            CanonicalSection.DISCUSSION,
            "exact_alias",
            "Government Research was funded by NSF after public consultation.",
            "funding_statement",
            None,
        ),
        (
            "Methods",
            CanonicalSection.METHODS,
            "exact_alias",
            "A waiver was granted by the ethics committee.",
            "ethics_statement",
            "A waiver was granted by the ethics committee.",
        ),
    ],
)
def test_external_review_capitalized_subject_and_waiver_regressions(
    heading: str,
    section_type: CanonicalSection,
    source: str,
    text: str,
    field: str,
    expected: str | None,
):
    contents = _contents([(1, heading, section_type, source, 0.999, [text])])

    _resolution, metadata = _resolve_apply(contents, "active")

    assert getattr(metadata, field) == expected


def test_topical_waiver_question_is_not_an_ethics_declaration():
    text = "The analysis examined whether a waiver was granted by the ethics committee."
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.ethics_statement is None


@pytest.mark.parametrize(
    "text",
    [
        "This study examines projects funded by the National Science Foundation.",
        "Our research analyzes interventions supported by NIH grants.",
        "The study compares investigators who received funding from NSF.",
        "We surveyed organizations supported by the Wellcome Trust.",
    ],
)
def test_paper_subject_topical_funding_mentions_are_not_recovered(text: str):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.funding_statement is None


@pytest.mark.parametrize(
    "text",
    [
        "Written informed consent was obtained from all participants.",
        "Participants provided informed consent before enrollment.",
        "Patients gave informed consent to participate.",
    ],
)
def test_declarative_participant_consent_remains_recoverable(text: str):
    contents = _contents([(1, "Methods", CanonicalSection.METHODS, "exact_alias", 1.0, [text])])

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.ethics_statement == text


@pytest.mark.parametrize(
    "text",
    [
        "Participants discussed informed consent requirements during training.",
        "Written informed consent is a widely debated ethical practice.",
        "This study examines whether procedures were approved by an ethics committee.",
        "Our research discusses informed consent as a legal concept.",
        "The study analyzes how informed consent affects recruitment.",
        "All participants debated informed consent during the workshop.",
    ],
)
def test_topical_consent_mentions_are_not_recovered(text: str):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.ethics_statement is None


@pytest.mark.parametrize(
    "text",
    [
        "Raw data are available from the corresponding author on reasonable request.",
        "Anonymized data are available in the Zenodo repository.",
    ],
)
def test_modified_data_subject_declaration_remains_recoverable(text: str):
    contents = _contents(
        [(1, "Declarations", CanonicalSection.ENDNOTE, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.data_availability == text


@pytest.mark.parametrize(
    "text",
    [
        "The data show that materials used in other studies are available from the "
        "Zenodo repository.",
        "Our data indicate that code produced by competitors is available on GitHub.",
        "Data reveal that datasets generated by prior surveys are available upon request.",
    ],
)
def test_nested_topical_data_availability_mentions_are_not_recovered(text: str):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.data_availability is None


@pytest.mark.parametrize(
    "text",
    [
        "This study examines authors who declare no conflicts of interest.",
        "Participants reported no conflicts of interest during interviews.",
        "The model assumes no competing interests among firms.",
        "We found no conflict of interest between the two measures.",
    ],
)
def test_topical_conflict_mentions_are_not_recovered(text: str):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.coi_statement is None


def test_generic_exact_alias_is_not_copy_permission_for_topical_prose():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                ["Funding shapes the priorities debated in science policy."],
            )
        ]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.funding_statement is None


def test_strong_heading_does_not_promote_publisher_license_boilerplate():
    contents = _contents(
        [
            (
                1,
                "Data availability statement",
                CanonicalSection.OPEN_DATA,
                "exact_alias",
                1.0,
                ["Published under a Creative Commons license by Example Publisher."],
            )
        ]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.data_availability is None


@pytest.mark.parametrize("heading", ["Funding", "Funding statement"])
@pytest.mark.parametrize("source", ["model", "exact_alias"])
def test_cross_field_negative_label_is_not_owned_by_canonical_funding(heading: str, source: str):
    contents = _contents(
        [
            (
                1,
                heading,
                CanonicalSection.FUNDING,
                source,
                0.99,
                ["Ethics Not applicable."],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.funding_statement is None


def test_bare_ethics_label_does_not_promote_topical_not_applicable_prose():
    contents = _contents(
        [
            (
                1,
                "Discussion",
                CanonicalSection.DISCUSSION,
                "exact_alias",
                1.0,
                [
                    ("Ethics", 7),
                    ("Not applicable to the scope of this philosophical discussion.", 7),
                ],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.ethics_statement is None


def test_bare_funding_label_recovers_compact_negative_declaration():
    contents = _contents(
        [
            (
                1,
                "Declarations",
                CanonicalSection.ENDNOTE,
                "exact_alias",
                1.0,
                [("Funding", 7), ("None.", 7)],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.funding_statement == "Funding None."


def test_data_corresponding_author_phrase_is_allowed_but_standalone_contact_is_boundary():
    contents = _contents(
        [
            (
                1,
                "Declarations",
                CanonicalSection.ENDNOTE,
                "exact_alias",
                1.0,
                [
                    (
                        "The datasets generated during this study are available from the "
                        "corresponding author on reasonable request.",
                        7,
                    ),
                    ("Corresponding author: Jane Doe, jane@example.org", 7),
                ],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.data_availability == (
        "The datasets generated during this study are available from the corresponding "
        "author on reasonable request."
    )


def test_data_available_upon_request_from_corresponding_author_is_not_clipped():
    text = (
        "The datasets generated during this study are available upon reasonable request "
        "from the corresponding author."
    )
    contents = _contents(
        [
            (
                1,
                "Data availability",
                CanonicalSection.OPEN_DATA,
                "exact_alias",
                1.0,
                [text],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.data_availability == text


@pytest.mark.parametrize(
    "text",
    [
        "The data are available by contacting the corresponding author.",
        "The data are accessible through the corresponding author.",
    ],
)
def test_data_corresponding_author_contact_constructions_are_not_clipped(text: str):
    contents = _contents(
        [
            (
                1,
                "Data availability",
                CanonicalSection.OPEN_DATA,
                "exact_alias",
                1.0,
                [text],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.data_availability == text


def test_prefers_research_ethics_over_publication_consent_decoy():
    contents = _synthetic_contents("publication_consent_decoy")
    resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.funding_statement == (
        "This research was supported by the Example Research Council under grant TEST-001."
    )
    assert metadata.data_availability == (
        "The datasets generated during this study are available from the authors on reasonable request."
    )
    assert metadata.ethics_statement == (
        "This study was approved by the institutional review board at Example University. "
        "All participants gave informed consent before the interview."
    )
    assert metadata.coi_statement == (
        "The authors report no competing interests related to this research."
    )
    section_candidates = {
        candidate.section_ids[0]: candidate
        for candidate in resolution.candidates
        if candidate.section_ids
        and candidate.field == "ethics_statement"
        and candidate.method != "anchored_paragraph"
    }
    assert section_candidates[3].text_ids == (3, 4)
    assert section_candidates[3].paragraph_ids == (3,)
    assert section_candidates[3].pages == (2,)
    assert section_candidates[3].classification_score == 0.55
    assert section_candidates[3].accepted is True
    assert section_candidates[4].text_ids == (5,)
    assert section_candidates[4].paragraph_ids == (4,)
    assert section_candidates[4].pages == (2,)
    assert section_candidates[4].classification_score == 0.95
    assert section_candidates[4].accepted is False


@pytest.mark.parametrize(
    "heading",
    [
        "Consent for publication:",
        "Consent for publication?",
        "Consent for publication!",
        "Consent for publication。",
        "Consent for publication：",
        "“Consent for publication”",
        "（Consent for publication）",
    ],
)
def test_publication_consent_heading_with_terminal_or_wrapping_punctuation_is_rejected(
    heading: str,
):
    contents = _contents(
        [
            (
                25,
                heading,
                CanonicalSection.ETHICS,
                "model",
                0.95,
                ["Not applicable."],
            )
        ]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.ethics_statement is None


def test_methods_embedded_ethics_is_a_valid_anchored_paragraph():
    contents = _contents(
        [
            (
                1,
                "Methods",
                CanonicalSection.METHODS,
                "exact_alias",
                1.0,
                [
                    "Participants completed the survey.",
                    "The institutional review board approved the study and all participants "
                    "gave informed consent.",
                ],
            )
        ]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.ethics_statement == (
        "The institutional review board approved the study and all participants gave "
        "informed consent."
    )


@pytest.mark.parametrize(
    ("field", "text"),
    [
        (
            "ethics_statement",
            "The role of the ethics committee in public deliberation remains contested.",
        ),
        (
            "coi_statement",
            "Authors report conflicts of interest as a recurring topic in the literature.",
        ),
        (
            "data_availability",
            "Data availability remains a major challenge for the research community.",
        ),
    ],
)
def test_topical_anchor_prose_is_not_a_declaration(field: str, text: str):
    contents = _contents(
        [(1, "Discussion", CanonicalSection.DISCUSSION, "exact_alias", 1.0, [text])]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert getattr(metadata, field) is None


def test_shadow_preserves_legacy_lexical_scalar_while_active_rejects_topical_anchor():
    contents = _contents(
        [
            (
                1,
                "Discussion",
                CanonicalSection.DISCUSSION,
                "exact_alias",
                1.0,
                ["The role of the ethics committee in public deliberation remains contested."],
            )
        ]
    )
    _legacy_resolution, legacy = _resolve_apply(contents, "legacy")
    shadow_resolution, shadow = _resolve_apply(contents, "shadow")
    _active_resolution, active = _resolve_apply(contents, "active")

    assert legacy.ethics_statement == contents.sentences[0].text
    assert shadow.ethics_statement == legacy.ethics_statement
    assert active.ethics_statement is None
    assert any(
        issue.code == "VAL_STATEMENT_SUSPECT" and "ethics_statement" in issue.evidence_ids
        for issue in shadow_resolution.issues
    )


def test_only_independently_accepted_adjacent_declarations_are_joined():
    contents = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                ["Supported by NSF grant 123."],
            ),
            (
                2,
                "Financial support",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                ["Additional support came from the Wellcome Trust."],
            ),
            (
                3,
                "Funding in public policy",
                CanonicalSection.FUNDING,
                "model",
                0.99,
                ["Funding can change institutional incentives."],
            ),
        ]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.funding_statement == (
        "Supported by NSF grant 123.\n\nAdditional support came from the Wellcome Trust."
    )


def test_trusted_section_stops_when_another_category_starts_in_same_sentence():
    contents = _contents(
        [
            (
                1,
                "Funding statement",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                [
                    "This work was supported by NSF grant 123. "
                    "Competing interests: The authors declare none."
                ],
            )
        ]
    )
    resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.funding_statement == "This work was supported by NSF grant 123."
    section_candidate = next(
        candidate
        for candidate in resolution.candidates
        if candidate.method == "trusted_section" and candidate.field == "funding_statement"
    )
    assert "category_boundary_clipped" in section_candidate.reason_flags


def test_same_paragraph_bare_category_label_stops_funding_and_recovers_negative_ethics():
    contents = _contents(
        [
            (
                1,
                "Declarations",
                CanonicalSection.ENDNOTE,
                "exact_alias",
                1.0,
                [
                    ("This work was supported by NSF grant 123.", 7),
                    ("Ethics: Not applicable.", 7),
                ],
            )
        ]
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.funding_statement == "This work was supported by NSF grant 123."
    assert metadata.ethics_statement == "Ethics: Not applicable."


def test_truly_bare_next_category_label_splits_and_recovers_following_declaration():
    contents = _contents(
        [
            (
                1,
                "Declarations",
                CanonicalSection.ENDNOTE,
                "exact_alias",
                1.0,
                [
                    ("This work was supported by NSF grant 123.", 7),
                    ("Ethics", 7),
                    ("Not applicable.", 7),
                ],
            )
        ]
    )

    _resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.funding_statement == "This work was supported by NSF grant 123."
    assert metadata.ethics_statement == "Ethics Not applicable."


def test_later_valid_funding_anchor_survives_rejected_anchor_in_same_paragraph():
    contents = _contents(
        [
            (
                1,
                "Discussion",
                CanonicalSection.DISCUSSION,
                "exact_alias",
                1.0,
                [
                    ("The conclusion was supported by University data.", 9),
                    ("This research was supported by NSF grant 123.", 9),
                ],
            )
        ]
    )

    resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.funding_statement == "This research was supported by NSF grant 123."
    funding_candidates = [
        candidate
        for candidate in resolution.candidates
        if candidate.field == "funding_statement" and candidate.method == "anchored_paragraph"
    ]
    assert len(funding_candidates) == 2
    assert funding_candidates[0].accepted is False
    assert funding_candidates[1].accepted is True


def test_topical_support_false_funding_preserves_output_provenance():
    contents = _synthetic_contents("topical_support")

    resolution, metadata = _resolve_apply(contents, "active")

    assert metadata.funding_statement is None
    candidate = next(
        candidate
        for candidate in resolution.candidates
        if candidate.field == "funding_statement" and candidate.method == "anchored_paragraph"
    )
    assert candidate.text_ids == (5,)
    assert candidate.paragraph_ids == (2,)
    assert candidate.section_ids == (2,)
    assert candidate.pages == (1,)
    assert candidate.classification_source == "llm"
    assert candidate.classification_score == 0.85
    assert candidate.accepted is False


@pytest.mark.parametrize("native", [False, True], ids=["pdf_or_docx", "native_jats"])
def test_pure_resolver_works_without_llm_for_native_and_non_native_inputs(native: bool):
    contents = _contents(
        [
            (
                1,
                "Competing interests",
                CanonicalSection.COI,
                "exact_alias",
                1.0,
                ["The authors declare no competing interests."],
            )
        ],
        native=native,
    )
    _resolution, metadata = _resolve_apply(contents, "active")
    assert metadata.coi_statement == "The authors declare no competing interests."


def test_apply_preserves_prepopulated_native_scalar_and_is_idempotent():
    contents = _contents(
        [
            (
                1,
                "Competing interests",
                CanonicalSection.COI,
                "exact_alias",
                1.0,
                ["The PDF text declares no competing interests."],
            )
        ],
        native=True,
    )
    module = _resolver_module()
    resolution = module.resolve_integrity_statements(contents, mode="active")
    metadata = PaperMetadata(doi="", title="T", coi_statement="Native JATS declaration.")
    module.apply_integrity_resolution(contents, metadata, resolution)
    module.apply_integrity_resolution(contents, metadata, resolution)
    assert metadata.coi_statement == "Native JATS declaration."


def test_active_replaces_non_native_scalar_only_with_an_accepted_candidate():
    accepted = _contents(
        [
            (
                1,
                "Funding statement",
                CanonicalSection.FUNDING,
                "exact_alias",
                1.0,
                ["This work was supported by NSF grant 123."],
            )
        ]
    )
    module = _resolver_module()
    accepted_resolution = module.resolve_integrity_statements(accepted, mode="active")
    accepted_metadata = PaperMetadata(
        doi="", title="T", funding_statement="Unbounded legacy funding prose."
    )
    module.apply_integrity_resolution(accepted, accepted_metadata, accepted_resolution)
    assert accepted_metadata.funding_statement == "This work was supported by NSF grant 123."

    rejected = _contents(
        [
            (
                1,
                "Funding",
                CanonicalSection.FUNDING,
                "model",
                0.99,
                ["Funding changes political incentives."],
            )
        ]
    )
    rejected_resolution = module.resolve_integrity_statements(rejected, mode="active")
    rejected_metadata = PaperMetadata(
        doi="", title="T", funding_statement="Existing reviewed statement."
    )
    module.apply_integrity_resolution(rejected, rejected_metadata, rejected_resolution)
    assert rejected_metadata.funding_statement == "Existing reviewed statement."


def test_candidate_pages_omit_unknown_page_numbers():
    contents = _contents(
        [
            (
                1,
                "Competing interests",
                CanonicalSection.COI,
                "exact_alias",
                1.0,
                ["The authors declare no competing interests."],
            )
        ]
    )
    contents.sentences[0].page_number = None
    resolution = _resolver_module().resolve_integrity_statements(contents, mode="active")
    assert resolution.candidates
    assert all(candidate.pages == () for candidate in resolution.candidates)


def test_legacy_shadow_active_controls_and_typed_shadow_evidence():
    contents = _contents(
        [
            (
                200,
                "Ethical Choices in Shared Spaces",
                CanonicalSection.ETHICS,
                "model",
                0.999,
                ["This section discusses ethical choices in shared public spaces."],
            )
        ]
    )

    legacy_resolution, legacy = _resolve_apply(contents, "legacy")
    shadow_resolution, shadow = _resolve_apply(contents, "shadow")
    active_resolution, active = _resolve_apply(contents, "active")

    assert legacy.ethics_statement == shadow.ethics_statement
    assert legacy.ethics_statement is not None
    assert active.ethics_statement is None
    assert legacy_resolution.issues == ()
    assert any(
        isinstance(issue, ValidationIssue)
        and issue.code == "VAL_STATEMENT_SUSPECT"
        and "ethics_statement" in issue.evidence_ids
        for issue in shadow_resolution.issues
    )
    assert active_resolution.issues == ()

    repeated = _resolver_module().resolve_integrity_statements(contents, mode="shadow")
    assert repeated.issues == shadow_resolution.issues


def test_legacy_preserves_canonical_scalar_when_heading_points_to_another_field():
    contents = _contents(
        [
            (
                1,
                "Competing interests",
                CanonicalSection.FUNDING,
                "model",
                0.91,
                ["The authors declare no competing interests."],
            )
        ]
    )

    _legacy_resolution, legacy = _resolve_apply(contents, "legacy")
    shadow_resolution, shadow = _resolve_apply(contents, "shadow")
    _active_resolution, active = _resolve_apply(contents, "active")

    assert legacy.funding_statement == "The authors declare no competing interests."
    assert shadow.funding_statement == legacy.funding_statement
    assert active.funding_statement is None
    assert active.coi_statement == "The authors declare no competing interests."
    assert any(
        issue.code == "VAL_STATEMENT_SUSPECT" and "funding_statement" in issue.evidence_ids
        for issue in shadow_resolution.issues
    )
