"""Selected front-matter ownership for core metadata extraction."""

from unittest import mock

import pandas as pd
import pytest

from bibr.extract.ref_locator import RefLocator
from bibr.paper import PaperAuthor
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)

_CANONICAL_AFFILIATION_MARKERS = (
    "department",
    "division",
    "faculty",
    "school",
    "college",
    "university",
    "universite",
    "université",
    "institute",
    "institution",
    "hospital",
    "centre",
    "center",
    "laboratory",
    "academy",
)


def _candidate(
    candidate_id: str,
    text: str,
    *,
    roles: frozenset[str],
    text_ids: tuple[int, ...] = (),
    source_kind: str = "paragraph",
):
    from bibr.extract.front_matter import FrontMatterCandidate

    return FrontMatterCandidate(
        candidate_id=candidate_id,
        source_kind=source_kind,
        reading_order=int(candidate_id.removeprefix("c")),
        page=1,
        bbox=None,
        region_label="paragraph_title" if source_kind == "heading" else "text",
        font_size=None,
        font_bold=None,
        section_id=1,
        text_ids=text_ids,
        paragraph_id=None,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        roles=roles,
    )


def _resolution(*candidates, selected_ids: tuple[str, ...] | None = None):
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

    selected_ids = selected_ids or tuple(candidate.candidate_id for candidate in candidates)
    block = FrontMatterBlock(
        block_id="selected",
        candidate_ids=selected_ids,
        title_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates if "title" in candidate.roles
        ),
    )
    allowed = frozenset(
        text_id
        for candidate in candidates
        if candidate.candidate_id in selected_ids
        for text_id in candidate.text_ids
    )
    return FrontMatterResolution(
        candidates=tuple(candidates),
        blocks=(block,),
        selected_block_id="selected",
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=allowed,
        allowed_section_ids=frozenset({1}),
    )


def test_core_rows_are_intersected_with_allowed_text_ids():
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {
            "text_id": [1, 2, 3],
            "section_name": ["Title", "Title", "Title"],
            "text": ["First record", "Selected record", "Third record"],
        }
    )
    locator = RefLocator(contents)

    selected = locator.collect_core_metadata_rows(
        cutoff_iloc=3,
        allowed_text_ids=frozenset({2}),
    )

    assert selected["text_id"].tolist() == [2]


def test_render_block_context_uses_selected_candidates_and_heading_only_byline():
    from bibr.extract.core_metadata import render_block_context

    first = _candidate("c1", "Wrong record title", roles=frozenset({"title"}), text_ids=(1,))
    title = _candidate("c2", "Selected record title", roles=frozenset({"title"}), text_ids=(2,))
    heading_name = _candidate(
        "c3",
        "María de la Cruz",
        roles=frozenset({"heading", "byline"}),
        source_kind="heading",
    )
    abstract = _candidate("c4", "Selected abstract", roles=frozenset({"abstract"}), text_ids=(3,))
    resolution = _resolution(first, title, heading_name, abstract, selected_ids=("c2", "c3", "c4"))

    context = render_block_context(resolution)

    assert context == "Selected record title\nMaría de la Cruz\nSelected abstract"
    assert "Wrong record title" not in context


@pytest.mark.parametrize("boundary_role", ["abstract", "doi"])
def test_author_context_falls_back_when_byline_roles_cross_record_boundary(boundary_role):
    from bibr.extract.core_metadata import render_author_context, render_block_context

    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    first_byline = _candidate(
        "c2",
        "Alice Example",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )
    boundary = _candidate(
        "c3",
        "Boundary evidence",
        roles=frozenset({boundary_role}),
        text_ids=(3,),
    )
    late_byline = _candidate(
        "c4",
        "Bob Sample",
        roles=frozenset({"byline"}),
        text_ids=(4,),
    )
    resolution = _resolution(title, first_byline, boundary, late_byline)
    full_text = render_block_context(resolution)

    assert render_author_context(resolution, full_text=full_text) == full_text


def test_author_context_zone_guards_fall_back_to_full_text():
    from bibr.extract.core_metadata import render_author_context, render_block_context

    byline = _candidate(
        "c2", "Alice Example, Bob Sample", roles=frozenset({"byline"}), text_ids=(2,)
    )
    affiliation = _candidate(
        "c3",
        "Department of Biology, Example University",
        roles=frozenset({"affiliation"}),
        text_ids=(3,),
    )
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    second_title = _candidate("c4", "Parallel title", roles=frozenset({"title"}), text_ids=(4,))

    # Zero title-role rows: the zone cannot anchor.
    resolution = _resolution(byline, affiliation)
    full_text = render_block_context(resolution)
    assert render_author_context(resolution, full_text=full_text) == full_text

    # Two title-role rows (parallel-language records): ambiguous anchor.
    resolution = _resolution(title, byline, second_title, affiliation)
    full_text = render_block_context(resolution)
    assert render_author_context(resolution, full_text=full_text) == full_text

    # Byline printed above the title: the zone cannot represent the layout.
    resolution = _resolution(byline, title, affiliation)
    full_text = render_block_context(resolution)
    assert render_author_context(resolution, full_text=full_text) == full_text


def _author(author_id: int, given: str, family: str, *, role: list[str] | None = None):
    return PaperAuthor(
        author_id=author_id,
        given=given,
        family=family,
        affiliation="",
        role=role or [],
    )


def test_author_grounding_ignores_page_one_name_outside_selected_block():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    selected = _candidate(
        "c1", "Alice van der Meer", roles=frozenset({"heading", "byline"}), source_kind="heading"
    )
    outside = _candidate(
        "c2", "Mallory Example", roles=frozenset({"heading", "byline"}), source_kind="heading"
    )
    resolution = _resolution(selected, outside, selected_ids=("c1",))

    issues = assess_author_grounding(
        [_author(1, "Alice", "van der Meer"), _author(2, "Mallory", "Example")],
        build_byline_group(resolution),
    )

    assert [issue.code for issue in issues] == ["VAL_AUTHOR_UNGROUNDED"]
    assert issues[0].count == 1
    assert "author:2" in issues[0].evidence_ids
    assert "c2" not in issues[0].evidence_ids


def test_author_grounding_reports_empty_and_strict_prefix_as_missing():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    byline = _candidate(
        "c1", "Alice Example, Bob Sample", roles=frozenset({"byline"}), text_ids=(2,)
    )
    group = build_byline_group(_resolution(byline))

    empty_issues = assess_author_grounding([], group)
    prefix_issues = assess_author_grounding([_author(1, "Alice", "Example")], group)

    assert [issue.code for issue in empty_issues] == ["VAL_AUTHOR_MISSING"]
    assert [issue.code for issue in prefix_issues] == ["VAL_AUTHOR_MISSING"]
    assert "strict prefix" in prefix_issues[0].message.casefold()
    assert "reason:strict_prefix_or_truncation" in prefix_issues[0].evidence_ids


@pytest.mark.parametrize(
    ("byline", "authors"),
    [
        (
            "María de la Cruz Jr.; Global Health Consortium",
            [
                _author(1, "María", "de la Cruz Jr."),
                _author(2, "", "Global Health Consortium", role=["organization"]),
            ],
        ),
        ("王 小明; 李 静", [_author(1, "小明", "王"), _author(2, "静", "李")]),
    ],
)
def test_author_grounding_preserves_particles_suffixes_consortiums_and_name_order(byline, authors):
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate("c1", byline, roles=frozenset({"byline"}), text_ids=(2,))

    assert assess_author_grounding(authors, build_byline_group(_resolution(candidate))) == ()


def test_author_grounding_accepts_clean_forty_author_byline():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    authors = [_author(index, f"Given{index}", f"Family{index}") for index in range(1, 41)]
    candidate = _candidate(
        "c1",
        "; ".join(f"Given{index} Family{index}" for index in range(1, 41)),
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )

    assert assess_author_grounding(authors, build_byline_group(_resolution(candidate))) == ()


def test_author_grounding_reports_internal_byline_omission():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "Alice Example, Bob Sample, Carol Other",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )
    authors = [_author(1, "Alice", "Example"), _author(2, "Carol", "Other")]

    issues = assess_author_grounding(authors, build_byline_group(_resolution(candidate)))

    assert [issue.code for issue in issues] == ["VAL_AUTHOR_MISSING"]
    assert issues[0].count == 1
    assert "reason:byline_omission" in issues[0].evidence_ids


@pytest.mark.parametrize(
    ("source_name", "given", "family"),
    [
        ("Alice O’Brien", "Alice", "O'Brien"),
        ("Jean–Luc Picard", "Jean-Luc", "Picard"),
    ],
)
def test_author_grounding_normalizes_unicode_punctuation_variants(source_name, given, family):
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate("c1", source_name, roles=frozenset({"byline"}), text_ids=(2,))

    assert (
        assess_author_grounding(
            [_author(1, given, family)],
            build_byline_group(_resolution(candidate)),
        )
        == ()
    )


def test_author_grounding_does_not_treat_affiliation_prose_as_truncation():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "Alice Example National Health Service",
        roles=frozenset({"byline", "affiliation"}),
        text_ids=(2,),
    )

    assert (
        assess_author_grounding(
            [_author(1, "Alice", "Example")],
            build_byline_group(_resolution(candidate)),
        )
        == ()
    )


@pytest.mark.parametrize(
    ("author", "expected_codes"),
    [
        (
            _author(1, "", "Health Consortium", role=["organization"]),
            ["VAL_AUTHOR_UNGROUNDED"],
        ),
        (
            _author(1, "", "Smith"),
            ["VAL_AUTHOR_UNGROUNDED", "VAL_AUTHOR_MISSING"],
        ),
    ],
)
def test_author_grounding_rejects_partial_consortium_and_surname_only_matches(
    author, expected_codes
):
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    source = "Global Health Consortium" if author.role else "Alice Smith"
    candidate = _candidate("c1", source, roles=frozenset({"byline"}), text_ids=(2,))

    issues = assess_author_grounding([author], build_byline_group(_resolution(candidate)))

    assert [issue.code for issue in issues] == expected_codes


def test_author_converter_preserves_real_organization_author():
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.schemas import AuthorLLM

    authors = CoreMetadataExtractor._convert_llm_authors(
        [
            AuthorLLM(
                given="",
                family="The WHO Study Group",
                affiliation="",
                role=["organization"],
            )
        ]
    )

    assert [(author.given, author.family, author.role) for author in authors] == [
        ("", "The WHO Study Group", ["organization"])
    ]


def test_composite_record_prose_does_not_expand_expected_author_set():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    composite = _candidate(
        "c1",
        "SELECTED HEALTH OUTCOMES Alice Example, Bob Sample, Carol Other "
        "National Health Service Abstract This study reports outcomes.",
        roles=frozenset({"title", "byline", "affiliation", "abstract"}),
        text_ids=(1, 2, 3),
    )
    authors = [
        _author(1, "Alice", "Example"),
        _author(2, "Bob", "Sample"),
        _author(3, "Carol", "Other"),
    ]

    assert (
        assess_author_grounding(
            authors,
            build_byline_group(_resolution(composite)),
        )
        == ()
    )


def test_proceedings_composite_candidate_does_not_fabricate_missing_authors():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group
    from bibr.pipeline.identity import ExpectedIdentity
    from tests.extract.test_front_matter import (
        _front_matter_module,
        _proceedings_composite_contents,
    )

    resolution, issues = _front_matter_module().resolve_front_matter(
        _proceedings_composite_contents(),
        expected_identity=ExpectedIdentity(
            queue_record_id="synthetic-proceedings",
            expected_title=(
                "What Do Students Learn About River Systems? "
                "An Analysis of Introductory Science Textbooks"
            ),
            doi_required=True,
        ),
        target_required=True,
    )
    target_candidate = next(
        candidate
        for candidate in resolution.candidates
        if candidate.text_ids == tuple(range(38, 54))
    )

    assert len(target_candidate.raw_text) > 1_800
    assert issues == ()
    grounding_issues = assess_author_grounding(
        [
            _author(1, "Morgan", "Sample"),
            _author(2, "Avery", "Reader"),
            _author(3, "Riley", "Example"),
        ],
        build_byline_group(resolution),
    )

    assert {issue.code for issue in grounding_issues}.isdisjoint(
        {"VAL_AUTHOR_MISSING", "VAL_AUTHOR_UNGROUNDED"}
    )
    partial_issues = assess_author_grounding(
        [_author(1, "Morgan", "Sample"), _author(2, "Avery", "Reader")],
        build_byline_group(resolution),
    )
    assert [(issue.code, issue.count) for issue in partial_issues] == [("VAL_AUTHOR_MISSING", 1)]
    empty_issues = assess_author_grounding([], build_byline_group(resolution))
    assert [(issue.code, issue.count) for issue in empty_issues] == [("VAL_AUTHOR_MISSING", 1)]
    assert "reason:byline_present_authors_empty" in empty_issues[0].evidence_ids


@pytest.mark.parametrize(
    ("authors", "expected_count"),
    [
        ([_author(1, "Alice", "Example")], 1),
        ([], 2),
    ],
)
def test_mixed_byline_affiliation_infers_only_names_before_boundary(authors, expected_count):
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "Alice Example, Bob Sample; Department of Biology, Example University",
        roles=frozenset({"byline", "affiliation"}),
        text_ids=(2,),
    )

    issues = assess_author_grounding(authors, build_byline_group(_resolution(candidate)))

    assert [(issue.code, issue.count) for issue in issues] == [
        ("VAL_AUTHOR_MISSING", expected_count)
    ]


def test_mixed_byline_abstract_infers_only_names_before_boundary():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "Alice Example, Bob Sample Abstract Results follow",
        roles=frozenset({"byline", "abstract"}),
        text_ids=(2,),
    )

    issues = assess_author_grounding(
        [_author(1, "Alice", "Example")],
        build_byline_group(_resolution(candidate)),
    )

    assert [(issue.code, issue.count) for issue in issues] == [("VAL_AUTHOR_MISSING", 1)]


@pytest.mark.parametrize(
    ("text", "roles"),
    [
        (
            "CLINICAL OUTCOMES Alzheimer Disease, Alice Example, Bob Sample Abstract Results",
            frozenset({"title", "byline", "abstract"}),
        ),
        (
            "CLINICAL OUTCOMES IN Alzheimer Disease, Alice Example, Bob Sample Abstract Results",
            frozenset({"title", "byline", "abstract"}),
        ),
        (
            "CLINICAL OUTCOMES Alzheimer Disease, Alice Example, Bob Sample "
            "1. Department of Biology",
            frozenset({"title", "byline", "affiliation"}),
        ),
        (
            "CLINICAL OUTCOMES IN Alzheimer Disease, Alice Example, Bob Sample "
            "1. Department of Biology",
            frozenset({"title", "byline", "affiliation"}),
        ),
    ],
)
def test_title_composite_uses_emitted_authors_as_missing_inference_anchors(text, roles):
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        text,
        roles=roles,
        text_ids=(2,),
    )
    group = build_byline_group(_resolution(candidate))

    complete_issues = assess_author_grounding(
        [_author(1, "Alice", "Example"), _author(2, "Bob", "Sample")],
        group,
    )
    alice_only_issues = assess_author_grounding([_author(1, "Alice", "Example")], group)
    empty_issues = assess_author_grounding([], group)

    assert complete_issues == ()
    assert [(issue.code, issue.count) for issue in alice_only_issues] == [("VAL_AUTHOR_MISSING", 1)]
    assert [(issue.code, issue.count) for issue in empty_issues] == [("VAL_AUTHOR_MISSING", 1)]
    assert "reason:byline_present_authors_empty" in empty_issues[0].evidence_ids


def test_title_composite_does_not_ground_authors_from_post_boundary_prose():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "CLINICAL OUTCOMES Alice Example Abstract Carol Other described outcomes "
        "and Bob Sample confirmed them",
        roles=frozenset({"title", "byline", "abstract"}),
        text_ids=(2,),
    )
    group = build_byline_group(_resolution(candidate))

    post_boundary_issues = assess_author_grounding(
        [_author(1, "Carol", "Other"), _author(2, "Bob", "Sample")],
        group,
    )
    alice_issues = assess_author_grounding([_author(1, "Alice", "Example")], group)

    assert [(issue.code, issue.count) for issue in post_boundary_issues] == [
        ("VAL_AUTHOR_UNGROUNDED", 2)
    ]
    assert alice_issues == ()


@pytest.mark.parametrize("marker", _CANONICAL_AFFILIATION_MARKERS)
def test_title_composite_grounding_honors_every_canonical_affiliation_marker(marker):
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        f"CLINICAL OUTCOMES Alice Example {marker} of Biology "
        "Carol Other described outcomes and Bob Sample confirmed them",
        roles=frozenset({"title", "byline", "affiliation"}),
        text_ids=(2,),
    )
    group = build_byline_group(_resolution(candidate))

    post_boundary_issues = assess_author_grounding(
        [_author(1, "Carol", "Other"), _author(2, "Bob", "Sample")],
        group,
    )

    assert [(issue.code, issue.count) for issue in post_boundary_issues] == [
        ("VAL_AUTHOR_UNGROUNDED", 2)
    ]
    assert assess_author_grounding([_author(1, "Alice", "Example")], group) == ()


def test_core_grounding_reuses_front_matter_affiliation_marker_contract():
    from bibr.extract import core_metadata, front_matter

    assert front_matter.AFFILIATION_MARKERS == _CANONICAL_AFFILIATION_MARKERS
    assert core_metadata.AFFILIATION_MARKER_RE is front_matter.AFFILIATION_MARKER_RE


@pytest.mark.parametrize(
    ("text", "roles"),
    [
        (
            "Clinical Outcomes and Social Care Alice Example Abstract Results follow",
            frozenset({"title", "byline", "abstract"}),
        ),
        (
            "Alice Example Department of Health and Social Care",
            frozenset({"byline", "affiliation"}),
        ),
        (
            "Alice Example Abstract We measured health and social outcomes",
            frozenset({"byline", "abstract"}),
        ),
    ],
)
def test_composite_candidate_residual_prose_does_not_create_missing_authors(text, roles):
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate("c1", text, roles=roles, text_ids=(2,))

    assert (
        assess_author_grounding(
            [_author(1, "Alice", "Example")],
            build_byline_group(_resolution(candidate)),
        )
        == ()
    )


def test_conjoined_exact_organization_author_grounds_without_allowing_partial_match():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "Alice Example and Global Health Consortium",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )
    group = build_byline_group(_resolution(candidate))

    exact_issues = assess_author_grounding(
        [
            _author(1, "Alice", "Example"),
            _author(2, "", "Global Health Consortium", role=["organization"]),
        ],
        group,
    )
    partial_issues = assess_author_grounding(
        [
            _author(1, "Alice", "Example"),
            _author(2, "", "Health Consortium", role=["organization"]),
        ],
        group,
    )

    assert exact_issues == ()
    assert "VAL_AUTHOR_UNGROUNDED" in {issue.code for issue in partial_issues}


def test_author_grounding_emits_missing_and_ungrounded_together():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "Alice Example, Bob Sample, Carol Other",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )
    authors = [
        _author(1, "Alice", "Example"),
        _author(2, "Mallory", "Outside"),
        _author(3, "Carol", "Other"),
    ]

    issues = assess_author_grounding(authors, build_byline_group(_resolution(candidate)))

    assert [issue.code for issue in issues] == [
        "VAL_AUTHOR_UNGROUNDED",
        "VAL_AUTHOR_MISSING",
    ]
    assert [issue.count for issue in issues] == [1, 1]
    assert "reason:byline_omission" in issues[1].evidence_ids


def test_author_grounding_is_one_to_one_for_duplicate_llm_names():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1", "Alice Example, Bob Sample", roles=frozenset({"byline"}), text_ids=(2,)
    )
    authors = [
        _author(1, "Alice", "Example"),
        _author(2, "Alice", "Example"),
        _author(3, "Bob", "Sample"),
    ]

    issues = assess_author_grounding(authors, build_byline_group(_resolution(candidate)))

    assert [issue.code for issue in issues] == ["VAL_AUTHOR_UNGROUNDED"]
    assert issues[0].count == 1
    assert "author:2" in issues[0].evidence_ids


def test_author_grounding_does_not_reuse_overlapping_name_span():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate("c1", "Alice Example Jr.", roles=frozenset({"byline"}), text_ids=(2,))
    authors = [
        _author(1, "Alice", "Example"),
        _author(2, "Alice", "Example Jr."),
    ]

    issues = assess_author_grounding(authors, build_byline_group(_resolution(candidate)))

    assert issues[0].code == "VAL_AUTHOR_UNGROUNDED"
    assert issues[0].count == 1
    assert "author:2" in issues[0].evidence_ids


def _paper_contents(rows: list[tuple[int, str, int]]) -> PaperContents:
    sections = [
        PaperSection(0, "Title", 0, None, CanonicalSection.TITLE, 1.0),
        PaperSection(1, "Introduction", 1, None, CanonicalSection.INTRODUCTION, 1.0),
    ]
    return PaperContents(
        sentences=[
            PaperSentence(
                text_id=text_id,
                text=text,
                section_id=section_id,
                paragraph_id=text_id,
                page_number=1,
            )
            for text_id, text, section_id in rows
        ],
        sections=sections,
        tables=[],
        links=[],
        sections_text={0: "", 1: ""},
    )


def make_extractor_with_captured_core_call(resolution, monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.schemas import CoreMetadataLLM

    contents = _paper_contents(
        [
            (text_id, candidate.raw_text, 0)
            for candidate in resolution.candidates
            for text_id in candidate.text_ids
        ]
        + [(99, "Body begins", 1)]
    )
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(title="Selected title", authors=[])
    )
    from bibr.schemas import AuthorsLLM

    llm.extract_authors = mock.AsyncMock(return_value=AuthorsLLM(authors=[]))
    settings = snapshot_settings()
    settings.llm.per_task_context = True
    settings.llm.merged_core_metadata = False
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        settings=settings,
        front_matter_resolution=resolution,
    )
    monkeypatch.setattr(
        extractor,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )
    return extractor, llm


def make_empty_author_extractor(
    monkeypatch,
    *,
    source: str,
    recovered=None,
    recovery_error=None,
    mark_byline: bool = True,
):
    """Extractor whose primary core call returns schema-valid empty authors.

    ``source`` is the printed author row; ``recovered`` is what the single
    free-JSON recovery attempt returns (ignored when ``recovery_error`` is
    given).
    """

    from bibr.schemas import AuthorsLLM

    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate(
        "c2",
        source,
        roles=frozenset({"byline"}) if mark_byline else frozenset(),
        text_ids=(2,),
    )
    affiliation = _candidate(
        "c3",
        "Department of Biology, Example University",
        roles=frozenset({"affiliation"}),
        text_ids=(3,),
    )
    resolution = _resolution(title, byline, affiliation)
    extractor, llm = make_extractor_with_captured_core_call(resolution, monkeypatch)
    if recovery_error is not None:
        llm.extract_authors = mock.AsyncMock(side_effect=recovery_error)
    else:
        llm.extract_authors = mock.AsyncMock(return_value=AuthorsLLM(authors=list(recovered or [])))
    return extractor, llm


async def test_affiliation_only_role_slice_falls_back_to_full_selected_block(monkeypatch):
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    first_author = _candidate(
        "c2",
        "Mahdieh Khorsandifard1",
        roles=frozenset(),
        text_ids=(2,),
    )
    second_author = _candidate(
        "c3",
        "Kian Jafari1",
        roles=frozenset(),
        text_ids=(3,),
    )
    third_author = _candidate(
        "c4",
        "Arash Sheikhaleh1",
        roles=frozenset(),
        text_ids=(4,),
    )
    affiliation = _candidate(
        "c5",
        "1 Department of Engineering, Example University",
        roles=frozenset({"affiliation"}),
        text_ids=(5,),
    )
    resolution = _resolution(
        title,
        first_author,
        second_author,
        third_author,
        affiliation,
    )
    extractor, llm = make_extractor_with_captured_core_call(resolution, monkeypatch)

    await extractor.extract()

    authors_text = llm.extract_core_metadata.await_args.kwargs["authors_text"]
    assert "Mahdieh Khorsandifard" in authors_text
    assert "Kian Jafari" in authors_text
    assert "Arash Sheikhaleh" in authors_text


async def test_recognized_byline_keeps_narrow_author_slice(monkeypatch):
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate(
        "c2",
        "Alice Example, Bob Sample",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )
    abstract = _candidate(
        "c3",
        "Alice and Bob studied a large cohort.",
        roles=frozenset({"abstract"}),
        text_ids=(3,),
    )
    resolution = _resolution(title, byline, abstract)
    extractor, llm = make_extractor_with_captured_core_call(resolution, monkeypatch)

    await extractor.extract()

    authors_text = llm.extract_core_metadata.await_args.kwargs["authors_text"]
    assert "Alice Example" in authors_text
    assert "studied a large cohort" not in authors_text


async def test_partial_byline_role_keeps_adjacent_unlabelled_author_rows(monkeypatch):
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    recognized = _candidate(
        "c2",
        "Smith, Alice",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )
    second_author = _candidate(
        "c3",
        "Bob Sample",
        roles=frozenset(),
        text_ids=(3,),
    )
    third_author = _candidate(
        "c4",
        "Carla Scholar",
        roles=frozenset(),
        text_ids=(4,),
    )
    affiliation = _candidate(
        "c5",
        "Department of Biology, Example University",
        roles=frozenset({"affiliation"}),
        text_ids=(5,),
    )
    abstract = _candidate(
        "c6",
        "Abstract evidence must stay outside the author prompt.",
        roles=frozenset({"abstract"}),
        text_ids=(6,),
    )
    resolution = _resolution(
        title,
        recognized,
        second_author,
        third_author,
        affiliation,
        abstract,
    )
    extractor, llm = make_extractor_with_captured_core_call(resolution, monkeypatch)

    await extractor.extract()

    authors_text = llm.extract_core_metadata.await_args.kwargs["authors_text"]
    assert "Smith, Alice" in authors_text
    assert "Bob Sample" in authors_text
    assert "Carla Scholar" in authors_text
    assert "Department of Biology" in authors_text
    assert "Abstract evidence" not in authors_text


async def test_false_byline_role_cannot_suppress_preceding_unlabelled_authors(monkeypatch):
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    first_author = _candidate(
        "c2",
        "Alice Example",
        roles=frozenset(),
        text_ids=(2,),
    )
    second_author = _candidate(
        "c3",
        "Bob Sample",
        roles=frozenset(),
        text_ids=(3,),
    )
    false_byline = _candidate(
        "c4",
        "Children’s Health · Women’s Studies",
        roles=frozenset({"byline"}),
        text_ids=(4,),
    )
    affiliation = _candidate(
        "c5",
        "Department of Biology, Example University",
        roles=frozenset({"affiliation"}),
        text_ids=(5,),
    )
    abstract = _candidate(
        "c6",
        "Abstract evidence must stay outside the author prompt.",
        roles=frozenset({"abstract"}),
        text_ids=(6,),
    )
    resolution = _resolution(
        title,
        first_author,
        second_author,
        false_byline,
        affiliation,
        abstract,
    )
    extractor, llm = make_extractor_with_captured_core_call(resolution, monkeypatch)

    await extractor.extract()

    authors_text = llm.extract_core_metadata.await_args.kwargs["authors_text"]
    assert "Alice Example" in authors_text
    assert "Bob Sample" in authors_text
    assert "Abstract evidence" not in authors_text


@pytest.mark.parametrize(
    "topic",
    [
        "Clinical Medicine · Health Economics",
        "Evidence-Based Medicine · Patient-Centered Care",
        "WHO Guidance · OECD Report",
        "Phase2 Study · Model3 Evaluation",
    ],
)
async def test_false_separator_topic_does_not_narrow_away_unlabelled_authors(
    monkeypatch,
    topic,
):
    from bibr.extract.front_matter import resolve_front_matter

    contents = _paper_contents(
        [
            (1, "Selected Study of Community Health", 0),
            (2, "Alice Example", 0),
            (3, "Bob Sample", 0),
            (4, topic, 0),
            (5, "Department of Biology, Example University", 0),
            (6, "Body begins", 1),
        ]
    )
    resolution, issues = resolve_front_matter(contents)
    topic = next(candidate for candidate in resolution.candidates if candidate.text_ids == (4,))

    assert issues == ()
    assert resolution.selected_block_id is not None
    assert "byline" not in topic.roles

    extractor, llm = make_extractor_with_captured_core_call(resolution, monkeypatch)
    await extractor.extract()

    authors_text = llm.extract_core_metadata.await_args.kwargs["authors_text"]
    assert "Alice Example" in authors_text
    assert "Bob Sample" in authors_text


async def test_core_llm_receives_only_selected_block_and_heading_byline(monkeypatch):
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    contents = _paper_contents(
        [
            (1, "Wrong record and wrong.person@example.test", 0),
            (2, "Selected record title", 0),
            (3, "Body begins", 1),
        ]
    )
    wrong = _candidate("c1", "Wrong record", roles=frozenset({"title"}), text_ids=(1,))
    title = _candidate("c2", "Selected record title", roles=frozenset({"title"}), text_ids=(2,))
    heading_name = _candidate(
        "c3",
        "María de la Cruz",
        roles=frozenset({"heading", "byline"}),
        source_kind="heading",
    )
    resolution = _resolution(wrong, title, heading_name, selected_ids=("c2", "c3"))
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="Selected record title",
            authors=[AuthorLLM(given="María", family="de la Cruz")],
        )
    )
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        front_matter_resolution=resolution,
    )
    monkeypatch.setattr(
        extractor,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )

    await extractor.extract()

    (full_text,), kwargs = llm.extract_core_metadata.await_args
    assert full_text == "Selected record title\nMaría de la Cruz"
    assert "Wrong record" not in full_text
    if kwargs["authors_text"] is not None:
        assert "María de la Cruz" in kwargs["authors_text"]


async def test_core_extractor_retains_ungrounded_author_and_records_typed_issue(monkeypatch):
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    contents = _paper_contents([(1, "Selected title", 0), (2, "Alice Example", 0), (3, "Body", 1)])
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate("c2", "Alice Example", roles=frozenset({"byline"}), text_ids=(2,))
    resolution = _resolution(title, byline)
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="Selected title",
            authors=[
                AuthorLLM(given="Alice", family="Example"),
                AuthorLLM(given="Mallory", family="Outside"),
            ],
        )
    )
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        front_matter_resolution=resolution,
    )
    monkeypatch.setattr(
        extractor,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )

    metadata = await extractor.extract()

    assert [author.family for author in metadata.authors] == ["Example", "Outside"]
    assert [issue.code for issue in extractor.validation_issues] == ["VAL_AUTHOR_UNGROUNDED"]


async def test_unresolved_resolution_skips_core_llm_and_returns_empty_scalars():
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

    contents = _paper_contents([(1, "First record", 0), (2, "Second record", 0)])
    candidates = (
        _candidate("c1", "First record", roles=frozenset({"title"}), text_ids=(1,)),
        _candidate("c2", "Second record", roles=frozenset({"title"}), text_ids=(2,)),
    )
    resolution = FrontMatterResolution(
        candidates=candidates,
        blocks=(
            FrontMatterBlock("b1", ("c1",), ("c1",)),
            FrontMatterBlock("b2", ("c2",), ("c2",)),
        ),
        selected_block_id=None,
        selection_method="abstained",
        reason_flags=("multiple_plausible_blocks",),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset({0}),
    )
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock()

    metadata = await CoreMetadataExtractor(
        contents,
        llm_client=llm,
        front_matter_resolution=resolution,
    ).extract()

    llm.extract_core_metadata.assert_not_awaited()
    assert metadata.title == ""
    assert metadata.doi == ""
    assert metadata.abstract == ""
    assert metadata.authors == []
    assert metadata.keywords == []


async def test_selected_block_does_not_harvest_email_from_unowned_page_one(monkeypatch):
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    contents = _paper_contents(
        [
            (1, "Selected title", 0),
            (2, "Alice Example", 0),
            (3, "Corresponding author: Alice Example alice@outside.test", 0),
            (4, "Body begins", 1),
        ]
    )
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate("c2", "Alice Example", roles=frozenset({"byline"}), text_ids=(2,))
    outside = _candidate(
        "c3",
        "Corresponding author: Alice Example alice@outside.test",
        roles=frozenset(),
        text_ids=(3,),
    )
    resolution = _resolution(title, byline, outside, selected_ids=("c1", "c2"))
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="Selected title",
            authors=[AuthorLLM(given="Alice", family="Example")],
        )
    )
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        front_matter_resolution=resolution,
    )
    monkeypatch.setattr(
        extractor,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )

    metadata = await extractor.extract()

    assert metadata.authors[0].email is None
    assert metadata.authors[0].corresponding is False


@pytest.mark.parametrize(
    ("block_doi", "headers", "footers", "expected"),
    [
        # Printed only in a running-header citation line above the title.
        (
            None,
            ["2017. Proc Example Soc 2, 20:1-15. https://doi.org/10.1234/pes.4064."],
            [],
            "10.1234/pes.4064",
        ),
        # The selected block's DOI is never overridden by the furniture's.
        ("DOI: 10.1234/block", ["https://doi.org/10.1234/other"], [], "10.1234/block"),
        # Furniture DOIs still pass the non-self rules.
        (
            None,
            ["Supplementary DOI: 10.1234/pes.supp"],
            ["12. Smith J. https://doi.org/10.1234/ref"],
            "",
        ),
    ],
)
async def test_selected_block_doi_falls_back_to_page_furniture(
    monkeypatch, block_doi, headers, footers, expected
):
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    rows = [(1, "Selected title", 0), (2, "Alice Example", 0)]
    candidates = [
        _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,)),
        _candidate("c2", "Alice Example", roles=frozenset({"byline"}), text_ids=(2,)),
    ]
    if block_doi is not None:
        rows.append((3, block_doi, 0))
        candidates.append(_candidate("c3", block_doi, roles=frozenset({"doi"}), text_ids=(3,)))
    contents = _paper_contents([*rows, (4, "Body begins", 1)])
    contents.detected_headers = headers
    contents.detected_footers = footers
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="Selected title",
            authors=[AuthorLLM(given="Alice", family="Example")],
        )
    )
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        front_matter_resolution=_resolution(*candidates),
    )
    monkeypatch.setattr(
        extractor,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )

    metadata = await extractor.extract()

    assert metadata.doi == expected
    (full_text,), _kwargs = llm.extract_core_metadata.await_args
    assert "Proc Example Soc" not in full_text


async def test_extract_phase_threads_resolution_to_metadata_extractor(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.pipeline.stages import post_parse as post_parse_module

    contents = _paper_contents([(1, "Selected title", 0), (2, "Body", 1)])
    candidate = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    resolution = _resolution(candidate)
    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False
    captured = {}

    class FakeExtractor:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        async def extract_all_metadata(self):
            from bibr.paper import PaperMetadata

            return PaperMetadata(doi="", title="Selected title")

    monkeypatch.setattr("bibr.extract.extractor.MetadataExtractor", FakeExtractor)

    metadata = await post_parse_module._extract_metadata_and_equations(
        contents,
        file_hash="hash",
        no_llm=False,
        llm_client=mock.MagicMock(),
        front_matter_resolution=resolution,
        settings=settings,
    )

    assert captured["front_matter_resolution"] is resolution
    assert metadata.title == "Selected title"


async def test_preparsed_native_metadata_ignores_abstained_resolution(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.extract.front_matter import FrontMatterResolution
    from bibr.paper import PaperMetadata
    from bibr.pipeline.stages import post_parse as post_parse_module

    native = PaperMetadata(doi="10.1234/native", title="Native title")
    contents = _paper_contents([])
    contents.preparsed_metadata = native
    resolution = FrontMatterResolution(
        candidates=(),
        blocks=(),
        selected_block_id=None,
        selection_method="abstained",
        reason_flags=("multiple_plausible_blocks",),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )
    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False

    metadata = await post_parse_module._extract_metadata_and_equations(
        contents,
        file_hash="hash",
        no_llm=False,
        llm_client=mock.MagicMock(),
        front_matter_resolution=resolution,
        ref_parse_strategy="off",
        settings=settings,
    )

    assert metadata is native
    assert metadata.title == "Native title"
    assert metadata.doi == "10.1234/native"


async def test_untargeted_no_llm_abstention_preserves_non_llm_fallbacks(monkeypatch):
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution
    from bibr.pipeline.stages import post_parse as post_parse_module

    contents = _paper_contents(
        [
            (1, "First record", 0),
            (2, "Abstract text from an unresolved record.", 0),
            (3, "Body", 1),
        ]
    )
    contents.detected_title = "Layout fallback title"
    contents.sections[0].section_type = CanonicalSection.ABSTRACT
    candidates = (
        _candidate("c1", "First record", roles=frozenset({"title"}), text_ids=(1,)),
        _candidate("c2", "Second record", roles=frozenset({"title"}), text_ids=(2,)),
    )
    resolution = FrontMatterResolution(
        candidates=candidates,
        blocks=(
            FrontMatterBlock("b1", ("c1",), ("c1",)),
            FrontMatterBlock("b2", ("c2",), ("c2",)),
        ),
        selected_block_id=None,
        selection_method="abstained",
        reason_flags=("multiple_plausible_blocks",),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset({0}),
    )

    async def no_op(*_args, **_kwargs):
        return None

    def attach(actual_contents, *_args, **_kwargs):
        actual_contents.front_matter_resolution = resolution
        return ()

    monkeypatch.setattr(post_parse_module, "_classify_sections", no_op)
    monkeypatch.setattr(post_parse_module, "_normalize_section_structure", no_op)
    monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach)

    paper = await post_parse_module.post_parse(
        contents,
        "multi.pdf",
        "hash",
        no_llm=True,
        ocr_metadata={
            "title": "OCR fallback title",
            "doi": "10.1234/ocr",
            "keywords": ["fallback"],
            "authors": ["Fallback Author"],
        },
    )

    assert paper.metadata.title == "Layout fallback title"
    assert paper.metadata.doi == "10.1234/ocr"
    assert "Abstract text from an unresolved record." in paper.metadata.abstract
    assert paper.metadata.keywords == ["fallback"]
    assert [author.family for author in paper.metadata.authors] == ["Author"]
    assert paper.validation_issues == []


async def test_selected_block_ignores_global_detected_title_and_ocr_fallbacks(monkeypatch):
    from bibr.paper import PaperMetadata
    from bibr.pipeline.stages import post_parse as post_parse_module

    contents = _paper_contents([(1, "Selected title", 0), (2, "Alice Example", 0)])
    contents.detected_title = "Outside layout title"
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate("c2", "Alice Example", roles=frozenset({"byline"}), text_ids=(2,))
    resolution = _resolution(title, byline)

    async def no_op(*_args, **_kwargs):
        return None

    async def extract(*_args, **_kwargs):
        return PaperMetadata(doi="", title="Selected title", authors=[])

    def attach(actual_contents, *_args, **_kwargs):
        actual_contents.front_matter_resolution = resolution
        return ()

    monkeypatch.setattr(post_parse_module, "_classify_sections", no_op)
    monkeypatch.setattr(post_parse_module, "_normalize_section_structure", no_op)
    monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach)
    monkeypatch.setattr(post_parse_module, "_extract_metadata_and_equations", extract)
    monkeypatch.setattr(post_parse_module, "_link_citations", no_op)
    monkeypatch.setattr(
        "bibr.extract.research_integrity.extract_structured_integrity",
        no_op,
    )

    class UnusedLlm:
        pass

    paper = await post_parse_module.post_parse(
        contents,
        "selected.pdf",
        "hash",
        llm_client=UnusedLlm(),
        ref_parse_strategy="off",
        ocr_metadata={
            "title": "Outside OCR title",
            "doi": "10.9999/outside.ocr",
            "keywords": ["outside"],
            "authors": ["Outside Author"],
        },
    )

    assert paper.metadata.title == "Selected title"
    assert paper.metadata.doi == ""
    assert paper.metadata.keywords == []
    assert paper.metadata.authors == []


async def test_author_grounding_issue_reaches_paper_and_export(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.pipeline.stages import post_parse as post_parse_module
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    contents = _paper_contents([(1, "Selected title", 0), (2, "Alice Example", 0), (3, "Body", 1)])
    contents.detected_title = "Selected title"
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate("c2", "Alice Example", roles=frozenset({"byline"}), text_ids=(2,))
    resolution = _resolution(title, byline)
    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False

    class FakeLlm:
        async def extract_core_metadata(self, *_args, **_kwargs):
            return CoreMetadataLLM(
                title="Selected title",
                authors=[
                    AuthorLLM(given="Alice", family="Example"),
                    AuthorLLM(given="Mallory", family="Outside"),
                ],
            )

    async def no_op(*_args, **_kwargs):
        return None

    def attach(actual_contents, *_args, **_kwargs):
        actual_contents.front_matter_resolution = resolution
        return ()

    monkeypatch.setattr(post_parse_module, "_classify_sections", no_op)
    monkeypatch.setattr(post_parse_module, "_normalize_section_structure", no_op)
    monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach)
    monkeypatch.setattr(post_parse_module, "_link_citations", no_op)
    monkeypatch.setattr(
        "bibr.extract.research_integrity.extract_structured_integrity",
        no_op,
    )
    monkeypatch.setattr(
        CoreMetadataExtractor,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )

    paper = await post_parse_module.post_parse(
        contents,
        "selected.pdf",
        "hash",
        llm_client=FakeLlm(),
        ref_parse_strategy="off",
        settings=settings,
    )

    assert [author.family for author in paper.metadata.authors] == ["Example", "Outside"]
    assert [issue.code for issue in paper.validation_issues] == ["VAL_AUTHOR_UNGROUNDED"]
    exported = paper.export_to_json()
    exported_codes = {issue["code"] for issue in exported["extraction"]["validation"]["issues"]}
    assert "VAL_AUTHOR_UNGROUNDED" in exported_codes


async def test_active_untargeted_abstention_is_blocking_and_not_promotable(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution
    from bibr.pipeline.stages import post_parse as post_parse_module
    from bibr.validation import IssueSeverity, ValidationIssue

    contents = _paper_contents([(1, "First record", 0), (2, "Second record", 0)])
    candidates = (
        _candidate("c1", "First record", roles=frozenset({"title"}), text_ids=(1,)),
        _candidate("c2", "Second record", roles=frozenset({"title"}), text_ids=(2,)),
    )
    resolution = FrontMatterResolution(
        candidates=candidates,
        blocks=(
            FrontMatterBlock("b1", ("c1",), ("c1",)),
            FrontMatterBlock("b2", ("c2",), ("c2",)),
        ),
        selected_block_id=None,
        selection_method="abstained",
        reason_flags=("multiple_plausible_blocks",),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )
    issue = ValidationIssue(
        code="VAL_METADATA_MULTI_ITEM",
        severity=IssueSeverity.ERROR,
        message="No unique active metadata record",
        origin_stage="extract",
        evidence_ids=("b1", "b2"),
        blocking=True,
    )
    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False

    async def no_op(*_args, **_kwargs):
        return None

    def resolve(_contents, *, expected_identity=None, target_required=False, settings=None):
        assert expected_identity is None
        assert target_required is True
        return resolution, (issue,)

    monkeypatch.setattr(post_parse_module, "_classify_sections", no_op)
    monkeypatch.setattr(post_parse_module, "_normalize_section_structure", no_op)
    monkeypatch.setattr(post_parse_module, "_link_citations", no_op)
    monkeypatch.setattr("bibr.extract.front_matter.resolve_front_matter", resolve)
    monkeypatch.setattr(
        "bibr.extract.research_integrity.extract_structured_integrity",
        no_op,
    )

    class UnusedLlm:
        pass

    paper = await post_parse_module.post_parse(
        contents,
        "multi.pdf",
        "hash",
        llm_client=UnusedLlm(),
        ref_parse_strategy="off",
        settings=settings,
    )

    assert paper.metadata.title == ""
    assert [item.code for item in paper.validation_issues] == ["VAL_METADATA_MULTI_ITEM"]
    exported = paper.export_to_json()
    assert exported["extraction"]["validation"]["blocking"] == 1
    assert exported["extraction"]["validation"]["promotable"] is False


async def test_end_to_end_clean_forty_author_byline_has_no_grounding_issue(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from bibr.pipeline.stages import post_parse as post_parse_module
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    byline_text = "; ".join(f"Given{index} Family{index}" for index in range(1, 41))
    contents = _paper_contents([(1, "Selected title", 0), (2, byline_text, 0)])
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate("c2", byline_text, roles=frozenset({"byline"}), text_ids=(2,))
    resolution = _resolution(title, byline)
    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False

    class FakeLlm:
        async def extract_core_metadata(self, *_args, **_kwargs):
            return CoreMetadataLLM(
                title="Selected title",
                authors=[
                    AuthorLLM(given=f"Given{index}", family=f"Family{index}")
                    for index in range(1, 41)
                ],
            )

    async def no_op(*_args, **_kwargs):
        return None

    def attach(actual_contents, *_args, **_kwargs):
        actual_contents.front_matter_resolution = resolution
        return ()

    monkeypatch.setattr(post_parse_module, "_classify_sections", no_op)
    monkeypatch.setattr(post_parse_module, "_normalize_section_structure", no_op)
    monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach)
    monkeypatch.setattr(post_parse_module, "_link_citations", no_op)
    monkeypatch.setattr(
        "bibr.extract.research_integrity.extract_structured_integrity",
        no_op,
    )
    monkeypatch.setattr(
        CoreMetadataExtractor,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )

    paper = await post_parse_module.post_parse(
        contents,
        "forty.pdf",
        "hash",
        llm_client=FakeLlm(),
        ref_parse_strategy="off",
        settings=settings,
    )

    assert len(paper.metadata.authors) == 40
    assert not {
        "VAL_AUTHOR_MISSING",
        "VAL_AUTHOR_UNGROUNDED",
    }.intersection(issue.code for issue in paper.validation_issues)


# --- Task 2: extractor-owned grounded empty-author recovery -------------------


async def test_empty_authors_receive_one_instructor_recovery_and_grounded_names_survive(
    monkeypatch,
):
    from bibr.schemas import AuthorLLM

    extractor, llm = make_empty_author_extractor(
        monkeypatch,
        source="Alice Example, Bob Sample",
        recovered=[
            AuthorLLM(given="Alice", family="Example"),
            AuthorLLM(given="Bob", family="Sample"),
        ],
    )

    metadata = await extractor.extract()

    assert [author.family for author in metadata.authors] == ["Example", "Sample"]
    assert llm.extract_authors.await_count == 1
    assert llm.extract_authors.await_args.kwargs["json_mode"] is True


async def test_missed_byline_full_context_can_ground_recovered_authors(monkeypatch):
    # 10.20448-class shape: the printed middle-dot author row carries no byline
    # role, so recovery must ground against the full selected-block context.
    from bibr.schemas import AuthorLLM

    extractor, _ = make_empty_author_extractor(
        monkeypatch,
        source="Mahdieh Khorsandifard1 · Kian Jafari1 · Arash Sheikhaleh1",
        recovered=[AuthorLLM(given="Kian", family="Jafari")],
        mark_byline=False,
    )

    metadata = await extractor.extract()

    assert [author.family for author in metadata.authors] == ["Jafari"]


async def test_recovery_drops_ungrounded_names_and_records_receipt(monkeypatch):
    from bibr.schemas import AuthorLLM

    extractor, llm = make_empty_author_extractor(
        monkeypatch,
        source="Alice Example",
        recovered=[
            AuthorLLM(given="Alice", family="Example"),
            AuthorLLM(given="Mallory", family="Outside"),
        ],
    )

    metadata = await extractor.extract()

    assert [author.family for author in metadata.authors] == ["Example"]
    assert llm.extract_authors.await_count == 1
    assert "VAL_AUTHOR_RECOVERY_UNGROUNDED" in {issue.code for issue in extractor.validation_issues}


async def test_recovery_reserves_source_spans_one_to_one(monkeypatch):
    # Two identical recovered authors compete for one printed name: exactly one
    # may survive.
    from bibr.schemas import AuthorLLM

    extractor, _ = make_empty_author_extractor(
        monkeypatch,
        source="Alice Example, Bob Sample",
        recovered=[
            AuthorLLM(given="Alice", family="Example"),
            AuthorLLM(given="Alice", family="Example"),
        ],
    )

    metadata = await extractor.extract()

    assert [author.family for author in metadata.authors] == ["Example"]


async def test_recovery_still_empty_keeps_empty_authors_after_one_attempt(monkeypatch):
    extractor, llm = make_empty_author_extractor(
        monkeypatch,
        source="Alice Example, Bob Sample",
        recovered=[],
    )

    metadata = await extractor.extract()

    assert metadata.authors == []
    assert llm.extract_authors.await_count == 1


async def test_recovery_upstream_error_degrades_to_empty_authors(monkeypatch):
    from bibr.exceptions import UpstreamServiceError

    extractor, llm = make_empty_author_extractor(
        monkeypatch,
        source="Alice Example, Bob Sample",
        recovery_error=UpstreamServiceError("LLM", "recovery failed"),
    )

    metadata = await extractor.extract()

    assert metadata.authors == []
    assert llm.extract_authors.await_count == 1


async def test_recovery_invalid_output_error_degrades_to_empty_authors(monkeypatch):
    from bibr.exceptions import ProcessingError
    from bibr.models import ErrorCode

    extractor, llm = make_empty_author_extractor(
        monkeypatch,
        source="Alice Example, Bob Sample",
        recovery_error=ProcessingError(
            "LLM returned invalid structured output",
            error_code=ErrorCode.LLM_INVALID_OUTPUT.value,
        ),
    )

    metadata = await extractor.extract()

    assert metadata.authors == []
    assert llm.extract_authors.await_count == 1


async def test_recovery_unrelated_processing_error_propagates(monkeypatch):
    from bibr.exceptions import ProcessingError

    extractor, _ = make_empty_author_extractor(
        monkeypatch,
        source="Alice Example, Bob Sample",
        recovery_error=ProcessingError("unrelated failure", error_code="other_code"),
    )

    with pytest.raises(ProcessingError):
        await extractor.extract()


def test_grounded_authors_in_context_preserves_order_and_reservations():
    from bibr.extract.core_metadata import grounded_authors_in_context
    from bibr.schemas import AuthorLLM

    authors = [
        AuthorLLM(given="Bob", family="Sample"),
        AuthorLLM(given="Alice", family="Example"),
        AuthorLLM(given="Alice", family="Example"),
        AuthorLLM(given="Mallory", family="Outside"),
    ]

    grounded, rejected = grounded_authors_in_context(
        authors,
        "Alice Example, Bob Sample and colleagues",
    )

    assert [(author.given, author.family) for author in grounded] == [
        ("Bob", "Sample"),
        ("Alice", "Example"),
    ]
    assert [(author.given, author.family) for author in rejected] == [
        ("Alice", "Example"),
        ("Mallory", "Outside"),
    ]


class TestBylineRecoveryContext:
    """Empty-author recovery uses focused byline evidence instead of repeating the failed wide context."""

    def test_returns_selected_byline_verbatim(self):
        from bibr.extract.core_metadata import byline_recovery_context

        resolution = _resolution(
            _candidate("c1", "A Computational Approach", roles=frozenset({"title"})),
            _candidate("c2", "Fabian Vexler and Anika Q Mboro", roles=frozenset({"byline"})),
        )
        assert byline_recovery_context(resolution) == "Fabian Vexler and Anika Q Mboro"

    def test_preserves_printed_case(self):
        """_name_tokens casefolds; the model must see the printed form."""
        from bibr.extract.core_metadata import byline_recovery_context

        resolution = _resolution(
            _candidate("c1", "T", roles=frozenset({"title"})),
            _candidate("c2", "Lisa A. M. van der Quix", roles=frozenset({"byline"})),
        )
        assert byline_recovery_context(resolution) == "Lisa A. M. van der Quix"

    def test_joins_multiple_byline_rows_deduped(self):
        from bibr.extract.core_metadata import byline_recovery_context

        resolution = _resolution(
            _candidate("c1", "T", roles=frozenset({"title"})),
            _candidate("c2", "Fabian Vexler", roles=frozenset({"byline"})),
            _candidate("c3", "Fabian Vexler", roles=frozenset({"byline"})),
            _candidate("c4", "Anika Q Mboro", roles=frozenset({"byline"})),
        )
        assert byline_recovery_context(resolution) == "Fabian Vexler\nAnika Q Mboro"

    def test_none_without_byline(self):
        """No name-like entry parses → caller keeps its existing context."""
        from bibr.extract.core_metadata import byline_recovery_context

        resolution = _resolution(
            _candidate("c1", "A Computational Approach", roles=frozenset({"title"})),
        )
        assert byline_recovery_context(resolution) is None

    def test_none_when_no_resolution(self):
        from bibr.extract.core_metadata import byline_recovery_context

        assert byline_recovery_context(None) is None

    def test_none_when_block_not_selected(self):
        from bibr.extract.core_metadata import byline_recovery_context
        from bibr.extract.front_matter import FrontMatterResolution

        resolution = _resolution(
            _candidate("c1", "T", roles=frozenset({"title"})),
            _candidate("c2", "Fabian Vexler", roles=frozenset({"byline"})),
        )
        abstained = FrontMatterResolution(
            candidates=resolution.candidates,
            blocks=resolution.blocks,
            selected_block_id=None,
            selection_method="abstained",
            reason_flags=(),
            allowed_text_ids=frozenset(),
            allowed_section_ids=frozenset(),
        )
        assert byline_recovery_context(abstained) is None


class TestAffiliationReconciliationUnderOwnership:
    """Ownership scopes the model input while deterministic affiliation reconcilers retain the full frame.

    Numbered footnotes and repeated author-name blocks can lie outside the selected front-matter record. Reconciliation remains anchored to an already-extracted author name."""

    @staticmethod
    def _extractor(full_df, block_rows, llm_authors, resolution):
        from bibr.extract.core_metadata import CoreMetadataExtractor
        from bibr.schemas import CoreMetadataLLM

        contents = mock.Mock(spec=PaperContents)
        contents.sentences_df = full_df
        contents.detected_headers = []
        contents.detected_footers = []
        contents.layout_hints = []
        contents.sections = []
        contents.sentences = []
        contents.processing_warnings = []

        locator = mock.MagicMock()
        locator.get_cutoff_index.return_value = len(full_df)
        # Ownership scope: the metadata frame is the selected block's rows only.
        locator.collect_core_metadata_rows.return_value = full_df.iloc[block_rows].copy()

        llm_client = mock.MagicMock()
        llm_client.extract_core_metadata = mock.AsyncMock(
            return_value=CoreMetadataLLM(title="A Study", authors=llm_authors, keywords=[])
        )
        return CoreMetadataExtractor(
            contents,
            llm_client=llm_client,
            locator=locator,
            email_harvester=mock.MagicMock(),
            front_matter_resolution=resolution,
        )

    async def test_numbered_page1_footnote_survives_ownership_scope(self):
        from bibr.schemas import AuthorLLM

        full_df = pd.DataFrame(
            {
                "text_id": [1, 2, 3, 4],
                "section_name": ["Title", "Title", "Footnote", "Footnote"],
                "page_number": [1, 1, 1, 1],
                "text": [
                    "A Study",
                    "Saskia M Kelders1,2 and Hanneke Kip1,3",
                    "1 Department of Health, Psychology and Technology, University of Twente, "
                    "Enschede, The Netherlands 2 Optentia Research Unit, North-West University, "
                    "Vanderbijlpark, South Africa",
                    "3 Department of Research, Transfore, Deventer, The Netherlands",
                ],
            }
        )
        resolution = _resolution(
            _candidate("c1", "A Study", roles=frozenset({"title"}), text_ids=(1,)),
            _candidate(
                "c2",
                "Saskia M Kelders1,2 and Hanneke Kip1,3",
                roles=frozenset({"byline"}),
                text_ids=(2,),
            ),
        )
        # The footnote rows are outside the block: ownership hides them.
        assert 3 not in resolution.allowed_text_ids
        assert 4 not in resolution.allowed_text_ids

        extractor = self._extractor(
            full_df,
            [0, 1],
            [
                AuthorLLM(given="Saskia M", family="Kelders", affiliation=None),
                AuthorLLM(given="Hanneke", family="Kip", affiliation=None),
            ],
            resolution,
        )
        metadata = await extractor.extract()

        assert metadata.authors[0].affiliation.startswith("Department of Health")
        assert "; Optentia Research Unit" in metadata.authors[0].affiliation
        assert "; Department of Research" in metadata.authors[1].affiliation

    async def test_backmatter_author_information_survives_ownership_scope(self):
        from bibr.schemas import AuthorLLM

        full_df = pd.DataFrame(
            {
                "text_id": [1, 2, 3],
                "section_name": ["Title", "Title", "Author information"],
                "page_number": [1, 1, 11],
                "text": [
                    "A Study",
                    "Saskia M Kelders and Hanneke Kip",
                    "Saskia M Kelders, Department of Health, University of Twente, Enschede, "
                    "The Netherlands; Hanneke Kip, Department of Research, Transfore, "
                    "Deventer, The Netherlands",
                ],
            }
        )
        resolution = _resolution(
            _candidate("c1", "A Study", roles=frozenset({"title"}), text_ids=(1,)),
            _candidate(
                "c2",
                "Saskia M Kelders and Hanneke Kip",
                roles=frozenset({"byline"}),
                text_ids=(2,),
            ),
        )
        assert 3 not in resolution.allowed_text_ids

        extractor = self._extractor(
            full_df,
            [0, 1],
            [
                AuthorLLM(given="Saskia M", family="Kelders", affiliation=None),
                AuthorLLM(given="Hanneke", family="Kip", affiliation=None),
            ],
            resolution,
        )
        metadata = await extractor.extract()

        assert metadata.authors[0].affiliation.startswith("Department of Health")
        assert metadata.authors[1].affiliation.startswith("Department of Research")

    def test_ambiguous_marker_abstains_instead_of_taking_the_last_definition(self):
        """A page carrying several articles restarts affiliation numbering at 1.

        Restoring the full sentence frame is what makes the page-1 footnote
        readable again, but it also puts a *neighbouring* article's numbered
        block in scope. Assigning the last definition would ship someone else's
        institution under this author's name, which is worse than shipping
        nothing — so an ambiguous marker must resolve to nothing at all.
        """
        from bibr.extract.core_metadata import CoreMetadataExtractor
        from bibr.models import PaperAuthor

        author = PaperAuthor(author_id=1, given="Saskia M", family="Kelders", affiliation="")
        frame = pd.DataFrame(
            {
                "page_number": [1, 1, 1],
                "text": [
                    "Saskia M Kelders1",
                    "1 University of Twente, Enschede, The Netherlands",
                    "1 Karolinska Institutet, Stockholm, Sweden",
                ],
            }
        )

        CoreMetadataExtractor._reconcile_numbered_affiliations([author], frame)

        assert author.affiliation in (None, "")

    def test_marker_repeated_with_identical_text_still_resolves(self):
        """Duplicate rows are an OCR artefact, not an ambiguity.

        The abstention above must key on the affiliation *meaning*, otherwise a
        header echoed once per column would silently disable the reconciler on
        ordinary single-article papers.
        """
        from bibr.extract.core_metadata import CoreMetadataExtractor
        from bibr.models import PaperAuthor

        author = PaperAuthor(author_id=1, given="Saskia M", family="Kelders", affiliation="")
        frame = pd.DataFrame(
            {
                "page_number": [1, 1, 1],
                "text": [
                    "Saskia M Kelders1",
                    "1 University of Twente, Enschede, The Netherlands",
                    "1 University of Twente, Enschede, The Netherlands",
                ],
            }
        )

        CoreMetadataExtractor._reconcile_numbered_affiliations([author], frame)

        assert author.affiliation == "University of Twente, Enschede, The Netherlands"


def test_author_context_keeps_surrounding_rows_when_title_is_only_byline_candidate():
    from bibr.extract.core_metadata import render_author_context, render_block_context

    citation_and_byline = _candidate(
        "c1",
        "Cite this article: 10.1234/example. Alice AUTHOR, Bob WRITER",
        roles=frozenset({"doi"}),
        text_ids=(1,),
    )
    title = _candidate(
        "c2",
        "EVALUATION OF ORGANIC FERTILIZATION AS AN ALTERNATIVE",
        roles=frozenset({"title", "byline"}),
    )
    resolution = _resolution(citation_and_byline, title)
    full_text = render_block_context(resolution)
    assert render_author_context(resolution, full_text=full_text) == full_text


def test_author_table_is_not_borrowed_across_records_or_from_unlabelled_data():
    from dataclasses import replace
    from types import SimpleNamespace

    from bibr.extract.core_metadata import author_table_context
    from bibr.paper_contents import PaperTable

    table = PaperTable(
        table_id=1,
        section_id=9,
        tbl_html="",
        page_number=1,
        df=pd.DataFrame([["Alice Example"]], columns=["Authors"]),
    )
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    other = _candidate("c2", "Other title", roles=frozenset({"title"}), text_ids=(2,))
    resolution = _resolution(title, other, selected_ids=("c1",))
    second = replace(resolution.blocks[0], block_id="other", candidate_ids=("c2",))
    resolution = replace(resolution, blocks=(*resolution.blocks, second))
    contents = SimpleNamespace(tables=[table])
    assert author_table_context(contents, resolution) == ""
    assert author_table_context(contents, None) == ""
    table.df.columns = ["Participants"]
    assert author_table_context(contents, _resolution(title)) == ""
    # An uncaptained literature-summary table may also appear on the front page.
    # A generic Authors column does not identify this paper's byline.
    table.df = pd.DataFrame(
        [["Alice Example", "2020", "Results from another paper"]],
        columns=["Authors", "Year", "Findings"],
    )
    assert author_table_context(contents, _resolution(title)) == ""


@pytest.mark.parametrize("label", ["Author information", "INFO PENULIS"])
def test_author_information_table_is_available_as_front_matter(label):
    from types import SimpleNamespace

    from bibr.extract.core_metadata import author_table_context
    from bibr.paper_contents import PaperTable

    table = PaperTable(
        table_id=1,
        section_id=9,
        tbl_html="",
        page_number=1,
        df=pd.DataFrame([["Alice Example"], ["Example University"]], columns=[label]),
    )
    title = _candidate("c1", "Selected title", roles=frozenset({"title"}), text_ids=(1,))
    resolution = _resolution(title)
    contents = SimpleNamespace(tables=[table])
    assert "Alice Example" in author_table_context(contents, resolution)
    assert "Example University" in author_table_context(contents, resolution)
    table.page_number = 2
    assert author_table_context(contents, resolution) == ""
    table.page_number = 1
    table.caption = "Table 1. Authors in the reviewed literature"
    assert author_table_context(contents, resolution) == ""
