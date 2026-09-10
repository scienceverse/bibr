from __future__ import annotations

import pytest

from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.pipeline.identity import ExpectedIdentity


def _contents(
    rows: list[tuple[str, CanonicalSection, str, int]],
    *,
    headers: list[str] | None = None,
    footers: list[str] | None = None,
) -> PaperContents:
    sections = [PaperSection(0, "Root", 0, None, CanonicalSection.TITLE)]
    sentences = []
    for index, (header, section_type, text, page) in enumerate(rows, start=1):
        sections.append(PaperSection(index, header, 1, 0, section_type))
        sentences.append(
            PaperSentence(
                text_id=index,
                text=text,
                section_id=index,
                paragraph_id=index,
                page_number=page,
                region_meta={"region_type": "text", "region_index": index + 10},
            )
        )
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
        detected_headers=list(headers or []),
        detected_footers=list(footers or []),
    )


def test_candidate_collection_preserves_sentence_provenance_and_context():
    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _contents(
        [
            (
                "Article",
                CanonicalSection.TITLE,
                "Article DOI: https://doi.org/10.1234/SELF.1",
                1,
            ),
            (
                "References",
                CanonicalSection.REFERENCES,
                "[1] Prior work. https://doi.org/10.9999/reference.1",
                7,
            ),
        ]
    )

    candidates = collect_doi_candidates(contents)

    own = next(c for c in candidates if c.normalized == "10.1234/self.1")
    ref = next(c for c in candidates if c.normalized == "10.9999/reference.1")
    assert (own.page, own.section_id, own.text_id) == (1, 1, 1)
    assert (own.region_index, own.region_type) == (11, "text")
    assert own.marker_kind == "article_doi"
    assert own.semantic_context == "article_self"
    assert own.selection_tier == 3
    assert ref.semantic_context == "reference"
    assert ref.rejection_reason == "reference_candidate"


def test_expected_identity_selects_only_visible_eligible_candidate():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.1111/front", 1),
            ("Body", CanonicalSection.INTRODUCTION, "See 10.2222/other", 3),
        ]
    )
    expected = ExpectedIdentity(
        queue_record_id="record-1", expected_doi="10.2222/other", doi_required=True
    )

    selection = select_doi_candidates(collect_doi_candidates(contents), expected)

    assert selection.selected is not None
    assert selection.selected.normalized == "10.2222/other"
    assert selection.selected.selection_tier == 4
    assert selection.issues == ()


def test_expected_identity_is_never_fabricated_when_absent_from_source():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, "A paper with no DOI printed", 1)])
    expected = ExpectedIdentity(
        queue_record_id="record-1", expected_doi="10.1234/not-visible", doi_required=True
    )

    selection = select_doi_candidates(collect_doi_candidates(contents), expected)

    assert selection.selected is None
    assert all(c.normalized != expected.expected_doi for c in selection.candidates)
    assert [issue.code for issue in selection.issues] == ["VAL_EXPECTED_ID_MISSING"]
    assert selection.issues[0].blocking is True


def test_equal_tier_conflict_abstains_with_typed_ambiguity_issue():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.1234/one", 1),
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.1234/two", 1),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]
    assert selection.issues[0].blocking is False


def test_reference_repository_and_component_candidates_cannot_shadow_article_doi():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "References",
                CanonicalSection.REFERENCES,
                "[2] Prior article. https://doi.org/10.9999/reference",
                1,
            ),
            (
                "Data availability",
                CanonicalSection.OPEN_DATA,
                "Data are on Zenodo: https://doi.org/10.5281/zenodo.123",
                2,
            ),
            (
                "Figure 1",
                CanonicalSection.FIGURE,
                "doi:10.1371/journal.pone.0130688.g001",
                3,
            ),
            (
                "Article",
                CanonicalSection.TITLE,
                "Please cite as: https://doi.org/10.1016/j.nmni.2024.101483",
                1,
            ),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1016/j.nmni.2024.101483"
    rejected = {c.normalized: c.rejection_reason for c in selection.candidates}
    assert rejected["10.9999/reference"] == "reference_candidate"
    assert rejected["10.5281/zenodo.123"] == "data_or_code_candidate"
    assert rejected["10.1371/journal.pone.0130688.g001"] == "component_candidate"


def test_explicit_article_self_label_overrides_incidental_repository_words():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Title",
                CanonicalSection.TITLE,
                "Article DOI: 10.1234/paper.data reports the data analysis",
                1,
            )
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.semantic_context == "article_self"


def test_reference_doi_label_is_rejected_outside_references_section():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Front matter",
                CanonicalSection.TITLE,
                "Reference DOI: https://doi.org/10.1234/cited-work",
                1,
            )
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.marker_kind == "reference_doi"
    assert candidate.semantic_context == "reference"
    assert candidate.rejection_reason == "reference_candidate"


@pytest.mark.parametrize(
    ("label", "marker_kind"),
    [("Parent DOI", "parent_doi"), ("Component DOI", "component_doi")],
)
def test_parent_and_component_doi_labels_are_rejected_before_generic_doi(label, marker_kind):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Front matter", CanonicalSection.TITLE, f"{label}: 10.1234/parent", 1)])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.marker_kind == marker_kind
    assert candidate.semantic_context == "parent_or_component"
    assert candidate.rejection_reason == "component_candidate"


def test_incidental_data_word_does_not_reject_front_matter_article_doi():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Title",
                CanonicalSection.TITLE,
                "Data-driven science https://doi.org/10.1234/article",
                1,
            )
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/article"
    assert selection.selected.rejection_reason is None


def test_clear_data_availability_phrase_still_rejects_repository_doi():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Methods",
                CanonicalSection.METHODS,
                "Research data are available at https://doi.org/10.1234/dataset-record",
                5,
            )
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.rejection_reason == "data_or_code_candidate"


@pytest.mark.parametrize("label", ["Data DOI", "Dataset DOI", "Code DOI", "Repository DOI"])
def test_explicit_data_record_doi_label_is_rejected_before_generic_doi(label):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [("Front matter", CanonicalSection.TITLE, f"{label}: 10.1234/repository-record", 1)]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.marker_kind == "data_doi"
    assert candidate.semantic_context == "data_or_code"
    assert candidate.rejection_reason == "data_or_code_candidate"


@pytest.mark.parametrize(
    ("raw_doi", "normalized"),
    [
        ("10.1234/wrapped- 2024", "10.1234/wrapped-2024"),
        ("10.1371/journal. pone.0279511", "10.1371/journal.pone.0279511"),
        ("10.1136/ vetrec-2018-105253", "10.1136/vetrec-2018-105253"),
    ],
)
def test_candidate_receipt_preserves_exact_raw_spelling_while_normalizing(raw_doi, normalized):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, f"Article DOI: {raw_doi}", 1)])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.raw == raw_doi
    assert selection.selected.normalized == normalized


def test_truncated_journal_prefix_loses_to_full_article_doi():
    """Ladder rule 1 — receipt shape of ``10.30574/wjarr.2022.14.3.0574``."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Title",
                CanonicalSection.TITLE,
                "Article DOI: https://doi.org/10.30574/wjarr.2022.14.3.0574",
                1,
            )
        ],
        headers=["DOI: 10.30574/wjarr"],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.30574/wjarr.2022.14.3.0574"
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]
    assert selection.issues[0].blocking is False


def test_supplement_path_extension_never_displaces_the_article_doi():
    """Rule 1 must not treat an ``/s1`` supplement as a longer spelling."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.3390/ijerph19148408", 1),
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.3390/ijerph19148408/s1", 1),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]


def test_funder_registry_doi_is_rejected_before_selection():
    """Receipt shape of ``10.1371/journal.pone.0279511`` — a Funder Registry id."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.1371/journal.pone.0279511", 1),
            (
                "Introduction",
                CanonicalSection.INTRODUCTION,
                "Funded by MCIN/AEI (10.13039/501100011033).",
                2,
            ),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1371/journal.pone.0279511"
    rejected = {c.normalized: c.rejection_reason for c in selection.candidates}
    assert rejected["10.13039/501100011033"] == "funder_registry_candidate"


def test_bare_co_tier_doi_loses_to_marked_front_matter_doi():
    """Ladder rule 2 — an unlabelled front-matter DOI never outranks a marked one."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.1371/journal.pone.0279511", 1),
            (
                "Introduction",
                CanonicalSection.INTRODUCTION,
                "Grant agreement (10.5555/501100011033).",
                2,
            ),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1371/journal.pone.0279511"
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]


def test_footer_ocr_twin_does_not_suppress_page_one_sentence_doi():
    """Ladder rule 3 — receipt shape of ``10.3390/ijerph19148408``."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [("Title", CanonicalSection.TITLE, "https://doi.org/10.3390/ijerph19148408", 1)],
        footers=["https://doi.org/10.3390/jeirph19148408"],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.3390/ijerph19148408"
    assert selection.selected.source_kind == "sentence"


def test_reference_dois_in_headers_do_not_suppress_front_matter_doi():
    """Ladder rule 3 — receipt shape of ``10.1007/s12671-023-02077-9``."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [("Title", CanonicalSection.TITLE, "https://doi.org/10.1007/s12671-023-02077-9", 1)],
        headers=[
            "https://doi.org/10.1212/WNL.00000000000008534",
            "https://doi.org/10.1126/sciadv.1700489",
        ],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1007/s12671-023-02077-9"


def test_footnote_reference_dois_lose_to_lowest_page_article_doi():
    """Ladder rule 4 — receipt shape of ``10.3389/fpsyg.2022.890524``."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Front matter", CanonicalSection.UNKNOWN, "DOI 10.3389/fpsyg.2022.890524", 1),
            (
                "Notes",
                CanonicalSection.FOOTNOTE,
                "Zhang, Y. (2015). Cleaner production. doi: 10.1016/j.jclepro.2014.11.086",
                16,
            ),
            (
                "Notes",
                CanonicalSection.FOOTNOTE,
                "Cohen, J. (2006). Accounting policy. doi: 10.1016/j.jaccpubpol.2006.03.004",
                16,
            ),
        ],
        headers=["DOI 10.3389/fpsyg.2022.890524"],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.3389/fpsyg.2022.890524"
    assert selection.selected.page == 1


def test_www_doi_org_label_separates_journal_and_article_dois():
    """Receipt shape of ``10.46654/rjmp.14033`` — the printed marker uses ``www.``."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [("Title", CanonicalSection.TITLE, "Article DOI: www.doi.org/10.46654/RJMP.14033", 1)],
        headers=["Journal DOI: www.doi.org/10.46654/RJMP"],
    )

    candidates = collect_doi_candidates(contents)

    by_doi = {c.normalized: c for c in candidates}
    assert by_doi["10.46654/rjmp"].marker_kind == "journal_doi"
    assert by_doi["10.46654/rjmp.14033"].marker_kind == "article_doi"

    selection = select_doi_candidates(candidates)

    assert selection.selected is not None
    assert selection.selected.normalized == "10.46654/rjmp.14033"
    assert selection.issues == ()


def test_marker_gated_registrant_wrap_is_bridged():
    """Live text of ``10.1016/j.lanwpc.2023.100933`` wraps between ``10.`` and ``1016``."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Front matter",
                CanonicalSection.TITLE,
                "Published Online xxxhttps://doi.org/10. 1016/j.lanwpc.2023. 100933",
                1,
            )
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1016/j.lanwpc.2023.100933"


@pytest.mark.parametrize(
    "text",
    [
        "Row 10. 1016/j.made-up.2020.1 of the table",
        "See section 10. 1016/2020 for details",
    ],
)
def test_unmarked_ten_dot_break_is_never_bridged(text):
    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _contents([("Body", CanonicalSection.RESULTS, text, 4)])

    assert collect_doi_candidates(contents) == ()
