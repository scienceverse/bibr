from __future__ import annotations

import re

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
                region_meta={
                    "region_type": "text",
                    "region_page": page,
                    "region_index": index + 10,
                },
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


def _ocr_region(label: str, content: str, top: int, *, native_label: str | None = None) -> dict:
    return {
        "native_label": native_label or label,
        "label": label,
        "content": content,
        "bbox_2d": [100, top, 900, top + 40],
    }


def _parse_ocr_pages(pages: list[list[dict]]) -> PaperContents:
    """OCR-stage post-processing, then the real parser and segmentation handoff."""
    from bibr.pipeline.stages.ocr import _postprocess_ocr_regions, _to_typed_regions
    from bibr.structure.pdf_parser import PDFParser

    for page in pages:
        for slot, region in enumerate(page):
            region["index"] = slot
    parser = PDFParser(_to_typed_regions(_postprocess_ocr_regions(pages)))
    contents = parser.parse()
    parser.apply_segmentation(
        contents,
        [re.split(r"(?<=[.!?])\s+", text) for text in parser.assembler.segmentable_texts],
    )
    parser.create_content_sections(contents)
    return contents


def _two_page_ocr_pages() -> list[list[dict]]:
    return [
        [
            _ocr_region("doc_title", "Tracing identifiers to layout regions", 60),
            _ocr_region("formula", "E = mc^2", 120, native_label="display_formula"),
            _ocr_region("text", "(1)", 120, native_label="formula_number"),
            # Layout slot 3; post-processing merges the equation number into
            # the formula and renumbers this region to index 2.
            _ocr_region("text", "https://doi.org/10.1234/self.1", 200),
            _ocr_region("text", "Received 1 January 2024; accepted 2 February 2024.", 260),
            _ocr_region("text", "The companion paper describes the data and", 900),
        ],
        [
            _ocr_region(
                "text",
                "the analysis code. It is cited as https://doi.org/10.5555/companion.2 here.",
                80,
            ),
        ],
    ]


def test_parsed_sentence_candidate_names_its_post_processing_region():
    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _parse_ocr_pages(_two_page_ocr_pages())

    own = next(c for c in collect_doi_candidates(contents) if c.normalized == "10.1234/self.1")
    assert (own.source_kind, own.page, own.region_index) == ("sentence", 1, 2)
    # ``(page, region_index)`` is the RegionSummary key exported as the
    # ``page``/``index`` of an ``extraction.regions`` row.
    [region] = [
        summary
        for summary in contents.region_summaries
        if (summary.page, summary.index) == (own.page, own.region_index)
    ]
    assert region.content == "https://doi.org/10.1234/self.1"
    assert region.label == own.region_type == "text"


def test_sentence_continued_onto_a_later_page_has_no_region_index():
    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _parse_ocr_pages(_two_page_ocr_pages())

    continued = next(
        c for c in collect_doi_candidates(contents) if c.normalized == "10.5555/companion.2"
    )
    # The paragraph began in region 4 of page 1; page 2 has no such region.
    assert continued.page == 2
    assert continued.region_index is None


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
    """Ladder, drop truncated prefixes — receipt shape of ``10.30574/wjarr.2022.14.3.0574``."""

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
    """An MDPI ``/s1`` supplement is a component, never a longer spelling."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.3390/ijerph19148408", 1),
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.3390/ijerph19148408/s1", 1),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.3390/ijerph19148408"
    assert selection.issues == ()
    rejected = {c.normalized: c.rejection_reason for c in selection.candidates}
    assert rejected["10.3390/ijerph19148408/s1"] == "component_candidate"


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
    """Ladder, prefer marked — an unlabelled front-matter DOI never outranks a marked one."""

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
    """Ladder, prefer body sources — receipt shape of ``10.3390/ijerph19148408``."""

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
    """Ladder, prefer body sources — receipt shape of ``10.1007/s12671-023-02077-9``."""

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
    """Ladder, prefer the lowest page — receipt shape of ``10.3389/fpsyg.2022.890524``."""

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


def test_year_led_citation_header_names_the_paper():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [("Title", CanonicalSection.TITLE, "A paper title", 1)],
        headers=["2017. Proc Example Soc 2, 20:1-15. https://doi.org/10.1234/pes.v2i0.4064."],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/pes.v2i0.4064"
    assert selection.selected.source_kind == "header"


@pytest.mark.parametrize(
    "line",
    ["12. Smith J. Earlier work. https://doi.org/10.1234/ref", "[3] https://doi.org/10.1234/ref"],
)
def test_numbered_reference_line_in_furniture_is_still_rejected(line):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, "A paper title", 1)], footers=[line])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert [c.rejection_reason for c in selection.candidates] == ["reference_candidate"]


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


_APA_MASTHEAD = (
    "Journal of Experimental Psychology: General 2023, Vol. 152, No. 1, 1-20 "
    "© 2022 American Psychological Association ISSN: 0096-3445 "
    "https://doi.org/10.1037/xge0001234"
)
_APA_SUPPLEMENT_NOTE = "Supplemental materials: https://doi.org/10.1037/xge0001234.supp"


@pytest.mark.parametrize("masthead_in_header", [True, False])
def test_apa_supplemental_materials_doi_never_names_the_paper(masthead_in_header):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    rows = [("Footnote 1", CanonicalSection.FOOTNOTE, _APA_SUPPLEMENT_NOTE, 1)]
    if not masthead_in_header:
        rows.insert(0, ("Title", CanonicalSection.TITLE, _APA_MASTHEAD, 1))
    contents = _contents(rows, headers=[_APA_MASTHEAD] if masthead_in_header else None)

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1037/xge0001234"
    assert selection.issues == ()
    [supplement] = [c for c in selection.candidates if c.normalized.endswith(".supp")]
    assert supplement.marker_kind == "component_doi"
    assert supplement.rejection_reason == "component_candidate"


@pytest.mark.parametrize(
    "supplement_doi",
    [
        "10.1037/xge0001234.supp",
        "10.5194/acp-16-8389-2016-supplement",
        "10.1371/journal.pone.0130688.s003",
        "10.7717/peerj.5478/supp-1",
        "10.7717/peerj.5478/fig-1",
        "10.7717/peerj.5478/table-2",
    ],
)
def test_supplement_suffix_doi_is_a_component_candidate(supplement_doi):
    """Receipt shape of ``10.5194/acp-16-8389-2016``: the page-13 supplement won."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Appendix",
                CanonicalSection.APPENDIX,
                f"The supplement related to this article is available online at doi:{supplement_doi}.",
                13,
            )
        ],
        headers=["Atmos. Chem. Phys., 16, 8389-8403, 2016 doi:10.5194/acp-16-8389-2016"],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.5194/acp-16-8389-2016"
    assert selection.issues == ()
    [supplement] = [c for c in selection.candidates if c.normalized == supplement_doi]
    assert supplement.rejection_reason == "component_candidate"


@pytest.mark.parametrize(
    ("text", "doi"),
    [
        ("BMJ 2013;346:f1049 doi: 10.1136/bmj.f1049", "10.1136/bmj.f1049"),
        ("Cite this as: BMJ 2014;348:g2276 doi: 10.1136/bmj.g2276", "10.1136/bmj.g2276"),
        (
            "Annals of Oncology 30 (Supplement 5): v1-v10, 2019. doi:10.1093/annonc/mdz239",
            "10.1093/annonc/mdz239",
        ),
        (
            "Annals of Oncology, Volume 30, Supplement 5, October 2019, doi:10.1093/annonc/mdz394",
            "10.1093/annonc/mdz394",
        ),
    ],
)
def test_article_doi_resembling_a_component_is_selected(text, doi):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, text, 1)])
    candidates = collect_doi_candidates(contents)

    selection = select_doi_candidates(candidates)
    expected = select_doi_candidates(
        candidates, ExpectedIdentity(queue_record_id="record-1", expected_doi=doi)
    )

    assert selection.selected is not None
    assert selection.selected.normalized == doi
    assert selection.selected.selection_tier == 3
    assert selection.issues == ()
    assert expected.selected is not None
    assert expected.selected.normalized == doi
    assert expected.issues == ()


def test_plos_figure_doi_on_page_one_is_still_a_component():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [("Title", CanonicalSection.TITLE, "https://doi.org/10.1371/journal.pone.0130688.g001", 1)]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.rejection_reason == "component_candidate"


@pytest.mark.parametrize(
    ("text", "doi"),
    [
        ("DOI: 10.31234/osf.io/abc12", "10.31234/osf.io/abc12"),
        ("https://doi.org/10.31235/osf.io/xyz98", "10.31235/osf.io/xyz98"),
        ("DOI: 10.31219/osf.io/qwe45", "10.31219/osf.io/qwe45"),
        ("DOI: 10.31234/osf.io/abc12_v1", "10.31234/osf.io/abc12_v1"),
    ],
)
def test_osf_hosted_preprint_doi_is_the_paper_doi(text, doi):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, text, 1)])
    candidates = collect_doi_candidates(contents)

    selection = select_doi_candidates(candidates)
    expected = select_doi_candidates(
        candidates,
        ExpectedIdentity(queue_record_id="record-1", expected_doi=doi, doi_required=True),
    )

    assert selection.selected is not None
    assert selection.selected.normalized == doi
    assert selection.selected.rejection_reason is None
    assert expected.selected is not None
    assert expected.selected.normalized == doi
    assert expected.issues == ()


@pytest.mark.parametrize(
    "doi",
    [
        "10.17605/OSF.IO/ABCDE",
        "10.6084/m9.figshare.1234567",
        "10.5061/dryad.abc123",
        "10.5281/zenodo.123",
    ],
)
def test_data_repository_registrant_doi_is_still_rejected(doi):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, f"DOI: {doi}", 1)])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.rejection_reason == "data_or_code_candidate"


def test_repository_doi_beside_a_cited_doi_still_marks_a_software_citation():
    """eLife 79461: a Zenodo DOI next to Cutadapt's article DOI in Methods."""

    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _contents(
        [
            (
                "Methods",
                CanonicalSection.METHODS,
                "Reads were trimmed using TrimGalore (ver. 0.6.0, Cutadapt ver. 1.18; "
                "DOI:10.5281/zenodo.5127899, DOI:10.14806/ej.17.1.200).",
                6,
            )
        ]
    )

    rejected = {c.normalized: c.rejection_reason for c in collect_doi_candidates(contents)}

    assert rejected == {
        "10.5281/zenodo.5127899": "data_or_code_candidate",
        "10.14806/ej.17.1.200": "data_or_code_candidate",
    }


def test_repeated_running_header_doi_is_not_displaced_by_early_footnote_citation():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "A study of things", 1),
            (
                "Notes",
                CanonicalSection.FOOTNOTE,
                "1 See Smith (2019), https://doi.org/10.2222/cited.1 for details.",
                2,
            ),
        ],
        headers=["https://doi.org/10.1111/own.123", "https://doi.org/10.1111/own.123"],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]
    assert selection.issues[0].message == "Conflicting source-visible DOI candidates at tier 2"


@pytest.mark.parametrize(
    "rows",
    [
        [
            (
                "Methods",
                CanonicalSection.METHODS,
                "We used the stimuli of Smith et al. (2019; https://doi.org/10.1037/xge0000123).",
                5,
            )
        ],
        [
            (
                "Literatur",
                CanonicalSection.UNKNOWN,
                "Smith, J. (2019). A title. Journal, 1, 1-2. https://doi.org/10.1037/xge0000123",
                20,
            ),
            (
                "Literatur",
                CanonicalSection.UNKNOWN,
                "Doe, J. (2020). Other. Journal, 2, 3-4. "
                "https://doi.org/10.1016/j.cognition.2020.104",
                21,
            ),
        ],
        [
            (
                "A Theory of Attention",
                CanonicalSection.UNKNOWN,
                "We build on Smith et al. (2019; https://doi.org/10.1037/xge0000123).",
                3,
            ),
            ("Method", CanonicalSection.METHODS, "Participants were 120 adults.", 4),
        ],
    ],
    ids=["one-cited-doi", "unclassified-reference-list", "unclassified-intro-on-page-3"],
)
def test_untyped_body_doi_never_names_a_manuscript_without_its_own_doi(rows):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    candidates = collect_doi_candidates(_contents(rows))

    selection = select_doi_candidates(candidates)
    required = select_doi_candidates(
        candidates, ExpectedIdentity(queue_record_id="record-1", doi_required=True)
    )
    mismatched = select_doi_candidates(
        candidates,
        ExpectedIdentity(
            queue_record_id="record-1", expected_doi="10.9999/own.1", doi_required=True
        ),
    )

    assert {c.selection_tier for c in candidates} == {1}
    assert selection.selected is None
    assert selection.issues == ()
    assert required.selected is None
    assert [(i.code, i.blocking) for i in required.issues] == [("VAL_EXPECTED_ID_MISSING", True)]
    assert mismatched.selected is None
    assert [(i.code, i.blocking) for i in mismatched.issues] == [("VAL_EXPECTED_ID_MISSING", True)]


def test_lone_journal_doi_does_not_name_the_paper():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([], headers=["Journal DOI: www.doi.org/10.46654/RJMP"])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.semantic_context == "journal_identity"


def test_structured_article_doi_wins_over_component_dois_in_body_text():
    """eLife JATS: figure DOIs extend the article DOI with a number."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates
    from bibr.models import PaperMetadata

    contents = _contents(
        [
            ("Figure 1", CanonicalSection.UNKNOWN, "DOI: 10.7554/eLife.00013.005", None),
            ("Figure 2", CanonicalSection.UNKNOWN, "DOI: 10.7554/eLife.00013.006", None),
        ]
    )
    contents.preparsed_metadata = PaperMetadata(doi="10.7554/eLife.00013", title="")

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.7554/elife.00013"
    assert selection.selected.source_kind == "structured_metadata"
    # A DOI label in the body, outside the front matter, marks a cited or
    # component DOI: it no longer ties with the article's own.
    assert {c.selection_tier for c in selection.candidates if c.source_kind == "sentence"} == {1}
    assert selection.issues == ()


_SICI_DOI = "10.1002/(SICI)1097-4679(199901)55:1<1::AID-JCLP1>3.0.CO;2-K"


def test_sici_doi_is_selected_whole():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, f"doi: {_SICI_DOI}", 1)])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.raw == _SICI_DOI
    assert selection.selected.normalized == _SICI_DOI.casefold()


@pytest.mark.parametrize(
    ("text", "doi"),
    [
        ("https://doi.org/10.1234/abc.5<sup>1</sup>", "10.1234/abc.5"),
        ("<https://doi.org/10.1234/abc.5>", "10.1234/abc.5"),
        ("doi: 10.1016/S0140-6736(20)30183-5", "10.1016/s0140-6736(20)30183-5"),
    ],
)
def test_angle_brackets_join_a_doi_only_in_the_sici_shape(text, doi):
    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _contents([("Title", CanonicalSection.TITLE, text, 1)])

    assert [c.normalized for c in collect_doi_candidates(contents)] == [doi]


@pytest.mark.parametrize(
    "caption",
    [
        "Figure 1. Rosette development in S. rosetta. DOI: 10.7554/eLife.00013.003",
        "Figure supplement 1. Frequency of rosette colonies. DOI: 10.7554/eLife.00013.004",
    ],
)
def test_labelled_caption_doi_stays_a_component(caption):
    """eLife PDFs print a numbered DOI under each figure; the suffix rule cannot see it."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    footer = "Alegado et al. eLife 2012;1:e00013. DOI: 10.7554/eLife.00013"
    contents = _contents(
        [("Results", CanonicalSection.RESULTS, caption, 3)], footers=[footer, footer]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.7554/elife.00013"
    assert selection.issues == ()
    [caption_candidate] = [c for c in selection.candidates if c.source_kind == "sentence"]
    assert caption_candidate.marker_kind == "explicit_doi"
    assert caption_candidate.rejection_reason == "component_candidate"


def _docx_contents(front: list[str], body_heading: str, body: list[str]):
    """Parse a DOCX and type its headings the way the no-LLM PostParse does."""

    import io

    docx = pytest.importorskip("docx")

    from bibr.input.docx_native import DocxParser
    from bibr.structure.section_classifier import classify_headers_batch

    document = docx.Document()
    for line in front:
        document.add_paragraph(line)
    document.add_heading("Abstract", level=1)
    document.add_paragraph("We studied attention in older adults.")
    document.add_heading(body_heading, level=1)
    for line in body:
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)

    parser = DocxParser(buffer.getvalue())
    contents = parser.parse()
    parser.apply_segmentation(contents, [[entry.text] for entry in parser.assembler.entries])
    contents.sentences = parser.sentences
    headed = [section for section in contents.sections if section.level > 0]
    for section, (section_type, _score) in zip(
        headed, classify_headers_batch([section.header for section in headed]), strict=True
    ):
        section.section_type = section_type
    return contents


_DOCX_TITLE_PAGE = ["Running head: ATTENTION", "Attention in Older Adults", "Jane Doe"]


@pytest.mark.parametrize(
    "doi_line",
    [
        "https://doi.org/10.1037/xge0001234",
        "10.1037/xge0001234",
        "This is the accepted manuscript of https://doi.org/10.1037/xge0001234",
    ],
)
def test_pageless_title_page_doi_names_the_paper(doi_line):
    """DOCX sentences have no page, so the block before the Abstract is the front matter."""

    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _docx_contents(
        [*_DOCX_TITLE_PAGE, doi_line], "Method", ["Participants were 120 adults."]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1037/xge0001234"
    assert selection.selected.page is None
    assert selection.selected.semantic_context == "front_matter"
    assert selection.selected.selection_tier == 2


def test_pageless_body_doi_still_abstains():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _docx_contents(
        _DOCX_TITLE_PAGE,
        "Method",
        ["We used the stimuli of Smith et al. (2019; https://doi.org/10.1037/xge0000123)."],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert (candidate.semantic_context, candidate.selection_tier) == ("untyped", 1)


def test_pageless_input_without_a_classified_section_has_no_front_block():
    """Without a classified section nothing bounds the front block, so nothing counts."""

    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _contents(
        [
            ("Root text", CanonicalSection.UNKNOWN, "https://doi.org/10.1037/xge0001234", None),
            ("Body", CanonicalSection.UNKNOWN, "Participants were adults.", None),
        ]
    )

    [candidate] = collect_doi_candidates(contents)

    assert candidate.selection_tier == 1


def test_doi_candidate_regex_does_not_backtrack_on_an_unclosed_bracket():
    import time

    from bibr.utils.text import DOI_CANDIDATE_RE

    started = time.perf_counter()
    for text in ("doi: 10.1234/<" + ":" * 100_000, "doi: 10.1234/<" + "a::" * 33_000):
        assert DOI_CANDIDATE_RE.search(text) is None
    assert time.perf_counter() - started < 2.0


def test_a_doi_ending_in_a_slash_ran_on_into_the_next_field():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [], footers=["https://doi.org/10.1234/jex.2026.04.0061234-5678/\u00a9 2026 The Authors."]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    [candidate] = selection.candidates
    assert candidate.normalized == "10.1234/jex.2026.04.0061234-5678/"
    assert candidate.rejection_reason == "line_join_overrun"


@pytest.mark.parametrize(
    ("section_type", "page", "tier"),
    [
        (CanonicalSection.INTRODUCTION, 7, 1),
        (CanonicalSection.INTRODUCTION, 2, 3),
        (CanonicalSection.ABSTRACT, 3, 3),
    ],
)
def test_a_doi_label_outranks_other_candidates_only_in_the_front_matter(section_type, page, tier):
    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _contents(
        [("Section", section_type, "As shown before (doi: 10.1234/cited.1).", page)]
    )

    [candidate] = collect_doi_candidates(contents)

    assert candidate.marker_kind == "explicit_doi"
    assert candidate.selection_tier == tier


def test_a_doi_label_in_the_running_footer_still_names_the_paper():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [("Introduction", CanonicalSection.INTRODUCTION, "Body text (doi: 10.1234/cited.1).", 7)],
        footers=["doi: 10.1234/own.9"],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert (selection.selected.normalized, selection.selected.source_kind) == (
        "10.1234/own.9",
        "footer",
    )
    assert selection.selected.selection_tier == 3
    assert selection.issues == ()


def test_a_labelled_doi_outside_the_front_matter_names_the_paper_when_alone():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    own = (
        "The present work has been shared as a preprint on an example server, "
        "https://example.org/abc/, doi: 10.1234/own.9."
    )
    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "A study of examples", 1),
            (
                "References",
                CanonicalSection.REFERENCES,
                "Doe J (2020). Prior work. doi: 10.1234/x.1",
                12,
            ),
            ("Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, own, 15),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/own.9"
    assert (selection.selected.selection_tier, selection.selected.semantic_context) == (
        1,
        "labelled_body",
    )
    assert selection.issues == ()


def test_a_labelled_doi_outside_the_front_matter_never_beats_the_front_matter():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "https://doi.org/10.1234/own.1", 1),
            (
                "Discussion",
                CanonicalSection.DISCUSSION,
                "As shown before (doi: 10.1234/cited.2).",
                7,
            ),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/own.1"
    assert selection.issues == ()


def test_two_labelled_dois_outside_the_front_matter_are_ambiguous():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "A study of examples", 1),
            (
                "Discussion",
                CanonicalSection.DISCUSSION,
                "As shown before (doi: 10.1234/cited.2).",
                7,
            ),
            (
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                "Preprint doi: 10.1234/own.9.",
                15,
            ),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is None
    assert [issue.code for issue in selection.issues] == ["VAL_DOI_AMBIGUOUS"]


@pytest.mark.parametrize(
    ("text", "selected"),
    [
        # A figure's source note cites another work.
        (
            "From: Doe J, Roe R, The Example Group (2009). Reporting items. "
            "Example Med 6(7): e1000097. doi:10.1234/cited.3",
            None,
        ),
        # The article citing itself is still its own DOI.
        (
            "Cite this article: Doe J, Roe R (2020). A study of examples. "
            "Example J 1: 2. doi: 10.1234/own.9",
            "10.1234/own.9",
        ),
    ],
)
def test_a_labelled_doi_in_a_citation_is_a_cited_work(text, selected):
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            ("Title", CanonicalSection.TITLE, "A study of examples", 1),
            ("Summary", CanonicalSection.UNKNOWN, text, 24),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert (selection.selected.normalized if selection.selected else None) == selected


@pytest.mark.parametrize(
    ("header", "rejected"),
    [
        # A reference entry at the top of a page, read as a running header.
        (
            "Doe, J., 2012b. How children cope. J. Example Psychol. 32, 225-233.\r\n"
            "https://doi.org/10.1234/cited.4.",
            True,
        ),
        # The paper's own citation lines stay page furniture.
        ("2017. Proc Example Soc 2, 20:1-15. https://doi.org/10.1234/own.4.", False),
        ("Doe et al. Example J 2020;9:e12345. DOI: https://doi.org/10.1234/own.4", False),
        (
            "Please cite this article as: Doe, J., A study, Example J (2020), "
            "https://doi.org/10.1234/own.4",
            False,
        ),
        ("Cite as: Doe, J. (2020). A study. Example J, 1, 2. https://doi.org/10.1234/own.4", False),
        (
            "Suggested citation: Doe, J. (2020). A study. Example J, 1, 2. "
            "https://doi.org/10.1234/own.4",
            False,
        ),
    ],
)
def test_a_reference_entry_read_as_page_furniture_is_a_reference(header, rejected):
    from bibr.extract.doi_identity import collect_doi_candidates

    contents = _contents([], headers=[header])

    [candidate] = collect_doi_candidates(contents)

    assert (candidate.rejection_reason == "reference_candidate") is rejected


def test_a_correction_notice_names_the_original_article_as_its_parent():
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    contents = _contents(
        [
            (
                "Correction",
                CanonicalSection.TITLE,
                "DOI of original article: 10.1234/original.1",
                1,
            ),
        ],
        footers=["https://doi.org/10.1234/notice.2"],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    parent = next(c for c in selection.candidates if c.normalized == "10.1234/original.1")
    assert (parent.marker_kind, parent.rejection_reason) == ("parent_doi", "component_candidate")
    assert selection.selected is not None
    assert selection.selected.normalized == "10.1234/notice.2"
