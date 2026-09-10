"""Author-side guards on the core-metadata path.

Covers the fabrication guard (an author list absent from the model's own
input), the shared printed-marker stripper used by both grounding paths, the
email guard in the author sanitizer, and the two last-resort recovery evidence
sources (title+byline composites and CRediT contribution lines).
"""

from unittest import mock

import pandas as pd

from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.paper import PaperAuthor
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.schemas import AuthorLLM, AuthorsLLM, CoreMetadataLLM


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


def _author(author_id: int, given: str, family: str, *, role: list[str] | None = None):
    return PaperAuthor(
        author_id=author_id,
        given=given,
        family=family,
        affiliation="",
        role=role or [],
    )


def _contents(
    *,
    sentences: list[PaperSentence] | None = None,
    sections: list[PaperSection] | None = None,
):
    sentences = sentences or []
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {
            "text_id": [sentence.text_id for sentence in sentences] or [1],
            "section_name": ["Front matter"] * (len(sentences) or 1),
            "text": [sentence.text for sentence in sentences] or ["Front matter"],
            "page_number": [1] * (len(sentences) or 1),
        }
    )
    contents.sentences = sentences
    contents.sections = sections or []
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.processing_warnings = []
    return contents


def _extractor(resolution, llm_metadata, *, contents=None, recovered=None):
    contents = contents if contents is not None else _contents()
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(return_value=llm_metadata)
    llm_client.extract_authors = mock.AsyncMock(
        return_value=AuthorsLLM(authors=list(recovered or []))
    )
    return CoreMetadataExtractor(
        contents,
        llm_client=llm_client,
        front_matter_resolution=resolution,
    )


# --------------------------------------------------------------------------
# 1. Fabrication guard
# --------------------------------------------------------------------------


def test_authors_are_reported_unchecked_when_no_byline_was_printed():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    title = _candidate("c1", "A Study Of Something", roles=frozenset({"title"}), text_ids=(1,))
    group = build_byline_group(_resolution(title))
    assert group.raw_texts == ()

    issues = assess_author_grounding([_author(1, "Anika Q", "Vexler")], group)

    assert [issue.code for issue in issues] == ["VAL_AUTHOR_UNCHECKED"]
    assert issues[0].count == 1
    assert "reason:no_printed_byline" in issues[0].evidence_ids
    assert "author:1" in issues[0].evidence_ids


def test_no_byline_and_no_authors_still_reports_a_code():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    title = _candidate("c1", "A Study Of Something", roles=frozenset({"title"}), text_ids=(1,))

    issues = assess_author_grounding([], build_byline_group(_resolution(title)))

    assert [issue.code for issue in issues] == ["VAL_AUTHOR_MISSING"]
    assert "reason:no_front_matter_byline" in issues[0].evidence_ids
    assert not issues[0].blocking


async def test_all_ungrounded_author_list_is_dropped():
    # The gesec shape: no byline reaches the selected block and the model
    # returns the prompt's illustrative names verbatim.
    title = _candidate(
        "c1",
        "APRESENTACAO E ANALISE DOS RESULTADOS",
        roles=frozenset({"title"}),
        text_ids=(1,),
    )
    resolution = _resolution(title)
    ext = _extractor(
        resolution,
        CoreMetadataLLM(
            title="APRESENTACAO E ANALISE DOS RESULTADOS",
            authors=[
                AuthorLLM(given="Anika Q", family="Vexler"),
                AuthorLLM(given="C.", family="Mboro"),
                AuthorLLM(given="Lisa A. M.", family="van der Quix"),
                AuthorLLM(given="Thomas", family="Henry"),
            ],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert metadata.authors == []
    assert "VAL_AUTHOR_FABRICATED" in [issue.code for issue in ext.validation_issues]


async def test_partially_grounded_author_list_survives():
    # All-or-nothing: one printed author is enough to keep the whole list, so a
    # real author printed outside the rendered block is never deleted.
    title = _candidate("c1", "A Study Of Something", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate("c2", "Regina Klein", roles=frozenset({"byline"}), text_ids=(2,))
    ext = _extractor(
        _resolution(title, byline),
        CoreMetadataLLM(
            title="A Study Of Something",
            authors=[
                AuthorLLM(given="Regina", family="Klein"),
                AuthorLLM(given="Luiz", family="Hardt"),
            ],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert [author.family for author in metadata.authors] == ["Klein", "Hardt"]
    assert "VAL_AUTHOR_FABRICATED" not in [issue.code for issue in ext.validation_issues]


async def test_printed_byline_blocks_the_drop_even_when_nothing_grounds():
    title = _candidate("c1", "A Study Of Something", roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate(
        "c2",
        "Patrik Felipe Nazarioa , Luciana Ferreirab , Jorge Bothc",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )
    ext = _extractor(
        _resolution(title, byline),
        CoreMetadataLLM(
            title="A Study Of Something",
            authors=[
                AuthorLLM(given="Patrik Felipe", family="Nazario"),
                AuthorLLM(given="Luciana Ferreira", family=""),
                AuthorLLM(given="Jorge Both", family=""),
            ],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert len(metadata.authors) == 3
    assert "VAL_AUTHOR_FABRICATED" not in [issue.code for issue in ext.validation_issues]


async def test_surname_only_author_never_triggers_the_drop():
    # A family-only, non-organization author yields no comparable token variant,
    # so it can never be "grounded" — the guard must not read that as fabrication.
    title = _candidate("c1", "A Study Of Something", roles=frozenset({"title"}), text_ids=(1,))
    ext = _extractor(
        _resolution(title),
        CoreMetadataLLM(
            title="A Study Of Something",
            authors=[AuthorLLM(given="", family="Klein")],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert [author.family for author in metadata.authors] == ["Klein"]


def test_prompts_carry_no_invented_person_names():
    from bibr.clients import prompts

    invented = ("Vexler", "Mboro", "van der Quix", "Thomas Henry", "Anika")
    for name in invented:
        assert name not in prompts._AUTHORS_PROMPT, name
        assert name not in prompts._CORE_METADATA_PROMPT, name


# --------------------------------------------------------------------------
# 2 + 3. Shared printed-marker stripper
# --------------------------------------------------------------------------


def test_recovery_grounding_accepts_glued_letter_marker():
    from bibr.extract.core_metadata import grounded_authors_in_context

    grounded, rejected = grounded_authors_in_context(
        [_author(1, "Htin", "Aung"), _author(2, "Kyaw", "Myint")],
        "Htin AungA, Kyaw MyintB",
    )

    assert [author.family for author in grounded] == ["Aung", "Myint"]
    assert rejected == []


def test_marker_stripper_leaves_all_caps_and_ordinary_names_intact():
    from bibr.extract.core_metadata import _strip_attached_markers

    assert _strip_attached_markers("NIKOLAI DINEV") == "NIKOLAI DINEV"
    assert _strip_attached_markers("Alice McDonald") == "Alice McDonald"
    assert _strip_attached_markers("Kim H") == "Kim H"
    assert _strip_attached_markers("Htin AungA") == "Htin Aung"
    assert _strip_attached_markers("Kian Jafari1") == "Kian Jafari"


def test_grounding_accepts_lowercase_affiliation_marker_lists():
    from bibr.extract.core_metadata import grounded_authors_in_context

    authors = [_author(1, "Kaisin", "Yee"), _author(2, "Hiang Khoon", "Tan")]
    grounded, rejected = grounded_authors_in_context(
        authors, "Kaisin Yeea,b,c and Hiang Khoon Tana,b,d,e,∗"
    )
    assert grounded == authors
    assert rejected == []


def test_observability_grounding_strips_glued_affiliation_digits():
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    candidate = _candidate(
        "c1",
        "Alice Example1, Bob Sample2",
        roles=frozenset({"byline"}),
        text_ids=(2,),
    )

    assert (
        assess_author_grounding(
            [_author(1, "Alice", "Example"), _author(2, "Bob", "Sample")],
            build_byline_group(_resolution(candidate)),
        )
        == ()
    )


# --------------------------------------------------------------------------
# 4. Email guard in the author sanitizer
# --------------------------------------------------------------------------


def test_author_sanitizer_drops_email_addresses():
    authors = CoreMetadataExtractor._convert_llm_authors(
        [
            AuthorLLM(given="Muhammad", family="Azzahid"),
            AuthorLLM(given="Muhammadazzahid21@gmail.com", family=""),
            AuthorLLM(given="", family="nengsih@uinjambi.ac.id"),
            AuthorLLM(given="Victor", family="Dirwan victordirwan@gmail.com"),
        ]
    )

    # Pure-address entries vanish; a name with an address glued to it keeps the name.
    assert [(author.given, author.family) for author in authors] == [
        ("Muhammad", "Azzahid"),
        ("Victor", "Dirwan"),
    ]


# --------------------------------------------------------------------------
# 5a. Composite title+byline records yield recovery evidence
# --------------------------------------------------------------------------


def test_composite_title_byline_record_yields_recovery_context():
    from bibr.extract.core_metadata import byline_recovery_context

    composite = _candidate(
        "c1",
        "EVALUATION OF SOIL QUALITY INDICATORS IN ORGANIC FERTILIZATION AS AN ALTERNATIVE "
        "TO SUSTAINABLE AGRICULTURE Mariana HRISTOVA*1 , Ana KATSAROVA2 , "
        "Veselina VASILEVA2 , Nikolai DINEV2",
        roles=frozenset({"title", "byline"}),
        text_ids=(2,),
    )

    context = byline_recovery_context(_resolution(composite))

    assert context is not None
    assert "Mariana HRISTOVA" in context
    assert "Nikolai DINEV" in context


def test_composite_title_only_record_yields_no_recovery_context():
    from bibr.extract.core_metadata import byline_recovery_context

    composite = _candidate(
        "c1",
        "EVALUATION OF SOIL QUALITY INDICATORS IN ORGANIC FERTILIZATION AS AN ALTERNATIVE "
        "TO SUSTAINABLE AGRICULTURE",
        roles=frozenset({"title", "byline"}),
        text_ids=(2,),
    )

    assert byline_recovery_context(_resolution(composite)) is None


def test_composite_recovery_evidence_does_not_change_expected_entries():
    # The composite record must not become an "expected author entry" — that
    # would report the title chunk as a missing author on every such paper.
    from bibr.extract.core_metadata import assess_author_grounding, build_byline_group

    composite = _candidate(
        "c1",
        "EVALUATION OF SOIL QUALITY INDICATORS IN ORGANIC FERTILIZATION AS AN ALTERNATIVE "
        "TO SUSTAINABLE AGRICULTURE Mariana HRISTOVA*1 , Ana KATSAROVA2",
        roles=frozenset({"title", "byline"}),
        text_ids=(2,),
    )
    group = build_byline_group(_resolution(composite))

    assert group.missing_inference_raw_texts == ()
    issues = assess_author_grounding(
        [_author(1, "Mariana", "Hristova"), _author(2, "Ana", "Katsarova")],
        group,
    )
    assert [issue.code for issue in issues] == []


# --------------------------------------------------------------------------
# 5b. CRediT last-resort harvest
# --------------------------------------------------------------------------


async def test_credit_lines_recover_authors_when_the_list_is_empty():
    sentences = [
        PaperSentence(
            text_id=47,
            text="Dalia Zayed: Writing - review & editing, Writing - original draft.",
            section_id=1,
            paragraph_id=1,
        ),
        PaperSentence(
            text_id=48,
            text="Mus'ab Banat: Writing - review & editing, Data curation.",
            section_id=1,
            paragraph_id=1,
        ),
        PaperSentence(
            text_id=49,
            text="Ala'a B. Al-Tammemi: Supervision, Conceptualization.",
            section_id=1,
            paragraph_id=1,
        ),
    ]
    sections = [
        PaperSection(
            1,
            "Credit Authorship Contribution Statement",
            2,
            None,
            CanonicalSection.AUTHOR_CONTRIBUTIONS,
            1.0,
        )
    ]
    title = _candidate("c1", "A Study Of Something", roles=frozenset({"title"}), text_ids=(1,))
    ext = _extractor(
        _resolution(title),
        CoreMetadataLLM(title="A Study Of Something", authors=[], keywords=[]),
        contents=_contents(sentences=sentences, sections=sections),
    )

    metadata = await ext.extract()

    assert [(author.given, author.family) for author in metadata.authors] == [
        ("Dalia", "Zayed"),
        ("Mus'ab", "Banat"),
        ("Ala'a B.", "Al-Tammemi"),
    ]
    assert "VAL_AUTHOR_CREDIT_RECOVERED" in [issue.code for issue in ext.validation_issues]


async def test_credit_harvest_never_overwrites_extracted_authors():
    sentences = [
        PaperSentence(
            text_id=47,
            text="Dalia Zayed: Writing - review & editing.",
            section_id=1,
            paragraph_id=1,
        ),
    ]
    sections = [
        PaperSection(
            1,
            "Credit Authorship Contribution Statement",
            2,
            None,
            CanonicalSection.AUTHOR_CONTRIBUTIONS,
            1.0,
        )
    ]
    byline = _candidate("c1", "Regina Klein", roles=frozenset({"byline"}), text_ids=(1,))
    ext = _extractor(
        _resolution(byline),
        CoreMetadataLLM(
            title="A Study",
            authors=[AuthorLLM(given="Regina", family="Klein")],
            keywords=[],
        ),
        contents=_contents(sentences=sentences, sections=sections),
    )

    metadata = await ext.extract()

    assert [(author.given, author.family) for author in metadata.authors] == [("Regina", "Klein")]


async def test_credit_harvest_ignores_role_first_contribution_lines():
    sentences = [
        PaperSentence(
            text_id=47,
            text="Conceptualization: D.Z. and M.B.; methodology: D.Z.",
            section_id=1,
            paragraph_id=1,
        ),
        PaperSentence(
            text_id=48,
            text="All authors have read and agreed to the published version.",
            section_id=1,
            paragraph_id=1,
        ),
    ]
    sections = [
        PaperSection(
            1,
            "Author Contributions",
            2,
            None,
            CanonicalSection.AUTHOR_CONTRIBUTIONS,
            1.0,
        )
    ]
    title = _candidate("c1", "A Study Of Something", roles=frozenset({"title"}), text_ids=(1,))
    ext = _extractor(
        _resolution(title),
        CoreMetadataLLM(title="A Study Of Something", authors=[], keywords=[]),
        contents=_contents(sentences=sentences, sections=sections),
    )

    metadata = await ext.extract()

    assert metadata.authors == []


class TestBrokenDiacriticGrounding:
    """Broken PDF font encodings must not discard correctly extracted accented names.

    A spacing acute accent can follow a dotless i or be separated from a capital A. Generic Unicode normalization does not repair that encoding damage."""

    @staticmethod
    def _tokens(value):
        from bibr.extract.core_metadata import _name_tokens

        return _name_tokens(value)

    def test_spacing_accent_forms_ground_against_the_printed_name(self):
        # The exact strings from that paper's text layer.
        assert self._tokens("Marı´a A´ ngeles Cintado") == self._tokens("María Ángeles Cintado")
        assert self._tokens("Lucı´a Ca´rcel") == self._tokens("Lucía Cárcel")
        assert self._tokens("Gabriel Gonza´lez") == self._tokens("Gabriel González")

    def test_umlaut_and_slashed_letter_forms_also_ground(self):
        assert self._tokens("Jo¨rg Mu¨ller") == self._tokens("Jörg Müller")
        assert self._tokens("Łukasz Wis´niewski") == self._tokens("Łukasz Wiśniewski")

    def test_folding_does_not_dissolve_names_into_each_other(self):
        assert self._tokens("Smith") != self._tokens("Smyth")
        assert self._tokens("Hansen") != self._tokens("Hanson")

    def test_apostrophes_and_hyphens_survive_the_fold(self):
        assert self._tokens("O'Brien") == self._tokens("O’Brien") == ("o'brien",)
        assert self._tokens("Anne-Marie Smith-Jones") == ("anne-marie", "smith-jones")
