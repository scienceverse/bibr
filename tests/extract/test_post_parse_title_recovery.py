"""Null-title recovery and bare-heading guards in the post-parse title path.

Findings H11 (under ownership scope the layout ``detected_title`` is never
consulted, so a printed title bibr already holds is shipped as null) and M11
(the bare-heading guard only tests exact membership, so a Romance-language body
heading is asserted as the article title; and there is no byline-adjacency
policy for multilingual front matter).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.extract.title_candidates import _is_ordinary_body_heading


def _candidate(candidate_id, text, roles, *, reading_order=0, region_label="text"):
    from bibr.extract.front_matter import FrontMatterCandidate

    return FrontMatterCandidate(
        candidate_id=candidate_id,
        source_kind="paragraph",
        reading_order=reading_order,
        page=1,
        bbox=None,
        region_label=region_label,
        font_size=None,
        font_bold=None,
        section_id=1,
        text_ids=(),
        paragraph_id=1,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        roles=frozenset(roles),
    )


def _decide(
    selected,
    *,
    llm_title=None,
    journal=None,
    publisher=None,
    detected_title="",
    prefer_byline_adjacent=False,
):
    """The title decision of an ownership-scoped run whose record is *selected*."""
    from bibr.extract.field_decisions import FieldCandidate, decide_title
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

    block = FrontMatterBlock(
        block_id="selected",
        candidate_ids=tuple(candidate.candidate_id for candidate in selected),
        title_candidate_ids=tuple(
            candidate.candidate_id for candidate in selected if "title" in candidate.roles
        ),
    )
    resolution = FrontMatterResolution(
        candidates=tuple(selected),
        blocks=(block,),
        selected_block_id="selected",
        selection_method="test",
        reason_flags=(),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )
    return decide_title(
        FieldCandidate("title", "llm", llm_title),
        resolution=resolution,
        detected_title=detected_title,
        sections=[],
        journal=journal,
        publisher=publisher,
        scoped=True,
        abstained=False,
        prefer_byline_adjacent=prefer_byline_adjacent,
        doc_info=None,
    )


class TestDetectedTitleFallback:
    """H11: the last-resort detected-title branch, with the existing filters."""

    LSM_TITLE = (
        "Comparative Efficacy of Fractional CO2 Laser and Microneedling "
        "Radiofrequency in Atrophic Acne Scars"
    )

    def test_lsm_printed_title_is_recovered(self):
        decision = _decide([], detected_title=self.LSM_TITLE)

        assert decision.rule == "layout_title_fallback"
        assert decision.value == self.LSM_TITLE
        assert decision.source == "layout_title"
        issues = decision.issues
        assert issues[-1].code == "VAL_TITLE_RECOVERED"
        assert issues[-1].severity == "warning"
        assert not issues[-1].blocking

    @pytest.mark.parametrize(
        ("detected", "journal"),
        [
            ("SCIENTIFIC REPORTS", "Scientific Reports"),
            ("You may also like", None),
            ("COPYRIGHT", None),
        ],
    )
    def test_masthead_and_furniture_detected_titles_stay_null(self, detected, journal):
        decision = _decide([], journal=journal, detected_title=detected)

        assert decision.selected is None
        assert not decision.value
        assert decision.issues == ()

    @pytest.mark.parametrize("detected", ["", "   ", None])
    def test_absent_detected_title_stays_null(self, detected):
        decision = _decide([], detected_title=detected)

        assert decision.selected is None
        assert not decision.value

    def test_never_overwrites_a_title_already_asserted(self):
        decision = _decide([], llm_title="Model title", detected_title=self.LSM_TITLE)

        assert decision.rule == "extracted"
        assert decision.value == "Model title"

    def test_body_heading_detected_title_stays_null(self):
        decision = _decide([], detected_title="APRESENTAÇÃO E ANÁLISE DOS RESULTADOS")

        assert decision.selected is None
        assert not decision.value

    async def test_ownership_scope_recovers_detected_title_end_to_end(self, monkeypatch):
        """The gate is at the call site, so exercise the whole stage."""
        import bibr.pipeline.stages.post_parse as post_parse_module
        from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution
        from bibr.models import PaperMetadata
        from bibr.paper_contents import PaperContents, PaperSection

        contents = PaperContents(
            sentences=[],
            sections=[
                PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
                PaperSection(section_id=1, header="Flurbulations", level=1, parent_section_id=0),
            ],
            tables=[],
            links=[],
            sections_text={0: "", 1: "something unknown"},
            detected_title=self.LSM_TITLE,
        )
        # A selected block with no safe title candidate: the selected-record
        # fallback fails closed, and only the detected-title branch can recover.
        byline = _candidate("byline", "Alice Author, Bob Author", roles={"byline"})
        resolution = FrontMatterResolution(
            candidates=(byline,),
            blocks=(
                FrontMatterBlock(
                    block_id="selected",
                    candidate_ids=(byline.candidate_id,),
                    title_candidate_ids=(),
                ),
            ),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=frozenset(),
            allowed_section_ids=frozenset({1}),
        )

        async def noop(*_args, **_kwargs):
            return None

        def attach(actual_contents, *_args, **_kwargs):
            actual_contents.front_matter_resolution = resolution
            return ()

        async def extract(*_args, **_kwargs):
            return PaperMetadata(doi="", title="")

        monkeypatch.setattr(post_parse_module, "_classify_sections", noop)
        monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach)
        monkeypatch.setattr(post_parse_module, "_normalize_section_structure", noop)
        monkeypatch.setattr(post_parse_module, "_extract_metadata_and_equations", extract)
        monkeypatch.setattr(post_parse_module, "_link_citations", AsyncMock())
        monkeypatch.setattr(
            "bibr.extract.research_integrity.extract_structured_integrity", AsyncMock()
        )

        paper = await post_parse_module.post_parse(
            contents,
            "source.pdf",
            "source-hash",
            llm_client=MagicMock(),
        )

        assert paper.metadata.title == self.LSM_TITLE
        assert "VAL_TITLE_RECOVERED" in {issue.code for issue in paper.validation_issues}


class TestOrdinaryBodyHeadingGuard:
    """M11: bare body headings in Romance languages must fail closed."""

    @pytest.mark.parametrize(
        "heading",
        [
            "apresentação e análise dos resultados",
            "4. resultados e discussão",
            "metodologia",
            "materiales y métodos",
            "conclusiones finales",
            "considerações finais",
            "revisão da literatura",
            "risultati e discussione",
            "introdução",
            "palavras-chave",
        ],
    )
    def test_body_headings_are_rejected(self, heading):
        assert _is_ordinary_body_heading(heading)

    @pytest.mark.parametrize(
        "title",
        [
            "análise dos impactos da inovação em pequenas empresas",
            "resultados de um programa de treinamento cognitivo",
            "gestão do conhecimento em organizações públicas",
            "the effect of sleep deprivation on working memory",
            "conclusiones de la reforma educativa chilena",
        ],
    )
    def test_real_titles_are_not_rejected(self, title):
        assert not _is_ordinary_body_heading(title)

    def test_portuguese_body_heading_is_not_asserted_as_the_title(self):
        decision = _decide(
            [_candidate("heading", "APRESENTAÇÃO E ANÁLISE DOS RESULTADOS", roles={"title"})]
        )

        assert decision.selected is None
        assert not decision.value
        assert decision.issues == ()

    def test_a_real_portuguese_title_still_recovers(self):
        real = "Gestão do conhecimento em pequenas empresas de tecnologia"
        decision = _decide([_candidate("title", real, roles={"title"})])

        assert decision.rule == "selected_record_title"
        assert decision.value == real


class TestBylineAdjacentTitlePreference:
    """M11: prophylactic byline-adjacency policy — default OFF."""

    FRENCH = "Le contrôle de constitutionnalité des lois de finances"
    ENGLISH = "Constitutional review of finance acts"

    def _redp_case(self, *, enabled):
        return _decide(
            [
                _candidate("fr", self.FRENCH, roles={"title"}, reading_order=0),
                _candidate("byline", "Alice Auteur", roles={"byline"}, reading_order=1),
                _candidate("en", self.ENGLISH, roles={"title"}, reading_order=7),
            ],
            llm_title=self.ENGLISH,
            prefer_byline_adjacent=enabled,
        )

    def test_default_settings_leave_the_model_title_untouched(self):
        from bibr.config import Settings

        assert Settings.pipeline.title_prefer_byline_adjacent is False
        decision = self._redp_case(enabled=Settings.pipeline.title_prefer_byline_adjacent)

        assert decision.value == self.ENGLISH
        assert decision.issues == ()

    def test_disabled_flag_leaves_the_model_title_untouched(self):
        decision = self._redp_case(enabled=False)

        assert decision.value == self.ENGLISH

    def test_enabled_prefers_the_row_printed_above_the_byline(self):
        decision = self._redp_case(enabled=True)

        assert decision.rule == "byline_adjacent_title"
        assert decision.value == self.FRENCH
        issues = decision.issues
        assert issues[-1].code == "VAL_TITLE_BYLINE_ADJACENT"
        assert issues[-1].evidence_ids == ("fr",)

    def test_enabled_keeps_a_model_title_grounded_in_the_adjacent_row(self):
        decision = _decide(
            [
                _candidate("fr", self.FRENCH, roles={"title"}, reading_order=0),
                _candidate("byline", "Alice Auteur", roles={"byline"}, reading_order=1),
            ],
            llm_title=self.FRENCH,
            prefer_byline_adjacent=True,
        )

        assert decision.value == self.FRENCH
        assert decision.selected.source == "llm"

    def test_enabled_never_promotes_an_unsafe_row(self):
        decision = _decide(
            [
                _candidate("heading", "RESUMO", roles={"title"}, reading_order=0),
                _candidate("byline", "Alice Auteur", roles={"byline"}, reading_order=1),
            ],
            llm_title=self.ENGLISH,
            prefer_byline_adjacent=True,
        )

        assert decision.value == self.ENGLISH

    def test_enabled_is_a_no_op_without_a_byline_row(self):
        decision = _decide(
            [_candidate("fr", self.FRENCH, roles={"title"}, reading_order=0)],
            llm_title=self.ENGLISH,
            prefer_byline_adjacent=True,
        )

        assert decision.value == self.ENGLISH
