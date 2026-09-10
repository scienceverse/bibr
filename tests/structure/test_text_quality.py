"""Parse-quality ("garbage text") detector — ported from Docling.

Covers the pure per-region rater (``rate_text_quality``), the page-grouped
aggregator (``paper_text_quality``), the post-parse wiring / settings gate, and
the ``info.text_quality`` export field. The detector is report-only: it must
never alter extracted data.
"""

from pathlib import Path

import pytest

from bibr.config import Settings
from bibr.structure.text_quality import (
    SCOREABLE_TREATMENTS,
    paper_text_quality,
    rate_text_quality,
)

# ── rate_text_quality: clean text calibrates to 1.0 (no false positives) ──


class TestRateCleanText:
    @pytest.mark.parametrize(
        "text",
        [
            "The quick brown fox jumps over the lazy dog and then returns home.",
            # Scientific prose with stats/equations-adjacent punctuation.
            "We found a significant effect, t(28) = 3.42, p = .003, d = 0.71 overall.",
            # Author initials must not read as fragmented/spaced letters.
            "J. R. R. Tolkien wrote extensively; see also R. A. Fisher (1925).",
            # A DOI / URL is punctuation-dense but legitimate.
            "Available at https://doi.org/10.1177/0956797617707270 for download.",
            # Short headings.
            "Introduction",
            "Materials and Methods",
            # A single spaced-out word is not enough to penalise.
            "F á b o s was the lead author on the study.",
            "",
        ],
    )
    def test_clean_text_scores_one(self, text):
        assert rate_text_quality(text) == 1.0


# ── rate_text_quality: hard-fail garbage → 0.0 ───────────────────────────


class TestRateHardFail:
    def test_replacement_char_fails(self):
        assert rate_text_quality("The quick brown fox � jumps over the dog.") == 0.0

    def test_repeated_token_loop_fails(self):
        # A whitespace token repeated >=5 times in a row (VLM decode loop).
        assert rate_text_quality("brown the the the the the the fox jumps") == 0.0

    def test_repeated_token_run_below_threshold_ok(self):
        # Four repeats is under the >=5 loop threshold.
        assert rate_text_quality("brown the the the the fox jumps over") == 1.0

    def test_symbol_soup_fails(self):
        # >=20 chars that are >=50% non-alphanumeric, non-space.
        assert rate_text_quality("()[]{}()[]{}()[]{}()[]{}") == 0.0

    def test_short_symbol_string_not_flagged(self):
        # Under the length floor, so exempt from the symbol-soup rule.
        assert rate_text_quality("(a=b)") == 1.0


# ── rate_text_quality: spaced-letter fragmentation penalty ───────────────


class TestRateSpacedLetters:
    def test_three_spaced_words_penalised(self):
        # Three runs of >=4 single-char tokens, separated by normal words, so
        # the runs are distinct: 0.1 penalty each -> 0.7.
        text = "F á b o s and J á n o s met G á b o r today"
        assert rate_text_quality(text) == pytest.approx(0.7)

    def test_two_spaced_words_below_threshold(self):
        # Only two runs -> under the >=3 threshold -> no penalty.
        text = "F á b o s and J á n o s wrote the paper"
        assert rate_text_quality(text) == 1.0


# ── paper_text_quality: label filtering + 10th-percentile aggregation ────


class TestPaperAggregation:
    def test_tenth_percentile_math_single_page(self):
        # Nine clean regions (1.0) + one replacement-char region (0.0):
        # np.nanquantile([1]*9 + [0], 0.10) == 0.9 (linear interpolation).
        regions = [(1, "text", f"clean sentence number {i} here") for i in range(9)]
        regions.append((1, "text", "broken � glyph line"))
        report = paper_text_quality(regions, label_treatment={"text": "content"})

        assert report is not None
        assert report.n_regions == 10
        assert report.score == pytest.approx(0.9)
        assert report.page_scores == {1: pytest.approx(0.9)}

    def test_per_page_scores_are_grouped(self):
        regions = [
            (1, "text", "clean text on page one alpha"),
            (1, "text", "broken � glyph on page one"),
            (2, "text", "clean text on page two beta"),
        ]
        report = paper_text_quality(regions, label_treatment={"text": "content"})

        assert report is not None
        assert set(report.page_scores) == {1, 2}
        # Page 2 is entirely clean; page 1 carries the bad region.
        assert report.page_scores[2] == 1.0
        assert report.page_scores[1] < 1.0

    def test_nonscoreable_labels_return_none(self):
        # Formula / image / table regions are non-scoreable (no prose to rate).
        # With NO scoreable region at all, the report is None — the "nothing to
        # score" case is about labels, not empty text.
        regions = [
            (1, "formula", "\\alpha � \\beta"),
            (1, "image", "�����"),
            (1, "table", "| � | � |"),
        ]
        assert (
            paper_text_quality(
                regions,
                label_treatment={
                    "formula": "formula",
                    "image": "figure",
                    "table": "table",
                },
            )
            is None
        )

    def test_empty_scoreable_regions_score_zero(self):
        # Empty scoreable regions are an OCR-coverage failure: layout detected
        # text here but OCR produced nothing. They are scored 0.0 (not skipped),
        # so the 10th-percentile paper score collapses to ~0 and a report IS
        # returned (all_scores is non-empty).
        regions = [(1, "text", t) for t in ("", "   ", "", "  ", "")]
        report = paper_text_quality(regions, label_treatment={"text": "content"})
        assert report is not None
        assert report.n_regions == 5
        assert report.score == pytest.approx(0.0)
        assert report.page_scores == {1: pytest.approx(0.0)}

    def test_same_regions_populated_score_high(self):
        # The SAME five regions, now populated with clean prose, score ~1.0 —
        # proving the low score above measures OCR coverage, not the labels.
        regions = [(1, "text", f"a perfectly clean sentence of prose {i}") for i in range(5)]
        report = paper_text_quality(regions, label_treatment={"text": "content"})
        assert report is not None
        assert report.n_regions == 5
        assert report.score == pytest.approx(1.0)

    def test_empty_scoreable_mixed_with_clean_drags_score(self):
        # One empty scoreable region among nine clean ones pulls the 10th-pct
        # paper score below 1.0 (np.nanquantile([1]*9 + [0], 0.10) == 0.9).
        regions = [(1, "text", f"clean sentence number {i} here") for i in range(9)]
        regions.append((1, "text", ""))
        report = paper_text_quality(regions, label_treatment={"text": "content"})
        assert report is not None
        assert report.n_regions == 10
        assert report.score == pytest.approx(0.9)

    def test_scoreable_treatments_set(self):
        # section_hint (abstract/reference regions) is scored: reference text is
        # where the spaced-letter OCR failure has been observed in production.
        assert sorted(SCOREABLE_TREATMENTS) == ["content", "footnote", "heading", "section_hint"]

    def test_default_label_treatment_uses_real_mapping(self):
        # No explicit map -> lazily resolves PDFParser.LABEL_TREATMENT. "table"
        # is skipped (not scoreable); only the clean "text" region is rated.
        regions = [
            (1, "table", "| � | garbage |"),
            (1, "text", "a perfectly clean sentence of prose"),
        ]
        report = paper_text_quality(regions)
        assert report is not None
        assert report.n_regions == 1
        assert report.score == 1.0


# ── post-parse wiring + settings gate ────────────────────────────────────


def _contents_with_regions(regions):
    """Minimal PaperContents carrying the given (page, label, content) regions."""
    from bibr.paper_contents import PaperContents, PaperSection, RegionSummary

    return PaperContents(
        sentences=[],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: ""},
        region_summaries=[
            RegionSummary(page=page, index=i, label=label, bbox=None, content=content)
            for i, (page, label, content) in enumerate(regions)
        ],
    )


class TestPostParseWiring:
    async def test_score_attached_and_warning_below_threshold(self, monkeypatch):
        from bibr.pipeline.stages.post_parse import post_parse

        monkeypatch.setattr(Settings.pipeline, "text_quality_report", True)
        monkeypatch.setattr(Settings.pipeline, "text_quality_warn_threshold", 0.5)

        # Every content region is garbage -> paper score 0.0 -> below threshold.
        contents = _contents_with_regions(
            [(1, "text", "broken � glyph line"), (1, "text", "another � broken line")]
        )
        paper = await post_parse(
            contents=contents, file_name="x.pdf", file_hash="deadbeef", no_llm=True
        )

        assert paper.text_quality == 0.0
        assert any(w.startswith("low_text_quality:") for w in paper.processing_warnings)

    async def test_clean_paper_no_warning(self, monkeypatch):
        from bibr.pipeline.stages.post_parse import post_parse

        monkeypatch.setattr(Settings.pipeline, "text_quality_report", True)

        contents = _contents_with_regions(
            [(1, "text", "a clean sentence"), (1, "text", "another clean sentence")]
        )
        paper = await post_parse(
            contents=contents, file_name="x.pdf", file_hash="deadbeef", no_llm=True
        )

        assert paper.text_quality == 1.0
        assert not any(w.startswith("low_text_quality:") for w in paper.processing_warnings)

    async def test_gate_off_skips_scoring(self, monkeypatch):
        from bibr.pipeline.stages.post_parse import post_parse

        monkeypatch.setattr(Settings.pipeline, "text_quality_report", False)

        contents = _contents_with_regions([(1, "text", "broken � glyph line")])
        paper = await post_parse(
            contents=contents, file_name="x.pdf", file_hash="deadbeef", no_llm=True
        )

        assert paper.text_quality is None
        assert not any(w.startswith("low_text_quality:") for w in paper.processing_warnings)


# ── export: info.text_quality scalar field ───────────────────────────────


def _minimal_paper(**overrides):
    from bibr.input.file import InputFile, InputFormat
    from bibr.models import PaperMetadata
    from bibr.paper import Paper
    from bibr.paper_contents import PaperContents, PaperSection

    input_file = InputFile(
        path=Path("/tmp/test.pdf"),
        file_hash="abc123",
        input_format=InputFormat(
            file_extension=".pdf", detected_mime_type="application/pdf", file_type="pdf"
        ),
    )
    contents = PaperContents(
        sentences=[],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: ""},
    )
    defaults = {
        "input_file": input_file,
        "metadata": PaperMetadata(doi="10.1234/test", title="Test Paper"),
        "contents": contents,
    }
    defaults.update(overrides)
    return Paper(**defaults)


class TestExportTextQuality:
    """v11 moved the score out of paper metadata: it is a parse-quality
    diagnostic, so it rides ``extraction.diagnostics.text_quality`` (populated
    by ``_build_extraction`` — see TestExtractionProvenance in
    tests/test_export_units.py) and is absent when a Paper is exported outside
    the pipeline."""

    def test_export_emits_text_quality(self):
        from bibr.export.json_export import export_paper_to_json, validate_export
        from tests.export.conftest import extraction_block

        paper = _minimal_paper(text_quality=0.42)
        paper.extraction = extraction_block(diagnostics={"text_quality": paper.text_quality})
        result = export_paper_to_json(paper)

        assert result["extraction"]["diagnostics"]["text_quality"] == pytest.approx(0.42)
        assert "text_quality" not in result["metadata"]
        assert validate_export(result) == []

    def test_export_emits_null_when_unscored(self):
        from bibr.export.json_export import export_paper_to_json, validate_export
        from tests.export.conftest import extraction_block

        paper = _minimal_paper()
        paper.extraction = extraction_block(diagnostics={"text_quality": paper.text_quality})
        result = export_paper_to_json(paper)

        assert result["extraction"]["diagnostics"]["text_quality"] is None
        assert validate_export(result) == []
