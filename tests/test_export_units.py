"""Unit tests for JSON export pipeline components.

Covers: canonicalize_orcid, _compute_paper_id, _merge_ocr_metadata,
section remapping, xref resolution, match serialization.
"""

import json
from pathlib import Path

import pandas as pd

from bibr.export.json_export import build_paper_export, export_paper_to_json
from bibr.input.file import InputFile, InputFormat
from bibr.models import (
    BibAuthor,
    ExternalMatch,
    MatchSource,
    PaperAuthor,
    PaperMetadata,
    PaperReference,
    canonicalize_orcid,
)
from bibr.paper import Paper, _merge_ocr_metadata
from bibr.paper_contents import (
    CanonicalSection,
    CitationCandidate,
    CitationLinkingReceipt,
    PaperContents,
    PaperEquation,
    PaperFigure,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperURLLink,
    PaperXref,
)
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.structure.xref_utils import detect_xrefs
from tests.export.conftest import extraction_block as _extraction_block

# ── helpers ────────────────────────────────────────────────────────────


def _input_file(name: str = "test.pdf") -> InputFile:
    return InputFile(
        path=Path(f"/tmp/{name}"),
        file_hash="abc123",
        input_format=InputFormat(
            file_extension=".pdf", detected_mime_type="application/pdf", file_type="pdf"
        ),
    )


def _minimal_contents(**overrides) -> PaperContents:
    defaults = {
        "sentences": [PaperSentence(text_id=1, text="Hello.", section_id=1, paragraph_id=1)],
        "sections": [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Intro",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.INTRODUCTION,
            ),
        ],
        "tables": [],
        "links": [],
        "sections_text": {1: "Hello."},
    }
    defaults.update(overrides)
    return PaperContents(**defaults)


def _minimal_paper(**overrides) -> Paper:
    defaults = {
        "input_file": _input_file(),
        "metadata": PaperMetadata(doi="10.1234/test", title="Test Paper"),
        "contents": _minimal_contents(),
    }
    defaults.update(overrides)
    return Paper(**defaults)


def test_build_paper_export_matches_dictionary_export():
    from bibr.export import PaperExport

    paper = _minimal_paper()

    model = build_paper_export(paper, validate=False)

    assert isinstance(model, PaperExport)
    assert model.model_dump(by_alias=True, exclude_unset=True) == export_paper_to_json(
        paper, validate=False
    )


def test_citation_linking_receipt_exports_losslessly_and_round_trips_schema():
    from bibr.export import PaperExport

    candidate = CitationCandidate(
        text_id=7,
        start=12,
        end=17,
        raw="[1-3]",
        style="numeric",
        bib_ids=(1, 2, 3),
        evidence=("bracket_marker", "recurring_sentences:4"),
        confidence=0.95,
        accepted=True,
        rejection_reasons=(),
    )
    receipt = CitationLinkingReceipt(
        style_scores={"numeric": 1.0, "author-year": 0.0},
        candidates=(candidate,),
        resolved_candidate_fraction=1.0,
        unique_linked_bib_fraction=0.3,
    )
    paper = _minimal_paper(contents=_minimal_contents(citation_receipt=receipt))
    paper.extraction = _extraction_block()

    out = export_paper_to_json(paper, validate=False)

    assert out["extraction"]["diagnostics"]["citation_linking"] == {
        "style_scores": {"numeric": 1.0, "author-year": 0.0},
        "candidates": [
            {
                "text_id": 7,
                "start": 12,
                "end": 17,
                "raw": "[1-3]",
                "style": "numeric",
                "bib_ids": [1, 2, 3],
                "evidence": ["bracket_marker", "recurring_sentences:4"],
                "confidence": 0.95,
                "accepted": True,
                "rejection_reasons": [],
            }
        ],
        "resolved_candidate_fraction": 1.0,
        "unique_linked_bib_fraction": 0.3,
    }
    assert PaperExport.model_validate(out).extraction.diagnostics.citation_linking is not None


def test_citation_linking_receipt_is_omitted_when_unavailable_with_or_without_validation():
    paper = _minimal_paper()
    paper.extraction = _extraction_block()

    for validate in (False, True):
        out = export_paper_to_json(paper, validate=validate)
        assert "citation_linking" not in out
        assert "citation_linking" not in out["extraction"]["diagnostics"]


def test_export_sanitizes_lone_surrogates_for_utf8_json_response():
    contents = _minimal_contents(
        sentences=[
            PaperSentence(
                text_id=1,
                text="Bad OCR scalar \ud800 in body.",
                section_id=1,
                paragraph_id=1,
            )
        ],
        sections_text={1: "Bad OCR scalar \udc00 in body."},
    )
    paper = _minimal_paper(
        metadata=PaperMetadata(doi="10.1234/test", title="Title with \ud800"),
        contents=contents,
    )

    out = export_paper_to_json(paper)

    json.dumps(out, ensure_ascii=False).encode("utf-8")
    assert out["metadata"]["title"] == "Title with �"
    assert out["text"][0]["text"] == "Bad OCR scalar � in body."


def test_export_runs_recursive_string_sanitizer_once(monkeypatch):
    from bibr.export import json_export

    calls = []

    def identity(payload):
        calls.append(payload)
        return payload

    monkeypatch.setattr(json_export, "_sanitize_json_strings", identity)
    json_export.export_paper_to_json(_minimal_paper(), validate=True)

    assert len(calls) == 1


def test_sane_url_rejects_invalid_bracketed_authority():
    from bibr.export.json_export import _is_sane_url

    assert _is_sane_url("https://doi.org[10.1037/example") is False


class TestConsolidatedExportValidates:
    def test_consolidated_export_passes_schema_validation(self):
        # Consolidation records its receipt under extraction.diagnostics after
        # export; the strict models must accept it — consolidated output
        # shipped to clients has to validate against our own schema.
        from bibr.enrich.consolidate import consolidate_bibs
        from bibr.export.json_export import validate_export

        ref = PaperReference(
            bib_id=1,
            title="T",
            first_page=None,
            volume=None,
            authors=None,
            year=2020,
            container=None,
        )
        ref.match = {MatchSource.CROSSREF: ExternalMatch(doi="10.1234/x", score=99.0)}
        meta = PaperMetadata(doi="10.1234/x", title="T", references=[ref])
        paper = _minimal_paper(metadata=meta)
        paper.extraction = _extraction_block()
        out = export_paper_to_json(paper)

        n = consolidate_bibs(out, mode="fill")

        assert n == 1
        assert out["extraction"]["diagnostics"]["consolidation"] == [
            {"bib_id": 1, "fields": ["doi"]}
        ]
        assert validate_export(out) == []


# ── structured enrichment completeness ────────────────────────────────


class TestEnrichmentExport:
    """``extraction.enrichment`` must surface enrichment completeness as
    structured data so a consumer can tell a partial (timed-out) enrichment
    from a complete one instead of grepping warnings."""

    @staticmethod
    def _ref(bib_id: int, matched: bool) -> PaperReference:
        r = PaperReference(
            bib_id=bib_id,
            title="T",
            first_page=None,
            volume=None,
            authors=None,
            year=2020,
            container=None,
        )
        if matched:
            r.match = {MatchSource.CROSSREF: ExternalMatch(doi="10.1/x", score=99.0)}
        return r

    def test_partial_enrichment_reports_incomplete(self):
        meta = PaperMetadata(
            doi="10.1/x",
            title="T",
            references=[self._ref(1, True), self._ref(2, True), self._ref(3, False)],
        )
        meta.enrichment_complete = False
        paper = _minimal_paper(metadata=meta)
        paper.extraction = _extraction_block()
        out = export_paper_to_json(paper)
        assert out["extraction"]["enrichment"] == {
            "complete": False,
            "refs_enriched": 2,
            "refs_total": 3,
        }

    def test_complete_enrichment(self):
        meta = PaperMetadata(
            doi="10.1/x",
            title="T",
            references=[self._ref(1, True), self._ref(2, True)],
        )
        meta.enrichment_complete = True
        paper = _minimal_paper(metadata=meta)
        paper.extraction = _extraction_block()
        enrichment = export_paper_to_json(paper)["extraction"]["enrichment"]
        assert enrichment["complete"] is True
        assert enrichment["refs_enriched"] == 2
        assert enrichment["refs_total"] == 2

    def test_enrichment_omitted_when_it_never_ran(self):
        # enrichment_complete left at its default (None) → the subsystem did not
        # run, so the key is absent rather than a zeroed row.
        meta = PaperMetadata(doi="10.1/x", title="T", references=[self._ref(1, False)])
        paper = _minimal_paper(metadata=meta)
        paper.extraction = _extraction_block()
        out = export_paper_to_json(paper)
        assert "enrichment" not in out["extraction"]
        assert "enrichment" not in out


# ── structured affiliations ────────────────────────────────────────────


class TestAffiliationsExport:
    """Top-level ``affiliation`` mirrors ``funding``: a 1-based list carrying
    our verbatim ``text`` plus the LLM's best-effort structured components."""

    def test_empty_by_default(self):
        out = export_paper_to_json(_minimal_paper())
        assert out["affiliation"] == []

    def test_roundtrips_with_ids_and_null_components(self):
        from bibr.models import Affiliation

        meta = PaperMetadata(
            doi="10.1/x",
            title="T",
            affiliations=[
                Affiliation(
                    text="Dept of Psychology, Univ X, London, UK",
                    institution="Univ X",
                    department="Dept of Psychology",
                    city="London",
                    country="UK",
                    author_ids=[1, 2],
                ),
                Affiliation(text="Univ Y"),
            ],
        )
        out = export_paper_to_json(_minimal_paper(metadata=meta))
        assert out["affiliation"] == [
            {
                "affiliation_id": 1,
                "text": "Dept of Psychology, Univ X, London, UK",
                "institution": "Univ X",
                "department": "Dept of Psychology",
                "city": "London",
                "country": "UK",
                "author_ids": [1, 2],
            },
            {
                "affiliation_id": 2,
                "text": "Univ Y",
                "institution": None,
                "department": None,
                "city": None,
                "country": None,
                "author_ids": [],
            },
        ]


# ── canonicalize_orcid ─────────────────────────────────────────────────


class TestCanonicalizeOrcid:
    def test_none_returns_none(self):
        assert canonicalize_orcid(None) is None

    def test_empty_string_returns_none(self):
        assert canonicalize_orcid("") is None

    def test_whitespace_only_returns_none(self):
        assert canonicalize_orcid("   ") is None

    def test_non_string_returns_none(self):
        assert canonicalize_orcid(12345) is None

    def test_bare_orcid_gets_prefix(self):
        assert canonicalize_orcid("0000-0002-1825-0097") == "https://orcid.org/0000-0002-1825-0097"

    def test_bare_orcid_with_x_checksum(self):
        assert canonicalize_orcid("0000-0001-5109-381X") == "https://orcid.org/0000-0001-5109-381X"

    def test_https_url_unchanged(self):
        assert (
            canonicalize_orcid("https://orcid.org/0000-0002-1825-0097")
            == "https://orcid.org/0000-0002-1825-0097"
        )

    def test_http_url_upgraded_to_https(self):
        assert (
            canonicalize_orcid("http://orcid.org/0000-0002-1825-0097")
            == "https://orcid.org/0000-0002-1825-0097"
        )

    def test_leading_trailing_whitespace_stripped(self):
        assert (
            canonicalize_orcid("  0000-0002-1825-0097  ") == "https://orcid.org/0000-0002-1825-0097"
        )

    def test_invalid_format_dropped(self):
        assert canonicalize_orcid("not-an-orcid") is None

    def test_partial_orcid_dropped(self):
        # Too few groups
        assert canonicalize_orcid("0000-0002-1825") is None

    def test_byline_superscript_dropped(self):
        # Author-index superscripts must not be exported as an ORCID.
        assert canonicalize_orcid("1") is None


# ── _compute_paper_id ──────────────────────────────────────────────────


class TestComputePaperId:
    def test_user_supplied_id_takes_priority(self):
        paper = _minimal_paper(paper_id="custom-id")
        assert paper._compute_paper_id() == "custom-id"

    def test_file_stem_used_when_no_user_id(self):
        # Not the DOI: the id must not change when a later bibr reads the DOI
        # differently, and it matches ``bibr batch`` and metacheck's stem ids.
        paper = _minimal_paper()
        assert paper.metadata.doi == "10.1234/test"
        assert paper._compute_paper_id() == "test"

    def test_file_stem_without_doi(self):
        paper = _minimal_paper(
            metadata=PaperMetadata(doi="", title="No DOI Paper"),
        )
        assert paper._compute_paper_id() == "test"

    def test_user_id_beats_doi(self):
        paper = _minimal_paper(paper_id="my-id")
        paper.metadata.doi = "10.9999/should-not-use"
        assert paper._compute_paper_id() == "my-id"

    def test_no_metadata_falls_to_file_stem(self):
        paper = _minimal_paper(metadata=None)
        assert paper._compute_paper_id() == "test"


# ── _merge_ocr_metadata ───────────────────────────────────────────────


class TestOcrMetadataSchema:
    def test_parse_valid_dict(self):
        from bibr.paper import OcrFallbackMetadata

        m = OcrFallbackMetadata.from_raw(
            {"title": "Test", "doi": "10.1/x", "keywords": ["ML"], "authors": ["Jane Doe"]}
        )
        assert m.title == "Test"
        assert m.doi == "10.1/x"
        assert m.keywords == ["ML"]
        assert m.authors == ["Jane Doe"]

    def test_parse_empty_dict(self):
        from bibr.paper import OcrFallbackMetadata

        m = OcrFallbackMetadata.from_raw({})
        assert m.title is None
        assert m.doi is None
        assert m.keywords == []
        assert m.authors == []

    def test_non_list_keywords_coerced_to_empty(self):
        from bibr.paper import OcrFallbackMetadata

        m = OcrFallbackMetadata.from_raw({"keywords": "not a list"})
        assert m.keywords == []

    def test_non_string_keyword_items_coerced_to_empty(self):
        from bibr.paper import OcrFallbackMetadata

        m = OcrFallbackMetadata.from_raw({"keywords": [1, 2, 3]})
        assert m.keywords == []

    def test_non_list_authors_coerced_to_empty(self):
        from bibr.paper import OcrFallbackMetadata

        m = OcrFallbackMetadata.from_raw({"authors": "not a list"})
        assert m.authors == []


# ── _merge_ocr_metadata ───────────────────────────────────────────────


def _fill_from_doc_info(meta, ocr):
    """Unscoped doc-info fills: the title, keyword and author decisions, then the DOI."""
    from bibr.extract.field_decisions import (
        apply_decision,
        decide_authors,
        decide_keywords,
        decide_title,
        incumbent_candidate,
    )
    from bibr.paper import doc_info_candidates

    doc_info = doc_info_candidates(ocr)
    authors = [incumbent_candidate(meta, "author", source=None)]
    if "author" in doc_info:
        authors.append(doc_info["author"])
    apply_decision(meta, decide_authors(authors))
    apply_decision(
        meta,
        decide_title(
            incumbent_candidate(meta, "title", source=None),
            resolution=None,
            detected_title=None,
            sections=[],
            journal=None,
            publisher=None,
            scoped=False,
            abstained=False,
            prefer_byline_adjacent=False,
            doc_info=doc_info.get("title"),
        ),
    )
    apply_decision(
        meta,
        decide_keywords(
            incumbent_candidate(meta, "keywords", source=None),
            doc_info=doc_info.get("keywords"),
            section=None,
            abstained=False,
        ),
    )
    _merge_ocr_metadata(meta, ocr)


class TestMergeOcrMetadata:
    def test_fills_empty_title(self):
        meta = PaperMetadata(doi="10.1/x", title="")
        _fill_from_doc_info(meta, {"title": "OCR Title"})
        assert meta.title == "OCR Title"

    def test_does_not_overwrite_existing_title(self):
        meta = PaperMetadata(doi="10.1/x", title="LLM Title")
        _fill_from_doc_info(meta, {"title": "OCR Title"})
        assert meta.title == "LLM Title"

    def test_fills_empty_doi(self):
        meta = PaperMetadata(doi="", title="T")
        _fill_from_doc_info(meta, {"doi": "10.1234/ocr"})
        assert meta.doi == "10.1234/ocr"

    def test_rejects_invalid_doi(self):
        meta = PaperMetadata(doi="", title="T")
        _fill_from_doc_info(meta, {"doi": "not-a-doi"})
        assert meta.doi == ""

    def test_does_not_overwrite_existing_doi(self):
        meta = PaperMetadata(doi="10.1/existing", title="T")
        _fill_from_doc_info(meta, {"doi": "10.1/ocr"})
        assert meta.doi == "10.1/existing"

    def test_fills_keywords(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _fill_from_doc_info(meta, {"keywords": ["ML", "NLP"]})
        assert meta.keywords == ["ML", "NLP"]

    def test_rejects_non_list_keywords(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _fill_from_doc_info(meta, {"keywords": "not a list"})
        assert meta.keywords == []

    def test_rejects_non_string_keyword_items(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _fill_from_doc_info(meta, {"keywords": [1, 2, 3]})
        assert meta.keywords == []

    def test_fills_authors_from_name_strings(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _fill_from_doc_info(meta, {"authors": ["Jane Smith", "Bob Lee"]})
        assert len(meta.authors) == 2
        assert meta.authors[0].given == "Jane"
        assert meta.authors[0].family == "Smith"
        assert meta.authors[0].author_id == 1
        assert meta.authors[1].given == "Bob"
        assert meta.authors[1].family == "Lee"
        assert meta.authors[1].author_id == 2

    def test_single_name_author(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _fill_from_doc_info(meta, {"authors": ["Madonna"]})
        assert len(meta.authors) == 1
        assert meta.authors[0].given == ""
        assert meta.authors[0].family == "Madonna"

    def test_skips_blank_author_names(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _fill_from_doc_info(meta, {"authors": ["", "  ", "Jane Doe"]})
        assert len(meta.authors) == 1
        assert meta.authors[0].family == "Doe"

    def test_does_not_overwrite_existing_authors(self):
        meta = PaperMetadata(
            doi="10.1/x",
            title="T",
            authors=[
                PaperAuthor(author_id=1, given="Existing", family="Author", affiliation=""),
            ],
        )
        _fill_from_doc_info(meta, {"authors": ["New Person"]})
        assert len(meta.authors) == 1
        assert meta.authors[0].family == "Author"

    def test_empty_ocr_dict_is_noop(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _fill_from_doc_info(meta, {})
        assert meta.title == "T"

    def test_title_stripped(self):
        meta = PaperMetadata(doi="10.1/x", title="")
        _fill_from_doc_info(meta, {"title": "  Padded Title  "})
        assert meta.title == "Padded Title"


# ── xref detection (detect_xrefs) ─────────────────────────────────────


class TestDetectXrefs:
    def test_table_xref_detected(self):
        sentences = [
            PaperSentence(text_id=1, text="See Table 2 for results.", section_id=1, paragraph_id=1)
        ]
        tables = [PaperTable(table_id=2, df=pd.DataFrame(), tbl_html="", section_id=1, label="2")]
        xrefs = detect_xrefs(sentences, tables, [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "table"
        assert xrefs[0].xref_id == 2
        assert xrefs[0].text_id == 1
        assert xrefs[0].tier == "label"

    def test_figure_xref_detected(self):
        sentences = [
            PaperSentence(text_id=1, text="As shown in Figure 3.", section_id=1, paragraph_id=1)
        ]
        figures = [PaperFigure(figure_id=3, section_id=1, image_b64=None, caption=None, label="3")]
        xrefs = detect_xrefs(sentences, [], figures)
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "figure"
        assert xrefs[0].xref_id == 3

    def test_abbreviated_table_ref(self):
        sentences = [
            PaperSentence(text_id=1, text="Tab. 1 shows values.", section_id=1, paragraph_id=1)
        ]
        tables = [PaperTable(table_id=1, df=pd.DataFrame(), tbl_html="", section_id=1)]
        xrefs = detect_xrefs(sentences, tables, [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "table"

    def test_abbreviated_figure_ref(self):
        sentences = [
            PaperSentence(text_id=1, text="See Fig. 1 and Fig. 2.", section_id=1, paragraph_id=1)
        ]
        figures = [
            PaperFigure(figure_id=1, section_id=1, image_b64=None, caption=None),
            PaperFigure(figure_id=2, section_id=1, image_b64=None, caption=None),
        ]
        xrefs = detect_xrefs(sentences, [], figures)
        assert len(xrefs) == 2

    def test_nonexistent_table_keeps_a_row_without_target(self):
        sentences = [
            PaperSentence(text_id=1, text="Table 99 is referenced.", section_id=1, paragraph_id=1)
        ]
        tables = [PaperTable(table_id=1, df=pd.DataFrame(), tbl_html="", section_id=1)]
        xrefs = detect_xrefs(sentences, tables, [])
        assert [(x.xref_type, x.xref_id, x.tier) for x in xrefs] == [("table", 0, "position")]

    def test_multiple_tables_in_one_sentence(self):
        sentences = [
            PaperSentence(text_id=1, text="Tables 1, 2, 3 show.", section_id=1, paragraph_id=1)
        ]
        tables = [
            PaperTable(table_id=i, df=pd.DataFrame(), tbl_html="", section_id=1)
            for i in range(1, 4)
        ]
        xrefs = detect_xrefs(sentences, tables, [])
        assert len(xrefs) == 3

    def test_mentions_without_items_have_no_target(self):
        sentences = [
            PaperSentence(text_id=1, text="Table 1 and Figure 1.", section_id=1, paragraph_id=1)
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert [(x.xref_type, x.xref_id) for x in xrefs] == [("table", 0), ("figure", 0)]

    # ── Compound "and" refs ──

    def test_tables_and_compound(self):
        sentences = [
            PaperSentence(text_id=1, text="Tables 5 and 6 show.", section_id=1, paragraph_id=1)
        ]
        tables = [
            PaperTable(table_id=i, df=pd.DataFrame(), tbl_html="", section_id=1, label=str(i))
            for i in range(5, 7)
        ]
        xrefs = detect_xrefs(sentences, tables, [])
        assert len(xrefs) == 2
        assert {x.xref_id for x in xrefs} == {5, 6}

    def test_figures_and_compound(self):
        sentences = [
            PaperSentence(
                text_id=1, text="Figures 1 and 2 illustrate.", section_id=1, paragraph_id=1
            )
        ]
        figures = [
            PaperFigure(figure_id=i, section_id=1, image_b64=None, caption=None)
            for i in range(1, 3)
        ]
        xrefs = detect_xrefs(sentences, [], figures)
        assert len(xrefs) == 2
        assert {x.xref_id for x in xrefs} == {1, 2}

    # ── Range expansion ──

    def test_table_range_expansion(self):
        """Tables 5-7 should produce xrefs for 5, 6, and 7."""
        sentences = [PaperSentence(text_id=1, text="See Tables 5-7.", section_id=1, paragraph_id=1)]
        tables = [
            PaperTable(table_id=i, df=pd.DataFrame(), tbl_html="", section_id=1, label=str(i))
            for i in range(5, 8)
        ]
        xrefs = detect_xrefs(sentences, tables, [])
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {5, 6, 7}

    def test_figure_range_expansion(self):
        """Figs. 1-3 should produce xrefs for 1, 2, and 3."""
        sentences = [PaperSentence(text_id=1, text="See Figs. 1-3.", section_id=1, paragraph_id=1)]
        figures = [
            PaperFigure(figure_id=i, section_id=1, image_b64=None, caption=None)
            for i in range(1, 4)
        ]
        xrefs = detect_xrefs(sentences, [], figures)
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {1, 2, 3}

    # ── Supplementary xrefs ──

    def test_supp_table_xref(self):
        sentences = [
            PaperSentence(text_id=1, text="See Table S1 for details.", section_id=1, paragraph_id=1)
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "supplementary"
        assert xrefs[0].xref_id == 1
        assert "Table S1" in xrefs[0].contents

    def test_supp_figure_xref(self):
        sentences = [
            PaperSentence(text_id=1, text="Fig. S2 shows data.", section_id=1, paragraph_id=1)
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "supplementary"
        assert xrefs[0].xref_id == 2

    def test_supplementary_table_named(self):
        sentences = [
            PaperSentence(
                text_id=1, text="Supplementary Table 3 lists.", section_id=1, paragraph_id=1
            )
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "supplementary"
        assert xrefs[0].xref_id == 3

    def test_supplemental_material_no_number(self):
        sentences = [
            PaperSentence(
                text_id=1, text="See Supplemental Material.", section_id=1, paragraph_id=1
            )
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "supplementary"
        assert xrefs[0].xref_id == 0

    def test_supplementary_figure_named(self):
        sentences = [
            PaperSentence(
                text_id=1, text="Supplementary Figure 2 shows.", section_id=1, paragraph_id=1
            )
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "supplementary"
        assert xrefs[0].xref_id == 2

    def test_supp_table_not_confused_with_regular_table(self):
        """Table S1 should be 'supplementary', not 'table'."""
        sentences = [
            PaperSentence(text_id=1, text="Table S1 and Table 2.", section_id=1, paragraph_id=1)
        ]
        tables = [PaperTable(table_id=2, df=pd.DataFrame(), tbl_html="", section_id=1)]
        xrefs = detect_xrefs(sentences, tables, [])
        types = {x.xref_type for x in xrefs}
        assert "supplementary" in types
        assert "table" in types
        assert len(xrefs) == 2

    # ── Equation xrefs ──

    def test_equation_xref(self):
        sentences = [PaperSentence(text_id=1, text="See Equation 3.", section_id=1, paragraph_id=1)]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "equation"
        assert xrefs[0].xref_id == 3

    def test_eq_abbreviated_xref(self):
        sentences = [PaperSentence(text_id=1, text="As in Eq. 5.", section_id=1, paragraph_id=1)]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "equation"
        assert xrefs[0].xref_id == 5

    def test_eqs_range_xref(self):
        sentences = [PaperSentence(text_id=1, text="See Eqs. 1-3.", section_id=1, paragraph_id=1)]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {1, 2, 3}

    def test_equation_parenthesized(self):
        sentences = [
            PaperSentence(text_id=1, text="Equation (3) yields.", section_id=1, paragraph_id=1)
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 3

    # ── Section xrefs ──

    def test_section_xref(self):
        sentences = [PaperSentence(text_id=1, text="See Section 3.", section_id=1, paragraph_id=1)]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "section"
        assert xrefs[0].xref_id == 3

    def test_section_symbol_xref(self):
        sentences = [
            PaperSentence(text_id=1, text="As described in §2.1.", section_id=1, paragraph_id=1)
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "section"
        assert xrefs[0].xref_id == 2
        assert "§2.1" in xrefs[0].contents

    def test_sections_compound(self):
        sentences = [
            PaperSentence(text_id=1, text="Sections 2 and 3 discuss.", section_id=1, paragraph_id=1)
        ]
        xrefs = detect_xrefs(sentences, [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "section"
        assert "Sections 2 and 3" in xrefs[0].contents


# ── JSON export: section remapping ─────────────────────────────────────


class TestExportSectionRemapping:
    def test_section_id_zero_excluded(self):
        """Root section (id=0) should not appear in export."""
        paper = _minimal_paper()
        result = export_paper_to_json(paper)
        section_ids = [s["section_id"] for s in result["section"]]
        assert 0 not in section_ids

    def test_parent_section_id_zero_remapped_to_null(self):
        """parent_section_id=0 should become None in export."""
        paper = _minimal_paper()
        result = export_paper_to_json(paper)
        intro_section = result["section"][0]
        assert intro_section["parent_section_id"] is None

    def test_section_type_exported_as_string(self):
        paper = _minimal_paper()
        result = export_paper_to_json(paper)
        assert result["section"][0]["section_type"] == "intro"

    def test_appendix_section_type_exported(self):
        """The APPENDIX section type serializes as the "appendix" literal."""
        contents = _minimal_contents(
            sections=[
                PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
                PaperSection(
                    section_id=1,
                    header="Appendix A",
                    level=1,
                    parent_section_id=0,
                    section_type=CanonicalSection.APPENDIX,
                ),
            ],
        )
        result = export_paper_to_json(_minimal_paper(contents=contents))
        assert result["section"][0]["section_type"] == "appendix"

    def test_level_exported(self):
        """Section level is included in the exported JSON."""
        paper = _minimal_paper()
        result = export_paper_to_json(paper)
        assert "level" in result["section"][0]
        assert result["section"][0]["level"] == 1

    def test_text_section_id_zero_remapped_to_null(self):
        """Sentences with section_id=0 should export as section_id=null."""
        contents = _minimal_contents(
            sentences=[PaperSentence(text_id=1, text="Root text.", section_id=0, paragraph_id=1)],
        )
        paper = _minimal_paper(contents=contents)
        result = export_paper_to_json(paper)
        assert result["text"][0]["section_id"] is None


# ── JSON export: match serialization ───────────────────────────────────


class TestExportMatchSerialization:
    def test_bib_match_structure(self):
        ref = PaperReference(
            bib_id=1,
            title="Ref",
            first_page="1",
            volume="10",
            authors="Smith, J.",
            year=2020,
            container="Nature",
            match={
                MatchSource.CROSSREF: ExternalMatch(id="10.1234/CR", score=95.0, title="Ref"),
            },
        )
        meta = PaperMetadata(doi="10.1234/test", title="Test", references=[ref])
        paper = _minimal_paper(metadata=meta)
        result = export_paper_to_json(paper)

        # v10.1: matches are in top-level bib_match, not nested in bib
        assert "match" not in result["bib"][0]
        bib_match = result["bib_match"]
        assert len(bib_match) == 1
        # A DOI service id is lowercase like every exported DOI, and the
        # enrichment score (0-100) is published on the 0-1 scale.
        assert bib_match[0]["service_id"] == "10.1234/cr"
        assert bib_match[0]["score"] == 0.95
        assert bib_match[0]["service"] == "crossref"
        assert bib_match[0]["bib_id"] == 1

    def test_no_matches_empty_bib_match(self):
        ref = PaperReference(
            bib_id=1,
            title="Ref",
            first_page="1",
            volume="10",
            authors="Smith, J.",
            year=2020,
            container="Nature",
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        paper = _minimal_paper(metadata=meta)
        result = export_paper_to_json(paper)

        assert result["bib_match"] == []

    def test_multiple_match_sources(self):
        ref = PaperReference(
            bib_id=1,
            title="Ref",
            first_page=None,
            volume=None,
            authors="Doe, A.",
            year=2021,
            container="Science",
            match={
                MatchSource.CROSSREF: ExternalMatch(id="10.1/cr", score=0.9),
                MatchSource.OPENALEX: ExternalMatch(id="W123", score=0.85),
            },
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        paper = _minimal_paper(metadata=meta)
        result = export_paper_to_json(paper)

        bib_match = result["bib_match"]
        assert len(bib_match) == 2
        services = {m["service"] for m in bib_match}
        assert services == {"crossref", "openalex"}


class TestNerOnlyBibFields:
    """The five the decoder used to drop reach ``bib[]`` in schema 10.8."""

    def test_they_reach_the_export(self):
        ref = PaperReference(
            bib_id=1,
            title="Bert: pre-training of deep bidirectional transformers",
            first_page=None,
            volume=None,
            authors="Devlin J, Chang M-W",
            year=2018,
            container=None,
            arxiv="1810.04805",
            pmid="28919116",
            series="Lecture Notes in Computer Science",
            access_date="Accessed 12 March 2020",
            note="in Russian",
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        bib = export_paper_to_json(_minimal_paper(metadata=meta))["bib"][0]
        assert bib["arxiv"] == "1810.04805"
        assert bib["pmid"] == "28919116"
        assert bib["series"] == "Lecture Notes in Computer Science"
        assert bib["access_date"] == "Accessed 12 March 2020"
        assert bib["note"] == "in Russian"

    def test_they_are_null_when_the_parser_tags_none_of_them(self):
        # The shipped v4.5 checkpoint never emits one -- its corpus had no
        # examples -- so existing output keeps its shape with explicit nulls.
        ref = PaperReference(
            bib_id=1,
            title="A study of X",
            first_page=None,
            volume=None,
            authors="Smith, J.",
            year=None,
            container=None,
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        bib = export_paper_to_json(_minimal_paper(metadata=meta))["bib"][0]
        for field in ("arxiv", "pmid", "series", "access_date", "note"):
            assert bib[field] is None


class TestBibIsolatedFromMatches:
    """`bib[]` mirrors the printed reference verbatim. External matches
    (Crossref, OpenAlex) are never copied into `bib[]`, regardless of score —
    they live in `bib_match[]` only. Rationale: bibliographic search can
    return a confidently-matched but distinct publication (a book review of
    the cited book, a conference abstract sharing a journal article's title),
    and any backfill fabricates fields the PDF never claimed."""

    def test_high_score_match_does_not_fill_any_field(self):
        ref = PaperReference(
            bib_id=1,
            title="A study of X",
            first_page=None,
            volume=None,
            authors="Smith, J.",
            year=None,
            container=None,
            doi=None,
            issue=None,
            last_page=None,
            publisher=None,
            url=None,
            match={
                MatchSource.CROSSREF: ExternalMatch(
                    id="10.1234/cr",
                    score=100.0,
                    doi="10.1234/cr",
                    title="A Study of X",
                    year=2020,
                    container="Nature",
                    volume="10",
                    issue="2",
                    first_page="100",
                    last_page="115",
                    publisher="Nature Publishing",
                    url="https://doi.org/10.1234/cr",
                ),
            },
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        result = export_paper_to_json(_minimal_paper(metadata=meta))
        bib = result["bib"][0]
        # Every match-only field stays as it was in the printed reference.
        assert bib["doi"] is None
        assert bib["url"] is None
        assert bib["issue"] is None
        assert bib["year"] is None
        assert bib["container"] is None
        assert bib["volume"] is None
        assert bib["first_page"] is None
        assert bib["last_page"] is None
        assert bib["publisher"] is None
        # Raw title (already non-null) is preserved.
        assert bib["title"] == "A study of X"
        # The Crossref match is preserved in bib_match[] for traceability.
        assert len(result["bib_match"]) == 1
        assert result["bib_match"][0]["doi"] == "10.1234/cr"

    def test_book_reference_not_contaminated_by_journal_match(self):
        """Regression for the v12 corpus pattern: book entries (Cohen 1988,
        Kahneman 1973, etc.) printed with no journal/volume/pages were being
        backfilled with metadata from a Crossref match to an unrelated journal
        review of the book. `bib[]` must reflect only what the PDF prints."""
        ref = PaperReference(
            bib_id=1,
            bib_type="book",
            title="Statistical power analysis for the behavioral sciences",
            authors="Cohen, J.",
            year=1988,
            publisher="Erlbaum",
            container=None,
            volume=None,
            first_page=None,
            last_page=None,
            match={
                MatchSource.CROSSREF: ExternalMatch(
                    id="10.x/jasa-review",
                    score=100.0,
                    container="Journal of the American Statistical Association",
                    volume="84",
                    first_page="1096",
                    last_page="1097",
                ),
            },
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        result = export_paper_to_json(_minimal_paper(metadata=meta))
        bib = result["bib"][0]
        assert bib["container"] is None
        assert bib["volume"] is None
        assert bib["first_page"] is None
        assert bib["last_page"] is None
        # Match is still recorded for downstream consumers that want it.
        assert len(result["bib_match"]) == 1

    def test_existing_raw_values_preserved(self):
        ref = PaperReference(
            bib_id=1,
            title="A study of X",
            first_page="1",
            volume="10",
            authors="Smith, J. & Jones, K.",
            year=2020,
            container="Nature",
            doi="10.9999/RAW",
            match={
                MatchSource.CROSSREF: ExternalMatch(
                    id="10.1234/abc",
                    score=100.0,
                    doi="10.1234/abc",
                    title="Different Title",
                    container="Different Journal",
                ),
            },
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        result = export_paper_to_json(_minimal_paper(metadata=meta))
        bib = result["bib"][0]
        # Printed values stay, except that a DOI is lowercased like every DOI.
        assert bib["doi"] == "10.9999/raw"
        assert bib["title"] == "A study of X"
        assert bib["container"] == "Nature"
        assert bib["authors"] == "Smith, J. & Jones, K."


# ── JSON export: xref serialization ────────────────────────────────────


class TestExportXrefSerialization:
    def test_xref_type_routing(self):
        """Each xref exports its type and the export id of the row it names:
        the parser's table 2 and figure 3 are the paper's only ones, so they
        export as 1, and a footnote reference names the footnote row."""
        xrefs = [
            PaperXref(xref_id=1, xref_type="bib", contents="[1]", text_id=1),
            PaperXref(xref_id=2, xref_type="table", contents="Table 2", text_id=1),
            PaperXref(xref_id=3, xref_type="figure", contents="Figure 3", text_id=1),
            PaperXref(xref_id=4, xref_type="foot", contents="*", text_id=1),
        ]
        sentences = [
            PaperSentence(text_id=1, text="Hello.", section_id=1, paragraph_id=1),
            PaperSentence(text_id=4, text="* A note.", section_id=2, paragraph_id=2),
        ]
        sections = [
            *_minimal_contents().sections,
            PaperSection(
                section_id=2,
                header="Footnote 1",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.FOOTNOTE,
                synthetic_kind="footnote",
                footnote_label="*",
            ),
        ]
        contents = _minimal_contents(
            xrefs=xrefs,
            sentences=sentences,
            sections=sections,
            tables=[PaperTable(2, pd.DataFrame(), "", 1)],
            figures=[PaperFigure(3, 1, None, None)],
        )
        paper = _minimal_paper(contents=contents)
        result = export_paper_to_json(paper)

        assert [(x["xref_type"], x["target_id"]) for x in result["xref"]] == [
            ("bib", 1),
            ("table", 1),
            ("figure", 1),
            ("foot", 1),
        ]
        assert result["footnote"] == [{"footnote_id": 1, "label": "*", "text_id": 4}]

    def test_unresolved_targets_are_null(self):
        """A reference to a table, figure or footnote the export does not have
        names no row."""
        xrefs = [
            PaperXref(xref_id=2, xref_type="table", contents="Table 2", text_id=1),
            PaperXref(xref_id=3, xref_type="figure", contents="Figure 3", text_id=1),
            PaperXref(xref_id=4, xref_type="foot", contents="*", text_id=1),
        ]
        result = export_paper_to_json(_minimal_paper(contents=_minimal_contents(xrefs=xrefs)))
        assert [x["target_id"] for x in result["xref"]] == [None, None, None]


# ── JSON export: display math replacement ──────────────────────────────


class TestExportDisplayMath:
    def test_display_math_replaced_with_placeholder(self):
        sentences = [
            PaperSentence(
                text_id=1,
                text="E = mc^2",
                section_id=1,
                paragraph_id=1,
                is_display_formula=True,
            ),
        ]
        contents = _minimal_contents(sentences=sentences)
        paper = _minimal_paper(contents=contents)
        result = export_paper_to_json(paper)

        assert result["text"][0]["text"] == "[equation]"
        assert result["text"][0]["formatted"] == "E = mc^2"

    def test_normal_text_no_formatted(self):
        paper = _minimal_paper()
        result = export_paper_to_json(paper)
        assert result["text"][0]["formatted"] is None


# ── JSON export: url sanity guard ───────────────────────────────────────


class TestExportUrlSanity:
    """Links whose href fails a basic sanity parse are dropped at export —
    truncated host-continuation wraps like "https://blog" or
    "http://scikit-learn" that slipped past upstream URL-join heuristics."""

    def _paper_with_links(self, urls: list[str]) -> Paper:
        links = [PaperURLLink(url=u, section_id=1, paragraph_id=1, text_id=1) for u in urls]
        return _minimal_paper(contents=_minimal_contents(links=links))

    def test_drops_truncated_host_no_dot(self):
        paper = self._paper_with_links(["https://blog"])
        result = export_paper_to_json(paper)
        assert result["url"] == []

    def test_drops_truncated_host_hyphenated(self):
        paper = self._paper_with_links(["http://scikit-learn"])
        result = export_paper_to_json(paper)
        assert result["url"] == []

    def test_keeps_well_formed_https_url(self):
        paper = self._paper_with_links(["https://blog.openai.com/better-language-models/"])
        result = export_paper_to_json(paper)
        assert [u["href"] for u in result["url"]] == [
            "https://blog.openai.com/better-language-models/"
        ]

    def test_keeps_doi_org_link(self):
        paper = self._paper_with_links(["https://doi.org/10.1177/0956797614520714"])
        result = export_paper_to_json(paper)
        assert len(result["url"]) == 1

    def test_keeps_urls_with_trailing_slash_and_query(self):
        urls = [
            "https://osf.io/mwzuq/",
            "https://gluebenchmark.com/leaderboard?query=1",
        ]
        paper = self._paper_with_links(urls)
        result = export_paper_to_json(paper)
        assert [u["href"] for u in result["url"]] == urls

    def test_mixed_list_drops_only_malformed_entries(self):
        urls = ["https://blog", "https://openai.com/research", "http://scikit-learn"]
        paper = self._paper_with_links(urls)
        result = export_paper_to_json(paper)
        assert [u["href"] for u in result["url"]] == ["https://openai.com/research"]


# ── JSON export: top-level fields ──────────────────────────────────────


class TestExportTopLevel:
    def test_paper_id_from_file_stem(self):
        paper = _minimal_paper()
        result = export_paper_to_json(paper)
        assert result["paper_id"] == "test"

    def test_metadata_and_source_blocks(self):
        paper = _minimal_paper()
        result = export_paper_to_json(paper)
        assert result["schema_version"] == "12.1"
        assert result["metadata"]["title"] == "Test Paper"
        assert result["metadata"]["doi"] == "10.1234/test"
        # v11 split the input artifact's identity out of the paper's metadata;
        # v12 carries the whole SHA-256, null when the bytes were never seen.
        assert result["source"] == {
            "file_name": "test.pdf",
            "sha256": None,
            "input_format": "pdf",
        }
        assert "sha256" not in result["metadata"]
        assert "bibr_version" not in result["metadata"]

    def test_source_sha256_is_the_full_digest(self):
        paper = _minimal_paper()
        paper.input_file.sha256 = "ab" * 32
        result = export_paper_to_json(paper)
        assert result["source"]["sha256"] == "ab" * 32

    def test_xml_input_is_jats(self):
        # The vocabulary names the format: bibr reads XML only as JATS, and
        # ``tei`` is left for GROBID converters.
        paper = _minimal_paper(
            input_file=InputFile(
                path=Path("/tmp/test.xml"),
                file_hash="abc123",
                input_format=InputFormat(
                    file_extension=".xml",
                    detected_mime_type="application/xml",
                    file_type="XML",
                ),
            )
        )
        result = export_paper_to_json(paper)
        assert result["source"]["input_format"] == "jats"

    def test_input_format_lowercased(self):
        # The pipeline stores enum member names ("PDF", "DOCX", "XML"); the
        # schema contract is lowercase ("pdf", "docx", "jats", "unknown").
        paper = _minimal_paper(
            input_file=InputFile(
                path=Path("/tmp/test.pdf"),
                file_hash="abc123",
                input_format=InputFormat(
                    file_extension=".pdf",
                    detected_mime_type="application/pdf",
                    file_type="PDF",
                ),
            )
        )
        result = export_paper_to_json(paper)
        assert result["source"]["input_format"] == "pdf"

    def test_author_orcid_canonicalized(self):
        meta = PaperMetadata(
            doi="10.1/x",
            title="T",
            authors=[
                PaperAuthor(
                    author_id=1,
                    given="Jane",
                    family="Doe",
                    affiliation="MIT",
                    orcid="0000-0002-1825-0097",
                )
            ],
        )
        paper = _minimal_paper(metadata=meta)
        result = export_paper_to_json(paper)
        assert result["author"][0]["orcid"] == "https://orcid.org/0000-0002-1825-0097"

    def test_no_contents_raises(self):
        paper = _minimal_paper(contents=None)
        try:
            export_paper_to_json(paper)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


# ── Abstract export ───────────────────────────────────────────────────


class TestExportAbstract:
    """The export reads ``paper.metadata.abstract`` VERBATIM. The policy layer
    (ABSTRACT-section fallback, keyword recovery, commentary guard) runs in
    post-parse — see tests/pipeline/test_metadata_finalize.py — so by export
    time the metadata is final."""

    def _abstract_paper(self, *, metadata_abstract: str | None, abstract_sentences: list[str]):
        sentences = [
            PaperSentence(text_id=i, text=t, section_id=2, paragraph_id=1)
            for i, t in enumerate(abstract_sentences, start=1)
        ]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=2,
                header="Abstract",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.ABSTRACT,
            ),
        ]
        contents = PaperContents(
            sentences=sentences, sections=sections, tables=[], links=[], sections_text={}
        )
        meta = PaperMetadata(doi="10.1/abstract", title="T")
        meta.abstract = metadata_abstract or ""
        return _minimal_paper(metadata=meta, contents=contents)

    def test_metadata_abstract_exported_verbatim(self):
        paper = self._abstract_paper(
            metadata_abstract="The clean LLM abstract.",
            abstract_sentences=["Section text that must NOT be re-derived at export."],
        )
        result = export_paper_to_json(paper)
        assert result["metadata"]["abstract"] == "The clean LLM abstract."

    def test_no_export_side_section_fallback(self):
        """Export does NOT rebuild the abstract from section text — that policy
        moved to post-parse. Empty metadata.abstract exports as null even when
        an ABSTRACT section exists."""
        paper = self._abstract_paper(
            metadata_abstract=None,
            abstract_sentences=["Sentence one.", "Sentence two."],
        )
        result = export_paper_to_json(paper)
        assert result["metadata"]["abstract"] is None

    def test_blank_abstract_exports_null(self):
        paper = self._abstract_paper(metadata_abstract="   ", abstract_sentences=[])
        result = export_paper_to_json(paper)
        assert result["metadata"]["abstract"] is None


# ── Schema validation ─────────────────────────────────────────────────


class TestSchemaValidation:
    """Test that export output validates against the v10.1 Pydantic schema."""

    def test_minimal_export_validates(self):
        from bibr.export.json_export import validate_export

        paper = _minimal_paper()
        data = export_paper_to_json(paper)
        errors = validate_export(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_full_export_validates(self):
        """Export with all optional fields populated passes schema validation."""
        from bibr.export.json_export import validate_export

        contents = _minimal_contents(
            figures=[
                PaperFigure(figure_id=1, section_id=1, image_b64=None, caption=None, page_number=1)
            ],
            tables=[
                PaperTable(
                    table_id=1,
                    df=pd.DataFrame(),
                    tbl_html="<table></table>",
                    section_id=1,
                    page_number=1,
                )
            ],
            links=[
                PaperURLLink(
                    url="https://example.com",
                    section_id=1,
                    paragraph_id=1,
                    text_id=1,
                    link_text="link",
                )
            ],
            xrefs=[PaperXref(xref_id=1, xref_type="bib", contents="[1]", text_id=1)],
            equations=[PaperEquation(text_id=1, grp_id=1, lhs="x", comp="=", rhs="1")],
        )
        meta = PaperMetadata(
            doi="10.1234/test",
            title="Full Paper",
            authors=[PaperAuthor(author_id=1, given="J", family="D", affiliation="MIT")],
            references=[
                PaperReference(
                    bib_id=1,
                    title="Ref 1",
                    first_page="1",
                    volume="10",
                    authors=None,
                    year=2020,
                    container="J. Test",
                )
            ],
        )
        paper = _minimal_paper(contents=contents, metadata=meta)
        data = export_paper_to_json(paper)
        errors = validate_export(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_invalid_data_caught(self):
        """Deliberately invalid data produces validation errors."""
        from bibr.export.json_export import validate_export

        bad_data = {"paper_id": 123}  # wrong type + missing required keys
        errors = validate_export(bad_data)
        assert len(errors) > 0

    def test_export_validates_by_default(self):
        """export_paper_to_json runs validation and returns valid data."""
        paper = _minimal_paper()
        data = export_paper_to_json(paper)
        from bibr.export.json_export import PaperExport

        PaperExport.model_validate(data)

    def test_export_output_revalidates_cleanly(self):
        """The built dict round-trips through validate_export with no errors
        (validate_export remains the tool for checking dicts from elsewhere,
        e.g. consolidated output or stored JSON)."""
        from bibr.export.json_export import validate_export

        data = export_paper_to_json(_minimal_paper())
        assert validate_export(data) == []

    def test_optional_validation_and_diagnostics_fields_default_when_absent(self):
        """A payload missing the optional flags still validates, with defaults."""
        from bibr.export.json_export import PaperExport

        paper = _minimal_paper(metadata=PaperMetadata(doi="10.1/x", title="Untitled"))
        paper.extraction = _extraction_block()
        data = export_paper_to_json(paper)
        data["extraction"]["diagnostics"].pop("references_complete")
        data["extraction"]["validation"].pop("blocking")
        data["extraction"]["validation"].pop("promotable")
        for issue in data["extraction"]["validation"]["issues"]:
            issue.pop("origin_stage")
            issue.pop("evidence_ids")
            issue.pop("blocking")

        validated = PaperExport.model_validate(data)

        assert validated.extraction.diagnostics.references_complete is True
        validation = validated.extraction.validation
        assert validation is not None
        assert validation.blocking == 0
        assert validation.promotable is True
        assert validation.issues[0].origin_stage == "export"
        assert validation.issues[0].evidence_ids == []
        assert validation.issues[0].blocking is False


class TestSchemaIsContract:
    """The Pydantic models must reject keys the hand-built dicts don't declare
    (extra='forbid') and must declare everything the dicts emit — otherwise
    the schema and the export drift apart silently."""

    def test_undeclared_top_level_key_caught(self):
        from bibr.export.json_export import validate_export

        data = export_paper_to_json(_minimal_paper())
        data["_bogus"] = True
        assert validate_export(data), "undeclared top-level key must fail validation"

    def test_undeclared_text_key_caught(self):
        from bibr.export.json_export import validate_export

        data = export_paper_to_json(_minimal_paper())
        data["text"][0]["_bogus"] = 1
        assert validate_export(data), "undeclared text key must fail validation"

    def test_region_meta_is_declared_under_extraction_not_on_text_rows(self):
        """v12: the v4 layout features are processing payload
        (``extraction.text_regions``); text rows reject them."""
        from bibr.export.json_export import validate_export

        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        data = export_paper_to_json(paper)
        data["extraction"]["text_regions"] = [
            {
                "text_id": 1,
                "font_size": 9.5,
                "font_bold": True,
                "is_italic": False,
                "page_number": 1,
                "bbox": [10.0, 20.0, 110.0, 40.0],
                "region_type": "text",
            }
        ]
        assert validate_export(data) == []
        data["text"][0]["_bbox_2d"] = [10.0, 20.0, 110.0, 40.0]
        assert validate_export(data), "text rows must not accept layout features"

    def test_regions_payload_is_declared(self):
        """include_regions output (extraction.regions) validates against the schema."""
        from bibr.export.json_export import validate_export

        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        data = export_paper_to_json(paper)
        data["extraction"]["regions"] = [
            {
                "page": 1,
                "index": 0,
                "label": "text",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "font_size": 9.5,
                "font_weight": 400,
                "font_bold": False,
                "section_id": 1,
                "content": "Hello.",
                "char_density": 0.5,
                "estimated_line_height": 1.1,
            }
        ]
        errors = validate_export(data)
        assert errors == [], f"Validation errors: {errors}"


class TestExportBuiltThroughModels:
    """The export dict is CONSTRUCTED through the PaperExport models, so a
    schema violation fails fatally at build time instead of warn-and-ship."""

    def test_invalid_data_raises_at_export(self):
        import pytest
        from pydantic import ValidationError

        paper = _minimal_paper()
        paper.contents.xrefs.append(
            PaperXref(xref_id=1, xref_type="bogus", contents="[1]", text_id=1)
        )
        with pytest.raises(ValidationError):
            export_paper_to_json(paper)

    def test_unconsolidated_bib_has_no_consolidated_fields_key(self):
        # ``consolidated_fields`` is injected by consolidate_bibs AFTER export;
        # unconsolidated output must not carry the key at all (not even null).
        ref = PaperReference(
            bib_id=1,
            title="T",
            first_page=None,
            volume=None,
            authors=None,
            year=2020,
            container=None,
        )
        meta = PaperMetadata(doi="10.1/x", title="T", references=[ref])
        out = export_paper_to_json(_minimal_paper(metadata=meta))
        assert "consolidated_fields" not in out["bib"][0]

    def test_output_is_plain_json_types(self):
        # model_dump must yield JSON-primitive values (no enums, no models).
        import json

        out = export_paper_to_json(_minimal_paper())
        json.dumps(out)  # raises TypeError on any non-JSON value


# ── Typed BibMatchExport ───────────────────────────────────────────────


class TestTypedBibMatchExport:
    def test_bib_match_validates_fields(self):
        """BibMatchExport validates fields correctly."""
        from bibr.export.json_export import BibMatchExport

        m = BibMatchExport(
            bib_id=1,
            service="crossref",
            service_id="10.1/cr",
            score=0.95,
            title="A Paper",
            author=[{"given": "J.", "family": "Smith"}],
            year=2020,
        )
        assert m.service_id == "10.1/cr"
        assert m.score == 0.95

    def test_bib_match_export_rejects_bad_score_type(self):
        """BibMatchExport rejects non-numeric score."""
        import pytest
        from pydantic import ValidationError

        from bibr.export.json_export import BibMatchExport

        with pytest.raises(ValidationError):
            BibMatchExport(bib_id=1, service="crossref", score="not_a_number", title="T")

    def test_full_export_with_typed_match_validates(self):
        """Full paper export with crossref match passes schema validation."""
        from bibr.export.json_export import validate_export

        ref = PaperReference(
            bib_id=1,
            title="Ref",
            first_page="1",
            volume="10",
            authors="Smith, J.",
            year=2020,
            container="Nature",
            match={
                MatchSource.CROSSREF: ExternalMatch(
                    id="10.1/cr",
                    score=0.95,
                    title="Ref",
                    authors=[BibAuthor(given="J.", family="Smith")],
                    year=2020,
                ),
            },
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        paper = _minimal_paper(metadata=meta)
        data = export_paper_to_json(paper)
        errors = validate_export(data)
        assert errors == [], f"Validation errors: {errors}"


class TestExportInPressYearHandling:
    """``is_in_press=True`` is a flag, not a year-replacement: a real numeric
    year must survive into the export. Only the legacy ``year=0`` sentinel
    collapses to ``None``."""

    def test_in_press_with_real_year_preserves_year(self):
        ref = PaperReference(
            bib_id=1,
            title="Advance",
            first_page="",
            volume="",
            authors="Smith, J.",
            year=2024,
            container="Nature",
            is_in_press=True,
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        paper = _minimal_paper(metadata=meta)
        result = export_paper_to_json(paper)
        assert result["bib"][0]["year"] == 2024
        assert result["bib"][0]["is_in_press"] is True

    def test_year_zero_sentinel_becomes_none(self):
        ref = PaperReference(
            bib_id=1,
            title="Pending",
            first_page="",
            volume="",
            authors="",
            year=0,
            container="",
        )
        meta = PaperMetadata(doi="10.1/test", title="Test", references=[ref])
        paper = _minimal_paper(metadata=meta)
        result = export_paper_to_json(paper)
        assert result["bib"][0]["year"] is None
        assert result["bib"][0]["is_in_press"] is True


class TestEngineExportsRejectExtraKeys:
    def test_extra_key_raises(self):
        import pytest
        from pydantic import ValidationError

        from bibr.export.json_export import LlmEngineExport, OcrEngineExport

        with pytest.raises(ValidationError):
            OcrEngineExport(backend="x", surprise="y")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            LlmEngineExport(provider="x", surprise="y")  # type: ignore[call-arg]


# ── extraction.warnings ────────────────────────────────────────────────


class TestProcessingWarnings:
    def test_default_empty_list(self):
        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        result = export_paper_to_json(paper)
        assert result["extraction"]["warnings"] == []
        # v11 moved warnings under ``extraction``; the root key is gone and
        # ``info`` stays scalar-only.
        assert "processing_warnings" not in result
        assert "processing_warnings" not in result["metadata"]

    def test_warnings_surfaced_under_extraction(self):
        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        paper.processing_warnings = [
            ProcessingWarning(WarningCode.OCR_PAGE_FAILED, "page 4: TimeoutError: timeout"),
            ProcessingWarning(WarningCode.CROSSREF_ENRICHMENT_TIMEOUT, "timed out after 120s"),
        ]
        result = export_paper_to_json(paper)
        assert result["extraction"]["warnings"] == [
            {"code": "OCR_PAGE_FAILED", "message": "page 4: TimeoutError: timeout"},
            {"code": "CROSSREF_ENRICHMENT_TIMEOUT", "message": "timed out after 120s"},
        ]

    def test_warnings_round_trip_through_schema(self):
        from bibr.export.json_export import validate_export

        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        paper.processing_warnings = [ProcessingWarning(WarningCode.LOW_TEXT_QUALITY, "0.12")]
        result = export_paper_to_json(paper)
        assert validate_export(result) == []

    def test_block_and_paper_warnings_are_unioned_in_order(self):
        """The stage snapshots the warnings into the block as objects; the
        Paper may gain more afterwards. Both reach the export once each."""
        seen = ProcessingWarning(WarningCode.REF_SEG_CRF_FALLBACK, "LLM seg tier disabled")
        later = ProcessingWarning(WarningCode.CONSOLIDATE_WITHOUT_ENRICHMENT, "nothing to merge")
        paper = _minimal_paper()
        paper.extraction = _extraction_block(warnings=[seen.to_dict()])
        paper.processing_warnings = [seen, later]
        result = export_paper_to_json(paper)
        assert result["extraction"]["warnings"] == [seen.to_dict(), later.to_dict()]

    def test_prose_and_malformed_warnings_fail_the_export(self):
        import pytest

        for bad in ("OCR failed for page index 3", {"code": "lowercase", "message": "m"}):
            paper = _minimal_paper()
            paper.extraction = _extraction_block(warnings=[bad])
            with pytest.raises(ValueError):
                export_paper_to_json(paper)


# ── output validation gate wiring ──────────────────────────────────────


class TestValidationGateWiring:
    def test_clean_paper_has_empty_validation_block(self):
        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        result = export_paper_to_json(paper)
        assert result["extraction"]["validation"] == {
            "errors": 0,
            "warnings": 0,
            "blocking": 0,
            "promotable": True,
            "issues": [],
        }
        assert result["extraction"]["warnings"] == []

    def test_reference_failure_exports_blocking_issue_and_incomplete_extraction(self):
        metadata = PaperMetadata(
            doi="10.1234/core",
            title="Durable Core",
            references_incomplete=True,
        )
        paper = _minimal_paper(metadata=metadata)
        paper.extraction = _extraction_block(producer={"name": "bibr", "version": "9.9.9-test"})

        result = export_paper_to_json(paper)

        assert result["metadata"]["title"] == "Durable Core"
        assert result["metadata"]["doi"] == "10.1234/core"
        assert result["bib"] == []
        assert result["extraction"]["diagnostics"]["references_complete"] is False
        assert result["extraction"]["validation"]["errors"] == 1
        assert result["extraction"]["validation"]["blocking"] == 1
        assert result["extraction"]["validation"]["promotable"] is False
        issue = next(
            issue
            for issue in result["extraction"]["validation"]["issues"]
            if issue["code"] == "VAL_REFERENCES_INCOMPLETE"
        )
        assert issue["severity"] == "error"
        assert issue["origin_stage"] == "extract"
        assert issue["blocking"] is True

    def test_stage_and_replay_issues_merge_deterministically(self, monkeypatch):
        from bibr.validation import IssueSeverity, ValidationIssue

        duplicate = ValidationIssue(
            code="VAL_DUPLICATE",
            severity=IssueSeverity.WARNING,
            message="same finding",
            origin_stage="extract",
            evidence_ids=("evidence-1",),
        )
        replay_only = ValidationIssue(
            code="VAL_REPLAY",
            severity=IssueSeverity.WARNING,
            message="replayed finding",
        )
        paper = _minimal_paper()
        paper.validation_issues = [duplicate]
        monkeypatch.setattr(
            "bibr.export.validation.validate_export",
            lambda _payload: [replay_only, duplicate],
        )

        result = export_paper_to_json(paper)

        assert [issue["code"] for issue in result["extraction"]["validation"]["issues"]] == [
            "VAL_DUPLICATE",
            "VAL_REPLAY",
        ]
        assert result["extraction"]["validation"]["warnings"] == 2

    def test_source_abstract_warning_remains_one_canonical_issue_through_export(self):
        from bibr.extract.front_matter import (
            FrontMatterBlock,
            FrontMatterCandidate,
            FrontMatterResolution,
        )
        from bibr.pipeline.stages.post_parse import _finalize_abstract_and_keywords

        source = "Grounded source abstract."
        body = "Body sentence outside the selected block."
        abstract = "Invented abstract. " + ("A" * 2500) + " " + body
        sections = [
            PaperSection(0, "Root", 0, None),
            PaperSection(1, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
            PaperSection(2, "Abstract", 1, 0, CanonicalSection.ABSTRACT),
        ]
        contents = _minimal_contents(
            sections=sections,
            sentences=[
                PaperSentence(1, source, 2, 1, page_number=1),
                PaperSentence(2, body, 1, 2, page_number=2),
            ],
            sections_text={1: body, 2: source},
        )
        candidate = FrontMatterCandidate(
            candidate_id="abstract",
            source_kind="paragraph",
            reading_order=1,
            page=1,
            bbox=None,
            region_label="abstract",
            font_size=None,
            font_bold=None,
            section_id=2,
            text_ids=(1,),
            paragraph_id=1,
            raw_text=source,
            normalized_text=source.casefold(),
            roles=frozenset({"abstract"}),
        )
        resolution = FrontMatterResolution(
            candidates=(candidate,),
            blocks=(
                FrontMatterBlock(
                    block_id="selected",
                    candidate_ids=(candidate.candidate_id,),
                    title_candidate_ids=(),
                ),
            ),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=frozenset({1}),
            allowed_section_ids=frozenset({2}),
        )
        metadata = PaperMetadata(doi="10.1234/test", title="Test Paper", abstract=abstract)
        source_issues = []
        _finalize_abstract_and_keywords(
            contents,
            metadata,
            resolution=resolution,
            validation_issue_sink=source_issues,
        )
        paper = _minimal_paper(contents=contents, metadata=metadata)
        paper.validation_issues = source_issues

        result = export_paper_to_json(paper)

        suspect = [
            issue
            for issue in result["extraction"]["validation"]["issues"]
            if issue["code"] == "VAL_ABSTRACT_SUSPECT"
        ]
        assert len(suspect) == 1
        assert suspect[0]["message"] == (
            "abstract suspicion: ungrounded, cross_boundary, length_gt_2500, "
            "non_reference_share_gt_20pct"
        )
        assert suspect[0]["origin_stage"] == "extract"
        assert suspect[0]["count"] == 1
        assert len(suspect[0]["evidence_ids"]) <= 20
        assert result["metadata"]["abstract"] == abstract

    def test_defect_surfaces_only_in_the_structured_block(self):
        """v11 stopped mirroring gate findings into prose warnings: the finding
        carries structure (code, severity, evidence) a string throws away, and
        two shapes of one finding made every consumer reconcile them."""
        paper = _minimal_paper(metadata=PaperMetadata(doi="10.1/x", title="Untitled"))
        paper.extraction = _extraction_block()
        retried = ProcessingWarning(WarningCode.OCR_REGION_FAILED, "ocr retried page 3")
        paper.processing_warnings = [retried]

        result = export_paper_to_json(paper)

        block = result["extraction"]["validation"]
        assert block["warnings"] >= 1
        codes = {i["code"] for i in block["issues"]}
        assert "VAL_TITLE_GENERIC" in codes
        assert result["extraction"]["warnings"] == [retried.to_dict()]

    def test_validate_false_skips_gate(self):
        paper = _minimal_paper(metadata=PaperMetadata(doi="10.1/x", title="Untitled"))
        paper.extraction = _extraction_block()
        result = export_paper_to_json(paper, validate=False)
        assert "validation" not in result
        assert result["extraction"]["warnings"] == []

    def test_gate_exception_does_not_break_export(self, monkeypatch):
        def _boom(_payload):
            raise RuntimeError("gate exploded")

        monkeypatch.setattr("bibr.export.validation.validate_export", _boom)
        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        result = export_paper_to_json(paper)
        # export still produced a dict, with the failure surfaced as VAL_INTERNAL
        assert result["extraction"]["validation"]["warnings"] >= 1
        assert "VAL_INTERNAL" in {i["code"] for i in result["extraction"]["validation"]["issues"]}
        assert result["extraction"]["warnings"] == []


# ── metadata is scalar-only (consumed by R via as.data.frame) ──────────


class TestMetadataIsScalarOnly:
    """R consumers (metacheck's .read_bibr) call ``as.data.frame(metadata)``,
    which errors when its fields are nested objects or non-scalar lists.
    Pipeline telemetry (engines, warnings) lives under ``extraction`` instead."""

    def test_engines_are_under_extraction_not_metadata(self):
        paper = _minimal_paper()
        paper.extraction = _extraction_block(
            ocr={"backend": "glm-http", "model": None, "profile": "glm"},
            llm={"provider": "google", "model": "gemini", "backend": "cloud"},
        )
        result = export_paper_to_json(paper)
        assert "ocr_config" not in result["metadata"]
        assert "ocr_config" not in result
        assert result["extraction"]["ocr"]["backend"] == "glm-http"
        assert result["extraction"]["llm"]["provider"] == "google"

    def test_all_metadata_values_are_scalar_or_flat_list_of_strings(self):
        paper = _minimal_paper()
        paper.extraction = _extraction_block(
            ocr={"backend": "glm-http", "model": None, "profile": "glm"},
        )
        paper.processing_warnings = [
            ProcessingWarning(WarningCode.OCR_TABLE_DROPPED, "w1"),
            ProcessingWarning(WarningCode.OCR_CONTROL_CHARS, "w2"),
        ]
        metadata = export_paper_to_json(paper)["metadata"]
        for k, v in metadata.items():
            if k == "keywords":
                assert isinstance(v, list)
                assert all(isinstance(s, str) for s in v)
            else:
                assert v is None or isinstance(v, (str, int, float, bool)), (
                    f"info[{k!r}] = {v!r} is not scalar"
                )


# ── _regions toggle ────────────────────────────────────────────────────


class TestIncludeRegionsToggle:
    def test_regions_omitted_by_default(self):
        from bibr.paper_contents import RegionSummary

        paper = _minimal_paper()
        paper.contents.region_summaries = [
            RegionSummary(
                page=1,
                index=0,
                label="header",
                bbox=(0, 0, 10, 10),
                font_size=12.0,
                font_weight=400,
                font_bold=False,
                section_id=None,
                content="hello",
                bbox_height=10.0,
                bbox_width=10.0,
                char_density=0.5,
                estimated_line_height=12.0,
            )
        ]
        paper.extraction = _extraction_block()
        result = export_paper_to_json(paper)
        assert "_regions" not in result
        assert "regions" not in result["extraction"]

    def test_regions_included_when_opted_in(self):
        from bibr.paper_contents import RegionSummary

        paper = _minimal_paper()
        paper.contents.region_summaries = [
            RegionSummary(
                page=1,
                index=0,
                label="header",
                bbox=(0, 0, 10, 10),
                font_size=12.0,
                font_weight=400,
                font_bold=False,
                section_id=None,
                content="hello",
                bbox_height=10.0,
                bbox_width=10.0,
                char_density=0.5,
                estimated_line_height=12.0,
            )
        ]
        paper.extraction = _extraction_block()
        result = export_paper_to_json(paper, include_regions=True)
        assert "_regions" not in result
        assert result["extraction"]["regions"][0]["content"] == "hello"

    def test_raw_ocr_content_is_only_in_opted_in_regions(self):
        from bibr.paper_contents import RegionSummary

        raw_content = "raw diagnostic " * 40
        paper = _minimal_paper()
        paper.contents.region_summaries = [
            RegionSummary(
                page=1,
                index=0,
                label="text",
                bbox=(0, 0, 10, 10),
                content="canonical diagnostic",
                raw_ocr_content=raw_content,
            )
        ]

        paper.extraction = _extraction_block()
        normal = export_paper_to_json(paper)
        opted_in = export_paper_to_json(paper, include_regions=True)

        assert "regions" not in normal["extraction"]
        assert opted_in["extraction"]["regions"][0]["content"] == "canonical diagnostic"
        assert opted_in["extraction"]["regions"][0]["raw_ocr_content"] == raw_content


# ── v4 training region metadata on text blocks ─────────────────────────


class TestV4RegionMetadataInExtraction:
    """The v4 training layout features are opt-in (include_region_meta) and,
    since v12, ride ``extraction.text_regions`` keyed by ``text_id``: they are
    debug/training payload, not paper content."""

    _REGION_META = {
        "font_size": 10.5,
        "font_bold": True,
        "is_italic": False,
        "bbox": [100.0, 50.0, 900.0, 250.0],  # 0..1000 layout space
        "region_type": "text",
        "region_page": 2,
        "region_index": 3,
    }

    def _paper(self, region_meta):
        sent = PaperSentence(
            text_id=1,
            text="Some text.",
            section_id=1,
            paragraph_id=1,
            page_number=2,
            region_meta=region_meta,
        )
        contents = _minimal_contents(sentences=[sent])
        contents.page_sizes = {2: (612.0, 792.0)}  # US Letter, points
        paper = _minimal_paper(contents=contents)
        paper.extraction = _extraction_block()
        return paper

    def test_region_meta_omitted_by_default(self):
        result = export_paper_to_json(self._paper(dict(self._REGION_META)))
        assert "text_regions" not in result["extraction"]
        assert not [k for k in result["text"][0] if k.startswith("_")]

    def test_region_meta_rows_present_when_opted_in(self):
        """The region's 0..1000 layout box is exported in points from the
        top-left of its page, inside the page size ``extraction.pages`` gives."""
        result = export_paper_to_json(
            self._paper(dict(self._REGION_META)), include_region_meta=True
        )
        (row,) = result["extraction"]["text_regions"]
        assert row == {
            "text_id": 1,
            "font_size": 10.5,
            "font_bold": True,
            "is_italic": False,
            "page_number": 2,
            "bbox": [61.2, 39.6, 550.8, 198.0],
            "region_type": "text",
        }
        (page,) = result["extraction"]["pages"]
        assert page == {"page_number": 2, "width": 612.0, "height": 792.0}
        x1, y1, x2, y2 = row["bbox"]
        assert 0 <= x1 <= x2 <= page["width"]
        assert 0 <= y1 <= y2 <= page["height"]
        assert not [k for k in result["text"][0] if k.startswith("_")]

    def test_no_row_for_a_sentence_without_region_meta(self):
        """Opted in but region_meta is None (DOCX, pre-v4): no row at all."""
        result = export_paper_to_json(self._paper(None), include_region_meta=True)
        assert "text_regions" not in result["extraction"]

    def test_reference_region_type_preserved(self):
        meta = {**self._REGION_META, "region_type": "reference_content"}
        result = export_paper_to_json(self._paper(meta), include_region_meta=True)
        assert result["extraction"]["text_regions"][0]["region_type"] == "reference_content"


# The commentary abstract-fabrication guard and the keyword-recovery policy
# moved to post-parse; their cases live in tests/pipeline/test_metadata_finalize.py.


# ── extraction provenance block (v10.3) ────────────────────────────────


class TestExtractionProvenance:
    """The top-level ``extraction`` block records the producing engines, bibr
    version, the resolved reference strategies, seg-fallback, enrichment flags,
    LLM usage, and stage timings."""

    def _ctx(self, *, crossref=None, consolidate=None, timings=None, no_llm=False, settings=None):
        import types

        from bibr.config import snapshot_settings
        from bibr.ocr.profiles import OcrRuntimeIdentity
        from bibr.pipeline.context import RunConfig

        scratch = {
            "ocr_runtime_identity": OcrRuntimeIdentity(
                backend="glm-http",
                model="glm-ocr",
                profile="glm",
                normalizer_version="glm-canonical-v1",
            )
        }
        if timings is not None:
            scratch["stage_timings"] = timings
        return types.SimpleNamespace(
            config=RunConfig(crossref=crossref, consolidate=consolidate, no_llm=no_llm),
            settings=settings if settings is not None else snapshot_settings(),
            scratch=scratch,
        )

    def test_schema_version_is_at_the_root_and_package_version_under_extraction(self):
        # v11 put the schema version at the root (its presence is the reader's
        # dispatch signal); the producing *package* version is only
        # ``extraction.producer.version`` (``extraction.bibr_version`` before 12).
        paper = _minimal_paper()
        paper.extraction = _extraction_block(producer={"name": "bibr", "version": "9.9.9-test"})
        result = export_paper_to_json(paper)
        assert result["schema_version"] == "12.1"
        assert result["extraction"]["producer"]["version"] == "9.9.9-test"
        assert "schema_version" not in result["metadata"]
        assert "bibr_version" not in result["metadata"]

    def test_minimal_extraction_when_unset_and_validates(self):
        """Outside the pipeline the exporter still emits ``extraction``: the
        producer and export time, no settings (v12)."""
        from bibr.export.json_export import validate_export

        result = export_paper_to_json(_minimal_paper())
        assert set(result["extraction"]) >= {"producer", "completed_at", "validation"}
        assert "settings" not in result["extraction"]
        assert validate_export(result) == []

    def test_export_passes_through_paper_extraction(self):
        from bibr.export.json_export import validate_export

        paper = _minimal_paper()
        paper.extraction = _extraction_block(
            producer={"name": "bibr", "version": "9.9.9-test"},
            settings={
                "ref_seg": "llm",
                "ref_parse": "ner",
                "crossref_enrich": False,
                "consolidate": "off",
            },
            diagnostics={"ref_seg_fallback_used": True},
            timings={"stages": {"extract": 1.0}, "total_seconds": 1.0},
        )
        result = export_paper_to_json(paper)
        assert result["extraction"]["producer"]["version"] == "9.9.9-test"
        assert result["extraction"]["settings"]["ref_parse"] == "ner"
        assert result["extraction"]["diagnostics"]["ref_seg_fallback_used"] is True
        assert result["extraction"]["timings"]["total_seconds"] == 1.0
        assert validate_export(result) == []

    def test_completed_at_is_utc_iso8601_to_the_second(self):
        import datetime as dt
        import re

        from bibr.pipeline.stages.export import _build_extraction

        ext = _build_extraction(self._ctx(), _minimal_paper())

        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ext["completed_at"])
        parsed = dt.datetime.fromisoformat(ext["completed_at"].replace("Z", "+00:00"))
        assert parsed.tzinfo == dt.UTC

    def test_build_extraction_carries_text_quality_into_diagnostics(self):
        # v11 home of the parse-quality score: it is telemetry, not metadata.
        from bibr.pipeline.stages.export import _build_extraction

        paper = _minimal_paper()
        paper.text_quality = 0.42
        assert _build_extraction(self._ctx(), paper)["diagnostics"]["text_quality"] == 0.42
        assert _build_extraction(self._ctx(), _minimal_paper())["diagnostics"]["text_quality"] is (
            None
        )

    def test_build_extraction_reports_both_engines(self):
        from bibr.pipeline.stages.export import _build_extraction

        ctx = self._ctx()
        ext = _build_extraction(ctx, _minimal_paper())

        assert ext["ocr"] == {"backend": "glm-http", "model": "glm-ocr", "profile": "glm"}
        assert ext["llm"]["provider"] == ctx.settings.llm.provider
        assert ext["llm"]["model"] == ctx.settings.llm.model

    def test_build_extraction_ocr_is_null_on_the_native_parse_path(self):
        """DOCX/XML/HTML/EPUB are parsed natively and never reach an OCR engine,
        so the block must not report the *configured* backend as if it had
        produced this output. ``resolve_ocr_runtime_identity`` cannot know this
        — it is pure config resolution — so the input format is the signal."""
        from bibr.pipeline.stages.export import _build_extraction

        native = (
            ("docx", ".docx"),
            ("xml", ".xml"),
            ("html", ".html"),
            ("htm", ".htm"),
            ("epub", ".epub"),
        )
        for file_type, extension in native:
            paper = _minimal_paper(
                input_file=InputFile(
                    path=Path(f"/tmp/paper{extension}"),
                    file_hash="abc123",
                    input_format=InputFormat(
                        file_extension=extension,
                        detected_mime_type="application/octet-stream",
                        # The pipeline stores the enum member name upper-cased.
                        file_type=file_type.upper(),
                    ),
                )
            )

            ext = _build_extraction(self._ctx(), paper)

            assert ext["ocr"] is None, file_type
            # ``null``, not omitted — the run happened, without that engine.
            paper.extraction = ext
            assert export_paper_to_json(paper)["extraction"]["ocr"] is None

    def test_build_extraction_reports_ocr_for_a_pdf(self):
        from bibr.pipeline.stages.export import _build_extraction

        ext = _build_extraction(self._ctx(), _minimal_paper())
        assert ext["ocr"]["backend"] == "glm-http"

    def test_build_extraction_llm_is_null_when_the_run_had_none(self):
        """``null`` (not omitted): the run happened, deliberately without an LLM."""
        from bibr.pipeline.stages.export import _build_extraction

        ext = _build_extraction(self._ctx(no_llm=True), _minimal_paper())

        assert ext["llm"] is None
        assert ext["ocr"] is not None

    def test_build_extraction_aggregates_usage_from_the_label_triples(self):
        from bibr.pipeline.stages.export import _build_extraction

        paper = _minimal_paper()
        paper.llm_usage_labels = {
            ("extract_authors", "google", "gemini-x"): {
                "calls": 2,
                "input_tokens": 100,
                "cached_input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 120,
            },
            ("parse_refs", "openai", "gpt-x"): {
                "calls": 1,
                "input_tokens": 50,
                "cached_input_tokens": 0,
                "output_tokens": 5,
                "total_tokens": 55,
            },
        }

        usage = _build_extraction(self._ctx(), paper)["usage"]

        assert usage["totals"] == {
            "calls": 3,
            "input_tokens": 150,
            "cached_input_tokens": 10,
            "output_tokens": 25,
            "total_tokens": 175,
        }
        assert [(row["label"], row["provider"], row["model"]) for row in usage["breakdown"]] == [
            ("extract_authors", "google", "gemini-x"),
            ("parse_refs", "openai", "gpt-x"),
        ]

    def test_build_extraction_usage_is_none_when_no_llm_ran(self):
        from bibr.pipeline.stages.export import _build_extraction

        assert _build_extraction(self._ctx(), _minimal_paper())["usage"] is None

    def test_build_extraction_defaults(self):
        import bibr
        from bibr.config import Settings
        from bibr.pipeline.stages.export import _build_extraction

        ctx = self._ctx(timings={"validate": 0.10, "extract": 1.2345})
        ext = _build_extraction(ctx, _minimal_paper())

        assert ext["producer"] == {"name": "bibr", "version": bibr.__version__, "build_sha": None}
        assert ext["settings"]["ref_seg"] == "llm"
        assert ext["settings"]["ref_parse"] == "llm"
        assert ext["diagnostics"]["references_complete"] is True
        assert ext["diagnostics"]["ref_seg_fallback_used"] is False
        assert ext["settings"]["crossref_enrich"] is bool(Settings.crossref.enrich)
        assert ext["settings"]["consolidate"] == Settings.crossref.consolidate
        assert ext["timings"]["stages"]["extract"] == 1.234
        assert ext["timings"]["total_seconds"] == round(0.10 + 1.2345, 3)

    def test_build_extraction_includes_exact_build_sha(self):
        from bibr.pipeline.stages.export import _build_extraction

        ctx = self._ctx()
        ctx.settings.BIBR_BUILD_SHA = "a" * 40

        ext = _build_extraction(ctx, _minimal_paper())

        assert ext["producer"]["build_sha"] == "a" * 40

    def test_build_extraction_detects_seg_fallback(self):
        from bibr.pipeline.stages.export import _build_extraction

        paper = _minimal_paper()
        paper.processing_warnings = [
            ProcessingWarning(WarningCode.REF_SEG_CRF_FALLBACK, "produced 0 spans")
        ]
        ext = _build_extraction(self._ctx(), paper)
        assert ext["diagnostics"]["ref_seg_fallback_used"] is True
        assert ext["warnings"] == [{"code": "REF_SEG_CRF_FALLBACK", "message": "produced 0 spans"}]

    def test_build_extraction_detects_geom_cascade(self):
        # A geom->LLM cascade is a fall-back from the configured (geom) segmenter
        # and must flip ref_seg_fallback_used, even though it has a distinct code.
        from bibr.pipeline.stages.export import _build_extraction

        paper = _minimal_paper()
        paper.processing_warnings = [
            ProcessingWarning(
                WarningCode.REF_SEG_GEOM_CASCADE, "geom low-confidence (0.947 < 0.95) or empty"
            )
        ]
        ext = _build_extraction(self._ctx(), paper)
        assert ext["diagnostics"]["ref_seg_fallback_used"] is True

    def test_build_extraction_detects_a_segmenter_error(self):
        # A region or CRF segmenter that raised did not produce the final
        # segmentation either (both were filed under the CRF-fallback prefix).
        from bibr.pipeline.stages.export import _build_extraction

        for code in (WarningCode.REF_SEG_REGION_ERROR, WarningCode.REF_SEG_CRF_ERROR):
            paper = _minimal_paper()
            paper.processing_warnings = [ProcessingWarning(code, "error: RuntimeError('x')")]
            ext = _build_extraction(self._ctx(), paper)
            assert ext["diagnostics"]["ref_seg_fallback_used"] is True, code

    def test_build_extraction_merge_split_is_not_fallback(self):
        # Merge-split is a correction on top of the segmenter, not a fall-back.
        from bibr.pipeline.stages.export import _build_extraction

        paper = _minimal_paper()
        paper.processing_warnings = [
            ProcessingWarning(WarningCode.REF_SEG_MERGE_SPLIT, "split into 1 more segment(s)")
        ]
        ext = _build_extraction(self._ctx(), paper)
        assert ext["diagnostics"]["ref_seg_fallback_used"] is False

    def test_build_extraction_effective_consolidate_and_crossref_off(self):
        from bibr.pipeline.stages.export import _build_extraction

        ctx = self._ctx(crossref=False, consolidate="replace")
        ext = _build_extraction(ctx, _minimal_paper())
        assert ext["settings"]["crossref_enrich"] is False
        assert ext["settings"]["consolidate"] == "replace"

    def test_build_extraction_crossref_enrich_reflects_effective_per_run_value(self):
        """``extraction.crossref_enrich`` is the resolved tri-state, not the raw setting."""
        from bibr.config import GlobalSettings
        from bibr.pipeline.stages.export import _build_extraction

        off = GlobalSettings()
        off.crossref.enrich = False
        on = GlobalSettings()
        on.crossref.enrich = True

        # None follows the setting either way.
        assert _build_extraction(self._ctx(settings=off), _minimal_paper())["settings"][
            "crossref_enrich"
        ] is (False)
        assert _build_extraction(self._ctx(settings=on), _minimal_paper())["settings"][
            "crossref_enrich"
        ] is (True)
        # A per-run override wins over the setting.
        forced_on = _build_extraction(self._ctx(crossref=True, settings=off), _minimal_paper())
        assert forced_on["settings"]["crossref_enrich"] is True
        forced_off = _build_extraction(self._ctx(crossref=False, settings=on), _minimal_paper())
        assert forced_off["settings"]["crossref_enrich"] is False

    def test_build_extraction_total_excludes_overlapped_prefetch_timing(self):
        """The enrichment prefetch overlaps the extract stage: reported, never summed."""
        from bibr.pipeline.stages.export import _build_extraction

        ctx = self._ctx(timings={"extract": 2.0, "enrich_prefetch": 1.5, "enrich": 0.5})
        ext = _build_extraction(ctx, _minimal_paper())

        assert ext["timings"]["stages"]["enrich_prefetch"] == 1.5
        assert ext["timings"]["total_seconds"] == 2.5

    def test_build_extraction_no_timings(self):
        """Absent, not a zeroed row: ``{"stages": null, "total_seconds": null}``
        is exactly the shape the absence rule exists to prevent."""
        from bibr.pipeline.stages.export import _build_extraction

        paper = _minimal_paper()
        ext = _build_extraction(self._ctx(), paper)
        assert ext["timings"] is None

        paper.extraction = ext
        assert "timings" not in export_paper_to_json(paper)["extraction"]

    def test_build_extraction_prefers_per_file_timings(self):
        # In a multi-file chunk the chunk wall clock must not be attributed to
        # every paper: each export reports the file's own stage_times.
        import types

        from bibr.pipeline.stages.export import _build_extraction

        ctx = self._ctx(timings={"ocr": 100.0, "extract": 50.0})  # chunk totals
        fs = types.SimpleNamespace(stage_times={"ocr": 2.0, "extract": 1.0009})
        ext = _build_extraction(ctx, _minimal_paper(), fs)
        assert ext["timings"]["stages"] == {"ocr": 2.0, "extract": 1.001}
        assert ext["timings"]["total_seconds"] == round(2.0 + 1.0009, 3)

    def test_build_extraction_falls_back_to_chunk_timings(self):
        # A caller without per-file times (or an empty dict) keeps the old
        # chunk-level numbers rather than exporting nothing.
        import types

        from bibr.pipeline.stages.export import _build_extraction

        ctx = self._ctx(timings={"ocr": 4.0})
        fs = types.SimpleNamespace(stage_times={})
        ext = _build_extraction(ctx, _minimal_paper(), fs)
        assert ext["timings"]["stages"] == {"ocr": 4.0}
        assert ext["timings"]["total_seconds"] == 4.0


class TestQualificationProvenanceRidesExtraction:
    """v12 moved the deployment-qualification surface from the root
    ``qualification_provenance`` key to ``extraction.qualification``. It follows
    the ``extraction`` absence rule: omitted when no LLM task ran."""

    def test_omitted_when_no_llm_ran(self):
        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        assert paper.qualification_provenance is None

        result = export_paper_to_json(paper)

        assert "qualification" not in result["extraction"]
        assert "qualification_provenance" not in result

    def test_populated_when_an_llm_ran(self):
        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        paper.qualification_provenance = {"model_id": "nuextract3", "fallback_outcome": "recovered"}

        result = export_paper_to_json(paper)

        assert result["extraction"]["qualification"]["model_id"] == "nuextract3"
        assert "qualification_provenance" not in result


# ── paper self-identity: scalar fields + info_match ────────────────────


class TestPaperSelfIdentityExport:
    """The paper's OWN bibliographic identity: scalar fields round-trip into
    ``metadata`` and the enrichment match flattens into ``metadata_match``."""

    def test_biblio_scalars_round_trip(self):
        meta = PaperMetadata(
            doi="10.1234/test",
            title="Test Paper",
            journal="Psychological Science",
            volume="31",
            issue="1",
            first_page="65",
            last_page="74",
            issn="0956-7976",
            publisher="SAGE Publications",
            published="2020-01-01",
            license="CC BY 4.0",
        )
        exported = export_paper_to_json(_minimal_paper(metadata=meta))["metadata"]
        assert exported["journal"] == "Psychological Science"
        assert exported["volume"] == "31"
        assert exported["issue"] == "1"
        assert exported["first_page"] == "65"
        assert exported["last_page"] == "74"
        assert exported["issn"] == "0956-7976"
        assert exported["publisher"] == "SAGE Publications"
        assert exported["published"] == "2020-01-01"
        assert exported["license"] == "CC BY 4.0"

    def test_absent_biblio_is_null(self):
        meta = PaperMetadata(doi="10.1234/test", title="Test Paper")
        exported = export_paper_to_json(_minimal_paper(metadata=meta))["metadata"]
        assert exported["journal"] is None
        assert exported["volume"] is None
        assert exported["publisher"] is None
        assert exported["published"] is None
        assert exported["license"] is None

    def test_metadata_match_serializes_like_bib_match(self):
        meta = PaperMetadata(
            doi="10.1234/test",
            title="Test Paper",
            match={
                MatchSource.CROSSREF: ExternalMatch(
                    id="10.1234/test",
                    score=100.0,
                    title="Test Paper",
                    container="Psychological Science",
                    volume="31",
                    issue="1",
                    first_page="65",
                    last_page="74",
                    publisher="SAGE Publications",
                    authors=[BibAuthor(given="A", family="B")],
                    doi="10.1234/test",
                )
            },
        )
        out = export_paper_to_json(_minimal_paper(metadata=meta))
        metadata_match = out["metadata_match"]
        assert len(metadata_match) == 1
        entry = metadata_match[0]
        assert "bib_id" not in entry
        assert entry["service"] == "crossref"
        assert entry["service_id"] == "10.1234/test"
        assert entry["score"] == 1.0
        assert entry["container"] == "Psychological Science"
        assert entry["volume"] == "31"
        assert entry["first_page"] == "65"
        assert entry["author"] == [{"given": "A", "family": "B"}]

    def test_no_match_empty_metadata_match(self):
        meta = PaperMetadata(doi="10.1234/test", title="Test Paper")
        out = export_paper_to_json(_minimal_paper(metadata=meta))
        assert out["metadata_match"] == []


# ── xref tier export ──────────────────────────────────────────────────


def test_xref_tier_is_exported_under_diagnostics_not_on_the_row():
    from bibr.export.json_export import XrefExport

    x = XrefExport(xref_id=1, target_id=1, xref_type="bib", contents="[1]", text_id=3)
    assert "tier" not in x.model_dump()

    paper = _minimal_paper()
    paper.extraction = _extraction_block()
    paper.contents.xrefs.append(
        PaperXref(xref_id=1, xref_type="bib", contents="[1]", text_id=1, tier="numeric")
    )
    result = export_paper_to_json(paper)
    xref_id = result["xref"][-1]["xref_id"]
    assert result["extraction"]["diagnostics"]["xref_tier"] == [
        {"xref_id": xref_id, "tier": "numeric"}
    ]
