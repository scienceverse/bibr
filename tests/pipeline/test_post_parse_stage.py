"""PostParseStage — extractor invocation (post_parse helper)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.models import PaperAuthor
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.post_parse import PostParseStage, post_parse
from bibr.pipeline.state import FileState
from bibr.processing_warnings import WarningCode


def _ctx(file_states):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )


@pytest.mark.asyncio
async def test_populates_paper_on_success():
    fs = FileState(path=Path("x.pdf"))
    fs.contents = MagicMock(layout_hints=None)
    fs.file_hash = "abc"

    paper = MagicMock()
    with patch("bibr.pipeline.stages.post_parse.post_parse", AsyncMock(return_value=paper)) as mk:
        await PostParseStage().run(_ctx([fs]))

    mk.assert_awaited_once()
    assert mk.await_args.kwargs["file_hash"] == "abc"
    assert fs.paper is paper
    assert fs.error is None
    assert "extract" in fs.stage_times


@pytest.mark.asyncio
async def test_stage_threads_equation_switch_to_post_parse():
    fs = FileState(path=Path("x.pdf"))
    fs.contents = MagicMock(layout_hints=None)
    ctx = _ctx([fs])
    ctx.config.equations = False

    with patch(
        "bibr.pipeline.stages.post_parse.post_parse", AsyncMock(return_value=MagicMock())
    ) as mk:
        await PostParseStage().run(ctx)

    assert mk.await_args.kwargs["extract_equations"] is False


@pytest.mark.asyncio
async def test_sets_error_on_exception():
    fs = FileState(path=Path("x.pdf"))
    fs.contents = MagicMock(layout_hints=None)

    async def boom(**kw):
        raise RuntimeError("nope")

    with patch("bibr.pipeline.stages.post_parse.post_parse", boom):
        await PostParseStage().run(_ctx([fs]))

    assert fs.error is not None
    assert fs.error_code == "extraction_failed"
    assert fs.failed_stage == "extract"


@pytest.mark.asyncio
async def test_typed_llm_failure_keeps_its_code():
    from bibr.exceptions import LlmTimeoutError

    fs = FileState(path=Path("x.pdf"))
    fs.contents = MagicMock(layout_hints=None)
    error = LlmTimeoutError("Failed to extract title/keywords", cause="timed out after 240s")

    with patch("bibr.pipeline.stages.post_parse.post_parse", AsyncMock(side_effect=error)):
        await PostParseStage().run(_ctx([fs]))

    assert fs.original_error is error
    assert fs.error_code == "llm_timeout"
    assert fs.failed_stage == "extract"


@pytest.mark.asyncio
async def test_typed_processing_error_keeps_code_identity_and_clears_traceback():
    from bibr.exceptions import ProcessingError

    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
    )
    try:
        raw_completion = "RAW-COMPLETION-SENTINEL"
        raise RuntimeError("safe native diagnostic")
    except RuntimeError as caught:
        assert raw_completion
        cause = caught
    try:
        raise error from cause
    except ProcessingError as caught:
        error = caught
    assert error.__traceback__ is not None
    assert cause.__traceback__ is not None

    fs = FileState(path=Path("x.pdf"))
    fs.contents = MagicMock(layout_hints=None)
    with patch(
        "bibr.pipeline.stages.post_parse.post_parse",
        AsyncMock(side_effect=error),
    ):
        await PostParseStage().run(_ctx([fs]))

    assert fs.original_error is error
    assert fs.error_code == "llm_invalid_output"
    assert fs.failed_stage == "extract"
    assert error.__traceback__ is None
    assert cause.__traceback__ is None


@pytest.mark.asyncio
async def test_failed_post_parse_attaches_usage_before_file_buckets_are_evicted():
    from bibr.exceptions import ProcessingError, SafeLlmDiagnostics

    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
        safe_diagnostics=SafeLlmDiagnostics(
            invalid_category="non_json",
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            cached_input_tokens=3,
        ),
    )
    client = MagicMock()
    client._track_usage = True
    client.usage_pop_file.return_value = {
        "numind/NuExtract3-FP8": {
            "input_tokens": 21,
            "output_tokens": 9,
            "total_tokens": 30,
            "cached_input_tokens": 4,
        }
    }
    client.usage_labels_pop_file.return_value = {
        ("extract_title_keywords", "openai", "numind/NuExtract3-FP8"): {
            "input_tokens": 21,
            "output_tokens": 9,
            "total_tokens": 30,
            "cached_input_tokens": 4,
            "calls": 2,
            "logical_calls": 1,
            "attempts": 2,
            "retries": 0,
            "failed_calls": 1,
            "native_attempts": 1,
            "instructor_attempts": 1,
            "protocol_fallbacks": 1,
            "native_invalid_outputs": 1,
            "native_invalid_non_json": 1,
        }
    }
    contents = _minimal_contents()

    with patch(
        "bibr.pipeline.stages.post_parse._classify_sections",
        AsyncMock(side_effect=error),
    ):
        with pytest.raises(ProcessingError) as raised:
            await post_parse(
                contents=contents,
                file_name="x.pdf",
                file_hash="deadbeef",
                llm_client=client,
            )

    assert raised.value is error
    assert error.safe_diagnostics.total_tokens == 30
    labels = error.safe_diagnostics.to_dict()["llm_usage_by_label"]
    assert labels["extract_title_keywords"]["attempts"] == 2
    assert labels["extract_title_keywords"]["native_invalid_non_json"] == 1
    client.usage_pop_file.assert_called_once()
    client.usage_labels_pop_file.assert_called_once()


@pytest.mark.asyncio
async def test_owned_client_close_cancellation_propagates_after_typed_failure():
    import asyncio

    from bibr.exceptions import ProcessingError, SafeLlmDiagnostics

    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
        safe_diagnostics=SafeLlmDiagnostics(invalid_category="non_json"),
    )
    client = MagicMock()
    client._track_usage = False
    client.usage_pop_file.return_value = {}
    client.usage_labels_pop_file.return_value = {}
    client.close = AsyncMock(side_effect=asyncio.CancelledError())

    with (
        patch("bibr.clients.llm.LLMClient", return_value=client),
        patch(
            "bibr.pipeline.stages.post_parse._classify_sections",
            AsyncMock(side_effect=error),
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await post_parse(
                contents=MagicMock(),
                file_name="x.pdf",
                file_hash="deadbeef",
            )

    client.close.assert_awaited_once()


def _minimal_contents() -> PaperContents:
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(section_id=1, header="Introduction", level=1, parent_section_id=0),
        PaperSection(section_id=2, header="References", level=1, parent_section_id=0),
        PaperSection(section_id=3, header="Flurbulations", level=1, parent_section_id=0),
    ]
    return PaperContents(
        sentences=[],
        sections=sections,
        tables=[],
        links=[],
        sections_text={0: "", 1: "intro body", 2: "ref 1; ref 2", 3: "something unknown"},
        detected_title="My Paper",
    )


@pytest.mark.asyncio
async def test_no_llm_skips_all_llm_calls():
    """no_llm=True must not instantiate LLMClient or call citation_linker / implicit_sections / MetadataExtractor."""
    contents = _minimal_contents()

    with (
        patch("bibr.clients.llm.LLMClient") as mock_llm,
        patch(
            "bibr.structure.section_classifier.classify_headers_batch_async",
            AsyncMock(side_effect=AssertionError("must not be called")),
        ),
        patch(
            "bibr.extract.extractor.MetadataExtractor",
            side_effect=AssertionError("must not be instantiated"),
        ),
        patch(
            "bibr.structure.implicit_sections.detect_implicit_sections",
            AsyncMock(side_effect=AssertionError("must not be called")),
        ),
        patch(
            "bibr.structure.citation_linker.detect_bib_xrefs",
            AsyncMock(side_effect=AssertionError("must not be called")),
        ),
    ):
        paper = await post_parse(
            contents=contents,
            file_name="x.pdf",
            file_hash="deadbeef",
            no_llm=True,
        )

    mock_llm.assert_not_called()

    assert paper.metadata.doi == ""
    assert paper.metadata.title == "My Paper"  # from detected_title
    assert paper.metadata.authors == []
    assert paper.metadata.references == []
    assert paper.contents.equations == []
    assert paper.contents.xrefs == []

    intro = next(s for s in paper.contents.sections if s.header == "Introduction")
    refs = next(s for s in paper.contents.sections if s.header == "References")
    unknown = next(s for s in paper.contents.sections if s.header == "Flurbulations")
    assert intro.section_type == CanonicalSection.INTRODUCTION
    assert refs.section_type == CanonicalSection.REFERENCES
    assert unknown.section_type == CanonicalSection.UNKNOWN


@pytest.mark.asyncio
async def test_no_llm_empty_authors_keep_author_specific_funding_conservative():
    from bibr.config import snapshot_settings

    text = "Jane Doe was funded by NSF grant 123."
    contents = PaperContents(
        sentences=[
            PaperSentence(
                text_id=1,
                text=text,
                section_id=1,
                paragraph_id=1,
            )
        ],
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Discussion",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.DISCUSSION,
            ),
        ],
        tables=[],
        links=[],
        sections_text={},
        detected_title="T",
    )
    settings = snapshot_settings()
    settings.pipeline.integrity_statement_mode = "active"

    paper = await post_parse(
        contents=contents,
        file_name="no-llm.pdf",
        file_hash="no-llm",
        no_llm=True,
        settings=settings,
    )

    assert paper.metadata.authors == []
    assert paper.metadata.funding_statement is None


def _root_only_contents() -> PaperContents:
    """A single level-0 section and no sentences — nothing to classify, so the
    section classifier and its LLM path stay dormant without patching."""
    return PaperContents(
        sentences=[],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: ""},
    )


@pytest.mark.asyncio
async def test_integrity_runs_after_linking_and_final_text_cleaning(monkeypatch):
    """Selected statement IDs are rendered only after citation work and cleaning."""
    from bibr.models import FundingEntry, PaperMetadata
    from bibr.paper_contents import PaperXref

    monkeypatch.setattr("bibr.config.Settings.EQUATION_EXTRACTION", False)

    order = []

    async def fake_integrity(contents, metadata, llm_client, file_hash, *, integrity_resolution):
        assert order == ["linking"]
        assert integrity_resolution is not None
        order.append("integrity")
        metadata.funding = [FundingEntry(funder="ACME Foundation", award_ids=["42"])]

    async def fake_detect(**kwargs):
        order.append("linking")
        return [PaperXref(xref_id=1, xref_type="bib", contents="[1]", text_id=0)]

    extractor = MagicMock()
    extractor.extract_all_metadata = AsyncMock(return_value=PaperMetadata(doi="", title="T"))

    with (
        patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor),
        patch(
            "bibr.structure.implicit_sections.detect_implicit_sections",
            AsyncMock(return_value=None),
        ),
        patch("bibr.extract.research_integrity.extract_structured_integrity", fake_integrity),
        patch("bibr.structure.citation_linker.detect_bib_xrefs", fake_detect),
    ):
        paper = await post_parse(
            contents=_root_only_contents(),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=MagicMock(),
        )

    assert order == ["linking", "integrity"]
    assert paper.metadata.funding == [FundingEntry(funder="ACME Foundation", award_ids=["42"])]
    assert any(x.xref_id == 1 and x.xref_type == "bib" for x in paper.contents.xrefs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extracted_authors", "ocr_metadata", "expected_names"),
    [
        (
            [
                PaperAuthor(
                    author_id=1,
                    given="Jane W.",
                    family="Doe Jr.",
                    affiliation="",
                )
            ],
            None,
            (("Jane W.", "Doe Jr."),),
        ),
        ([], {"authors": ["Ana de la Cruz"]}, (("Ana de la", "Cruz"),)),
    ],
)
async def test_integrity_resolution_receives_available_author_snapshot(
    monkeypatch,
    extracted_authors,
    ocr_metadata,
    expected_names,
):
    from bibr.extract.integrity_statements import resolve_integrity_statements
    from bibr.models import PaperMetadata

    seen = {}

    def capture_resolution(contents, *, mode, author_names=()):
        seen["author_names"] = author_names
        return resolve_integrity_statements(
            contents,
            mode=mode,
            author_names=author_names,
        )

    extractor = MagicMock()
    extractor.extract_all_metadata = AsyncMock(
        return_value=PaperMetadata(
            doi="",
            title="T",
            authors=extracted_authors,
        )
    )
    monkeypatch.setattr(
        "bibr.extract.integrity_statements.resolve_integrity_statements",
        capture_resolution,
    )
    contents = _root_only_contents()
    if ocr_metadata:
        # Native/preparsed ownership is the path on which fill-empty metadata
        # remains permitted without a selected PDF front-matter block.
        contents.preparsed_metadata = PaperMetadata(doi="", title="T")

    with (
        patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor),
        patch(
            "bibr.structure.implicit_sections.detect_implicit_sections",
            AsyncMock(return_value=None),
        ),
        patch(
            "bibr.structure.citation_linker.detect_bib_xrefs",
            AsyncMock(return_value=[]),
        ),
    ):
        paper = await post_parse(
            contents=contents,
            file_name="authors.pdf",
            file_hash="authors",
            ocr_metadata=ocr_metadata,
            llm_client=MagicMock(),
            extract_equations=False,
        )

    assert seen["author_names"] == expected_names
    assert (
        tuple((author.given, author.family) for author in paper.metadata.authors) == expected_names
    )


@pytest.mark.asyncio
async def test_default_shadow_preserves_legacy_statement_and_emits_typed_issue(monkeypatch):
    contents = PaperContents(
        sentences=[
            PaperSentence(
                text_id=1,
                text="This section discusses the ethics of discounting climate harms.",
                section_id=200,
                paragraph_id=1,
                page_number=12,
            )
        ],
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=200,
                header="Time Discounting: An Ethical Problem",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.ETHICS,
                classification_source="model",
                classification_score=0.999,
            ),
        ],
        tables=[],
        links=[],
        sections_text={},
        detected_title="T",
    )

    async def preserve_classification(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "bibr.pipeline.stages.post_parse._classify_sections", preserve_classification
    )
    paper = await post_parse(
        contents=contents,
        file_name="synthetic-topical-ethics.pdf",
        file_hash="synthetic-topical-ethics",
        no_llm=True,
    )

    assert paper.metadata.ethics_statement == contents.sentences[0].text
    issue = next(
        issue for issue in paper.validation_issues if issue.code == "VAL_STATEMENT_SUSPECT"
    )
    assert issue.origin_stage == "post_parse"
    assert "ethics_statement" in issue.evidence_ids
    assert not any(
        "VAL_STATEMENT_SUSPECT" in f"{w.code}: {w.message}" for w in paper.processing_warnings
    )


@pytest.mark.asyncio
async def test_default_shadow_keeps_pre_finalize_scalar_bytes_through_post_parse(monkeypatch):
    contents = PaperContents(
        sentences=[
            PaperSentence(
                text_id=1,
                text="This work was supported by NSF grant $^{123}$ .",
                section_id=1,
                paragraph_id=7,
                page_number=1,
            ),
            PaperSentence(
                text_id=2,
                text="Ethics: Not applicable.",
                section_id=1,
                paragraph_id=7,
                page_number=1,
            ),
        ],
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Funding",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.FUNDING,
                classification_source="exact_alias",
                classification_score=1.0,
            ),
        ],
        tables=[],
        links=[],
        sections_text={},
        detected_title="T",
    )

    async def preserve_classification(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "bibr.pipeline.stages.post_parse._classify_sections", preserve_classification
    )
    paper = await post_parse(
        contents=contents,
        file_name="shadow-compat.pdf",
        file_hash="shadow-compat",
        no_llm=True,
    )

    assert paper.metadata.funding_statement == (
        "This work was supported by NSF grant $^{123}$ . Ethics: Not applicable."
    )
    assert paper.contents.sentences[0].text == "This work was supported by NSF grant 123 ."
    funding_issues = [
        issue
        for issue in paper.validation_issues
        if issue.code == "VAL_STATEMENT_SUSPECT" and "funding_statement" in issue.evidence_ids
    ]
    assert len(funding_issues) == 1


@pytest.mark.asyncio
async def test_native_metadata_passed_as_ocr_metadata(monkeypatch):
    """The stage feeds fs.native_metadata (doc-info harvest) into post_parse's
    fill-empty merge parameter."""
    import bibr.pipeline.stages.post_parse as pp_mod

    seen = {}

    async def _fake_post_parse(contents, file_name, file_hash, **kwargs):
        seen["ocr_metadata"] = kwargs.get("ocr_metadata")
        return MagicMock(name="paper")

    monkeypatch.setattr(pp_mod, "post_parse", _fake_post_parse)

    fs = FileState(path=Path("a.pdf"))
    fs.contents = MagicMock(layout_hints=None)
    fs.native_metadata = {"title": "Doc Info Title", "doi": "10.1/x"}
    ctx = _ctx([fs])

    await PostParseStage().run(ctx)

    assert seen["ocr_metadata"] == {"title": "Doc Info Title", "doi": "10.1/x"}


@pytest.mark.asyncio
async def test_native_reference_failure_preserves_core_and_marks_incomplete():
    """Native/JATS reference parsing has the same durable-core contract as PDF."""
    import pandas as pd

    from bibr.config import snapshot_settings
    from bibr.models import PaperMetadata
    from bibr.pipeline.stages.post_parse import _resolve_preparsed_references

    metadata = PaperMetadata(doi="10.1234/native", title="Native Core")
    contents = MagicMock()
    contents.native_references = None
    extractor = MagicMock()
    extractor._collect_reference_rows.return_value = pd.DataFrame({"text": ["Reference"]})
    extractor._extract_references = AsyncMock(side_effect=RuntimeError("native parser down"))

    with patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor):
        result = await _resolve_preparsed_references(
            contents,
            metadata,
            "deadbeef",
            MagicMock(),
            "native",
            "ner",
            settings=snapshot_settings(),
        )

    assert result is metadata
    assert result.title == "Native Core"
    assert result.doi == "10.1234/native"
    assert result.references == []
    assert result.references_incomplete is True


@pytest.mark.asyncio
async def test_native_reference_row_collection_failure_marks_incomplete():
    from bibr.config import snapshot_settings
    from bibr.models import PaperMetadata
    from bibr.pipeline.stages.post_parse import _resolve_preparsed_references

    metadata = PaperMetadata(doi="10.1234/native", title="Native Core")
    contents = MagicMock()
    contents.native_references = None
    extractor = MagicMock()
    extractor._collect_reference_rows.side_effect = RuntimeError("native locator crashed")

    with patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor):
        result = await _resolve_preparsed_references(
            contents,
            metadata,
            "deadbeef",
            MagicMock(),
            "native",
            "ner",
            settings=snapshot_settings(),
        )

    assert result is metadata
    assert result.references == []
    assert result.references_incomplete is True
    assert "native locator crashed" in result._references_incomplete_diagnostic


@pytest.mark.parametrize("control_error", [SystemExit("stop"), KeyboardInterrupt()])
@pytest.mark.asyncio
async def test_native_reference_parse_control_errors_propagate(control_error):
    import pandas as pd

    from bibr.config import snapshot_settings
    from bibr.models import PaperMetadata
    from bibr.pipeline.stages.post_parse import _resolve_preparsed_references

    metadata = PaperMetadata(doi="10.1234/native", title="Native Core")
    contents = MagicMock()
    contents.native_references = None
    extractor = MagicMock()
    extractor._collect_reference_rows.return_value = pd.DataFrame({"text": ["Reference"]})
    extractor._extract_references = AsyncMock(side_effect=control_error)

    with (
        patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor),
        pytest.raises(type(control_error)),
    ):
        await _resolve_preparsed_references(
            contents,
            metadata,
            "deadbeef",
            MagicMock(),
            "native",
            "ner",
            settings=snapshot_settings(),
        )


def test_attach_text_quality_counts_empty_scoreable_regions():
    """Empty scoreable region summaries (OCR coverage failures) must reach the
    scorer. post_parse previously dropped them with an ``if rs.content`` filter,
    making the coverage-failure score inert on real papers."""
    from types import SimpleNamespace

    from bibr.pipeline.stages.post_parse import _attach_text_quality

    # label "text" -> treatment "content" (scoreable) via PDFParser.LABEL_TREATMENT.
    summaries = [SimpleNamespace(page=1, label="text", content="") for _ in range(5)]
    contents = SimpleNamespace(region_summaries=summaries)
    paper = SimpleNamespace(text_quality=None, processing_warnings=[])

    _attach_text_quality(paper, contents)

    # 10th percentile of all-0.0 scores is 0.0 (< default warn threshold 0.5).
    assert paper.text_quality == 0.0
    assert any(w.code == WarningCode.LOW_TEXT_QUALITY for w in paper.processing_warnings)


def test_attach_text_quality_high_when_regions_populated():
    """The same regions populated with clean prose score high — no warning —
    so the coverage penalty is specific to blank regions, not all-of-them."""
    from types import SimpleNamespace

    from bibr.pipeline.stages.post_parse import _attach_text_quality

    summaries = [
        SimpleNamespace(page=1, label="text", content="This is a clean sentence of prose.")
        for _ in range(5)
    ]
    contents = SimpleNamespace(region_summaries=summaries)
    paper = SimpleNamespace(text_quality=None, processing_warnings=[])

    _attach_text_quality(paper, contents)

    assert paper.text_quality == 1.0
    assert paper.processing_warnings == []


def _decide_unscoped_title(detected, llm_title, *, journal=None, publisher=None, resolution=None):
    """The title decision of an unscoped run, where the layout title is weighed."""
    from bibr.extract.field_decisions import FieldCandidate, decide_title

    return decide_title(
        FieldCandidate("title", "llm", llm_title),
        resolution=resolution,
        detected_title=detected,
        sections=[],
        journal=journal,
        publisher=publisher,
        scoped=False,
        abstained=False,
        prefer_byline_adjacent=False,
        doc_info=None,
    )


class TestResolveTitleMastheadGuard:
    """Prefer an extracted article title when the layout title has specific masthead evidence.

    Journal/publisher agreement or a URL/ISSN marker identifies page furniture. A title disagreement alone must not trigger replacement."""

    @staticmethod
    def _run(detected_title, llm_title, journal, publisher=None):
        return _decide_unscoped_title(
            detected_title, llm_title, journal=journal, publisher=publisher
        ).value

    def test_masthead_exact_journal_match_uses_llm_title(self):
        # srep45484: layout grabbed "SCIENTIFIC REPORTS" == journal.
        real = "Single crystalline superstructured stable single domain magnetite nanoparticles"
        assert self._run("SCIENTIFIC REPORTS", real, "SCIENTIFIC REPORTS") == real

    def test_masthead_journal_substring_plus_url_uses_llm_title(self):
        # ijlgc: detected is the journal name (uppercased) + a URL suffix.
        detected = (
            "INTERNATIONAL JOURNAL OF LAW, GOVERNMENT AND COMMUNICATION (IJLGC) www.ijlgc.com"
        )
        real = "Navigating Digital Dialogue: Asnaf Students' Experiences Of Communication"
        journal = "International Journal of Law, Government and Communication"
        assert self._run(detected, real, journal) == real

    def test_masthead_publisher_match_uses_llm_title(self):
        # Banner is the PUBLISHER, not the journal (journal-only would miss it).
        real = "A Randomized Trial of Widget Efficacy"
        assert self._run("ELSEVIER", real, "The Lancet", publisher="Elsevier") == real

    def test_masthead_url_marker_uses_llm_title(self):
        # Banner carries a bare URL and matches neither journal nor publisher.
        real = "Deep Learning for Protein Folding"
        assert self._run("www.somerepository.org", real, "Bioinformatics", "OUP") == real

    def test_normal_paper_keeps_verbatim_detected_title(self):
        # Real doc_title, journal differs → NOT a masthead → keep layout verbatim.
        detected = "A Study of Widgets in the Wild"
        assert (
            self._run(detected, "A Study of Widgets in the Wild", "Journal of Widget Science")
            == detected
        )

    def test_no_journal_keeps_detected_title(self):
        # --no-llm path: journal is None → cannot detect masthead → keep layout.
        assert self._run("My Paper", "My Paper", None) == "My Paper"

    def test_llm_agreement_keeps_detected_even_if_journalish(self):
        # Disagreement gate: if the LLM title matches the detected title, keep
        # the verbatim layout title (do not override on a paraphrase).
        title = "Advances in Journal Science"
        assert self._run(title, title, title) == title

    def test_empty_llm_title_keeps_detected_masthead(self):
        # Nothing to recover with: no LLM title → keep the (masthead) layout title.
        assert self._run("SCIENTIFIC REPORTS", None, "SCIENTIFIC REPORTS") == "SCIENTIFIC REPORTS"

    def test_regression_guard_correct_layout_title_kept_over_wrong_llm_who(self):
        detected = "Global strategy for the prevention and control of noncommunicable diseases"
        assert (
            self._run(
                detected,
                "Report by the Director-General",
                "WORLD HEALTH ORGANIZATION FIFTY-THIRD WORLD HEALTH ASSEMBLY",
            )
            == detected
        )

    def test_regression_guard_correct_layout_title_kept_over_wrong_llm_historyka(self):
        detected = (
            "FAMILY HISTORIES – A REFLECTION ON THE DEVELOPMENT OF HISTORIOGRAPHIC GENRE FORMS"
        )
        assert (
            self._run(
                detected,
                "THE SCOPE OF HISTORIOGRAPHIC RESEARCH AROUND THE IMAGE OF FAMILY HISTORIES",
                "HISTORYKA. Studies in Historical Methods",
            )
            == detected
        )


class TestExactGenericArticleTitles:
    @pytest.mark.parametrize(
        "label",
        [
            "  RESEARCH   ARTICLE. ",
            "(Research Article)",
            "“Original Article”",
            "— Original Research —",
            "\u200fResearch Article\u200f",
            "【Original Article】",
        ],
    )
    def test_helper_strips_unicode_surrounding_punctuation_and_format_symbols(self, label):
        from bibr.utils.metadata import is_exact_generic_article_label

        assert is_exact_generic_article_label(label)

    @pytest.mark.parametrize(
        "label",
        [
            "Research Article: Effects of X",
            "An Original Research Article",
            "Research—Article",
            "Original (Research)",
            "Original Article — Effects of X",
        ],
    )
    def test_helper_never_strips_internal_punctuation_or_matches_substrings(self, label):
        from bibr.utils.metadata import is_exact_generic_article_label

        assert not is_exact_generic_article_label(label)

    @staticmethod
    def _run(detected, llm_title, *, grounded, roles=frozenset({"title"})):
        from bibr.extract.front_matter import (
            FrontMatterBlock,
            FrontMatterCandidate,
            FrontMatterResolution,
        )

        candidate = FrontMatterCandidate(
            candidate_id="title-candidate",
            source_kind="paragraph",
            reading_order=0,
            page=1,
            bbox=None,
            region_label="doc_title",
            font_size=None,
            font_bold=None,
            section_id=1,
            text_ids=(1,),
            paragraph_id=1,
            raw_text=llm_title if grounded else "Different source title",
            normalized_text=(llm_title if grounded else "Different source title").casefold(),
            roles=roles,
        )
        block = FrontMatterBlock(
            block_id="selected",
            candidate_ids=(candidate.candidate_id,),
            title_candidate_ids=(candidate.candidate_id,),
        )
        resolution = FrontMatterResolution(
            candidates=(candidate,),
            blocks=(block,),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=frozenset({1}),
            allowed_section_ids=frozenset({1}),
        )
        return _decide_unscoped_title(detected, llm_title, resolution=resolution).value

    def test_exact_generic_label_yields_to_grounded_non_generic_title(self):
        real = "Effects of X on Y"
        assert self._run("Research Article", real, grounded=True) == real

    def test_exact_generic_label_does_not_yield_to_ungrounded_title(self):
        assert (
            self._run("Research Article", "Effects of X on Y", grounded=False) == "Research Article"
        )

    def test_exact_generic_label_yields_to_grounded_heading_title(self):
        real = "Effects of X on Y"
        assert (
            self._run(
                "Research Article",
                real,
                grounded=True,
                roles=frozenset({"title", "heading"}),
            )
            == real
        )

    def test_exact_generic_label_does_not_yield_to_composite_title(self):
        assert (
            self._run(
                "Research Article",
                "Effects of X on Y",
                grounded=True,
                roles=frozenset({"title", "byline"}),
            )
            == "Research Article"
        )

    def test_generic_prefix_with_real_title_is_retained(self):
        detected = "Research Article: Effects of X"
        assert self._run(detected, "Effects of X", grounded=True) == detected

    async def test_active_selected_pipeline_executes_safe_generic_resolution(self, monkeypatch):
        import bibr.pipeline.stages.post_parse as post_parse_module
        from bibr.extract.front_matter import (
            FrontMatterBlock,
            FrontMatterCandidate,
            FrontMatterResolution,
        )
        from bibr.models import PaperMetadata

        real = "Effects of X on Y"
        contents = _minimal_contents()
        contents.detected_title = "Research Article"
        candidate = FrontMatterCandidate(
            candidate_id="selected-title",
            source_kind="paragraph",
            reading_order=0,
            page=1,
            bbox=None,
            region_label="doc_title",
            font_size=None,
            font_bold=None,
            section_id=1,
            text_ids=(),
            paragraph_id=1,
            raw_text=real,
            normalized_text=real.casefold(),
            roles=frozenset({"title"}),
        )
        block = FrontMatterBlock(
            block_id="selected",
            candidate_ids=(candidate.candidate_id,),
            title_candidate_ids=(candidate.candidate_id,),
        )
        resolution = FrontMatterResolution(
            candidates=(candidate,),
            blocks=(block,),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=frozenset(),
            allowed_section_ids=frozenset({1}),
        )

        async def classify(*_args, **_kwargs):
            return None

        def attach(actual_contents, *_args, **_kwargs):
            actual_contents.front_matter_resolution = resolution
            return ()

        async def extract(*_args, **_kwargs):
            return PaperMetadata(doi="", title=real)

        monkeypatch.setattr(post_parse_module, "_classify_sections", classify)
        monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach)
        monkeypatch.setattr(post_parse_module, "_normalize_section_structure", classify)
        monkeypatch.setattr(post_parse_module, "_extract_metadata_and_equations", extract)
        monkeypatch.setattr(post_parse_module, "_link_citations", AsyncMock())
        monkeypatch.setattr(
            "bibr.extract.research_integrity.extract_structured_integrity", AsyncMock()
        )
        client = MagicMock()

        paper = await post_parse_module.post_parse(
            contents,
            "source.pdf",
            "source-hash",
            llm_client=client,
        )

        assert paper.metadata.title == real


class TestResolveSelectedTitle:
    """Task 4: a null LLM title may use exactly one safe selected candidate."""

    @staticmethod
    def _candidate(candidate_id, text, roles):
        from bibr.extract.front_matter import FrontMatterCandidate

        return FrontMatterCandidate(
            candidate_id=candidate_id,
            source_kind="paragraph",
            reading_order=0,
            page=1,
            bbox=None,
            region_label="text",
            font_size=None,
            font_bold=None,
            section_id=1,
            text_ids=(),
            paragraph_id=1,
            raw_text=text,
            normalized_text=" ".join(text.casefold().split()),
            roles=frozenset(roles),
        )

    @classmethod
    def _case(cls, selected, *, journal=None, publisher=None):
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
        return resolution, {"journal": journal, "publisher": publisher}

    def test_null_llm_title_uses_one_safe_selected_title(self):
        from bibr.extract.title_candidates import selected_title_candidate

        resolution, record = self._case(
            [self._candidate("title", "Grounded selected title", roles={"title"})],
        )

        candidate, _ = selected_title_candidate(resolution, **record)
        assert candidate.value == "Grounded selected title"
        assert candidate.source == "front_matter_candidate"
        issues = candidate.issues
        assert issues[-1].code == "VAL_TITLE_RECOVERED"
        assert issues[-1].evidence_ids == ("title",)
        assert not issues[-1].blocking

    def test_normalized_duplicate_copies_collapse_to_one_candidate(self):
        from bibr.extract.title_candidates import selected_title_candidate

        resolution, record = self._case(
            [
                self._candidate("a", "Grounded selected title", roles={"title"}),
                self._candidate("b", "Grounded  Selected  Title", roles={"title"}),
            ],
        )

        candidate, _ = selected_title_candidate(resolution, **record)
        assert candidate.value == "Grounded selected title"

    @pytest.mark.parametrize(
        "selected",
        [
            [
                ("a", "First possible title", {"title"}),
                ("b", "Second possible title", {"title"}),
            ],
            [("masthead", "SCIENTIFIC REPORTS", {"title"})],
            [("related", "You may also like", {"title"})],
            [("copyright", "COPYRIGHT", {"title"})],
            [("abstract_heading", "RESUMO", {"title"})],
            [("kicker", "SHORT COMMUNICATION", {"title"})],
            [("generic", "Research Article", {"title"})],
            [("journalish", "Journal of Community Health", {"title"})],
            [("composite", "Title Alice Author", {"title", "byline"})],
            [("empty", "   ", {"title"})],
        ],
    )
    def test_unsafe_or_ambiguous_selected_titles_remain_null(self, selected):
        from bibr.extract.title_candidates import selected_title_candidate

        resolution, record = self._case(
            [self._candidate(*args) for args in selected],
            journal="Scientific Reports",
            publisher="Nature Publishing Group",
        )

        candidate, _ = selected_title_candidate(resolution, **record)
        assert candidate is None

    def test_abstention_never_recovers(self):
        from bibr.extract.title_candidates import selected_title_candidate

        resolution, record = self._case(
            [self._candidate("title", "Grounded selected title", roles={"title"})],
        )
        resolution = resolution.__class__(
            candidates=resolution.candidates,
            blocks=resolution.blocks,
            selected_block_id=None,
            selection_method="abstained",
            reason_flags=("multiple_plausible_blocks",),
            allowed_text_ids=frozenset(),
            allowed_section_ids=frozenset(),
        )

        candidate, reason = selected_title_candidate(resolution, **record)
        assert candidate is None
        assert reason == "no selected record"

    async def test_ownership_scope_recovers_selected_title_and_ignores_outside_evidence(
        self, monkeypatch
    ):
        import bibr.pipeline.stages.post_parse as post_parse_module
        from bibr.models import PaperMetadata

        real = "Grounded selected title"
        contents = _minimal_contents()
        # An outside layout banner and an unknown section header
        # ("Flurbulations" in _minimal_contents) must both stay ignored.
        contents.detected_title = "INTERNATIONAL JOURNAL OF EXAMPLES"
        candidate = self._candidate("selected-title", real, roles={"title"})
        from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

        block = FrontMatterBlock(
            block_id="selected",
            candidate_ids=(candidate.candidate_id,),
            title_candidate_ids=(candidate.candidate_id,),
        )
        resolution = FrontMatterResolution(
            candidates=(candidate,),
            blocks=(block,),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=frozenset(),
            allowed_section_ids=frozenset({1}),
        )

        async def classify(*_args, **_kwargs):
            return None

        def attach(actual_contents, *_args, **_kwargs):
            actual_contents.front_matter_resolution = resolution
            return ()

        async def extract(*_args, **_kwargs):
            return PaperMetadata(doi="", title="")

        monkeypatch.setattr(post_parse_module, "_classify_sections", classify)
        monkeypatch.setattr(post_parse_module, "_attach_front_matter_resolution", attach)
        monkeypatch.setattr(post_parse_module, "_normalize_section_structure", classify)
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

        assert paper.metadata.title == real
        assert "VAL_TITLE_RECOVERED" in {issue.code for issue in paper.validation_issues}
