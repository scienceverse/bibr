"""Document orchestration through real post-parse, identity and export stages."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bibr.config import snapshot_settings
from bibr.export.document_models import DocumentExport
from bibr.models import PaperMetadata, PaperReference
from bibr.pipeline.context import RunConfig
from bibr.pipeline.document import process_document
from bibr.pipeline.stages.export import ExportStage
from bibr.pipeline.stages.identity import IdentityValidationStage
from bibr.pipeline.stages.post_parse import PostParseStage
from tests.extract.test_document_scope import _document


def _pipeline(monkeypatch, *, fail_record=None, bad_boundary=False, leading_fragment=False):
    contents, _resolution = _document(same_page=True)
    if leading_fragment:
        from bibr.paper_contents import PaperSentence

        contents.sentences.insert(
            0, PaperSentence(99, "Remaining fragment of an earlier abstract.", 0, 99, 1)
        )
    if bad_boundary:
        contents.sentences[3], contents.sentences[6] = contents.sentences[6], contents.sentences[3]
    observations = {"front_calls": 0, "records": []}

    class ParsedDocument:
        name = "parse"

        async def run(self, ctx):
            observations["front_calls"] += 1
            fs = ctx.file_states[0]
            fs.contents = deepcopy(contents)
            fs.content_sha256 = "a" * 64
            fs.file_hash = "a" * 16
            fs.native_metadata = {"doi": "10.9999/global-leak", "title": "GLOBAL TITLE"}

    async def metadata(scoped, *args, **kwargs):
        from bibr.extract.ref_locator import RefLocator

        record_id = scoped.front_matter_resolution.selected_block_id
        observations["records"].append(record_id)
        if record_id == fail_record:
            raise ValueError("source text and secret-token must not be exported")
        refs = RefLocator(scoped).collect_reference_rows()
        return PaperMetadata(
            doi="",
            title=scoped.detected_title,
            abstract="The controlled study measured seedling development.",
            references=[
                PaperReference(
                    bib_id=1,
                    title=refs.iloc[0].text,
                    first_page="1",
                    volume="1",
                    authors="River A.",
                    year=2020,
                    container="Forest Journal",
                    text_id=int(refs.iloc[0].text_id),
                )
            ],
        )

    # Mock only the shared parse boundary and model-driven extraction. Article
    # detection/scoping, post-processing, DOI selection and serialization are real.
    monkeypatch.setattr(
        "bibr.pipeline.document.build_stage_plan",
        lambda **kwargs: (
            ParsedDocument(),
            PostParseStage(),
            IdentityValidationStage(),
            ExportStage(),
        ),
    )
    monkeypatch.setattr("bibr.pipeline.stages.post_parse._extract_metadata_and_equations", metadata)
    settings = snapshot_settings()
    pipeline = SimpleNamespace(
        settings=settings,
        _config=RunConfig(
            no_llm=True,
            crossref=False,
            equations=False,
            ref_parse_strategy="off",
            consolidate="off",
        ),
        _resources=SimpleNamespace(classifiers=None),
    )
    return pipeline, observations


async def test_all_articles_have_independent_identity_references_and_body(monkeypatch):
    pipeline, observations = _pipeline(monkeypatch)

    payload = await process_document(pipeline, "collected.pdf")

    result = DocumentExport.model_validate(payload)
    assert result.status == "complete", payload
    assert observations["front_calls"] == 1
    assert len(observations["records"]) == len(result.records) == 2
    assert result.document_id == "sha256:" + "a" * 64
    assert result.diagnostics.detected_record_ids == [r.record_id for r in result.records]
    assert result.diagnostics.unassigned_source_text_ids == []
    for index, record in enumerate(payload["records"], start=1):
        paper = record["paper"]
        assert paper["source"] == payload["source"]
        assert paper["metadata"]["doi"] == f"10.9999/study-{index}"
        assert f"Study {index}." in paper["bib"][0]["title"]
        assert f"Study {3 - index}." not in str(paper)
        assert f"Article {3 - index} body" not in str(paper)
        assert "GLOBAL TITLE" not in str(paper)
        assert paper["schema_version"] == "11.1"
        assert (
            paper["extraction"]["diagnostics"]["front_matter"]["selected_block_id"]
            == record["record_id"]
        )


async def test_first_record_failure_does_not_drop_or_block_the_second(monkeypatch):
    pipeline, observations = _pipeline(monkeypatch, fail_record="front-matter-block-1")

    payload = await process_document(pipeline, "collected.pdf")

    assert payload["status"] == "partial"
    assert [row["status"] for row in payload["records"]] == ["failed", "extracted"]
    assert observations["records"] == ["front-matter-block-1", "front-matter-block-2"]
    assert payload["records"][0]["paper"] is None
    assert "secret-token" not in str(payload)
    assert "source text" not in str(payload)


async def test_ambiguous_boundaries_keep_all_candidates_without_extraction(monkeypatch):
    pipeline, observations = _pipeline(monkeypatch, bad_boundary=True)

    payload = await process_document(pipeline, "collected.pdf")

    assert payload["status"] == "unresolved"
    assert len(payload["records"]) == 2
    assert observations["records"] == []
    assert all(row["paper"] is None and row["reason_flags"] for row in payload["records"])
    assert "unassigned_document_text" in payload["diagnostics"]["reason_flags"]
    assert payload["diagnostics"]["unassigned_source_text_ids"]


async def test_no_detected_records_is_explicitly_unresolved(monkeypatch):
    from dataclasses import replace

    pipeline, observations = _pipeline(monkeypatch)
    _, resolution = _document()
    empty = replace(resolution, candidates=(), blocks=())
    monkeypatch.setattr(
        "bibr.pipeline.document.resolve_front_matter", lambda *args, **kwargs: (empty, ())
    )

    payload = await process_document(pipeline, "collected.pdf")

    assert payload["status"] == "unresolved"
    assert payload["records"] == []
    assert "no_article_records_detected" in payload["diagnostics"]["reason_flags"]
    assert observations["records"] == []


async def test_local_pipeline_exposes_document_path(monkeypatch):
    from bibr.local.pipeline import LocalPipeline

    process = AsyncMock(return_value={"document_schema_version": "1.0"})
    monkeypatch.setattr("bibr.pipeline.document.process_document", process)
    pipeline = object.__new__(LocalPipeline)

    assert await pipeline.process_document("bundle.pdf") == {"document_schema_version": "1.0"}
    assert process.await_args.args == (pipeline, "bundle.pdf")


async def test_bad_record_source_is_failed_without_losing_other_records(monkeypatch):
    from bibr.pipeline import document

    pipeline, _ = _pipeline(monkeypatch)
    plan = document.build_stage_plan

    class BadSourceExport(ExportStage):
        async def run(self, ctx):
            await super().run(ctx)
            payload = ctx.file_states[0].result_json
            if payload["metadata"]["doi"] == "10.9999/study-1":
                payload["source"]["file_name"] = "foreign.pdf"

    monkeypatch.setattr(
        document,
        "build_stage_plan",
        lambda **kwargs: (*plan(**kwargs)[:-1], BadSourceExport()),
    )

    payload = await process_document(pipeline, "collected.pdf")

    assert payload["status"] == "partial"
    assert [row["status"] for row in payload["records"]] == ["failed", "extracted"]
    assert "foreign.pdf" not in str(payload)


async def test_leading_fragment_is_reported_without_assigning_it_to_an_article(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, leading_fragment=True)

    payload = await process_document(pipeline, "collected.pdf")

    assert payload["status"] == "complete"
    assert len(payload["records"]) == 2
    assert payload["diagnostics"]["unassigned_source_text_ids"] == [99]
    assert "unassigned_document_text" in payload["diagnostics"]["reason_flags"]
    assert all(99 not in row["source_text_ids"] for row in payload["records"])
    assert "Remaining fragment" not in str(payload)
