from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from bibr.config import Settings
from bibr.input.file import InputFile
from bibr.models import PaperMetadata
from bibr.paper import Paper
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.identity import ExpectedIdentity
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState


def _paper(text: str, *, doi: str = "") -> Paper:
    contents = PaperContents(
        sentences=[PaperSentence(1, text, 1, 1, page_number=1)],
        sections=[
            PaperSection(0, "Root", 0, None, CanonicalSection.TITLE),
            PaperSection(1, "Title", 1, 0, CanonicalSection.TITLE),
        ],
        tables=[],
        links=[],
        sections_text={},
    )
    input_file = InputFile(path="paper.pdf")
    input_file.file_hash = "a" * 16
    return Paper(
        input_file=input_file,
        contents=contents,
        metadata=PaperMetadata(doi=doi, title="Paper"),
    )


def _ctx(state: FileState) -> PipelineContext:
    return PipelineContext(
        file_states=[state],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(no_llm=True),
        settings=Settings,
    )


async def test_identity_stage_blocks_source_hash_mismatch_without_setting_file_error():
    from bibr.pipeline.stages.identity import IdentityValidationStage

    expected = ExpectedIdentity(
        queue_record_id="record-1",
        expected_doi="10.1234/paper",
        source_sha256="b" * 64,
        doi_required=True,
    )
    state = FileState(
        path=Path("paper.pdf"),
        expected_identity=expected,
        content_sha256="a" * 64,
        paper=_paper("Article DOI: 10.1234/paper", doi="10.1234/paper"),
    )

    await IdentityValidationStage().run(_ctx(state))

    assert state.error is None
    assert state.doi_selection is not None
    assert state.paper is not None
    assert "VAL_SOURCE_INTEGRITY" in {issue.code for issue in state.paper.validation_issues}
    issue = next(i for i in state.paper.validation_issues if i.code == "VAL_SOURCE_INTEGRITY")
    assert issue.blocking is True


async def test_validate_stage_recomputes_full_hash_for_manifest_disk_input(tmp_path, monkeypatch):
    from bibr.pipeline.stages.validate import ValidateStage

    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\ncurrent bytes")
    valid = MagicMock(is_valid=True, native_artifact=None)
    monkeypatch.setattr("bibr.input.validate.validate_input_file", lambda *_args, **_kw: valid)
    state = FileState(path=source, content_sha256="0" * 64)

    await ValidateStage().run(_ctx(state))

    assert state.content_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


async def test_identity_stage_reports_mismatch_and_keeps_source_selected_identity():
    from bibr.pipeline.stages.identity import IdentityValidationStage

    expected = ExpectedIdentity(
        queue_record_id="record-1", expected_doi="10.1234/expected", doi_required=True
    )
    state = FileState(
        path=Path("paper.pdf"),
        expected_identity=expected,
        content_sha256="a" * 64,
        paper=_paper("Article DOI: 10.1234/visible", doi="10.1234/stale"),
    )

    await IdentityValidationStage().run(_ctx(state))

    assert state.error is None
    assert state.paper is not None and state.paper.metadata is not None
    assert state.paper.metadata.doi == "10.1234/visible"
    assert "VAL_EXPECTED_ID_MISMATCH" in {issue.code for issue in state.paper.validation_issues}


async def test_identity_stage_clears_unrepresented_metadata_doi_without_manifest():
    from bibr.pipeline.stages.identity import IdentityValidationStage

    state = FileState(
        path=Path("paper.pdf"),
        paper=_paper("A paper with no source-visible DOI", doi="10.1234/stale"),
    )

    await IdentityValidationStage().run(_ctx(state))

    assert state.doi_selection is not None
    assert state.doi_selection.selected is None
    assert state.paper is not None and state.paper.metadata is not None
    assert state.paper.metadata.doi == ""


async def test_identity_stage_exports_additive_expected_identity_and_candidate_receipt():
    from bibr.pipeline.stages.export import _build_extraction
    from bibr.pipeline.stages.identity import IdentityValidationStage

    doi = "10.1234/paper"
    expected = ExpectedIdentity(
        queue_record_id="record-1",
        expected_doi=doi,
        expected_doi_sha256=hashlib.sha256(doi.encode()).hexdigest(),
        source_sha256="a" * 64,
        doi_required=True,
    )
    state = FileState(
        path=Path("paper.pdf"),
        expected_identity=expected,
        content_sha256="a" * 64,
        paper=_paper(f"Article DOI: {doi}", doi=""),
    )
    ctx = _ctx(state)

    await IdentityValidationStage().run(ctx)
    state.paper.extraction = _build_extraction(ctx, state.paper, state)
    payload = state.paper.export_to_json()

    assert payload["schema_version"] == "11.0"
    assert payload["metadata"]["doi"] == doi
    identity = payload["extraction"]["identity"]
    assert identity["expected"]["queue_record_id"] == "record-1"
    receipt = identity["receipt"]
    assert receipt["selected"]["normalized"] == doi
    assert receipt["selected"]["selection_tier"] == 4
    assert receipt["candidates"]


def test_identity_stage_is_immediately_after_post_parse_in_local_and_serve_plans():
    from bibr.pipeline.plans import build_stage_plan

    for mode in ("local", "serve"):
        stages = build_stage_plan(mode=mode, stream_backhalf=False, enrichers=[])
        names = [stage.name for stage in stages]
        assert names[names.index("extract") + 1] == "identity"


async def test_streaming_backhalf_runs_identity_immediately_after_post_parse():
    from bibr.pipeline.stages.render_ocr import StreamingRenderOcrStage

    events = []

    class FakeStage:
        def __init__(self, name):
            self.name = name

        async def run(self, ctx):
            events.append(self.name)

    state = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    ctx = _ctx(state)
    stage = StreamingRenderOcrStage(
        parse=FakeStage("parse"),
        post_parse=FakeStage("extract"),
        identity=FakeStage("identity"),
        enrich=FakeStage("enrich"),
        export=FakeStage("export"),
        layout=FakeStage("layout"),
        native_text=FakeStage("native_text"),
        ocr=FakeStage("ocr"),
    )

    await stage._run_backhalf(ctx, [state], asyncio.Semaphore(1))

    assert events == ["parse", "extract", "identity", "enrich", "export"]
