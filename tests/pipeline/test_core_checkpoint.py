from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState
from bibr.validation import ValidationIssue


def _payload(*, promotable: bool = True) -> dict:
    issues = []
    if not promotable:
        issues = [
            {
                "code": "VAL_EXPECTED_ID_MISSING",
                "severity": "error",
                "message": "missing",
                "origin_stage": "identity",
                "evidence_ids": ["queue-1"],
                "count": 1,
                "blocking": True,
            }
        ]
    return {
        "schema_version": "11.0",
        "metadata": {"title": "Core"},
        "bib": [],
        "bib_match": [],
        "metadata_match": [],
        # A pipeline-produced core always carries ``extraction``; enrichment
        # replay now rejects a core without it rather than dropping the
        # completeness receipt silently.
        "extraction": {
            "bibr_version": "0.0.0-test",
            "completed_at": "2026-07-24T10:00:00Z",
            "settings": {
                "ref_seg": "geom",
                "ref_parse": "ner",
                "crossref_enrich": True,
                "consolidate": "off",
            },
            "warnings": [],
        },
        "validation": {
            "errors": len(issues),
            "warnings": 0,
            "blocking": len(issues),
            "promotable": promotable,
            "issues": issues,
        },
    }


def _ctx(fs: FileState, *, config: RunConfig | None = None) -> PipelineContext:
    # Enrichment is opt-in (CROSSREF_ENRICH off by default); the checkpoint
    # tests below exercise the "enrichment requested" gate, so force it on
    # per run unless a test supplies its own config.
    return PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=config or RunConfig(crossref=True),
    )


def _schema_valid_paper(*, issues=()):
    from bibr.input.file import InputFile, InputFormat
    from bibr.models import PaperMetadata
    from bibr.paper import Paper
    from bibr.paper_contents import PaperContents, PaperSection, PaperSentence

    input_file = InputFile(
        path=Path("paper.pdf"),
        file_hash="abc123",
        input_format=InputFormat(
            file_extension=".pdf",
            detected_mime_type="application/pdf",
            file_type="pdf",
        ),
    )
    contents = PaperContents(
        sentences=[PaperSentence(text_id=1, text="Body.", section_id=1, paragraph_id=1)],
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ],
        tables=[],
        links=[],
        sections_text={1: "Body."},
    )
    return Paper(
        input_file=input_file,
        metadata=PaperMetadata(doi="10.1/test", title="Core"),
        contents=contents,
        validation_issues=list(issues),
    )


async def test_core_checkpoint_writes_unenriched_payload_before_enrichment(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import ArtifactDisposition, RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    fs = FileState(path=tmp_path / "paper.pdf")
    fs.paper = MagicMock()
    fs.paper.validation_issues = []
    fs.paper.export_to_json.return_value = _payload()
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")

    await CoreCheckpointStage().run(_ctx(fs))

    assert fs.artifact_disposition is ArtifactDisposition.PROMOTABLE
    assert fs.core_sha256
    written = json.loads((tmp_path / "paper.json").read_text(encoding="utf-8"))
    assert written["bib_match"] == []
    assert written["metadata_match"] == []
    receipt = json.loads(fs.artifact_sink.receipt_path(fs).read_text(encoding="utf-8"))
    assert [event["state"] for event in receipt["events"]] == [
        RunState.STARTED,
        RunState.CORE_WRITTEN,
    ]
    assert fs.paper is not None


async def test_checkpoint_interruption_keeps_quarantined_core_readable_and_retryable(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import ArtifactDisposition
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.enrich import EnrichmentStage

    fs = FileState(path=tmp_path / "paper.pdf")
    issue = ValidationIssue(
        "VAL_EXPECTED_ID_MISSING",
        "error",
        "missing",
        origin_stage="identity",
        blocking=True,
    )
    fs.paper = _schema_valid_paper(issues=[issue])
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = _ctx(fs)

    await CoreCheckpointStage(enrichment_requested=True).run(ctx)

    enricher = MagicMock()
    enricher.enrich = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await EnrichmentStage([enricher]).run(ctx)

    assert fs.artifact_disposition is ArtifactDisposition.IDENTITY_CONFLICT
    destination = fs.artifact_sink.destination_path(fs)
    core = json.loads(destination.read_text(encoding="utf-8"))
    from bibr.export.json_export import validate_export

    assert validate_export(core) == []
    assert core["schema_version"] == "11.0"
    assert core["validation"]["promotable"] is False
    receipt = json.loads(fs.artifact_sink.receipt_path(fs).read_text(encoding="utf-8"))
    assert receipt["events"][-1]["state"] == "cutoff_interrupted"
    assert receipt["retryable"] is True


async def test_requested_enrichment_checkpoint_adds_pending_gate_without_changing_disposition(
    tmp_path,
):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import ArtifactDisposition
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    fs = FileState(path=tmp_path / "paper.pdf")
    fs.paper = MagicMock(validation_issues=[])
    fs.paper.export_to_json.return_value = _payload(promotable=True)
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")

    await CoreCheckpointStage(enrichment_requested=True).run(_ctx(fs))

    core = json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8"))
    assert fs.artifact_disposition is ArtifactDisposition.PROMOTABLE
    assert core["validation"]["promotable"] is False
    assert any(issue["code"] == "VAL_ENRICHMENT_PENDING" for issue in core["validation"]["issues"])


def test_local_plans_checkpoint_after_identity_before_enrichment():
    from bibr.pipeline.plans import build_stage_plan
    from bibr.pipeline.stages.render_ocr import StreamingRenderOcrStage

    barrier = build_stage_plan(mode="local", stream_backhalf=False, enrichers=[])
    names = [stage.name for stage in barrier]
    assert names.index("identity") < names.index("core_checkpoint") < names.index("enrich")

    streaming = build_stage_plan(mode="local", stream_backhalf=True, enrichers=[])
    composite = next(stage for stage in streaming if isinstance(stage, StreamingRenderOcrStage))
    assert composite._identity.name == "identity"
    assert composite._checkpoint.name == "core_checkpoint"


async def test_export_writes_atomic_sidecar_and_replays_enrichment(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState, enrichment_settings_digest
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    enriched = copy.deepcopy(core)
    enriched.update(
        {
            "bib_match": [{"bib_id": 1, "service": "crossref", "doi": "10.1/ref"}],
            "metadata_match": [{"service": "crossref", "doi": "10.1/self"}],
            "enrichment": {"complete": True, "refs_enriched": 1, "refs_total": 1},
        }
    )
    core["bib"] = [{"bib_id": 1}]
    enriched["bib"] = [{"bib_id": 1}]
    fs = FileState(path=tmp_path / "paper.pdf")
    fs.paper = MagicMock()
    fs.paper.validation_issues = []
    fs.paper.export_to_json.side_effect = [core, enriched]
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = _ctx(fs, config=RunConfig(crossref=True, consolidate="off"))

    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE
    await ExportStage().run(ctx)

    assert fs.error is None
    assert fs.result_json["bib_match"] == enriched["bib_match"]
    assert fs.result_json["metadata_match"] == enriched["metadata_match"]
    assert fs.result_json["validation"]["promotable"] is True
    assert not any(
        issue["code"] == "VAL_ENRICHMENT_PENDING"
        for issue in fs.result_json["validation"]["issues"]
    )
    sidecar = json.loads(fs.artifact_sink.sidecar_path(fs).read_text(encoding="utf-8"))
    assert sidecar["core_sha256"] == fs.core_sha256
    assert sidecar["settings_digest"] == enrichment_settings_digest(ctx)
    assert sidecar["completeness"] == "complete"
    assert set(sidecar) == {
        "schema_version",
        "core_sha256",
        "settings_digest",
        "completeness",
        "bib_match",
        "metadata_match",
    }
    receipt = json.loads(fs.artifact_sink.receipt_path(fs).read_text(encoding="utf-8"))
    assert receipt["events"][-1]["state"] == "enrichment_complete"
    immutable = json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8"))
    assert immutable["bib_match"] == []
    assert immutable["validation"]["promotable"] is False
    assert (
        json.loads(fs.artifact_sink.destination_path(fs).read_text(encoding="utf-8"))
        == fs.result_json
    )

    from bibr.local.cli import _write_chunk_results

    _write_chunk_results(
        [fs],
        output_path=tmp_path / "paper.json",
        json_kwargs={"indent": 2, "ensure_ascii": False},
        console=MagicMock(),
        is_batch=False,
        total_files=1,
        total_t0=0.0,
    )
    materialized = json.loads(fs.artifact_sink.destination_path(fs).read_text(encoding="utf-8"))
    assert materialized["bib_match"] == enriched["bib_match"]
    assert json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8")) == immutable


async def test_sidecar_write_failure_keeps_checkpoint_core_valid(tmp_path, monkeypatch):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload(promotable=False)
    enriched = {**core, "metadata_match": [{"service": "crossref", "doi": "10.1/self"}]}
    fs = FileState(path=tmp_path / "paper.pdf")
    fs.paper = MagicMock()
    fs.paper.validation_issues = [
        ValidationIssue("VAL_EXPECTED_ID_MISSING", "error", "missing", blocking=True)
    ]
    fs.paper.export_to_json.side_effect = [core, enriched]
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = _ctx(fs)
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    checkpoint = json.loads(fs.artifact_sink.destination_path(fs).read_text(encoding="utf-8"))
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE

    def fail_sidecar(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fs.artifact_sink, "write_enrichment", fail_sidecar)
    await ExportStage().run(ctx)

    assert fs.error is None
    assert fs.result_json == checkpoint
    assert (
        json.loads(fs.artifact_sink.destination_path(fs).read_text(encoding="utf-8")) == checkpoint
    )
    receipt = json.loads(fs.artifact_sink.receipt_path(fs).read_text(encoding="utf-8"))
    assert receipt["events"][-1]["state"] == "enrichment_partial"


async def test_terminal_receipt_is_recorded_after_public_materialization(tmp_path, monkeypatch):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    enriched = copy.deepcopy(core)
    enriched["metadata_match"] = [{"service": "crossref", "doi": "10.1/self"}]
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.side_effect = [core, enriched]
    sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.artifact_sink = sink
    ctx = _ctx(fs)
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE

    real_record = sink.record

    def assert_materialized_before_terminal(state_fs, state, **kwargs):
        if state == RunState.ENRICHMENT_COMPLETE:
            durable = json.loads(sink.destination_path(state_fs).read_text())
            assert durable["metadata_match"] == enriched["metadata_match"]
            assert durable == state_fs.result_json
        return real_record(state_fs, state, **kwargs)

    monkeypatch.setattr(sink, "record", assert_materialized_before_terminal)
    await ExportStage().run(ctx)

    receipt = json.loads(sink.receipt_path(fs).read_text())
    assert receipt["events"][-1]["state"] == "enrichment_complete"


async def test_final_materialization_failure_preserves_checkpoint_and_retryable_receipt(
    tmp_path, monkeypatch
):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    enriched = copy.deepcopy(core)
    enriched["metadata_match"] = [{"service": "crossref", "doi": "10.1/self"}]
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.side_effect = [core, enriched]
    sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.artifact_sink = sink
    ctx = _ctx(fs)
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    checkpoint = json.loads(sink.destination_path(fs).read_text())
    immutable = sink.core_path(fs).read_bytes()
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE

    monkeypatch.setattr(sink, "materialize", MagicMock(side_effect=OSError("disk full")))
    await ExportStage().run(ctx)

    assert fs.error is None
    assert fs.result_json == checkpoint
    assert json.loads(sink.destination_path(fs).read_text()) == checkpoint
    assert sink.core_path(fs).read_bytes() == immutable
    receipt = json.loads(sink.receipt_path(fs).read_text())
    assert receipt["events"][-1]["state"] == "enrichment_partial"
    assert receipt["retryable"] is True


async def test_disposition_comes_from_serialized_output_validation_before_pending(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import ArtifactDisposition
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    payload = _payload()
    payload["validation"] = {
        "errors": 1,
        "warnings": 0,
        "blocking": 1,
        "promotable": False,
        "issues": [
            {
                "code": "VAL_OUTPUT_ONLY_BLOCKER",
                "severity": "error",
                "message": "output-only",
                "origin_stage": "export",
                "evidence_ids": [],
                "count": 1,
                "blocking": True,
            }
        ],
    }
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.return_value = payload
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")

    await CoreCheckpointStage(enrichment_requested=True).run(_ctx(fs))

    assert fs.artifact_disposition is ArtifactDisposition.BLOCKED
    assert fs.artifact_sink.destination_path(fs).parts[-3:] == (
        "_quarantine",
        "blocked",
        "paper.json",
    )


async def test_corrupt_durable_core_never_becomes_publishable_result(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink, atomic_write_json
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    enriched = copy.deepcopy(core)
    enriched["metadata_match"] = [{"service": "crossref", "doi": "10.1/self"}]
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.side_effect = [core, enriched]
    sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.artifact_sink = sink
    ctx = _ctx(fs)
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    verified_checkpoint = copy.deepcopy(fs.result_json)
    durable_public = sink.destination_path(fs).read_bytes()
    corrupted = copy.deepcopy(verified_checkpoint)
    corrupted["metadata"]["title"] = "CORRUPTED"
    atomic_write_json(sink.core_path(fs), corrupted)
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE

    await ExportStage().run(ctx)

    assert fs.result_json == verified_checkpoint
    assert sink.destination_path(fs).read_bytes() == durable_public
    assert fs.result_json["metadata"]["title"] != "CORRUPTED"


async def test_export_replays_sidecar_read_back_from_disk(tmp_path, monkeypatch):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    enriched = copy.deepcopy(core)
    enriched["metadata_match"] = [{"service": "crossref", "doi": "10.1/self"}]
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.side_effect = [core, enriched]
    sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.artifact_sink = sink
    ctx = _ctx(fs)
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    checkpoint = copy.deepcopy(fs.result_json)
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE

    monkeypatch.setattr(
        sink,
        "read_enrichment",
        MagicMock(return_value={"schema_version": "corrupt-on-disk"}),
        raising=False,
    )
    await ExportStage().run(ctx)

    assert fs.result_json == checkpoint
    receipt = json.loads(sink.receipt_path(fs).read_text())
    assert receipt["events"][-1]["state"] == "enrichment_partial"


@pytest.mark.parametrize(
    ("bib_rows", "metadata_rows"),
    [
        (({"bib_id": 1, "service": "crossref", "score": "bad"},), ()),
        (({"bib_id": 1, "service": "crossref", "authors": "bad"},), ()),
        ((), ({"service": "crossref", "unknown_field": "bad"},)),
    ],
)
async def test_malformed_typed_durable_sidecar_fails_closed_before_publication(
    tmp_path, monkeypatch, bib_rows, metadata_rows
):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import (
        ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        EnrichmentSidecar,
        RunState,
        enrichment_settings_digest,
    )
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    core["bib"] = [{"bib_id": 1}]
    enriched = copy.deepcopy(core)
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.side_effect = [core, enriched]
    sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.artifact_sink = sink
    ctx = _ctx(fs)
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    checkpoint = copy.deepcopy(fs.result_json)
    public_before = sink.destination_path(fs).read_bytes()
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE
    # The current sidecar version, deliberately: this test must reach the typed
    # row validation, not short-circuit on the sidecar version check.
    malicious = EnrichmentSidecar(
        schema_version=ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        core_sha256=fs.core_sha256,
        settings_digest=enrichment_settings_digest(ctx),
        completeness="complete",
        bib_match=bib_rows,
        metadata_match=metadata_rows,
    )
    monkeypatch.setattr(sink, "read_enrichment", MagicMock(return_value=malicious))

    await ExportStage().run(ctx)

    assert fs.error is None
    assert fs.result_json == checkpoint
    assert sink.destination_path(fs).read_bytes() == public_before
    receipt = json.loads(sink.receipt_path(fs).read_text())
    assert receipt["events"][-1]["state"] == "enrichment_partial"
    assert all(event["state"] != "enrichment_complete" for event in receipt["events"])
    assert receipt["retryable"] is True


@pytest.mark.parametrize(
    ("enrichment_requested", "config"),
    [
        (False, RunConfig(crossref=True)),
        (True, RunConfig(crossref=True, ref_parse_strategy="off")),
        (True, RunConfig(crossref=False)),
        (True, RunConfig()),  # crossref=None follows CROSSREF_ENRICH (off)
    ],
)
async def test_authoritative_no_enrichment_core_skips_second_serialization(
    tmp_path, enrichment_requested, config
):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.side_effect = [core, AssertionError("serialized twice")]
    sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.artifact_sink = sink
    ctx = _ctx(fs, config=config)

    await CoreCheckpointStage(enrichment_requested=enrichment_requested).run(ctx)
    checkpoint = copy.deepcopy(fs.result_json)
    public_before = sink.destination_path(fs).read_bytes()
    await ExportStage().run(ctx)

    assert fs.paper is None
    assert fs.error is None
    assert fs.result_json == checkpoint
    assert sink.destination_path(fs).read_bytes() == public_before
    receipt = json.loads(sink.receipt_path(fs).read_text())
    assert receipt["events"][-1] == {
        "state": "core_written",
        "detail": "enrichment_not_requested",
    }
    assert receipt["retryable"] is False


@pytest.mark.parametrize("mode", ["fill", "replace"])
async def test_sink_backed_consolidation_is_deterministic_by_bib_id_and_service(tmp_path, mode):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _payload()
    core["bib"] = [
        {"bib_id": 2, "doi": "printed", "volume": "1"},
        {"bib_id": 1, "doi": None},
    ]
    enriched = copy.deepcopy(core)
    enriched["bib_match"] = [
        {"bib_id": 1, "service": "other", "doi": "other"},
        {"bib_id": 2, "service": "crossref", "doi": "printed", "volume": "2"},
        {"bib_id": 1, "service": "crossref", "doi": "preferred"},
    ]
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.side_effect = [core, enriched]
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = _ctx(fs, config=RunConfig(crossref=True, consolidate=mode))
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE

    await ExportStage().run(ctx)

    by_id = {row["bib_id"]: row for row in fs.result_json["bib"]}
    assert by_id[1]["doi"] == "preferred"
    assert by_id[2]["doi"] == "printed"
    assert by_id[2]["volume"] == ("2" if mode == "replace" else "1")


def test_sink_bound_chunk_writer_reports_without_rewriting(tmp_path, monkeypatch):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.local.cli import _write_chunk_results

    fs = FileState(path=tmp_path / "paper.pdf", result_json=_payload())
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.core_sha256 = "a" * 64
    fs.artifact_sink.destination_path(fs).write_text('{"durable":true}')
    monkeypatch.setattr(
        "bibr.local.artifacts.atomic_write_json",
        MagicMock(side_effect=AssertionError("CLI must not rewrite sink output")),
    )

    processed, errors = _write_chunk_results(
        [fs],
        output_path=tmp_path / "paper.json",
        json_kwargs={"indent": 2},
        console=MagicMock(),
        is_batch=False,
        total_files=1,
        total_t0=0,
    )

    assert (processed, errors) == (1, 0)
    assert fs.artifact_sink.destination_path(fs).read_text() == '{"durable":true}'


async def test_immutable_core_identity_survives_public_checkpoint_failure(tmp_path, monkeypatch):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.return_value = _payload()
    sink = LocalArtifactSink(tmp_path / "paper.json")
    fs.artifact_sink = sink
    monkeypatch.setattr(sink, "materialize", MagicMock(side_effect=OSError("public failed")))

    await CoreCheckpointStage().run(_ctx(fs))

    assert fs.core_sha256
    assert sink.core_path(fs).is_file()
    receipt = json.loads(sink.receipt_path(fs).read_text())
    assert any(event["state"] == "core_written" for event in receipt["events"])
    assert receipt["retryable"] is True


@pytest.mark.parametrize(
    ("enrichment_requested", "config"),
    [
        (False, RunConfig(crossref=True)),
        (True, RunConfig(crossref=True, ref_parse_strategy="off")),
    ],
)
async def test_no_enrichment_or_refs_off_checkpoint_is_not_pending(
    tmp_path, enrichment_requested, config
):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.return_value = _payload()
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")

    await CoreCheckpointStage(enrichment_requested=enrichment_requested).run(
        _ctx(fs, config=config)
    )

    assert fs.result_json["validation"]["promotable"] is True
    assert not any(
        issue["code"] == "VAL_ENRICHMENT_PENDING"
        for issue in fs.result_json["validation"]["issues"]
    )
    receipt = json.loads(fs.artifact_sink.receipt_path(fs).read_text())
    assert receipt["retryable"] is False


@pytest.mark.parametrize(
    ("setting", "config", "pending"),
    [
        (False, RunConfig(), False),
        (True, RunConfig(), True),
        (False, RunConfig(crossref=True), True),
        (True, RunConfig(crossref=False), False),
    ],
)
async def test_pending_gate_follows_effective_enrichment_switch(tmp_path, setting, config, pending):
    """The checkpoint's pending gate resolves the tri-state ``crossref`` per run."""
    from bibr.config import GlobalSettings
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    settings = GlobalSettings()
    settings.crossref.enrich = setting
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.return_value = _payload(promotable=True)
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=config,
        settings=settings,
    )

    await CoreCheckpointStage(enrichment_requested=True).run(ctx)

    core = json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8"))
    has_gate = any(
        issue["code"] == "VAL_ENRICHMENT_PENDING" for issue in core["validation"]["issues"]
    )
    assert has_gate is pending
    receipt = json.loads(fs.artifact_sink.receipt_path(fs).read_text(encoding="utf-8"))
    detail = receipt["events"][-1].get("detail")
    assert (detail == "enrichment_not_requested") is (not pending)


@pytest.mark.parametrize(("refs", "pending"), [(None, False), ("ner", True), ("off", False)])
async def test_checkpoint_pending_gate_obeys_request_override_of_refs_off(tmp_path, refs, pending):
    from bibr.config import GlobalSettings
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    settings = GlobalSettings()
    settings.REF_PARSE_STRATEGY = "off"
    fs = FileState(path=tmp_path / "paper.pdf", paper=MagicMock(validation_issues=[]))
    fs.paper.export_to_json.return_value = _payload(promotable=True)
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = PipelineContext(
        [fs],
        NullProgress(),
        MagicMock(),
        RunConfig(crossref=True, ref_parse_strategy=refs),
        settings,
    )
    await CoreCheckpointStage(enrichment_requested=True).run(ctx)
    core = json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8"))
    assert (
        any(issue["code"] == "VAL_ENRICHMENT_PENDING" for issue in core["validation"]["issues"])
        is pending
    )
