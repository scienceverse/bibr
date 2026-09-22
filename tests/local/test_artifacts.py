from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _core_payload(*, title: str = "Café") -> dict:
    return {
        "schema_version": "12.0",
        "metadata": {"title": title},
        "bib": [{"bib_id": 1}],
        "bib_match": [],
        "metadata_match": [],
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
    }


def test_canonical_json_hash_is_order_independent_and_utf8():
    from bibr.pipeline.artifacts import canonical_json_bytes, canonical_json_sha256

    left = {"z": "Café", "a": [2, 1]}
    right = {"a": [2, 1], "z": "Café"}

    assert canonical_json_bytes(left) == b'{"a":[2,1],"z":"Caf\xc3\xa9"}'
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_atomic_writer_uses_same_directory_temp_fsync_and_replace(tmp_path, monkeypatch):
    from bibr.local.artifacts import atomic_write_json

    destination = tmp_path / "nested" / "paper.json"
    replace_calls: list[tuple[Path, Path]] = []
    fsync_calls: list[int] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def recording_replace(source, target):
        replace_calls.append((Path(source), Path(target)))
        real_replace(source, target)

    def recording_fsync(fd):
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "replace", recording_replace)
    monkeypatch.setattr(os, "fsync", recording_fsync)

    atomic_write_json(destination, {"title": "Café"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"title": "Café"}
    assert len(replace_calls) == 1
    temp_path, target = replace_calls[0]
    assert temp_path.parent == destination.parent
    assert target == destination
    # One fsync for the flushed file and one for the containing directory on
    # filesystems that support directory descriptors.
    assert len(fsync_calls) >= 1


def test_atomic_writer_preserves_prior_destination_on_pre_replace_failure(tmp_path):
    from bibr.local.artifacts import atomic_write_json

    destination = tmp_path / "paper.json"
    destination.write_text('{"old":true}', encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json(destination, {"bad": object()})

    assert destination.read_text(encoding="utf-8") == '{"old":true}'
    assert list(tmp_path.glob(".paper.json.*.tmp")) == []


def test_disposition_precedence_prefers_source_then_identity_then_references():
    from bibr.pipeline.artifacts import ArtifactDisposition, disposition_for_issues
    from bibr.validation import ValidationIssue

    refs = ValidationIssue("VAL_REFERENCES_INCOMPLETE", "error", "refs", blocking=True)
    identity = ValidationIssue("VAL_EXPECTED_ID_MISMATCH", "error", "doi", blocking=True)
    source = ValidationIssue("VAL_SOURCE_INTEGRITY", "error", "sha", blocking=True)

    assert disposition_for_issues([]) is ArtifactDisposition.PROMOTABLE
    assert disposition_for_issues([refs]) is ArtifactDisposition.REFERENCES_INCOMPLETE
    assert disposition_for_issues([refs, identity]) is ArtifactDisposition.IDENTITY_CONFLICT
    assert disposition_for_issues([identity, source, refs]) is ArtifactDisposition.SOURCE_INTEGRITY


def test_unknown_blocking_issue_fails_closed():
    from bibr.pipeline.artifacts import ArtifactDisposition, disposition_for_issues
    from bibr.validation import ValidationIssue

    unknown = ValidationIssue("VAL_FUTURE_BLOCKER", "error", "future", blocking=True)
    assert disposition_for_issues([unknown]) is ArtifactDisposition.BLOCKED


def test_sidecar_replay_requires_core_settings_and_schema_match():
    from bibr.pipeline.artifacts import (
        ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        ArtifactReplayError,
        EnrichmentSidecar,
        canonical_json_sha256,
        replay_enrichment_sidecar,
    )

    core = _core_payload()
    core_hash = canonical_json_sha256(core)
    sidecar = EnrichmentSidecar(
        schema_version=ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        core_sha256=core_hash,
        settings_digest="settings-v1",
        completeness="complete",
        bib_match=({"bib_id": 1, "service": "crossref", "doi": "10.1/ref"},),
        metadata_match=({"service": "crossref", "doi": "10.1/self"},),
    )

    replayed = replay_enrichment_sidecar(core, sidecar, expected_settings_digest="settings-v1")
    assert replayed["bib_match"] == list(sidecar.bib_match)
    assert replayed["metadata_match"] == list(sidecar.metadata_match)
    assert replayed["extraction"]["enrichment"] == {
        "complete": True,
        "refs_enriched": 1,
        "refs_total": 1,
    }
    assert core["bib_match"] == []

    mismatches = [
        (sidecar, "settings-v2"),
        (
            EnrichmentSidecar(
                schema_version="999",
                core_sha256=core_hash,
                settings_digest="settings-v1",
                completeness="complete",
            ),
            "settings-v1",
        ),
        (
            EnrichmentSidecar(
                schema_version=ENRICHMENT_SIDECAR_SCHEMA_VERSION,
                core_sha256="0" * 64,
                settings_digest="settings-v1",
                completeness="complete",
            ),
            "settings-v1",
        ),
    ]
    for invalid, expected_digest in mismatches:
        with pytest.raises(ArtifactReplayError):
            replay_enrichment_sidecar(core, invalid, expected_settings_digest=expected_digest)


def test_local_sink_routes_blocking_core_and_receipts_to_quarantine(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import ArtifactDisposition, RunState
    from bibr.pipeline.state import FileState

    fs = FileState(path=tmp_path / "paper.pdf")
    fs.artifact_disposition = ArtifactDisposition.IDENTITY_CONFLICT
    sink = LocalArtifactSink(tmp_path / "results" / "paper.json")

    sink.record(fs, RunState.STARTED)
    fs.core_sha256 = sink.write_core(fs, _core_payload())
    sink.materialize(fs, _core_payload())
    sink.record(fs, RunState.CORE_WRITTEN)

    destination = tmp_path / "results" / "_quarantine" / "identity_conflict" / "paper.json"
    assert json.loads(destination.read_text(encoding="utf-8"))["metadata"]["title"] == "Café"
    immutable_core = sink.core_path(fs)
    assert json.loads(immutable_core.read_text(encoding="utf-8"))["metadata"]["title"] == "Café"
    receipt = json.loads(sink.receipt_path(fs).read_text(encoding="utf-8"))
    assert [event["state"] for event in receipt["events"]] == ["started", "core_written"]
    assert receipt["disposition"] == "identity_conflict"
    assert receipt["retryable"] is True


async def test_chunk_processor_binds_sink_only_for_explicit_file_outputs(tmp_path):
    from bibr.local.cli import ChunkProcessor

    seen = []
    pipeline = type("Pipeline", (), {})()

    async def process_chunk(states, **_kwargs):
        seen.extend(states)

    pipeline.process_chunk = process_chunk
    explicit = ChunkProcessor(
        pipeline=pipeline,
        paper_id=None,
        is_batch=False,
        active_stages=[],
        output_path=tmp_path / "paper.json",
    )
    await explicit.run([tmp_path / "paper.pdf"], chunk_index=1, total_chunks=1, console=object())
    assert seen[-1].artifact_sink is not None
    receipt = json.loads(seen[-1].artifact_sink.receipt_path(seen[-1]).read_text())
    assert receipt["events"][0]["state"] == "started"

    stdout = ChunkProcessor(
        pipeline=pipeline,
        paper_id=None,
        is_batch=False,
        active_stages=[],
        output_path=None,
    )
    await stdout.run([tmp_path / "stdout.pdf"], chunk_index=1, total_chunks=1, console=object())
    assert seen[-1].artifact_sink is None


def test_chunk_writer_uses_quarantine_destination_for_blocking_payload(tmp_path):
    from unittest.mock import MagicMock

    from bibr.local.artifacts import LocalArtifactSink
    from bibr.local.cli import _write_chunk_results
    from bibr.pipeline.artifacts import ArtifactDisposition
    from bibr.pipeline.state import FileState

    fs = FileState(path=tmp_path / "paper.pdf", result_json=_core_payload())
    fs.artifact_disposition = ArtifactDisposition.REFERENCES_INCOMPLETE
    fs.artifact_sink = LocalArtifactSink(tmp_path / "out" / "paper.json")
    fs.artifact_sink.materialize(fs, fs.result_json)

    _write_chunk_results(
        [fs],
        output_path=tmp_path / "out" / "paper.json",
        json_kwargs={"indent": 2, "ensure_ascii": False},
        console=MagicMock(),
        is_batch=False,
        total_files=1,
        total_t0=0.0,
    )

    quarantined = tmp_path / "out" / "_quarantine" / "references_incomplete" / "paper.json"
    assert json.loads(quarantined.read_text(encoding="utf-8"))["metadata"]["title"] == "Café"
    assert not (tmp_path / "out" / "paper.json").exists()


def test_receipt_transitions_are_idempotent_monotonic_and_survive_restart(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import ArtifactDisposition, RunState
    from bibr.pipeline.state import FileState

    destination = tmp_path / "paper.json"
    first_fs = FileState(path=tmp_path / "paper.pdf")
    first = LocalArtifactSink(destination)
    first.record(first_fs, RunState.STARTED)
    first.record(first_fs, RunState.STARTED)
    first.record(first_fs, RunState.CORE_WRITTEN)
    first.record(first_fs, RunState.ENRICHMENT_COMPLETE)
    first.record(first_fs, RunState.CUTOFF_INTERRUPTED)
    first.record(first_fs, RunState.CUTOFF_INTERRUPTED)

    first_receipt = json.loads(first.receipt_path(first_fs).read_text())
    assert [event["state"] for event in first_receipt["attempts"][0]["events"]] == [
        "started",
        "core_written",
        "enrichment_complete",
    ]

    second_fs = FileState(path=tmp_path / "paper.pdf")
    second_fs.artifact_disposition = ArtifactDisposition.IDENTITY_CONFLICT
    second = LocalArtifactSink(destination)
    second.record(second_fs, RunState.STARTED)
    second.record(second_fs, RunState.CORE_WRITTEN, detail="enrichment_not_requested")

    restarted = json.loads(second.receipt_path(second_fs).read_text())
    assert len(restarted["attempts"]) == 2
    assert restarted["attempts"][0]["attempt_id"] != restarted["attempts"][1]["attempt_id"]
    assert restarted["attempts"][0]["events"] == first_receipt["attempts"][0]["events"]
    # Quarantined artifacts remain retryable even after a terminal complete or
    # enrichment-not-requested attempt.
    assert restarted["retryable"] is True


def test_blocked_completed_receipt_remains_retryable(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import ArtifactDisposition, RunState
    from bibr.pipeline.state import FileState

    fs = FileState(path=tmp_path / "paper.pdf")
    fs.artifact_disposition = ArtifactDisposition.BLOCKED
    sink = LocalArtifactSink(tmp_path / "paper.json")
    sink.record(fs, RunState.STARTED)
    sink.record(fs, RunState.CORE_WRITTEN)
    sink.record(fs, RunState.ENRICHMENT_COMPLETE)

    assert json.loads(sink.receipt_path(fs).read_text())["retryable"] is True


@pytest.mark.parametrize(
    "rows",
    [
        (
            {"bib_id": 1, "service": "crossref"},
            {"bib_id": 1, "service": "crossref"},
        ),
        ({"bib_id": 99, "service": "crossref"},),
        ({"bib_id": 1},),
        ("not-a-row",),
    ],
)
def test_sidecar_rejects_duplicate_malformed_and_unknown_bibliography_rows(rows):
    from bibr.pipeline.artifacts import (
        ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        ArtifactReplayError,
        EnrichmentSidecar,
        canonical_json_sha256,
        replay_enrichment_sidecar,
    )

    core = _core_payload()
    sidecar = EnrichmentSidecar(
        schema_version=ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        core_sha256=canonical_json_sha256(core),
        settings_digest="settings",
        completeness="complete",
        bib_match=rows,
    )
    with pytest.raises(ArtifactReplayError):
        replay_enrichment_sidecar(core, sidecar, expected_settings_digest="settings")


def test_sidecar_rejects_duplicate_or_malformed_metadata_service_rows():
    from bibr.pipeline.artifacts import (
        ENRICHMENT_SIDECAR_SCHEMA_VERSION,
        ArtifactReplayError,
        EnrichmentSidecar,
        canonical_json_sha256,
        replay_enrichment_sidecar,
    )

    core = _core_payload()
    for rows in (
        ({"service": "crossref"}, {"service": "crossref"}),
        ({"doi": "10.1/no-service"},),
        ("not-a-row",),
    ):
        sidecar = EnrichmentSidecar(
            schema_version=ENRICHMENT_SIDECAR_SCHEMA_VERSION,
            core_sha256=canonical_json_sha256(core),
            settings_digest="settings",
            completeness="complete",
            metadata_match=rows,
        )
        with pytest.raises(ArtifactReplayError):
            replay_enrichment_sidecar(core, sidecar, expected_settings_digest="settings")


def test_sidecar_replay_rejects_a_core_without_an_extraction_block():
    """v11 hangs the completeness receipt and the enrichment diagnostics off
    ``extraction``. A core lacking it would replay bib_match fine while both
    vanish without trace — a partial enrichment reading as clean. Fail the
    replay instead, so the caller falls back to the verified checkpoint."""
    from bibr.pipeline.artifacts import (
        ArtifactReplayError,
        canonical_json_sha256,
        make_enrichment_sidecar,
        replay_enrichment_sidecar,
    )

    core = _core_payload()
    core.pop("extraction")
    sidecar = make_enrichment_sidecar(
        core,
        core_sha256=canonical_json_sha256(core),
        settings_digest="settings",
        completeness="partial",
        warnings=("transport failure",),
    )

    with pytest.raises(ArtifactReplayError, match="no extraction block"):
        replay_enrichment_sidecar(core, sidecar, expected_settings_digest="settings")


def test_partial_sidecar_persists_bounded_diagnostics_and_replay_restores_them():
    from bibr.pipeline.artifacts import (
        canonical_json_sha256,
        make_enrichment_sidecar,
        replay_enrichment_sidecar,
    )

    core = _core_payload()
    enriched = dict(core)
    enriched["bib_match"] = [{"bib_id": 1, "service": "crossref", "doi": "10.1/ref"}]
    sidecar = make_enrichment_sidecar(
        enriched,
        core_sha256=canonical_json_sha256(core),
        settings_digest="settings",
        completeness="partial",
        warnings=("transport failure " + "x" * 2_000,),
        detail="1 of 1 terminal requests failed " + "y" * 2_000,
    )

    encoded = sidecar.to_dict()
    assert len(encoded["warnings"][0]) <= 512
    assert len(encoded["detail"]) <= 512
    assert encoded["warnings"][0].startswith("transport failure")

    replayed = replay_enrichment_sidecar(core, sidecar, expected_settings_digest="settings")
    warnings = replayed["extraction"]["warnings"]
    assert any("transport failure" in warning for warning in warnings)
    assert any("1 of 1 terminal requests failed" in warning for warning in warnings)


def test_enrichment_settings_digest_includes_resolver_result_settings():
    from unittest.mock import MagicMock

    from bibr.pipeline.artifacts import enrichment_settings_digest
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress

    def context(*, url, authoritative, limit):
        settings = MagicMock()
        settings.crossref.enrich = True
        settings.crossref.consolidate = "off"
        settings.crossref.enrich_concurrency = 4
        settings.crossref.enrich_timeout = 30
        settings.crossref.request_timeout = 10
        settings.crossref.rate_limit_rpm = 50
        settings.crossref.cache_size = 100
        settings.crossref.redis_cache = False
        settings.crossref.cache_ttl_seconds = 60
        settings.resolver.url = url
        settings.resolver.enrich = True
        settings.resolver.timeout = 10
        settings.resolver.limit = limit
        settings.resolver.search_concurrency = 8
        settings.resolver.authoritative = authoritative
        settings.ror.enrich = True
        settings.ror.url = "https://api.ror.org/v2"
        settings.ror.enrich_timeout = 60.0
        return PipelineContext(
            file_states=[],
            progress=NullProgress(),
            resources=MagicMock(),
            config=RunConfig(crossref=True),
            settings=settings,
        )

    baseline = enrichment_settings_digest(
        context(url="http://resolver-a", authoritative=False, limit=20)
    )
    assert baseline != enrichment_settings_digest(
        context(url="http://resolver-b", authoritative=False, limit=20)
    )
    assert baseline != enrichment_settings_digest(
        context(url="http://resolver-a", authoritative=True, limit=20)
    )
    assert baseline != enrichment_settings_digest(
        context(url="http://resolver-a", authoritative=False, limit=5)
    )


def test_ror_rows_round_trip_through_the_sidecar():
    from bibr.pipeline.artifacts import (
        ArtifactReplayError,
        EnrichmentSidecar,
        canonical_json_sha256,
        make_enrichment_sidecar,
        replay_enrichment_sidecar,
    )

    core = {
        **_core_payload(),
        "affiliation": [{"affiliation_id": 1, "text": "Uni"}],
        "funding": [{"funding_id": 1, "funder": "NSF"}],
        "affiliation_match": [],
        "funding_match": [],
    }
    affiliation_row = {
        "affiliation_id": 1,
        "service": "ror",
        "service_id": "https://ror.org/0abcde123",
        "score": 1.0,
        "name": "Uni",
        "country_code": "NL",
    }
    funding_row = {
        "funding_id": 1,
        "service": "ror",
        "service_id": "https://ror.org/021nxhr62",
        "score": 1.0,
        "name": "U.S. National Science Foundation",
        "country_code": "US",
        "funder_doi": "10.13039/100000001",
    }
    enriched = {**core, "affiliation_match": [affiliation_row], "funding_match": [funding_row]}
    sidecar = make_enrichment_sidecar(
        enriched,
        core_sha256=canonical_json_sha256(core),
        settings_digest="s",
        completeness="complete",
    )
    assert EnrichmentSidecar.from_dict(sidecar.to_dict()) == sidecar

    replayed = replay_enrichment_sidecar(core, sidecar, expected_settings_digest="s")
    assert replayed["affiliation_match"] == [affiliation_row]
    assert replayed["funding_match"] == [funding_row]

    dangling = {**enriched, "affiliation_match": [{**affiliation_row, "affiliation_id": 9}]}
    bad = make_enrichment_sidecar(
        dangling,
        core_sha256=canonical_json_sha256(core),
        settings_digest="s",
        completeness="complete",
    )
    with pytest.raises(ArtifactReplayError, match="unknown affiliation_id"):
        replay_enrichment_sidecar(core, bad, expected_settings_digest="s")


def test_a_sidecar_without_ror_rows_stays_compact():
    from bibr.pipeline.artifacts import make_enrichment_sidecar

    sidecar = make_enrichment_sidecar(
        _core_payload(), core_sha256="x", settings_digest="s", completeness="complete"
    )
    assert "affiliation_match" not in sidecar.to_dict()
    assert "funding_match" not in sidecar.to_dict()
