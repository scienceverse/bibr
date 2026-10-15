"""The identity-only lookup: the paper's own DOI, without reference enrichment.

``CROSSREF_IDENTITY`` (per run: ``crossref_identity``) makes one request per
paper with a selected DOI when reference enrichment does not run. No-LLM runs
never make it, a paper without a DOI costs nothing, and ``refs=off`` does not
stop it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.config import GlobalSettings
from bibr.models import ExternalMatch, MatchSource, PaperMetadata
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.enricher import (
    CrossrefIdentityEnricher,
    EnrichmentOutcome,
    EnrichmentStatus,
    needs_identity_lookup,
)
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.enrich import EnrichmentStage
from bibr.pipeline.state import FileState
from bibr.processing_warnings import ProcessingWarning, WarningCode


def _settings(*, enrich: bool = False, identity: bool = False) -> GlobalSettings:
    settings = GlobalSettings()
    settings.crossref.enrich = enrich
    settings.crossref.identity = identity
    return settings


def _ctx(file_states, config: RunConfig, settings: GlobalSettings) -> PipelineContext:
    return PipelineContext(
        file_states=list(file_states),
        progress=NullProgress(),
        resources=MagicMock(),
        config=config,
        settings=settings,
    )


def _fs(name: str, doi: str | None) -> FileState:
    fs = FileState(path=Path(name))
    fs.paper = MagicMock()
    fs.paper.metadata = PaperMetadata(doi=doi or "", title="Printed title")
    return fs


# --- RunConfig.enrichment_mode ----------------------------------------------


@pytest.mark.parametrize(
    ("config", "enrich", "identity", "mode"),
    [
        # Off by default.
        (RunConfig(), False, False, "off"),
        # The setting turns it on; an explicit per-run value wins either way.
        (RunConfig(), False, True, "identity"),
        (RunConfig(crossref_identity=True), False, False, "identity"),
        (RunConfig(crossref_identity=False), False, True, "off"),
        # Reference enrichment already looks the paper up.
        (RunConfig(crossref=True), False, True, "full"),
        (RunConfig(), True, True, "full"),
        (RunConfig(crossref=False), True, True, "identity"),
        # refs=off stops reference enrichment, not the identity lookup.
        (RunConfig(crossref=True, ref_parse_strategy="off"), False, True, "identity"),
        (RunConfig(crossref=True, ref_parse_strategy="off"), False, False, "off"),
        # A no-LLM run makes no external call at all.
        (RunConfig(no_llm=True, crossref_identity=True), False, True, "off"),
        (RunConfig(no_llm=True, crossref=True), True, True, "off"),
    ],
)
def test_enrichment_mode(config, enrich, identity, mode):
    assert config.enrichment_mode(_settings(enrich=enrich, identity=identity)) == mode


def test_identity_setting_is_off_by_default():
    assert GlobalSettings().crossref.identity is False


def test_needs_identity_lookup_requires_a_doi():
    assert needs_identity_lookup(_fs("a.pdf", "10.1/x").paper)
    assert not needs_identity_lookup(_fs("b.pdf", None).paper)
    assert not needs_identity_lookup(None)


# --- CrossrefIdentityEnricher -------------------------------------------------


async def test_enricher_without_doi_makes_no_request(monkeypatch):
    lookup = AsyncMock()
    monkeypatch.setattr("bibr.enrich.references.enrich_paper_identity", lookup)

    outcome = await CrossrefIdentityEnricher(settings=_settings()).enrich(_fs("a.pdf", None))

    assert outcome.status is EnrichmentStatus.NO_WORK
    lookup.assert_not_awaited()


async def test_enricher_looks_up_the_paper_doi_once(monkeypatch):
    from bibr.enrich.references import EnrichmentReport

    async def lookup(metadata, **_kwargs):
        metadata.match[MatchSource.CROSSREF] = ExternalMatch(doi=metadata.doi, score=100.0)
        return EnrichmentReport(attempted=1, matched=1)

    spy = AsyncMock(side_effect=lookup)
    monkeypatch.setattr("bibr.enrich.references.enrich_paper_identity", spy)
    fs = _fs("a.pdf", "10.1234/self")

    outcome = await CrossrefIdentityEnricher(settings=_settings()).enrich(fs)

    assert outcome == EnrichmentOutcome(EnrichmentStatus.COMPLETE)
    spy.assert_awaited_once()
    assert fs.paper.metadata.match[MatchSource.CROSSREF].doi == "10.1234/self"
    # The printed fields are the paper's; the record only goes to ``match``.
    assert fs.paper.metadata.title == "Printed title"


async def test_enricher_failure_is_a_warning_not_a_partial_enrichment(monkeypatch):
    monkeypatch.setattr(
        "bibr.enrich.references.enrich_paper_identity",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    outcome = await CrossrefIdentityEnricher(settings=_settings()).enrich(_fs("a.pdf", "10.1/x"))

    assert outcome.status is EnrichmentStatus.COMPLETE
    assert [w.code for w in outcome.warnings] == [WarningCode.CROSSREF_ENRICHMENT_FAILED]


async def test_enricher_timeout_is_a_warning_not_a_partial_enrichment(monkeypatch):
    async def slow(*_args, **_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr("bibr.enrich.references.enrich_paper_identity", slow)

    outcome = await CrossrefIdentityEnricher(timeout=0.01, settings=_settings()).enrich(
        _fs("a.pdf", "10.1/x")
    )

    assert outcome.status is EnrichmentStatus.COMPLETE
    assert [w.code for w in outcome.warnings] == [WarningCode.CROSSREF_ENRICHMENT_TIMEOUT]


async def test_enricher_passes_lookup_failure_details_on(monkeypatch):
    from bibr.enrich.references import EnrichmentReport

    detail = ProcessingWarning(WarningCode.ENRICHMENT_LOOKUP_FAILED, "bib_id=0 lookup failed")
    monkeypatch.setattr(
        "bibr.enrich.references.enrich_paper_identity",
        AsyncMock(return_value=EnrichmentReport(attempted=1, failed=1, details=(detail,))),
    )

    outcome = await CrossrefIdentityEnricher(settings=_settings()).enrich(_fs("a.pdf", "10.1/x"))

    assert outcome.status is EnrichmentStatus.COMPLETE
    assert outcome.warnings == (detail,)


# --- EnrichmentStage gating ---------------------------------------------------


class _Recorder:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[str] = []

    async def enrich(self, fs):
        self.calls.append(fs.path.name)
        return EnrichmentOutcome(EnrichmentStatus.COMPLETE)


@pytest.mark.parametrize(
    ("config", "settings", "full_calls", "identity_calls"),
    [
        (RunConfig(), _settings(), [], []),
        (RunConfig(), _settings(identity=True), [], ["doi.pdf"]),
        (RunConfig(crossref_identity=True), _settings(), [], ["doi.pdf"]),
        (RunConfig(crossref=True), _settings(identity=True), ["doi.pdf", "nodoi.pdf"], []),
        (
            RunConfig(crossref=True, ref_parse_strategy="off"),
            _settings(identity=True),
            [],
            ["doi.pdf"],
        ),
        (RunConfig(no_llm=True), _settings(identity=True), [], []),
    ],
)
async def test_stage_runs_identity_enrichers_only_in_identity_mode(
    config, settings, full_calls, identity_calls
):
    full, identity = _Recorder("crossref"), _Recorder("crossref_identity")
    with_doi, without_doi = _fs("doi.pdf", "10.1/x"), _fs("nodoi.pdf", None)

    await EnrichmentStage([full], identity_enrichers=[identity]).run(
        _ctx([with_doi, without_doi], config, settings)
    )

    assert sorted(full.calls) == full_calls
    assert identity.calls == identity_calls
    looked_up = {fs.path.name for fs in (with_doi, without_doi) if fs.enrichment_state}
    assert looked_up == set(full_calls or identity_calls)


async def test_stage_without_identity_enricher_skips_identity_mode():
    full = _Recorder("crossref")
    fs = _fs("doi.pdf", "10.1/x")

    await EnrichmentStage([full]).run(_ctx([fs], RunConfig(crossref_identity=True), _settings()))

    assert full.calls == []
    assert fs.enrichment_state is None


# --- core checkpoint, export and replay -------------------------------------


def _core_payload(doi: str | None) -> dict:
    return {
        "schema_version": "12.3",
        "metadata": {"title": "Printed title", "doi": doi},
        "bib": [{"bib_id": 1}],
        "bib_match": [],
        "metadata_match": [],
        "extraction": {
            "warnings": [],
            "validation": {
                "errors": 0,
                "warnings": 0,
                "blocking": 0,
                "promotable": True,
                "issues": [],
            },
        },
    }


def _pending(payload: dict) -> bool:
    return any(
        issue["code"] == "VAL_ENRICHMENT_PENDING"
        for issue in payload["extraction"]["validation"]["issues"]
    )


@pytest.mark.parametrize(
    ("doi", "config", "pending"),
    [
        ("10.1234/self", RunConfig(crossref_identity=True), True),
        (None, RunConfig(crossref_identity=True), False),
        ("10.1234/self", RunConfig(), False),
        ("10.1234/self", RunConfig(crossref_identity=True, no_llm=True), False),
        ("10.1234/self", RunConfig(crossref_identity=True, ref_parse_strategy="off"), True),
    ],
)
async def test_checkpoint_marks_identity_lookups_pending(tmp_path, doi, config, pending):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    fs = _fs("paper.pdf", doi)
    fs.path = tmp_path / "paper.pdf"
    fs.paper.validation_issues = []
    fs.paper.export_to_json.return_value = _core_payload(doi)
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")

    await CoreCheckpointStage(identity_requested=True).run(_ctx([fs], config, _settings()))

    core = json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8"))
    assert _pending(core) is pending


async def test_checkpoint_without_identity_enricher_is_not_pending(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage

    fs = _fs("paper.pdf", "10.1234/self")
    fs.path = tmp_path / "paper.pdf"
    fs.paper.validation_issues = []
    fs.paper.export_to_json.return_value = _core_payload("10.1234/self")
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")

    await CoreCheckpointStage().run(_ctx([fs], RunConfig(crossref_identity=True), _settings()))

    assert not _pending(json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8")))


async def test_identity_sidecar_replay_carries_the_record_and_no_reference_receipt(tmp_path):
    """The replayed export has the record, no pending gate, and no
    ``extraction.enrichment`` row: no reference was looked up."""
    import copy

    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _core_payload("10.1234/self")
    enriched = copy.deepcopy(core)
    enriched["metadata_match"] = [{"service": "crossref", "doi": "10.1234/self", "score": 1.0}]
    fs = _fs("paper.pdf", "10.1234/self")
    fs.path = tmp_path / "paper.pdf"
    fs.paper.validation_issues = []
    fs.paper.export_to_json.side_effect = [core, enriched]
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = _ctx([fs], RunConfig(crossref_identity=True, consolidate="off"), _settings())

    await CoreCheckpointStage(identity_requested=True).run(ctx)
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE
    await ExportStage().run(ctx)

    assert fs.error is None
    assert fs.result_json["metadata_match"] == enriched["metadata_match"]
    assert fs.result_json["bib_match"] == []
    assert "enrichment" not in fs.result_json["extraction"]
    assert not _pending(fs.result_json)
    assert fs.result_json["extraction"]["validation"]["promotable"] is True


async def test_identity_replay_compares_the_front_matter_with_the_record(tmp_path):
    """The replayed export, and the -o file on disk, say whether the title and
    authors agree with the record; the printed values stay."""
    import copy

    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.export import ExportStage

    core = _core_payload("10.1234/self")
    core["author"] = [{"given": "Leila", "family": "Ben Ammar Elgaaied", "literal": None}]
    core["extraction"]["fields"] = {
        name: {"state": "extracted", "source": "llm", "issues": []} for name in ("title", "author")
    }
    enriched = copy.deepcopy(core)
    enriched["metadata_match"] = [
        {
            "service": "crossref",
            "doi": "10.1234/self",
            "score": 1.0,
            "title": "Printed",
            "subtitle": "Title",
            "author": [{"given": "Leila Ben Ammar", "family": "Elgaaied"}],
        }
    ]
    fs = _fs("paper.pdf", "10.1234/self")
    fs.path = tmp_path / "paper.pdf"
    fs.paper.validation_issues = []
    fs.paper.export_to_json.side_effect = [core, enriched]
    fs.artifact_sink = LocalArtifactSink(tmp_path / "paper.json")
    ctx = _ctx([fs], RunConfig(crossref_identity=True, consolidate="off"), _settings())

    await CoreCheckpointStage(identity_requested=True).run(ctx)
    fs.enrichment_state = RunState.ENRICHMENT_COMPLETE
    await ExportStage().run(ctx)

    assert fs.error is None
    expected = {"service": "crossref", "outcome": "agree"}
    fields = fs.result_json["extraction"]["fields"]
    assert fields["title"]["registry"] == expected
    assert fields["author"]["registry"] == expected
    assert fs.result_json["metadata"]["title"] == "Printed title"
    on_disk = json.loads((tmp_path / "paper.json").read_text(encoding="utf-8"))
    assert on_disk["extraction"]["fields"] == fields
    # The immutable core predates the lookup.
    stored_core = json.loads(fs.artifact_sink.core_path(fs).read_text(encoding="utf-8"))
    assert "registry" not in stored_core["extraction"]["fields"]["title"]


def test_settings_digest_separates_identity_from_off():
    from bibr.pipeline.artifacts import enrichment_settings_digest

    settings = _settings()
    off = enrichment_settings_digest(_ctx([], RunConfig(), settings))
    identity = enrichment_settings_digest(_ctx([], RunConfig(crossref_identity=True), settings))
    assert off != identity


# --- pipeline construction ----------------------------------------------------


def _enrichment_stage(pipeline) -> EnrichmentStage:
    candidates = list(pipeline._stages) + [
        stage._enrich for stage in pipeline._stages if hasattr(stage, "_enrich")
    ]
    (stage,) = [s for s in candidates if isinstance(s, EnrichmentStage)]
    return stage


@pytest.mark.parametrize(
    ("kwargs", "enrich", "identity", "full_names", "identity_names"),
    [
        ({}, False, False, [], []),
        ({}, False, True, [], ["crossref_identity"]),
        ({"crossref_identity": True}, False, False, [], ["crossref_identity"]),
        ({"crossref_identity": True, "no_llm": True}, False, True, [], []),
        ({"crossref": True}, False, True, ["crossref"], []),
        ({"crossref": True, "ref_parse_strategy": "off"}, False, True, [], ["crossref_identity"]),
    ],
)
def test_local_pipeline_builds_the_identity_enricher(
    kwargs, enrich, identity, full_names, identity_names
):
    from bibr.local.pipeline import LocalPipeline

    settings = _settings(enrich=enrich, identity=identity)
    settings.ror.enrich = False
    stage = _enrichment_stage(LocalPipeline(llm_backend="cloud", settings=settings, **kwargs))

    assert [e.name for e in stage._enrichers] == full_names
    assert [e.name for e in stage._identity_enrichers] == identity_names
