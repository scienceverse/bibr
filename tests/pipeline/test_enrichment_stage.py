"""EnrichmentStage — drives a list of Enricher objects concurrently."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.enrich import EnrichmentStage
from bibr.pipeline.state import FileState
from bibr.processing_warnings import ProcessingWarning, WarningCode


def _ctx_with_progress(file_states, prog):
    return PipelineContext(
        file_states=file_states,
        progress=prog,
        resources=MagicMock(),
        config=RunConfig(crossref=True),
    )


class _FakeEnricher:
    name = "fake"

    def __init__(self):
        self.calls = []
        self.enrich = AsyncMock(side_effect=self._record)

    async def _record(self, fs):
        self.calls.append(fs.path.name)


def _ctx(file_states):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(crossref=True),
    )


@pytest.mark.asyncio
async def test_runs_each_enricher_on_each_alive_paper():
    fs1 = FileState(path=Path("a.pdf"))
    fs1.paper = MagicMock()
    fs2 = FileState(path=Path("b.pdf"))
    fs2.paper = MagicMock()

    e1 = _FakeEnricher()
    e2 = _FakeEnricher()
    await EnrichmentStage(enrichers=[e1, e2]).run(_ctx([fs1, fs2]))

    assert sorted(e1.calls) == ["a.pdf", "b.pdf"]
    assert sorted(e2.calls) == ["a.pdf", "b.pdf"]


@pytest.mark.asyncio
async def test_skips_files_without_paper():
    fs = FileState(path=Path("x.pdf"))
    fs.paper = None
    e = _FakeEnricher()
    await EnrichmentStage(enrichers=[e]).run(_ctx([fs]))
    assert e.calls == []


@pytest.mark.asyncio
async def test_empty_pipe_is_noop():
    fs = FileState(path=Path("x.pdf"))
    fs.paper = MagicMock()
    await EnrichmentStage(enrichers=[]).run(_ctx([fs]))  # must not raise


@pytest.mark.asyncio
async def test_emits_brackets_with_enrichers_but_no_papers():
    """When enrichers exist but no file has a paper, stage still emits brackets."""
    fs = FileState(path=Path("x.pdf"))
    fs.paper = None
    prog = MagicMock(wraps=NullProgress())
    ctx = _ctx_with_progress([fs], prog)
    await EnrichmentStage(enrichers=[_FakeEnricher()]).run(ctx)
    prog.stage_start.assert_called_once_with("enrich")
    prog.stage_end.assert_called_once_with("enrich")


@pytest.mark.asyncio
async def test_raising_enricher_does_not_abort_chunk():
    """One file's enricher blowing up must not abort the other files —
    the failure becomes a warning on that file, the stage never raises."""
    fs_ok = FileState(path=Path("ok.pdf"))
    fs_ok.paper = MagicMock()
    fs_bad = FileState(path=Path("bad.pdf"))
    fs_bad.paper = MagicMock()

    class _ExplodingEnricher:
        name = "exploding"

        async def enrich(self, fs):
            if fs.path.name == "bad.pdf":
                raise RuntimeError("boom")

    follow_up = _FakeEnricher()
    await EnrichmentStage(enrichers=[_ExplodingEnricher(), follow_up]).run(_ctx([fs_ok, fs_bad]))

    # The later enricher still ran on every file…
    assert sorted(follow_up.calls) == ["bad.pdf", "ok.pdf"]
    # …the failing file carries a warning, not an error…
    assert fs_bad.warnings == [
        ProcessingWarning(
            WarningCode.ENRICHER_FAILED, "_ExplodingEnricher failed: RuntimeError: boom"
        )
    ]
    assert fs_bad.enrichment_warnings == fs_bad.warnings
    assert fs_bad.error is None
    # …and the healthy file is untouched.
    assert fs_ok.warnings == []


@pytest.mark.asyncio
async def test_explicit_partial_outcome_drives_partial_state_without_warning_inference():
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.enricher import EnrichmentOutcome, EnrichmentStatus

    fs = FileState(path=Path("partial.pdf"), paper=MagicMock())
    enricher = MagicMock()
    enricher.enrich = AsyncMock(
        return_value=EnrichmentOutcome(
            status=EnrichmentStatus.PARTIAL,
            warnings=(),
            detail="one terminal lookup failure",
        )
    )

    await EnrichmentStage([enricher]).run(_ctx([fs]))

    assert fs.enrichment_state is RunState.ENRICHMENT_PARTIAL


@pytest.mark.asyncio
async def test_no_work_outcome_completes_requested_enrichment_gate():
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.enricher import EnrichmentOutcome, EnrichmentStatus

    fs = FileState(path=Path("empty.pdf"), paper=MagicMock())
    enricher = MagicMock()
    enricher.enrich = AsyncMock(return_value=EnrichmentOutcome(status=EnrichmentStatus.NO_WORK))

    await EnrichmentStage([enricher]).run(_ctx([fs]))

    assert fs.enrichment_state is RunState.ENRICHMENT_COMPLETE


@pytest.mark.asyncio
async def test_cutoff_does_not_regress_completed_sibling_and_is_recorded_once(tmp_path):
    from bibr.local.artifacts import LocalArtifactSink
    from bibr.pipeline.artifacts import RunState
    from bibr.pipeline.enricher import EnrichmentOutcome, EnrichmentStatus

    complete = FileState(path=Path("complete.pdf"), paper=MagicMock())
    interrupted = FileState(path=Path("interrupted.pdf"), paper=MagicMock())
    for fs in (complete, interrupted):
        fs.core_sha256 = "a" * 64
        fs.artifact_sink = LocalArtifactSink(tmp_path / f"{fs.path.stem}.json")
        fs.artifact_sink.record(fs, RunState.STARTED)
        fs.artifact_sink.record(fs, RunState.CORE_WRITTEN)
    complete.artifact_sink.record(complete, RunState.ENRICHMENT_COMPLETE)

    class _InterruptedEnricher:
        async def enrich(self, fs):
            if fs is interrupted:
                raise asyncio.CancelledError()
            return EnrichmentOutcome(status=EnrichmentStatus.COMPLETE)

    with pytest.raises(asyncio.CancelledError):
        await EnrichmentStage([_InterruptedEnricher()]).run(_ctx([complete, interrupted]))

    complete_events = json.loads(complete.artifact_sink.receipt_path(complete).read_text())[
        "events"
    ]
    interrupted_events = json.loads(
        interrupted.artifact_sink.receipt_path(interrupted).read_text()
    )["events"]
    assert [event["state"] for event in complete_events].count("cutoff_interrupted") == 0
    assert [event["state"] for event in interrupted_events].count("cutoff_interrupted") == 1
