"""CrossrefEnricher — per-paper Crossref enrichment."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.pipeline.enricher import CrossrefEnricher, Enricher
from bibr.pipeline.stages.enrich import EnrichmentStage
from bibr.pipeline.state import FileState


def _make_fs_with_refs(n: int = 2, doi: str = "") -> FileState:
    fs = FileState(path=Path("p.pdf"))
    fs.paper = MagicMock()
    fs.paper.metadata = MagicMock()
    fs.paper.metadata.references = [MagicMock() for _ in range(n)]
    # Default to no DOI so the self-DOI path stays out of tests that only
    # exercise reference enrichment; opt in explicitly where needed.
    fs.paper.metadata.doi = doi
    return fs


def test_crossref_enricher_is_an_enricher():
    assert isinstance(CrossrefEnricher(), Enricher)


@pytest.mark.asyncio
async def test_enrichment_stage_skips_per_request_refs_off():
    enricher = MagicMock()
    enricher.enrich = AsyncMock()
    fs = MagicMock(paper=MagicMock())
    ctx = MagicMock()
    ctx.config.ref_parse_strategy = "off"
    ctx.alive.return_value = [fs]

    await EnrichmentStage([enricher]).run(ctx)

    enricher.enrich.assert_not_awaited()


@pytest.mark.asyncio
async def test_crossref_enricher_calls_enrich_references():
    fs = _make_fs_with_refs()
    fake = AsyncMock()
    enricher = CrossrefEnricher()
    with patch("bibr.enrich.references.enrich_references", fake):
        await enricher.enrich(fs)
    fake.assert_awaited_once_with(fs.paper.metadata.references, settings=enricher._settings)
    assert fs.warnings == []


@pytest.mark.asyncio
async def test_crossref_enricher_skips_when_no_references():
    fs = FileState(path=Path("p.pdf"))
    fs.paper = MagicMock(metadata=MagicMock(references=[], doi=""))
    fake = AsyncMock()
    with patch("bibr.enrich.references.enrich_references", fake):
        await CrossrefEnricher().enrich(fs)
    fake.assert_not_awaited()


@pytest.mark.asyncio
async def test_crossref_enricher_warns_on_timeout():
    fs = _make_fs_with_refs()

    async def slow(_refs, **_kwargs):
        raise TimeoutError()

    with patch("bibr.enrich.references.enrich_references", slow):
        await CrossrefEnricher(timeout=0.01).enrich(fs)
    assert any("timed out" in w for w in fs.warnings)


@pytest.mark.asyncio
async def test_crossref_enricher_warns_on_exception():
    fs = _make_fs_with_refs()

    async def boom(_refs, **_kwargs):
        raise RuntimeError("nope")

    with patch("bibr.enrich.references.enrich_references", boom):
        await CrossrefEnricher().enrich(fs)
    assert any("failed: nope" in w for w in fs.warnings)


@pytest.mark.asyncio
async def test_marks_enrichment_complete_on_success():
    fs = _make_fs_with_refs()
    with patch("bibr.enrich.references.enrich_references", AsyncMock()):
        await CrossrefEnricher().enrich(fs)
    assert fs.paper.metadata.enrichment_complete is True


@pytest.mark.asyncio
async def test_self_doi_enrichment_runs_when_doi_present():
    fs = _make_fs_with_refs(doi="10.1/self")
    identity = AsyncMock()
    enricher = CrossrefEnricher()
    with (
        patch("bibr.enrich.references.enrich_paper_identity", identity),
        patch("bibr.enrich.references.enrich_references", AsyncMock()),
    ):
        await enricher.enrich(fs)
    identity.assert_awaited_once_with(fs.paper.metadata, settings=enricher._settings)


@pytest.mark.asyncio
async def test_self_doi_enrichment_runs_without_references():
    # A DOI-bearing paper with no references still gets self-identity enrichment;
    # enrichment_complete stays None (it tracks reference enrichment only).
    fs = FileState(path=Path("p.pdf"))
    fs.paper = MagicMock(metadata=MagicMock(references=[], doi="10.1/self"))
    identity = AsyncMock()
    refs = AsyncMock()
    enricher = CrossrefEnricher()
    with (
        patch("bibr.enrich.references.enrich_paper_identity", identity),
        patch("bibr.enrich.references.enrich_references", refs),
    ):
        await enricher.enrich(fs)
    identity.assert_awaited_once_with(fs.paper.metadata, settings=enricher._settings)
    refs.assert_not_awaited()


@pytest.mark.asyncio
async def test_marks_enrichment_incomplete_on_timeout():
    fs = _make_fs_with_refs()

    async def slow(_refs, **_kwargs):
        raise TimeoutError()

    with patch("bibr.enrich.references.enrich_references", slow):
        await CrossrefEnricher(timeout=0.01).enrich(fs)
    assert fs.paper.metadata.enrichment_complete is False


@pytest.mark.asyncio
async def test_crossref_enricher_reports_no_work_for_empty_identity_and_references():
    from bibr.pipeline.enricher import EnrichmentStatus

    fs = FileState(path=Path("p.pdf"))
    fs.paper = MagicMock(metadata=MagicMock(references=[], doi=""))

    outcome = await CrossrefEnricher().enrich(fs)

    assert outcome.status is EnrichmentStatus.NO_WORK
    assert outcome.warnings == ()


@pytest.mark.asyncio
async def test_crossref_enricher_reports_clean_doi_miss_as_complete():
    from bibr.enrich.references import EnrichmentReport
    from bibr.pipeline.enricher import EnrichmentStatus

    fs = _make_fs_with_refs(n=0, doi="10.1/self")
    identity = AsyncMock(return_value=EnrichmentReport(attempted=1))
    with patch("bibr.enrich.references.enrich_paper_identity", identity):
        outcome = await CrossrefEnricher().enrich(fs)

    assert outcome.status is EnrichmentStatus.COMPLETE
    assert outcome.warnings == ()


@pytest.mark.asyncio
async def test_crossref_enricher_reports_swallowed_terminal_failure_as_partial():
    from bibr.enrich.references import EnrichmentReport
    from bibr.pipeline.enricher import EnrichmentStatus

    fs = _make_fs_with_refs(n=1)
    report = EnrichmentReport(
        attempted=1,
        failed=1,
        details=("bib_id=1 DOI lookup failed: transport",),
    )
    with patch("bibr.enrich.references.enrich_references", AsyncMock(return_value=report)):
        outcome = await CrossrefEnricher().enrich(fs)

    assert outcome.status is EnrichmentStatus.PARTIAL
    assert outcome.warnings == report.details
    assert fs.paper.metadata.enrichment_complete is False


@pytest.mark.parametrize("exc", [KeyboardInterrupt("stop"), SystemExit(2)])
def test_crossref_enricher_never_swallows_process_interrupts(exc):
    # Deliberately synchronous, driving its own loop. ``enrich`` awaits through
    # ``asyncio.wait_for``, which on Python < 3.12 wraps the coroutine in a Task;
    # a BaseException raised inside a Task is delivered to the awaiter *and*
    # re-raised out of the loop runner. As an async test that second copy escapes
    # ``run_until_complete`` into pytest-asyncio's runner and interrupts the whole
    # session (KeyboardInterrupt aborts the run ~36% in). Owning the ``asyncio.run``
    # call puts both delivery points inside ``pytest.raises`` on every version.
    fs = _make_fs_with_refs(n=1)

    async def interrupt(*_args, **_kwargs):
        raise exc

    async def enrich():
        with patch("bibr.enrich.references.enrich_references", interrupt):
            await CrossrefEnricher().enrich(fs)

    with pytest.raises(type(exc)):
        asyncio.run(enrich())
