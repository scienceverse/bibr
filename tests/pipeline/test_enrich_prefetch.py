"""Enrichment prefetch overlaps the extract stage's LLM tail.

``bibr/pipeline/enrich_prefetch.py`` starts ``prefetch_enrichment`` the moment
references are parsed and hangs the handle on the ``Paper``; the enricher
consumes it and every non-enriching path cancels it. Nothing here touches the
network: ``prefetch_enrichment`` is replaced by fakes with controlled delays.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from bibr.config import GlobalSettings
from bibr.models import PaperMetadata
from bibr.paper import PaperReference
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.enrich_prefetch import (
    EnrichmentPrefetchHandle,
    PrefetchUnavailable,
    cancel_leftover_prefetches,
    discard_prefetches,
    prefetch_handle_of,
    start_enrichment_prefetch,
    take_prefetch_handle,
)
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.enrich import EnrichmentStage
from bibr.pipeline.state import FileState

PREFETCH = "bibr.enrich.references.prefetch_enrichment"


def _ref(i: int = 1, doi: str | None = None) -> PaperReference:
    return PaperReference(
        bib_id=i,
        title="A Great Paper on Testing Methods",
        first_page=None,
        volume=None,
        authors="Smith, J.",
        year=2020,
        container=None,
        doi=doi,
    )


def _settings(enrich: bool = True) -> GlobalSettings:
    settings = GlobalSettings()
    settings.crossref.enrich = enrich
    return settings


def _pending_tasks() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not current and not t.done()]


async def _fake_prefetch(delay: float = 0.0, result: object = "prefetch"):
    """A ``prefetch_enrichment`` stand-in returning *result* after *delay*."""

    async def fake(references, *, settings=None, **_kwargs):  # noqa: ARG001
        if delay:
            await asyncio.sleep(delay)
        return result

    return fake


# --- the handle ------------------------------------------------------------------


async def test_start_returns_none_for_empty_references():
    assert start_enrichment_prefetch([], settings=_settings()) is None
    assert _pending_tasks() == []


async def test_handle_runs_prefetch_and_records_timing():
    settings = _settings()
    with patch(PREFETCH, await _fake_prefetch(0.01, result="done")) as fake:
        handle = start_enrichment_prefetch([_ref()], settings=settings)
        assert handle is not None
        assert handle.n_references == 1
        assert not handle.done()
        assert handle.seconds is None
        assert await handle.result() == "done"
    assert handle.done()
    assert handle.seconds is not None and handle.seconds >= 0.01
    # ``prefetch_enrichment`` is not a Mock here; the fake was called through the patch.
    assert fake is not None
    assert _pending_tasks() == []


async def test_failed_prefetch_surfaces_as_unavailable_not_as_an_error():
    async def boom(references, *, settings=None, **_kwargs):  # noqa: ARG001
        raise RuntimeError("resolver exploded")

    with patch(PREFETCH, boom):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        assert handle is not None
        with pytest.raises(PrefetchUnavailable, match="resolver exploded"):
            await handle.result()
    assert handle.done()
    assert _pending_tasks() == []


async def test_independently_cancelled_prefetch_is_unavailable_not_cancelled():
    with patch(PREFETCH, await _fake_prefetch(10.0)):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        assert handle is not None
        handle.cancel()
        with pytest.raises(PrefetchUnavailable):
            await handle.result()
    assert _pending_tasks() == []


async def test_caller_cancellation_propagates_and_cancels_the_prefetch():
    """An enrich timeout cancels the waiter; the prefetch must die with it."""
    with patch(PREFETCH, await _fake_prefetch(10.0)):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        assert handle is not None

        async def wait_for_it():
            return await handle.result()

        waiter = asyncio.create_task(wait_for_it())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.wait([handle.task])
    assert handle.task.cancelled()
    assert _pending_tasks() == []


async def test_discard_cancels_and_settles():
    with patch(PREFETCH, await _fake_prefetch(10.0)):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        assert handle is not None
        await handle.discard()
    assert handle.task.cancelled()
    assert _pending_tasks() == []


def test_overlap_accounting():
    task = MagicMock()
    task.add_done_callback = MagicMock()
    handle = EnrichmentPrefetchHandle(task, started_at=100.0, n_references=3)
    handle.finished_at = 105.0
    # Finished before the enrich stage started waiting: fully hidden.
    assert handle.overlap(110.0) == (5.0, 0.0)
    # The stage started waiting at t=103: 3 s hidden, 2 s exposed.
    assert handle.overlap(103.0) == (3.0, 2.0)
    assert handle.seconds == 5.0


async def test_paper_helpers_tolerate_doubles_and_detach():
    assert prefetch_handle_of(MagicMock()) is None
    assert prefetch_handle_of(None) is None
    assert take_prefetch_handle(MagicMock()) is None
    with patch(PREFETCH, await _fake_prefetch(10.0)):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        paper = MagicMock()
        paper.enrichment_prefetch = handle
        assert prefetch_handle_of(paper) is handle
        assert take_prefetch_handle(paper) is handle
        assert paper.enrichment_prefetch is None
        fs = FileState(path=Path("p.pdf"), paper=MagicMock())
        fs.paper.enrichment_prefetch = handle
        assert await discard_prefetches([fs]) == 1
    assert handle.task.cancelled()
    assert _pending_tasks() == []


async def test_cancel_leftovers_and_free_all_cancel_the_task():
    with patch(PREFETCH, await _fake_prefetch(10.0)):
        first = start_enrichment_prefetch([_ref()], settings=_settings())
        second = start_enrichment_prefetch([_ref(2)], settings=_settings())
        fs_a = FileState(path=Path("a.pdf"), paper=MagicMock())
        fs_a.paper.enrichment_prefetch = first
        fs_b = FileState(path=Path("b.pdf"), paper=MagicMock())
        fs_b.paper.enrichment_prefetch = second

        assert cancel_leftover_prefetches([fs_a]) == 1
        fs_b.free_all()  # an errored file freed at a stage boundary
        assert fs_b.paper is None
        await asyncio.wait([first.task, second.task])
    assert first.task.cancelled() and second.task.cancelled()
    assert _pending_tasks() == []


# --- extractor listener ------------------------------------------------------------


def _extractor_with_ref_section(**kwargs):
    from bibr.extract.extractor import MetadataExtractor

    sections = ["Introduction"] * 3 + ["References"] * 3
    texts = [f"Intro {i}" for i in range(3)] + [
        "Smith J. (2020). Paper A. Nature, 10, 1-5.",
        "Jones A. (2019). Paper B. Science, 20, 10-15.",
        "Brown B. (2021). Paper C. Cell, 30, 100-110.",
    ]
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame({"section_name": sections, "text": texts})
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
        PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
    ]
    contents.sentences = []
    return MetadataExtractor(contents, **kwargs)


async def test_on_references_ready_fires_before_core_metadata_completes():
    ext = _extractor_with_ref_section()
    events: list = []
    parsed = [_ref(1), _ref(2)]

    async def slow_core():
        await asyncio.sleep(0.05)
        events.append("core_done")
        ext.metadata = PaperMetadata(doi="", title="T", keywords=[], authors=[])

    async def fast_refs(_ref_df):
        await asyncio.sleep(0.001)
        events.append("refs_done")
        return parsed

    ext.extract_core_metadata = slow_core
    ext._extract_references = fast_refs

    def listener(references):
        events.append(("listener", len(references), references is parsed))

    meta = await ext.extract_all_metadata(on_references_ready=listener)

    assert events == ["refs_done", ("listener", 2, True), "core_done"]
    assert meta.references is parsed


async def test_on_references_ready_skips_empty_and_failed_reference_tasks():
    ext = _extractor_with_ref_section()
    calls: list = []

    async def core():
        ext.metadata = PaperMetadata(doi="", title="T", keywords=[], authors=[])

    async def empty_refs(_ref_df):
        return []

    ext.extract_core_metadata = core
    ext._extract_references = empty_refs
    await ext.extract_all_metadata(on_references_ready=calls.append)
    assert calls == []

    ext = _extractor_with_ref_section()

    async def core2():
        ext.metadata = PaperMetadata(doi="", title="T", keywords=[], authors=[])

    async def failing_refs(_ref_df):
        raise RuntimeError("parser died")

    ext.extract_core_metadata = core2
    ext._extract_references = failing_refs
    meta = await ext.extract_all_metadata(on_references_ready=calls.append)
    assert calls == []
    assert meta.references_incomplete is True


async def test_listener_failure_never_breaks_extraction(caplog):
    ext = _extractor_with_ref_section()
    parsed = [_ref(1)]

    async def core():
        await asyncio.sleep(0.01)
        ext.metadata = PaperMetadata(doi="", title="T", keywords=[], authors=[])

    async def refs(_ref_df):
        return parsed

    ext.extract_core_metadata = core
    ext._extract_references = refs

    def bad_listener(_references):
        raise ValueError("listener bug")

    meta = await ext.extract_all_metadata(on_references_ready=bad_listener)
    assert meta.references is parsed
    assert any("on_references_ready callback failed" in r.message for r in caplog.records)


# --- post_parse wiring ----------------------------------------------------------------


def _contents() -> PaperContents:
    return PaperContents(
        sentences=[],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: ""},
        detected_title="My Paper",
    )


def _post_parse_patches(extract):
    return (
        patch("bibr.pipeline.stages.post_parse._classify_sections", AsyncMock(return_value=None)),
        patch(
            "bibr.structure.implicit_sections.detect_implicit_sections",
            AsyncMock(return_value=None),
        ),
        patch("bibr.pipeline.stages.post_parse._extract_metadata_and_equations", extract),
        patch("bibr.pipeline.stages.post_parse._link_citations", AsyncMock(return_value=None)),
        patch(
            "bibr.extract.research_integrity.extract_structured_integrity",
            AsyncMock(return_value=None),
        ),
    )


async def test_post_parse_attaches_prefetch_handle_started_by_the_listener():
    from bibr.pipeline.stages.post_parse import post_parse

    parsed = [_ref(1, doi="10.1/a")]

    async def extract(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        kwargs["on_references_ready"](parsed)
        return PaperMetadata(doi="", title="T", references=parsed)

    a, b, c, d, e = _post_parse_patches(extract)
    with a, b, c, d, e, patch(PREFETCH, await _fake_prefetch(0.0, result="ready")):
        paper = await post_parse(
            contents=_contents(),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=MagicMock(),
            settings=_settings(),
            enrichment_prefetch=True,
        )
        handle = prefetch_handle_of(paper)
        assert handle is not None
        assert await handle.result() == "ready"
    assert _pending_tasks() == []


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"enrichment_prefetch": False}, "not requested"),
        ({"enrichment_prefetch": True, "ref_parse_strategy": "off"}, "refs off"),
    ],
)
async def test_post_parse_does_not_start_prefetch_when_not_wanted(kwargs, reason):
    from bibr.pipeline.stages.post_parse import post_parse

    seen: dict = {}

    async def extract(contents, file_hash, no_llm, llm_client, **kw):  # noqa: ARG001
        seen["listener"] = kw.get("on_references_ready")
        return PaperMetadata(doi="", title="T")

    a, b, c, d, e = _post_parse_patches(extract)
    with a, b, c, d, e:
        paper = await post_parse(
            contents=_contents(),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=MagicMock(),
            settings=_settings(),
            **kwargs,
        )
    assert seen["listener"] is None, reason
    assert paper.enrichment_prefetch is None
    assert _pending_tasks() == []


async def test_post_parse_failure_after_prefetch_started_cancels_it():
    from bibr.exceptions import ProcessingError
    from bibr.pipeline.stages.post_parse import post_parse

    started: list[EnrichmentPrefetchHandle] = []
    real_start = start_enrichment_prefetch

    def recording_start(references, *, settings):
        handle = real_start(references, settings=settings)
        started.append(handle)
        return handle

    async def extract(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        kwargs["on_references_ready"]([_ref(1)])
        await asyncio.sleep(0)
        raise ProcessingError("extraction died", error_code="extraction_failed")

    a, b, c, d, e = _post_parse_patches(extract)
    with (
        a,
        b,
        c,
        d,
        e,
        patch(PREFETCH, await _fake_prefetch(10.0)),
        patch("bibr.pipeline.enrich_prefetch.start_enrichment_prefetch", recording_start),
    ):
        with pytest.raises(ProcessingError):
            await post_parse(
                contents=_contents(),
                file_name="x.pdf",
                file_hash="deadbeef",
                llm_client=MagicMock(),
                settings=_settings(),
                enrichment_prefetch=True,
            )
        (handle,) = started
        await asyncio.wait([handle.task])
    assert handle.task.cancelled()
    assert _pending_tasks() == []


async def test_post_parse_stage_requests_prefetch_only_when_run_enriches():
    from bibr.pipeline.stages.post_parse import PostParseStage

    seen: list[bool] = []

    async def fake_post_parse(contents, file_name, file_hash, **kwargs):  # noqa: ARG001
        seen.append(kwargs["enrichment_prefetch"])
        return MagicMock(name="paper")

    settings = _settings(enrich=False)
    for crossref, expected in ((None, False), (True, True), (False, False)):
        fs = FileState(path=Path("a.pdf"))
        fs.contents = MagicMock(layout_hints=None)
        ctx = PipelineContext(
            file_states=[fs],
            progress=NullProgress(),
            resources=MagicMock(),
            config=RunConfig(crossref=crossref),
            settings=settings,
        )
        with patch("bibr.pipeline.stages.post_parse.post_parse", fake_post_parse):
            await PostParseStage().run(ctx)
        assert seen[-1] is expected, crossref


# --- stage-level cancellation ----------------------------------------------------------


def _ctx_for(fs: FileState, config: RunConfig, settings: GlobalSettings) -> PipelineContext:
    return PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=config,
        settings=settings,
    )


@pytest.mark.parametrize(
    ("config", "setting"),
    [
        (RunConfig(crossref=True, ref_parse_strategy="off"), True),
        (RunConfig(crossref=False), True),
        (RunConfig(), False),
    ],
)
async def test_enrichment_stage_skip_paths_leave_no_pending_tasks(config, setting):
    enricher = MagicMock()
    enricher.enrich = AsyncMock()
    with patch(PREFETCH, await _fake_prefetch(10.0)):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        fs = FileState(path=Path("p.pdf"), paper=MagicMock())
        fs.paper.enrichment_prefetch = handle

        await EnrichmentStage([enricher]).run(_ctx_for(fs, config, _settings(setting)))

    enricher.enrich.assert_not_awaited()
    assert handle.task.cancelled()
    assert fs.paper.enrichment_prefetch is None
    assert _pending_tasks() == []


async def test_enrichment_stage_settles_handle_an_enricher_ignored():
    class _Ignores:
        name = "ignores"

        async def enrich(self, fs):  # noqa: ARG002
            return None

    with patch(PREFETCH, await _fake_prefetch(10.0)):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        fs = FileState(path=Path("p.pdf"), paper=MagicMock())
        fs.paper.enrichment_prefetch = handle

        await EnrichmentStage([_Ignores()]).run(_ctx_for(fs, RunConfig(crossref=True), _settings()))

    assert handle.task.cancelled()
    assert _pending_tasks() == []


async def test_process_chunk_cancels_leftovers_for_files_that_errored_after_post_parse():
    from bibr.pipeline.pipeline import Pipeline

    class _PostParse:
        name = "extract"
        requires = ()
        produces = ("paper",)

        def __init__(self, handle):
            self._handle = handle

        async def run(self, ctx):
            for fs in ctx.alive():
                fs.paper = MagicMock()
                fs.paper.enrichment_prefetch = self._handle
                fs.result_json = {"kept": True}  # not freed at the boundary

    class _Fails:
        name = "identity"
        requires = ("paper",)
        produces = ()

        async def run(self, ctx):
            for fs in ctx.alive():
                fs.set_error("boom", code="x", stage=self.name)

    with patch(PREFETCH, await _fake_prefetch(10.0)):
        handle = start_enrichment_prefetch([_ref()], settings=_settings())
        pipeline = Pipeline(
            stages=[_PostParse(handle), _Fails()],
            resources=MagicMock(),
            config=RunConfig(crossref=True),
            settings=_settings(),
        )
        fs = FileState(path=Path("p.pdf"))
        await pipeline.process_chunk([fs])
        await asyncio.wait([handle.task])

    assert fs.error == "boom"
    assert handle.task.cancelled()
    assert _pending_tasks() == []


# --- the core checkpoint sees unenriched references ----------------------------------


async def test_prefetch_reads_only_and_checkpoint_payload_has_no_match_data():
    """The prefetch (resolver candidates + Crossref cache) never touches
    ``PaperReference.match``; only ``enrich_references`` does, later."""
    from bibr.enrich.references import enrich_references, prefetch_enrichment
    from bibr.input.file import InputFile, InputFormat
    from bibr.paper import MatchSource, Paper

    ref = _ref(1)
    crossref = MagicMock()
    crossref.enrich_semaphore = asyncio.Semaphore(4)
    crossref.prefetch_works_by_doi = AsyncMock()
    resolver = AsyncMock()
    resolver.healthy = AsyncMock(return_value=True)
    resolver.search_many = AsyncMock(
        return_value=[
            [{"title": "A Great Paper on Testing Methods", "doi": "10.1/x", "source": "openalex"}]
        ]
    )
    settings = _settings()

    prefetch = await prefetch_enrichment(
        [ref], settings=settings, crossref_client=crossref, resolver_client=resolver
    )
    assert prefetch.resolver_prefetch and id(ref) in prefetch.resolver_prefetch
    assert ref.match == {}

    input_file = InputFile(path="p.pdf")
    input_file.file_hash = "deadbeef"
    input_file.input_format = InputFormat(
        file_extension="pdf", detected_mime_type="application/pdf", file_type="PDF"
    )
    paper = Paper(
        input_file=input_file,
        metadata=PaperMetadata(doi="", title="T", references=[ref]),
        contents=PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={}),
    )
    handle = MagicMock(spec=EnrichmentPrefetchHandle)
    paper.enrichment_prefetch = handle
    payload = paper.export_to_json()
    assert payload["bib_match"] == []
    assert "enrichment_prefetch" not in payload

    await enrich_references([ref], settings=settings, prefetch=prefetch)
    assert MatchSource.OPENALEX in ref.match
    assert resolver.healthy.await_count == 1
    assert resolver.search_many.await_count == 1


def test_handle_timing_uses_monotonic_clock():
    task = MagicMock()
    task.add_done_callback = MagicMock()
    before = time.monotonic()
    handle = EnrichmentPrefetchHandle(task, started_at=before, n_references=1)
    assert handle.seconds is None
    handle.finished_at = before + 2.0
    assert handle.seconds == pytest.approx(2.0)
