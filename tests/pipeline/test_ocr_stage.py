"""OcrStage — runs per-region OCR on layout results."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import (
    CONCURRENT_MANAGED_OCR_BACKENDS,
    REMOTE_OCR_BACKENDS,
    OcrStage,
    _is_remote_ocr,
    _local_region_limit,
)
from bibr.pipeline.state import FileState


def _ctx(file_states, *, resources=None, config=None, signals=None):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=resources or MagicMock(),
        config=config or RunConfig(),
        signals=signals or StageSignals(any_needs_ocr=True, preloading_ocr=False),
    )


def test_paddle_http_backend_uses_remote_ocr_scheduling():
    assert _is_remote_ocr(RunConfig(ocr_backend="paddle-http"))


@pytest.mark.asyncio
async def test_skips_when_no_file_needs_ocr():
    fs = FileState(path=Path("x.docx"))
    fs.contents = MagicMock()
    rm = MagicMock()
    ctx = _ctx([fs], resources=rm, signals=StageSignals(any_needs_ocr=False, preloading_ocr=False))

    await OcrStage().run(ctx)

    rm.await_ocr.assert_not_called()


@pytest.mark.asyncio
async def test_local_sequential_per_file_ocr_sets_regions():
    fs = FileState(path=Path("x.pdf"))
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[]]
    rm = MagicMock()
    rm.ocr = MagicMock(
        recognize=AsyncMock(return_value="text"),
        loaded=True,
    )
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    # no wait_for_server attr; skip branch
    del rm.ocr.wait_for_server
    cfg = RunConfig(ocr_backend="glm-llama")

    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "hello"}]),
        ),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(_ctx([fs], resources=rm, config=cfg))

    assert fs.ocr_regions is not None
    assert fs.error is None
    # Typed Region IR seam: the stage hands PDFParser (and the QA gate)
    # OcrRegionResult objects, not wire-format dicts.
    from bibr.ocr.types import OcrRegionResult

    assert all(isinstance(r, OcrRegionResult) for page in fs.ocr_regions for r in page)
    assert fs.ocr_regions[0][0].content == "hello"


@pytest.mark.asyncio
async def test_upstream_ocr_error_fails_file_even_if_other_pages_succeed():
    """A systemic OCR outage (UpstreamServiceError — e.g. circuit breaker open)
    on ANY page must fail the whole file. Otherwise the request returns a
    partial result with silently blank pages (the H1 silent-200 defect)."""
    from bibr.exceptions import UpstreamServiceError

    fs = FileState(path=Path("x.pdf"))
    fs.page_images = [MagicMock(), MagicMock()]
    fs.page_indices = [0, 1]
    fs.layout_results = [[], []]
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="t"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    cfg = RunConfig(ocr_backend="glm-http", ocr_url="http://x")

    def _page(page_img, regions, orig_idx, *a, **k):
        if orig_idx == 1:
            raise UpstreamServiceError("ocr", "circuit breaker open")
        return [{"content": "ok"}]

    with (
        patch("bibr.pipeline.stages.ocr.ocr_page_regions", AsyncMock(side_effect=_page)),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(_ctx([fs], resources=rm, config=cfg))

    assert fs.error is not None
    assert isinstance(fs.original_error, UpstreamServiceError)


@pytest.mark.asyncio
async def test_http_path_fires_requests_across_all_files():
    fs1 = FileState(path=Path("a.pdf"))
    fs1.page_images = [MagicMock()]
    fs1.page_indices = [0]
    fs1.layout_results = [[]]
    fs2 = FileState(path=Path("b.pdf"))
    fs2.page_images = [MagicMock()]
    fs2.page_indices = [0]
    fs2.layout_results = [[]]
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="t"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    cfg = RunConfig(ocr_backend="glm-http", ocr_url="http://x")

    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "x"}]),
        ) as mk,
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(_ctx([fs1, fs2], resources=rm, config=cfg))

    # Two files x one page each = 2 calls scheduled concurrently
    assert mk.await_count == 2
    assert fs1.ocr_regions is not None
    assert fs2.ocr_regions is not None


@pytest.mark.asyncio
async def test_run_always_uses_async_loader_no_preload():
    """OcrStage._run must use await_ocr even when no preload was scheduled."""
    fs = FileState(path=Path("x.pdf"))
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[]]
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="text"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server  # skip wait_for_server branch
    cfg = RunConfig(ocr_backend="glm-llama")

    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "hello"}]),
        ),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(
            _ctx(
                [fs],
                resources=rm,
                config=cfg,
                signals=StageSignals(any_needs_ocr=True, preloading_ocr=False),
            )
        )

    rm.await_ocr.assert_awaited_once()
    from bibr.pipeline.resources import ResourceManager

    assert not hasattr(ResourceManager, "ensure_ocr")


@pytest.mark.asyncio
async def test_serve_http_backend_uses_remote_batching(monkeypatch):
    """``ocr_backend == 'serve-http'`` routes through the remote batching path."""
    from bibr.pipeline.stages.ocr import OcrStage

    stage = OcrStage()
    called = {"remote": False, "local": False}

    async def fake_remote(self, ctx, ocr_fn):  # noqa: ARG001
        called["remote"] = True

    async def fake_local(self, ctx, ocr_fn, sem):  # noqa: ARG001
        called["local"] = True

    monkeypatch.setattr(OcrStage, "_run_remote", fake_remote)
    monkeypatch.setattr(OcrStage, "_run_local", fake_local)

    # Minimal ctx — exercise the backend-selection branch only.
    from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals

    rm = MagicMock()
    rm.ocr.recognize = AsyncMock(return_value="")
    rm.ocr.wait_for_server = AsyncMock()
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    ctx = PipelineContext(
        file_states=[],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(ocr_backend="serve-http"),
        signals=StageSignals(any_needs_ocr=True),
    )
    await stage._run(ctx)
    assert called["remote"] is True
    assert called["local"] is False


@pytest.mark.asyncio
async def test_keep_all_skips_per_chunk_shutdown(monkeypatch):
    """``memory_mode='keep_all'`` must avoid the per-chunk OCR shutdown so
    local backends don't pay full re-startup between chunks."""
    from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
    from bibr.pipeline.stages.ocr import OcrStage

    async def fake_remote(self, ctx, ocr_fn):  # noqa: ARG001
        return None

    monkeypatch.setattr(OcrStage, "_run_remote", fake_remote)

    rm = MagicMock()
    rm.ocr.recognize = AsyncMock(return_value="")
    rm.ocr.wait_for_server = AsyncMock()
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    ctx = PipelineContext(
        file_states=[],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(ocr_backend="serve-http", memory_mode="keep_all"),
        signals=StageSignals(any_needs_ocr=True),
    )
    await OcrStage()._run(ctx)
    rm.shutdown_ocr.assert_not_called()


@pytest.mark.asyncio
async def test_balanced_local_llm_calls_per_chunk_shutdown(monkeypatch):
    """In balanced mode a *local* LLM server shares the GPU, so OCR is still
    torn down per chunk to make room for the extract stage."""
    from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
    from bibr.pipeline.stages.ocr import OcrStage

    async def fake_remote(self, ctx, ocr_fn):  # noqa: ARG001
        return None

    monkeypatch.setattr(OcrStage, "_run_remote", fake_remote)

    rm = MagicMock()
    rm.ocr.recognize = AsyncMock(return_value="")
    rm.ocr.wait_for_server = AsyncMock()
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    ctx = PipelineContext(
        file_states=[],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(ocr_backend="serve-http", memory_mode="balanced", llm_backend="local"),
        signals=StageSignals(any_needs_ocr=True),
    )
    await OcrStage()._run(ctx)
    rm.shutdown_ocr.assert_awaited_once()


@pytest.mark.asyncio
async def test_progress_ocr_end_runs_even_on_error(monkeypatch):
    """``ocr_end`` (which stops the Rich Live progress) must run even when
    the run path raises — otherwise the terminal stays in alt-screen mode."""
    from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
    from bibr.pipeline.stages.ocr import OcrStage

    async def fake_remote(self, ctx, ocr_fn):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(OcrStage, "_run_remote", fake_remote)

    progress = NullProgress()
    progress.ocr_start = MagicMock()
    progress.ocr_end = MagicMock()
    progress.ocr_region_done = MagicMock()

    rm = MagicMock()
    rm.memory_mode = "balanced"
    rm.ocr.recognize = AsyncMock(return_value="")
    rm.ocr.wait_for_server = AsyncMock()
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    ctx = PipelineContext(
        file_states=[],
        progress=progress,
        resources=rm,
        config=RunConfig(ocr_backend="serve-http"),
        signals=StageSignals(any_needs_ocr=True),
    )
    with pytest.raises(RuntimeError):
        await OcrStage()._run(ctx)
    progress.ocr_end.assert_called_once()


@pytest.mark.asyncio
async def test_ocr_init_failure_marks_all_files_errored_without_raising():
    """OCR backend init failure (missing model, unreachable URL) must fail
    per-file via set_error, not crash the whole chunk."""
    fs1 = FileState(path=Path("a.pdf"))
    fs2 = FileState(path=Path("b.pdf"))
    rm = MagicMock()
    rm.await_ocr = AsyncMock(side_effect=RuntimeError("model not found"))
    rm.shutdown_ocr = AsyncMock(return_value=None)

    await OcrStage().run(_ctx([fs1, fs2], resources=rm))

    for fs in (fs1, fs2):
        assert fs.error is not None
        assert fs.error_code == "ocr_failed"
        assert fs.failed_stage == "ocr"
        assert isinstance(fs.original_error, RuntimeError)


@pytest.mark.asyncio
async def test_wait_for_server_failure_errors_files_without_unpaired_ocr_end():
    """wait_for_server failure happens before ocr_start; ocr_end must not be
    called without a matching ocr_start."""
    fs = FileState(path=Path("a.pdf"))
    rm = MagicMock()
    rm.await_ocr = AsyncMock(return_value=None)
    rm.ocr.wait_for_server = AsyncMock(side_effect=RuntimeError("server unreachable"))
    rm.shutdown_ocr = AsyncMock(return_value=None)

    progress = NullProgress()
    progress.ocr_start = MagicMock()
    progress.ocr_end = MagicMock()

    ctx = PipelineContext(
        file_states=[fs],
        progress=progress,
        resources=rm,
        config=RunConfig(),
        signals=StageSignals(any_needs_ocr=True, preloading_ocr=False),
    )
    await OcrStage().run(ctx)

    assert fs.error is not None
    assert fs.error_code == "ocr_failed"
    progress.ocr_start.assert_not_called()
    progress.ocr_end.assert_not_called()


@pytest.mark.asyncio
async def test_process_chunk_continues_after_ocr_init_failure():
    """process_chunk must not raise when shared OCR init fails — the per-file
    error model carries the failure to the caller."""
    from bibr.config import GlobalSettings
    from bibr.pipeline.pipeline import Pipeline

    class _StandaloneOcrStage:
        name = "ocr"

        async def run(self, ctx):
            await OcrStage().run(ctx)

    class _P(Pipeline):
        def __init__(self, resources, config):
            super().__init__(
                stages=[_StandaloneOcrStage()],
                resources=resources,
                config=config,
                settings=GlobalSettings(),
            )

    rm = MagicMock()
    rm.await_ocr = AsyncMock(side_effect=RuntimeError("backend down"))
    rm.shutdown_ocr = AsyncMock(return_value=None)
    fs = FileState(path=Path("a.pdf"))

    await _P(rm, RunConfig()).process_chunk([fs])

    assert fs.error is not None
    assert fs.error_code == "ocr_failed"


@pytest.mark.asyncio
async def test_ocr_init_failure_not_retried_within_chunk():
    """A hard OCR-init failure records a chunk-scoped marker so later windows
    fast-fail without re-running the (slow, doomed) engine constructor once per
    file. The interleaved stage replaces file_states but shares one signals
    object."""
    fs1 = FileState(path=Path("a.pdf"))
    fs2 = FileState(path=Path("b.pdf"))
    rm = MagicMock()
    rm.await_ocr = AsyncMock(side_effect=RuntimeError("model not found"))
    rm.shutdown_ocr = AsyncMock(return_value=None)

    # One StageSignals shared across both windows (dataclasses.replace keeps it
    # by reference — the whole point of using a mutable object here instead of
    # plain dataclass fields).
    signals = StageSignals(any_needs_ocr=True, preloading_ocr=False)

    await OcrStage().run(_ctx([fs1], resources=rm, signals=signals))
    await OcrStage().run(_ctx([fs2], resources=rm, signals=signals))

    # Constructor attempted exactly once, not once per file/window.
    assert rm.await_ocr.await_count == 1
    assert signals.ocr_init_error is not None
    for fs in (fs1, fs2):
        assert fs.error_code == "ocr_failed"
        assert isinstance(fs.original_error, RuntimeError)


@pytest.mark.asyncio
async def test_signals_shared_by_reference_across_dataclasses_replace():
    """CRITICAL invariant: a signal written on a ``dataclasses.replace``d
    sub-context must be visible on the original context. This is what makes
    ``StageSignals`` correct for ``InterleavedRenderOcrStage``'s per-window
    sub-contexts — plain dataclass fields on ``PipelineContext`` would NOT
    propagate, since each ``replace`` copy gets independent field slots."""
    from dataclasses import replace

    ctx = _ctx([])
    sub = replace(ctx, file_states=[FileState(path=Path("x.pdf"))])

    assert sub.signals is ctx.signals  # same object, not a copy

    sub.signals.ocr_init_error = RuntimeError("boom")
    sub.signals.defer_ocr_teardown = True

    assert ctx.signals.ocr_init_error is not None
    assert str(ctx.signals.ocr_init_error) == "boom"
    assert ctx.signals.defer_ocr_teardown is True


@pytest.mark.asyncio
async def test_remote_ocr_records_stage_time():
    """_run_remote must set fs.stage_times['ocr'] so remote/cloud-vision OCR is
    counted in the exported total_seconds — _run_local set it, _run_remote did
    not, silently undercounting the whole OCR cost for those backends."""
    fs = FileState(path=Path("a.pdf"))
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[]]
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="t"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    cfg = RunConfig(ocr_backend="glm-http", ocr_url="http://x")

    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "x"}]),
        ),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(_ctx([fs], resources=rm, config=cfg))

    assert fs.ocr_regions is not None
    assert fs.error is None
    assert "ocr" in fs.stage_times
    assert fs.stage_times["ocr"] >= 0.0


# ---------------------------------------------------------------------------
# OCR-mostly-failed detection (catches silent-engine-crash regressions)
# ---------------------------------------------------------------------------


def _fs_with_regions(name: str, regions: list[list[dict]]) -> FileState:
    from bibr.pipeline.stages.ocr import _to_typed_regions

    fs = FileState(path=Path(name))
    fs.ocr_regions = _to_typed_regions(regions)
    return fs


def test_check_ocr_success_fails_file_when_all_text_regions_empty():
    """Engine-crash scenario: text regions present, but OCR returned no content."""
    fs = _fs_with_regions(
        "broken.pdf",
        [[{"native_label": "text", "content": ""} for _ in range(10)]],
    )
    ctx = _ctx([fs])
    OcrStage._check_ocr_success(ctx)
    assert fs.error is not None
    assert fs.error_code == "ocr_mostly_failed"
    assert "0/10" in fs.error


def test_check_ocr_success_passes_when_all_regions_have_content():
    fs = _fs_with_regions(
        "ok.pdf",
        [[{"native_label": "text", "content": f"line {i}"} for i in range(5)]],
    )
    ctx = _ctx([fs])
    OcrStage._check_ocr_success(ctx)
    assert fs.error is None


def test_check_ocr_success_ignores_figure_and_skip_regions():
    """Figures legitimately have empty content; they shouldn't pull the rate down."""
    fs = _fs_with_regions(
        "figs.pdf",
        [
            [
                {"native_label": "text", "content": "hello"},
                {"native_label": "figure", "content": ""},
                {"native_label": "skip", "content": ""},
                {"native_label": "abandon", "content": ""},
            ]
        ],
    )
    ctx = _ctx([fs])
    OcrStage._check_ocr_success(ctx)
    assert fs.error is None


def test_check_ocr_success_ignores_production_skip_and_abandon_labels():
    """Real skip/abandon regions carry PP-DocLayoutV3 native labels
    (image/chart/number/...), not the task-type names. A figure-heavy page
    must not be spuriously flagged ocr_mostly_failed."""
    fs = _fs_with_regions(
        "figheavy.pdf",
        [
            [
                {"native_label": "text", "content": "hello"},
                {"native_label": "text", "content": "world"},
                # skip-task regions (never OCR'd, legitimately empty)
                {"native_label": "image", "content": ""},
                {"native_label": "chart", "content": ""},
                # abandon-task regions
                {"native_label": "number", "content": ""},
                {"native_label": "aside_text", "content": ""},
                {"native_label": "footer_image", "content": ""},
                {"native_label": "header_image", "content": ""},
            ]
        ],
    )
    ctx = _ctx([fs])
    OcrStage._check_ocr_success(ctx)
    assert fs.error is None


def test_check_ocr_success_excludes_native_text_bypass_regions():
    """Native-text-bypassed regions never went through OCR, so they must not
    pad the success denominator/numerator. A mostly-native doc whose only truly
    OCR'd region came back empty (engine crash) must still be flagged."""
    regions = [
        [
            {"native_label": "text", "content": "real text", "_native_text_used": True}
            for _ in range(9)
        ]
        + [{"native_label": "text", "content": ""}]  # the lone OCR'd region: empty
    ]
    fs = _fs_with_regions("mostly_native.pdf", regions)
    ctx = _ctx([fs])
    OcrStage._check_ocr_success(ctx)
    assert fs.error_code == "ocr_mostly_failed"
    assert "0/1" in fs.error


def test_check_ocr_success_fully_native_doc_passes():
    """A fully native-text doc invokes OCR on nothing; with every region
    excluded, ocr_needed == 0 and the gate must not fire."""
    regions = [
        [{"native_label": "text", "content": "", "_native_text_used": True} for _ in range(5)]
    ]
    fs = _fs_with_regions("fully_native.pdf", regions)
    ctx = _ctx([fs])
    OcrStage._check_ocr_success(ctx)
    assert fs.error is None


def test_check_ocr_success_threshold_zero_disables_check():
    fs = _fs_with_regions("zero.pdf", [[{"native_label": "text", "content": ""}]])
    ctx = _ctx([fs])
    ctx.settings.ocr.min_success_rate = 0.0
    OcrStage._check_ocr_success(ctx)
    assert fs.error is None


# ---------------------------------------------------------------------------
# _deduplicate_reference_regions — content-based duplicate suppression
# ---------------------------------------------------------------------------

# Mirrors data/prereg.pdf pages 22-23: PP-DocLayoutV3 emits a full-column
# ``reference`` region AND per-entry ``text`` regions over the same content.
# Both get filled, so the same references enter the parser twice. The
# per-entry provenance bboxes are too coarse for a geometric union test
# (several rows share one bbox), so duplication must be detected by content.

_REF_ENTRIES = [
    "Burke, C. J. (1953). A brief note on one-tailed tests. "
    "Psychological Bulletin, 50(5), 384-387.",
    "Chambers, C. D., Feredoes, E., Muthukumaraswamy, S. D., & Etchells, P. "
    '(2014). Instead of" playing the game" it is time to change the rules. '
    "AIMS Neuroscience, 1(1), 4-17.",
    "Jennison, C., & Turnbull, B. W. (2000). Group sequential methods with "
    "applications to clinical trials. Boca Raton: Chapman & Hall/CRC.",
]

# The page-level region's copy of the same text, with the punctuation/spacing
# mangling observed in the wild ("trialsBoca RatonChapman & Hall/CRC").
_GIANT_REF_CONTENT = (
    _REF_ENTRIES[0]
    + " "
    + _REF_ENTRIES[1]
    + " Jennison, C., & Turnbull, B. W. (2000). Group sequential methods with "
    "applications to clinical trialsBoca RatonChapman & Hall/CRC"
)


def _ref_dup_page(giant_content=_GIANT_REF_CONTENT, entries=_REF_ENTRIES):
    return [
        {
            "native_label": "reference",
            "label": "text",
            "content": giant_content,
            "bbox_2d": [112, 160, 888, 878],
        },
        *(
            {
                "native_label": "text",
                "label": "text",
                "content": e,
                # Coarse shared bbox, as produced by the native-text path.
                "bbox_2d": [113, 437, 886, 549],
            }
            for e in entries
        ),
    ]


def test_dedup_blanks_reference_region_duplicated_by_text_regions():
    from bibr.pipeline.stages.ocr import _deduplicate_reference_regions

    pages = [_ref_dup_page()]
    result = _deduplicate_reference_regions(pages)

    # Region survives (it keys the References section and the layout hint)
    # but its duplicate content is blanked.
    ref_regions = [r for r in result[0] if r["native_label"] == "reference"]
    assert len(ref_regions) == 1
    assert ref_regions[0]["content"] == ""
    # The per-entry text regions are untouched.
    text_contents = [r["content"] for r in result[0] if r["native_label"] == "text"]
    assert text_contents == _REF_ENTRIES


def test_dedup_keeps_reference_region_with_unique_content():
    from bibr.pipeline.stages.ocr import _deduplicate_reference_regions

    unique = (
        "Vazire, S. (2016). Editorial. Social Psychological and Personality "
        "Science, 7(1), 3-7. Nosek, B. A., & Lakens, D. (2014). Registered "
        "reports: A method to increase the credibility of published results."
    )
    pages = [_ref_dup_page(giant_content=unique)]
    result = _deduplicate_reference_regions(pages)

    ref_regions = [r for r in result[0] if r["native_label"] == "reference"]
    assert ref_regions[0]["content"] == unique


def test_dedup_keeps_reference_region_longer_than_text_blob():
    from bibr.pipeline.stages.ocr import _deduplicate_reference_regions

    # Reference region holds MORE than the text regions cover — blanking
    # would lose the extra entries, so it must be kept.
    extra = _GIANT_REF_CONTENT + (
        " Vazire, S. (2016). Editorial. Social Psychological and Personality "
        "Science, 7(1), 3-7. Rice, W. R., & Gaines, S. D. (1994). Heads I "
        "win, tails you lose: testing directional alternative hypotheses in "
        "ecological and evolutionary research. Trends in Ecology and "
        "Evolution, 9(6), 235-237."
    )
    pages = [_ref_dup_page(giant_content=extra)]
    result = _deduplicate_reference_regions(pages)

    ref_regions = [r for r in result[0] if r["native_label"] == "reference"]
    assert ref_regions[0]["content"] == extra


def test_dedup_blanks_reference_region_with_ocr_noise():
    from bibr.pipeline.stages.ocr import _deduplicate_reference_regions

    # A handful of character-level OCR differences must not defeat the match.
    noisy = _GIANT_REF_CONTENT.replace("Burke", "Burke,").replace("(1953)", "(l953)")
    pages = [_ref_dup_page(giant_content=noisy)]
    result = _deduplicate_reference_regions(pages)

    ref_regions = [r for r in result[0] if r["native_label"] == "reference"]
    assert ref_regions[0]["content"] == ""


def test_dedup_still_removes_reference_overlapping_reference_content():
    from bibr.pipeline.stages.ocr import _deduplicate_reference_regions

    # Pre-existing geometric pass: reference region overlapping a
    # reference_content region is removed outright.
    pages = [
        [
            {
                "native_label": "reference",
                "label": "text",
                "content": "dup",
                "bbox_2d": [100, 100, 500, 500],
            },
            {
                "native_label": "reference_content",
                "label": "text",
                "content": "dup",
                "bbox_2d": [100, 100, 500, 500],
            },
        ]
    ]
    result = _deduplicate_reference_regions(pages)
    labels = [r["native_label"] for r in result[0]]
    assert labels == ["reference_content"]


# --- per-chunk OCR teardown heuristic (balanced keeps OCR across chunks) -----
# balanced mode reloaded OCR weights every chunk; with a cloud/remote LLM there
# is no GPU consumer for the freed VRAM, so OCR now stays resident across chunks
# (torn down once at aclose). Local-LLM and aggressive still tear down per chunk.


def _ocr_rm_for_teardown():
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="text"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    return rm


async def _run_one_chunk(rm, cfg):
    fs = FileState(path=Path("x.pdf"))
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[]]
    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "hello"}]),
        ),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(_ctx([fs], resources=rm, config=cfg))


@pytest.mark.asyncio
async def test_balanced_cloud_keeps_ocr_loaded_across_chunk():
    rm = _ocr_rm_for_teardown()
    await _run_one_chunk(
        rm, RunConfig(ocr_backend="glm-llama", memory_mode="balanced", llm_backend="cloud")
    )
    rm.shutdown_ocr.assert_not_called()


@pytest.mark.asyncio
async def test_balanced_local_llm_unloads_ocr():
    rm = _ocr_rm_for_teardown()
    await _run_one_chunk(
        rm, RunConfig(ocr_backend="glm-llama", memory_mode="balanced", llm_backend="local")
    )
    rm.shutdown_ocr.assert_awaited_once()


@pytest.mark.asyncio
async def test_aggressive_unloads_ocr():
    rm = _ocr_rm_for_teardown()
    await _run_one_chunk(
        rm, RunConfig(ocr_backend="glm-llama", memory_mode="aggressive", llm_backend="cloud")
    )
    rm.shutdown_ocr.assert_awaited_once()


@pytest.mark.asyncio
async def test_keep_all_keeps_ocr_loaded():
    rm = _ocr_rm_for_teardown()
    await _run_one_chunk(
        rm, RunConfig(ocr_backend="glm-llama", memory_mode="keep_all", llm_backend="local")
    )
    rm.shutdown_ocr.assert_not_called()


def _automatic_fallback_client(prompt_log):
    client = MagicMock(loaded=True)

    async def recognize(_image, prompt):
        prompt_log.append(prompt)
        return "<fcel>A<nl>" if prompt == "Table Recognition:" else "plain text"

    client.recognize = AsyncMock(side_effect=recognize)
    client.wait_for_server = AsyncMock()
    client.shutdown = AsyncMock()
    return client


def _automatic_chunk_file(name):
    from PIL import Image

    fs = FileState(path=Path(name), file_hash=name)
    fs.page_images = [Image.new("RGB", (100, 100), "white")]
    fs.page_indices = [0]
    fs.layout_results = [
        [
            {
                "label": "text",
                "native_label": "text",
                "bbox_2d": [0, 0, 500, 500],
                "task_type": "text",
            },
            {
                "label": "table",
                "native_label": "table",
                "bbox_2d": [500, 0, 1000, 500],
                "task_type": "table",
            },
        ]
    ]
    return fs


class _AutomaticIdentityCaptureStage:
    name = "ocr"

    def __init__(self):
        self.captures = []

    async def run(self, ctx):
        from bibr.pipeline.stages.export import _build_ocr_config

        await OcrStage().run(ctx)
        self.captures.append(
            {
                "identity": ctx.scratch["ocr_runtime_identity"],
                "profile": ctx.scratch["ocr_profile"],
                "provenance": _build_ocr_config(ctx),
                "regions": ctx.file_states[0].ocr_regions,
            }
        )


def _automatic_pipeline(resources, config, stage):
    from bibr.config import snapshot_settings
    from bibr.pipeline.pipeline import Pipeline

    return Pipeline(
        stages=[stage],
        resources=resources,
        config=config,
        settings=snapshot_settings(),
    )


def _assert_glm_chunk_captures(captures, cache_store):
    assert [capture["identity"].backend for capture in captures] == [
        "glm-llama",
        "glm-llama",
    ]
    assert [capture["profile"].name for capture in captures] == ["glm", "glm"]
    assert [capture["regions"][0][1].content for capture in captures] == [
        "<fcel>A<nl>",
        "<fcel>A<nl>",
    ]
    assert [capture["regions"][0][1].raw_content for capture in captures] == [None, None]
    assert [
        (
            capture["provenance"]["ocr_backend"],
            capture["provenance"]["ocr_model"],
            capture["provenance"]["ocr_profile"],
        )
        for capture in captures
    ] == [
        ("glm-llama", "THUDM/GLM-OCR", "glm"),
        ("glm-llama", "THUDM/GLM-OCR", "glm"),
    ]
    assert [call.args[2].profile for call in cache_store.call_args_list] == ["glm", "glm"]


@pytest.mark.asyncio
async def test_automatic_glm_fallback_identity_survives_two_retained_process_chunks(
    monkeypatch,
):
    from bibr.ocr.registry import OcrBackendCandidate
    from bibr.pipeline import ocr_cache
    from bibr.pipeline.resources import ResourceManager

    candidates = (
        OcrBackendCandidate("paddle-vllm", "paddle-ocr-vl-1.6", "paddle"),
        OcrBackendCandidate("glm-llama", "THUDM/GLM-OCR", "glm"),
    )
    prompts = []
    glm_client = _automatic_fallback_client(prompts)
    rm = ResourceManager(ocr_backend="paddle")
    stage = _AutomaticIdentityCaptureStage()
    pipeline = _automatic_pipeline(
        rm,
        RunConfig(ocr_backend="paddle", memory_mode="balanced", llm_backend="cloud"),
        stage,
    )
    requested_names = []

    def resolve(name, settings):  # noqa: ARG001
        requested_names.append(name)
        return candidates

    monkeypatch.setattr("bibr.ocr.registry.resolve_backend_candidates", resolve)

    with (
        patch.object(
            rm,
            "_create_ocr_client_for",
            side_effect=[RuntimeError("Paddle unavailable"), glm_client],
        ) as create,
        patch.object(ocr_cache, "is_enabled", return_value=True),
        patch.object(ocr_cache, "load", return_value=None),
        patch.object(ocr_cache, "store") as cache_store,
    ):
        await pipeline.process_chunk([_automatic_chunk_file("first.pdf")])
        await pipeline.process_chunk([_automatic_chunk_file("second.pdf")])

    assert [call.args[0].backend for call in create.call_args_list] == [
        "paddle-vllm",
        "glm-llama",
    ]
    assert requested_names == ["paddle"]
    assert prompts == [
        "Text Recognition:",
        "Table Recognition:",
        "Text Recognition:",
        "Table Recognition:",
    ]
    _assert_glm_chunk_captures(stage.captures, cache_store)


@pytest.mark.asyncio
async def test_automatic_glm_fallback_restarts_chain_after_unloaded_process_chunk(
    monkeypatch,
):
    from bibr.ocr.registry import OcrBackendCandidate
    from bibr.pipeline import ocr_cache
    from bibr.pipeline.resources import ResourceManager

    candidates = (
        OcrBackendCandidate("paddle-vllm", "paddle-ocr-vl-1.6", "paddle"),
        OcrBackendCandidate("glm-llama", "THUDM/GLM-OCR", "glm"),
    )
    prompts = []
    first_glm = _automatic_fallback_client(prompts)
    second_glm = _automatic_fallback_client(prompts)
    rm = ResourceManager(ocr_backend="paddle")
    stage = _AutomaticIdentityCaptureStage()
    pipeline = _automatic_pipeline(
        rm,
        RunConfig(ocr_backend="paddle", memory_mode="aggressive", llm_backend="cloud"),
        stage,
    )
    requested_names = []

    def resolve(name, settings):  # noqa: ARG001
        requested_names.append(name)
        if name == "paddle":
            return candidates
        return (OcrBackendCandidate("glm-llama", "THUDM/GLM-OCR", "glm"),)

    monkeypatch.setattr("bibr.ocr.registry.resolve_backend_candidates", resolve)

    with (
        patch.object(
            rm,
            "_create_ocr_client_for",
            side_effect=[
                RuntimeError("Paddle unavailable"),
                first_glm,
                RuntimeError("Paddle still unavailable"),
                second_glm,
            ],
        ) as create,
        patch.object(ocr_cache, "is_enabled", return_value=True),
        patch.object(ocr_cache, "load", return_value=None),
        patch.object(ocr_cache, "store") as cache_store,
    ):
        await pipeline.process_chunk([_automatic_chunk_file("first.pdf")])
        assert rm.ocr is None
        assert rm.ocr_runtime_identity is None
        await pipeline.process_chunk([_automatic_chunk_file("second.pdf")])

    assert [call.args[0].backend for call in create.call_args_list] == [
        "paddle-vllm",
        "glm-llama",
        "paddle-vllm",
        "glm-llama",
    ]
    assert requested_names == ["paddle", "paddle"]
    assert prompts == [
        "Text Recognition:",
        "Table Recognition:",
        "Text Recognition:",
        "Table Recognition:",
    ]
    first_glm.shutdown.assert_awaited_once()
    second_glm.shutdown.assert_awaited_once()
    _assert_glm_chunk_captures(stage.captures, cache_store)


@pytest.mark.asyncio
async def test_automatic_selector_refreshes_identity_when_retained_client_dies_between_chunks(
    monkeypatch,
):
    from bibr.ocr.registry import OcrBackendCandidate
    from bibr.pipeline import ocr_cache
    from bibr.pipeline.resources import ResourceManager

    candidates = (
        OcrBackendCandidate("paddle-vllm", "paddle-ocr-vl-1.6", "paddle"),
        OcrBackendCandidate("glm-llama", "THUDM/GLM-OCR", "glm"),
    )
    prompts = []
    first_glm = _automatic_fallback_client(prompts)
    replacement_paddle = _automatic_fallback_client(prompts)
    rm = ResourceManager(ocr_backend="paddle")
    stage = _AutomaticIdentityCaptureStage()
    pipeline = _automatic_pipeline(
        rm,
        RunConfig(ocr_backend="paddle", memory_mode="balanced", llm_backend="cloud"),
        stage,
    )
    requested_names = []

    def resolve(name, settings):  # noqa: ARG001
        requested_names.append(name)
        return candidates

    monkeypatch.setattr("bibr.ocr.registry.resolve_backend_candidates", resolve)

    with (
        patch.object(
            rm,
            "_create_ocr_client_for",
            side_effect=[
                RuntimeError("Paddle unavailable"),
                first_glm,
                replacement_paddle,
            ],
        ) as create,
        patch.object(ocr_cache, "is_enabled", return_value=True),
        patch.object(ocr_cache, "load", return_value=None),
        patch.object(ocr_cache, "store") as cache_store,
    ):
        await pipeline.process_chunk([_automatic_chunk_file("first.pdf")])
        first_glm.loaded = False
        await pipeline.process_chunk([_automatic_chunk_file("second.pdf")])

    assert [call.args[0].backend for call in create.call_args_list] == [
        "paddle-vllm",
        "glm-llama",
        "paddle-vllm",
    ]
    assert requested_names == ["paddle", "paddle"]
    assert prompts == [
        "Text Recognition:",
        "Table Recognition:",
        "OCR:",
        "Table Recognition:",
    ]
    assert [capture["identity"].backend for capture in stage.captures] == [
        "glm-llama",
        "paddle-vllm",
    ]
    assert [capture["profile"].name for capture in stage.captures] == ["glm", "paddle"]
    assert [capture["regions"][0][1].content for capture in stage.captures] == [
        "<fcel>A<nl>",
        "<table><tr><td>A</td></tr></table>",
    ]
    assert [capture["regions"][0][1].raw_content for capture in stage.captures] == [
        None,
        "<fcel>A<nl>",
    ]
    assert [
        (
            capture["provenance"]["ocr_backend"],
            capture["provenance"]["ocr_model"],
            capture["provenance"]["ocr_profile"],
        )
        for capture in stage.captures
    ] == [
        ("glm-llama", "THUDM/GLM-OCR", "glm"),
        ("paddle-vllm", "paddle-ocr-vl-1.6", "paddle"),
    ]
    assert [call.args[2].backend for call in cache_store.call_args_list] == [
        "glm-llama",
        "paddle-vllm",
    ]


# ---------------------------------------------------------------------------
# Corrupted-OCR retry: a region whose OCR text is riddled with control chars
# (the vllm-mlx --mllm NUL failure mode) is re-OCR'd once; the cleaner of the
# two results wins. STX soft-hyphen marks are deliberate GLM-OCR output and
# must not trigger a retry.
# ---------------------------------------------------------------------------


def _corruption_page():
    from PIL import Image

    img = Image.new("RGB", (100, 100), "white")
    regions = [{"label": "text", "bbox_2d": [0, 0, 500, 500], "task_type": "text"}]
    return img, regions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile_name", "expected_prompt"),
    [("paddle", "OCR:"), ("glm", "Text Recognition:")],
)
async def test_ocr_page_regions_uses_resolved_profile_prompt(profile_name, expected_prompt):
    from bibr.ocr.profiles import GLM_PROFILE, PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()
    prompts: list[str] = []

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        prompts.append(prompt)
        return "result"

    profile = {"paddle": PADDLE_PROFILE, "glm": GLM_PROFILE}[profile_name]
    await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None, profile=profile)

    assert prompts == [expected_prompt]


@pytest.mark.asyncio
async def test_paddle_output_is_normalized_before_region_postprocessing():
    from PIL import Image

    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img = Image.new("RGB", (100, 100), "white")
    regions = [
        {"label": "display_formula", "bbox_2d": [0, 0, 500, 500], "task_type": "formula"},
        {"label": "table", "bbox_2d": [500, 0, 1000, 500], "task_type": "table"},
    ]
    outputs = iter(["$$\nx^2\n$$", "<fcel>A<ecel>B<nl>"])

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        return next(outputs)

    out = await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None, profile=PADDLE_PROFILE)

    assert out[0]["content"] == "x^2"
    assert out[0]["_raw_ocr_content"] == "$$\nx^2\n$$"
    assert out[1]["content"] == "<table><tr><td>A</td><td>B</td></tr></table>"
    assert out[1]["_raw_ocr_content"] == "<fcel>A<ecel>B<nl>"


@pytest.mark.asyncio
async def test_malformed_paddle_otsl_warns_but_keeps_canonical_content():
    from PIL import Image

    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    warnings: list[str] = []
    img = Image.new("RGB", (100, 100), "white")
    regions = [{"label": "table", "bbox_2d": [0, 0, 500, 500], "task_type": "table"}]

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        return "<lcel>orphan<nl>"

    out = await ocr_page_regions(
        img,
        regions,
        0,
        "f.pdf",
        ocr_fn,
        None,
        profile=PADDLE_PROFILE,
        warning_sink=warnings.append,
    )

    assert out[0]["content"] == "<table><tr><td>orphan</td></tr></table>"
    assert warnings == [
        "OCR table output incomplete (page 1, region 0, finish_reason=unknown, "
        "reasons: malformed_structure)",
        "Malformed Paddle OTSL: left continuation has no left anchor",
    ]


@pytest.mark.asyncio
async def test_length_finish_reason_is_retained_and_warned_with_region_context():
    from PIL import Image

    from bibr.ocr.backend import OcrText
    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    warnings: list[str] = []
    img = Image.new("RGB", (100, 100), "white")
    regions = [{"label": "table", "bbox_2d": [0, 0, 500, 500], "task_type": "table"}]

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        return OcrText("<fcel>partial<nl>", finish_reason="length")

    out = await ocr_page_regions(
        img,
        regions,
        0,
        "f.pdf",
        ocr_fn,
        None,
        profile=PADDLE_PROFILE,
        warning_sink=warnings.append,
    )

    assert out[0]["_ocr_finish_reason"] == "length"
    assert warnings == [
        "OCR table output incomplete (page 1, region 0, finish_reason=length, "
        "reasons: finish_reason_length)"
    ]


@pytest.mark.asyncio
async def test_stopped_incomplete_paddle_table_warns_before_tolerant_normalization():
    from PIL import Image

    from bibr.ocr.backend import OcrText
    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    warnings: list[str] = []
    img = Image.new("RGB", (100, 100), "white")
    regions = [{"label": "table", "bbox_2d": [0, 0, 500, 500], "task_type": "table"}]

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        return OcrText("<fcel>A<fcel>B", finish_reason="stop")

    out = await ocr_page_regions(
        img,
        regions,
        0,
        "f.pdf",
        ocr_fn,
        None,
        profile=PADDLE_PROFILE,
        warning_sink=warnings.append,
    )

    assert out[0]["content"] == "<table><tr><td>A</td><td>B</td></tr></table>"
    assert out[0]["_raw_ocr_content"] == "<fcel>A<fcel>B"
    assert out[0]["_ocr_finish_reason"] == "stop"
    assert warnings == [
        "OCR table output incomplete (page 1, region 0, finish_reason=stop, "
        "reasons: missing_terminal_nl)"
    ]


@pytest.mark.asyncio
async def test_stopped_complete_paddle_table_retains_finish_reason_without_warning():
    from PIL import Image

    from bibr.ocr.backend import OcrText
    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    warnings: list[str] = []
    img = Image.new("RGB", (100, 100), "white")
    regions = [{"label": "table", "bbox_2d": [0, 0, 500, 500], "task_type": "table"}]

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        return OcrText("<fcel>A<nl>", finish_reason="stop")

    out = await ocr_page_regions(
        img,
        regions,
        0,
        "f.pdf",
        ocr_fn,
        None,
        profile=PADDLE_PROFILE,
        warning_sink=warnings.append,
    )

    assert out[0]["_ocr_finish_reason"] == "stop"
    assert warnings == []


@pytest.mark.asyncio
async def test_stage_propagates_normalization_warnings_to_file_state():
    from PIL import Image

    fs = FileState(path=Path("table.pdf"))
    fs.page_images = [Image.new("RGB", (100, 100), "white")]
    fs.page_indices = [0]
    fs.layout_results = [[{"label": "table", "bbox_2d": [0, 0, 500, 500], "task_type": "table"}]]
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="<lcel>orphan<nl>"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server

    await OcrStage().run(_ctx([fs], resources=rm, config=RunConfig(ocr_profile="paddle")))

    assert fs.ocr_regions is not None
    assert fs.warnings == [
        "OCR table output incomplete (page 1, region 0, finish_reason=unknown, "
        "reasons: malformed_structure)",
        "Malformed Paddle OTSL: left continuation has no left anchor",
    ]


@pytest.mark.asyncio
async def test_stage_uses_settings_backend_when_local_run_config_backend_is_none():
    """Library/default construction may defer the backend to owned settings."""
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value=""), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    ctx = _ctx([], resources=rm, config=RunConfig(ocr_backend=None))

    await OcrStage()._run(ctx)

    assert ctx.scratch["ocr_profile"].name == "paddle"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["gemini", "openai", "anthropic"])
async def test_stage_uses_glm_compatibility_profile_for_cloud_ocr(backend):
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value=""), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    ctx = _ctx([], resources=rm, config=RunConfig(ocr_backend=backend))

    await OcrStage()._run(ctx)

    assert ctx.scratch["ocr_profile"].name == "glm"


@pytest.mark.asyncio
async def test_native_text_region_has_no_raw_ocr_diagnostic():
    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()
    regions[0].update(content="native text", _native_text_used=True)

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        raise AssertionError("native text must bypass OCR")

    out = await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None, profile=PADDLE_PROFILE)

    assert out[0]["content"] == "native text"
    assert "_raw_ocr_content" not in out[0]


@pytest.mark.asyncio
async def test_pua_fallback_calls_ocr_once_and_preserves_all_variants():
    from PIL import Image

    from bibr.ocr.profiles import GLM_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    fixture = Path(__file__).parent.parent / "fixtures" / "native_text_pua.json"
    source = json.loads(fixture.read_text())
    clean, corrupt = source["regions"]
    regions = [
        {
            "label": clean["label"],
            "bbox_2d": clean["bbox_2d"],
            "content": clean["native_text"],
            "_native_text_used": True,
        },
        {
            "label": corrupt["label"],
            "bbox_2d": corrupt["bbox_2d"],
            "_native_text_candidate": corrupt["native_text"],
            "_native_text_rejection_reason": "private_use",
        },
    ]
    calls = 0

    async def ocr_fn(_cropped, _prompt):
        nonlocal calls
        calls += 1
        return corrupt["ocr_response"]

    out = await ocr_page_regions(
        Image.new("RGB", (1000, 1000)), regions, 2, "synthetic-pua.pdf", ocr_fn, profile=GLM_PROFILE
    )

    assert calls == 1
    assert out[0]["content"] == clean["native_text"]
    assert out[0]["_native_text_used"] is True
    assert "_raw_ocr_content" not in out[0]
    assert out[1]["content"] == corrupt["ocr_response"]
    assert out[1]["_raw_ocr_content"] == corrupt["ocr_response"]
    assert out[1]["_native_text_candidate"] == corrupt["native_text"]
    assert out[1]["_native_text_rejection_reason"] == "private_use"


def test_postprocess_does_not_merge_away_pua_fallback_provenance():
    from bibr.pipeline.stages.ocr import _postprocess_ocr_regions

    fallback = {
        "index": 1,
        "native_label": "text",
        "label": "text",
        "content": "dology recovered by OCR",
        "bbox_2d": [0, 10, 100, 20],
        "_raw_ocr_content": "dology recovered by OCR",
        "_native_text_candidate": "dology recovered by native ",
        "_native_text_rejection_reason": "private_use",
    }
    pages = [
        [
            {
                "index": 0,
                "native_label": "text",
                "label": "text",
                "content": "metho-",
                "bbox_2d": [0, 0, 100, 10],
            },
            fallback,
        ]
    ]

    out = _postprocess_ocr_regions(pages)

    assert len(out[0]) == 2
    assert out[0][1]["content"] == fallback["content"]
    assert out[0][1]["_native_text_candidate"] == fallback["_native_text_candidate"]
    assert out[0][1]["_native_text_rejection_reason"] == "private_use"


def test_postprocess_does_not_blank_pua_reference_fallback_as_duplicate():
    from bibr.pipeline.stages.ocr import _postprocess_ocr_regions

    canonical = "Lau, Ernst (1927). A recovered reference with sufficient identifying detail."
    pages = [
        [
            {
                "index": 0,
                "native_label": "text",
                "label": "text",
                "content": canonical,
                "bbox_2d": [0, 0, 100, 10],
            },
            {
                "index": 1,
                "native_label": "reference",
                "label": "reference",
                "content": canonical,
                "bbox_2d": [0, 10, 100, 20],
                "_raw_ocr_content": canonical,
                "_native_text_candidate": canonical.replace("1927", ""),
                "_native_text_rejection_reason": "private_use",
            },
            {
                "index": 2,
                "native_label": "reference_content",
                "label": "reference_content",
                "content": "Another reference anchor.",
                "bbox_2d": [200, 10, 300, 20],
            },
        ]
    ]

    out = _postprocess_ocr_regions(pages)

    recovered = next(region for region in out[0] if region.get("_native_text_candidate"))
    assert recovered["content"] == canonical
    assert recovered["_raw_ocr_content"] == canonical


@pytest.mark.asyncio
async def test_inference_exception_does_not_attempt_a_second_backend_call():
    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()
    calls = 0

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise RuntimeError("backend unavailable")

    out = await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None, profile=PADDLE_PROFILE)

    assert calls == 1
    assert out[0]["content"] == ""


@pytest.mark.asyncio
async def test_corrupt_ocr_region_retried_once():
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()
    calls: list[int] = []

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        calls.append(1)
        return "\x00Res\x00ults\x00" if len(calls) == 1 else "Results"

    out = await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None)
    assert len(calls) == 2
    assert out[0]["content"] == "Results"


@pytest.mark.asyncio
async def test_clean_ocr_region_not_retried():
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()
    calls: list[int] = []

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        calls.append(1)
        return "Results"

    out = await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None)
    assert len(calls) == 1
    assert out[0]["content"] == "Results"


@pytest.mark.asyncio
async def test_cancelled_ocr_region_propagates():
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None)


@pytest.mark.asyncio
async def test_stx_soft_hyphen_marks_do_not_trigger_retry():
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()
    calls: list[int] = []

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        calls.append(1)
        return "off\x02line and cross\x02national text"

    await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_keeps_original_when_retry_is_worse():
    from bibr.pipeline.stages.ocr import ocr_page_regions

    img, regions = _corruption_page()
    calls: list[int] = []

    async def ocr_fn(cropped, prompt):  # noqa: ARG001
        calls.append(1)
        if len(calls) == 1:
            return "\x00Res\x00ults\x00"
        return "\x00\x00\x00\x00\x00garbage\x00"

    out = await ocr_page_regions(img, regions, 0, "f.pdf", ocr_fn, None)
    assert len(calls) == 2
    assert out[0]["content"] == "\x00Res\x00ults\x00"


def _limit_settings(*, server_wide: int, per_file: int):
    return SimpleNamespace(
        ocr=SimpleNamespace(
            max_concurrent_regions=server_wide,
            concurrent_regions_per_file=per_file,
        )
    )


def test_managed_batching_server_uses_the_server_wide_region_cap():
    """paddle-vllm is a continuous-batching server bibr owns: holding the
    client at the per-file anti-starvation cap leaves it under-subscribed
    (its vLLM admits --max-num-seqs 12 against a client cap of 6)."""
    settings = _limit_settings(server_wide=16, per_file=6)
    assert _local_region_limit(settings, "paddle-vllm") == 16


def test_serialized_engines_keep_the_per_file_region_cap():
    """MLX prefill serializes on the device and the llama.cpp OCR client
    clamps to its single slot, so these must not inherit the batching cap."""
    settings = _limit_settings(server_wide=16, per_file=6)
    assert _local_region_limit(settings, "glm-llama") == 6
    assert _local_region_limit(settings, "paddle-rapid-mlx") == 6


def test_server_wide_cap_binds_the_per_file_cap():
    """OCR_MAX_CONCURRENT_REGIONS is documented as the server-wide ceiling;
    the sequential-file path used to ignore it entirely."""
    settings = _limit_settings(server_wide=2, per_file=6)
    assert _local_region_limit(settings, "glm-llama") == 2


def test_region_limit_never_drops_below_one():
    settings = _limit_settings(server_wide=0, per_file=0)
    assert _local_region_limit(settings, "paddle-vllm") == 1
    assert _local_region_limit(settings, "glm-llama") == 1


def test_apple_silicon_autotune_still_serializes_regions():
    """compute_ocr_concurrency pins both knobs to 1 on Apple Silicon; the
    batching branch must not resurrect concurrency there."""
    settings = _limit_settings(server_wide=1, per_file=1)
    assert _local_region_limit(settings, "paddle-rapid-mlx") == 1
    assert _local_region_limit(settings, "paddle-vllm") == 1


def test_managed_batching_backend_is_not_classified_remote():
    """It stays on the sequential-file path — only its region cap changes —
    so page-image RAM is still bounded to one file."""
    assert "paddle-vllm" in CONCURRENT_MANAGED_OCR_BACKENDS
    assert "paddle-vllm" not in REMOTE_OCR_BACKENDS
    assert not _is_remote_ocr(RunConfig(ocr_backend="paddle-vllm"))


@pytest.mark.asyncio
async def test_region_cap_follows_resolved_runtime_not_requested_backend():
    """OCR_BACKEND=paddle is a selector, not a runtime. The cap must be sized
    against the candidate that actually started."""
    from bibr.ocr.profiles import OcrRuntimeIdentity

    fs = FileState(path=Path("x.pdf"))
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[]]
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="text"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server

    ctx = _ctx([fs], resources=rm, config=RunConfig(ocr_backend="paddle"))
    ctx.scratch["ocr_runtime_identity"] = OcrRuntimeIdentity(
        backend="paddle-vllm",
        model="paddle-ocr-vl-1.6",
        profile="paddle",
        normalizer_version="paddle-canonical-v1",
    )

    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "hello"}]),
        ),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
        patch(
            "bibr.pipeline.stages.ocr._local_region_limit",
            side_effect=_local_region_limit,
        ) as limit,
    ):
        await OcrStage().run(ctx)

    assert fs.error is None
    assert limit.call_args.args[1] == "paddle-vllm"
