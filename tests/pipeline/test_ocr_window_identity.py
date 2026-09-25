"""OCR identity across per-window runs, and pending-only init failures.

``InterleavedRenderOcrStage`` drives ``OcrStage`` once per file-window with
sub-contexts sharing one ``scratch`` dict. These tests pin what every window
of a multi-window chunk must use: the concrete runtime's identity, prompts,
profile and provenance — even when the first window was covered entirely by
native text and never started the engine. They also pin that init failures
only fail files still needing OCR, on both stages.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from bibr.ocr.profiles import GLM_PROFILE, OcrRuntimeIdentity
from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import OcrStage
from bibr.pipeline.state import FileState

GLM_IDENTITY = OcrRuntimeIdentity(
    backend="glm-llama",
    model="THUDM/GLM-OCR",
    profile="glm",
    normalizer_version="glm-canonical-v1",
)

STATIC_PADDLE_IDENTITY = OcrRuntimeIdentity(
    backend="paddle",
    model="PaddlePaddle/PaddleOCR-VL-1.6",
    profile="paddle",
    normalizer_version="paddle-canonical-v1",
)

GLM_TABLE_HTML = "<table><tr><td>A</td><td>B</td></tr></table>"


def _ctx(file_states, *, resources, config=None, signals=None, scratch=None):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=resources,
        config=config or RunConfig(ocr_backend="paddle"),
        signals=signals or StageSignals(any_needs_ocr=True, preloading_ocr=False),
        scratch=scratch if scratch is not None else {},
    )


def _chain_rm(calls):
    """Fake ResourceManager for the automatic chain paddle -> glm-llama."""

    async def await_ocr():
        calls["starts"] += 1
        rm.ocr_runtime_identity = GLM_IDENTITY
        rm.ocr = client

    async def recognize(image, prompt):  # noqa: ARG001
        calls["prompts"].append(prompt)
        if prompt == GLM_PROFILE.prompts["table"]:
            return GLM_TABLE_HTML
        return "some text"

    client = SimpleNamespace(recognize=recognize, loaded=True)
    rm = SimpleNamespace(
        ocr=None,
        ocr_runtime_identity=None,
        await_ocr=await_ocr,
        shutdown_ocr=AsyncMock(return_value=None),
    )
    return rm


def _native_only_file(name="native.pdf"):
    fs = FileState(path=Path(name))
    fs.page_images = [Image.new("RGB", (64, 64))]
    fs.page_indices = [0]
    fs.layout_results = [
        [{"label": "text", "task_type": "text", "_native_text_used": True, "content": "native"}]
    ]
    return fs


def _scan_file(name="scan.pdf"):
    fs = FileState(path=Path(name))
    fs.page_images = [Image.new("RGB", (64, 64))]
    fs.page_indices = [0]
    fs.layout_results = [
        [
            {"label": "text", "task_type": "text"},
            {"label": "table", "task_type": "table", "bbox_2d": [0, 0, 1000, 1000]},
        ]
    ]
    return fs


async def _run_window(stage, ctx, fs):
    sub = replace(ctx, file_states=[fs])
    sub.scratch = ctx.scratch
    await stage.run(sub)
    return sub


@pytest.mark.asyncio
async def test_native_first_window_does_not_poison_later_windows():
    """A native-only first window must not fix the static identity in scratch.

    Window 2 then resolves the chain's concrete runtime and uses its prompts,
    profile and provenance; a GLM HTML table passes through instead of being
    mangled by OTSL decoding.
    """
    calls = {"starts": 0, "prompts": []}
    rm = _chain_rm(calls)
    ctx = _ctx([], resources=rm)
    stage = OcrStage()

    fs1 = _native_only_file()
    await _run_window(stage, ctx, fs1)
    assert fs1.error is None
    assert calls["starts"] == 0
    assert "ocr_runtime_identity" not in ctx.scratch

    fs2 = _scan_file()
    sub2 = await _run_window(stage, ctx, fs2)
    assert fs2.error is None
    assert calls["starts"] == 1
    assert ctx.scratch["ocr_runtime_identity"] == GLM_IDENTITY
    assert sub2.scratch["ocr_profile"].name == "glm"
    assert calls["prompts"] == [GLM_PROFILE.prompts["text"], GLM_PROFILE.prompts["table"]]
    regions = fs2.ocr_regions[0]
    assert regions[0].content == "some text"
    assert regions[1].content == GLM_TABLE_HTML

    from bibr.pipeline.stages.export import _build_engines

    ocr, _ = _build_engines(sub2)
    assert ocr["backend"] == "glm-llama"
    assert ocr["profile"] == "glm"


@pytest.mark.asyncio
async def test_scan_first_window_control_uses_concrete_identity():
    """Control: when the engine starts in the first window, all windows agree."""
    calls = {"starts": 0, "prompts": []}
    rm = _chain_rm(calls)
    ctx = _ctx([], resources=rm)
    stage = OcrStage()

    fs1 = _scan_file("scan.pdf")
    await _run_window(stage, ctx, fs1)
    fs2 = _native_only_file("native.pdf")
    await _run_window(stage, ctx, fs2)

    assert fs1.error is None
    assert fs2.error is None
    assert ctx.scratch["ocr_runtime_identity"] == GLM_IDENTITY
    assert calls["prompts"] == [GLM_PROFILE.prompts["text"], GLM_PROFILE.prompts["table"]]
    assert fs1.ocr_regions[0][1].content == GLM_TABLE_HTML


@pytest.mark.asyncio
async def test_engine_start_adopts_concrete_identity_over_stale_static():
    """A window needing the engine adopts the started runtime's identity even
    when scratch already holds the static fallback (e.g. seeded by an older
    revision's native-only window)."""
    calls = {"starts": 0, "prompts": []}
    rm = _chain_rm(calls)
    ctx = _ctx([], resources=rm)
    ctx.scratch["ocr_runtime_identity"] = STATIC_PADDLE_IDENTITY

    fs = _scan_file()
    sub = await _run_window(OcrStage(), ctx, fs)

    assert fs.error is None
    assert ctx.scratch["ocr_runtime_identity"] == GLM_IDENTITY
    assert sub.scratch["ocr_profile"].name == "glm"
    assert calls["prompts"] == [GLM_PROFILE.prompts["text"], GLM_PROFILE.prompts["table"]]


@pytest.mark.asyncio
async def test_explicit_backend_still_persists_static_identity():
    """Guard: non-automatic backends keep the old persist-static behaviour."""
    fs = _native_only_file()
    rm = MagicMock()
    rm.ocr = None
    rm.ocr_runtime_identity = None
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(side_effect=AssertionError("no engine for native-only"))
    ctx = _ctx([fs], resources=rm, config=RunConfig(ocr_backend="glm-llama"))

    await OcrStage().run(ctx)

    assert fs.error is None
    assert ctx.scratch["ocr_runtime_identity"].backend == "glm-llama"
    rm.await_ocr.assert_not_awaited()


@pytest.mark.asyncio
async def test_retained_runtime_shared_by_probe_and_stage():
    """Both stages reuse one retained loaded client without restarting it."""
    from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage

    calls = {"starts": 0, "prompts": []}
    rm = _chain_rm(calls)
    rm.ocr_runtime_identity = GLM_IDENTITY

    async def recognize(image, prompt):  # noqa: ARG001
        calls["prompts"].append(prompt)
        return "some text"

    rm.ocr = SimpleNamespace(recognize=recognize, loaded=True)
    cfg = RunConfig(ocr_backend="paddle")
    fs = FileState(path=Path("paper.pdf"))
    ctx = _ctx([fs], resources=rm, config=cfg)

    with (
        patch("bibr.pipeline.ocr_cache.is_enabled", return_value=True),
        patch("bibr.pipeline.ocr_cache.load_bundle", return_value=False),
    ):
        pending = await InterleavedRenderOcrStage()._probe_cache(ctx)

    assert pending == [fs]
    assert calls["starts"] == 0
    assert ctx.scratch["ocr_runtime_identity"] == GLM_IDENTITY

    fs.page_images = [Image.new("RGB", (64, 64))]
    fs.page_indices = [0]
    fs.layout_results = [[{"label": "text", "task_type": "text"}]]
    await _run_window(OcrStage(), ctx, fs)

    assert fs.error is None
    assert calls["prompts"] == [GLM_PROFILE.prompts["text"]]
    assert ctx.scratch["ocr_runtime_identity"] == GLM_IDENTITY


def _pdf_needing_ocr(name):
    fs = FileState(path=Path(name))
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[{"label": "text", "task_type": "text"}]]
    return fs


@pytest.mark.asyncio
async def test_engine_start_failure_spares_cache_hit_files():
    """An engine-start failure after the regions-cache probe fails only the
    files still pending OCR — cache hits keep their output."""
    cached = _pdf_needing_ocr("cached.pdf")
    cached.ocr_regions = []
    pending = _pdf_needing_ocr("pending.pdf")
    rm = SimpleNamespace(
        ocr=SimpleNamespace(loaded=True),
        ocr_runtime_identity=None,
        await_ocr=AsyncMock(side_effect=RuntimeError("engine missing")),
        shutdown_ocr=AsyncMock(return_value=None),
    )
    ctx = _ctx([cached, pending], resources=rm, config=RunConfig(ocr_backend="glm-llama"))

    with patch(
        "bibr.pipeline.stages.ocr.ocr_page_regions",
        AsyncMock(return_value=[{"content": "x"}]),
    ):
        await OcrStage().run(ctx)

    assert pending.error is not None
    assert pending.error_code == "ocr_failed"
    assert cached.error is None


@pytest.mark.asyncio
async def test_prior_init_error_spares_native_parses():
    """A hard init failure in an earlier window fails only OCR-needing files;
    a DOCX in the same window still proceeds (as in the interleaved probe)."""
    pdf = _pdf_needing_ocr("a.pdf")
    docx = FileState(path=Path("b.docx"))
    docx.contents = MagicMock()
    rm = SimpleNamespace(
        ocr=None,
        ocr_runtime_identity=None,
        await_ocr=AsyncMock(side_effect=AssertionError("must not re-run")),
        shutdown_ocr=AsyncMock(return_value=None),
    )
    signals = StageSignals(any_needs_ocr=True, preloading_ocr=False)
    signals.ocr_init_error = RuntimeError("earlier window")
    ctx = _ctx([pdf, docx], resources=rm, signals=signals)

    await OcrStage().run(ctx)

    assert pdf.error_code == "ocr_failed"
    assert docx.error is None
    rm.await_ocr.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_paddle_table_counts_as_unfilled():
    """Blank Paddle table output decodes to empty content, so the OCR
    success-rate gate still catches a silently degraded engine."""
    from bibr.ocr.normalization import normalize_ocr_output
    from bibr.ocr.profiles import PADDLE_PROFILE
    from bibr.pipeline.stages.ocr import _to_typed_regions

    blank = normalize_ocr_output(PADDLE_PROFILE, "table", "").content
    fs = FileState(path=Path("broken.pdf"))
    fs.ocr_regions = _to_typed_regions(
        [[{"native_label": "table", "label": "table", "content": blank} for _ in range(5)]]
    )
    ctx = _ctx([fs], resources=MagicMock())

    OcrStage._check_ocr_success(ctx)

    assert fs.error is not None
    assert fs.error_code == "ocr_mostly_failed"


def test_filled_paddle_table_still_counts_as_filled():
    """Guard: a real decoded table keeps counting toward the success rate."""
    from bibr.pipeline.stages.ocr import _to_typed_regions

    fs = FileState(path=Path("ok.pdf"))
    fs.ocr_regions = _to_typed_regions(
        [[{"native_label": "table", "label": "table", "content": GLM_TABLE_HTML}]]
    )
    ctx = _ctx([fs], resources=MagicMock())

    OcrStage._check_ocr_success(ctx)

    assert fs.error is None
