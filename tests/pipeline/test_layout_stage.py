"""LayoutStage — page rendering + layout detection."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.layout import LayoutStage
from bibr.pipeline.state import FileState


def _ctx(file_states, resources=None, config=None):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=resources or MagicMock(),
        config=config or RunConfig(),
    )


@pytest.mark.asyncio
async def test_skips_files_with_existing_contents():
    fs = FileState(path=Path("x.docx"))
    fs.contents = MagicMock()  # native-DOCX already parsed
    rm = MagicMock()
    ctx = _ctx([fs], resources=rm)

    await LayoutStage().run(ctx)

    rm.ensure_layout.assert_not_called()
    assert ctx.signals.any_needs_ocr is False
    assert ctx.signals.preloading_ocr is False


@pytest.mark.asyncio
async def test_preloads_ocr_when_balanced_and_not_http():
    fs = FileState(path=Path("x.pdf"))
    fs.pdf_bytes = b"%PDF"
    rm = MagicMock()
    rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[[]]))
    rm.ocr = None
    cfg = RunConfig(memory_mode="balanced", ocr_backend="glm-llama")
    with (
        patch(
            "bibr.pipeline.stages.layout._iter_pdf_pages",
            return_value=iter([(0, MagicMock(width=800, height=1000))]),
        ),
    ):
        await LayoutStage().run(_ctx([fs], resources=rm, config=cfg))

    rm.start_ocr_preload.assert_called_once()


@pytest.mark.asyncio
async def test_does_not_preload_in_aggressive_mode():
    fs = FileState(path=Path("x.pdf"))
    fs.pdf_bytes = b"%PDF"
    rm = MagicMock()
    rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[[]]))
    cfg = RunConfig(memory_mode="aggressive")
    with (
        patch(
            "bibr.pipeline.stages.layout._iter_pdf_pages",
            return_value=iter([(0, MagicMock(width=800, height=1000))]),
        ),
    ):
        await LayoutStage().run(_ctx([fs], resources=rm, config=cfg))

    rm.start_ocr_preload.assert_not_called()
    rm.unload_layout.assert_called_once()


@pytest.mark.asyncio
async def test_empty_pages_sets_error():
    fs = FileState(path=Path("x.pdf"))
    fs.pdf_bytes = b"%PDF"
    rm = MagicMock()
    rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[]))
    with patch("bibr.pipeline.stages.layout._iter_pdf_pages", return_value=iter([])):
        await LayoutStage().run(_ctx([fs], resources=rm))

    assert fs.error is not None
    assert fs.error_code == "layout_failed"
    assert fs.failed_stage == "layout"
    assert fs.error_outage is False  # this PDF's own failure


@pytest.mark.asyncio
async def test_renders_next_file_while_detecting_current():
    """File B's CPU-side render must overlap file A's detect_batch call —
    otherwise batch throughput serializes render and inference end to end."""
    events: list[tuple] = []

    def fake_render(pdf_bytes, dpi, start_page, end_page, max_pixels, max_dimension):
        events.append(("render", pdf_bytes))
        return iter([(0, MagicMock(width=800, height=1000))])

    async def fake_detect(images_list):
        await asyncio.sleep(0.15)  # window in which the prefetch render must land
        events.append(("detect_done",))
        return [[]]

    fs_a = FileState(path=Path("a.pdf"))
    fs_a.pdf_bytes = b"%PDF-A"
    fs_b = FileState(path=Path("b.pdf"))
    fs_b.pdf_bytes = b"%PDF-B"
    rm = MagicMock()
    rm.layout = MagicMock(detect_batch=AsyncMock(side_effect=fake_detect))

    with (
        patch("bibr.pipeline.stages.layout._iter_pdf_pages", side_effect=fake_render),
    ):
        await LayoutStage().run(_ctx([fs_a, fs_b], resources=rm))

    assert fs_a.layout_results == [[]]
    assert fs_b.layout_results == [[]]
    assert fs_a.error is None and fs_b.error is None
    first_detect_done = events.index(("detect_done",))
    assert ("render", b"%PDF-B") in events[:first_detect_done]


class TestMaxPagesClamping:
    """Settings.pipeline.max_pages is applied inside LayoutStage."""

    @pytest.mark.asyncio
    async def test_clamps_end_page_to_max_pages(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "max_pages", 5)

        fs = FileState(path=Path("x.pdf"))
        fs.pdf_bytes = b"%PDF"
        rm = MagicMock()
        rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[[]]))
        captured = {}

        def fake_render(pdf_bytes, dpi, start_page, end_page, max_pixels, max_dimension):
            captured["start_page"] = start_page
            captured["end_page"] = end_page
            return iter([(0, MagicMock(width=800, height=1000))])

        cfg = RunConfig(start_page=0, end_page=19)
        with (
            patch("bibr.pipeline.stages.layout._iter_pdf_pages", side_effect=fake_render),
        ):
            await LayoutStage().run(_ctx([fs], resources=rm, config=cfg))

        assert captured["start_page"] == 0
        assert captured["end_page"] == 4  # 0 + 5 - 1

    @pytest.mark.asyncio
    async def test_clamps_when_no_end_page_given(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "max_pages", 3)

        fs = FileState(path=Path("x.pdf"))
        fs.pdf_bytes = b"%PDF"
        rm = MagicMock()
        rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[[]]))
        captured = {}

        def fake_render(pdf_bytes, dpi, start_page, end_page, max_pixels, max_dimension):
            captured["start_page"] = start_page
            captured["end_page"] = end_page
            return iter([(0, MagicMock(width=800, height=1000))])

        cfg = RunConfig(start_page=2, end_page=None)
        with (
            patch("bibr.pipeline.stages.layout._iter_pdf_pages", side_effect=fake_render),
        ):
            await LayoutStage().run(_ctx([fs], resources=rm, config=cfg))

        assert captured["start_page"] == 2
        assert captured["end_page"] == 4  # 2 + 3 - 1

    @pytest.mark.asyncio
    async def test_zero_max_pages_means_unlimited(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "max_pages", 0)

        fs = FileState(path=Path("x.pdf"))
        fs.pdf_bytes = b"%PDF"
        rm = MagicMock()
        rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[[]]))
        captured = {}

        def fake_render(pdf_bytes, dpi, start_page, end_page, max_pixels, max_dimension):
            captured["start_page"] = start_page
            captured["end_page"] = end_page
            return iter([(0, MagicMock(width=800, height=1000))])

        cfg = RunConfig(start_page=0, end_page=99)
        with (
            patch("bibr.pipeline.stages.layout._iter_pdf_pages", side_effect=fake_render),
        ):
            await LayoutStage().run(_ctx([fs], resources=rm, config=cfg))

        # max_pages=0 means unlimited — end_page must not be clamped.
        assert captured["end_page"] == 99


@pytest.mark.parametrize("streaming", [False, True])
async def test_layout_startup_failure_spares_native_sibling(streaming):
    from types import SimpleNamespace

    from bibr.config import GlobalSettings
    from bibr.pipeline.pipeline import Pipeline
    from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage, StreamingRenderOcrStage

    native = FileState(path=Path("paper.xml"), contents=object())
    pdf = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    rm = MagicMock()
    failure = RuntimeError("layout model unavailable")
    rm.ensure_layout.side_effect = failure

    async def export(ctx):
        for fs in ctx.alive():
            fs.result_json = {"file": fs.path.name}

    exporter = SimpleNamespace(name="export", run=export)
    if streaming:

        def noop(name):
            return SimpleNamespace(name=name, run=AsyncMock())

        stages = [
            StreamingRenderOcrStage(
                parse=noop("parse"),
                post_parse=noop("extract"),
                enrich=noop("enrich"),
                export=exporter,
            )
        ]
    else:
        stages = [InterleavedRenderOcrStage(), exporter]
    pipeline = Pipeline(
        stages=stages,
        resources=rm,
        settings=GlobalSettings(),
        config=RunConfig(ocr_backend="glm-http"),
    )
    await pipeline.process_chunk([pdf, native])
    assert native.error is None
    assert native.result_json == {"file": "paper.xml"}
    assert pdf.error_code == "layout_failed"
    assert pdf.original_error is failure
    assert pdf.error_outage is True


async def test_optional_preload_failure_defers_to_ocr_without_aborting_layout():
    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    rm = MagicMock()
    rm.ocr = None
    rm.start_ocr_preload.side_effect = RuntimeError("preload unavailable")
    rm.layout.detect_batch = AsyncMock(return_value=[[]])
    context = _ctx([fs], resources=rm, config=RunConfig(ocr_backend="glm-llama"))
    with patch("bibr.pipeline.stages.layout._iter_pdf_pages", return_value=[(0, MagicMock())]):
        await LayoutStage().run(context)
    assert fs.error is None
    assert fs.layout_results == [[]]
    assert context.signals.preloading_ocr is False
