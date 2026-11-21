"""Regression tests for Fix 1E pipeline stage concurrency and overlap."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.config import GlobalSettings
from bibr.ocr.registry import TEXT_LAYER_BACKEND
from bibr.paper_contents import PaperContents
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.enrich import EnrichmentStage
from bibr.pipeline.stages.parse_segment import ParseSegmentStage
from bibr.pipeline.stages.post_parse import _extract_metadata_and_equations
from bibr.pipeline.stages.render_ocr import StreamingRenderOcrStage
from bibr.pipeline.state import FileState
from bibr.processing_warnings import ProcessingWarning, WarningCode


@pytest.mark.asyncio
async def test_parse_segment_overlaps_parse_roles_and_tagger_and_preserves_warning_order() -> None:
    fs = FileState(path=Path("paper.pdf"))
    fs.ocr_regions = [[{"label": "text", "content": "Hello", "bbox_2d": [0, 0, 100, 100]}]]
    rm = MagicMock()
    rm.ensure_segmenter = MagicMock()
    rm.segmenter = MagicMock()
    rm.segmenter.segment_batch = AsyncMock(return_value=[])
    ctx = PipelineContext(
        file_states=[fs], progress=NullProgress(), resources=rm, config=RunConfig()
    )

    parse_entered = threading.Event()
    roles_entered = asyncio.Event()
    tagger_entered = asyncio.Event()

    def fake_parse(*_a, **_kw):
        parse_entered.set()
        p = MagicMock()
        p.assembler = []
        p.create_content_sections = MagicMock()
        c = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text="")
        return p, c

    async def fake_roles(*_a, warnings=None, **_kw):
        roles_entered.set()
        await asyncio.wait_for(tagger_entered.wait(), timeout=5)
        if warnings is not None:
            warnings.append(ProcessingWarning(WarningCode.OCR_REGION_FAILED, "role-warn"))
        return None

    async def fake_tagger(*_a, warnings=None, **_kw):
        tagger_entered.set()
        await asyncio.wait_for(roles_entered.wait(), timeout=5)
        if warnings is not None:
            warnings.append(ProcessingWarning(WarningCode.OCR_REGION_FAILED, "tagger-warn"))
        return None

    with (
        patch("bibr.pipeline.stages.parse_segment._parse_pdf", side_effect=fake_parse),
        patch("bibr.pipeline.stages.parse_segment._predict_front_roles", side_effect=fake_roles),
        patch("bibr.pipeline.stages.parse_segment._tag_front_matter", side_effect=fake_tagger),
    ):
        await ParseSegmentStage().run(ctx)

    assert fs.error is None
    assert parse_entered.is_set()
    assert [w.message for w in fs.warnings] == ["role-warn", "tagger-warn"]


@pytest.mark.asyncio
async def test_extract_metadata_overlaps_regex_equations_and_keeps_them_on_llm_timeout() -> None:
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text="")
    settings = GlobalSettings()
    settings.EQUATION_EXTRACTION_TIMEOUT_SECONDS = 0.05

    eq_started = threading.Event()
    meta_saw_eq_running = False

    def fake_regex(*_a, **_kw):
        eq_started.set()
        time.sleep(0.03)
        return ["eq1"]

    async def fake_meta(*_a, **_kw):
        nonlocal meta_saw_eq_running
        # Poll briefly to verify regex thread ran concurrently with meta_coro.
        for _ in range(20):
            if eq_started.is_set():
                meta_saw_eq_running = True
                break
            await asyncio.sleep(0.005)
        return MagicMock()

    async def slow_llm_fb(*_a, **_kw):
        await asyncio.sleep(1.0)
        return []

    fake_ext = MagicMock()
    fake_ext.extract_all_metadata = fake_meta
    fake_ext.validation_issues = []

    with (
        patch("bibr.extract.extractor.MetadataExtractor", return_value=fake_ext),
        patch(
            "bibr.extract.equation_extractor.EquationExtractor.extract_from_sentences",
            side_effect=fake_regex,
        ),
        patch(
            "bibr.extract.equation_extractor.EquationExtractor.extract_with_llm_fallback",
            side_effect=slow_llm_fb,
        ),
    ):
        await _extract_metadata_and_equations(
            contents, "h", False, MagicMock(), extract_equations=True, settings=settings
        )

    assert meta_saw_eq_running is True
    assert contents.equations == ["eq1"]
    assert any(
        w.code == WarningCode.EQUATION_LLM_FALLBACK_TIMEOUT for w in contents.processing_warnings
    )


@pytest.mark.asyncio
async def test_enrichment_stage_runs_enrichers_concurrently_with_deterministic_warnings() -> None:
    first_in = asyncio.Event()
    second_in = asyncio.Event()

    class _BarrierEnricher:
        def __init__(self, name: str, mine: asyncio.Event, other: asyncio.Event):
            self.name = name
            self._mine = mine
            self._other = other

        async def enrich(self, _fs):
            self._mine.set()
            await asyncio.wait_for(self._other.wait(), timeout=5)
            raise RuntimeError(f"{self.name}-err")

    fs = FileState(path=Path("p.pdf"))
    fs.paper = MagicMock()
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(crossref=True),
    )
    e1 = _BarrierEnricher("first", first_in, second_in)
    e2 = _BarrierEnricher("second", second_in, first_in)
    await EnrichmentStage([e1, e2]).run(ctx)

    # Both ran concurrently (barrier did not deadlock) and warnings folded in enricher order.
    assert [w.message for w in fs.warnings] == [
        "_BarrierEnricher failed: RuntimeError: first-err",
        "_BarrierEnricher failed: RuntimeError: second-err",
    ]


@pytest.mark.asyncio
async def test_streaming_text_layer_scales_backhalf_to_max_concurrent_post_parse() -> None:
    in_flight = 0
    peak = 0
    all_four = asyncio.Event()

    class _Noop:
        name = "noop"

        async def run(self, ctx):
            for fs in ctx.alive():
                if fs.ocr_regions is None:
                    fs.ocr_regions = []

    class _BarrierPost:
        name = "post_parse"

        async def run(self, _ctx):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            if in_flight >= 4:
                all_four.set()
            await asyncio.wait_for(all_four.wait(), timeout=5)
            in_flight -= 1

    class _DoneExport:
        name = "export"

        async def run(self, ctx):
            for fs in ctx.alive():
                fs.result_json = {"ok": True}

    files = [FileState(path=Path(f"f{i}.pdf")) for i in range(4)]
    for fs in files:
        fs.pdf_bytes = b"%PDF"

    stage = StreamingRenderOcrStage(
        layout=_Noop(),
        native_text=_Noop(),
        ocr=_Noop(),
        parse=_Noop(),
        post_parse=_BarrierPost(),
        enrich=_Noop(),
        export=_DoneExport(),
    )
    rm = MagicMock()
    rm.shutdown_ocr = AsyncMock(return_value=None)
    ctx = PipelineContext(
        file_states=files,
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(
            ocr_backend=TEXT_LAYER_BACKEND, llm_backend="cloud", memory_mode="balanced"
        ),
    )
    await stage.run(ctx)
    assert peak == 4
