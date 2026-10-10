"""The scan path wired into the parse and OCR stages."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from bibr.config import snapshot_settings
from bibr.ocr.pdf_inspection import PdfPageInspection
from bibr.ocr.profiles import GLM_PROFILE
from bibr.ocr.types import OcrRegionResult
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import OcrStage
from bibr.pipeline.stages.parse_segment import ParseSegmentStage
from bibr.pipeline.state import FileState
from bibr.processing_warnings import WarningCode
from bibr.scan.consensus import Recognizer


def _settings(**pipeline):
    settings = snapshot_settings()
    return settings.model_copy(update={"pipeline": settings.pipeline.model_copy(update=pipeline)})


def _scan_inspection(n):
    pages = tuple(
        PdfPageInspection(
            index=i,
            width=612.0,
            height=792.0,
            crop_box=(0.0, 0.0, 612.0, 792.0),
            char_count=0,
        )
        for i in range(n)
    )
    return SimpleNamespace(pages=pages)


def _r(index, label, content):
    return OcrRegionResult(
        index=index, native_label=label, label=label, content=content, bbox_2d=[0, 0, 9, 9]
    )


def _scanned_file():
    fs = FileState(path=Path("scan.pdf"))
    fs.pdf_inspection = _scan_inspection(2)
    fs.ocr_regions = [
        [_r(0, "doc_title", "A Paper About Things"), _r(1, "reference", "1. Our ref.")],
        [_r(0, "doc_title", "The Next Article Entirely"), _r(1, "text", "Its intro.")],
    ]
    return fs


async def _parse(fs, settings):
    parser = MagicMock()
    parser.parse.return_value = MagicMock()
    rm = MagicMock()
    rm.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(),
        settings=settings,
    )
    with patch("bibr.structure.pdf_parser.PDFParser", return_value=parser) as PP:
        await ParseSegmentStage().run(ctx)
    return PP.call_args.args[0]


@pytest.mark.asyncio
async def test_parse_stage_splits_scanned_articles_when_enabled():
    fs = _scanned_file()
    parsed = await _parse(fs, _settings(scan_article_split=True))
    assert [[r.content for r in page] for page in parsed] == [
        ["A Paper About Things", "1. Our ref."],
        [],
    ]
    assert fs.page_kinds == {0: "scan", 1: "scan"}
    assert [w.code for w in fs.warnings] == [WarningCode.SCAN_ARTICLE_SPLIT]


@pytest.mark.asyncio
async def test_parse_stage_leaves_regions_alone_by_default():
    fs = _scanned_file()
    parsed = await _parse(fs, _settings())
    assert [len(page) for page in parsed] == [2, 2]
    assert fs.warnings == []


class _Backend:
    async def recognize(self, image, prompt):
        return "completely different reading"

    async def shutdown(self):
        pass


def test_ocr_stage_consensus_flags_disagreeing_scan_regions():
    fs = FileState(path=Path("scan.pdf"))
    fs.pdf_inspection = _scan_inspection(1)
    fs.layout_results = [[{"label": "text", "bbox_2d": [0, 0, 500, 100]}]]
    fs.page_indices = [0]
    fs.page_images = [Image.new("RGB", (1000, 1000), "white")]
    pages = [
        [
            {
                "index": 0,
                "native_label": "text",
                "label": "text",
                "content": "The primary reading",
                "bbox_2d": [0, 0, 500, 100],
            }
        ]
    ]
    second = Recognizer(role="consensus", backend=_Backend(), profile=GLM_PROFILE)
    ctx = SimpleNamespace(
        scratch={"scan_recognizers": (second, None)}, settings=snapshot_settings()
    )

    asyncio.run(OcrStage._apply_consensus(fs, ctx, pages))

    assert pages[0][0]["_ocr_consensus"]["escalated"] is True
    assert pages[0][0]["content"] == "The primary reading"
    assert [w.code for w in fs.warnings] == [WarningCode.OCR_RECOGNIZERS_DISAGREE]
    typed = OcrRegionResult.from_dict(pages[0][0])
    assert typed.ocr_consensus == pages[0][0]["_ocr_consensus"]
    assert typed.to_dict()["_ocr_consensus"] == typed.ocr_consensus


def test_ocr_stage_consensus_is_a_no_op_without_recognizers():
    fs = FileState(path=Path("scan.pdf"))
    pages = [[{"content": "x", "bbox_2d": [0, 0, 1, 1], "label": "text"}]]
    ctx = SimpleNamespace(scratch={}, settings=snapshot_settings())
    asyncio.run(OcrStage._apply_consensus(fs, ctx, pages))
    assert "_ocr_consensus" not in pages[0][0]
    assert fs.page_kinds is None


def test_page_kinds_survive_freeing_the_inspection():
    # OCR frees the inspection and layout (also on an OCR-cache hit) before
    # parse, where the split reads the page classes.
    fs = FileState(path=Path("scan.pdf"))
    fs.pdf_inspection = _scan_inspection(2)
    fs.layout_results = [[{"label": "text"}], [{"label": "text"}]]
    fs.page_indices = [0, 1]
    fs.free_pre_ocr()
    assert fs.pdf_inspection is None
    assert fs.page_kinds == {0: "scan", 1: "scan"}


@pytest.mark.asyncio
async def test_parse_stage_splits_after_ocr_freed_the_inspection():
    fs = _scanned_file()
    fs.free_pre_ocr()
    parsed = await _parse(fs, _settings(scan_article_split=True))
    assert [len(page) for page in parsed] == [2, 0]
