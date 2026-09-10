"""Stage requires/produces contracts.

Each pipeline stage declares which FileState fields it consumes
(``requires``) and which it may populate (``produces``). The contract is
enforced STATICALLY at pipeline construction: every required field must be
produced by an earlier stage in the list. This turns a stage-reordering
mistake into a loud construction-time error instead of a silent None-skip
that quietly drops every file.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from bibr.pipeline.pipeline import StageContractError, validate_stage_contracts
from bibr.pipeline.plans import build_stage_plan
from bibr.pipeline.stages.classifiers import ClassifierStage
from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
from bibr.pipeline.stages.docx import DocxHandlingStage
from bibr.pipeline.stages.enrich import EnrichmentStage
from bibr.pipeline.stages.export import ExportStage
from bibr.pipeline.stages.identity import IdentityValidationStage
from bibr.pipeline.stages.jats import JatsHandlingStage
from bibr.pipeline.stages.layout import LayoutStage
from bibr.pipeline.stages.llm_server import LlmServerStage
from bibr.pipeline.stages.native_text import NativeTextStage
from bibr.pipeline.stages.ocr import OcrStage
from bibr.pipeline.stages.parse_segment import ParseSegmentStage
from bibr.pipeline.stages.post_parse import PostParseStage
from bibr.pipeline.stages.validate import ValidateStage

PRODUCTION_STAGES = [
    ValidateStage,
    DocxHandlingStage,
    JatsHandlingStage,
    LayoutStage,
    NativeTextStage,
    OcrStage,
    ClassifierStage,
    LlmServerStage,
    ParseSegmentStage,
    PostParseStage,
    IdentityValidationStage,
    CoreCheckpointStage,
    EnrichmentStage,
    ExportStage,
]


class TestDeclarations:
    def test_every_production_stage_declares_contracts(self):
        for cls in PRODUCTION_STAGES:
            assert hasattr(cls, "requires"), cls.__name__
            assert hasattr(cls, "produces"), cls.__name__
            assert isinstance(cls.requires, tuple), cls.__name__
            assert isinstance(cls.produces, tuple), cls.__name__

    def test_declared_fields_exist_on_filestate(self):
        from dataclasses import fields

        from bibr.pipeline.state import FileState

        valid = {f.name for f in fields(FileState)}
        for cls in PRODUCTION_STAGES:
            for name in (*cls.requires, *cls.produces):
                assert name in valid, f"{cls.__name__} declares unknown field {name!r}"


def _production_instances():
    return [cls(enrichers=[]) if cls is EnrichmentStage else cls() for cls in PRODUCTION_STAGES]


class TestOrderingValidation:
    def test_production_stage_order_is_valid(self):
        validate_stage_contracts(_production_instances())

    def test_missing_producer_raises(self):
        # OCR before layout: ocr requires fields only layout produces.
        with pytest.raises(StageContractError, match="OcrStage"):
            validate_stage_contracts([ValidateStage(), OcrStage(), LayoutStage()])

    def test_stage_without_declarations_is_tolerated(self):
        # Third-party / test stages without contracts are skipped, not rejected.
        class Bare:
            name = "bare"

            async def run(self, ctx):
                pass

        validate_stage_contracts([ValidateStage(), Bare()])

    def test_local_pipeline_validates_at_construction(self, monkeypatch):
        # The production orchestrator runs the check when it builds its list.
        import bibr.pipeline.pipeline as pipeline_mod

        called = {}

        def _spy(stages):
            called["stages"] = list(stages)

        monkeypatch.setattr(pipeline_mod, "validate_stage_contracts", _spy)
        from bibr.local.pipeline import LocalPipeline

        # Barrier path (managed local LLM backend): layout + native_text + ocr
        # fuse into a single InterleavedRenderOcrStage (per-window render/free
        # to cap batch RAM); the serve pipeline keeps the three separate for
        # cross-request batching. memory_mode pinned so the stage shape does
        # not depend on the host's RAM auto-detect.
        LocalPipeline(llm_backend="vllm", memory_mode="balanced")
        names = [s.name for s in called["stages"]]
        assert "render_ocr" in names
        assert "layout" not in names and "ocr" not in names
        assert names[0] == "validate" and names[-1] == "export"

        # Streaming path (cloud LLM): the back-half stages fold into the
        # terminal composite, which is also handed to the validator.
        LocalPipeline(llm_backend="cloud", memory_mode="balanced")
        names = [s.name for s in called["stages"]]
        assert names[0] == "validate" and names[-1] == "render_ocr_stream"
        assert "layout" not in names and "ocr" not in names and "export" not in names

    def test_local_and_serve_plans_share_canonical_backbone(self):
        local = build_stage_plan(mode="local", stream_backhalf=False, enrichers=[])
        serve = build_stage_plan(mode="serve", stream_backhalf=False, enrichers=[])

        local_names = [s.name for s in local if s.name != "llm_server"]
        serve_names = [s.name for s in serve]

        assert local_names == serve_names

    def test_serve_pipeline_validates_at_construction(self, monkeypatch):
        import bibr.pipeline.pipeline as pipeline_mod
        from bibr.serve.pipeline import ServePipeline

        calls = []
        monkeypatch.setattr(pipeline_mod, "validate_stage_contracts", calls.append)

        ServePipeline(
            layout=MagicMock(),
            segmenter=MagicMock(),
            http_client=MagicMock(),
            ocr_base_url="http://ocr.local",
            ocr_sem_global=asyncio.Semaphore(2),
            ocr_breaker=MagicMock(),
        )

        assert len(calls) == 1
