"""Pipeline audit fixes: input-error taxonomy, OCR concurrency, page gate.

M11 (``InputValidationError`` flattened to code="unknown"), M12
(``OCR_MAX_CONCURRENT_REGIONS`` ignored on the local remote path), M13 (an
inert figure tier accepted silently), L3 (pages that failed outright were
invisible to the success gate).
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibr.exceptions import InputValidationError
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import OcrStage
from bibr.pipeline.stages.validate import ValidateStage
from bibr.pipeline.state import FileState


def _ctx(file_states, config=None, settings=None):
    kwargs = {
        "file_states": file_states,
        "progress": NullProgress(),
        "resources": MagicMock(),
        "config": config or RunConfig(),
    }
    if settings is not None:
        kwargs["settings"] = settings
    return PipelineContext(**kwargs)


class TestInputValidationTaxonomy:
    """M11: a 4xx input rejection reported as an internal processing failure."""

    async def test_rejected_input_keeps_its_own_error_code(self, tmp_path):
        path = tmp_path / "paper.tei.xml"
        path.write_bytes(b"<TEI/>")
        fs = FileState(path=path)

        with patch(
            "bibr.input.validate.validate_input_file",
            side_effect=InputValidationError("TEI XML is not a supported input"),
        ):
            await ValidateStage().run(_ctx([fs]))

        assert fs.error_code == "invalid_input"
        assert isinstance(fs.original_error, InputValidationError)

    async def test_an_unexpected_failure_is_still_unknown(self, tmp_path):
        path = tmp_path / "paper.pdf"
        path.write_bytes(b"%PDF-1.4")
        fs = FileState(path=path)

        with patch(
            "bibr.input.validate.validate_input_file",
            side_effect=RuntimeError("something broke"),
        ):
            await ValidateStage().run(_ctx([fs]))

        assert fs.error_code == "unknown"

    async def test_process_file_reraises_the_input_error_unwrapped(self):
        """It is a sibling of ProcessingError, so it fell through the wrap."""
        from bibr.config import Settings
        from bibr.pipeline.pipeline import Pipeline

        pipeline = Pipeline(
            stages=[],
            resources=MagicMock(),
            config=RunConfig(),
            settings=Settings,
        )

        async def _chunk(file_states, progress=None, config=None):
            file_states[0].set_error(
                "unsupported",
                code="invalid_input",
                stage="validate",
                exc=InputValidationError("unsupported"),
            )

        with (
            patch.object(pipeline, "process_chunk", side_effect=_chunk),
            pytest.raises(InputValidationError),
        ):
            await pipeline.process_file(Path("x.tei.xml"), content=b"<TEI/>")


class TestRegionConcurrencyCeiling:
    """M12: the server-wide ceiling did not bind on the local remote path."""

    async def test_the_server_wide_ceiling_binds(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.ocr, "max_concurrent_files", 4)
        monkeypatch.setattr(Settings.ocr, "concurrent_regions_per_file", 6)
        monkeypatch.setattr(Settings.ocr, "max_concurrent_regions", 2)

        captured = {}

        async def fake_one(fs, ctx, ocr_fn, region_sem):
            captured["value"] = region_sem._value

        stage = OcrStage()
        ctx = _ctx([FileState(path=Path("a.pdf"))])
        with patch.object(OcrStage, "_ocr_one_file", side_effect=fake_one):
            await stage._run_remote(ctx, MagicMock())

        assert captured["value"] == 2

    async def test_the_per_file_product_still_binds_when_it_is_lower(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.ocr, "max_concurrent_files", 2)
        monkeypatch.setattr(Settings.ocr, "concurrent_regions_per_file", 3)
        monkeypatch.setattr(Settings.ocr, "max_concurrent_regions", 64)

        captured = {}

        async def fake_one(fs, ctx, ocr_fn, region_sem):
            captured["value"] = region_sem._value

        stage = OcrStage()
        ctx = _ctx([FileState(path=Path("a.pdf"))])
        with patch.object(OcrStage, "_ocr_one_file", side_effect=fake_one):
            await stage._run_remote(ctx, MagicMock())

        assert captured["value"] == 6


class TestPageFailureGate:
    """L3: a wholly-failed page contributed to neither side of the ratio."""

    def _fs(self, *, attempted, failed):
        fs = FileState(path=Path("a.pdf"))
        fs.ocr_regions = [[]]
        fs.ocr_pages_attempted = attempted
        fs.ocr_pages_failed = failed
        return fs

    def test_mostly_failed_pages_fail_the_file(self):
        fs = self._fs(attempted=10, failed=9)
        OcrStage._check_ocr_success(_ctx([fs]))

        assert fs.error_code == "ocr_mostly_failed"
        assert "9/10 pages" in fs.error

    def test_a_few_failed_pages_do_not(self):
        fs = self._fs(attempted=10, failed=1)
        OcrStage._check_ocr_success(_ctx([fs]))

        assert fs.error is None

    def test_no_recorded_pages_leaves_the_region_gate_in_charge(self):
        fs = self._fs(attempted=0, failed=0)
        OcrStage._check_ocr_success(_ctx([fs]))

        assert fs.error is None


class TestInertFigureTier:
    """M13: FIG_EXTRACT=meta was accepted and then did nothing at all."""

    def test_a_non_off_tier_warns(self, caplog, monkeypatch):
        import bibr.pipeline.context as ctx_mod

        monkeypatch.setattr(ctx_mod, "_figure_tier_warned", False)
        from bibr.config import Settings

        with caplog.at_level(logging.WARNING, logger="bibr.pipeline.context"):
            RunConfig(figure_extract="meta").figure_extract_tier(Settings)

        assert any("not implemented" in r.message for r in caplog.records)

    def test_the_default_is_silent(self, caplog, monkeypatch):
        import bibr.pipeline.context as ctx_mod

        monkeypatch.setattr(ctx_mod, "_figure_tier_warned", False)
        from bibr.config import Settings

        with caplog.at_level(logging.WARNING, logger="bibr.pipeline.context"):
            RunConfig().figure_extract_tier(Settings)

        assert caplog.records == []
