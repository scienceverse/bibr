"""Tests for bibr/pipeline/progress.py — progress reporting helpers."""

from io import StringIO

import pytest

from bibr.pipeline.progress import (
    STAGES,
    NullProgress,
    ProgressTracker,
    RichProgress,
    _stage_label,
)


class TestStageLabel:
    def test_known_stages_have_humanized_labels(self):
        assert _stage_label("validate") == "Checking file"
        assert _stage_label("ocr") == "Reading text (OCR)"
        assert _stage_label("export") == "Writing output"

    def test_unknown_stage_falls_back_to_titlecase(self):
        assert _stage_label("custom_stage") == "Custom_Stage"


class TestNullProgress:
    def test_implements_protocol(self):
        np = NullProgress()
        assert isinstance(np, ProgressTracker)

    def test_all_methods_no_op(self):
        np = NullProgress()
        # None of these should raise or return anything meaningful
        assert np.stage_start("validate") is None
        assert np.stage_end("validate") is None
        assert np.ocr_start(10) is None
        assert np.ocr_region_done() is None
        assert np.ocr_end() is None


class TestRichProgress:
    def _make(self, stages=None):
        # Direct stderr-bound Console output to a buffer for inspection
        from rich.console import Console

        rp = RichProgress(stages=stages)
        rp._console = Console(file=StringIO(), force_terminal=False, no_color=True)
        return rp

    def test_uses_default_stages_when_none(self):
        rp = self._make()
        assert rp._stages == STAGES
        assert rp._total_stages == len(STAGES)

    def test_filtered_stages_renumbered(self):
        rp = self._make(stages=["validate", "ocr", "export"])
        assert rp._total_stages == 3
        assert rp._stage_index["ocr"] == 1
        assert rp._stage_index["export"] == 2

    @pytest.mark.parametrize(
        ("stage", "label"),
        [("docx", "Reading DOCX"), ("jats", "Reading JATS XML"), ("html", "Reading HTML/ePub")],
    )
    def test_native_reading_stage_is_reported(self, stage, label):
        rp = self._make(stages=["validate", stage, "parse", "export"])
        rp.stage_start(stage)
        output = rp._console.file.getvalue()
        assert "[2/4]" in output
        assert label in output

    def test_explicit_empty_stage_list_stays_empty(self):
        rp = self._make(stages=[])
        rp.stage_start("docx")
        assert rp._console.file.getvalue() == ""

    def test_stage_start_logs_position(self):
        rp = self._make(stages=["validate", "ocr"])
        rp.stage_start("ocr", detail="42 regions")
        out = rp._console.file.getvalue()
        assert "[2/2]" in out
        assert "Reading text (OCR)" in out
        assert "42 regions" in out

    def test_stage_start_unknown_stage_silent(self):
        rp = self._make(stages=["validate"])
        rp.stage_start("nonexistent")
        assert rp._console.file.getvalue() == ""

    def test_stage_end_clears_timer(self):
        rp = self._make()
        rp.stage_start("validate")
        rp.stage_end("validate")
        assert rp._stage_start_time is None

    def test_stage_end_without_start_is_safe(self):
        rp = self._make()
        rp.stage_end("validate")  # no start — should not raise

    def test_ocr_lifecycle(self):
        rp = self._make()
        rp.ocr_start(5)
        assert rp._ocr_total == 5
        assert rp._ocr_done == 0
        for _ in range(5):
            rp.ocr_region_done()
        assert rp._ocr_done == 5
        rp.ocr_end()
        assert rp._ocr_progress is None

    def test_ocr_region_done_without_start_is_safe(self):
        rp = self._make()
        rp.ocr_region_done()  # no start — should not raise

    def test_ocr_end_without_start_is_safe(self):
        rp = self._make()
        rp.ocr_end()  # no start — should not raise


class TestProtocolCompliance:
    @pytest.mark.parametrize("cls", [NullProgress, RichProgress])
    def test_implements_protocol(self, cls):
        instance = cls()
        assert isinstance(instance, ProgressTracker)
