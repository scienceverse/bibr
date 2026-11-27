"""Wave 4B tests: ORT CPU arena shrinkage and StatementTagger CUDA EP integration."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bibr.config import snapshot_settings
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.statement_tagger import pipeline as st_pipeline
from bibr.statement_tagger.runtime import StatementTagger
from bibr.utils import onnx_providers
from bibr.utils.locks import LOCAL_INFERENCE_LOCK


def _lock_free_from_other_thread() -> bool:
    result: dict[str, bool] = {}

    def probe() -> None:
        acquired = LOCAL_INFERENCE_LOCK.acquire(blocking=False)
        result["acquired"] = acquired
        if acquired:
            LOCAL_INFERENCE_LOCK.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return result["acquired"]


def test_cpu_arena_shrinkage_run_option_accepted_and_matches_no_arena(monkeypatch) -> None:
    """Real ORT RunOptions accepts cpu:0 shrinkage and produces bit-identical NER outputs."""
    from bibr.ner.parser_onnx import OnnxRefParser
    from tests import test_onnx_runtime_core as core

    monkeypatch.setenv("ML_ONNX_CPU_ARENA", "false")
    parser_off = OnnxRefParser(core._bundle("ner"), device="cpu")
    out_off = parser_off.parse_batch(core.REFS)

    monkeypatch.setenv("ML_ONNX_CPU_ARENA", "true")
    monkeypatch.setenv("ML_ONNX_CPU_ARENA_SHRINK", "true")
    parser_shrink = OnnxRefParser(core._bundle("ner"), device="cpu")
    assert isinstance(parser_shrink.session, onnx_providers._ArenaShrinkingSession)
    assert (
        parser_shrink.session._run_options.get_run_config_entry(
            "memory.enable_memory_arena_shrinkage"
        )
        == "cpu:0"
    )
    out_shrink = parser_shrink.parse_batch(core.REFS)

    assert out_shrink == out_off


def test_statement_tagger_pipeline_uses_create_session_and_cuda_oom_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    bundle = write_bundle(tmp_path / "tagger")
    real_create_session = onnx_providers.create_session

    # Wrap a real CPU session to pretend it is on CUDA and trigger a GPU OOM
    # on any batch larger than 1, forcing split_on_gpu_oom -> single-item or CPU fallback.
    created: list[dict] = []

    class _FakeCudaOomSession:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.arm_oom = False
            self.oom_calls = 0

        def get_inputs(self):
            return self._inner.get_inputs()

        def get_outputs(self):
            return self._inner.get_outputs()

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def run(self, output_names, input_feed, run_options=None):
            if self.arm_oom:
                self.oom_calls += 1
                raise RuntimeError("CUDA out of memory while running statement tagger")
            return self._inner.run(output_names, input_feed, run_options)

    fake_cuda_holder: list[_FakeCudaOomSession] = []

    def fake_create_session(
        model_path, *, device=None, model_name="", disable_cpu_arena=False, **_kw
    ):
        created.append(
            {"device": device, "model_name": model_name, "disable_cpu_arena": disable_cpu_arena}
        )
        cpu_sess, _ = real_create_session(
            model_path, device="cpu", model_name=model_name, disable_cpu_arena=disable_cpu_arena
        )
        if device == "cpu":
            return cpu_sess, "cpu"
        wrapped = _FakeCudaOomSession(cpu_sess)
        fake_cuda_holder.append(wrapped)
        return wrapped, "cuda"

    st_pipeline.reset_statement_tagger()
    monkeypatch.setattr(onnx_providers, "create_session", fake_create_session)
    try:
        settings = snapshot_settings()
        settings.ml.statement_tagger_model_id = str(bundle)
        settings.ml.statement_tagger_device = None
        tagger = st_pipeline.get_statement_tagger(settings)
        assert tagger is not None
        assert tagger.device == "cuda"
        assert created[0]["device"] is None
        assert created[0]["model_name"] == "statement tagger (ONNX)"

        blocks = [
            "Methods",
            "This work was funded by a grant.",
            "Competing interests: none.",
        ]
        # Arm runtime GPU OOM after the load probe passes -> falls back to lazy CPU session.
        fake_cuda_holder[0].arm_oom = True
        got = tagger.probabilities(blocks)
        ref = StatementTagger(bundle, threads=1).probabilities(blocks)
        np.testing.assert_allclose(got, ref, atol=1e-6)
        assert fake_cuda_holder[0].oom_calls > 0
        assert any(c["device"] == "cpu" and c["disable_cpu_arena"] is True for c in created[1:])
    finally:
        st_pipeline.reset_statement_tagger()


def test_predict_statement_tags_unlocks_cpu_and_locks_cuda() -> None:
    contents = PaperContents(
        sentences=[PaperSentence(1, "This work was funded by NSF.", 1, 1, page_number=1)],
        sections=[
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(1, "Funding", 1, 0, CanonicalSection.FUNDING, 1.0, "exact_alias"),
        ],
        tables=[],
        links=[],
        sections_text={},
    )
    seen: dict[str, bool] = {}

    class _MockTagger:
        def __init__(self, key: str, providers: list[str]) -> None:
            self._key = key
            self._session = MagicMock()
            self._session.get_providers.return_value = providers

        def tag(self, texts):
            seen[self._key] = _lock_free_from_other_thread()
            return []

    st_pipeline.predict_statement_tags(
        contents, None, tagger=_MockTagger("cpu_free", ["CPUExecutionProvider"])
    )
    st_pipeline.predict_statement_tags(
        contents,
        None,
        tagger=_MockTagger("cuda_free", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    assert seen == {"cpu_free": True, "cuda_free": False}


def test_statement_tagger_reloads_when_device_setting_changes(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    bundle = write_bundle(tmp_path / "reload")
    st_pipeline.reset_statement_tagger()
    try:
        s_auto = snapshot_settings()
        s_auto.ml.statement_tagger_model_id = str(bundle)
        s_auto.ml.statement_tagger_device = None
        first = st_pipeline.get_statement_tagger(s_auto)

        s_cpu = snapshot_settings()
        s_cpu.ml.statement_tagger_model_id = str(bundle)
        s_cpu.ml.statement_tagger_device = "cpu"
        second = st_pipeline.get_statement_tagger(s_cpu)

        assert first is not None and second is not None
        assert first is not second
    finally:
        st_pipeline.reset_statement_tagger()
