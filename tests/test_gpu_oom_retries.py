"""GPU out-of-memory failures in bibr's ONNX models split the batch, then use the CPU.

Torch-free: the models are the committed fixture bundles on CPU sessions,
with their GPU session swapped for one that raises the errors ONNX Runtime
raised on a nearly full RTX 3090 (E23, 2026-10-26). The CPU last resort
then runs the real graph, so its answers must match the golden outputs.
"""

from __future__ import annotations

import logging
import threading

import pytest
from PIL import Image

from bibr.ocr.registry import GpuMemory
from bibr.utils import gpu_oom
from bibr.utils.gpu_oom import is_gpu_oom_error, is_memory_error, split_on_gpu_oom
from tests import test_onnx_runtime_core as core

# A section-classifier arena failure (E23 C/chk1 04:28:39.203).
ARENA_OOM = (
    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while "
    "running Add node. Name:'/inner/encoder/encoder/layer.0/attention/self/query/Add' "
    "Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* "
    "onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) "
    "Failed to allocate memory for requested buffer of size 25165824"
)
# The layout failure 12 ms later, which named no CUDA error at all.
_CUDA_CALL = (
    "/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:{line} "
    "std::conditional_t<THRW, void, onnxruntime::common::Status> onnxruntime::CudaCall(ERRTYPE, "
    "const char*, const char*, SUCCTYPE, const char*, const char*, int) [with ERRTYPE = "
    "cudaError; bool THRW = true; SUCCTYPE = cudaError; std::conditional_t<THRW, void, "
    "common::Status> = void] "
)
BARE_CUDA_ERROR = (
    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running "
    "Conv node. Name:'/inner/model/backbone/model/encoder/stages.0/blocks.0/layers.0/"
    "convolution/Conv' Status Message: " + _CUDA_CALL.format(line=122) + _CUDA_CALL.format(line=114)
)
ILLEGAL_ADDRESS = (
    "[ONNXRuntimeError] : 1 : FAIL : "
    + _CUDA_CALL.format(line=129)
    + _CUDA_CALL.format(line=121)
    + "CUDA failure 700: an illegal memory access was encountered ; GPU=0 ; hostname=fridge ; "
    "file=/onnxruntime_src/onnxruntime/core/providers/cuda/gpu_data_transfer.cc ; line=73 ; "
    "expr=cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, stream);"
)
CUDNN_EXECUTION_FAILED = (
    "[ONNXRuntimeError] : 1 : FAIL : CUDNN failure 8: CUDNN_STATUS_EXECUTION_FAILED ; GPU=0 ; "
    "hostname=fridge ; expr=cudnnConvolutionForward(...); [with ERRTYPE = cudnnStatus_t]"
)


def _nearly_full(monkeypatch, free_gb: float | None) -> None:
    reading = None if free_gb is None else GpuMemory(total_gb=23.56, free_gb=free_gb, own_gb=2.0)
    monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: reading)


class TestRecognition:
    @pytest.mark.parametrize(
        "message",
        [
            ARENA_OOM,
            "CUDA out of memory. Tried to allocate 20.00 MiB",
            "CUDA failure 2: out of memory ; GPU=0",
            "CUDNN failure 4: CUDNN_STATUS_ALLOC_FAILED ; GPU=0",
            "CUBLAS failure 3: CUBLAS_STATUS_ALLOC_FAILED ; GPU=0",
            "cudaErrorMemoryAllocation",
            "CUDA_ERROR_OUT_OF_MEMORY",
        ],
    )
    def test_allocation_failures(self, message):
        assert is_gpu_oom_error(RuntimeError(message))

    @pytest.mark.parametrize("message", [ILLEGAL_ADDRESS, CUDNN_EXECUTION_FAILED, "bad input"])
    def test_real_faults_are_not_oom_even_on_a_full_card(self, monkeypatch, message):
        _nearly_full(monkeypatch, 0.1)
        gpu_oom._note_oom()
        assert not is_gpu_oom_error(RuntimeError(message))

    def test_bare_cuda_error_on_a_nearly_full_card(self, monkeypatch):
        _nearly_full(monkeypatch, 0.3)
        assert is_gpu_oom_error(RuntimeError(BARE_CUDA_ERROR))

    def test_bare_cuda_error_right_after_an_arena_failure(self, monkeypatch):
        _nearly_full(monkeypatch, None)
        assert is_gpu_oom_error(RuntimeError(ARENA_OOM))
        assert is_gpu_oom_error(RuntimeError(BARE_CUDA_ERROR))

    @pytest.mark.parametrize("free_gb", [None, 8.0])
    def test_bare_cuda_error_with_room_is_a_real_error(self, monkeypatch, free_gb):
        _nearly_full(monkeypatch, free_gb)
        assert not is_gpu_oom_error(RuntimeError(BARE_CUDA_ERROR))

    def test_an_old_arena_failure_no_longer_counts(self, monkeypatch):
        _nearly_full(monkeypatch, None)
        monkeypatch.setattr(gpu_oom, "_last_oom_at", -1e9)
        assert not is_gpu_oom_error(RuntimeError(BARE_CUDA_ERROR))

    def test_a_failing_probe_is_not_oom(self, monkeypatch):
        def broken():
            raise OSError("nvidia-smi hung")

        monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", broken)
        assert not is_gpu_oom_error(RuntimeError(BARE_CUDA_ERROR))


class TestCpuSessions:
    """ORT's CPU arena fails in the same words as its CUDA arena; that is no GPU OOM."""

    def test_a_cpu_arena_failure_is_not_a_gpu_oom(self, monkeypatch):
        _nearly_full(monkeypatch, None)
        assert not is_gpu_oom_error(RuntimeError(ARENA_OOM), on_gpu=False)
        # Nor does it open the window for a bare cudaError on the GPU.
        assert gpu_oom._last_oom_at is None
        assert not is_gpu_oom_error(RuntimeError(BARE_CUDA_ERROR))

    def test_a_cpu_session_still_splits_on_its_arena_failures(self):
        assert is_memory_error(RuntimeError(ARENA_OOM), on_gpu=False)
        assert is_memory_error(RuntimeError("out of memory"), on_gpu=False)
        assert not is_memory_error(RuntimeError(BARE_CUDA_ERROR), on_gpu=False)
        assert gpu_oom._last_oom_at is None

    def test_split_raises_a_cpu_sessions_failure_at_once(self, caplog):
        calls = []

        def run(items):
            calls.append(len(items))
            raise RuntimeError(ARENA_OOM)

        with (
            caplog.at_level(logging.WARNING, logger="bibr.utils.gpu_oom"),
            pytest.raises(RuntimeError, match="BFCArena"),
        ):
            split_on_gpu_oom([1, 2, 3], run, label="Model", on_gpu=False)
        assert calls == [3]
        assert not [r for r in caplog.records if "GPU out of memory" in r.getMessage()]


class TestSplit:
    def test_halves_until_it_fits_and_keeps_order(self, caplog):
        calls = []

        def run(items):
            calls.append(len(items))
            if len(items) > 2:
                raise RuntimeError(ARENA_OOM)
            return [item * 10 for item in items]

        with caplog.at_level(logging.WARNING, logger="bibr.utils.gpu_oom"):
            assert split_on_gpu_oom([1, 2, 3, 4, 5], run, label="Model") == [10, 20, 30, 40, 50]
        assert calls == [5, 2, 3, 1, 2]
        assert any(
            "Model GPU out of memory with batch size 5 (RuntimeError), retrying as 2 + 3"
            in r.getMessage()
            for r in caplog.records
        )

    def test_single_input_goes_to_the_cpu(self, caplog):
        def run(items):
            raise RuntimeError(ARENA_OOM)

        with caplog.at_level(logging.WARNING, logger="bibr.utils.gpu_oom"):
            out = split_on_gpu_oom(
                [1, 2], run, label="Model", run_on_cpu=lambda xs: [-x for x in xs]
            )
        assert out == [-1, -2]
        assert sum("on a single input" in r.getMessage() for r in caplog.records) == 2

    def test_single_input_without_a_cpu_reraises(self):
        def run(items):
            raise RuntimeError(ARENA_OOM)

        with pytest.raises(RuntimeError, match="BFCArena"):
            split_on_gpu_oom([1, 2], run, label="Model")

    def test_other_errors_propagate_untouched(self):
        def run(items):
            raise RuntimeError(ILLEGAL_ADDRESS)

        with pytest.raises(RuntimeError, match="illegal memory access"):
            split_on_gpu_oom([1, 2], run, label="Model", run_on_cpu=lambda xs: xs)


class _FailingGpuSession:
    """Stands in for a CUDA session: raises *message* for batches over *fits*."""

    def __init__(self, real, message: str, *, fits: int = 0, batch_input: str | None = None):
        self._real = real
        self._message = message
        self._fits = fits
        self._batch_input = batch_input
        self.batches: list[int] = []

    def get_inputs(self):
        return self._real.get_inputs()

    def run(self, output_names, feeds, run_options=None):
        name = self._batch_input or next(iter(feeds))
        batch = int(feeds[name].shape[0])
        self.batches.append(batch)
        if batch > self._fits:
            raise RuntimeError(self._message)
        return self._real.run(output_names, feeds)


def _on_failing_gpu(model, message: str, *, fits: int = 0):
    model.device = "cuda"
    model.session = _FailingGpuSession(model.session, message, fits=fits)
    return model.session


class TestSectionClassifier:
    """A classifier OOM sent every heading of the paper to the alias/LLM tier."""

    def _golden(self, preds):
        return [(p.canonical_type.value, p.is_top_level, round(p.score, 4)) for p in preds]

    def test_splits_then_matches_golden(self):
        from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

        model = OnnxSectionClassifierModel(core._bundle("section"), "cpu")
        session = _on_failing_gpu(model, ARENA_OOM, fits=1)
        preds = model.classify_batch(core.CONTEXTS, max_length=32)
        assert self._golden(preds) == core.SECTION_GOLDEN
        assert session.batches[0] == len(core.CONTEXTS)
        assert model._cpu_session is None

    def test_single_heading_runs_on_the_cpu(self):
        from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

        model = OnnxSectionClassifierModel(core._bundle("section"), "cpu")
        _on_failing_gpu(model, ARENA_OOM)
        preds = model.classify_batch(core.CONTEXTS, max_length=32)
        assert self._golden(preds) == core.SECTION_GOLDEN
        assert model._cpu_session is not None

    def test_real_fault_still_raises_for_the_callers_fallback(self):
        from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

        model = OnnxSectionClassifierModel(core._bundle("section"), "cpu")
        _on_failing_gpu(model, ILLEGAL_ADDRESS)
        with pytest.raises(RuntimeError, match="illegal memory access"):
            model.classify_batch(core.CONTEXTS, max_length=32)

    def test_a_cpu_session_that_fails_is_not_retried(self, caplog):
        from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

        model = OnnxSectionClassifierModel(core._bundle("section"), "cpu")
        session = _FailingGpuSession(model.session, ARENA_OOM)  # device stays "cpu"
        model.session = session
        with (
            caplog.at_level(logging.WARNING, logger="bibr.utils.gpu_oom"),
            pytest.raises(RuntimeError, match="BFCArena"),
        ):
            model.classify_batch(core.CONTEXTS, max_length=32)
        # Raised as before: no split, and not logged as a GPU out-of-memory.
        assert session.batches == [len(core.CONTEXTS)]
        assert not [r for r in caplog.records if "GPU out of memory" in r.getMessage()]
        assert model._cpu_session is None

    def test_cpu_fallback_session_skips_the_cpu_arena(self, monkeypatch):
        from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

        model = OnnxSectionClassifierModel(core._bundle("section"), "cpu")
        calls = _spy_create_session(monkeypatch)
        model._session_on_cpu()
        assert calls == [{"device": "cpu", "disable_cpu_arena": True}]


class TestNerParser:
    """A parser OOM dropped the paper's references (11 papers in E23's long run)."""

    def test_splits_then_matches_golden(self):
        from bibr.ner.parser_onnx import OnnxRefParser

        parser = OnnxRefParser(core._bundle("ner"), device="cpu")
        session = _on_failing_gpu(parser, ARENA_OOM, fits=1)
        assert parser.parse_batch(core.REFS) == core.NER_GOLDEN
        assert session.batches == [2, 1, 1]

    def test_single_reference_runs_on_the_cpu(self):
        from bibr.ner.parser_onnx import OnnxRefParser

        parser = OnnxRefParser(core._bundle("ner"), device="cpu")
        _on_failing_gpu(parser, ARENA_OOM)
        assert parser.parse_batch(core.REFS) == core.NER_GOLDEN
        assert parser.parse_batch(["", core.REFS[1], "  "]) == [{}, core.NER_GOLDEN[1], {}]

    def test_truncation_count_survives_the_split(self):
        from bibr.ner.parser_onnx import OnnxRefParser

        parser = OnnxRefParser(core._bundle("ner"), device="cpu")
        expected = (parser.parse_batch(core.REFS), parser.last_truncated_count)
        _on_failing_gpu(parser, ARENA_OOM, fits=1)
        assert (parser.parse_batch(core.REFS), parser.last_truncated_count) == expected
        assert expected[1] > 0

    def test_real_fault_still_raises(self):
        from bibr.ner.parser_onnx import OnnxRefParser

        parser = OnnxRefParser(core._bundle("ner"), device="cpu")
        _on_failing_gpu(parser, ILLEGAL_ADDRESS)
        with pytest.raises(RuntimeError, match="illegal memory access"):
            parser.parse_batch(core.REFS)

    def test_a_cpu_session_that_fails_is_not_retried(self, caplog):
        from bibr.ner.parser_onnx import OnnxRefParser

        parser = OnnxRefParser(core._bundle("ner"), device="cpu")
        session = _FailingGpuSession(parser.session, ARENA_OOM)  # device stays "cpu"
        parser.session = session
        with (
            caplog.at_level(logging.WARNING, logger="bibr.utils.gpu_oom"),
            pytest.raises(RuntimeError, match="BFCArena"),
        ):
            parser.parse_batch(core.REFS)
        assert session.batches == [len(core.REFS)]
        assert not [r for r in caplog.records if "GPU out of memory" in r.getMessage()]
        assert parser._cpu_session is None

    def test_cpu_fallback_session_skips_the_cpu_arena(self, monkeypatch):
        from bibr.ner.parser_onnx import OnnxRefParser

        parser = OnnxRefParser(core._bundle("ner"), device="cpu")
        calls = _spy_create_session(monkeypatch)
        parser._session_on_cpu()
        assert calls == [{"device": "cpu", "disable_cpu_arena": True}]


def _spy_create_session(monkeypatch) -> list[dict]:
    """Record the device and CPU-arena choice of each session opened from now on."""
    from bibr.utils import onnx_providers

    real = onnx_providers.create_session
    calls: list[dict] = []

    def spy(model_path, **kwargs):
        calls.append(
            {"device": kwargs.get("device"), "disable_cpu_arena": kwargs.get("disable_cpu_arena")}
        )
        return real(model_path, **kwargs)

    monkeypatch.setattr(onnx_providers, "create_session", spy)
    return calls


def _layout_detector(message: str, *, fits: int = 0, device: str = "cuda"):
    from bibr.layout_base import BaseLayoutDetector
    from bibr.layout_onnx import OnnxLayoutBackend

    backend = OnnxLayoutBackend(core._bundle("layout"), threshold=0.05)
    backend.device = device
    backend.session = _FailingGpuSession(
        backend.session, message, fits=fits, batch_input=backend._input_name
    )
    det = object.__new__(BaseLayoutDetector)
    det._compiled = False
    det._runtime = "onnx"
    det._model = backend
    det._postprocess = lambda result, width, height: result  # type: ignore[method-assign]
    return det, backend


class TestLayout:
    """The bare ``cudaError`` failed its paper with ``layout_failed``."""

    def test_bare_cuda_error_on_a_full_card_splits_and_runs_the_last_page_on_cpu(
        self, monkeypatch, caplog
    ):
        _nearly_full(monkeypatch, 0.2)
        det, backend = _layout_detector(BARE_CUDA_ERROR)
        with caplog.at_level(logging.WARNING, logger="bibr.layout_base"):
            out = det._detect_images(core._sample_pages())
        assert [core._rounded(d) for d in out] == core.LAYOUT_GOLDEN
        assert backend.session.batches == [2, 1, 1]
        messages = [r.getMessage() for r in caplog.records]
        assert "Layout OOM with batch size 2, retrying as 1 + 1" in messages
        assert messages.count("Layout OOM on a single page; running it on the CPU") == 2

    def test_arena_failure_splits(self):
        det, backend = _layout_detector(ARENA_OOM, fits=1)
        out = det._detect_images(core._sample_pages())
        assert [core._rounded(d) for d in out] == core.LAYOUT_GOLDEN
        assert backend._cpu_session is None

    def test_bare_cuda_error_with_room_still_fails_the_page(self, monkeypatch):
        _nearly_full(monkeypatch, 9.0)
        det, _backend = _layout_detector(BARE_CUDA_ERROR)
        with pytest.raises(RuntimeError, match="ERRTYPE = cudaError"):
            det._detect_images(core._sample_pages())

    def test_real_fault_still_fails_the_page(self, monkeypatch):
        _nearly_full(monkeypatch, 0.2)
        det, _backend = _layout_detector(ILLEGAL_ADDRESS)
        with pytest.raises(RuntimeError, match="illegal memory access"):
            det._detect_images(core._sample_pages())

    def test_a_cpu_session_splits_as_before_but_never_reruns_a_page_on_itself(self, caplog):
        det, backend = _layout_detector(ARENA_OOM, device="cpu")
        with (
            caplog.at_level(logging.WARNING, logger="bibr.layout_base"),
            pytest.raises(RuntimeError, match="BFCArena"),
        ):
            det._detect_images(core._sample_pages())
        # The batch halves as it always did; the single page then raises
        # instead of running again on the same CPU session.
        assert backend.session.batches == [2, 1]
        messages = [r.getMessage() for r in caplog.records]
        assert "Layout OOM with batch size 2, retrying as 1 + 1" in messages
        assert "Layout OOM on a single page; running it on the CPU" not in messages
        assert gpu_oom._last_oom_at is None

    def test_torch_runtime_has_no_cpu_fallback(self):
        from bibr.layout_base import BaseLayoutDetector

        det = object.__new__(BaseLayoutDetector)
        det._compiled = False

        def detect(images, orig_sizes):
            raise RuntimeError("CUDA out of memory")

        det._detect_pytorch = detect  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="out of memory"):
            det._detect_images([Image.new("RGB", (10, 10))])


class _FakeSat:
    def __init__(self, *, fail_with: str | None = None, fits: int = 0):
        self.fail_with = fail_with
        self.fits = fits
        self.batches: list[int] = []

    def split(self, texts, **kwargs):
        self.batches.append(len(texts))
        if self.fail_with and len(texts) > self.fits:
            raise RuntimeError(self.fail_with)
        return [[t] for t in texts]


def _segmenter(gpu: _FakeSat, *, device: str = "cuda", cpu: _FakeSat | None = None):
    from bibr.segmenter_base import BaseSentenceSegmenter

    seg = object.__new__(BaseSentenceSegmenter)
    seg.model = gpu
    seg._threshold = None
    seg._device = device
    seg._cpu_model = cpu
    seg._cpu_model_lock = threading.Lock()
    return seg


class TestSentenceSegmenter:
    """One 3,361-character text that did not fit failed its paper (parse_failed)."""

    def test_single_text_goes_to_the_cpu_copy(self, caplog):
        cpu = _FakeSat()
        seg = _segmenter(_FakeSat(fail_with=ARENA_OOM), cpu=cpu)
        with caplog.at_level(logging.WARNING, logger="bibr.segmenter_base"):
            out = seg._split_batch_with_retry(["a", "b", "c"])
        assert out == [["a"], ["b"], ["c"]]
        assert cpu.batches == [1, 1, 1]
        messages = [r.getMessage() for r in caplog.records]
        assert "OOM with batch size 3, retrying as 1 + 2" in messages
        assert "OOM on single text (1 chars); splitting it on the CPU" in messages

    def test_halving_alone_when_it_fits(self):
        gpu = _FakeSat(fail_with=ARENA_OOM, fits=2)
        seg = _segmenter(gpu, cpu=_FakeSat())
        assert seg._split_batch_with_retry(["a", "b", "c", "d"]) == [["a"], ["b"], ["c"], ["d"]]
        assert gpu.batches == [4, 2, 2]

    def test_cpu_session_single_text_still_raises(self):
        seg = _segmenter(_FakeSat(fail_with=ARENA_OOM), device="cpu")
        with pytest.raises(RuntimeError, match="BFCArena"):
            seg._split_batch_with_retry(["a"])

    def test_cpu_session_halves_without_counting_a_gpu_oom(self):
        gpu = _FakeSat(fail_with=ARENA_OOM, fits=1)
        seg = _segmenter(gpu, device="cpu")
        assert seg._split_batch_with_retry(["a", "b"]) == [["a"], ["b"]]
        assert gpu.batches == [2, 1, 1]
        assert gpu_oom._last_oom_at is None

    def test_real_fault_is_not_retried(self):
        gpu = _FakeSat(fail_with=ILLEGAL_ADDRESS)
        seg = _segmenter(gpu, cpu=_FakeSat())
        with pytest.raises(RuntimeError, match="illegal memory access"):
            seg._split_batch_with_retry(["a", "b"])
        assert gpu.batches == [2]


# The sentence segmenter's CUDA session could not even be created on an 8 GB
# card with a desktop beside the managed vLLM server (E26 E8-w4xd, 15:32:12):
# ORT's constructor raised its arena failure, and the chunk failed parse_failed.
SESSION_ARENA_OOM = (
    "[ONNXRuntimeError] : 1 : FAIL : Exception during initialization: "
    "/onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* "
    "onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) "
    "Failed to allocate memory for requested buffer of size 384003072"
)


def _cuda_sessions_fail(monkeypatch, message: str) -> list[dict]:
    """From now on every session asks for CUDA first, and opening a CUDA one raises *message*.

    A chain without CUDA opens the real graph on the CPU. Returns one record
    (provider names, whether the CPU arena was left on) per session opened.
    """
    import onnxruntime as ort

    from bibr.utils import onnx_providers

    real_providers = onnx_providers.get_ort_providers
    real_session = ort.InferenceSession
    opened: list[dict] = []

    def providers(*, enable_cuda: bool = True, **kwargs):
        chain = real_providers(enable_cuda=False, **kwargs)
        return [("CUDAExecutionProvider", {}), *chain] if enable_cuda else chain

    def session(path, sess_options=None, providers=None, **kwargs):
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        arena = None if sess_options is None else sess_options.enable_cpu_mem_arena
        opened.append({"providers": names, "cpu_arena": arena})
        if "CUDAExecutionProvider" in names:
            raise RuntimeError(message)
        return real_session(path, sess_options=sess_options, providers=providers, **kwargs)

    monkeypatch.setattr(onnx_providers, "get_ort_providers", providers)
    monkeypatch.setattr(ort, "InferenceSession", session)
    return opened


_CPU_FALLBACK_LINE = (
    "could not open its CUDA session: GPU out of memory (RuntimeError); opening it on the CPU"
)


class TestSessionCreation:
    """A CUDA session that cannot be opened for lack of memory opens on the CPU instead."""

    def test_section_classifier_opens_on_the_cpu_and_matches_golden(self, monkeypatch, caplog):
        from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

        opened = _cuda_sessions_fail(monkeypatch, SESSION_ARENA_OOM)
        with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
            model = OnnxSectionClassifierModel(core._bundle("section"), "cuda")
        assert model.device == "cpu"
        assert [o["providers"][0] for o in opened] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert f"section classifier (ONNX) {_CPU_FALLBACK_LINE}" in caplog.text
        assert "requested CUDA" not in caplog.text
        preds = model.classify_batch(core.CONTEXTS, max_length=32)
        assert [
            (p.canonical_type.value, p.is_top_level, round(p.score, 4)) for p in preds
        ] == core.SECTION_GOLDEN

    def test_ner_parser_opens_on_the_cpu_and_matches_golden(self, monkeypatch, caplog):
        from bibr.ner.parser_onnx import OnnxRefParser

        opened = _cuda_sessions_fail(monkeypatch, SESSION_ARENA_OOM)
        with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
            parser = OnnxRefParser(core._bundle("ner"), device="cuda")
        assert parser.device == "cpu"
        assert len(opened) == 2
        assert f"NER reference parser (ONNX) {_CPU_FALLBACK_LINE}" in caplog.text
        assert parser.parse_batch(core.REFS) == core.NER_GOLDEN

    def test_paper_classifier_opens_on_the_cpu(self, monkeypatch):
        from bibr.structure.paper_classifier_onnx import OnnxPaperClassifierModel

        _cuda_sessions_fail(monkeypatch, SESSION_ARENA_OOM)
        model = OnnxPaperClassifierModel(core._bundle("paper"), "cuda")
        assert model.device == "cpu"
        assert [p.paper_type for p in model.classify_batch(core.ITEMS)] == [
            row[4] for row in core.PAPER_GOLDEN
        ]

    def test_layout_opens_on_the_cpu_without_the_cpu_arena(self, monkeypatch, caplog):
        from bibr.layout_onnx import OnnxLayoutBackend

        opened = _cuda_sessions_fail(monkeypatch, SESSION_ARENA_OOM)
        with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
            backend = OnnxLayoutBackend(core._bundle("layout"), device="cuda", threshold=0.05)
        assert backend.device == "cpu"
        assert opened[1] == {"providers": ["CPUExecutionProvider"], "cpu_arena": False}
        assert _CPU_FALLBACK_LINE in caplog.text
        assert [
            core._rounded(det) for det in backend.run(core._sample_pages())
        ] == core.LAYOUT_GOLDEN

    def test_it_is_recorded_as_the_run_time_fallback_is(self, monkeypatch, caplog):
        """Neither CPU fallback has a counter: each logs a warning and opens the
        process-wide recent-out-of-memory window (``gpu_oom._note_oom``)."""
        from bibr.ner.parser_onnx import OnnxRefParser
        from bibr.utils import gpu_oom

        _cuda_sessions_fail(monkeypatch, SESSION_ARENA_OOM)
        assert not gpu_oom._recent_oom()
        with caplog.at_level(logging.WARNING):
            OnnxRefParser(core._bundle("ner"), device="cuda")
        assert gpu_oom._recent_oom()
        session_warnings = [r for r in caplog.records if r.levelno == logging.WARNING]

        monkeypatch.setattr(gpu_oom, "_last_oom_at", None)
        caplog.clear()

        def run(_items):
            raise RuntimeError(SESSION_ARENA_OOM)

        with caplog.at_level(logging.WARNING):
            assert gpu_oom.split_on_gpu_oom(
                ["x"], run, label="t", run_on_cpu=lambda _items: ["cpu"]
            ) == ["cpu"]
        assert gpu_oom._recent_oom()
        run_warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(session_warnings) == len(run_warnings) == 1

    def test_a_real_fault_while_opening_still_raises(self, monkeypatch):
        from bibr.ner.parser_onnx import OnnxRefParser

        opened = _cuda_sessions_fail(monkeypatch, ILLEGAL_ADDRESS)
        with pytest.raises(RuntimeError, match="illegal memory access"):
            OnnxRefParser(core._bundle("ner"), device="cuda")
        assert len(opened) == 1
