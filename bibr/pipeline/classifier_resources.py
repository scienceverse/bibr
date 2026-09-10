"""Managed lifecycle and cross-request batching for local classifiers."""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from bibr.pipeline.inference_batcher import InferenceBatcher
from bibr.utils.locks import LOCAL_INFERENCE_LOCK

logger = logging.getLogger(__name__)
_MIB = 1024 * 1024


class ClassifierState(StrEnum):
    UNCONFIGURED = "unconfigured"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED_REQUIRED = "failed_required"
    CLOSED = "closed"


@dataclass(frozen=True)
class ClassifierStatus:
    state: ClassifierState
    device: str | None = None
    error: str | None = None
    peak_vram_bytes: int | None = None


def choose_classifier_device(
    *,
    explicit_device: str | None,
    memory_mode: str,
    cuda_available: bool,
    free_vram_bytes: int | None,
    total_vram_bytes: int | None,
    managed_vllm_fraction: float,
    estimated_peak_bytes: int,
    safety_reserve_bytes: int,
) -> str:
    """Choose CUDA only when classifier and reserve fit after vLLM allocation."""
    if explicit_device:
        return explicit_device
    if memory_mode == "aggressive" or not cuda_available:
        return "cpu"
    if free_vram_bytes is None or total_vram_bytes is None:
        return "cpu"
    post_vllm_capacity = int(total_vram_bytes * max(0.0, 1.0 - managed_vllm_fraction))
    usable = min(free_vram_bytes, post_vllm_capacity) - safety_reserve_bytes
    return "cuda" if usable > estimated_peak_bytes else "cpu"


@dataclass
class _ManagedClassifier:
    name: str
    model_id: str | None
    revision: str
    explicit_device: str | None
    estimated_peak_bytes: int
    batch_size: int
    batch_timeout_ms: float
    loader: Any
    status: ClassifierStatus = ClassifierStatus(ClassifierState.UNCONFIGURED)
    model: Any = None
    batcher: InferenceBatcher[Any, Any] | None = None


class ClassifierResources:
    """Own both classifier models, sticky startup state, and micro-batchers."""

    def __init__(
        self,
        settings,
        *,
        memory_mode: str,
        managed_vllm_fraction: float,
        cuda_available: bool | None = None,
        free_vram_bytes: int | None = None,
        total_vram_bytes: int | None = None,
        paper_loader=None,
        section_loader=None,
    ) -> None:
        self._settings = settings
        self._memory_mode = memory_mode
        self._managed_vllm_fraction = managed_vllm_fraction
        self._cuda_available = cuda_available
        self._free_vram_bytes = free_vram_bytes
        self._total_vram_bytes = total_vram_bytes
        self._required = bool(settings.ml.classifiers_required)
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._paper = _ManagedClassifier(
            name="paper",
            model_id=settings.ml.paper_classifier_model_id,
            revision=settings.ml.paper_classifier_revision,
            explicit_device=settings.ml.paper_classifier_device,
            estimated_peak_bytes=settings.ml.paper_classifier_estimated_peak_mb * _MIB,
            batch_size=settings.ml.paper_classifier_batch_size,
            batch_timeout_ms=settings.ml.paper_classifier_batch_timeout_ms,
            loader=paper_loader or functools.partial(_load_paper_model, settings=settings),
        )
        self._section = _ManagedClassifier(
            name="section",
            model_id=settings.ml.section_classifier_model_id,
            revision=settings.ml.section_classifier_revision,
            explicit_device=settings.ml.section_classifier_device,
            estimated_peak_bytes=settings.ml.section_classifier_estimated_peak_mb * _MIB,
            batch_size=settings.ml.section_classifier_batch_size,
            batch_timeout_ms=settings.ml.section_classifier_batch_timeout_ms,
            loader=section_loader or functools.partial(_load_section_model, settings=settings),
        )

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("classifier resources are closed")
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self._cuda_available is None:
                (
                    self._cuda_available,
                    self._free_vram_bytes,
                    self._total_vram_bytes,
                ) = await asyncio.to_thread(_cuda_memory_info)
            await self._start_one(self._paper)
            await self._start_one(self._section)
            self._started = True

    async def classify_paper(self, item):
        await self.start()
        if self._paper.batcher is None:
            return None
        return await self._paper.batcher.submit(item)

    async def classify_section(self, item):
        await self.start()
        if self._section.batcher is None:
            return None
        return await self._section.batcher.submit(item)

    async def classify_papers(self, items: list[Any]) -> list[Any] | None:
        results = await asyncio.gather(*(self.classify_paper(item) for item in items))
        return None if all(result is None for result in results) else list(results)

    async def classify_sections(self, items: list[Any]) -> list[Any] | None:
        results = await asyncio.gather(*(self.classify_section(item) for item in items))
        return None if all(result is None for result in results) else list(results)

    def status(self) -> dict[str, ClassifierStatus]:
        return {"paper": self._paper.status, "section": self._section.status}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in (self._paper, self._section):
            if resource.batcher is not None:
                await resource.batcher.close()
                resource.batcher = None
            resource.model = None
            resource.status = ClassifierStatus(ClassifierState.CLOSED, resource.status.device)

    async def _start_one(self, resource: _ManagedClassifier) -> None:
        if not resource.model_id:
            resource.status = ClassifierStatus(ClassifierState.UNCONFIGURED)
            return
        device = choose_classifier_device(
            explicit_device=resource.explicit_device,
            memory_mode=self._memory_mode,
            cuda_available=bool(self._cuda_available),
            free_vram_bytes=self._free_vram_bytes,
            total_vram_bytes=self._total_vram_bytes,
            managed_vllm_fraction=self._managed_vllm_fraction,
            estimated_peak_bytes=resource.estimated_peak_bytes,
            safety_reserve_bytes=self._settings.ml.classifier_vram_safety_reserve_mb * _MIB,
        )
        resource.status = ClassifierStatus(ClassifierState.LOADING, device)
        try:
            resource.model = await asyncio.to_thread(
                _locked_load, resource.loader, resource.model_id, resource.revision, device
            )
        except Exception as exc:  # noqa: BLE001 - sticky degraded startup state
            state = ClassifierState.FAILED_REQUIRED if self._required else ClassifierState.DEGRADED
            resource.status = ClassifierStatus(state, device, _safe_error(exc))
            logger.warning("%s classifier load failed: %s", resource.name, exc)
            return

        peak_vram = _peak_vram_bytes(device)
        resource.status = ClassifierStatus(ClassifierState.READY, device, peak_vram_bytes=peak_vram)
        resource.batcher = InferenceBatcher(
            lambda items, model=resource.model: _locked_forward(model, items),
            resource.batch_size,
            resource.batch_timeout_ms,
            f"{resource.name}-classifier",
        )


def _locked_load(loader, model_id: str, revision: str, device: str):
    with LOCAL_INFERENCE_LOCK:
        return loader(model_id, revision, device)


def _locked_forward(model, items):
    with LOCAL_INFERENCE_LOCK:
        return model.classify_batch(items)


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _cuda_memory_info() -> tuple[bool, int | None, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return (False, None, None)
        free, total = torch.cuda.mem_get_info()
        return (True, int(free), int(total))
    except (ImportError, RuntimeError):
        return (False, None, None)


def _peak_vram_bytes(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None


def _load_paper_model(model_id: str, revision: str, device: str, settings=None):
    """Load the paper classifier on the runtime ``ML_RUNTIME`` selects (ONNX or torch)."""
    from bibr.structure.paper_classifier_common import load_paper_classifier

    return load_paper_classifier(model_id, revision=revision, device=device, settings=settings)


def _load_section_model(model_id: str, revision: str, device: str, settings=None):
    """Load the section classifier on the runtime ``ML_RUNTIME`` selects (ONNX or torch)."""
    from bibr.structure.section_classifier_common import load_section_classifier

    return load_section_classifier(model_id, revision=revision, device=device, settings=settings)
