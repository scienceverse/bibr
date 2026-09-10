"""Base ``Pipeline`` class shared by local and serve pipelines.

Subclasses populate ``_stages``, ``_resources`` and ``_config`` in their
own ``__init__``; this base class provides the shared orchestration
driver (``process_chunk`` and its single-file convenience wrapper
``process_file``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from bibr.pipeline.progress import NullProgress

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import ProgressTracker
    from bibr.pipeline.resources import ResourceManager
    from bibr.pipeline.stage import Stage
    from bibr.pipeline.state import FileState

logger = logging.getLogger(__name__)


class StageContractError(RuntimeError):
    """A stage list is mis-ordered: a stage requires a FileState field that no
    earlier stage produces."""


# FileState fields populated before any stage runs (constructor inputs).
_INPUT_FIELDS = frozenset({"path", "paper_id", "pdf_bytes"})


def validate_stage_contracts(stages: Sequence[Stage]) -> None:
    """Statically check a stage list's ``requires``/``produces`` declarations.

    Each stage class may declare ``requires`` (FileState fields it consumes)
    and ``produces`` (fields it may populate). Every required field must be an
    input field or produced by an EARLIER stage — otherwise a reordering
    mistake surfaces as a silent None-skip that quietly drops files. Stages
    without declarations (tests, third-party) are skipped.

    Raises :class:`StageContractError` at pipeline construction time.
    """
    available = set(_INPUT_FIELDS)
    for stage in stages:
        requires = getattr(stage, "requires", None)
        produces = getattr(stage, "produces", None)
        if requires is not None:
            missing = [f for f in requires if f not in available]
            if missing:
                raise StageContractError(
                    f"{type(stage).__name__} requires {missing} but no earlier "
                    f"stage produces them (available: {sorted(available)})"
                )
        if produces is not None:
            available.update(produces)


async def run_stage(ctx: PipelineContext, stage: Stage) -> None:
    """Run one stage with the timing and cleanup guarantees shared by all pipelines."""
    started = time.monotonic()
    try:
        await stage.run(ctx)
    finally:
        ctx.scratch.setdefault("stage_timings", {})[stage.name] = time.monotonic() - started
        ctx.free_after_stage(stage.name)


class Pipeline:
    """Shared orchestration for stage-list pipelines."""

    _stages: tuple[Stage, ...]
    _resources: ResourceManager
    _config: RunConfig
    _settings: GlobalSettings

    def __init__(
        self,
        *,
        stages: Sequence[Stage],
        resources: ResourceManager,
        config: RunConfig,
        settings: GlobalSettings,
    ) -> None:
        self._stages = tuple(stages)
        self._resources = resources
        self._config = config
        self._settings = settings
        validate_stage_contracts(self._stages)

    @property
    def settings(self) -> GlobalSettings:
        """Concrete immutable-by-convention settings snapshot owned by this pipeline."""
        return self._settings

    async def process_file(
        self,
        path: str | Path,
        paper_id: str | None = None,
        progress: ProgressTracker | None = None,
        content: bytes | None = None,
        content_hash: str | None = None,
        config: RunConfig | None = None,
    ) -> dict:
        """Process a single file through the full pipeline.

        ``content`` lets callers pre-populate ``fs.pdf_bytes`` so the validate
        stage can skip a disk read (used by the serve API to avoid the
        in-memory → tmp-file → re-read roundtrip).
        ``config`` overrides ``self._config`` for this call only — used to set
        per-request fields like ``start_page`` / ``include_figures`` while
        sharing a single long-lived pipeline across requests.
        """
        from bibr.pipeline.state import FileState

        fs = FileState(path=Path(path), paper_id=paper_id)
        if content is not None:
            fs.pdf_bytes = content
        fs.content_sha256 = content_hash
        await self.process_chunk([fs], progress=progress, config=config)
        if fs.error:
            from bibr.exceptions import (
                InputValidationError,
                ProcessingError,
                UpstreamServiceError,
            )

            # Preserve upstream-service semantics (→ HTTP 502) instead of masking
            # an OCR/LLM/Crossref outage as a client-side ProcessingError (422).
            if isinstance(fs.original_error, UpstreamServiceError):
                raise fs.original_error
            # Likewise a rejected input is the caller's 4xx, not a 422 — and
            # ``InputValidationError`` is a sibling of ``ProcessingError``, not
            # a subclass, so without this it fell through to the generic wrap.
            if isinstance(fs.original_error, InputValidationError):
                raise fs.original_error
            if isinstance(fs.original_error, ProcessingError):
                # PostParseStage clears retained tracebacks for the safe
                # invalid-output failure. Re-raise every typed processing
                # error unchanged so its code and diagnostics survive.
                raise fs.original_error from fs.original_error.__cause__
            raise ProcessingError(
                fs.error,
                error_code=fs.error_code,
                failed_stage=fs.failed_stage,
            ) from fs.original_error
        assert fs.result_json is not None  # noqa: S101 — set by export stage on the success path
        return cast(dict[str, Any], fs.result_json)

    async def process_chunk(
        self,
        file_states: list[FileState],
        progress: ProgressTracker | None = None,
        config: RunConfig | None = None,
    ) -> None:
        """Run each stage in ``self._stages`` against a shared ``PipelineContext``.

        This intentionally does **not** tear down long-lived resources
        (LLM server, OCR engine). Owners must call :meth:`aclose` once
        when the pipeline itself is being disposed — tearing down on every
        request would force the next request to pay the vllm-mlx startup
        cost again.
        """
        if not file_states:
            return

        from bibr.pipeline.context import PipelineContext

        prog = progress or NullProgress()
        ctx = PipelineContext(
            file_states=file_states,
            progress=prog,
            resources=self._resources,
            config=config or self._config,
            settings=self._settings,
        )

        for stage in self._stages:
            await run_stage(ctx, stage)
            if not ctx.alive() and stage.name != "export":
                break

    async def aclose(self) -> None:
        """Release pipeline-lifetime resources (LLM server, OCR engine).

        Idempotent — safe to call more than once; underlying shutdowns no-op
        when their resource is already None. Best-effort: a failure tearing
        down one resource is logged and never skips the others (otherwise a
        raised LLM-server shutdown would leak the OCR engine and LLM client).
        """
        try:
            await self._resources.close_classifiers()
        except Exception:  # noqa: BLE001 — best-effort teardown
            logger.warning("Classifier shutdown failed during aclose", exc_info=True)
        try:
            self._resources.shutdown_llm_server()
        except Exception:  # noqa: BLE001 — best-effort teardown
            logger.warning("LLM server shutdown failed during aclose", exc_info=True)
        try:
            await self._resources.shutdown_ocr()
        except Exception:  # noqa: BLE001 — best-effort teardown
            logger.warning("OCR shutdown failed during aclose", exc_info=True)
        try:
            await self._resources.close_llm_client()
        except Exception:  # noqa: BLE001 — best-effort teardown
            logger.warning("LLM client close failed during aclose", exc_info=True)
