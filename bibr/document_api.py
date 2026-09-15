"""Opt-in document API: all detected papers from one input file."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack

from bibr.api import (
    ChewOptions,
    Result,
    _pipeline_kwargs,
    _preflight_llm,
    _refs_kwargs,
    _require_file_path,
)
from bibr.export.document_models import DocumentExport, DocumentRecordExport

if TYPE_CHECKING:
    from bibr.config import GlobalSettings


class DocumentRecord:
    """An article candidate, including explicit unresolved/failed outcomes."""

    def __init__(self, model: DocumentRecordExport):
        self.model = model

    @property
    def record_id(self) -> str:
        return self.model.record_id

    @property
    def status(self) -> str:
        return self.model.status

    @property
    def paper(self) -> Result | None:
        return Result(self.model.paper) if self.model.paper is not None else None

    @property
    def partial_paper(self) -> Result | None:
        """Retained scoped fields with blocking validation; never a success."""
        return Result(self.model.partial_paper) if self.model.partial_paper is not None else None

    @property
    def reason_flags(self) -> list[str]:
        return list(self.model.reason_flags)

    @property
    def error(self) -> str | None:
        return self.model.error


class DocumentResult:
    """Validated document envelope; ``records`` retains every detected candidate.

    ``ok`` means every detected record was extracted. It does not certify
    exhaustive article detection or metadata quality. Inspect nested validation
    and ``reason_flags`` as well as unresolved/failed outcomes.
    """

    def __init__(self, data: dict[str, Any] | DocumentExport):
        self.model = (
            data if isinstance(data, DocumentExport) else DocumentExport.model_validate(data)
        )

    @property
    def data(self) -> dict[str, Any]:
        return self.model.model_dump(mode="json", by_alias=True)

    @property
    def document_id(self) -> str:
        return self.model.document_id

    @property
    def status(self) -> str:
        return self.model.status

    @property
    def ok(self) -> bool:
        return self.status == "complete"

    @property
    def records(self) -> list[DocumentRecord]:
        return [DocumentRecord(record) for record in self.model.records]

    @property
    def papers(self) -> list[Result]:
        """Extracted papers only; inspect ``records`` to account for failures."""
        return [Result(record.paper) for record in self.model.records if record.paper is not None]

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return destination

    def __repr__(self) -> str:
        return f"<bibr.DocumentResult {self.status}: {len(self.model.records)} records>"


async def achew_document(
    path: str | Path,
    *,
    refs: str | bool | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> DocumentResult:
    """Return every detected paper from exactly one input file asynchronously.

    Uses the same OCR/LLM/reference options as ``achew``. Printed translations
    remain variants of one paper when identity evidence supports that grouping.
    Document-wide input/OCR failures raise; individual record failures are rows.
    """
    from bibr.local.pipeline import LocalPipeline

    _require_file_path(path, api_name="achew_document")
    kwargs = {**_pipeline_kwargs(options), **_refs_kwargs(refs)}
    _preflight_llm(settings, kwargs)
    pipeline = LocalPipeline(settings=settings, **kwargs)
    try:
        return DocumentResult(await pipeline.process_document(path))
    finally:
        await pipeline.aclose()


def chew_document(
    path: str | Path,
    *,
    refs: str | bool | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> DocumentResult:
    """Synchronous ``achew_document``; use the async form in a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(achew_document(path, refs=refs, settings=settings, **options))
    raise RuntimeError("Use 'await bibr.achew_document(...)' inside a running event loop")


__all__ = ["DocumentRecord", "DocumentResult", "achew_document", "chew_document"]
