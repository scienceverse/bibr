"""`OcrBackend` Protocol — explicit contract for OCR clients.

Every concrete OCR client (local SGLang engine, HTTP fallback, managed
vllm-mlx subprocess, plus the serve-layer backend added in P4) must
satisfy this Protocol. Dispatch is done through ``bibr.ocr.registry``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable


class OcrText(str):
    """String-compatible OCR output carrying optional provider stop metadata."""

    finish_reason: str | None

    def __new__(cls, content: str, *, finish_reason: str | None = None):
        value = super().__new__(cls, content)
        value.finish_reason = finish_reason
        return value


@runtime_checkable
class OcrBackend(Protocol):
    """OCR backend contract.

    Attributes:
        name: Registry key. Unique per concrete backend.

    Properties:
        loaded: ``True`` if the backend is ready to serve requests.
    """

    name: ClassVar[str]

    @property
    def loaded(self) -> bool: ...

    async def recognize(self, image: Any, prompt: str) -> str:
        """OCR a single cropped region image and return recognised text."""
        ...

    async def wait_for_server(self) -> None:
        """Block until the backend is ready. No-op for in-process backends."""
        ...

    async def shutdown(self) -> None:
        """Release all resources held by the backend."""
        ...
