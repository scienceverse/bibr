"""`LlmClient` Protocol — structural contract for the LLM client public surface.

The real implementation is :class:`bibr.clients.llm.LLMClient`. Tests can
substitute a fake that implements the same methods. Private helpers
(``_invoke_structured``, ``_cap_input``) are intentionally excluded — those
are internal LLMClient mechanics and not stable extension points.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LlmClient(Protocol):
    """Public surface required by extract/structure callers."""

    async def extract_core_metadata(
        self,
        text: str,
        file_hash: str = "unknown",
        *,
        authors_text: str | None = None,
        classification_text: str | None = None,
        include_classification: bool = True,
    ) -> Any: ...
    async def extract_references(
        self, text: str, file_hash: str = "unknown", start_index: int = 1
    ) -> Any: ...
    async def extract_references_chunk(self, text: str, file_hash: str = "unknown") -> Any: ...
    async def segment_references(self, text: str, file_hash: str = "unknown") -> list[str]: ...
    async def extract_equations(
        self, sentences: list[tuple[int, str]], file_hash: str = "unknown"
    ) -> list: ...
    async def label_paper_type(
        self, title: str, abstract: str, file_hash: str = "unknown"
    ) -> Any: ...
    async def resolve_citations(
        self,
        ambiguous_citations: list[tuple[int, str]],
        reference_summary: list[dict],
        file_hash: str = "unknown",
    ) -> list: ...
    async def invoke_structured(
        self,
        response_model: Any,
        messages: list[dict],
        system_prompt: str,
        *,
        label: str,
        reasoning_effort: str | None = None,
        client_override: Any = None,
        max_tokens: int | None = None,
    ) -> Any: ...

    @property
    def limiter(self) -> Any: ...

    async def close(self) -> None: ...
