"""The LLM client of a fast run: one that makes no call.

Fast mode hands this to every stage that would take an :class:`LLMClient`. It
is falsy, so the ``if llm_client:`` gates skip their block, and the sites that
need more than a gate test :func:`is_skipped` and take their no-LLM path
without a record of failure. A call that reaches it anyway raises
:class:`LlmSkipped`, which the sites' own error handling turns into a visible
``*_FAILED`` warning, so a site that missed its guard is caught by the tests
and the export check, and can never reach the network.
"""

from __future__ import annotations

from typing import Any


class LlmSkipped(RuntimeError):
    """A call reached the fast-mode client; the site that made it has no fast-mode guard."""


def _refused(name: str) -> LlmSkipped:
    return LlmSkipped(f"fast mode makes no LLM calls, but {name}() was called")


class SkippedLlmClient:
    """Stands in for :class:`bibr.clients.llm.LLMClient` in a fast run."""

    _track_usage = False

    def __bool__(self) -> bool:
        return False

    @property
    def usage(self) -> dict:
        return {}

    @property
    def limiter(self) -> Any:
        return None

    def json_mode_reroll_is_distinct(self) -> bool:
        return False

    async def ensure_server_context(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def record_reference_usage(self, usage: dict) -> None:
        return None

    def usage_for_file(self, file_hash: str) -> dict:
        return {}

    def usage_pop_file(self, key: str) -> dict:
        return {}

    def usage_labels_pop_file(self, key: str) -> dict:
        return {}

    def protocol_hashes_pop_file(self, key: str) -> dict:
        return {}

    def traces_pop_file(self, key: str) -> list:
        return []

    async def extract_core_metadata(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_core_metadata")

    async def extract_core_metadata_merged(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_core_metadata_merged")

    async def extract_title_keywords(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_title_keywords")

    async def extract_authors(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_authors")

    async def extract_paper_classification(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_paper_classification")

    async def label_paper_type(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("label_paper_type")

    async def extract_references(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_references")

    async def extract_references_chunk(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_references_chunk")

    async def segment_references(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("segment_references")

    async def extract_research_integrity(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_research_integrity")

    async def resolve_citations(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("resolve_citations")

    async def extract_equations(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("extract_equations")

    async def invoke_structured(self, *args: Any, **kwargs: Any) -> Any:
        raise _refused("invoke_structured")


def is_skipped(client: object) -> bool:
    """Whether *client* is the fast-mode stand-in (so the run makes no LLM call)."""
    return isinstance(client, SkippedLlmClient)


def note_skipped(contents: Any, what: str, count: int = 1) -> None:
    """Add *count* to what this paper's fast run left undone (``PaperContents.llm_skipped``)."""
    if count > 0:
        contents.llm_skipped[what] = contents.llm_skipped.get(what, 0) + count
