import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from bibr.clients.llm import LLMClient, usage_file_context
from bibr.config import GlobalSettings
from bibr.exceptions import ProcessingError
from bibr.schemas import CitationMatch, CitationResolutionResult
from tests.structure.test_citation_shortlist import bibliography


def answer(text="Smith (2020)", bib_id=71, text_id=10):
    return CitationMatch(text_id=text_id, citation_text=text, bib_id=bib_id)


class Backend:
    def __init__(self, effects):
        self.effects = iter(effects)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        effect = next(self.effects)
        if isinstance(effect, BaseException):
            raise effect
        return CitationResolutionResult(matches=effect), SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        )


def client(effects, enabled=True):
    backend = Backend(effects)
    instance = LLMClient(
        settings=GlobalSettings(
            llm={
                "citation_shortlist": enabled,
                "track_usage": True,
            }
        ),
        backend=backend,
    )
    instance._limiter = mock.MagicMock()
    instance._limiter.acquire = mock.AsyncMock(return_value=None)
    return instance, backend


def prompt(request):
    return str(request["messages"])


async def test_first_pass_saves_refs_preserves_ids_and_records_usage():
    c, backend = client([[answer()]])
    with usage_file_context("paper"):
        matches = await c.resolve_citations([(10, "Smith (2020)")], bibliography())
    assert matches == [answer()]
    assert len(backend.requests) == 1
    assert "bib_id=71:" in prompt(backend.requests[0])
    assert "bib_id=100:" not in prompt(backend.requests[0])
    metrics = c.usage_labels_pop_file("paper")["resolve_citations"]
    assert metrics["logical_calls"] == metrics["calls"] == metrics["citation_shortlist_calls"] == 1
    assert metrics["citation_reference_rows_sent"] == 6
    assert metrics["citation_reference_rows_full"] == 46


async def test_expansion_only_retries_pending_and_cannot_overwrite_accepted():
    c, backend = client(
        [
            [answer(), answer("Jones (2020)", None, 20)],
            [answer(bib_id=42), answer("Jones (2020)", 15, 20)],
        ]
    )
    with usage_file_context("paper"):
        result = await c.resolve_citations(
            [(10, "Smith (2020)"), (20, "Jones (2020)")], bibliography()
        )
    assert result == [answer(), answer("Jones (2020)", 15, 20)]
    assert len(backend.requests) == 2
    assert "bib_id=100:" in prompt(backend.requests[1])
    assert "text_id=10: Smith" not in prompt(backend.requests[1])
    assert "text_id=20: Jones" in prompt(backend.requests[1])
    assert c._limiter.acquire.await_count == 2
    metrics = c.usage_labels_pop_file("paper")["resolve_citations"]
    assert metrics["logical_calls"] == 1
    assert metrics["calls"] == metrics["attempts"] == 2
    assert metrics["input_tokens"] == 200
    assert metrics["citation_full_expansions"] == 1
    assert metrics["citation_reference_rows_sent"] == 52


async def test_expansion_does_not_recurse_and_drops_unknown_ids():
    c, backend = client([[answer(bib_id=None)], [answer(bib_id=9999)]])
    assert await c.resolve_citations([(10, "Smith (2020)")], bibliography()) == []
    assert len(backend.requests) == 2


async def test_unavailable_expansion_preserves_accepted_matches(monkeypatch):
    c, _ = client([])
    monkeypatch.setattr(
        c,
        "_invoke_structured",
        mock.AsyncMock(
            side_effect=[CitationResolutionResult(matches=[answer()]), RuntimeError("unavailable")]
        ),
    )
    assert await c.resolve_citations(
        [(10, "Smith (2020)"), (20, "Jones (2020)")], bibliography()
    ) == [answer()]


@pytest.mark.parametrize("error", [ProcessingError("invalid output"), asyncio.CancelledError()])
async def test_expansion_preserves_processing_errors_and_cancellation(monkeypatch, error):
    c, _ = client([])
    monkeypatch.setattr(
        c,
        "_invoke_structured",
        mock.AsyncMock(side_effect=[CitationResolutionResult(matches=[answer()]), error]),
    )
    with pytest.raises(type(error)):
        await c.resolve_citations([(10, "Smith (2020)"), (20, "Jones (2020)")], bibliography())


@pytest.mark.parametrize("text,enabled", [("Smith (2020)", False), ("[1]", True)])
async def test_disabled_or_uncertain_retrieval_preserves_full_call(text, enabled):
    expected = answer(text, None)
    c, backend = client([[expected]], enabled=enabled)
    assert await c.resolve_citations([(10, text)], bibliography()) == [expected]
    assert len(backend.requests) == 1
    assert "bib_id=100:" in prompt(backend.requests[0])


def test_shortlist_setting_can_be_disabled_in_environment(monkeypatch):
    monkeypatch.setenv("LLM_CITATION_SHORTLIST", "false")
    assert not GlobalSettings().llm.citation_shortlist
