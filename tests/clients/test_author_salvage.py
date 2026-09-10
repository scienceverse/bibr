"""Author-JSON salvage + graceful degradation for truncated core-metadata.

A tight-context local server (e.g. NuExtract3 at max_model_len 8192) can cut the
author-list JSON mid-array, raising instructor's ``IncompleteOutputException``.
Two hardening behaviours:

1. ``extract_authors`` recovers the complete leading author objects from the
   truncated completion instead of losing the whole call.
2. ``extract_core_metadata`` keeps title/keywords when the author call fails
   outright, rather than aborting all core metadata.
"""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from bibr.clients.llm import LLMClient, _salvage_truncated_authors
from bibr.exceptions import UpstreamServiceError
from bibr.extract.ref_extractor import IncompleteOutputException
from bibr.schemas import AuthorsLLM, PaperClassificationLLM, TitleKeywordsLLM


def _openai_completion(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _degenerate_authors_completion():
    # A repetition loop: NuExtract3-FP8 emits the same handful of affiliation
    # fragments as author objects (role=["organization"], empty given) over and
    # over until the token cap cuts the array mid-object. salvage_array_objects
    # recovers every complete fragment (~132 here), which the degeneration guard
    # must reject rather than ship as a byline.
    orgs = [
        "Department of Psychology",
        "University of Testing",
        "Institute of Examples",
        "Center for Fixtures",
        "Laboratory of Mocks",
        "School of Assertions",
    ]
    objs = [
        json.dumps({"given": "", "family": o, "role": ["organization"]})
        for _ in range(22)
        for o in orgs
    ]
    raw = '{"authors": [' + ", ".join(objs) + ', {"given": "", "fam'
    return _openai_completion(raw)


def _long_distinct_byline_completion(n: int = 40):
    # A legitimate long consortium byline truncated mid-array: every recovered
    # object is a distinct author, so the guard must keep the salvage.
    objs = [json.dumps({"given": f"Given{i}", "family": f"Family{i}"}) for i in range(n)]
    raw = '{"authors": [' + ", ".join(objs) + ', {"given": "Zoe", "fam'
    return _openai_completion(raw)


def _truncated_authors_completion():
    # Two complete author objects, then a third cut off mid-object.
    raw = (
        '{"authors": ['
        '{"given": "Ada", "family": "Lovelace"}, '
        '{"given": "Alan", "family": "Turing"}, '
        '{"given": "Grace", "fam'
    )
    return _openai_completion(raw)


async def test_extract_authors_salvages_truncated_completion():
    client = LLMClient()
    lim = mock.Mock()
    lim.acquire = mock.AsyncMock()
    client._limiter = lim
    client._invoke_structured = mock.AsyncMock(
        side_effect=IncompleteOutputException(last_completion=_truncated_authors_completion())
    )

    result = await client.extract_authors("byline text", file_hash="h")

    assert isinstance(result, AuthorsLLM)
    assert [(a.given, a.family) for a in result.authors] == [
        ("Ada", "Lovelace"),
        ("Alan", "Turing"),
    ]


async def test_extract_authors_raises_when_nothing_salvageable():
    client = LLMClient()
    lim = mock.Mock()
    lim.acquire = mock.AsyncMock()
    client._limiter = lim
    # Truncated before the first object closes — nothing to salvage.
    client._invoke_structured = mock.AsyncMock(
        side_effect=IncompleteOutputException(
            last_completion=_openai_completion('{"authors": [{"given": "Ada", "fam')
        )
    )

    with pytest.raises(UpstreamServiceError):
        await client.extract_authors("byline text", file_hash="h")


async def test_extract_authors_raises_on_non_truncation_error():
    client = LLMClient()
    lim = mock.Mock()
    lim.acquire = mock.AsyncMock()
    client._limiter = lim
    client._invoke_structured = mock.AsyncMock(side_effect=RuntimeError("connection reset"))

    with pytest.raises(UpstreamServiceError):
        await client.extract_authors("byline text", file_hash="h")


async def test_author_failure_keeps_title_keywords():
    client = LLMClient()
    client.extract_title_keywords = mock.AsyncMock(
        return_value=TitleKeywordsLLM(title="A Real Title", abstract="A.", keywords=["k"])
    )
    # Persistent author failure: both the fan-out call and the re-roll raise.
    client.extract_authors = mock.AsyncMock(
        side_effect=UpstreamServiceError("LLM", "Failed to extract authors", RuntimeError("cut"))
    )
    client.extract_paper_classification = mock.AsyncMock(
        return_value=PaperClassificationLLM(paper_type="empirical", oecd_domain="Social Sciences")
    )

    result = await client.extract_core_metadata("byline: someone", file_hash="h")

    assert result.title == "A Real Title"
    assert result.keywords == ["k"]
    assert result.authors == []
    assert result.paper_type == "empirical"


def test_salvage_rejects_degenerate_repetition_loop():
    # A repetition-loop completion salvages many objects but is degenerate, so
    # the guard returns [] and lets the empty-author re-roll take over.
    exc = IncompleteOutputException(last_completion=_degenerate_authors_completion())
    assert _salvage_truncated_authors(exc) == []


def test_salvage_keeps_long_distinct_byline():
    # 40 distinct authors truncated mid-array: legitimate, must survive.
    exc = IncompleteOutputException(last_completion=_long_distinct_byline_completion(40))
    out = _salvage_truncated_authors(exc)
    assert len(out) == 40
    assert len({(a.given, a.family) for a in out}) == 40


async def test_extract_authors_reraises_on_degenerate_loop():
    client = LLMClient()
    lim = mock.Mock()
    lim.acquire = mock.AsyncMock()
    client._limiter = lim
    client._invoke_structured = mock.AsyncMock(
        side_effect=IncompleteOutputException(last_completion=_degenerate_authors_completion())
    )

    with pytest.raises(UpstreamServiceError):
        await client.extract_authors("byline text", file_hash="h")


class _RaisingBackend:
    """Minimal StructuredBackend stand-in whose create() always raises."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    async def create(self, **kwargs):
        raise self._exc


async def test_incomplete_output_records_partial_usage():
    # The truncated completion carries token usage; _invoke_structured must
    # record it before re-raising so llm_usage isn't understated.
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"authors": [', tool_calls=None))
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22, total_tokens=33),
    )
    exc = IncompleteOutputException(last_completion=completion)
    client = LLMClient(backend=_RaisingBackend(exc))
    client._track_usage = True

    with pytest.raises(IncompleteOutputException):
        await client._invoke_structured(AuthorsLLM, [{"role": "user", "content": "x"}], "sys")

    model = client._settings.llm.model
    assert client._usage[model]["input_tokens"] == 11
    assert client._usage[model]["output_tokens"] == 22
    assert client._usage[model]["total_tokens"] == 33
