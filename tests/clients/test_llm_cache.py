"""Content-addressed cache for structured LLM responses."""

import json
from unittest import mock

import pytest

from bibr.clients.llm import LLMClient
from bibr.clients.llm_cache import (
    LlmResponseCache,
    canonical_user_text,
    request_key,
)
from bibr.clients.prompts import fence
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM

_BOUNDARY_A = "0123456789abcdef0123456789abcdef"
_BOUNDARY_B = "fedcba9876543210fedcba9876543210"


def _key(**overrides):
    base = {
        "model": "gemini-3.5-flash-lite",
        "schema_name": "TitleKeywordsLLM",
        "system": "sys",
        "user_text": "hello",
    }
    base.update(overrides)
    return request_key(**base)


# --- keying -----------------------------------------------------------------


def test_same_request_keys_the_same():
    assert _key() == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other-model"),
        ("schema_name", "AuthorsLLM"),
        ("system", "different system"),
        ("user_text", "different text"),
        ("max_tokens", 4096),
        ("reasoning_effort", "high"),
        ("mode", "instructor:override"),
        ("schema_json", '{"properties": {"title": {"type": "string"}}}'),
        ("chat_template_json", '{"enable_thinking": false}'),
    ],
)
def test_every_answer_changing_field_changes_the_key(field, value):
    assert _key(**{field: value}) != _key()


def test_fence_boundary_is_canonicalised_out_of_the_key():
    """10 of 13 call sites mint a fresh uuid4 boundary per call. Without this
    the same logical request hashes differently every run and the cache could
    never hit."""
    a = _key(user_text="instructions" + fence(_BOUNDARY_A, "the document"))
    b = _key(user_text="instructions" + fence(_BOUNDARY_B, "the document"))

    assert a == b


def test_alternate_marker_wording_is_canonicalised():
    """The citation prompt fences with BEGIN REFERENCES / END CITATIONS rather
    than START / END."""
    text = "--- {b} BEGIN REFERENCES ---\nrefs\n--- {b} END REFERENCES ---"
    a = _key(user_text=text.format(b=_BOUNDARY_A))
    b = _key(user_text=text.format(b=_BOUNDARY_B))

    assert a == b


def test_documents_differing_only_in_a_bare_hex_token_stay_distinct():
    """A 32-hex string in a paper's own text (an MD5, say) must NOT be
    canonicalised — that would merge two genuinely different documents."""
    assert _key(user_text=f"checksum {_BOUNDARY_A}") != _key(user_text=f"checksum {_BOUNDARY_B}")


def test_canonical_text_leaves_unfenced_input_untouched():
    assert canonical_user_text("plain prompt") == "plain prompt"


# --- storage ----------------------------------------------------------------


def test_put_then_get_round_trips(tmp_path):
    cache = LlmResponseCache(tmp_path)
    body = TitleKeywordsLLM(title="A paper").model_dump(mode="json")

    cache.put("k" * 32, body, model="m", schema_name="TitleKeywordsLLM")

    assert cache.get("k" * 32) == body


def test_get_is_a_miss_for_an_unknown_key(tmp_path):
    assert LlmResponseCache(tmp_path).get("0" * 32) is None


def test_corrupt_entry_is_a_miss_not_an_error(tmp_path):
    cache = LlmResponseCache(tmp_path)
    path = cache.path_for("k" * 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    assert cache.get("k" * 32) is None


def test_entry_from_a_future_format_version_is_ignored(tmp_path):
    cache = LlmResponseCache(tmp_path)
    path = cache.path_for("k" * 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 999, "response": {"title": "x"}}))

    assert cache.get("k" * 32) is None


def test_unwritable_root_degrades_to_uncached(tmp_path):
    """Cache failures must never break an extraction."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    cache = LlmResponseCache(blocker)

    assert cache.put("k" * 32, {"title": "x"}, model="m", schema_name="S") is False
    assert cache.get("k" * 32) is None


# --- client integration -----------------------------------------------------


class _CountingBackend:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def create(self, **kwargs):  # noqa: ARG002
        self.calls += 1
        return self.result, None


def _client(tmp_path, backend, *, enabled=True):
    settings = GlobalSettings(cache={"llm": enabled, "llm_dir": str(tmp_path)})
    client = LLMClient(settings=settings, backend=backend)
    client._limiter = mock.MagicMock()
    client._limiter.acquire = mock.AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_second_identical_extraction_is_served_from_cache(tmp_path):
    backend = _CountingBackend(TitleKeywordsLLM(title="Cached paper"))

    first = await _client(tmp_path, backend).extract_title_keywords("body text")
    # A second client shares only the on-disk cache, proving it is not in-memory.
    second = await _client(tmp_path, backend).extract_title_keywords("body text")

    assert backend.calls == 1
    assert first.title == second.title == "Cached paper"


@pytest.mark.asyncio
async def test_cache_hit_survives_a_new_fence_boundary(tmp_path):
    """extract_title_keywords mints a fresh boundary per call, so this is the
    realistic repeat-run path rather than a synthetic one."""
    backend = _CountingBackend(TitleKeywordsLLM(title="Stable"))
    client = _client(tmp_path, backend)

    await client.extract_title_keywords("body text")
    await client.extract_title_keywords("body text")

    assert backend.calls == 1


@pytest.mark.asyncio
async def test_different_input_still_calls_live(tmp_path):
    backend = _CountingBackend(TitleKeywordsLLM(title="x"))
    client = _client(tmp_path, backend)

    await client.extract_title_keywords("paper one")
    await client.extract_title_keywords("paper two")

    assert backend.calls == 2


@pytest.mark.asyncio
async def test_disabled_by_default_so_nothing_is_cached(tmp_path):
    backend = _CountingBackend(TitleKeywordsLLM(title="x"))

    await _client(tmp_path, backend, enabled=False).extract_title_keywords("body")
    await _client(tmp_path, backend, enabled=False).extract_title_keywords("body")

    assert backend.calls == 2
    assert not list(tmp_path.rglob("*.json"))


# --- rate limiting ----------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_spends_no_rate_limit_slot(tmp_path):
    """A hit costs no provider quota, so it must not wait on the budget that
    exists to protect that quota — otherwise a fully-cached corpus re-run is
    still paced at LLM_RATE_LIMIT_RPM."""
    backend = _CountingBackend(TitleKeywordsLLM(title="x"))

    warm = _client(tmp_path, backend)
    await warm.extract_title_keywords("body text")
    assert warm._limiter.acquire.await_count == 1

    served = _client(tmp_path, backend)
    await served.extract_title_keywords("body text")

    assert backend.calls == 1
    assert served._limiter.acquire.await_count == 0


@pytest.mark.asyncio
async def test_a_live_call_still_takes_exactly_one_slot(tmp_path):
    backend = _CountingBackend(TitleKeywordsLLM(title="x"))
    client = _client(tmp_path, backend, enabled=False)

    await client.extract_title_keywords("one")
    await client.extract_title_keywords("two")

    assert backend.calls == 2
    assert client._limiter.acquire.await_count == 2


@pytest.mark.asyncio
async def test_every_call_site_routes_through_the_single_acquisition():
    """The 12 per-call-site acquisitions were replaced by one inside
    _invoke_structured; a stray re-addition would double-charge the budget."""
    import pathlib

    source = pathlib.Path(LLMClient.__module__.replace(".", "/") + ".py")
    if not source.exists():  # installed rather than in-tree
        pytest.skip("llm.py not resolvable from the working tree")
    hits = [
        line
        for line in source.read_text().splitlines()
        if line.strip() == "await self._acquire_rate_limit()"
    ]
    # Initial request, transient retry, the native-to-Instructor recovery and
    # the decoder-abort recovery each acquire a slot for their physical request.
    assert len(hits) == 4


@pytest.mark.parametrize(
    "abstract_fields",
    [
        {},
        {"abstract": None},
        {"abstract": ""},
        {"abstract": "  "},
        {"abstract": "verbatim-string"},
        {"abstract": "A printed summary."},
    ],
)
def test_cache_preserves_abstract_absence_intent(tmp_path, abstract_fields):
    client = LLMClient(settings=GlobalSettings())
    cache = LlmResponseCache(tmp_path)
    result = TitleKeywordsLLM(title="A printed title", **abstract_fields)
    client._cache_store(cache, "f" * 32, result, TitleKeywordsLLM)
    recovered = client._cached_result(cache, "f" * 32, TitleKeywordsLLM)
    assert recovered.abstract == result.abstract
    assert recovered._abstract_explicitly_absent == result._abstract_explicitly_absent
