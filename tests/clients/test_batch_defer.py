"""Round 1 of ``bibr batch --llm-batch``: a deferred cache miss is spooled, not sent.

The spool row must carry everything the batch needs to send the request the
live call would have sent, under the key the live call would have stored its
answer at; round 2 then counts how many deferred calls hit.
"""

import asyncio
import json
import stat

import anthropic
import httpx
import instructor
import pytest

from bibr.clients import batch_defer
from bibr.clients.batch import AnthropicBatchAdapter
from bibr.clients.batch_defer import (
    FRONT_MATTER_LABELS,
    collecting,
    iter_spool,
    load_response_model,
    observing,
    read_row,
    read_spool,
    request_for,
)
from bibr.clients.llm import LLMClient, usage_file_context
from bibr.clients.llm_cache import LlmResponseCache, response_body
from bibr.config import GlobalSettings
from bibr.exceptions import LlmDeferred, ProcessingError
from bibr.schemas import AuthorsLLM, TitleKeywordsLLM, TitleKeywordsPointerLLM
from tests.clients.test_batch_request import _ANSWERS, BOUNDARY, TEXT, _message, live_kwargs

FILE_HASH = "0123456789abcdef"


def _settings(tmp_path, **llm) -> GlobalSettings:
    # Usage labels name the deferred calls; the suite turns them off by default.
    return GlobalSettings(
        llm={
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "api_key": "test",
            "track_usage": True,
            **llm,
        },
        cache={"llm": True, "llm_dir": str(tmp_path / "llm-cache")},
    )


def _offline_client(settings) -> LLMClient:
    client = LLMClient(settings=settings)

    def no_live_call():
        raise AssertionError("a deferred call must not reach the provider")

    client._get_client = no_live_call
    return client


# The front-matter calls the client makes, by the label it gives each.
_CALLS = {
    "extract_title_keywords": lambda c: c.extract_title_keywords(TEXT, boundary=BOUNDARY),
    "extract_authors": lambda c: c.extract_authors(TEXT, boundary=BOUNDARY),
}


def _title_call(client, label="extract_title_keywords"):
    async def call():
        with usage_file_context(f"{FILE_HASH}#7"):
            return await _CALLS[label](client)

    return call


def _defer_title(tmp_path, settings, label="extract_title_keywords") -> dict:
    spool = tmp_path / "out" / "llm_batch" / "spool.jsonl"
    client = _offline_client(settings)
    with collecting(spool), pytest.raises(LlmDeferred) as caught:
        asyncio.run(_title_call(client, label)())
    assert caught.value.error_code == "llm_deferred"
    assert isinstance(caught.value, ProcessingError)
    (row,) = read_spool(spool)
    return row


def test_a_deferred_miss_is_spooled_privately_and_not_sent(tmp_path):
    settings = _settings(tmp_path)
    row = _defer_title(tmp_path, settings)

    spool = tmp_path / "out" / "llm_batch" / "spool.jsonl"
    assert stat.S_IMODE(spool.stat().st_mode) == 0o600
    assert stat.S_IMODE(spool.parent.stat().st_mode) == 0o700
    assert row["label"] == "extract_title_keywords"
    assert row["file_hash"] == FILE_HASH
    assert row["model"] == "claude-haiku-4-5"
    assert row["response_model"] == "bibr.schemas:TitleKeywordsLLM"
    # The prompt builder's parts, cache flags included.
    assert any(part.get("cache") for part in row["messages"][0]["content"])
    assert row["call_kwargs"]["temperature"] == 0.0


def test_the_spooled_key_is_where_the_live_call_stores_its_answer(tmp_path):
    settings = _settings(tmp_path)
    row = _defer_title(tmp_path, settings)

    # The same call made live, outside any batch round, stores its answer.
    live_kwargs(settings, lambda c: c.extract_title_keywords(TEXT, boundary="c" * 32))

    assert LlmResponseCache(settings=settings).get(row["key"]) is not None


# The response model each call sends, without and with LLM_POINTER_OUTPUTS.
_SCHEMAS = {
    ("extract_title_keywords", False): "TitleKeywordsLLM",
    ("extract_title_keywords", True): "TitleKeywordsPointerLLM",
    ("extract_authors", False): "AuthorsLLM",
    ("extract_authors", True): "AuthorsPointerLLM",
}


@pytest.mark.parametrize("pointers", [False, True], ids=["verbatim", "pointers"])
@pytest.mark.parametrize("label", ["extract_title_keywords", "extract_authors"])
@pytest.mark.parametrize(
    "llm",
    [{}, {"thinking_budget": 2048, "title_max_tokens": 8192}, {"model": "claude-haiku-5-5"}],
)
def test_the_batch_request_of_a_spool_row_is_the_live_call(tmp_path, llm, label, pointers):
    settings = _settings(tmp_path, **llm, pointer_outputs=pointers)
    row = _defer_title(tmp_path, settings, label)

    request = request_for(row)
    params = AnthropicBatchAdapter(model=row["model"], settings=settings).build_params(request)
    live = live_kwargs(settings, _CALLS[label])

    schema = _SCHEMAS[label, pointers]
    assert row["schema"] == schema
    assert row["response_model"] == f"bibr.schemas:{schema}"
    assert request.custom_id == row["key"]
    assert params == live
    assert [tool["name"] for tool in live["tools"]] == [schema]
    # The live call, made outside any batch round, stores its answer where
    # the round-2 replay will look for the batch's.
    assert LlmResponseCache(settings=settings).get(row["key"]) is not None


@pytest.mark.parametrize("pointers", [False, True], ids=["verbatim", "pointers"])
@pytest.mark.parametrize("label", ["extract_title_keywords", "extract_authors"])
def test_a_batch_answer_is_stored_exactly_as_the_live_answer(tmp_path, label, pointers):
    """The filler and the live client write the same body for the same answer."""
    settings = _settings(tmp_path, pointer_outputs=pointers)
    row = _defer_title(tmp_path, settings, label)
    live_kwargs(settings, _CALLS[label])
    stored = LlmResponseCache(settings=settings).get(row["key"])

    schema = load_response_model(row["response_model"])
    message = anthropic.types.Message.model_validate(
        _message(schema.__name__, _ANSWERS[schema.__name__])
    )
    parsed = AnthropicBatchAdapter._parse(message, schema, {})

    assert response_body(parsed) == stored


@pytest.mark.parametrize(
    "anchors, explicitly_absent",
    [
        ({}, False),  # omitted: the abstract stays omitted
        ({"abstract_first_words": None, "abstract_last_words": None}, True),  # "no abstract"
        ({"abstract_first_words": "", "abstract_last_words": None}, False),  # blank
    ],
)
def test_a_pointer_answer_keeps_its_abstract_absence_through_the_cache(anchors, explicitly_absent):
    """The cache holds the pointer answer before copy-back, so what it stores
    must validate back to the same answer: an omitted abstract and a "no
    abstract" give different exports. The stored body leaves out the anchor
    fields an answer did not fill unless the model said there is no abstract
    (``_abstract_fields``), for the batch filler and the live client alike."""
    from bibr.clients.pointer_outputs import resolve_title_keywords

    message = anthropic.types.Message.model_validate(
        _message("TitleKeywordsPointerLLM", {"title": "A Study of Things", **anchors})
    )
    parsed = AnthropicBatchAdapter._parse(message, TitleKeywordsPointerLLM, {})
    hit = TitleKeywordsPointerLLM.model_validate(response_body(parsed))

    assert parsed._abstract_explicitly_absent is explicitly_absent
    assert hit._abstract_explicitly_absent is explicitly_absent
    live_copy = resolve_title_keywords(parsed, TEXT).response
    hit_copy = resolve_title_keywords(hit, TEXT).response
    assert hit_copy.model_dump() == live_copy.model_dump()
    assert hit_copy.model_fields_set == live_copy.model_fields_set
    assert hit_copy._abstract_explicitly_absent is live_copy._abstract_explicitly_absent


def test_a_hit_is_served_and_not_spooled(tmp_path):
    settings = _settings(tmp_path)
    row = _defer_title(tmp_path, settings)
    LlmResponseCache(settings=settings).put(
        row["key"],
        {"title": "Cached", "keywords": []},
        model=row["model"],
        schema_name=row["schema"],
        source="batch",
    )
    spool = tmp_path / "second" / "spool.jsonl"

    with collecting(spool):
        result = asyncio.run(_title_call(_offline_client(settings))())

    assert result.title == "Cached"
    assert not spool.exists()


def test_other_labels_and_overrides_are_not_deferred(tmp_path):
    settings = _settings(tmp_path)
    spool = tmp_path / "spool.jsonl"
    client = LLMClient(settings=settings)
    captured = []

    def answer(request):
        captured.append(request)
        return httpx.Response(200, json=_message("AuthorsLLM", _ANSWERS["AuthorsLLM"]))

    raw = anthropic.AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer))
    )
    live = instructor.from_anthropic(raw, model="claude-haiku-4-5")
    client._get_client = lambda: live
    messages = [{"role": "user", "content": "authors please"}]

    async def calls():
        # A label outside the deferred set goes live.
        await client.invoke_structured(AuthorsLLM, messages, "sys", label="resolve_citations")
        # A caller-supplied client (the JSON-mode re-roll) goes live too.
        await client._run_labeled_call(
            "extract_authors",
            lambda: client._invoke_structured(AuthorsLLM, messages, "sys2", client_override=live),
        )

    with collecting(spool, FRONT_MATTER_LABELS):
        asyncio.run(calls())

    assert len(captured) == 2
    assert not spool.exists()


def _lookup(client, key, *, hit, label="extract_authors", file_hash=FILE_HASH):
    """One cache lookup of a labeled call, as ``_invoke_structured`` reports it."""

    async def hook():
        batch_defer.on_cache_lookup(
            client,
            key,
            hit=hit,
            deferrable=True,
            request=(AuthorsLLM, [{"role": "user", "content": "u"}], "sys", None, None),
        )
        return "went live"

    async def call():
        with usage_file_context(f"{file_hash}#1" if file_hash else None):
            return await client._run_labeled_call(label, hook)

    return asyncio.run(call())


def test_other_providers_are_never_deferred(tmp_path):
    settings = GlobalSettings(
        llm={"provider": "openai", "model": "gpt-5-mini", "api_key": "test", "track_usage": True},
        cache={"llm": True, "llm_dir": str(tmp_path / "c")},
    )
    spool = tmp_path / "spool.jsonl"
    with collecting(spool):
        assert _lookup(LLMClient(settings=settings), "a" * 32, hit=False) == "went live"
    assert not spool.exists()


def test_an_unwritable_spool_lets_the_call_go_live(tmp_path):
    spool = tmp_path / "spool.jsonl"
    spool.mkdir()  # a directory where the file should be
    client = LLMClient(settings=_settings(tmp_path))
    with collecting(spool) as mode:
        assert _lookup(client, "a" * 32, hit=False) == "went live"
    assert (mode.spooled, mode.write_failures) == (0, 1)


def test_a_call_without_a_label_is_not_deferred(tmp_path):
    spool = tmp_path / "spool.jsonl"
    client = LLMClient(settings=_settings(tmp_path))
    with collecting(spool):
        batch_defer.on_cache_lookup(
            client,
            "a" * 32,
            hit=False,
            deferrable=True,
            request=(AuthorsLLM, [], "sys", None, None),
        )
    assert not spool.exists()


def test_rounds_do_not_nest():
    with collecting("spool.jsonl"), pytest.raises(RuntimeError), observing(()):
        pass
    assert batch_defer._mode is None


def test_the_replay_counts_the_deferred_calls_only(tmp_path):
    client = LLMClient(settings=_settings(tmp_path))
    updates = []
    expected = {
        (FILE_HASH, "extract_authors", "AuthorsLLM"),
        ("ffffffffffffffff", "extract_authors", "AuthorsLLM"),
    }
    with observing(expected, unanswered_keys={"e" * 32}, on_update=updates.append) as stats:
        _lookup(client, "a" * 32, hit=True)
        _lookup(client, "b" * 32, hit=False)
        _lookup(client, "e" * 32, hit=False, file_hash="ffffffffffffffff")
        # Not deferred in round 1 (a later call of another label): not counted.
        _lookup(client, "c" * 32, hit=False, label="extract_paper_classification")
        _lookup(client, "d" * 32, hit=False, file_hash="1111111111111111")
        # A label outside the round is never counted.
        _lookup(client, "f" * 32, hit=False, label="resolve_citations")

    assert (stats.hits, stats.misses, stats.unanswered, stats.fallbacks) == (1, 1, 1, 0)
    assert stats.hit_rate == 0.5
    assert len(updates) == 3


def _pointer_lookup(client, key, *, hit, schema):
    """A lookup of extract_title_keywords in the form *schema*."""

    async def hook():
        batch_defer.on_cache_lookup(
            client,
            key,
            hit=hit,
            deferrable=True,
            request=(schema, [{"role": "user", "content": "u"}], "sys", None, None),
        )

    async def call():
        with usage_file_context(f"{FILE_HASH}#2"):
            await client._run_labeled_call("extract_title_keywords", hook)

    asyncio.run(call())


def test_a_verbatim_re_ask_after_a_pointer_answer_is_a_fallback_not_a_miss(tmp_path):
    """LLM_POINTER_OUTPUTS: round 1 deferred the pointer call. When bibr cannot
    use the batch's pointer answer, the verbatim call it makes instead was
    never batched; it goes live and must not pull the hit rate under the
    replay's stop threshold."""
    client = LLMClient(settings=_settings(tmp_path))
    expected = {(FILE_HASH, "extract_title_keywords", "TitleKeywordsPointerLLM")}
    with observing(expected) as stats:
        _pointer_lookup(client, "a" * 32, hit=True, schema=TitleKeywordsPointerLLM)
        _pointer_lookup(client, "b" * 32, hit=False, schema=TitleKeywordsLLM)

    assert (stats.hits, stats.misses, stats.fallbacks) == (1, 0, 1)
    assert stats.hit_rate == 1.0

    # Round 1 deferred the verbatim call (the pointer answer was cached and
    # unusable): the cached pointer answer is not counted, the verbatim call is.
    expected = {(FILE_HASH, "extract_title_keywords", "TitleKeywordsLLM")}
    with observing(expected) as stats:
        _pointer_lookup(client, "a" * 32, hit=True, schema=TitleKeywordsPointerLLM)
        _pointer_lookup(client, "b" * 32, hit=True, schema=TitleKeywordsLLM)

    assert (stats.hits, stats.misses, stats.fallbacks) == (1, 0, 0)


def test_read_spool_keeps_the_first_row_per_key_and_skips_junk(tmp_path):
    spool = tmp_path / "spool.jsonl"
    good = {"v": 1, "key": "a" * 32, "messages": [], "n": 1}
    spool.write_text(
        "\n".join(
            [
                json.dumps(good),
                "{not json",
                json.dumps({**good, "n": 2}),
                json.dumps({**good, "key": "../../etc"}),
                json.dumps({**good, "v": 99, "key": "b" * 32}),
                "",
            ]
        )
    )
    assert read_spool(spool) == [good]
    assert read_spool(tmp_path / "missing.jsonl") == []


def test_spool_rows_stream_and_load_again_by_offset(tmp_path):
    """The fill keeps offsets, not rows: a corpus-scale spool need not fit in memory."""
    spool = tmp_path / "spool.jsonl"
    rows = [
        {"v": 1, "key": f"{i:032x}", "messages": [{"role": "user", "content": "Müller ✓"}]}
        for i in range(3)
    ]
    lines = [json.dumps(rows[0], ensure_ascii=False), "{torn", json.dumps(rows[1]), ""]
    spool.write_text("\n".join([*lines, json.dumps(rows[2], ensure_ascii=False)]) + "\n")

    pairs = list(iter_spool(spool))

    assert [row for _, row in pairs] == rows
    with spool.open("rb") as handle:
        assert [read_row(handle, offset) for offset, _ in pairs] == rows
        with pytest.raises(ValueError):
            read_row(handle, pairs[0][0] + 1)
    assert list(iter_spool(tmp_path / "missing.jsonl")) == []


@pytest.mark.parametrize(
    "path", ["os:system", "bibr.schemas:json", "bibr.schemas", "pydantic:BaseModel"]
)
def test_only_bibr_response_models_load(path):
    with pytest.raises((ValueError, AttributeError)):
        load_response_model(path)


def test_bibr_response_models_load():
    assert load_response_model("bibr.schemas:AuthorsLLM") is AuthorsLLM
