"""The LLM client's pointer answers (LLM_POINTER_OUTPUTS, opt-in).

With the flag on, title/keywords asks for the abstract's first and last words
and authors for a numbered affiliation list; both are turned back into the
verbatim responses. A pointer answer that cannot be used falls back to the
verbatim call, and a service failure raises as before.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bibr.clients.llm import LLMClient
from bibr.config import Settings
from bibr.exceptions import LlmTimeoutError
from bibr.schemas import (
    AuthorLLM,
    AuthorsLLM,
    AuthorsPointerLLM,
    TitleKeywordsLLM,
    TitleKeywordsPointerLLM,
)

_TEXT = (
    "A Study of Things\n"
    "Jane Doe1, John Smith1,2\n"
    "1 University of Twente 2 North-West University\n"
    "Abstract\n"
    "We examined how things are done in practice and found\r\n"
    "that most of them are done well in all clinics.\n"
    "Keywords: things\n"
)
_POINTER_TITLE = TitleKeywordsPointerLLM(
    title="A Study of Things",
    abstract_first_words="We examined how things are done in practice",
    abstract_last_words="most of them are done well in all clinics.",
    keywords=["things"],
)
_POINTER_AUTHORS = AuthorsPointerLLM.model_validate(
    {
        "affiliations": ["University of Twente", "North-West University"],
        "authors": [
            {"given": "Jane", "family": "Doe", "affiliation_ids": [1]},
            {"given": "John", "family": "Smith", "affiliation_ids": [1, 2]},
        ],
    }
)
_VERBATIM_TITLE = TitleKeywordsLLM(title="A Study of Things", abstract="verbatim", keywords=[])
_VERBATIM_AUTHORS = AuthorsLLM(authors=[AuthorLLM(given="Jane", family="Doe")])


def _validation_error() -> ValidationError:
    try:
        AuthorsPointerLLM.model_validate({"authors": [{"corresponding": "not-a-boolean"}]})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


def _client(monkeypatch, answers: dict, *, pointers: bool = True):
    """A client whose calls answer from *answers* (response model -> value or
    exception), recording which response models were asked for."""
    monkeypatch.setattr(Settings.llm, "pointer_outputs", pointers)
    monkeypatch.setattr(Settings.llm, "merged_core_metadata", False)
    client = LLMClient()
    calls = []

    async def fake_invoke(
        response_model,
        messages,
        system_prompt,
        reasoning_effort=None,
        client_override=None,
        max_tokens=None,
    ):
        calls.append(response_model)
        answer = answers[response_model]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(client, "_invoke_structured", fake_invoke)
    return client, calls


async def test_flag_off_keeps_the_verbatim_calls(monkeypatch):
    client, calls = _client(
        monkeypatch,
        {TitleKeywordsLLM: _VERBATIM_TITLE, AuthorsLLM: _VERBATIM_AUTHORS},
        pointers=False,
    )
    result = await client.extract_core_metadata(_TEXT, include_classification=False)
    assert sorted(model.__name__ for model in calls) == ["AuthorsLLM", "TitleKeywordsLLM"]
    assert result.abstract == "verbatim"


async def test_flag_on_copies_the_abstract_and_expands_affiliations(monkeypatch):
    client, calls = _client(
        monkeypatch,
        {TitleKeywordsPointerLLM: _POINTER_TITLE, AuthorsPointerLLM: _POINTER_AUTHORS},
    )
    result = await client.extract_core_metadata(_TEXT, include_classification=False)
    assert sorted(model.__name__ for model in calls) == [
        "AuthorsPointerLLM",
        "TitleKeywordsPointerLLM",
    ]
    assert result.abstract.startswith("We examined how things are done")
    assert result.abstract.endswith("done well in all clinics.")
    assert [author.affiliation for author in result.authors] == [
        "University of Twente",
        "University of Twente; North-West University",
    ]


async def test_unplaced_abstract_falls_back_to_the_verbatim_call(monkeypatch):
    unplaced = _POINTER_TITLE.model_copy(
        update={"abstract_first_words": "these words are printed nowhere in the text"}
    )
    client, calls = _client(
        monkeypatch, {TitleKeywordsPointerLLM: unplaced, TitleKeywordsLLM: _VERBATIM_TITLE}
    )
    result = await client.extract_title_keywords(_TEXT)
    assert calls == [TitleKeywordsPointerLLM, TitleKeywordsLLM]
    assert result.abstract == "verbatim"


async def test_explicitly_absent_abstract_needs_no_verbatim_call(monkeypatch):
    absent = TitleKeywordsPointerLLM.model_validate(
        {"title": "T", "abstract_first_words": None, "abstract_last_words": None}
    )
    client, calls = _client(monkeypatch, {TitleKeywordsPointerLLM: absent})
    result = await client.extract_title_keywords(_TEXT)
    assert calls == [TitleKeywordsPointerLLM]
    assert result.abstract is None
    assert result._abstract_explicitly_absent is True


async def test_invalid_pointer_authors_fall_back_to_the_verbatim_call(monkeypatch):
    client, calls = _client(
        monkeypatch, {AuthorsPointerLLM: _validation_error(), AuthorsLLM: _VERBATIM_AUTHORS}
    )
    result = await client.extract_authors(_TEXT)
    assert calls == [AuthorsPointerLLM, AuthorsLLM]
    assert result.authors[0].family == "Doe"


async def test_numbered_affiliations_without_a_list_fall_back_to_the_verbatim_call(monkeypatch):
    unlisted = AuthorsPointerLLM.model_validate(
        {
            "affiliations": [],
            "authors": [
                {"given": "Jane", "family": "Doe", "affiliation_ids": [1]},
                {"given": "John", "family": "Smith", "affiliation_ids": [1, 2]},
            ],
        }
    )
    client, calls = _client(
        monkeypatch, {AuthorsPointerLLM: unlisted, AuthorsLLM: _VERBATIM_AUTHORS}
    )
    result = await client.extract_authors(_TEXT)
    assert calls == [AuthorsPointerLLM, AuthorsLLM]
    assert result is _VERBATIM_AUTHORS


async def test_a_service_failure_raises_without_a_second_call(monkeypatch):
    client, calls = _client(
        monkeypatch, {TitleKeywordsPointerLLM: TimeoutError("LLM call timed out after 30s")}
    )
    with pytest.raises(LlmTimeoutError):
        await client.extract_title_keywords(_TEXT)
    assert calls == [TitleKeywordsPointerLLM]


async def test_json_mode_reroll_stays_verbatim(monkeypatch):
    client, calls = _client(monkeypatch, {AuthorsLLM: _VERBATIM_AUTHORS})
    monkeypatch.setattr(client, "_get_json_mode_client", lambda: SimpleNamespace())
    await client.extract_authors(_TEXT, json_mode=True)
    assert calls == [AuthorsLLM]


async def test_nuextract_native_backend_stays_verbatim(monkeypatch):
    client, calls = _client(
        monkeypatch, {TitleKeywordsLLM: _VERBATIM_TITLE, AuthorsLLM: _VERBATIM_AUTHORS}
    )
    monkeypatch.setattr(client, "_backend_protocol", lambda backend: "nuextract-native")
    await client.extract_core_metadata(_TEXT, include_classification=False)
    assert sorted(model.__name__ for model in calls) == ["AuthorsLLM", "TitleKeywordsLLM"]


class _DictCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, body, **_):
        self.store[key] = body


@pytest.mark.parametrize(
    ("answer", "explicit"),
    [({"title": "T"}, False), ({"title": "T", "abstract_first_words": None}, True)],
)
def test_a_cached_pointer_answer_keeps_omitted_apart_from_absent(answer, explicit):
    client = LLMClient()
    cache = _DictCache()
    result = TitleKeywordsPointerLLM.model_validate(answer)
    client._cache_store(cache, "key", result, TitleKeywordsPointerLLM)
    hit = client._cached_result(cache, "key", TitleKeywordsPointerLLM)
    assert hit._abstract_explicitly_absent is explicit
