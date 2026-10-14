"""Batch requests are the live Anthropic call, byte for byte.

A batch answer can only stand in for a live one when the request is the
request the live path would have sent. The batch adapter used to build its
own: no temperature (Haiku 4.5 then samples at 1.0), a 1024-token cap, the
prompt parts flattened (losing the prompt-cache marks) and a strict
sanitized tool that is not Instructor's. These tests drive the real live path
(``LLMClient`` → Instructor → the anthropic SDK over a mock transport),
capture the arguments of the SDK call through Instructor's
``completion:kwargs`` hook, and compare them with the batch parameters.
"""

import asyncio

import httpx
import instructor
import pytest

from bibr.clients.batch import (
    _TOOL_ONLY_INSTRUCTION,
    AnthropicBatchAdapter,
    BatchError,
    BatchRequest,
)
from bibr.clients.llm import LLMClient, _task_max_tokens
from bibr.config import GlobalSettings
from bibr.schemas import AuthorsLLM, PaperClassificationLLM, PaperTypeLabel, TitleKeywordsLLM

# The SDK comes with the `batch` extra, which the core CI environments leave out.
anthropic = pytest.importorskip("anthropic")

BOUNDARY = "b" * 32
TEXT = "A Study of Things\nJane Doe, University of Somewhere\nAbstract. We studied things."

_ANSWERS = {
    "TitleKeywordsLLM": {"title": "A Study of Things", "abstract": None, "keywords": []},
    "AuthorsLLM": {"authors": [{"given": "Jane", "family": "Doe"}]},
    "PaperClassificationLLM": {"paper_type": None},
    "PaperTypeLabel": {"paper_type": "review", "confidence": 0.8},
    # LLM_POINTER_OUTPUTS: answers bibr can copy back out of TEXT, so the
    # live call makes no verbatim re-ask.
    "TitleKeywordsPointerLLM": {
        "title": "A Study of Things",
        "abstract_first_words": "We studied things.",
        "abstract_last_words": "We studied things.",
        "keywords": [],
    },
    "AuthorsPointerLLM": {
        "affiliations": ["University of Somewhere"],
        "authors": [{"given": "Jane", "family": "Doe", "affiliation_ids": [1]}],
    },
}


def _message(tool_name: str, payload: dict, *, stop_reason: str = "tool_use") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "tool_use", "id": "tu_1", "name": tool_name, "input": payload}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _settings(**llm) -> GlobalSettings:
    return GlobalSettings(
        llm={"provider": "anthropic", "model": "claude-haiku-4-5", "api_key": "test", **llm}
    )


def live_kwargs(settings: GlobalSettings, call, *, client: LLMClient | None = None) -> dict:
    """The SDK call arguments of one live structured call, via Instructor's hook."""
    captured: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = request.read().decode().replace(" ", "")
        name = next(n for n in _ANSWERS if f'"name":"{n}"' in body)
        return httpx.Response(200, json=_message(name, _ANSWERS[name]))

    raw = anthropic.AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer))
    )
    live = instructor.from_anthropic(raw, model=settings.llm.model)
    live.on("completion:kwargs", lambda *args, **kwargs: captured.append(kwargs))
    client = client if client is not None else LLMClient(settings=settings)
    client._get_client = lambda: live
    asyncio.run(call(client))
    assert len(captured) == 1
    return captured[0]


def _front_matter_requests(settings: GlobalSettings) -> dict[str, tuple]:
    """(live call, equivalent batch request) per front-matter task.

    With ``LLM_POINTER_OUTPUTS`` the title/keywords and authors calls send
    their pointer prompts and response models.
    """
    llm = settings.llm
    pointers = "_pointers" if llm.pointer_outputs else ""
    return {
        "title_keywords": (
            lambda c: c.extract_title_keywords(TEXT, boundary=BOUNDARY),
            BatchRequest.from_spec(
                "t",
                f"title_keywords{pointers}",
                boundary=BOUNDARY,
                text=TEXT,
                max_tokens=_task_max_tokens(settings, llm.title_max_tokens),
            ),
        ),
        "authors": (
            lambda c: c.extract_authors(TEXT, boundary=BOUNDARY),
            BatchRequest.from_spec(
                "a",
                f"authors{pointers}",
                boundary=BOUNDARY,
                text=TEXT,
                max_tokens=_task_max_tokens(settings, llm.authors_max_tokens),
                reasoning_effort=llm.reasoning_effort_authors,
            ),
        ),
        "classification": (
            lambda c: c.extract_paper_classification(TEXT, boundary=BOUNDARY),
            BatchRequest.from_spec(
                "c",
                "classification",
                boundary=BOUNDARY,
                text=TEXT[: llm.classification_max_chars],
                max_tokens=_task_max_tokens(settings, llm.paper_classification_max_tokens),
            ),
        ),
    }


SETTINGS_VARIANTS = {
    # The setup default: temperature 0.
    "haiku-4-5": {},
    # A thinking budget: no temperature, auto tool choice plus Instructor's note.
    "haiku-4-5-thinking": {"thinking_budget": 2048, "title_max_tokens": 8192},
    # Sampling-free: neither temperature nor thinking; forced tool kept.
    "haiku-5-5": {"model": "claude-haiku-5-5"},
}


@pytest.mark.parametrize("pointers", [False, True], ids=["verbatim", "pointers"])
@pytest.mark.parametrize("variant", sorted(SETTINGS_VARIANTS))
@pytest.mark.parametrize("task", ["title_keywords", "authors", "classification"])
def test_batch_params_equal_the_live_call(variant, task, pointers):
    settings = _settings(**SETTINGS_VARIANTS[variant], pointer_outputs=pointers)
    call, request = _front_matter_requests(settings)[task]

    live = live_kwargs(settings, call)
    adapter = AnthropicBatchAdapter(model=settings.llm.model, settings=settings)

    assert adapter.build_params(request) == live


@pytest.mark.parametrize(
    "task, tool",
    [("title_keywords", "TitleKeywordsPointerLLM"), ("authors", "AuthorsPointerLLM")],
)
def test_pointer_outputs_send_the_pointer_request(task, tool):
    """The parity above is not two verbatim requests compared: the flag is on."""
    settings = _settings(pointer_outputs=True)
    call, _ = _front_matter_requests(settings)[task]

    live = live_kwargs(settings, call)

    assert live["tool_choice"] == {"type": "tool", "name": tool}


def test_live_parity_keeps_the_prompt_cache_marks_and_temperature():
    """The three things the old builder dropped, spelled out."""
    settings = _settings()
    _, request = _front_matter_requests(settings)["title_keywords"]
    params = AnthropicBatchAdapter(model="claude-haiku-4-5", settings=settings).build_params(
        request
    )
    assert params["temperature"] == 0.0
    assert params["max_tokens"] == settings.llm.title_max_tokens
    blocks = params["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert params["tool_choice"] == {"type": "tool", "name": "TitleKeywordsLLM"}
    assert params["tools"][0]["input_schema"] == TitleKeywordsLLM.model_json_schema()


def test_spooled_call_kwargs_win_over_the_settings():
    settings = _settings()
    request = BatchRequest(
        custom_id="k",
        schema=PaperTypeLabel,
        system="sys",
        user_text="text",
        call_kwargs={"max_tokens": 777, "temperature": 0.0},
    )
    params = AnthropicBatchAdapter(settings=settings).build_params(request)
    assert params["max_tokens"] == 777
    assert params["messages"] == [{"role": "user", "content": "text"}]
    assert params["system"] == [{"type": "text", "text": "sys"}]


def test_model_defaults_to_the_batch_model_setting():
    settings = _settings(batch_model="claude-haiku-5-5")
    assert AnthropicBatchAdapter(settings=settings).model == "claude-haiku-5-5"


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"])
def test_forced_tool_choice_is_dropped_for_models_that_reject_it(model):
    settings = _settings(model=model)
    _, request = _front_matter_requests(settings)["authors"]
    params = AnthropicBatchAdapter(model=model, settings=settings).build_params(request)

    assert params["tool_choice"] == {"type": "auto"}
    assert params["system"][-1] == {"type": "text", "text": _TOOL_ONLY_INSTRUCTION}
    assert "temperature" not in params
    assert "thinking" not in params


def test_haiku_5_5_keeps_the_forced_tool():
    settings = _settings(model="claude-haiku-5-5")
    _, request = _front_matter_requests(settings)["authors"]
    params = AnthropicBatchAdapter(model="claude-haiku-5-5", settings=settings).build_params(
        request
    )
    assert params["tool_choice"] == {"type": "tool", "name": "AuthorsLLM"}


def test_from_spec_resolves_prompt_and_schema():
    req = BatchRequest.from_spec(
        "paper-1", "paper_type_label", title="A Title", abstract="An abstract"
    )
    assert req.custom_id == "paper-1"
    assert req.schema is PaperTypeLabel
    assert req.system == "You are a scientific paper classifier."
    assert "A Title" in req.user_text
    assert "An abstract" in req.user_text
    assert req.messages[0]["role"] == "user"


class _Rows:
    def __init__(self, rows):
        self.messages = type("M", (), {"batches": self})()
        self._rows = rows

    def results(self, batch_id):
        return iter(self._rows)


def _row(custom_id, message):
    succeeded = type("R", (), {"type": "succeeded", "message": message})()
    return type("Row", (), {"custom_id": custom_id, "result": succeeded})()


def test_results_validate_like_the_live_call():
    """Instructor's parser, so private validator state matches the live result."""
    message = anthropic.types.Message.model_validate(
        _message("TitleKeywordsLLM", {"title": "T", "abstract": None, "keywords": []})
    )
    adapter = AnthropicBatchAdapter(client=_Rows([_row("t", message)]), settings=_settings())
    request = BatchRequest(custom_id="t", schema=TitleKeywordsLLM, system="s", user_text="u")

    result = adapter.retrieve("batch_x", [request])["t"]

    assert isinstance(result, TitleKeywordsLLM)
    assert result.title == "T"
    assert result._abstract_explicitly_absent is True


def test_a_truncated_answer_is_a_row_error_not_a_result():
    message = anthropic.types.Message.model_validate(
        _message("AuthorsLLM", {"authors": []}, stop_reason="max_tokens")
    )
    adapter = AnthropicBatchAdapter(client=_Rows([_row("a", message)]), settings=_settings())
    request = BatchRequest(custom_id="a", schema=AuthorsLLM, system="s", user_text="u")

    result = adapter.retrieve("batch_x", [request])["a"]

    assert isinstance(result, BatchError)
    assert result.error_type == "validation_error"


def test_split_respects_the_request_and_byte_limits(monkeypatch):
    import bibr.clients.batch as batch

    settings = _settings()
    adapter = AnthropicBatchAdapter(settings=settings)
    requests = [
        BatchRequest(custom_id=f"k{i}", schema=PaperClassificationLLM, system="s", user_text="x")
        for i in range(5)
    ]
    monkeypatch.setattr(batch, "MAX_BATCH_REQUESTS", 2)
    assert [len(p) for p in adapter.split(requests)] == [2, 2, 1]

    monkeypatch.setattr(batch, "MAX_BATCH_REQUESTS", 100)
    one = len(batch.json.dumps(adapter._build_request(requests[0])).encode()) + 1
    monkeypatch.setattr(batch, "_BATCH_BYTE_BUDGET", one * 3)
    assert [len(p) for p in adapter.split(requests)] == [3, 2]
