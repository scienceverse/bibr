"""Extra request-body fields and reasoning-effort omission for OpenAI-compatible servers.

``LLM_EXTRA_BODY`` carries server-specific fields (e.g. a hosted API's
thinking-mode switch) that the OpenAI schema has no name for.
``LLM_CHAT_TEMPLATE_KWARGS`` keeps its own meaning and is merged in.
"""

from bibr.clients import providers
from bibr.config import GlobalSettings

_LOCAL = "http://localhost:8000/v1"
_DISABLE_THINKING = {"thinking": {"type": "disabled"}}


def _kwargs(per_call=None, **llm):
    settings = GlobalSettings(llm={"provider": "openai", **llm})
    return providers.get("openai", settings=settings).call_kwargs(per_call)


def test_extra_body_is_sent_to_a_custom_endpoint():
    kwargs = _kwargs(base_url=_LOCAL, extra_body=_DISABLE_THINKING)
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_extra_body_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    assert _kwargs(base_url=_LOCAL)["extra_body"] == _DISABLE_THINKING


def test_chat_template_kwargs_alone_send_exactly_what_they_did_before():
    kwargs = _kwargs(base_url=_LOCAL, chat_template_kwargs={"enable_thinking": False})
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_chat_template_kwargs_merge_into_extra_body_and_win_on_shared_keys():
    kwargs = _kwargs(
        base_url=_LOCAL,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True, "custom": 1},
        },
        chat_template_kwargs={"enable_thinking": False},
    )
    assert kwargs["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False, "custom": 1},
    }


def test_no_extra_body_without_either_setting():
    assert "extra_body" not in _kwargs(base_url=_LOCAL)


def test_real_openai_does_not_receive_extra_body():
    kwargs = _kwargs(extra_body=_DISABLE_THINKING, chat_template_kwargs={"x": 1})
    assert "extra_body" not in kwargs


def test_mutating_a_request_never_alters_the_settings():
    settings = GlobalSettings(
        llm={
            "provider": "openai",
            "base_url": _LOCAL,
            "extra_body": {"thinking": {"type": "disabled"}},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    provider = providers.get("openai", settings=settings)
    first = provider.call_kwargs(None)["extra_body"]
    first["thinking"]["type"] = "enabled"
    first["chat_template_kwargs"]["enable_thinking"] = True

    assert provider.call_kwargs(None)["extra_body"] == {
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert settings.llm.extra_body == {"thinking": {"type": "disabled"}}


# --- reasoning_effort -------------------------------------------------------


def test_reasoning_effort_is_sent_by_default():
    assert _kwargs(base_url=_LOCAL)["reasoning_effort"] == "minimal"


def test_empty_reasoning_effort_setting_omits_the_field(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    assert "reasoning_effort" not in _kwargs(base_url=_LOCAL)
    monkeypatch.delenv("LLM_REASONING_EFFORT")
    assert "reasoning_effort" not in _kwargs(base_url=_LOCAL, reasoning_effort=None)


def test_empty_per_call_override_omits_the_field(monkeypatch):
    # The authors/citations calls pass their own override; an empty
    # LLM_REASONING_EFFORT_AUTHORS reaches the adapter as "".
    monkeypatch.setenv("LLM_REASONING_EFFORT_AUTHORS", "")
    settings = GlobalSettings(llm={"provider": "openai", "base_url": _LOCAL})
    per_call = settings.llm.reasoning_effort_authors
    assert per_call == ""
    assert "reasoning_effort" not in (
        providers.get("openai", settings=settings).call_kwargs(per_call)
    )


def test_per_call_override_still_applies_when_the_default_is_empty():
    assert _kwargs("low", base_url=_LOCAL, reasoning_effort="")["reasoning_effort"] == "low"
