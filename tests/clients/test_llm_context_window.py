"""Inputs cut to fit the LLM server's context window.

Two blind_v1 papers failed export because the front-matter prompt ran past the
4B student's 32768-token vLLM context: the 300k-character input cap means
nothing to a CJK paper, where a character is about a token.
"""

from __future__ import annotations

import pytest

from bibr.clients.llm import LLMClient, _context_window_from_error
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM

OVERFLOW = (
    "Error code: 400 - {'error': {'message': \"This model's maximum context length is "
    "32768 tokens. However, you requested 4096 output tokens and your prompt contains "
    'at least 28673 input tokens, for a total of at least 32769 tokens."}}'
)


def _settings(**llm) -> GlobalSettings:
    s = GlobalSettings()
    for key, value in llm.items():
        setattr(s.llm, key, value)
    return s


def test_cjk_text_is_cut_to_fit_beside_the_completion():
    text = "研究" * 20_000
    capped = LLMClient._cap_input(text, _settings(context_tokens=32_768), max_tokens=8_192)

    assert capped == text[: len(capped)]
    assert len(capped) <= 32_768 - 8_192 - 2_048


def test_english_front_matter_that_fits_is_untouched():
    text = "word " * 16_000  # 80k characters, about 20k tokens
    assert LLMClient._cap_input(text, _settings(context_tokens=32_768), max_tokens=4_096) == text


def test_no_known_window_keeps_the_character_cap_only():
    text = "研究" * 20_000
    assert LLMClient._cap_input(text, _settings(), max_tokens=8_192) == text


def test_a_cap_past_the_window_still_leaves_room_for_text():
    capped = LLMClient._cap_input("x" * 50_000, _settings(context_tokens=8_192), max_tokens=8_192)
    assert len(capped) > 1_000


def test_the_window_is_read_through_wrapped_errors():
    try:
        try:
            raise ValueError(OVERFLOW)
        except ValueError as inner:
            raise RuntimeError("Max retries exceeded") from inner
    except RuntimeError as outer:
        assert _context_window_from_error(outer) == 32_768
    assert _context_window_from_error(RuntimeError("HTTP 400")) is None


@pytest.mark.parametrize(
    "message",
    [
        # vLLM 0.32's own wording (vllm/renderers/params.py). 0.32 no longer
        # rejects a prompt whose completion cap runs past the window: it
        # lowers the cap to what is left. A prompt longer than the window
        # alone is still rejected, by tokens or, before tokenizing, by
        # characters.
        "This model's maximum context length is 32768 tokens. However, your prompt contains "
        "at least 32769 input tokens, which exceeds this limit even without any output tokens. "
        "Please reduce the length of the input prompt.",
        "This model's maximum context length is 32768 tokens. However, your prompt contains "
        "500000 characters (more than 491520 characters, which is the upper bound for 32768 "
        "input tokens). Please reduce the length of the input prompt.",
    ],
)
def test_vllm_032_prompt_overflows_still_name_the_window(message):
    assert _context_window_from_error(RuntimeError(f"Error code: 400 - {message}")) == 32_768


@pytest.mark.asyncio
async def test_an_overflow_is_asked_again_with_the_input_cut():
    client = LLMClient(settings=_settings())
    sent: list[str] = []

    async def invoke(response_model, messages, system_prompt, **kwargs):
        sent.append("".join(part["text"] for part in messages[0]["content"]))
        if len(sent) == 1:
            raise RuntimeError(OVERFLOW)
        return TitleKeywordsLLM(title="研究")

    client._invoke_structured = invoke
    result = await client.extract_title_keywords("研究" * 20_000)

    assert result.title == "研究"
    assert client._settings.llm.context_tokens == 32_768
    assert len(sent) == 2 and len(sent[1]) < len(sent[0])


@pytest.mark.asyncio
async def test_an_overflow_at_the_known_window_is_not_retried():
    client = LLMClient(settings=_settings(context_tokens=32_768))
    calls = 0

    async def invoke(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(OVERFLOW)

    client._invoke_structured = invoke
    with pytest.raises(Exception):  # noqa: B017 - the classified LLM error
        await client.extract_title_keywords("研究" * 20_000)
    assert calls == 1


def test_managed_vllm_reports_the_window_it_was_started_with():
    from bibr.local.vllm_llm import max_model_len

    assert max_model_len("scienceverse/bibr-extract-qwen35-4b", None) == 32_768
    assert max_model_len("scienceverse/bibr-extract-qwen35-4b", "--max-model-len=16384") == 16_384
    assert max_model_len("some/other-model", "") is None
