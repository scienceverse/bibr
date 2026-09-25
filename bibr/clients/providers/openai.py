"""OpenAI provider adapter."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, ClassVar

import instructor

from bibr.clients.providers import register
from bibr.config import snapshot_settings
from bibr.utils.hosts import refuse_public_plaintext

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

# ``LLM_INSTRUCTOR_MODE`` value -> instructor mode for custom-base_url servers.
_MODE_ALIASES = {
    "json": instructor.Mode.JSON,
    "md_json": instructor.Mode.MD_JSON,
    "markdown_json": instructor.Mode.MD_JSON,
    "json_schema": instructor.Mode.JSON_SCHEMA,
    "tools": instructor.Mode.TOOLS,
}


def _resolve_instructor_mode(raw: str) -> instructor.Mode:
    """Resolve ``LLM_INSTRUCTOR_MODE`` to an instructor mode.

    Default (empty / unknown) is JSON_SCHEMA — strict guided decoding, the
    safe choice for capable models. ``json`` (json_object) trades strictness
    for prompt-driven field population, which small local models need to
    avoid skipping optional fields under a strict grammar.
    """
    return _MODE_ALIASES.get(raw.strip().lower(), instructor.Mode.JSON_SCHEMA)


@register
class OpenAIProvider:
    name: ClassVar[str] = "openai"

    def __init__(self, settings: GlobalSettings | None = None) -> None:
        self._settings = settings if settings is not None else snapshot_settings()

    def build_client(
        self, mode_override: instructor.Mode | None = None
    ) -> instructor.AsyncInstructor:
        api_key = self._settings.llm.api_key
        if not api_key and not self._settings.llm.base_url:
            raise ValueError("OpenAI API key required. Set LLM_API_KEY environment variable.")
        kwargs: dict = {"async_client": True, "api_key": api_key or "not-needed"}
        if self._settings.llm.base_url:
            if api_key and not self._settings.llm.allow_insecure_http:
                refuse_public_plaintext(
                    self._settings.llm.base_url,
                    credential="LLM API key",
                    opt_out="set LLM_ALLOW_INSECURE_HTTP=true",
                )
            kwargs["base_url"] = self._settings.llm.base_url
            # A custom base_url means a local/self-hosted OpenAI-compatible
            # server (LM Studio, vLLM, llama.cpp, …). Those commonly reject
            # instructor's default TOOLS mode (object-form ``tool_choice``)
            # but accept JSON_SCHEMA structured outputs. Real OpenAI keeps the
            # default TOOLS mode. ``LLM_INSTRUCTOR_MODE`` overrides (e.g. "json"
            # for small models that skip optional fields under strict grammar).
            # ``mode_override`` (set for the empty-author re-roll) forces a
            # specific mode regardless of the configured default — free-JSON
            # mode drops the strict grammar's []-is-minimal-valid escape.
            kwargs["mode"] = (
                mode_override
                if mode_override is not None
                else _resolve_instructor_mode(self._settings.llm.instructor_mode)
            )
        client: instructor.AsyncInstructor = instructor.from_provider(
            f"openai/{self._settings.llm.model}", **kwargs
        )
        return client

    def call_kwargs(self, reasoning_effort: str | None, max_tokens: int | None = None) -> dict:
        kwargs: dict = {"temperature": self._settings.llm.temperature}
        if self._settings.llm.base_url and self._settings.llm.chat_template_kwargs:
            kwargs["extra_body"] = {
                "chat_template_kwargs": copy.deepcopy(self._settings.llm.chat_template_kwargs)
            }
        # Self-hosted OpenAI-compatible servers (vLLM/SGLang/vllm-mlx via
        # LLM_BASE_URL) reject caps above their (to us, unknown) max_model_len
        # on tight-context servers. Omit only the implicit global default so
        # the server can auto-fit generation to max_model_len - prompt_len.
        # Explicit task-level caps are intentional runaway guards and must win.
        omit_cap = (
            self._settings.llm.base_url
            and "max_tokens" not in self._settings.llm.model_fields_set
            and max_tokens is None
        )
        if not omit_cap:
            cap = max_tokens or self._settings.llm.max_tokens
            if self._settings.llm.base_url:
                # OpenAI-compatible local runtimes generally implement the
                # older chat-completions field. vllm-mlx 0.4 ignores
                # max_completion_tokens entirely and falls back to its 32k
                # default, which can stall local batched inference.
                kwargs["max_tokens"] = cap
            else:
                kwargs["max_completion_tokens"] = cap
        effort = (
            reasoning_effort
            if reasoning_effort is not None
            else self._settings.llm.reasoning_effort
        )
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs
