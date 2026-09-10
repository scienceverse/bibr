"""NuExtract native-template structured backend.

NuExtract 3 was trained to read a JSON-like template from chat template
kwargs. For self-hosted OpenAI-compatible inference this uses the model's
native protocol instead of an external JSON-schema constrained decoder while
still validating the final result with bibr's Pydantic response models.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from bibr.clients.nuextract_schema import (
    NativeSchemaContract,
    NuExtractSchemaPolicy,
    native_contract_for_model,
    native_wire_value_is_valid,
)
from bibr.config import GlobalSettings, snapshot_settings

__all__ = [
    "NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256",
    "NUEXTRACT3_FP8_EXPECTED_REVISION",
    "NativeCompletionEnvelope",
    "NativeSchemaContract",
    "NativeInvalidCategory",
    "NuExtractNativeBackend",
    "NuExtractInvalidOutput",
    "NuExtractSchemaPolicy",
    "build_native_request_kwargs",
    "native_contract_for_model",
    "normalize_native_completion",
    "parse_native_completion",
    "template_for_model",
]

logger = logging.getLogger(__name__)

NUEXTRACT3_FP8_EXPECTED_REVISION = "d88964bad5ba47333cb721b351e19045ee6a6fc0"
NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256 = (
    "31e44d28615d268efdc3dcf59cb59bd2d51714d517455fbad97518d351b84119"
)


def _sha256_text(s: str) -> str:
    """Lowercase hex sha256 of the utf-8 bytes of *s*."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


_GENERIC_INSTRUCTIONS = (
    "Extract the requested fields from the document data and return only "
    "the completed JSON object. Use null or [] when a field is absent."
)

NativeInvalidCategory = Literal[
    "empty",
    "non_json",
    "truncated",
    "trailing_content",
    "non_object",
    "schema_invalid",
]


@dataclass(frozen=True, slots=True, repr=False)
class NativeCompletionEnvelope:
    """Normalized native response boundary shared by production and benchmarks.

    ``raw`` is intentionally excluded from the generated representation. It is
    consumed immediately by the strict parser and must not enter receipts or
    ordinary diagnostics.
    """

    raw: str
    finish_reason: str | None
    input_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_input_tokens: int
    has_usage: bool


@dataclass(repr=False)
class NuExtractInvalidOutput(Exception):
    """Safe diagnostic for a rejected native completion.

    Raw response text is deliberately represented only by its length and
    digest so this exception can cross retry, metric, log, and API boundaries.
    """

    category: NativeInvalidCategory
    model: str
    finish_reason: str | None
    response_chars: int
    response_sha256: str
    input_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_input_tokens: int
    response_model: str
    # Raw completion, populated ONLY for ``category == "truncated"``. A
    # truncated batch's completion holds the complete leading references
    # before the cut, and the batched reference parser salvages them
    # (:func:`bibr.extract.ref_extractor._salvage_truncated_batch`) instead of
    # losing the whole batch — a path the native backend could never reach,
    # because this exception carries no text and the Instructor-only
    # ``last_completion`` bridge found nothing to read. Excluded from
    # ``__str__``/``__repr__`` (and from equality) so the safety property in
    # the class docstring still holds: nothing here reaches receipts, metrics,
    # logs, or the API.
    salvage_raw: str = field(default="", repr=False, compare=False)

    def __str__(self) -> str:
        return (
            "NuExtractInvalidOutput("
            f"category={self.category!r}, "
            f"model={self.model!r}, "
            f"finish_reason={self.finish_reason!r}, "
            f"response_chars={self.response_chars}, "
            f"response_sha256={self.response_sha256!r}, "
            f"input_tokens={self.input_tokens}, "
            f"completion_tokens={self.completion_tokens}, "
            f"total_tokens={self.total_tokens}, "
            f"cached_input_tokens={self.cached_input_tokens}, "
            f"response_model={self.response_model!r}"
            ")"
        )

    def __repr__(self) -> str:
        return str(self)


def _flatten_content_parts(messages: list[dict]) -> list[dict]:
    """Join bibr prompt parts into plain OpenAI-compatible message content."""
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if (
            isinstance(content, list)
            and content
            and all(isinstance(part, dict) and "cache" in part for part in content)
        ):
            msg = {**msg, "content": "".join(str(part.get("text", "")) for part in content)}
        out.append(msg)
    return out


def _project_native_prompt(messages: list[dict]) -> tuple[list[dict], str | None]:
    """Separate role-annotated document data from NuExtract task instructions."""
    projected: list[dict] = []
    instruction_parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if not (
            isinstance(content, list)
            and content
            and all(
                isinstance(part, dict)
                and part.get("nuextract_role") in {"document", "instructions"}
                for part in content
            )
        ):
            projected.extend(_flatten_content_parts([message]))
            continue

        document = "".join(
            str(part.get("text", "")) for part in content if part["nuextract_role"] == "document"
        )
        instruction_parts.extend(
            str(part.get("text", ""))
            for part in content
            if part["nuextract_role"] == "instructions"
        )
        if document:
            projected.append({**message, "content": document})

    instructions = ""
    for instruction_part in instruction_parts:
        if (
            instructions
            and instruction_part
            and not instructions[-1].isspace()
            and not instruction_part[0].isspace()
        ):
            instructions += "\n"
        instructions += instruction_part
    return projected, instructions or None


_LEADING_THINK_RE = re.compile(r"\A<think>.*?</think>", flags=re.DOTALL)
_OUTER_JSON_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n?```\Z",
    flags=re.DOTALL,
)
_INCOMPLETE_LITERAL_RE = re.compile(r"(?:t|tr|tru|f|fa|fal|fals|n|nu|nul)\Z")
_INCOMPLETE_NUMBER_RE = re.compile(
    r"(?:-|(?:-?(?:0|[1-9]\d*)\.)|"
    r"(?:-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?))\Z"
)


class _StrictJSONFailure(ValueError):
    """Internal no-payload sentinel for non-standard JSON."""


def _unwrap_native_wrappers(raw: str) -> tuple[str, bool]:
    cleaned = raw.strip()
    if cleaned.startswith("<think>"):
        thinking = _LEADING_THINK_RE.match(cleaned)
        if thinking is None:
            return cleaned, False
        cleaned = cleaned[thinking.end() :].strip()
        if cleaned.startswith("<think>"):
            return cleaned, False

    if cleaned.startswith("```"):
        fence = _OUTER_JSON_FENCE_RE.fullmatch(cleaned)
        if fence is None:
            return cleaned, False
        cleaned = fence.group("body").strip()
    return cleaned, True


def _reject_json_constant(_value: str) -> None:
    raise _StrictJSONFailure


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise _StrictJSONFailure
        parsed[key] = value
    return parsed


def _contains_non_finite_number(value: Any) -> bool:
    if type(value) is float:
        return not math.isfinite(value)
    if type(value) is list:
        return any(_contains_non_finite_number(item) for item in value)
    if type(value) is dict:
        return any(_contains_non_finite_number(item) for item in value.values())
    return False


def _json_structure_state(raw: str) -> tuple[bool, bool, bool]:
    """Return (open_container, in_string, malformed_closing)."""
    stack: list[str] = []
    in_string = False
    escaped = False
    malformed_closing = False
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if not stack or stack.pop() != expected:
                malformed_closing = True
                break
    return bool(stack), in_string, malformed_closing


def _looks_like_incomplete_json(raw: str, error: json.JSONDecodeError) -> bool:
    stripped = raw.strip()
    if _INCOMPLETE_LITERAL_RE.fullmatch(stripped) or _INCOMPLETE_NUMBER_RE.fullmatch(stripped):
        return True

    open_container, in_string, malformed_closing = _json_structure_state(raw)
    if malformed_closing:
        return False
    if error.msg.startswith("Unterminated string"):
        return in_string
    if not open_container:
        return False

    remainder = raw[error.pos :].strip()
    if error.msg == "Expecting value":
        if not remainder:
            return True
        return bool(
            _INCOMPLETE_LITERAL_RE.fullmatch(remainder)
            or _INCOMPLETE_NUMBER_RE.fullmatch(remainder)
        )
    if error.msg in {
        "Expecting ':' delimiter",
        "Expecting ',' delimiter",
        "Expecting property name enclosed in double quotes",
    }:
        if not remainder:
            return True
        tail = re.split(r"[:,\[]", raw)[-1].strip()
        return bool(_INCOMPLETE_NUMBER_RE.fullmatch(tail))
    return False


def _json_decode_category(
    raw: str,
    error: json.JSONDecodeError,
) -> NativeInvalidCategory:
    if _looks_like_incomplete_json(raw, error):
        return "truncated"
    if error.msg == "Extra data":
        return "trailing_content"
    return "non_json"


def _invalid_output(
    category: NativeInvalidCategory,
    *,
    raw: str,
    finish_reason: str | None,
    contract: NativeSchemaContract,
    model: str,
    input_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_input_tokens: int,
) -> NuExtractInvalidOutput:
    return NuExtractInvalidOutput(
        category=category,
        model=model,
        finish_reason=finish_reason,
        response_chars=len(raw),
        response_sha256=hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest(),
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        response_model=contract.response_model.__name__,
        salvage_raw=raw if category == "truncated" else "",
    )


def parse_native_completion(
    raw: str,
    *,
    finish_reason: str | None,
    contract: NativeSchemaContract,
    model: str,
    input_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> BaseModel:
    """Parse and validate one entire NuExtract native completion."""

    def diagnostic(category: NativeInvalidCategory) -> NuExtractInvalidOutput:
        return _invalid_output(
            category,
            raw=raw,
            finish_reason=finish_reason,
            contract=contract,
            model=model,
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    if finish_reason == "length":
        raise diagnostic("truncated")

    cleaned, wrappers_valid = _unwrap_native_wrappers(raw)
    if not wrappers_valid:
        raise diagnostic("non_json")
    if not cleaned:
        raise diagnostic("empty")

    parsed: Any = None
    invalid_category: NativeInvalidCategory | None = None
    try:
        parsed = json.loads(
            cleaned,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except _StrictJSONFailure:
        invalid_category = "non_json"
    except json.JSONDecodeError as error:
        invalid_category = _json_decode_category(cleaned, error)
    except (ValueError, RecursionError):
        invalid_category = "non_json"
    if invalid_category is not None:
        raise diagnostic(invalid_category)

    non_finite = False
    numeric_scan_failed = False
    try:
        non_finite = _contains_non_finite_number(parsed)
    except (ValueError, RecursionError):
        numeric_scan_failed = True
    if numeric_scan_failed or non_finite:
        raise diagnostic("non_json")
    if type(parsed) is not dict:
        raise diagnostic("non_object")
    if not native_wire_value_is_valid(
        parsed,
        wire_schema=contract.wire_schema,
        template=contract.template,
    ):
        raise diagnostic("schema_invalid")

    result: BaseModel | None = None
    pydantic_failed = False
    try:
        result = contract.response_model.model_validate(parsed)
    except ValidationError:
        pydantic_failed = True
    if pydantic_failed:
        raise diagnostic("schema_invalid")
    if result is None:  # pragma: no cover - defensive invariant
        raise diagnostic("schema_invalid")
    return result


def template_for_model(response_model: type) -> dict[str, Any]:
    """Build a NuExtract template from a Pydantic response model."""
    return native_contract_for_model(response_model).template


def _max_tokens_kw(
    settings: GlobalSettings,
    max_tokens: int | None,
) -> dict[str, int]:
    if max_tokens is not None:
        return {"max_tokens": max_tokens}
    configured = settings.llm.max_tokens
    default = type(settings.llm).model_fields["max_tokens"].default
    if configured != default or "max_tokens" in settings.llm.model_fields_set:
        return {"max_tokens": configured}
    return {}


def build_native_request_kwargs(
    *,
    settings: GlobalSettings,
    response_model: type[BaseModel],
    system: str,
    messages: list[dict],
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build one NuExtract-native OpenAI request from the semantic prompt contract."""
    if any(message.get("role") == "assistant" for message in messages):
        raise ValueError("NuExtract native requests do not accept assistant input messages")

    projected_messages, instructions = _project_native_prompt(messages)
    contract = native_contract_for_model(response_model)
    chat_template_kwargs = {
        "template": json.dumps(contract.template, ensure_ascii=False, indent=2),
        "instructions": instructions or _GENERIC_INSTRUCTIONS,
        "enable_thinking": False,
    }
    request_kwargs: dict[str, Any] = {
        "model": settings.llm.model,
        "messages": [{"role": "system", "content": system}, *projected_messages],
        "temperature": settings.llm.temperature,
        "extra_body": {"chat_template_kwargs": chat_template_kwargs},
    }
    request_kwargs.update(_max_tokens_kw(settings, max_tokens))
    return request_kwargs


def _completion_member(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _token_count(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0


def _completion_usage(completion: Any) -> tuple[int, int, int, int]:
    usage = _completion_member(completion, "usage")
    if usage is None:
        usage = _completion_member(completion, "usage_metadata")
    if usage is None:
        return 0, 0, 0, 0

    input_tokens = _token_count(
        _completion_member(usage, "input_tokens")
        or _completion_member(usage, "prompt_tokens")
        or _completion_member(usage, "prompt_token_count")
    )
    completion_tokens = _token_count(
        _completion_member(usage, "output_tokens")
        or _completion_member(usage, "completion_tokens")
        or _completion_member(usage, "candidates_token_count")
    )
    total_tokens = _token_count(
        _completion_member(usage, "total_tokens") or _completion_member(usage, "total_token_count")
    )
    if total_tokens == 0:
        total_tokens = input_tokens + completion_tokens

    cached_input_tokens = _token_count(
        _completion_member(usage, "cached_content_token_count")
        or _completion_member(usage, "cache_read_input_tokens")
    )
    if cached_input_tokens == 0:
        details = _completion_member(usage, "prompt_tokens_details")
        if details is None:
            details = _completion_member(usage, "input_tokens_details")
        cached_input_tokens = _token_count(_completion_member(details, "cached_tokens"))

    return input_tokens, completion_tokens, total_tokens, cached_input_tokens


def normalize_native_completion(completion: Any) -> NativeCompletionEnvelope:
    """Normalize one provider completion without interpreting its JSON payload."""
    choices = _completion_member(completion, "choices")
    try:
        choice = choices[0]
    except (IndexError, KeyError, TypeError):
        choice = None

    message = _completion_member(choice, "message")
    content = _completion_member(message, "content")
    raw = content if isinstance(content, str) else ("" if content is None else str(content))
    finish_reason = _completion_member(choice, "finish_reason")
    if not isinstance(finish_reason, str):
        finish_reason = None

    usage = _completion_member(completion, "usage")
    if usage is None:
        usage = _completion_member(completion, "usage_metadata")
    input_tokens, completion_tokens, total_tokens, cached_input_tokens = _completion_usage(
        completion
    )
    return NativeCompletionEnvelope(
        raw=raw,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        has_usage=usage is not None,
    )


class NuExtractNativeBackend:
    """Structured backend using NuExtract 3's native template protocol."""

    _supports_dispatch_callback = True

    def __init__(self, settings=None):
        self._settings = settings if settings is not None else snapshot_settings()
        self._client = None
        self._client_sig: tuple | None = None

    def _settings_signature(self) -> tuple:
        return (
            self._settings.llm.base_url,
            self._settings.llm.api_key,
            self._settings.llm.model,
        )

    def _get_client(self):
        sig = self._settings_signature()
        if self._client is None or self._client_sig != sig:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {
                "api_key": self._settings.llm.api_key or "not-needed",
                "timeout": self._settings.llm.timeout_seconds * 2,
            }
            if self._settings.llm.base_url:
                kwargs["base_url"] = self._settings.llm.base_url
            self._client = AsyncOpenAI(**kwargs)
            self._client_sig = sig
        return self._client

    async def create(
        self,
        *,
        response_model: type,
        system: str,
        messages: list[dict],
        want_completion: bool,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        client_override: Any = None,
        on_dispatch: Callable[[], None] | None = None,
        on_protocol_hashes: Callable[[dict[str, str]], None] | None = None,
    ) -> tuple[Any, Any | None]:
        if client_override is not None:
            logger.debug("Ignoring Instructor client_override for NuExtract native backend")
        if reasoning_effort is not None:
            logger.debug(
                "Ignoring reasoning_effort=%s for NuExtract native backend", reasoning_effort
            )

        client = self._get_client()
        request_kwargs = build_native_request_kwargs(
            settings=self._settings,
            response_model=response_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        if on_protocol_hashes is not None:
            # Provenance capture is a best-effort side-channel: never let a hash
            # computation failure break the actual native call.
            try:
                chat_template_kwargs = request_kwargs["extra_body"]["chat_template_kwargs"]
                request_document = "".join(
                    str(m["content"])
                    for m in request_kwargs["messages"]
                    if m.get("role") != "system"
                )
                on_protocol_hashes(
                    {
                        "request_document_sha256": _sha256_text(request_document),
                        "instruction_sha256": _sha256_text(
                            str(chat_template_kwargs["instructions"])
                        ),
                        "converted_template_sha256": _sha256_text(
                            str(chat_template_kwargs["template"])
                        ),
                    }
                )
            except Exception:  # noqa: BLE001 — provenance must not affect extraction
                logger.debug("Failed to capture native protocol hashes", exc_info=True)
        if on_dispatch is not None:
            on_dispatch()
        completion = await client.chat.completions.create(**request_kwargs)
        envelope = normalize_native_completion(completion)
        contract = native_contract_for_model(response_model)
        result = parse_native_completion(
            envelope.raw,
            finish_reason=envelope.finish_reason,
            contract=contract,
            model=self._settings.llm.model,
            input_tokens=envelope.input_tokens,
            completion_tokens=envelope.completion_tokens,
            total_tokens=envelope.total_tokens,
            cached_input_tokens=envelope.cached_input_tokens,
        )
        return result, completion if want_completion else None
