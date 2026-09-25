import asyncio
import contextlib
import copy
import functools
import hashlib
import itertools
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal, cast

from bibr.clients.prompts import PROMPTS
from bibr.config import GlobalSettings, snapshot_settings
from bibr.exceptions import (
    LlmCallError,
    LlmInvalidOutputError,
    LlmRejectedError,
    LlmServiceError,
    LlmTimeoutError,
    LlmTruncatedError,
    LlmUnreachableError,
    ProcessingError,
    SafeLlmDiagnostics,
    UpstreamServiceError,
)
from bibr.schemas import (
    AuthorLLM,
    AuthorsLLM,
    CoreMetadataLLM,
    PaperClassificationLLM,
    PaperReferenceLLM,
    PaperTypeLabel,
    ResearchIntegrityLLM,
    TitleKeywordsLLM,
)
from bibr.utils.transient import is_transient_network_error

if TYPE_CHECKING:
    from bibr.clients.providers.base import LlmProvider
    from bibr.clients.structured import StructuredBackend
    from bibr.utils.rate_limiter import AsyncLocalRateLimiter, AsyncRedisRateLimiter

logger = logging.getLogger(__name__)

# Seconds allowed for the one-shot Redis reachability probe that decides
# between the shared and the local rate limiter. A limiter is a fallback, not a
# dependency: if Redis cannot answer this fast the local one is correct enough
# and vastly better than making every request wait on it.
_REDIS_PROBE_TIMEOUT = 1.0

# Cloud default: number of times Instructor attempts a response when its output
# fails schema validation (1 initial + 2 re-asks). Deterministic local servers
# default to one attempt; see :func:`_validation_attempts`.
_LLM_VALIDATION_ATTEMPTS = 3

# Cap on segmentation windows per call — bounds cost/fan-out for pathologically
# long (or abusive) reference blocks. With the smaller ref_seg_window_chars budget
# (~16K/window) a 40-window cap covers ~2000 references — any realistic
# bibliography; beyond it the tail is dropped with a warning.
_SEG_MAX_WINDOWS = 40


def _window_ref_text(text: str, budget: int) -> list[str]:
    """Split a references block into windows of at most ``budget`` chars.

    Splits on line boundaries so a reference's opening line is never cut across
    windows (each opening anchor stays locatable). A single line longer than the
    budget (e.g. a whole reference list extracted as one blob) is hard-split into
    budget-sized chunks. Returns ``[text]`` unchanged when it already fits — so
    normal papers keep the single-call behaviour. Windowing replaces a single
    truncating call, which silently dropped every reference past the cap on
    small-context local models (e.g. the 24K-context Gemma).
    """
    if len(text) <= budget:
        return [text]
    lines: list[str] = []
    for ln in text.splitlines(keepends=True):
        if len(ln) <= budget:
            lines.append(ln)
        else:
            lines.extend(ln[i : i + budget] for i in range(0, len(ln), budget))
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for ln in lines:
        if cur and cur_len + len(ln) > budget:
            windows.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln)
    if cur:
        windows.append("".join(cur))
    return windows


_REF_INDEX_MARKER_RE = re.compile(r"^(\d+)\.\s", re.MULTILINE)


def _expected_ref_indices(
    numbered_text: str, start_index: int, expected_count: int | None = None
) -> set[int]:
    """Indices of the numbered entries in a batched reference-parse prompt.

    The caller numbers entries contiguously from ``start_index``; the longest
    contiguous run of line-start ``N.`` markers recovers that set while
    ignoring spurious numbers inside reference bodies. ``expected_count`` (the
    known batch length) caps the run so an interior line that happens to begin
    with the next sequential ``N.`` inside a wrapped reference body cannot
    inflate the set with a phantom index.
    """
    markers = {int(m.group(1)) for m in _REF_INDEX_MARKER_RE.finditer(numbered_text)}
    expected: set[int] = set()
    idx = start_index
    limit = start_index + expected_count if expected_count is not None else None
    while idx in markers and (limit is None or idx < limit):
        expected.add(idx)
        idx += 1
    return expected


def _validation_attempts(settings: "GlobalSettings") -> int:
    """Resolve schema-validation attempts for the active endpoint.

    Local OpenAI-compatible servers run extraction at temperature zero. A
    validation re-ask is therefore expensive and commonly reproduces the same
    small-model failure. Cloud providers keep Instructor's self-correction
    loop, and ``LLM_VALIDATION_ATTEMPTS`` can explicitly override either path.
    """
    configured = settings.llm.validation_attempts
    if configured is not None:
        return configured
    is_custom_openai = settings.llm.provider.lower() == "openai" and bool(settings.llm.base_url)
    return 1 if is_custom_openai else _LLM_VALIDATION_ATTEMPTS


def _task_max_tokens(settings: GlobalSettings, task_limit: int | None) -> int | None:
    """Return a positive task cap bounded by LLM_MAX_TOKENS; 0/None disables it."""
    if task_limit is None or task_limit <= 0:
        return None
    return min(int(task_limit), int(settings.llm.max_tokens))


def _validation_retry(
    settings: "GlobalSettings",
    *,
    attempts: int | None = None,
) -> "Any":
    """Instructor retry policy: re-ask the model on schema/validation failures.

    A pydantic ``ValidationError`` means the model returned structurally wrong
    output; feeding the error back and re-asking (Instructor's headline feature)
    usually fixes it. Provider/transport errors (429, 5xx, timeouts) are
    deliberately NOT retried here — they propagate to ``_invoke_structured``'s
    own breaker-aware retry loop, so rate limits aren't double-counted.
    """
    from pydantic import ValidationError
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

    return AsyncRetrying(
        stop=stop_after_attempt(
            _validation_attempts(settings) if attempts is None else max(1, int(attempts))
        ),
        retry=retry_if_exception_type(ValidationError),
        reraise=True,
    )


def _get_provider(
    settings: "GlobalSettings | None" = None,
) -> "tuple[LlmProvider, str]":
    """Return the provider instance for the currently configured LLM, plus its name."""
    from bibr.clients import providers

    effective = settings if settings is not None else snapshot_settings()
    name = effective.llm.provider.lower()
    return providers.get(name, settings=effective), name


def _is_blank_completion_error(exc: BaseException) -> bool:
    """True when a failure is the model returning an empty/blank completion.

    Small local models (vllm-mlx, where ``json_schema`` is advisory) sometimes
    emit an immediate EOS, so Instructor receives ``''`` and — after exhausting
    its validation re-asks — raises a pydantic ``json_invalid`` ``ValidationError``
    with a blank ``input``. That is model flakiness, not a genuine schema/transport
    fault, so best-effort callers log it softly instead of as an error. A
    non-blank-but-malformed completion (real fidelity fault) is deliberately
    excluded. Walks the ``__cause__``/``__context__`` chain in case Instructor
    wraps the parse failure.
    """
    from pydantic import ValidationError

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ValidationError):
            for err in cur.errors():
                if err.get("type") == "json_invalid":
                    val = err.get("input")
                    if isinstance(val, str) and not val.strip():
                        return True
        cur = cur.__cause__ or cur.__context__
    return False


def _extract_cached_tokens(usage) -> int:
    """Cached-input-token count from a provider usage object, 0 if unreported.

    Shapes: google-genai ``cached_content_token_count``, Anthropic
    ``cache_read_input_tokens``, OpenAI ``prompt_tokens_details.cached_tokens``.
    """
    cached = getattr(usage, "cached_content_token_count", None) or getattr(
        usage, "cache_read_input_tokens", None
    )
    if cached:
        return cached
    details = getattr(usage, "prompt_tokens_details", None)
    return getattr(details, "cached_tokens", None) or 0 if details is not None else 0


def _create_client(mode_override=None, settings: "GlobalSettings | None" = None):
    """Construct an async Instructor client for the configured provider.

    Dispatch is delegated to ``bibr.clients.providers`` — each provider
    adapter knows how to read its own credentials and build the instructor
    client. Supported providers: google, openai, anthropic, groq, ollama.

    ``mode_override`` (an ``instructor.Mode``) forces a specific instructor
    mode for the openai provider only — used to build the free-JSON re-roll
    client in :meth:`LLMClient._get_json_mode_client`. Other providers don't
    expose a mode knob, so the override is ignored for them.
    """
    provider, _ = _get_provider(settings)
    if mode_override is not None and getattr(provider, "name", None) == "openai":
        return provider.build_client(mode_override=mode_override)
    return provider.build_client()


def preflight_credentials(settings: "GlobalSettings | None" = None) -> None:
    """Fail fast when the configured provider can't build a client.

    Raises the provider's ``ValueError`` (e.g. missing API key) without any
    network call, so the CLI can abort before OCR work starts.
    """
    _create_client(settings=settings)


def _extract_http_status(exc: BaseException) -> int | None:
    """Best-effort extraction of HTTP status from various provider exception shapes.

    Handles httpx/OpenAI (``.response.status_code`` / ``.status_code``) and the
    google-genai SDK. On genai's async path the client uses **aiohttp**, so
    ``ServerError.response`` is an ``aiohttp.ClientResponse`` whose code lives in
    ``.status`` (not ``.status_code``), and the int status is also on the
    exception as ``.code`` (``.status`` there is the string label, e.g.
    "UNAVAILABLE"). Without these fallbacks a Gemini 503 reads as ``None`` and is
    misclassified as non-transient, so it never retries.
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        s = getattr(resp, "status_code", None)
        if not isinstance(s, int):
            s = getattr(resp, "status", None)  # aiohttp.ClientResponse
        if isinstance(s, int):
            return s
    s = getattr(exc, "status_code", None)
    if isinstance(s, int):
        return s
    # genai APIError stores the int status in ``.code``; guard the range so a
    # non-HTTP ``.code`` on some other exception can't masquerade as a status.
    code = getattr(exc, "code", None)
    if isinstance(code, int) and 100 <= code <= 599:
        return code
    return None


def _find_last_completion(exc: BaseException) -> Any:
    """Walk an exception chain for a truncation error's raw completion.

    Instructor's ``IncompleteOutputException`` (raised when the model hits its
    output-token cap mid-object) carries the partial provider completion on
    ``last_completion``; :meth:`LLMClient.extract_references` wraps it in an
    ``UpstreamServiceError`` (``original_error``), so unwrap the same
    ``original_error``/``__cause__`` layers ``_is_degenerate_ref_failure``
    does. Returns the completion object, or ``None`` when none is present.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        lc = getattr(cur, "last_completion", None)
        if lc is not None:
            return lc
        nxt = getattr(cur, "original_error", None)
        cur = nxt if isinstance(nxt, BaseException) else cur.__cause__
    return None


# How far down ``original_error`` / ``__cause__`` / ``__context__`` the failure
# classifier looks. Instructor and the SDKs wrap an error two or three deep.
_FAILURE_CHAIN_DEPTH = 8

_TIMEOUT_EXC_NAMES = frozenset(
    {"TimeoutError", "TimeoutException", "APITimeoutError", "DeadlineExceeded"}
)
# A connection refused, dropped or never accepted, or the breaker open: the
# service is down, whatever the input. The same names as the batch resume's
# service-outage rule, so that rule can count ``LlmUnreachableError`` too.
_UNREACHABLE_EXC_NAMES = frozenset(
    {
        "ConnectionError",
        "NetworkError",
        "RemoteProtocolError",
        "ProxyError",
        "APIConnectionError",
        "ClientConnectionError",
        "CircuitOpenError",
    }
)
_INVALID_OUTPUT_EXC_NAMES = frozenset(
    {"ValidationError", "JSONDecodeError", "ResponseParsingError", "AsyncValidationError"}
)
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


def _failure_chain(exc: BaseException) -> list[BaseException]:
    """*exc* and the errors it wraps, outermost first, without repeats."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending and len(chain) < _FAILURE_CHAIN_DEPTH:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        pending.extend(
            nested
            for nested in (
                getattr(current, "original_error", None),
                current.__cause__,
                None if current.__suppress_context__ else current.__context__,
            )
            if isinstance(nested, BaseException)
        )
    return chain


def http_status_in_chain(exc: BaseException) -> int | None:
    """The first HTTP status carried by *exc* or an error it wraps.

    Instructor wraps a provider SDK's error in its own exception, which has
    no status; the status is on the error it wraps.
    """
    return next(
        (status for error in _failure_chain(exc) if (status := _extract_http_status(error))),
        None,
    )


def _exc_names(exc: BaseException) -> set[str]:
    return {klass.__name__ for klass in type(exc).__mro__}


def _finish_reason(completion: Any) -> str | None:
    """The provider's stop reason for *completion*, lowercased, if it has one."""
    choices = getattr(completion, "choices", None)
    reason = getattr(choices[0], "finish_reason", None) if choices else None
    if reason is None:
        reason = getattr(completion, "stop_reason", None)  # Anthropic
    if reason is None:
        candidates = getattr(completion, "candidates", None)  # google-genai
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if reason is None:
        return None
    return str(getattr(reason, "name", reason)).lower()


def _classify_llm_failure(exc: BaseException) -> type[LlmCallError]:
    """Pick the :class:`~bibr.exceptions.LlmCallError` class for a failed call.

    The order encodes precedence: a truncated response stays a truncation even
    when Instructor wrapped it, a status anywhere in the chain beats the
    wrapper's missing one, and only a failure with no service signal counts as
    invalid output.
    """
    if isinstance(exc, LlmCallError):
        return type(exc)
    chain = _failure_chain(exc)
    for error in chain:
        if "IncompleteOutputException" in _exc_names(error):
            return LlmTruncatedError
        completion = getattr(error, "last_completion", None)
        if completion is not None and _finish_reason(completion) in _TRUNCATED_FINISH_REASONS:
            return LlmTruncatedError
    # The host never accepted the connection: down, not slow.
    if any("ConnectTimeout" in _exc_names(error) for error in chain):
        return LlmUnreachableError
    if any(_exc_names(error) & _TIMEOUT_EXC_NAMES for error in chain):
        return LlmTimeoutError
    status = http_status_in_chain(exc)
    if status == 408:
        return LlmTimeoutError
    if status is not None and (status == 429 or status >= 500):
        return LlmServiceError
    if status is not None and 400 <= status < 500:
        return LlmRejectedError
    if any(_exc_names(error) & _UNREACHABLE_EXC_NAMES for error in chain):
        return LlmUnreachableError
    if any(is_transient_network_error(error) for error in chain):
        return LlmServiceError
    if any(_exc_names(error) & _INVALID_OUTPUT_EXC_NAMES for error in chain):
        return LlmInvalidOutputError
    return LlmCallError


def _bounded(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _failure_cause(error_class: type[LlmCallError], exc: BaseException) -> str:
    """A bounded description of why the call failed, without model output.

    Validation errors quote the rejected values, so for invalid output only
    the error locations and types are kept.
    """
    if isinstance(exc, LlmCallError) and exc.cause:
        return exc.cause
    chain = _failure_chain(exc)
    if error_class is LlmTruncatedError:
        return "the response stopped at the output-token limit"
    if error_class is LlmInvalidOutputError:
        from pydantic import ValidationError

        for error in chain:
            if isinstance(error, ValidationError):
                locations = ", ".join(
                    f"{'.'.join(str(part) for part in item['loc']) or '<root>'} [{item['type']}]"
                    for item in error.errors()[:3]
                )
                return _bounded(f"{error.error_count()} validation error(s): {locations}")
            if isinstance(error, json.JSONDecodeError):
                return _bounded(f"JSONDecodeError: {error.msg} at position {error.pos}")
        return f"invalid structured output ({type(chain[-1]).__name__})"
    root = chain[-1]
    for error in chain:
        if _extract_http_status(error) is not None:
            root = error
            break
    detail = str(root)
    status = _extract_http_status(root)
    prefix = f"HTTP {status}: " if status is not None and str(status) not in detail else ""
    return _bounded(f"{prefix}{type(root).__name__}: {detail}" if detail else type(root).__name__)


def llm_call_error(message: str, exc: BaseException) -> LlmCallError:
    """Wrap a failed LLM task call in the :class:`LlmCallError` that fits it.

    ``message`` names the task ("Failed to extract authors"); the cause is
    appended from the exception chain, and ``exc`` stays ``original_error``.
    """
    error_class = _classify_llm_failure(exc)
    return error_class(message, exc, cause=_failure_cause(error_class, exc))


def _recover_finished_response(exc: BaseException, response_model: type) -> Any | None:
    """Recover a finished response that Instructor rejected, or return ``None``.

    Only an invalid-output failure qualifies, never a truncation, a decoder
    abort or a service failure: the completion must be whole. The recovered
    value goes through ``response_model``'s validation like any response.
    """
    if _classify_llm_failure(exc) is not LlmInvalidOutputError:
        return None
    completion = _find_last_completion(exc)
    if completion is None or _finish_reason(completion) in {"abort", *_TRUNCATED_FINISH_REASONS}:
        return None
    from bibr.clients.structured_json import StructuredResponseError, recover_structured_object

    try:
        recovered = recover_structured_object(_completion_text(completion), response_model)
    except StructuredResponseError as invalid:
        logger.debug("Local %s recovery declined: %s", response_model.__name__, invalid.category)
        return None
    except Exception:  # noqa: BLE001 - a validator bug must not replace the typed error
        logger.debug("Local %s recovery failed", response_model.__name__, exc_info=True)
        return None
    logger.warning(
        "Recovered the %s response locally after it failed validation "
        "(%d invalid backslash escape(s) repaired)",
        response_model.__name__,
        recovered.repaired_backslashes,
    )
    return recovered.value


def llm_failure_code(exc: BaseException) -> str:
    """The stable error code for a failed LLM call, however it was raised.

    Callers that degrade on any exception use it to label their warning: a
    typed error keeps its code, a typed processing failure its own, and a raw
    exception (``invoke_structured`` does not wrap) is classified here.
    """
    if isinstance(exc, LlmCallError):
        return exc.error_code
    if isinstance(exc, ProcessingError) and exc.error_code:
        return exc.error_code
    return _classify_llm_failure(exc).error_code


def _completion_text(completion: Any) -> str:
    """Best-effort raw text from a provider completion across shapes.

    Handles OpenAI-style ``choices[0].message`` (content, or a tool call's
    ``function.arguments`` when the JSON rode a function call), google-genai
    (``.text``), and Anthropic ``Message`` content blocks (text or a
    ``tool_use`` input). Returns ``""`` when nothing is extractable so callers
    degrade to their existing failure path rather than raising.
    """
    choices = getattr(completion, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None) or choices[0]
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            return content
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            args = getattr(getattr(tool_calls[0], "function", None), "arguments", None)
            if isinstance(args, str) and args:
                return args
    text = getattr(completion, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(completion, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                inp = getattr(block, "input", None)
                if inp is not None:
                    import json

                    parts.append(json.dumps(inp))
        joined = "".join(parts)
        if joined:
            return joined
    if isinstance(completion, str):
        return completion
    return ""


def incomplete_output_text(exc: BaseException) -> str:
    """Raw completion text carried by a truncation error, ``""`` if unavailable.

    Bridges an ``IncompleteOutputException`` (possibly wrapped) to the salvage
    utility (:func:`bibr.utils.json_salvage.salvage_array_objects`): the batched
    reference parser recovers the complete leading objects from a completion cut
    off mid-array instead of losing the whole batch.
    """
    completion = _find_last_completion(exc)
    if completion is not None:
        return _completion_text(completion)
    # The NuExtract native backend raises its own diagnostic rather than an
    # Instructor exception, so there is no ``last_completion`` to read. It
    # carries the truncated text on ``salvage_raw`` for exactly this purpose.
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        raw = getattr(cur, "salvage_raw", None)
        if isinstance(raw, str) and raw:
            return raw
        nxt = getattr(cur, "original_error", None)
        cur = nxt if isinstance(nxt, BaseException) else cur.__cause__
    return ""


# Degeneration guard for author salvage. NuExtract3-FP8 can fall into a
# repetition loop that emits the same handful of affiliation fragments as author
# objects until the token cap, so the salvage recovers hundreds of near-identical
# "authors". A real truncated byline is a modest run of DISTINCT names, so reject
# a salvage that is either too long or dominated by duplicates and let the
# empty-author re-roll take over.
_MAX_SALVAGED_AUTHORS = 64
_MIN_SALVAGED_DISTINCT_RATIO = 0.5


def _salvage_truncated_authors(exc: BaseException) -> list[AuthorLLM]:
    """Recover the complete leading authors from a truncated author completion.

    A tight-context server (e.g. NuExtract3 at ``max_model_len`` 8192) can cut
    the ``"authors": [...]`` array mid-object, raising instructor's
    ``IncompleteOutputException``. The leading objects are usually complete and
    valid, so validate them against :class:`AuthorLLM` and keep the maximal
    leading run, stopping at the first object that fails validation. Returns
    ``[]`` (no salvage) when the completion is unavailable or nothing parses —
    e.g. a non-truncation error — so the caller falls back to raising unchanged.

    Also returns ``[]`` when the salvage looks like a model degeneration (a
    repetition loop): more than ``_MAX_SALVAGED_AUTHORS`` objects, or a distinct
    ``(given, family)`` ratio below ``_MIN_SALVAGED_DISTINCT_RATIO``. A genuine
    long byline is distinct names and survives.
    """
    from pydantic import ValidationError

    from bibr.utils.json_salvage import salvage_array_objects

    raw = incomplete_output_text(exc)
    if not raw:
        return []
    salvaged: list[AuthorLLM] = []
    for obj in salvage_array_objects(raw, "authors"):
        if not isinstance(obj, dict):
            break
        try:
            salvaged.append(AuthorLLM.model_validate(obj))
        except ValidationError:
            break

    if salvaged:
        distinct = len({((a.given or "").strip(), (a.family or "").strip()) for a in salvaged})
        ratio = distinct / len(salvaged)
        if len(salvaged) > _MAX_SALVAGED_AUTHORS or ratio < _MIN_SALVAGED_DISTINCT_RATIO:
            logger.warning(
                "Author salvage looks degenerate (%d objects, %d distinct); "
                "discarding and deferring to the empty-author re-roll",
                len(salvaged),
                distinct,
            )
            return []
    return salvaged


def _is_provider_429(exc: BaseException) -> bool:
    """Detect rate-limit (429) errors across httpx, OpenAI, and Google shapes."""
    if _extract_http_status(exc) == 429:
        return True
    type_name = type(exc).__name__
    if type_name in ("ResourceExhausted", "RateLimitError"):
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _is_provider_429(cause)
    return False


def _extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort parse of ``Retry-After`` / ``retry_delay`` from a provider error.

    Looks at:
    * httpx response headers (``retry-after``)
    * google.api_core ``ResourceExhausted`` (``retry_delay`` proto in details)
    * Wrapped ``__cause__`` / ``__context__`` chains
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        ra = headers.get("retry-after") if headers is not None and hasattr(headers, "get") else None
        if ra is not None:
            try:
                return float(ra)
            except (TypeError, ValueError):
                pass
    details_attr = getattr(exc, "details", None)
    details = details_attr() if callable(details_attr) else details_attr
    if details:
        try:
            for d in details:
                ra = getattr(d, "retry_delay", None)
                if ra is not None:
                    seconds = getattr(ra, "seconds", 0) or 0
                    nanos = getattr(ra, "nanos", 0) or 0
                    if seconds or nanos:
                        return float(seconds) + nanos / 1e9
        except TypeError:
            pass
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _extract_retry_after_seconds(cause)
    return None


# Credential-shaped tokens for the providers this codebase actually talks to:
# OpenAI (sk-...), Google (AIza...), the bibr-resolver (sv_...), Groq (gsk_...).
# Deliberately not a general secret-scanner — it does not attempt to catch every
# credential shape in existence (e.g. AWS AKIA... keys), only the ones that can
# plausibly appear in a bibr prompt/completion via LLM_API_KEY or similar.
_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{8,}|sv_[A-Za-z0-9_-]{8,}|"
    r"gsk_[A-Za-z0-9_-]{8,})\b"
)


def scrub_trace_text(text: str) -> str:
    """Redact credential-shaped tokens before a prompt/completion is persisted.

    A stored trace is precisely the artifact where a stray credential would
    survive; a 2026-07-23 security audit of this repo found ~35 sites where a
    credential can surface via a repr. Called on every message and completion
    before a ``LlmTraceExport`` row is assembled — never on raw text that
    might reach a log or export unscrubbed.
    """
    return _SECRET_RE.sub("[REDACTED]", text)


def _sanitize_trace_value(value: Any) -> Any:
    """Recursively coerce *value* to JSON-safe, scrubbed primitives for a trace row.

    ``LlmTraceExport.params`` is an untyped ``dict`` (pydantic doesn't validate
    its values), so it accepts anything at construction time. Every current
    provider adapter's ``call_kwargs`` is already scalars or nested dicts of
    scalars (Google nests ``temperature`` under ``generation_config``;
    Anthropic nests ``thinking``) — this is a safety net, not a workaround for
    something happening today. Without it, a future adapter threading an Enum
    or an SDK sentinel (e.g. instructor's ``NotGiven``) into ``call_kwargs``
    would construct a valid row and only fail later at
    ``model_dump(mode="json")`` — failing the *entire file's* export, far from
    where the bad value was introduced.

    Strings are scrubbed here too: ``params`` gets no exemption from the same
    credential-shaped-token check applied to ``messages``/``raw_completion``,
    even though no current adapter puts a credential into ``call_kwargs``
    (those live only in ``build_client()``).
    """
    if isinstance(value, str):
        return scrub_trace_text(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_trace_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_trace_value(v) for v in value]
    # Unknown/non-primitive type (Enum, SDK sentinel, ...): coerce via repr
    # rather than let it ride unsanitized into a "JSON-safe" row.
    return scrub_trace_text(repr(value))


def _trace_message_text(content: Any) -> str:
    """Render a message's ``content`` as plain text for a trace row.

    Handles the post-``transform_messages`` shape used by providers with a
    transform hook (currently only Anthropic): a list of content blocks
    (``{"type": "text", "text": ..., "cache_control": {...}}``) left behind by
    cache-marker parts (:func:`bibr.clients.prompts.part`) that
    ``_flatten_content_parts`` does not join, because the ``"cache"`` key it
    looks for has already been replaced with ``"cache_control"`` by the
    transform. Without this, an Anthropic trace row's message captured Python
    repr noise (``"[{'type': 'text', 'text': ..., 'cache_control': ...}]"``)
    instead of the actual prompt text — not a security issue (scrubbing still
    ran on the repr), but it defeats the point: these rows exist to be
    training data, not a debug dump of internal message shapes.
    """
    if isinstance(content, list):
        parts = [str(p["text"]) for p in content if isinstance(p, dict) and "text" in p]
        if parts:
            return "".join(parts)
    return str(content or "")


def _build_call_kwargs(
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    settings: "GlobalSettings | None" = None,
) -> dict:
    """Build provider-specific kwargs for a ``client.create()`` call."""
    provider, provider_name = _get_provider(settings)
    kwargs = provider.call_kwargs(reasoning_effort, max_tokens)

    if reasoning_effort is not None and provider_name != "openai":
        # reasoning_effort is an OpenAI-only parameter; log when it's supplied
        # for another provider so misconfigurations don't go unnoticed.
        logger.debug("reasoning_effort=%s ignored for provider=%s", reasoning_effort, provider_name)

    return kwargs


# Attribution key for per-file usage. asyncio tasks copy the context at
# creation, so a value set inside one file's post-parse task tree is visible to
# all its nested coroutines/tasks and invisible to sibling files — making
# per-file attribution race-free on a shared LLMClient. Callers should use a
# UNIQUE key per invocation (not the bare content hash) and pop the bucket via
# ``usage_pop_file`` when done, so reprocessing the same file never
# accumulates across runs and concurrent duplicate files don't share a bucket.
_usage_file_hash: ContextVar[str | None] = ContextVar("_usage_file_hash", default=None)

# Call-site attribution for per-file usage. Set by ``track_llm_usage`` (to
# ``func.__name__``) and by ``invoke_structured(label=...)`` for callers
# outside this class. ContextVars are asyncio-task-local, so the concurrent
# ``asyncio.gather`` fan-out in ``extract_core_metadata`` stays correctly
# attributed even though all call sites share one ``LLMClient``.
_usage_label: ContextVar[str | None] = ContextVar("_usage_label", default=None)

_usage_context_counter = itertools.count(1)


def new_usage_context_key(file_hash: str | None) -> str | None:
    """Unique per-invocation attribution key (``None`` when no file hash)."""
    if not file_hash:
        return None
    return f"{file_hash}#{next(_usage_context_counter)}"


@contextmanager
def usage_file_context(key: str | None):
    """Attribute LLM token usage recorded inside this context to *key*."""
    token = _usage_file_hash.set(key)
    try:
        yield
    finally:
        _usage_file_hash.reset(token)


def track_llm_usage(func):
    """Decorator that logs per-call token usage by snapshotting the handler before/after."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        return await self._run_labeled_call(
            func.__name__,
            lambda: func(self, *args, **kwargs),
        )

    return wrapper


def _flatten_content_parts(messages: list[dict]) -> list[dict]:
    """Join builder content parts (see :func:`bibr.clients.prompts.part`)
    into one string for providers whose transform did not consume them.
    Provider-native block lists (no ``"cache"`` key) pass through untouched.
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if (
            isinstance(content, list)
            and content
            and all(isinstance(p, dict) and "cache" in p for p in content)
        ):
            msg = {**msg, "content": "".join(p["text"] for p in content)}
        out.append(msg)
    return out


def _sha256_text(s: str) -> str:
    """Lowercase hex sha256 of the utf-8 bytes of *s*."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class InstructorBackend:
    """Default :class:`~bibr.clients.structured.StructuredBackend`: one
    structured call through the configured Instructor chat client.

    Holds the owning ``LLMClient`` (late-bound) for client construction; all
    provider dispatch (message transform, call kwargs, validation re-ask) is
    module-level machinery in this file.
    """

    _supports_dispatch_callback = True

    def __init__(
        self,
        owner: "LLMClient",
        *,
        suppress_reasoning: bool = False,
        validation_attempts: int | None = None,
    ):
        self._owner = owner
        self._suppress_reasoning = suppress_reasoning
        self._validation_attempts = validation_attempts

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
        client = client_override if client_override is not None else self._owner._get_client()
        provider, _ = _get_provider(self._owner._settings)
        full_messages = [{"role": "system", "content": system}] + messages
        if hasattr(provider, "transform_messages"):
            full_messages = provider.transform_messages(full_messages)
        full_messages = _flatten_content_parts(full_messages)
        if on_protocol_hashes is not None:
            # Provenance capture is a best-effort side-channel: never let a hash
            # computation failure break the actual structured-output call.
            try:
                request_document = "".join(
                    str(m["content"]) for m in full_messages if m.get("role") != "system"
                )
                on_protocol_hashes(
                    {
                        "request_document_sha256": _sha256_text(request_document),
                        "instruction_sha256": _sha256_text(str(full_messages[0]["content"])),
                        "converted_template_sha256": _sha256_text(
                            json.dumps(
                                response_model.model_json_schema(),
                                sort_keys=True,
                                ensure_ascii=False,
                            )
                        ),
                    }
                )
            except Exception:  # noqa: BLE001 — provenance must not affect extraction
                logger.debug("Failed to capture instructor protocol hashes", exc_info=True)
        call_kwargs = _build_call_kwargs(
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            settings=self._owner._settings,
        )
        if self._suppress_reasoning:
            # ``reasoning_effort=None`` normally inherits the configured
            # OpenAI effort. Native recovery must explicitly omit the field:
            # otherwise the fallback is not a protocol-only change.
            call_kwargs.pop("reasoning_effort", None)
        # Instructor re-asks the model on schema-validation failures
        # (validation_retry); transient 429/5xx are excluded here — they
        # propagate to LLMClient's breaker-aware retry loop.
        validation_retry = _validation_retry(
            self._owner._settings,
            attempts=self._validation_attempts,
        )
        if want_completion:
            if on_dispatch is not None:
                on_dispatch()
            result, completion = await client.create_with_completion(
                response_model=response_model,
                messages=full_messages,
                max_retries=validation_retry,
                **call_kwargs,
            )
            # Known limitation (see LlmTraceExport docstring): max_retries
            # above makes instructor re-ask internally on schema-validation
            # failure, so a rejected completion never reaches this line —
            # only the eventually-accepted call is ever captured here, hence
            # parsed_ok is unconditionally True and attempt is always 1.
            self._owner._record_trace(
                messages=full_messages,
                completion=completion,
                params=call_kwargs,
                parsed_ok=True,
            )
            return result, completion
        if on_dispatch is not None:
            on_dispatch()
        result = await client.create(
            response_model=response_model,
            messages=full_messages,
            max_retries=validation_retry,
            **call_kwargs,
        )
        return result, None


def _structured_backend_kind(settings) -> str:
    raw = getattr(settings.llm, "structured_backend", "auto").strip().lower()
    if raw in {"instructor", "nuextract-native"}:
        return raw
    if raw and raw != "auto":
        logger.warning("Unknown LLM_STRUCTURED_BACKEND=%r; falling back to Instructor", raw)
    return "instructor"


class LLMClient:
    def __init__(
        self,
        settings: "GlobalSettings | None" = None,
        backend: "StructuredBackend | None" = None,
    ):
        self._settings = settings if settings is not None else snapshot_settings()
        self._backend_explicit = backend is not None
        self._backend_kind: str | None = None
        self._backend: StructuredBackend = backend if backend is not None else self._make_backend()
        self._client = None  # Lazy-init instructor client
        self._client_sig: tuple | None = None
        # Second, free-JSON-mode client for the empty-author / null-classification
        # re-roll (see _get_json_mode_client). Lazy-init, rebuilt on settings change.
        self._json_client = None
        self._json_client_sig: tuple | None = None
        self._markdown_json_client = None
        self._markdown_json_client_sig: tuple | None = None
        self._limiter: AsyncRedisRateLimiter | AsyncLocalRateLimiter | None = None
        # Guards lazy-init across concurrent first callers; otherwise both
        # coroutines pass the ``self._limiter is None`` check, both probe
        # Redis, and one of the two limiters is orphaned. Bound to the loop it
        # was created in so a client reused across loops rebuilds it.
        self._limiter_init_lock: asyncio.Lock | None = None
        self._limiter_init_loop: asyncio.AbstractEventLoop | None = None
        self._concurrency_sem: asyncio.Semaphore | None = None
        self._llm_cache = None  # Lazy-init structured-response cache (CACHE_LLM)
        self._track_usage = self._settings.llm.track_usage
        self._usage: dict[str, dict[str, int]] = {}
        self._usage_by_file: dict[str, dict[str, dict[str, int]]] = {}
        # Keyed by (label, provider, model): a label used under two engines
        # within the same file must not blend one engine's stamp with the
        # other's token counts (see _label_bucket).
        self._labels_by_file: dict[str, dict[tuple[str, str, str], dict[str, int]]] = {}
        self._protocol_hashes_by_file: dict[str, dict[str, dict[str, str]]] = {}
        # Opt-in trace rows (LLM_CAPTURE_TRACE), keyed like the usage buckets
        # above. Always empty when capture is off — see _record_trace.
        self._traces_by_file: dict[str, list[dict]] = {}

        from bibr.utils.circuit_breaker import AsyncCircuitBreaker

        self._breaker = AsyncCircuitBreaker(
            failure_threshold=self._settings.cb.failure_threshold,
            reset_timeout=self._settings.cb.reset_timeout_seconds,
            name="llm",
        )

    def _make_backend(self) -> "StructuredBackend":
        kind = _structured_backend_kind(self._settings)
        self._backend_kind = kind
        if kind == "nuextract-native":
            from bibr.clients.nuextract import NuExtractNativeBackend

            logger.info(
                "LLM structured backend: NuExtract native template (model=%s)",
                self._settings.llm.model,
            )
            if self._settings.llm.capture_trace:
                # Loud, not silent: NuExtractNativeBackend has no _owner and
                # never calls _record_trace (see LlmTraceExport docstring).
                # Without this, an operator on the primary local/self-hosted
                # path believes they're collecting LoRA training data and
                # gets an empty extraction.trace — indistinguishable from
                # "off" or "nothing to capture" anywhere else in the export.
                logger.warning(
                    "LLM_CAPTURE_TRACE is on but the resolved structured backend is "
                    "nuextract-native (model=%s), which is not yet instrumented for "
                    "trace capture — extraction.trace will stay empty for calls routed "
                    "through this backend.",
                    self._settings.llm.model,
                )
            return NuExtractNativeBackend(settings=self._settings)
        return InstructorBackend(self)

    def _make_recovery_instructor_backend(self) -> "StructuredBackend":
        """Build the one-attempt, no-reasoning Instructor recovery transport."""
        return InstructorBackend(
            self,
            suppress_reasoning=True,
            validation_attempts=1,
        )

    def _supports_instructor_json_fallback(self) -> bool:
        """Pure preflight for the qualified native-to-JSON recovery route."""
        provider = self._settings.llm.provider
        base_url = self._settings.llm.base_url
        return provider.strip().lower() == "openai" and bool((base_url or "").strip())

    def _backend_protocol(
        self,
        backend: "StructuredBackend",
    ) -> Literal["nuextract-native", "instructor"]:
        """Return the protocol captured with *backend* for this logical call."""
        if self._backend_kind == "nuextract-native":
            return "nuextract-native"
        # Preserve correct behavior for callers that inject the concrete
        # native backend directly instead of selecting it through settings.
        if type(backend).__name__ == "NuExtractNativeBackend":
            return "nuextract-native"
        return "instructor"

    def _ensure_backend_current(self) -> None:
        if self._backend_explicit:
            return
        kind = _structured_backend_kind(self._settings)
        if kind != self._backend_kind:
            self._backend = self._make_backend()

    @staticmethod
    def _cap_input(text: str, settings: "GlobalSettings | None" = None) -> str:
        """Truncate input text to LLM_MAX_INPUT_CHARS to prevent token-cost abuse."""
        effective = settings if settings is not None else snapshot_settings()
        limit = effective.llm.max_input_chars
        if len(text) > limit:
            logger.warning("LLM input truncated from %d to %d chars", len(text), limit)
            return text[:limit]
        return text

    def _settings_signature(self) -> tuple:
        return (
            self._settings.llm.provider,
            self._settings.llm.base_url,
            self._settings.llm.api_key,
            self._settings.llm.model,
        )

    def _get_client(self):
        """Return the (cached) instructor async client. Rebuilds when relevant
        settings change (e.g. when a local LLM server reconfigures the
        provider mid-process)."""
        sig = self._settings_signature()
        if self._client is None:
            self._client = _create_client(settings=self._settings)
            self._client_sig = sig
        elif self._client_sig is None:
            # Client was set externally (e.g. in tests); adopt current sig as baseline.
            self._client_sig = sig
        elif self._client_sig != sig:
            self._client = _create_client(settings=self._settings)
            self._client_sig = sig
        return self._client

    def json_mode_reroll_is_distinct(self) -> bool:
        """Would a ``json_mode=True`` re-roll actually differ from the first call?

        ``_create_client`` honours ``mode_override`` for the openai provider
        only; every other adapter ignores it and hands back a client built
        exactly like the primary one. Since extraction runs at temperature 0,
        the "recovery" re-roll then re-sends a byte-identical request and gets
        the same answer — a wasted paid call on every affected paper, on the
        default (google) provider among others. Callers check this before
        spending it.
        """
        provider, _ = _get_provider(self._settings)
        return getattr(provider, "name", None) == "openai"

    def _get_json_mode_client(self):
        """Return a (cached) instructor client forced to free-JSON mode.

        Used for the empty-author / null-classification re-roll: a paper whose
        byline IS in the input occasionally comes back with ``authors=[]`` (and
        a null classification) under strict guided decoding, where ``[]`` is the
        grammar's minimal valid production. Re-rolling through this client drops
        that escape. For providers without a mode knob it is just an independent
        client (still a fresh re-roll). Rebuilt when relevant settings change."""
        import instructor

        sig = self._settings_signature()
        if self._json_client is None or self._json_client_sig != sig:
            self._json_client = _create_client(
                mode_override=instructor.Mode.JSON,
                settings=self._settings,
            )
            self._json_client_sig = sig
        return self._json_client

    def _get_markdown_json_client(self):
        """Validate JSON in Python without requesting a server-side grammar."""
        import instructor

        sig = self._settings_signature()
        if self._markdown_json_client is None or self._markdown_json_client_sig != sig:
            self._markdown_json_client = _create_client(
                mode_override=instructor.Mode.MD_JSON, settings=self._settings
            )
            self._markdown_json_client_sig = sig
        return self._markdown_json_client

    def _can_recover_decoder_abort(self, error: Exception, client_override: Any) -> bool:
        """Only recover explicit aborts from a custom endpoint's JSON grammar."""
        import instructor

        if not self._supports_instructor_json_fallback():
            return False
        completion = _find_last_completion(error)
        choices = getattr(completion, "choices", None)
        if not choices or getattr(choices[0], "finish_reason", None) != "abort":
            return False
        client = client_override if client_override is not None else self._get_client()
        return getattr(client, "mode", None) in {instructor.Mode.JSON_SCHEMA, instructor.Mode.JSON}

    # Number of slots in the limiter's sliding window. Sized to the
    # ``extract_core_metadata`` fan-out (title/keywords + authors +
    # classification) so the three concurrent calls don't queue serially
    # against a single-slot limiter. Average rate is preserved by scaling
    # the window proportionally.
    _BURST = 3

    @property
    def limiter(self):
        """The rate limiter, or ``None`` before the first request builds it.

        Built by :meth:`_ensure_limiter`, not here: the Redis probe must run on
        the event loop rather than block it. Mirrors ``CrossrefClient.limiter``.
        """
        return self._limiter

    async def _ensure_limiter(self) -> None:
        """Create the rate limiter on first request (Redis probe runs off-loop).

        The probe used to be a *synchronous* ``redis.Redis.ping()`` inside the
        ``limiter`` property, reached from a coroutine. ``bibr serve`` runs one
        LitServe worker with ``enable_async=True``, so a single event loop
        serves every concurrent request: a Redis that accepts the connection
        but never answers froze the whole worker, not just the caller — every
        in-flight paper stalled together (measured at 5.1 s of total freeze
        against a wedged-but-reachable Redis). ``socket_connect_timeout``
        bounds only the connect, not the command round-trip, so the read is
        bounded explicitly here as well.
        """
        if self._limiter is not None:
            return
        loop = asyncio.get_running_loop()
        if self._limiter_init_lock is None or self._limiter_init_loop is not loop:
            self._limiter_init_lock = asyncio.Lock()
            self._limiter_init_loop = loop
        async with self._limiter_init_lock:
            if self._limiter is not None:
                return
            interval = 60.0 / self._settings.llm.rate_limit_rpm
            window = interval * self._BURST
            try:
                if not self._settings.redis.url:
                    raise RuntimeError("Redis URL not configured")
                from redis.asyncio import Redis as AsyncRedis

                from bibr.utils.rate_limiter import AsyncRedisRateLimiter

                r = AsyncRedis.from_url(
                    self._settings.redis.url,
                    socket_connect_timeout=_REDIS_PROBE_TIMEOUT,
                    socket_timeout=_REDIS_PROBE_TIMEOUT,
                )
                try:
                    await asyncio.wait_for(r.ping(), timeout=_REDIS_PROBE_TIMEOUT)
                finally:
                    await r.aclose()

                self._limiter = AsyncRedisRateLimiter(
                    redis_url=self._settings.redis.url,
                    resource_id="llm",
                    max_requests=self._BURST,
                    window_seconds=window,
                )
            except Exception as exc:
                from bibr.utils.rate_limiter import AsyncLocalRateLimiter

                logger.info("Redis unavailable, using local rate limiter: %s", exc)
                self._limiter = AsyncLocalRateLimiter(
                    resource_id="llm",
                    max_requests=self._BURST,
                    window_seconds=window,
                )

    def _concurrency_gate(self) -> "asyncio.Semaphore | contextlib.nullcontext":
        """Semaphore bounding in-flight LLM requests (``LLM_MAX_CONCURRENCY``).

        Local single-device servers (vllm-mlx simple engine) crash under
        concurrent requests and thrash unified memory with parallel KV
        caches — ``LLM_MAX_CONCURRENCY=1`` serializes them. 0 (default)
        means unlimited, preserving cloud-provider behavior.
        """
        limit = getattr(self._settings.llm, "max_concurrency", 0) or 0
        if limit <= 0:
            return contextlib.nullcontext()
        if self._concurrency_sem is None:
            self._concurrency_sem = asyncio.Semaphore(limit)
        return self._concurrency_sem

    @contextlib.asynccontextmanager
    async def _measured_concurrency_gate(self):
        """Enter the concurrency gate and record its wait even if entry fails."""
        started = time.perf_counter()
        entered = False
        try:
            async with self._concurrency_gate():
                self._record_label_metric(
                    "concurrency_wait_ms",
                    round((time.perf_counter() - started) * 1000),
                )
                entered = True
                yield
        finally:
            if not entered:
                self._record_label_metric(
                    "concurrency_wait_ms",
                    round((time.perf_counter() - started) * 1000),
                )

    @property
    def usage(self) -> dict:
        """Returns aggregate token usage metadata across all calls, keyed by model name."""
        return self._usage

    def usage_for_file(self, file_hash: str) -> dict[str, dict[str, int]]:
        """Token usage by model for calls made inside ``usage_file_context(file_hash)``."""
        return copy.deepcopy(self._usage_by_file.get(file_hash, {}))

    def usage_pop_file(self, key: str) -> dict[str, dict[str, int]]:
        """Like :meth:`usage_for_file`, but removes the bucket — callers that
        own the attribution key evict it so the per-file map stays bounded on
        long-lived (serve) clients."""
        return self._usage_by_file.pop(key, {})

    def usage_labels_pop_file(self, key: str) -> dict[tuple[str, str, str], dict[str, int]]:
        """Usage for calls attributed to *key*, keyed by ``(label, provider, model)``;
        removes the bucket."""
        return self._labels_by_file.pop(key, {})

    def protocol_hashes_pop_file(self, key: str) -> dict[str, dict[str, str]]:
        """Per-label protocol hashes for calls attributed to *key*; removes the bucket."""
        return self._protocol_hashes_by_file.pop(key, {})

    def traces_pop_file(self, key: str) -> list[dict]:
        """Opt-in trace rows for calls attributed to *key*; removes the bucket.

        Always ``[]`` when ``LLM_CAPTURE_TRACE`` is off — mirrors
        ``usage_labels_pop_file``'s pop-and-evict contract so a shared
        (serve/local) client's per-file map stays bounded."""
        return self._traces_by_file.pop(key, [])

    @property
    def resolved_structured_backend(self) -> str:
        """The structured backend actually in effect (``instructor`` | ``nuextract-native``)."""
        return self._backend_kind or _structured_backend_kind(self._settings)

    def _label_bucket(self) -> dict[str, int] | None:
        if not self._track_usage:
            return None
        file_key = _usage_file_hash.get()
        if not file_key:
            return None
        label = _usage_label.get() or "unlabeled"
        # Keyed by the full (label, provider, model) triple — not just label —
        # so a label used under two engines within one file gets two buckets
        # instead of one bucket whose provider/model stamp goes stale mid-file.
        key = (label, self._settings.llm.provider, self._settings.llm.model)
        return self._labels_by_file.setdefault(file_key, {}).setdefault(
            key,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "calls": 0,
            },
        )

    def _record_label_metric(self, name: str, value: int) -> None:
        bucket = self._label_bucket()
        if bucket is not None:
            bucket[name] = bucket.get(name, 0) + max(0, int(value))

    def _record_protocol_hashes(self, hashes: dict[str, str]) -> None:
        """Capture one task's request-byte hashes, first-writer-wins per label.

        First-writer-wins so a native task's fallback to Instructor does not
        overwrite the hashes of the native request that was actually sent.
        """
        if not self._track_usage:
            return
        file_key = _usage_file_hash.get()
        if not file_key:
            return
        label = _usage_label.get() or "unlabeled"
        self._protocol_hashes_by_file.setdefault(file_key, {}).setdefault(label, hashes)

    def _record_trace(
        self,
        *,
        messages: list[dict],
        completion: Any,
        params: dict,
        parsed_ok: bool,
        attempt: int = 1,
        error: str | None = None,
    ) -> None:
        """Append one opt-in trace row (``LLM_CAPTURE_TRACE``) for this call.

        Mirrors ``_record_protocol_hashes``'s file/label attribution via the
        same contextvars. Best-effort: a capture bug must never break the
        actual structured-output call, so failures here are swallowed after a
        debug log — never let trace capture take extraction down with it.
        """
        if not self._settings.llm.capture_trace:
            return
        file_key = _usage_file_hash.get()
        if not file_key:
            return
        try:
            raw: str | None = None
            finish_reason: str | None = None
            choices = getattr(completion, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None)
                raw = str(content) if content else None
                finish_reason = getattr(choices[0], "finish_reason", None)
            self._traces_by_file.setdefault(file_key, []).append(
                {
                    "label": _usage_label.get() or "unlabeled",
                    "provider": self._settings.llm.provider,
                    "model": self._settings.llm.model,
                    "messages": [
                        {**m, "content": scrub_trace_text(_trace_message_text(m.get("content")))}
                        for m in messages
                    ],
                    "raw_completion": scrub_trace_text(raw) if raw else None,
                    "parsed_ok": parsed_ok,
                    "finish_reason": finish_reason,
                    # Resolved sampling params as _build_call_kwargs produced
                    # them for this call — shape is provider-dependent (e.g.
                    # Google nests temperature under generation_config), but
                    # always the actually-resolved values, never filtered down
                    # to a "safe" subset that could drop them silently.
                    # _sanitize_trace_value makes the (untyped) dict JSON-safe
                    # and scrubs any embedded credential-shaped strings.
                    "params": _sanitize_trace_value(dict(params)),
                    "attempt": attempt,
                    "error": error,
                }
            )
        except Exception:  # noqa: BLE001 — trace capture must not affect extraction
            logger.debug("Failed to capture LLM trace row", exc_info=True)

    async def _run_labeled_call(self, label: str, call):
        if not self._track_usage:
            return await call()
        before = copy.deepcopy(self._usage)
        label_token = _usage_label.set(label)
        started = time.perf_counter()
        self._record_label_metric("logical_calls", 1)
        for metric in (
            "attempts",
            "retries",
            "failed_calls",
            "total_ms",
            "rate_limit_wait_ms",
            "concurrency_wait_ms",
            "provider_ms",
            "native_attempts",
            "instructor_attempts",
            "protocol_fallbacks",
            "protocol_fallbacks_recovered",
            "native_invalid_outputs",
            "native_invalid_empty",
            "native_invalid_non_json",
            "native_invalid_truncated",
            "native_invalid_trailing_content",
            "native_invalid_non_object",
            "native_invalid_schema_invalid",
        ):
            self._record_label_metric(metric, 0)
        try:
            return await call()
        except BaseException:
            self._record_label_metric("failed_calls", 1)
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            self._record_label_metric("total_ms", elapsed_ms)
            _usage_label.reset(label_token)
            self._log_usage_delta(label, before)

    async def _acquire_rate_limit(self) -> None:
        started = time.perf_counter()
        try:
            await self._ensure_limiter()
            assert self._limiter is not None  # noqa: S101 — _ensure_limiter sets it
            await self._limiter.acquire()
        finally:
            self._record_label_metric(
                "rate_limit_wait_ms",
                round((time.perf_counter() - started) * 1000),
            )

    def _record_usage_counts(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cached_input_tokens: int,
    ) -> None:
        """Record already-normalized numeric usage for one completion."""
        model_name = self._settings.llm.model
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        total_tokens = max(0, int(total_tokens))
        cached_tokens = max(0, int(cached_input_tokens))

        buckets = [self._usage]
        file_hash = _usage_file_hash.get()
        if file_hash:
            buckets.append(self._usage_by_file.setdefault(file_hash, {}))

        for bucket in buckets:
            counts = bucket.setdefault(
                model_name,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_input_tokens": 0,
                },
            )
            counts["input_tokens"] += input_tokens
            counts["output_tokens"] += output_tokens
            counts["total_tokens"] += total_tokens
            counts["cached_input_tokens"] += cached_tokens

        if file_hash:
            lbucket = self._label_bucket()
            assert lbucket is not None  # noqa: S101 — file hash + tracking guarantee a bucket
            lbucket["input_tokens"] += input_tokens
            lbucket["output_tokens"] += output_tokens
            lbucket["total_tokens"] += total_tokens
            lbucket["cached_input_tokens"] += cached_tokens
            lbucket["calls"] += 1

    def _record_usage(self, completion) -> None:
        """Record token usage from an Instructor completion object.

        Understands three raw provider shapes:
        - OpenAI-style ``completion.usage`` (prompt_tokens/completion_tokens,
          prefix-cache hits in ``prompt_tokens_details.cached_tokens``)
        - Anthropic-style ``completion.usage`` (input_tokens/output_tokens,
          prompt-cache hits in ``cache_read_input_tokens``)
        - google-genai ``completion.usage_metadata`` (prompt_token_count/
          candidates_token_count, implicit-cache hits in
          ``cached_content_token_count``)
        """
        usage = getattr(completion, "usage", None) or getattr(completion, "usage_metadata", None)
        if usage is None:
            return

        input_tokens = (
            getattr(usage, "input_tokens", 0)
            or getattr(usage, "prompt_tokens", 0)
            or getattr(usage, "prompt_token_count", 0)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", 0)
            or getattr(usage, "completion_tokens", 0)
            or getattr(usage, "candidates_token_count", 0)
            or 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0)
            or getattr(usage, "total_token_count", 0)
            or (input_tokens + output_tokens)
        )
        self._record_usage_counts(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=_extract_cached_tokens(usage),
        )

    def _record_native_invalid(self, error: "BaseException") -> None:
        """Account safe usage carried by one rejected native completion."""
        category = str(getattr(error, "category", ""))
        self._record_label_metric("native_invalid_outputs", 1)
        if category in {
            "empty",
            "non_json",
            "truncated",
            "trailing_content",
            "non_object",
            "schema_invalid",
        }:
            self._record_label_metric(f"native_invalid_{category}", 1)
        if not self._track_usage:
            return
        self._record_usage_counts(
            input_tokens=getattr(error, "input_tokens", 0),
            output_tokens=getattr(error, "completion_tokens", 0),
            total_tokens=getattr(error, "total_tokens", 0),
            cached_input_tokens=getattr(error, "cached_input_tokens", 0),
        )

    def _log_usage_delta(self, label: str, before: dict[str, dict[str, int]]) -> None:
        """Log the per-model token-usage delta between *before* and ``self._usage``."""
        after = self._usage

        # Compute delta per model
        for model_name, after_counts in after.items():
            before_counts = before.get(model_name, {})
            delta_in = after_counts.get("input_tokens", 0) - before_counts.get("input_tokens", 0)
            delta_out = after_counts.get("output_tokens", 0) - before_counts.get("output_tokens", 0)
            delta_total = after_counts.get("total_tokens", 0) - before_counts.get("total_tokens", 0)
            if delta_total > 0:
                logger.info(
                    f"Token usage [{label}] model={model_name}: "
                    f"input={delta_in}, output={delta_out}, total={delta_total}"
                )

    _RETRY_MAX_ATTEMPTS = 3
    _RETRY_BASE_DELAY = 1.0  # seconds

    async def _invoke_protocol_with_retries(
        self,
        backend: "StructuredBackend",
        *,
        protocol: Literal["nuextract-native", "instructor"],
        response_model: Any,
        messages: list[dict],
        system_prompt: str,
        reasoning_effort: str | None = None,
        client_override: Any = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Invoke one captured protocol/backend with transient retries only.

        Args:
            response_model: Pydantic model class for structured output.
            messages: User messages (system prompt is prepended automatically).
            system_prompt: System prompt string.
            reasoning_effort: Per-call reasoning effort override (OpenAI only).
            client_override: Use this instructor client instead of the default.
            max_tokens: Per-call output-token cap override (``None`` keeps the
                configured ``LLM_MAX_TOKENS``).

        Returns:
            A validated Pydantic model instance.
        """
        import random

        per_request = self._settings.llm.timeout_seconds
        # Multiplier sized so 3 attempts + backoff fit within the pipeline
        # timeout (Settings.pipeline.timeout, default 300 s).
        hard_timeout = per_request * 2

        last_exc: BaseException | None = None
        physical_attempts = 0
        for attempt in range(self._RETRY_MAX_ATTEMPTS):
            is_last = attempt == self._RETRY_MAX_ATTEMPTS - 1
            if attempt > 0:
                # The caller acquired one slot for this logical call; every
                # retry is a further physical request and must take its own.
                # Without this a retry storm spends budget it never acquired —
                # exactly when the provider is already rate-limiting us, and
                # exactly when a shared limiter is meant to hold the fleet back.
                await self._acquire_rate_limit()

            async def invoke_backend():
                nonlocal physical_attempts
                provider_started: float | None = None
                dispatched = False

                def record_dispatch() -> None:
                    nonlocal dispatched, physical_attempts, provider_started
                    if dispatched:
                        return
                    if physical_attempts > 0:
                        self._record_label_metric("retries", 1)
                    self._record_label_metric("attempts", 1)
                    self._record_label_metric(
                        (
                            "native_attempts"
                            if protocol == "nuextract-native"
                            else "instructor_attempts"
                        ),
                        1,
                    )
                    physical_attempts += 1
                    dispatched = True
                    provider_started = time.perf_counter()

                create_kwargs = {
                    "response_model": response_model,
                    "system": system_prompt,
                    "messages": messages,
                    "want_completion": self._track_usage,
                    "reasoning_effort": (
                        None if protocol == "nuextract-native" else reasoning_effort
                    ),
                    "max_tokens": max_tokens,
                    "client_override": client_override,
                }
                if getattr(backend, "_supports_dispatch_callback", False) is True:
                    create_kwargs["on_dispatch"] = record_dispatch
                    create_kwargs["on_protocol_hashes"] = self._record_protocol_hashes
                else:
                    # Injected/testing backends retain their existing contract:
                    # entering ``create`` is the physical call boundary.
                    record_dispatch()
                try:
                    return await backend.create(**create_kwargs)
                finally:
                    if provider_started is not None:
                        self._record_label_metric(
                            "provider_ms",
                            round((time.perf_counter() - provider_started) * 1000),
                        )

            try:
                async with self._measured_concurrency_gate():
                    async with self._breaker:
                        result, completion = await asyncio.wait_for(
                            invoke_backend(),
                            timeout=hard_timeout,
                        )
                    if completion is not None:
                        self._record_usage(completion)
                    return result
            except TimeoutError:
                last_exc = TimeoutError(
                    f"LLM call timed out after {hard_timeout}s "
                    f"(per-request timeout: {per_request}s)"
                )
                if is_last:
                    raise last_exc from None
            except UpstreamServiceError as exc:
                last_exc = exc
                if is_last:
                    raise
            except Exception as exc:
                # Detect 429 across httpx (status_code=429), OpenAI
                # (RateLimitError), and Google (ResourceExhausted) shapes —
                # the SDK exception classes don't all expose .response.
                is_429 = _is_provider_429(exc)
                status = _extract_http_status(exc)
                # Connection resets, read timeouts and APIConnectionError carry
                # no HTTP status at all, so a status-only test re-raised them on
                # the first attempt — worst under the default google provider,
                # whose SDK sets stop_after_attempt(1), leaving exactly one
                # physical attempt for a blip.
                transient = (
                    is_429
                    or (status is not None and status >= 500)
                    or is_transient_network_error(exc)
                )
                if not transient:
                    # A truncation error (IncompleteOutputException) still burned
                    # tokens — its partial completion carries the usage. Record it
                    # before re-raising so llm_usage isn't understated on the
                    # degenerate call. Best-effort: never let it mask the error.
                    if self._track_usage:
                        try:
                            partial = _find_last_completion(exc)
                            if partial is not None:
                                self._record_usage(partial)
                        except Exception:  # noqa: BLE001 — usage is best-effort
                            logger.debug("Failed to record usage from partial completion")
                    raise
                last_exc = exc
                if is_last:
                    raise

            backoff = self._RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)  # noqa: S311
            retry_after = _extract_retry_after_seconds(last_exc) if last_exc else None
            if retry_after is not None:
                # Honor the provider's hint, but keep small jitter to avoid
                # synchronized stampede when many callers hit the same 429.
                delay = max(backoff, retry_after + random.uniform(0, 0.5))  # noqa: S311
            else:
                delay = backoff
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                self._RETRY_MAX_ATTEMPTS,
                last_exc,
                delay,
            )
            await asyncio.sleep(delay)

        assert last_exc is not None  # noqa: S101 — loop sets it on every retry path
        raise last_exc

    async def _invoke_with_protocol_fallback(
        self,
        *,
        backend: "StructuredBackend",
        protocol: Literal["nuextract-native", "instructor"],
        response_model: type,
        messages: list[dict],
        system_prompt: str,
        reasoning_effort: str | None,
        client_override: Any = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Run a captured route, allowing at most native -> Instructor once."""
        from bibr.clients.nuextract import NuExtractInvalidOutput

        try:
            return await self._invoke_protocol_with_retries(
                backend,
                protocol=protocol,
                response_model=response_model,
                messages=messages,
                system_prompt=system_prompt,
                reasoning_effort=(None if protocol == "nuextract-native" else reasoning_effort),
                client_override=client_override,
                max_tokens=max_tokens,
            )
        except NuExtractInvalidOutput as native_error:
            if protocol != "nuextract-native":
                raise
            self._record_native_invalid(native_error)
            if not self._supports_instructor_json_fallback():
                raise
            self._record_label_metric("protocol_fallbacks", 1)
            try:
                recovery_backend = self._make_recovery_instructor_backend()
                recovery_client = self._get_json_mode_client()
                # A further physical request: it takes its own limiter slot,
                # as the decoder-abort recovery below does.
                await self._acquire_rate_limit()
                recovered = await self._invoke_protocol_with_retries(
                    recovery_backend,
                    protocol="instructor",
                    response_model=response_model,
                    messages=messages,
                    system_prompt=system_prompt,
                    reasoning_effort=None,
                    client_override=recovery_client,
                    max_tokens=max_tokens,
                )
                self._record_label_metric("protocol_fallbacks_recovered", 1)
                return recovered
            except Exception:
                # Leave the nested handler before raising so Python does not
                # retain the potentially raw-bearing fallback error as
                # ``__context__`` on the safe native diagnostic.
                logger.warning(
                    "NuExtract native recovery failed; preserving the original safe diagnostic"
                )
            native_error.__traceback__ = None
            native_error.__cause__ = None
            native_error.__context__ = None
            native_error.__suppress_context__ = True
            raise native_error from None
        except Exception as error:
            if protocol != "instructor" or not self._can_recover_decoder_abort(
                error, client_override
            ):
                raise
            # JSON mode also uses a grammar on SGLang. MD_JSON supplies the
            # schema in the prompt and still validates the complete response,
            # avoiding the same broken decoder on the recovery request.
            logger.warning("Local JSON decoder aborted; retrying with validated unconstrained JSON")
            self._record_label_metric("decoder_abort_fallbacks", 1)
            await self._acquire_rate_limit()
            recovered = await self._invoke_protocol_with_retries(
                self._make_recovery_instructor_backend(),
                protocol="instructor",
                response_model=response_model,
                messages=messages,
                system_prompt=system_prompt,
                reasoning_effort=None,
                client_override=self._get_markdown_json_client(),
                max_tokens=max_tokens,
            )
            self._record_label_metric("decoder_abort_fallbacks_recovered", 1)
            return recovered

    @property
    def _response_cache(self):
        """The opt-in structured-response cache, or ``None`` when disabled."""
        if not self._settings.cache.llm:
            return None
        if self._llm_cache is None:
            from bibr.clients.llm_cache import LlmResponseCache

            self._llm_cache = LlmResponseCache(settings=self._settings)
        return self._llm_cache

    def _cache_lookup_key(
        self,
        response_model: Any,
        messages: list[dict],
        system_prompt: str,
        *,
        protocol: str,
        reasoning_effort: str | None,
        client_override: Any,
        max_tokens: int | None,
    ) -> tuple[Any, str | None]:
        """Resolve ``(cache, key)`` for one request, or ``(None, None)``."""
        cache = self._response_cache
        if cache is None:
            return None, None
        from bibr.clients.llm_cache import request_key

        try:
            flat = _flatten_content_parts(messages)
            user_text = "".join(m["content"] for m in flat if isinstance(m.get("content"), str))
            key = request_key(
                model=self._settings.llm.model,
                schema_name=getattr(response_model, "__name__", str(response_model)),
                system=system_prompt,
                user_text=user_text,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                # A caller-supplied client is the JSON-mode re-roll, which
                # resends identical text expecting a different answer.
                mode=f"{protocol}:{'override' if client_override is not None else 'default'}",
                schema_json=json.dumps(response_model.model_json_schema(), sort_keys=True),
                chat_template_json=json.dumps(
                    self._settings.llm.chat_template_kwargs, sort_keys=True
                ),
            )
        except Exception:  # noqa: BLE001 — caching is best-effort, never fatal
            logger.debug("LLM cache key construction failed; proceeding uncached", exc_info=True)
            return None, None
        return cache, key

    def _cached_result(self, cache, key: str, response_model: Any) -> Any:
        """Return a validated cached response, or ``None`` to call live."""
        body = cache.get(key)
        if body is None:
            return None
        try:
            result = response_model.model_validate(body)
        except Exception:  # noqa: BLE001 — a stale entry is a miss, not an error
            logger.debug("Discarding unusable LLM cache entry %s", key)
            return None
        self._record_label_metric("cache_hits", 1)
        return result

    def _cache_store(self, cache, key: str, result: Any, response_model: Any) -> None:
        try:
            body = result.model_dump(mode="json")
            # Blank/omitted model output is sanitized to None, but is not an
            # explicit refusal. Preserve that distinction when validating a hit.
            if (
                getattr(result, "_abstract_explicitly_absent", None) is False
                and body.get("abstract") is None
            ):
                body.pop("abstract", None)
            cache.put(
                key,
                body,
                model=self._settings.llm.model,
                schema_name=getattr(response_model, "__name__", str(response_model)),
                label=_usage_label.get(),
            )
        except Exception:  # noqa: BLE001 — never let caching break an answer
            logger.debug("LLM cache store failed for %s", key, exc_info=True)

    async def _invoke_structured(
        self,
        response_model: Any,
        messages: list[dict],
        system_prompt: str,
        reasoning_effort: str | None = None,
        client_override: Any = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Snapshot one structured route, then run bounded protocol recovery."""
        self._ensure_backend_current()
        backend = self._backend
        protocol = self._backend_protocol(backend)

        # A caller-provided Instructor client (the explicit json_mode re-roll)
        # deliberately bypasses native rather than being ignored by it.
        if protocol == "nuextract-native" and client_override is not None:
            backend = self._make_recovery_instructor_backend()
            protocol = "instructor"
            reasoning_effort = None

        from bibr.clients.nuextract import NuExtractInvalidOutput
        from bibr.models import ErrorCode

        cache, cache_key = self._cache_lookup_key(
            response_model,
            messages,
            system_prompt,
            protocol=protocol,
            reasoning_effort=reasoning_effort,
            client_override=client_override,
            max_tokens=max_tokens,
        )
        if cache is not None and cache_key is not None:
            hit = self._cached_result(cache, cache_key, response_model)
            if hit is not None:
                return hit

        # Below the cache check on purpose: a hit spends no provider quota, so
        # it must not wait on a budget that exists to protect that quota — an
        # otherwise fully-cached corpus re-run would still be paced at
        # LLM_RATE_LIMIT_RPM. Callers used to acquire before building their
        # prompt; acquiring here instead keeps it one slot per dispatched
        # request while making that request the thing the slot is spent on.
        await self._acquire_rate_limit()

        try:
            result = await self._invoke_with_protocol_fallback(
                backend=backend,
                protocol=protocol,
                response_model=response_model,
                messages=messages,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
                client_override=client_override,
                max_tokens=max_tokens,
            )
            if cache is not None and cache_key is not None:
                self._cache_store(cache, cache_key, result, response_model)
            return result
        except NuExtractInvalidOutput as native_error:
            diagnostics = SafeLlmDiagnostics.from_native_error(native_error)
            # Parser/provider frames can retain the raw completion in locals.
            # The safe diagnostic remains as the explicit cause, but its
            # traceback and any prior chain must not cross the public boundary.
            native_error.__traceback__ = None
            native_error.__cause__ = None
            native_error.__context__ = None
            native_error.__suppress_context__ = True
            raise ProcessingError(
                "LLM returned invalid structured output",
                error_code=ErrorCode.LLM_INVALID_OUTPUT.value,
                safe_diagnostics=diagnostics,
            ) from native_error

    async def invoke_structured(
        self,
        response_model: Any,
        messages: list[dict],
        system_prompt: str,
        *,
        label: str,
        reasoning_effort: str | None = None,
        client_override: Any = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Public, usage-attributed variant of _invoke_structured for callers
        outside this class (section classifier, implicit sections).
        """

        async def invoke():
            return await self._invoke_structured(
                response_model,
                messages,
                system_prompt,
                reasoning_effort=reasoning_effort,
                client_override=client_override,
                max_tokens=max_tokens,
            )

        return await self._run_labeled_call(label, invoke)

    @track_llm_usage
    async def extract_title_keywords(
        self, text: str, file_hash: str = "unknown", *, boundary: str | None = None
    ) -> TitleKeywordsLLM:
        logger.debug(
            f"Starting LLM title/keywords extraction (hash={file_hash}, input length: {len(text)})"
        )
        try:
            spec = PROMPTS["title_keywords"]
            boundary = boundary or uuid.uuid4().hex
            capped_text = self._cap_input(text, self._settings)
            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": spec.build_user(boundary=boundary, text=capped_text)}],
                spec.system,
                max_tokens=_task_max_tokens(self._settings, self._settings.llm.title_max_tokens),
            )

            logger.info(f"Successfully extracted title/keywords (hash={file_hash})")
            return cast("TitleKeywordsLLM", result)
        except ProcessingError:
            raise
        except Exception as e:
            # The anchor call alone gets local recovery: a finished response
            # with, say, LaTeX backslashes in the abstract otherwise loses the
            # whole record's title/keywords fields.
            recovered = _recover_finished_response(e, TitleKeywordsLLM)
            if recovered is not None:
                return cast("TitleKeywordsLLM", recovered)
            logger.error(
                f"LLM title/keywords extraction failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to extract title/keywords", e) from e

    @track_llm_usage
    async def extract_authors(
        self,
        text: str,
        file_hash: str = "unknown",
        *,
        json_mode: bool = False,
        boundary: str | None = None,
    ) -> AuthorsLLM:
        logger.debug(
            f"Starting LLM author extraction (hash={file_hash}, input length: {len(text)}, "
            f"json_mode={json_mode})"
        )
        try:
            spec = PROMPTS["authors"]
            boundary = boundary or uuid.uuid4().hex
            capped_text = self._cap_input(text, self._settings)
            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": spec.build_user(boundary=boundary, text=capped_text)}],
                spec.system,
                reasoning_effort=self._settings.llm.reasoning_effort_authors,
                client_override=self._get_json_mode_client() if json_mode else None,
                max_tokens=_task_max_tokens(self._settings, self._settings.llm.authors_max_tokens),
            )

            logger.info(f"Successfully extracted authors (hash={file_hash})")
            return cast("AuthorsLLM", result)
        except ProcessingError:
            raise
        except Exception as e:
            # A truncated author list (IncompleteOutputException from a
            # tight-context server) usually carries complete leading authors in
            # its raw completion — recover them instead of losing the byline.
            salvaged = _salvage_truncated_authors(e)
            if salvaged:
                code = llm_failure_code(e)
                logger.warning(
                    "Author extraction failed (%s); salvaged %d leading author(s) "
                    "from the partial completion (hash=%s)",
                    code,
                    len(salvaged),
                    file_hash,
                )
                result = AuthorsLLM(authors=salvaged)
                result._salvaged_after = code
                return result
            logger.error(
                f"LLM author extraction failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to extract authors", e) from e

    @track_llm_usage
    async def extract_paper_classification(
        self,
        text: str,
        file_hash: str = "unknown",
        *,
        json_mode: bool = False,
        boundary: str | None = None,
    ) -> PaperClassificationLLM:
        logger.debug(
            f"Starting LLM paper classification (hash={file_hash}, input length: {len(text)})"
        )
        try:
            spec = PROMPTS["classification"]
            boundary = boundary or uuid.uuid4().hex
            compact = text[: self._settings.llm.classification_max_chars]
            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": spec.build_user(boundary=boundary, text=compact)}],
                spec.system,
                client_override=self._get_json_mode_client() if json_mode else None,
                max_tokens=_task_max_tokens(
                    self._settings, self._settings.llm.paper_classification_max_tokens
                ),
            )

            logger.info(f"Successfully classified paper (hash={file_hash})")
            return cast("PaperClassificationLLM", result)
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(
                f"LLM paper classification failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to classify paper", e) from e

    @track_llm_usage
    async def label_paper_type(
        self, title: str, abstract: str, file_hash: str = "unknown"
    ) -> PaperTypeLabel:
        """Classify a paper's paper_type from its title + abstract only.

        The paper_type-only fallback tier for the trained multitask classifier:
        when the local classifier's paper_type confidence falls below
        ``ML_PAPER_CLASSIFIER_MIN_CONFIDENCE``, core-metadata extraction
        escalates to this method. Reuses ``PROMPTS["paper_type_label"]`` +
        ``PaperTypeLabel`` (shared with evaluation/label_paper_type_llm.py) so
        the prompt never drifts.
        """
        logger.debug(f"Starting LLM paper_type labeling (hash={file_hash})")
        try:
            spec = PROMPTS["paper_type_label"]
            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": spec.build_user(title=title, abstract=abstract)}],
                spec.system,
                max_tokens=_task_max_tokens(
                    self._settings, self._settings.llm.paper_type_max_tokens
                ),
            )

            logger.info(f"Successfully labeled paper_type (hash={file_hash})")
            return cast("PaperTypeLabel", result)
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(
                f"LLM paper_type labeling failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to label paper_type", e) from e

    @track_llm_usage
    async def extract_core_metadata_merged(
        self, text: str, file_hash: str = "unknown"
    ) -> CoreMetadataLLM:
        """Single-call variant of :meth:`extract_core_metadata`.

        Experimental (``LLM_MERGED_CORE_METADATA``): merges the three focused
        prompts into one call so the front-matter text is sent once instead
        of three times. Uses the authors reasoning effort — the strictest of
        the three — since author parsing is the most error-prone part.
        """
        logger.debug(
            f"Starting merged LLM core metadata extraction "
            f"(hash={file_hash}, input length: {len(text)})"
        )
        try:
            spec = PROMPTS["core_metadata"]
            boundary = uuid.uuid4().hex
            capped_text = self._cap_input(text, self._settings)
            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": spec.build_user(boundary=boundary, text=capped_text)}],
                spec.system,
                reasoning_effort=self._settings.llm.reasoning_effort_authors,
            )

            logger.info(f"Successfully extracted merged core metadata (hash={file_hash})")
            return cast("CoreMetadataLLM", result)
        except ProcessingError:
            raise
        except Exception as e:
            recovered = _recover_finished_response(e, CoreMetadataLLM)
            if recovered is not None:
                return cast("CoreMetadataLLM", recovered)
            logger.error(
                f"Merged LLM core metadata extraction failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to extract core metadata", e) from e

    async def extract_core_metadata(
        self,
        text: str,
        file_hash: str = "unknown",
        *,
        authors_text: str | None = None,
        classification_text: str | None = None,
        include_classification: bool = True,
    ) -> CoreMetadataLLM:
        """Run title/keywords, authors, and optional classification concurrently.

        Returns a combined CoreMetadataLLM for backward compatibility. With
        ``LLM_MERGED_CORE_METADATA`` set, its single-call schema is retained.
        Otherwise the fan-out uses two calls when ``include_classification`` is
        false, or three calls when it is true.

        ``authors_text`` / ``classification_text`` optionally carry task-specific
        context slices for the authors and classification calls (see
        ``LLM_PER_TASK_CONTEXT``); ``None`` falls back to the full ``text``.
        Title/keywords always receives the full ``text``.
        """
        if getattr(self._settings.llm, "merged_core_metadata", False):
            try:
                return await self.extract_core_metadata_merged(text, file_hash=file_hash)
            except (LlmTruncatedError, LlmInvalidOutputError) as exc:
                # The one call carries every field. Its response fails the same
                # way on a retry, so return an empty record with every field
                # marked failed; the extractor keeps references and the rest.
                logger.warning(
                    "Merged core metadata extraction failed (hash=%s); every core field "
                    "is marked failed: %s",
                    file_hash,
                    exc,
                )
                empty = CoreMetadataLLM(authors=[])
                empty._field_failures = dict.fromkeys(CoreMetadataLLM.model_fields, exc.error_code)
                return empty

        authors_text = authors_text or text
        classification_text = classification_text or text

        logger.debug(
            f"Starting LLM core metadata extraction (hash={file_hash}, input length: {len(text)})"
        )

        # One boundary for the whole fan-out: the three calls fence the SAME
        # front-matter text, and a shared boundary keeps the fenced-document
        # prefix byte-identical so provider prompt caches can reuse it.
        boundary = uuid.uuid4().hex

        # Start all three independent extraction calls together. They share the same fenced
        # front-matter prefix for provider cache reuse.
        tasks = [
            self.extract_title_keywords(text, file_hash=file_hash, boundary=boundary),
            self.extract_authors(authors_text, file_hash=file_hash, boundary=boundary),
        ]
        if include_classification:
            tasks.append(
                self.extract_paper_classification(
                    classification_text, file_hash=file_hash, boundary=boundary
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        title_kw, authors = results[:2]
        classification = results[2] if include_classification else PaperClassificationLLM()

        # Invalid structured output is a typed protocol failure, never an
        # optional title/author/classification miss. Inspect the entire fan-out
        # before applying the ordinary upstream degradation policy.
        for result in results:
            if isinstance(result, ProcessingError):
                raise result
        # A cancelled call is not a failed call.
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result

        # Response fields of a failed call, with the failure's error code.
        field_failures: dict[str, str] = {}
        # Title/keywords is the anchor of the record. When the service did not
        # answer, propagate: the paper fails, and a retry can complete it. A
        # truncated or invalid response fails the same way on every retry, so
        # keep what the other calls extracted and mark its fields failed.
        if isinstance(title_kw, BaseException):
            typed = {
                id(outcome): (
                    outcome
                    if isinstance(outcome, LlmCallError)
                    else llm_call_error(f"Failed to extract {task}", outcome)
                )
                for outcome, task in (
                    (title_kw, "title/keywords"),
                    (authors, "authors"),
                    (classification, "paper classification"),
                )
                if isinstance(outcome, BaseException)
            }
            # A partial record is kept only when no call failed for a reason a
            # retry could fix; otherwise the paper fails as it always did.
            for outcome in (title_kw, authors, classification):
                error = typed.get(id(outcome))
                if error is None or isinstance(error, (LlmTruncatedError, LlmInvalidOutputError)):
                    continue
                if error is outcome:
                    raise error
                raise error from cast(BaseException, outcome)
            title_error = typed[id(title_kw)]
            logger.warning(
                "Title/keywords extraction failed (hash=%s); keeping the other core "
                "metadata calls: %s",
                file_hash,
                title_error,
            )
            field_failures.update(
                dict.fromkeys(TitleKeywordsLLM.model_fields, title_error.error_code)
            )
            title_kw = TitleKeywordsLLM()
        # Authors failing outright (e.g. a persistent truncation the salvage
        # couldn't recover) must NOT sink the whole record. Degrade to an
        # empty list; the empty-author re-roll below gets one more recovery
        # shot before we accept it.
        if isinstance(authors, BaseException):
            logger.warning(
                "Author extraction failed (hash=%s); keeping title/keywords and "
                "attempting empty-author recovery: %s",
                file_hash,
                authors,
            )
            field_failures["authors"] = llm_failure_code(authors)
            authors = AuthorsLLM(authors=[])

        # Classification is optional — degrade gracefully
        classification_failure: str | None = None
        if isinstance(classification, BaseException):
            logger.warning(f"Paper classification failed, continuing without: {classification}")
            classification_failure = llm_failure_code(classification)
            classification = PaperClassificationLLM()

        # A schema-valid empty author list is a domain result. CoreMetadataExtractor owns the
        # single grounded recovery attempt, while this client preserves the blank response.
        authors_were_empty = not authors.authors
        if authors_were_empty:
            logger.warning(
                "Author extraction returned 0 authors (hash=%s); extractor-owned "
                "recovery may follow",
                file_hash,
            )

        classification_blank = (
            classification.paper_type is None
            and classification.oecd_domain is None
            and classification.oecd_subdomain is None
        )
        if (
            include_classification
            and authors_were_empty
            and classification_blank
            and self.json_mode_reroll_is_distinct()
        ):
            try:
                reclass = await self.extract_paper_classification(
                    classification_text, file_hash=file_hash, json_mode=True, boundary=boundary
                )
            except UpstreamServiceError as exc:
                logger.warning("Classification re-roll failed (hash=%s): %s", file_hash, exc)
                reclass = classification
            if reclass.paper_type is not None or reclass.oecd_domain is not None:
                classification = reclass
                classification_failure = None
        if classification_failure is not None:
            field_failures.update(
                dict.fromkeys(
                    ("paper_type", "oecd_domain", "oecd_subdomain"), classification_failure
                )
            )

        combined = CoreMetadataLLM(
            title=title_kw.title,
            **({"abstract": title_kw.abstract} if "abstract" in title_kw.model_fields_set else {}),
            keywords=title_kw.keywords,
            authors=authors.authors,
            oecd_domain=classification.oecd_domain,
            oecd_subdomain=classification.oecd_subdomain,
            paper_type=classification.paper_type,
            journal=title_kw.journal,
            volume=title_kw.volume,
            issue=title_kw.issue,
            first_page=title_kw.first_page,
            last_page=title_kw.last_page,
            issn=title_kw.issn,
            publisher=title_kw.publisher,
            published=title_kw.published,
            license=title_kw.license,
        )

        combined._abstract_explicitly_absent = title_kw._abstract_explicitly_absent
        combined._field_failures = field_failures
        salvaged_after = getattr(authors, "_salvaged_after", None)
        combined._authors_salvaged_after = (
            salvaged_after if isinstance(salvaged_after, str) else None
        )

        logger.info(f"Successfully extracted core metadata (hash={file_hash})")
        return combined

    @track_llm_usage
    async def extract_references(
        self,
        text: str,
        file_hash: str = "unknown",
        start_index: int = 1,
        expected_count: int | None = None,
    ) -> list[PaperReferenceLLM]:
        logger.debug(
            f"Starting LLM reference extraction (hash={file_hash}, input length: {len(text)})"
        )
        try:
            spec = PROMPTS["references_parse"]
            boundary = uuid.uuid4().hex
            capped_text = self._cap_input(text, self._settings)
            content = spec.build_user(boundary=boundary, text=capped_text, start_index=start_index)

            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": content}],
                spec.system,
                max_tokens=self._settings.REF_PARSE_MAX_TOKENS,
            )

            extracted_refs = result.references
            expected = _expected_ref_indices(capped_text, start_index, expected_count)
            reported = [ref.index for ref in extracted_refs]
            trusted = (
                bool(expected)
                and len(set(reported)) == len(reported)
                and all(idx in expected for idx in reported)
            )
            if trusted:
                extracted_refs.sort(key=lambda r: r.index)
                if len(extracted_refs) != len(expected):
                    logger.warning(
                        f"LLM returned {len(extracted_refs)}/{len(expected)} refs in batch "
                        f"(hash={file_hash}, start_index={start_index}); "
                        "keeping LLM-reported indices"
                    )
            else:
                if extracted_refs:
                    logger.warning(
                        f"LLM-reported ref indices invalid ({reported}) — re-indexing "
                        f"positionally (hash={file_hash}, start_index={start_index})"
                    )
                # Position N of the output is only input N when the LLM
                # returned one ref per numbered entry, in order. Two separate
                # things break that:
                #   - the count disagrees, so an entry was dropped or merged;
                #   - an index repeats, which means the model was not tracking
                #     entries one-to-one even where the count happens to line
                #     up — 15 rows labelled 1,2,3,3,5,… for 15 entries.
                # The second was checked for ``trusted`` but not here, so a
                # duplicate with a matching count took the positional path with
                # backfills left *on*.
                # Either way the mapping has slipped: mark these so downstream
                # segment-anchored backfills (issue recovery, DOI rescue) stay
                # off rather than copying the neighbouring segment's printed
                # DOI onto the wrong reference. Indices that are merely out of
                # range — a batch numbered 1..n instead of from start_index —
                # are a renumbering, which positional re-indexing fixes exactly,
                # so they deliberately do not count as slipped.
                repeated = len(set(reported)) != len(reported)
                slipped = bool(expected) and (repeated or len(extracted_refs) != len(expected))
                for offset, ref in enumerate(extracted_refs):
                    ref.index = start_index + offset
                    if slipped:
                        ref.mark_index_untrusted()
            logger.info(
                f"Successfully extracted {len(extracted_refs)} references "
                f"(hash={file_hash}, start_index={start_index})"
            )
            return cast("list[PaperReferenceLLM]", extracted_refs)
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(
                f"LLM reference extraction failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to extract references", e) from e

    @track_llm_usage
    async def extract_references_chunk(
        self,
        text: str,
        file_hash: str = "unknown",
    ) -> list[PaperReferenceLLM]:
        """Chunk-tolerant reference parse: *text* is a region-aligned slice of
        the reference list, not a pre-segmented numbered list. The model finds
        its own boundaries, so — unlike :meth:`extract_references` — there is
        no trusted numbered contract to validate against; every call
        re-indexes the returned refs positionally by their order in the
        response.
        """
        logger.debug(
            f"Starting LLM chunked reference extraction (hash={file_hash}, "
            f"input length: {len(text)})"
        )
        try:
            spec = PROMPTS["references_parse_chunk"]
            boundary = uuid.uuid4().hex
            capped_text = self._cap_input(text, self._settings)
            content = spec.build_user(boundary=boundary, text=capped_text)

            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": content}],
                spec.system,
                max_tokens=self._settings.REF_PARSE_MAX_TOKENS,
            )

            extracted_refs = result.references
            for offset, ref in enumerate(extracted_refs):
                ref.index = offset + 1
            logger.info(
                f"Successfully extracted {len(extracted_refs)} references from chunk "
                f"(hash={file_hash})"
            )
            return cast("list[PaperReferenceLLM]", extracted_refs)
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(
                f"LLM chunked reference extraction failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to extract references (chunk)", e) from e

    @track_llm_usage
    async def segment_references(self, text: str, file_hash: str = "unknown") -> list[str]:
        """Segment a references block into per-reference verbatim opening anchors.

        Cheap O(N)-output-token segmentation: the model emits one short anchor
        per reference (caller snaps them onto the text via
        :mod:`bibr.extract.anchor_snap`). Each window is rate-limited as one
        provider request, then routed through ``_invoke_structured`` so it
        shares usage tracking, breaker, concurrency gating, and retries.
        """
        logger.debug(
            f"Starting LLM reference segmentation (hash={file_hash}, input length: {len(text)})"
        )
        try:
            # Bound seg OUTPUT, not just input: window on the smaller of the input
            # budget and ref_seg_window_chars so a long bibliography is split into
            # several bounded-output calls instead of one that overruns the timeout.
            budget = min(
                self._settings.llm.max_input_chars,
                self._settings.llm.ref_seg_window_chars,
            )
            windows = _window_ref_text(text, budget)
            if len(windows) > _SEG_MAX_WINDOWS:
                logger.warning(
                    "Reference section spans %d windows (%d chars each); segmenting "
                    "the first %d and dropping the rest (hash=%s)",
                    len(windows),
                    budget,
                    _SEG_MAX_WINDOWS,
                    file_hash,
                )
                windows = windows[:_SEG_MAX_WINDOWS]

            # Segment each window independently, then concatenate the anchors in
            # window order. Windows partition the text on line
            # boundaries with no overlap, so no reference recurs across them; any
            # incidental repeat — including two sibling references (e.g. 2013a/2013b)
            # with byte-identical openings — is kept deliberately, because anchor_snap
            # claims a distinct text position per anchor. De-duplicating here would
            # merge such siblings into one, silently dropping a reference.
            per_window = await asyncio.gather(
                *(self._segment_window(w) for w in windows),
                return_exceptions=True,
            )
            for result in per_window:
                if isinstance(result, ProcessingError):
                    raise result
            for result in per_window:
                if isinstance(result, BaseException):
                    raise result
            anchors = [anchor for win_anchors in per_window for anchor in win_anchors]
            logger.info(
                f"Segmented references into {len(anchors)} anchors "
                f"across {len(windows)} window(s) (hash={file_hash})"
            )
            return anchors
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(
                f"LLM reference segmentation failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to segment references", e) from e

    async def _segment_window(self, window_text: str) -> list[str]:
        """LLM-segment one references window into verbatim opening anchors."""
        spec = PROMPTS["references_segment"]
        boundary = uuid.uuid4().hex
        result = await self._invoke_structured(
            spec.response_model,
            [{"role": "user", "content": spec.build_user(boundary=boundary, text=window_text)}],
            spec.system,
        )
        return list(result.anchors)

    @track_llm_usage
    async def extract_research_integrity(
        self,
        funding_text: str,
        contributions_text: str,
        author_names: list[tuple[str, str]],
        *,
        affiliation_list: list[str],
        file_hash: str = "unknown",
    ) -> ResearchIntegrityLLM:
        """Parse structured funding + author contributions + affiliations from
        the funding statement, author-contributions statement, and affiliation
        list.

        ``author_names`` is the paper's ``(given, family)`` byline, supplied to
        help the model read abbreviated names/initials in the contributions
        statement. ``affiliation_list`` is the deduped verbatim affiliation
        strings, presented as a numbered list. Callers must skip this call when
        all three inputs are empty.
        """
        logger.debug(
            f"Starting LLM research-integrity extraction (hash={file_hash}, "
            f"funding={len(funding_text)} chars, contributions={len(contributions_text)} chars, "
            f"affiliations={len(affiliation_list)})"
        )
        try:
            spec = PROMPTS["research_integrity"]
            boundary = uuid.uuid4().hex
            authors_block = (
                "\n".join(f"- {g} {f}".strip() for g, f in author_names) or "(none provided)"
            )
            affiliations_block = (
                "\n".join(f"[{i}] {t}" for i, t in enumerate(affiliation_list, start=1))
                or "(none provided)"
            )
            content = spec.build_user(
                boundary=boundary,
                funding_text=self._cap_input(funding_text, self._settings),
                contributions_text=self._cap_input(contributions_text, self._settings),
                affiliations_block=self._cap_input(affiliations_block, self._settings),
                authors_block=authors_block,
            )
            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": content}],
                spec.system,
                max_tokens=_task_max_tokens(
                    self._settings, self._settings.llm.integrity_max_tokens
                ),
            )
            logger.info(
                f"Extracted {len(result.funding)} funder(s), "
                f"{len(result.contributions)} contribution(s), "
                f"{len(result.affiliations)} affiliation(s) (hash={file_hash})"
            )
            return cast("ResearchIntegrityLLM", result)
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(
                f"LLM research-integrity extraction failed (hash={file_hash}): {e}",
                exc_info=True,
            )
            raise llm_call_error("Failed to extract research integrity", e) from e

    @track_llm_usage
    async def resolve_citations(
        self,
        ambiguous_citations: list[tuple[int, str]],
        reference_summary: list[dict],
        file_hash: str = "unknown",
    ) -> list:
        """Resolve ambiguous inline citations against the reference list using LLM.

        Args:
            ambiguous_citations: list of (text_id, citation_text) tuples
            reference_summary: list of dicts with bib_id, author, year, title
            file_hash: file hash for logging

        Returns:
            list of CitationMatch objects. A failed call raises an
            :class:`~bibr.exceptions.LlmCallError`; the caller degrades.
        """
        logger.debug(
            f"Starting LLM citation resolution (hash={file_hash}, "
            f"{len(ambiguous_citations)} candidates, {len(reference_summary)} refs)"
        )
        try:
            # Build compact reference table
            ref_lines = []
            for r in reference_summary:
                ref_lines.append(
                    f'  bib_id={r["bib_id"]}: {r["author"]} ({r["year"]}) "{r["title"]}"'
                )
            ref_block = "\n".join(ref_lines)

            # Build citation list
            cite_lines = []
            for text_id, cite_text in ambiguous_citations:
                cite_lines.append(f"  text_id={text_id}: {cite_text}")
            cite_block = "\n".join(cite_lines)

            spec = PROMPTS["citation_resolution"]
            boundary = uuid.uuid4().hex
            prompt = spec.build_user(
                boundary=boundary,
                ref_block=self._cap_input(ref_block, self._settings),
                cite_block=self._cap_input(cite_block, self._settings),
            )

            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": prompt}],
                spec.system,
                reasoning_effort=self._settings.llm.reasoning_effort_citations,
                max_tokens=_task_max_tokens(self._settings, self._settings.llm.citation_max_tokens),
            )

            matches = result.matches
            logger.info(
                f"LLM resolved {sum(1 for m in matches if m.bib_id is not None)}/{len(matches)} "
                f"citations (hash={file_hash})"
            )
            return cast("list", matches)

        except ProcessingError:
            raise
        except Exception as e:
            raise llm_call_error("Failed to resolve citations", e) from e

    @track_llm_usage
    async def extract_equations(
        self,
        sentences: list[tuple[int, str]],
        file_hash: str = "unknown",
    ) -> list:
        """Extract statistical equations from sentences using LLM.

        Args:
            sentences: list of (text_id, sentence_text) tuples
            file_hash: file hash for logging

        Returns:
            list of PaperEquation objects. A failed call raises an
            :class:`~bibr.exceptions.LlmCallError`; the caller degrades.
        """
        from bibr.paper_contents import PaperEquation

        logger.debug(
            f"Starting LLM equation extraction (hash={file_hash}, {len(sentences)} sentences)"
        )
        try:
            sent_lines = []
            text_id_map = {}
            for i, (text_id, text) in enumerate(sentences):
                sent_lines.append(f"  [{i}] (text_id={text_id}): {text}")
                text_id_map[i] = text_id
            sent_block = "\n".join(sent_lines)

            spec = PROMPTS["equations"]
            boundary = uuid.uuid4().hex
            prompt = spec.build_user(
                boundary=boundary,
                sent_block=self._cap_input(sent_block, self._settings),
            )

            result = await self._invoke_structured(
                spec.response_model,
                [{"role": "user", "content": prompt}],
                spec.system,
                max_tokens=_task_max_tokens(self._settings, self._settings.llm.equation_max_tokens),
            )

            llm_eqs = result.equations
            paper_eqs = []
            default_text_id = sentences[0][0] if sentences else 0
            for eq in llm_eqs:
                text_id = (
                    text_id_map.get(eq.sentence_index, default_text_id)
                    if eq.sentence_index is not None
                    else default_text_id
                )
                paper_eqs.append(
                    PaperEquation(
                        text_id=text_id,
                        grp_id=0,  # assigned by caller
                        lhs=eq.lhs,
                        df=eq.df or "",
                        comp=eq.comp,
                        rhs=eq.rhs,
                    )
                )

            logger.info(f"LLM extracted {len(paper_eqs)} equation components (hash={file_hash})")
            return paper_eqs

        except ProcessingError:
            raise
        except Exception as e:
            raise llm_call_error("Failed to extract equations", e) from e

    async def close(self):
        """Close the rate limiter and clean up resources."""
        self._usage.clear()
        self._usage_by_file.clear()
        self._labels_by_file.clear()
        self._traces_by_file.clear()
        if self._limiter:
            await self._limiter.close()
