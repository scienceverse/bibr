"""bibr exception hierarchy.

Concrete subclasses cover the failure modes the pipeline needs to
distinguish at HTTP / CLI boundaries:

- ``InputValidationError`` (4xx) — user-supplied input rejected
- ``UpstreamServiceError`` (5xx) — third-party (LLM, OCR, Crossref) failed
- ``ProcessingError``       (422 for stable processing failures) — processing failed mid-pipeline
- ``ConfigurationError``          — invalid ``.env`` / environment settings

A failed LLM task call raises an ``LlmCallError``, an ``UpstreamServiceError``
subclass whose class and ``error_code`` say how it failed: the service did not
answer (``LlmServiceError``, ``LlmTimeoutError``), refused the request
(``LlmRejectedError``), or answered with a truncated or invalid response
(``LlmTruncatedError``, ``LlmInvalidOutputError``). Serve maps the last two to
422, because retrying the same request cannot help.

Most call sites catch the bare ``Exception`` and surface a structured
``ProcessingStatus`` instead. The specific catches are the degrade sites that
keep a paper when an optional ``UpstreamServiceError`` call fails,
``Pipeline.process_file`` and serve's error translation. Keep the hierarchy
minimal; do not add new subclasses unless a new catch site needs to
discriminate them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

_SAFE_NATIVE_INVALID_CATEGORIES = frozenset(
    {
        "empty",
        "non_json",
        "truncated",
        "trailing_content",
        "non_object",
        "schema_invalid",
    }
)

_SAFE_LLM_LABELS = frozenset(
    {
        "unlabeled",
        "extract_title_keywords",
        "extract_authors",
        "extract_paper_classification",
        "label_paper_type",
        "extract_core_metadata_merged",
        "extract_references",
        "extract_references_chunk",
        "segment_references",
        "extract_research_integrity",
        "resolve_citations",
        "extract_equations",
        "section_classifier",
        "implicit_sections",
    }
)

SAFE_LLM_COUNTER_NAMES = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "calls",
    "logical_calls",
    "attempts",
    "retries",
    "failed_calls",
    "native_attempts",
    "instructor_attempts",
    "protocol_fallbacks",
    "native_invalid_outputs",
    "native_invalid_empty",
    "native_invalid_non_json",
    "native_invalid_truncated",
    "native_invalid_trailing_content",
    "native_invalid_non_object",
    "native_invalid_schema_invalid",
)


def _require_safe_count(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class SafeLlmLabelDiagnostics:
    """One allowlisted LLM call-site's numeric terminal counters.

    The tuple representation prevents arbitrary dictionary keys or values from
    entering exceptions, HTTP translation, metering, or async-job results.
    """

    label: str
    counters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if type(self.label) is not str or self.label not in _SAFE_LLM_LABELS:
            raise ValueError(f"label is not an allowlisted LLM call site: {self.label!r}")
        if type(self.counters) not in {list, tuple}:
            raise TypeError("counters must be a list or tuple of name/value pairs")
        if len(self.counters) != len(SAFE_LLM_COUNTER_NAMES):
            raise ValueError("counters must contain the fixed safe LLM counter set")
        canonical: list[tuple[str, int]] = []
        for expected_name, item in zip(
            SAFE_LLM_COUNTER_NAMES,
            self.counters,
            strict=True,
        ):
            item_object: object = item
            if type(item_object) is list:
                pair: list[object] | tuple[object, ...] = cast(list[object], item_object)
            elif type(item_object) is tuple:
                pair = cast(tuple[object, ...], item_object)
            else:
                raise TypeError("each counter must be a two-item list or tuple")
            if len(pair) != 2:
                raise TypeError("each counter must be a two-item list or tuple")
            name, value = pair
            if type(name) is not str or name != expected_name:
                raise ValueError("counters must contain the fixed safe LLM counter set")
            canonical.append((expected_name, _require_safe_count(expected_name, value)))
        object.__setattr__(self, "counters", tuple(canonical))

    @classmethod
    def from_counts(
        cls,
        label: str,
        counts: Mapping[str, object],
    ) -> SafeLlmLabelDiagnostics:
        if label not in _SAFE_LLM_LABELS:
            raise ValueError(f"label is not an allowlisted LLM call site: {label!r}")
        return cls(
            label=label,
            counters=tuple(
                (name, _require_safe_count(name, counts.get(name, 0)))
                for name in SAFE_LLM_COUNTER_NAMES
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return dict(self.counters)


@dataclass(frozen=True, slots=True)
class SafeLlmDiagnostics:
    """Filtered diagnostics for a terminal invalid structured completion.

    This carrier deliberately excludes prompts, completions, model responses,
    exception text, hashes, filenames, and arbitrary mappings. Only one stable
    invalid-output category plus numeric usage/protocol counters may cross the
    failed-file boundary.
    """

    invalid_category: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    labels: tuple[SafeLlmLabelDiagnostics, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.invalid_category) is not str
            or self.invalid_category not in _SAFE_NATIVE_INVALID_CATEGORIES
        ):
            raise ValueError(f"invalid_category is not allowlisted: {self.invalid_category!r}")
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
        ):
            _require_safe_count(name, getattr(self, name))
        if type(self.labels) is not tuple or not all(
            type(item) is SafeLlmLabelDiagnostics for item in self.labels
        ):
            raise TypeError("labels must be an exact tuple of SafeLlmLabelDiagnostics")
        label_names = tuple(item.label for item in self.labels)
        if len(label_names) != len(set(label_names)):
            raise ValueError("labels must be unique")

    @classmethod
    def from_native_error(cls, error: BaseException) -> SafeLlmDiagnostics:
        return cls(
            invalid_category=str(getattr(error, "category", "")),
            input_tokens=_require_safe_count("input_tokens", getattr(error, "input_tokens", 0)),
            output_tokens=_require_safe_count(
                "output_tokens", getattr(error, "completion_tokens", 0)
            ),
            total_tokens=_require_safe_count("total_tokens", getattr(error, "total_tokens", 0)),
            cached_input_tokens=_require_safe_count(
                "cached_input_tokens", getattr(error, "cached_input_tokens", 0)
            ),
        )

    def with_file_usage(
        self,
        usage: Mapping[str, Mapping[str, object]],
        usage_by_label: Mapping[str, Mapping[str, object]],
    ) -> SafeLlmDiagnostics:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }
        saw_usage = False
        for model_counts in usage.values():
            if not isinstance(model_counts, Mapping):
                continue
            for name in totals:
                value = model_counts.get(name, 0)
                if type(value) is int and value >= 0:
                    totals[name] += value
                    saw_usage = saw_usage or value > 0

        labels: list[SafeLlmLabelDiagnostics] = []
        for label, counts in usage_by_label.items():
            if label not in _SAFE_LLM_LABELS or not isinstance(counts, Mapping):
                continue
            try:
                labels.append(SafeLlmLabelDiagnostics.from_counts(label, counts))
            except (TypeError, ValueError):
                continue
        labels.sort(key=lambda item: item.label)

        return SafeLlmDiagnostics(
            invalid_category=self.invalid_category,
            input_tokens=totals["input_tokens"] if saw_usage else self.input_tokens,
            output_tokens=totals["output_tokens"] if saw_usage else self.output_tokens,
            total_tokens=totals["total_tokens"] if saw_usage else self.total_tokens,
            cached_input_tokens=(
                totals["cached_input_tokens"] if saw_usage else self.cached_input_tokens
            ),
            labels=tuple(labels),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "invalid_category": self.invalid_category,
            "llm_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "cached_input_tokens": self.cached_input_tokens,
            },
            "llm_usage_by_label": {item.label: item.to_dict() for item in self.labels},
        }


class BibrError(Exception):
    """Base exception for bibr"""

    pass


class UpstreamServiceError(BibrError):
    """Raised when an upstream service (OCR, LLM, Crossref) fails (5xx, connection error)"""

    def __init__(
        self,
        service_name: str,
        message: str,
        original_error: BaseException | None = None,
    ):
        self.service_name = service_name
        self.original_error = original_error
        super().__init__(f"Error in {service_name}: {message}")


class LlmCallError(UpstreamServiceError):
    """An LLM task call failed; the subclass says how.

    Every class here is an ``UpstreamServiceError``, so each site that degrades
    on an upstream failure keeps doing so. ``error_code`` is the stable code
    for warnings, issues and HTTP details, and the message keeps a bounded
    description of the cause. :func:`bibr.clients.llm.llm_call_error` picks the
    class from the exception chain; this base class is a failure no rule
    recognized, such as a bug in the call path.
    """

    error_code = "llm_failed"

    def __init__(
        self,
        message: str,
        original_error: BaseException | None = None,
        *,
        cause: str | None = None,
    ):
        super().__init__("LLM", f"{message} ({cause})" if cause else message, original_error)
        self.cause = cause


class LlmServiceError(LlmCallError):
    """The LLM service did not answer: a 429 or 5xx status, a transport failure
    or an open circuit breaker. Retrying later can succeed."""


class LlmTimeoutError(LlmServiceError):
    """The LLM call did not finish within its time budget."""

    error_code = "llm_timeout"


class LlmRejectedError(LlmCallError):
    """The LLM service refused the request with a 4xx status other than 429 —
    a credential, model or request-size problem rather than an outage."""


class LlmTruncatedError(LlmCallError):
    """The model stopped at its output-token limit before finishing the response.

    Deterministic for the same input and settings, so a retry does not help.
    """

    error_code = "llm_truncated"


class LlmInvalidOutputError(LlmCallError):
    """The model's finished response failed JSON parsing or schema validation,
    including any validation re-asks. Deterministic, like a truncation."""

    error_code = "llm_invalid_output"


class InputValidationError(BibrError):
    """Raised when the input file is invalid or causes an error in downstream services (4xx)"""

    pass


class ProcessingError(BibrError):
    """Raised when processing (text structuring, metadata extraction) fails.

    Stable, client-actionable processing codes are exposed as HTTP 422. Errors
    without a stable code retain the legacy generic processing response.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        failed_stage: str | None = None,
        safe_diagnostics: SafeLlmDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failed_stage = failed_stage
        self.safe_diagnostics = safe_diagnostics

    def __setattr__(self, name: str, value: object) -> None:
        if (
            name == "safe_diagnostics"
            and value is not None
            and type(value) is not SafeLlmDiagnostics
        ):
            raise TypeError("safe_diagnostics must be exact SafeLlmDiagnostics or None")
        super().__setattr__(name, value)


class ConfigurationError(BibrError):
    """Raised when the ``.env`` / environment fails settings validation.

    Wraps the pydantic ``ValidationError`` from constructing ``GlobalSettings``
    in a human-friendly message that names the actual environment variable(s)
    to fix and their allowed values, so ``bibr --help`` / ``bibr doctor`` /
    ``bibr setup`` — the tools a user reaches for to repair their config —
    surface a fixable problem instead of a raw traceback.

    ``problems`` holds one plain-text line per offending variable so ``bibr
    doctor`` can render each as its own failed check.
    """

    def __init__(self, message: str, *, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or []
