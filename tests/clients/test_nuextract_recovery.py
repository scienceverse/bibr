"""Bounded NuExtract-native recovery through Instructor JSON mode."""

from __future__ import annotations

import asyncio
import traceback
from collections import Counter
from unittest import mock

import pytest
from pydantic import BaseModel, ValidationError

from bibr.clients.llm import InstructorBackend, LLMClient, usage_file_context
from bibr.clients.nuextract import NuExtractInvalidOutput, NuExtractNativeBackend
from bibr.config import GlobalSettings
from bibr.exceptions import ProcessingError
from bibr.schemas import (
    AuthorsLLM,
    CoreMetadataLLM,
    PaperClassificationLLM,
    TitleKeywordsLLM,
)
from bibr.utils.circuit_breaker import CircuitOpenError


class _Transient503(Exception):
    status_code = 503


class _SequenceBackend:
    """One physical call per outcome, with complete call-argument recording."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def create(
        self,
        *,
        response_model,
        system,
        messages,
        want_completion,
        reasoning_effort=None,
        max_tokens=None,
        client_override=None,
    ):
        self.calls.append(
            {
                "response_model": response_model,
                "system": system,
                "messages": messages,
                "want_completion": want_completion,
                "reasoning_effort": reasoning_effort,
                "max_tokens": max_tokens,
                "client_override": client_override,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, None


class _NoopLimiter:
    async def acquire(self):
        return None


def _settings(**overrides) -> GlobalSettings:
    settings = GlobalSettings()
    defaults = {
        "provider": "openai",
        "base_url": "http://127.0.0.1:8767/v1",
        "model": "numind/NuExtract3-FP8",
        "structured_backend": "nuextract-native",
        "track_usage": False,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(settings.llm, key, value)
    return settings


def _title(value: str = "Recovered") -> TitleKeywordsLLM:
    return TitleKeywordsLLM(title=value, abstract=None, keywords=[])


def _invalid(
    category: str = "non_json",
    *,
    input_tokens: int = 11,
    completion_tokens: int = 7,
    cached_input_tokens: int = 3,
) -> NuExtractInvalidOutput:
    return NuExtractInvalidOutput(
        category=category,
        model="numind/NuExtract3-FP8",
        finish_reason="stop",
        response_chars=23,
        response_sha256="a" * 64,
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        total_tokens=input_tokens + completion_tokens,
        cached_input_tokens=cached_input_tokens,
        response_model="TitleKeywordsLLM",
    )


def _client(
    native: _SequenceBackend,
    instructor: _SequenceBackend,
    *,
    settings: GlobalSettings | None = None,
) -> tuple[LLMClient, Counter]:
    client = LLMClient(settings=settings or _settings(), backend=native)
    # Explicit injected transports are otherwise protocol-agnostic. This is
    # the captured route selected by an explicit native configuration.
    client._backend_kind = "nuextract-native"
    client._limiter = _NoopLimiter()
    client._make_recovery_instructor_backend = mock.Mock(return_value=instructor)
    client._get_json_mode_client = mock.Mock(return_value=object())
    metrics: Counter = Counter()
    record_label_metric = client._record_label_metric

    def record_both(name, value):
        metrics.update({name: value})
        record_label_metric(name, value)

    client._record_label_metric = record_both
    return client, metrics


def _capture_protocol_metrics(client: LLMClient) -> Counter:
    metrics: Counter = Counter()
    record_label_metric = client._record_label_metric

    def record_both(name, value):
        metrics.update({name: value})
        record_label_metric(name, value)

    client._record_label_metric = record_both
    return metrics


async def _invoke(client: LLMClient, *, client_override=None):
    return await client._invoke_structured(
        TitleKeywordsLLM,
        [{"role": "user", "content": "paper"}],
        "system",
        reasoning_effort="high",
        client_override=client_override,
    )


@pytest.fixture(autouse=True)
def _zero_backoff(monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(LLMClient, "_RETRY_BASE_DELAY", 0.0)


async def test_native_success_has_one_native_call_and_no_retry():
    native = _SequenceBackend([_title("Native")])
    instructor = _SequenceBackend([])
    client, metrics = _client(native, instructor)

    result = await _invoke(client)

    assert result.title == "Native"
    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (1, 0, 1, 0)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (1, 0)
    assert native.calls[0]["reasoning_effort"] is None


async def test_terminal_native_invalid_is_converted_only_at_outer_structured_boundary():
    native_error = _invalid("non_json")
    native = _SequenceBackend([native_error])
    instructor = _SequenceBackend([])
    client, _ = _client(
        native,
        instructor,
        settings=_settings(provider="anthropic", base_url="http://127.0.0.1/v1"),
    )

    with pytest.raises(ProcessingError) as raised:
        await _invoke(client)

    assert str(raised.value) == "LLM returned invalid structured output"
    assert raised.value.error_code == "llm_invalid_output"
    assert raised.value.safe_diagnostics.invalid_category == "non_json"
    assert raised.value.safe_diagnostics.total_tokens == 18
    assert raised.value.__cause__ is native_error
    assert native_error.__traceback__ is None


async def test_private_fallback_state_machine_keeps_native_diagnostic_typed():
    native_error = _invalid("schema_invalid")
    native = _SequenceBackend([native_error])
    instructor = _SequenceBackend([])
    client, _ = _client(
        native,
        instructor,
        settings=_settings(provider="anthropic", base_url="http://127.0.0.1/v1"),
    )

    with pytest.raises(NuExtractInvalidOutput) as raised:
        await client._invoke_with_protocol_fallback(
            backend=native,
            protocol="nuextract-native",
            response_model=TitleKeywordsLLM,
            messages=[{"role": "user", "content": "paper"}],
            system_prompt="system",
            reasoning_effort=None,
        )

    assert raised.value is native_error


async def test_invalid_native_falls_back_once_and_first_fallback_is_not_retry():
    native = _SequenceBackend([_invalid("non_json")])
    instructor = _SequenceBackend([_title()])
    client, metrics = _client(native, instructor)

    result = await _invoke(client)

    assert result.title == "Recovered"
    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (1, 1, 2, 0)
    assert metrics["protocol_fallbacks"] == 1
    assert metrics["native_attempts"] == 1
    assert metrics["instructor_attempts"] == 1
    assert instructor.calls[0]["reasoning_effort"] is None
    assert instructor.calls[0]["client_override"] is client._get_json_mode_client.return_value


async def test_native_503_then_success_stays_on_captured_native_backend(monkeypatch):
    native = _SequenceBackend([_Transient503("one"), _title("Native")])
    instructor = _SequenceBackend([])
    client, metrics = _client(native, instructor)
    ensure = mock.Mock(wraps=client._ensure_backend_current)
    monkeypatch.setattr(client, "_ensure_backend_current", ensure)

    result = await _invoke(client)

    assert result.title == "Native"
    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (2, 0, 2, 1)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (2, 0)
    ensure.assert_called_once_with()


async def test_native_503_exhaustion_never_falls_back():
    native = _SequenceBackend([_Transient503("one"), _Transient503("two"), _Transient503("three")])
    instructor = _SequenceBackend([])
    client, metrics = _client(native, instructor)

    with pytest.raises(_Transient503, match="three"):
        await _invoke(client)

    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (3, 0, 3, 2)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (3, 0)


async def test_native_503_then_invalid_falls_back_once():
    native = _SequenceBackend([_Transient503("one"), _invalid("truncated")])
    instructor = _SequenceBackend([_title()])
    client, metrics = _client(native, instructor)

    result = await _invoke(client)

    assert result.title == "Recovered"
    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (2, 1, 3, 1)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (2, 1)


async def test_fallback_503_retries_never_return_to_native():
    native = _SequenceBackend([_invalid("truncated")])
    instructor = _SequenceBackend([_Transient503("one"), _Transient503("two"), _title()])
    client, metrics = _client(native, instructor)

    result = await _invoke(client)

    assert result.title == "Recovered"
    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (1, 3, 4, 2)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (1, 3)


async def test_unsupported_fallback_configuration_stops_after_native():
    native_error = _invalid()
    native = _SequenceBackend([native_error])
    instructor = _SequenceBackend([])
    client, metrics = _client(
        native,
        instructor,
        settings=_settings(provider="anthropic", base_url="http://127.0.0.1/v1"),
    )

    with pytest.raises(ProcessingError) as raised:
        await _invoke(client)

    assert raised.value.__cause__ is native_error
    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (1, 0, 1, 0)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (1, 0)


async def test_json_mode_client_override_bypasses_native_up_front():
    native = _SequenceBackend([])
    instructor = _SequenceBackend([_title()])
    client, metrics = _client(native, instructor)
    override = object()

    result = await _invoke(client, client_override=override)

    assert result.title == "Recovered"
    assert (
        len(native.calls),
        len(instructor.calls),
        metrics["attempts"],
        metrics["retries"],
    ) == (0, 1, 1, 0)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (0, 1)
    assert instructor.calls[0]["client_override"] is override
    assert metrics["protocol_fallbacks"] == 0


def test_fallback_preflight_is_pure_and_normalized():
    supported = LLMClient(
        settings=_settings(provider=" OpenAI ", base_url="  http://127.0.0.1/v1  ")
    )
    no_url = LLMClient(settings=_settings(provider="openai", base_url="   "))
    wrong_provider = LLMClient(
        settings=_settings(provider="anthropic", base_url="http://127.0.0.1/v1")
    )

    assert supported._supports_instructor_json_fallback() is True
    assert no_url._supports_instructor_json_fallback() is False
    assert wrong_provider._supports_instructor_json_fallback() is False
    assert supported._json_client is None


async def test_fallback_validation_failure_re_raises_safe_native_diagnostic_only():
    class _Required(BaseModel):
        value: int

    def make_validation_error() -> ValidationError:
        try:
            _Required(value="not-an-int")
        except ValidationError as error:
            return error
        raise AssertionError("expected ValidationError")

    validation_error = make_validation_error()
    native_error = _invalid()
    native = _SequenceBackend([native_error])
    instructor = _SequenceBackend([validation_error])
    client, _ = _client(native, instructor)

    with pytest.raises(ProcessingError) as raised:
        await _invoke(client)

    assert raised.value.__cause__ is native_error
    assert native_error.__cause__ is None
    assert native_error.__context__ is None
    assert len(native.calls) == len(instructor.calls) == 1


async def test_fallback_failure_does_not_retain_or_format_raw_bearing_exception(caplog):
    raw_sentinel = "RAW-FALLBACK-SENTINEL"
    native_error = _invalid()
    native = _SequenceBackend([native_error])
    instructor = _SequenceBackend([RuntimeError(raw_sentinel)])
    client, _ = _client(native, instructor)

    with pytest.raises(ProcessingError) as raised:
        await _invoke(client)

    rendered = "".join(
        traceback.format_exception(type(raised.value), raised.value, raised.value.__traceback__)
    )
    assert raw_sentinel not in str(raised.value)
    assert raw_sentinel not in repr(raised.value)
    assert raw_sentinel not in rendered
    assert raw_sentinel not in caplog.text
    assert raised.value.__cause__ is native_error
    assert native_error.__cause__ is None
    assert native_error.__context__ is None


async def test_invalid_native_usage_is_recorded_exactly_once_before_successful_fallback():
    native = _SequenceBackend([_invalid(input_tokens=11, completion_tokens=7)])
    instructor = _SequenceBackend([_title()])
    settings = _settings(track_usage=True)
    client, _ = _client(native, instructor, settings=settings)

    with usage_file_context("paper#1"):
        result = await client._run_labeled_call("title", lambda: _invoke(client))

    assert result.title == "Recovered"
    assert client.usage[settings.llm.model] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "cached_input_tokens": 3,
    }
    labels = client.usage_labels_pop_file("paper#1")[
        ("title", settings.llm.provider, settings.llm.model)
    ]
    assert labels["input_tokens"] == 11
    assert labels["output_tokens"] == 7
    assert labels["total_tokens"] == 18
    assert labels["cached_input_tokens"] == 3
    assert labels["calls"] == 1
    assert labels["attempts"] == labels["native_attempts"] + labels["instructor_attempts"]
    assert labels["native_attempts"] == 1
    assert labels["instructor_attempts"] == 1
    assert labels["protocol_fallbacks"] == 1
    assert labels["native_invalid_outputs"] == 1
    assert labels["native_invalid_non_json"] == 1
    assert labels["native_invalid_outputs"] == sum(
        labels[f"native_invalid_{category}"]
        for category in (
            "empty",
            "non_json",
            "truncated",
            "trailing_content",
            "non_object",
            "schema_invalid",
        )
    )


async def test_all_invalid_category_counters_are_initialized_on_success():
    native = _SequenceBackend([_title("Native")])
    instructor = _SequenceBackend([])
    client, _ = _client(native, instructor, settings=_settings(track_usage=True))

    with usage_file_context("paper#category-zeros"):
        await client._run_labeled_call("extract_title_keywords", lambda: _invoke(client))

    labels = client.usage_labels_pop_file("paper#category-zeros")[
        ("extract_title_keywords", client._settings.llm.provider, client._settings.llm.model)
    ]
    assert labels["attempts"] == 1
    assert labels["native_attempts"] == 1
    assert labels["instructor_attempts"] == 0
    assert labels["protocol_fallbacks"] == 0
    assert labels["native_invalid_outputs"] == 0
    assert all(
        labels[f"native_invalid_{category}"] == 0
        for category in (
            "empty",
            "non_json",
            "truncated",
            "trailing_content",
            "non_object",
            "schema_invalid",
        )
    )


async def _call_public_wrapper(client: LLMClient, name: str):
    if name == "title":
        return await client.extract_title_keywords("front matter")
    if name == "authors":
        return await client.extract_authors("byline")
    if name == "classification":
        return await client.extract_paper_classification("title and abstract")
    if name == "paper_type":
        return await client.label_paper_type("Title", "Abstract")
    if name == "merged":
        return await client.extract_core_metadata_merged("front matter")
    if name == "references":
        return await client.extract_references("1. Citation", expected_count=1)
    if name == "reference_chunk":
        return await client.extract_references_chunk("Citation")
    if name == "segmentation":
        return await client.segment_references("Citation")
    if name == "integrity":
        return await client.extract_research_integrity(
            "funded by X",
            "",
            [("Ada", "Lovelace")],
            affiliation_list=[],
        )
    if name == "citations":
        return await client.resolve_citations(
            [(1, "[1]")],
            [{"bib_id": "b1", "author": "A", "year": "2024", "title": "T"}],
        )
    if name == "equations":
        return await client.extract_equations([(1, "t = 2.0")])
    raise AssertionError(name)


@pytest.mark.parametrize(
    "wrapper",
    [
        "title",
        "authors",
        "classification",
        "paper_type",
        "merged",
        "references",
        "reference_chunk",
        "segmentation",
        "integrity",
        "citations",
        "equations",
    ],
)
async def test_every_public_llm_wrapper_preserves_processing_error_identity(wrapper):
    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
    )
    backend = _SequenceBackend([error])
    client = LLMClient(settings=_settings(structured_backend="instructor"), backend=backend)
    client._backend_kind = "instructor"
    client._limiter = _NoopLimiter()

    with pytest.raises(ProcessingError) as raised:
        await _call_public_wrapper(client, wrapper)

    assert raised.value is error


@pytest.mark.parametrize("failed_branch", ["title", "authors", "classification"])
async def test_core_metadata_fanout_preserves_processing_error_before_degradation(failed_branch):
    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
    )
    client = LLMClient(settings=_settings(structured_backend="instructor"))
    client.extract_title_keywords = mock.AsyncMock(
        return_value=TitleKeywordsLLM(title="Title", abstract="Abstract", keywords=[])
    )
    client.extract_authors = mock.AsyncMock(return_value=AuthorsLLM(authors=[]))
    client.extract_paper_classification = mock.AsyncMock(return_value=PaperClassificationLLM())
    {
        "title": client.extract_title_keywords,
        "authors": client.extract_authors,
        "classification": client.extract_paper_classification,
    }[failed_branch].side_effect = error

    with pytest.raises(ProcessingError) as raised:
        await client.extract_core_metadata("front matter")

    assert raised.value is error


async def test_author_fanout_preserves_processing_error_identity():
    # The client no longer re-rolls schema-valid empty authors (the extractor
    # owns that recovery); a typed processing failure raised by the primary
    # author call must still propagate with identity from the fan-out.
    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
    )
    client = LLMClient(settings=_settings(structured_backend="instructor"))
    client.extract_title_keywords = mock.AsyncMock(
        return_value=TitleKeywordsLLM(title="Title", abstract="Abstract", keywords=[])
    )
    client.extract_authors = mock.AsyncMock(side_effect=error)
    client.extract_paper_classification = mock.AsyncMock(
        return_value=PaperClassificationLLM(paper_type="article")
    )

    with pytest.raises(ProcessingError) as raised:
        await client.extract_core_metadata("front matter")

    assert raised.value is error


@pytest.mark.parametrize("where", ["native", "fallback"])
async def test_cancellation_during_protocol_execution_propagates(where):
    cancelled = asyncio.CancelledError()
    native = _SequenceBackend([cancelled] if where == "native" else [_invalid()])
    instructor = _SequenceBackend([cancelled] if where == "fallback" else [])
    client, _ = _client(native, instructor)

    with pytest.raises(asyncio.CancelledError):
        await _invoke(client)

    assert len(native.calls) == 1
    assert len(instructor.calls) == (1 if where == "fallback" else 0)
    assert client._breaker._failure_count == 0


@pytest.mark.parametrize("where", ["native", "fallback"])
async def test_cancellation_during_each_protocol_backoff_propagates(monkeypatch, where):
    async def cancel_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", cancel_sleep)
    native = _SequenceBackend([_Transient503("native")] if where == "native" else [_invalid()])
    instructor = _SequenceBackend([_Transient503("fallback")] if where == "fallback" else [])
    client, _ = _client(native, instructor)

    with pytest.raises(asyncio.CancelledError):
        await _invoke(client)

    assert len(native.calls) == 1
    assert len(instructor.calls) == (1 if where == "fallback" else 0)


def _model_result(response_model):
    if response_model is AuthorsLLM:
        return AuthorsLLM(authors=[])
    if response_model is CoreMetadataLLM:
        return CoreMetadataLLM(title="Title", abstract=None, keywords=[], authors=[])
    if "matches" in response_model.model_fields:
        return response_model(matches=[])
    raise AssertionError(response_model)


@pytest.mark.parametrize(
    ("invoke_name", "expected_effort"),
    [
        ("authors", "author-effort"),
        ("citations", "citation-effort"),
        ("merged", "author-effort"),
    ],
)
@pytest.mark.parametrize("protocol", ["nuextract-native", "instructor"])
async def test_task_reasoning_is_suppressed_only_for_native(invoke_name, expected_effort, protocol):
    class _TaskBackend(_SequenceBackend):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return _model_result(kwargs["response_model"]), None

    backend = _TaskBackend([])
    settings = _settings(
        structured_backend=protocol,
        reasoning_effort_authors="author-effort",
        reasoning_effort_citations="citation-effort",
    )
    client = LLMClient(settings=settings, backend=backend)
    client._backend_kind = protocol
    client._limiter = _NoopLimiter()

    if invoke_name == "authors":
        await client.extract_authors("byline")
    elif invoke_name == "citations":
        await client.resolve_citations(
            [(1, "[1]")],
            [{"bib_id": "b1", "author": "A", "year": "2024", "title": "T"}],
        )
    else:
        await client.extract_core_metadata_merged("front matter")

    assert len(backend.calls) == 1
    assert backend.calls[0]["reasoning_effort"] == (
        None if protocol == "nuextract-native" else expected_effort
    )


async def test_recovery_client_construction_failure_re_raises_native_without_physical_attempt():
    native_error = _invalid()
    native = _SequenceBackend([native_error])
    instructor = _SequenceBackend([])
    client, metrics = _client(native, instructor)
    client._get_json_mode_client.side_effect = RuntimeError("client build failed")

    with pytest.raises(ProcessingError) as raised:
        await _invoke(client)

    assert raised.value.__cause__ is native_error
    assert (len(native.calls), len(instructor.calls), metrics["attempts"]) == (1, 0, 1)
    assert (metrics["native_attempts"], metrics["instructor_attempts"]) == (1, 0)
    assert native_error.__cause__ is None
    assert native_error.__context__ is None


@pytest.mark.parametrize("protocol", ["nuextract-native", "instructor"])
async def test_lazy_client_construction_failure_does_not_increment_physical_attempts(
    protocol,
):
    settings = _settings(structured_backend=protocol)
    client = LLMClient(settings=settings)
    if protocol == "nuextract-native":
        backend = NuExtractNativeBackend(settings)
        backend._get_client = mock.Mock(side_effect=RuntimeError("client build failed"))
    else:
        backend = InstructorBackend(client)
        client._get_client = mock.Mock(side_effect=RuntimeError("client build failed"))
    client._backend = backend
    client._backend_kind = protocol
    metrics: Counter = Counter()
    record_label_metric = client._record_label_metric

    def record_both(name, value):
        metrics.update({name: value})
        record_label_metric(name, value)

    client._record_label_metric = record_both

    with pytest.raises(RuntimeError, match="client build failed"):
        await _invoke(client)

    assert metrics["attempts"] == 0
    assert metrics["native_attempts"] == 0
    assert metrics["instructor_attempts"] == 0


async def test_native_request_preparation_failure_does_not_count_attempt(monkeypatch):
    settings = _settings(structured_backend="nuextract-native")
    client = LLMClient(settings=settings)
    backend = NuExtractNativeBackend(settings)
    backend._get_client = mock.Mock(return_value=mock.MagicMock())
    client._backend = backend
    client._backend_kind = "nuextract-native"
    metrics = _capture_protocol_metrics(client)

    def fail_request_preparation(**_kwargs):
        raise RuntimeError("native request preparation failed")

    monkeypatch.setattr(
        "bibr.clients.nuextract.build_native_request_kwargs",
        fail_request_preparation,
    )

    with pytest.raises(RuntimeError, match="native request preparation failed"):
        await _invoke(client)

    assert metrics["attempts"] == 0
    assert metrics["native_attempts"] == 0
    assert metrics["instructor_attempts"] == 0
    assert metrics["retries"] == 0


@pytest.mark.parametrize("failure_site", ["provider", "messages", "kwargs"])
async def test_instructor_local_preparation_failure_does_not_count_attempt(
    monkeypatch,
    failure_site,
):
    settings = _settings(structured_backend="instructor")
    client = LLMClient(settings=settings)
    sdk_client = mock.MagicMock()
    sdk_client.create = mock.AsyncMock()
    client._get_client = mock.Mock(return_value=sdk_client)
    client._backend = InstructorBackend(client)
    client._backend_kind = "instructor"
    metrics = _capture_protocol_metrics(client)

    if failure_site == "provider":

        def fail_provider(_settings):
            raise RuntimeError("instructor provider preparation failed")

        monkeypatch.setattr("bibr.clients.llm._get_provider", fail_provider)
    elif failure_site == "messages":
        provider = mock.MagicMock()
        provider.transform_messages.side_effect = RuntimeError(
            "instructor message preparation failed"
        )
        monkeypatch.setattr(
            "bibr.clients.llm._get_provider",
            lambda _settings: (provider, None),
        )
    else:
        provider = object()
        monkeypatch.setattr(
            "bibr.clients.llm._get_provider",
            lambda _settings: (provider, None),
        )

        def fail_kwargs(**_kwargs):
            raise RuntimeError("instructor kwargs preparation failed")

        monkeypatch.setattr("bibr.clients.llm._build_call_kwargs", fail_kwargs)

    with pytest.raises(RuntimeError, match="instructor .* preparation failed"):
        await _invoke(client)

    assert metrics["attempts"] == 0
    assert metrics["native_attempts"] == 0
    assert metrics["instructor_attempts"] == 0
    assert metrics["retries"] == 0
    sdk_client.create.assert_not_awaited()


@pytest.mark.parametrize(
    "entry_error",
    [
        CircuitOpenError("llm", retry_after=30.0),
        asyncio.CancelledError(),
    ],
    ids=["breaker-rejection", "pre-call-cancellation"],
)
async def test_pre_call_exit_does_not_increment_physical_attempts(entry_error):
    class _RejectingGate:
        async def __aenter__(self):
            raise entry_error

        async def __aexit__(self, *_args):
            return False

    native = _SequenceBackend([_title("must not run")])
    client, metrics = _client(native, _SequenceBackend([]))
    client._breaker = _RejectingGate()

    with pytest.raises(type(entry_error)):
        await _invoke(client)

    assert native.calls == []
    assert metrics["attempts"] == 0
    assert metrics["native_attempts"] == 0
    assert metrics["instructor_attempts"] == 0
