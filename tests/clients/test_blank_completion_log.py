"""Blank-completion recognition + soft logging.

Small local models (vllm-mlx with advisory ``json_schema``) occasionally emit
an immediate EOS, so Instructor receives an empty string and raises a
``json_invalid`` ``ValidationError`` (``input_value=''``). That is model
flakiness, not a genuine fault — best-effort callers (e.g. the equation LLM
fallback) already degrade gracefully, so the failure should read as an
INFO-level skip, not a scary WARNING that looks like a real error.
"""

import logging
from unittest import mock

from pydantic import ValidationError

from bibr.clients.llm import LLMClient, _is_blank_completion_error
from bibr.schemas import EquationExtractionResult


def _blank_json_error() -> ValidationError:
    """A real ValidationError from parsing an empty completion, mirroring the
    error Instructor raises after re-asking a model that keeps returning ''."""
    try:
        EquationExtractionResult.model_validate_json("")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError from empty JSON")


def test_blank_completion_detected():
    assert _is_blank_completion_error(_blank_json_error()) is True


def test_blank_completion_detected_through_exception_chain():
    """Instructor may surface the parse failure wrapped in its own exception."""
    outer = RuntimeError("instructor retry wrapper")
    outer.__cause__ = _blank_json_error()
    assert _is_blank_completion_error(outer) is True


def test_malformed_nonblank_json_is_not_blank():
    """Non-empty-but-invalid output is a real fidelity fault, not an empty EOS."""
    try:
        EquationExtractionResult.model_validate_json("{not json")
    except ValidationError as exc:
        assert _is_blank_completion_error(exc) is False
    else:
        raise AssertionError("expected a ValidationError")


def test_structural_validation_error_is_not_blank():
    """A wrong-type field (real schema mismatch) must stay a loud failure."""
    try:
        EquationExtractionResult.model_validate({"equations": "not-a-list"})
    except ValidationError as exc:
        assert _is_blank_completion_error(exc) is False
    else:
        raise AssertionError("expected a ValidationError")


def test_generic_exception_is_not_blank():
    assert _is_blank_completion_error(RuntimeError("network down")) is False


def _client_with_limiter() -> LLMClient:
    client = LLMClient()
    lim = mock.Mock()
    lim.acquire = mock.AsyncMock()
    client._limiter = lim
    return client


async def test_extract_equations_soft_logs_blank_completion(caplog):
    client = _client_with_limiter()
    client._invoke_structured = mock.AsyncMock(side_effect=_blank_json_error())

    with caplog.at_level(logging.INFO, logger="bibr.clients.llm"):
        result = await client.extract_equations([(1, "a sentence")], file_hash="h")

    assert result == []
    # No WARNING/ERROR — the empty completion is expected local-model flakiness.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("empty completion" in r.getMessage().lower() for r in caplog.records)


async def test_extract_equations_still_warns_on_real_failure(caplog):
    client = _client_with_limiter()
    client._invoke_structured = mock.AsyncMock(side_effect=RuntimeError("boom"))

    with caplog.at_level(logging.INFO, logger="bibr.clients.llm"):
        result = await client.extract_equations([(1, "a sentence")], file_hash="h")

    assert result == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)
