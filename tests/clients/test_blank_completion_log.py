"""Blank-completion recognition + soft logging.

Small local models (vllm-mlx with advisory ``json_schema``) occasionally emit
an immediate EOS, so Instructor receives an empty string and raises a
``json_invalid`` ``ValidationError`` (``input_value=''``). That is model
flakiness, not a genuine fault — best-effort callers (e.g. the equation LLM
fallback) already degrade gracefully, so the failure should read as an
INFO-level skip, not a scary WARNING that looks like a real error. The skip is
still counted: the export's EQUATION_LLM_FALLBACK_FAILED warning names it.
"""

import logging
from unittest import mock

import pytest
from pydantic import ValidationError

from bibr.clients.llm import LLMClient, _is_blank_completion_error
from bibr.exceptions import LlmServiceError
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


async def test_extract_equations_raises_the_typed_failure():
    """The call raises; the equation extractor decides how loudly to log. A
    blank completion is the server failing to answer, so a service failure."""
    client = _client_with_limiter()
    client._invoke_structured = mock.AsyncMock(side_effect=_blank_json_error())

    with pytest.raises(LlmServiceError):
        await client.extract_equations([(1, "a sentence")], file_hash="h")


def _fallback_inputs():
    from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence

    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Results",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.RESULTS,
        ),
    ]
    sentences = [
        PaperSentence(
            text_id=10,
            text="Weird stat layout (t: 3.42, p: .003) regex misses.",
            section_id=1,
            paragraph_id=1,
            page_number=None,
        )
    ]
    return sentences, sections


async def test_equation_fallback_soft_logs_blank_completion(caplog):
    from bibr.extract.equation_extractor import EquationExtractor

    client = _client_with_limiter()
    client._invoke_structured = mock.AsyncMock(side_effect=_blank_json_error())
    extractor = EquationExtractor()

    with caplog.at_level(logging.INFO):
        await extractor.extract_with_llm_fallback(*_fallback_inputs(), llm_client=client)

    # No WARNING/ERROR — the empty completion is expected local-model flakiness —
    # but the failed batch is still recorded for the export's warning.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("failed for batch" in r.getMessage() for r in caplog.records)
    assert extractor.llm_batch_failures == ["llm_failed"]


async def test_equation_fallback_still_warns_on_real_failure(caplog):
    from bibr.extract.equation_extractor import EquationExtractor

    client = _client_with_limiter()
    client._invoke_structured = mock.AsyncMock(side_effect=RuntimeError("boom"))
    extractor = EquationExtractor()

    with caplog.at_level(logging.INFO, logger="bibr.extract.equation_extractor"):
        await extractor.extract_with_llm_fallback(*_fallback_inputs(), llm_client=client)

    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert extractor.llm_batch_failures == ["llm_failed"]
