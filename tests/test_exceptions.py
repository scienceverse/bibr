"""Tests for the exception hierarchy."""

import pytest

from bibr.exceptions import (
    SAFE_LLM_COUNTER_NAMES,
    BibrError,
    InputValidationError,
    ProcessingError,
    SafeLlmDiagnostics,
    SafeLlmLabelDiagnostics,
    UpstreamServiceError,
)


def test_bibr_error_is_exception():
    assert issubclass(BibrError, Exception)


def test_input_validation_error_inherits():
    err = InputValidationError("bad file")
    assert isinstance(err, BibrError)
    assert str(err) == "bad file"


def test_processing_error_inherits():
    err = ProcessingError("parse failed")
    assert isinstance(err, BibrError)
    assert str(err) == "parse failed"
    assert err.error_code is None
    assert err.failed_stage is None


def test_processing_error_carries_metadata():
    err = ProcessingError("parse failed", error_code="ocr_failed", failed_stage="ocr")
    assert err.error_code == "ocr_failed"
    assert err.failed_stage == "ocr"


def test_processing_error_carries_only_typed_safe_llm_diagnostics():
    label = SafeLlmLabelDiagnostics.from_counts(
        "extract_title_keywords",
        {
            "input_tokens": 11,
            "output_tokens": 7,
            "attempts": 2,
            "native_attempts": 1,
            "instructor_attempts": 1,
            "protocol_fallbacks": 1,
            "native_invalid_outputs": 1,
            "native_invalid_non_json": 1,
        },
    )
    diagnostics = SafeLlmDiagnostics(
        invalid_category="non_json",
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        cached_input_tokens=3,
        labels=(label,),
    )

    err = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
        safe_diagnostics=diagnostics,
    )

    assert err.safe_diagnostics is diagnostics
    assert diagnostics.to_dict() == {
        "invalid_category": "non_json",
        "llm_usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "cached_input_tokens": 3,
        },
        "llm_usage_by_label": {
            "extract_title_keywords": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "calls": 0,
                "logical_calls": 0,
                "attempts": 2,
                "retries": 0,
                "failed_calls": 0,
                "native_attempts": 1,
                "instructor_attempts": 1,
                "protocol_fallbacks": 1,
                "native_invalid_outputs": 1,
                "native_invalid_empty": 0,
                "native_invalid_non_json": 1,
                "native_invalid_truncated": 0,
                "native_invalid_trailing_content": 0,
                "native_invalid_non_object": 0,
                "native_invalid_schema_invalid": 0,
            }
        },
    }


@pytest.mark.parametrize("category", ["raw-response", "", "NON_JSON"])
def test_safe_llm_diagnostics_reject_unknown_categories(category):
    with pytest.raises(ValueError, match="invalid_category"):
        SafeLlmDiagnostics(invalid_category=category)


def test_safe_llm_diagnostics_reject_arbitrary_labels_and_non_integer_values():
    with pytest.raises(ValueError, match="label"):
        SafeLlmLabelDiagnostics.from_counts("RAW-USER-CONTENT", {"attempts": 1})
    with pytest.raises(TypeError, match="attempts"):
        SafeLlmLabelDiagnostics.from_counts(
            "extract_title_keywords",
            {"attempts": "RAW-USER-CONTENT"},
        )
    with pytest.raises(TypeError, match="safe_diagnostics"):
        ProcessingError("safe", safe_diagnostics={"raw": "RAW-USER-CONTENT"})


def test_safe_label_diagnostics_owns_immutable_counter_snapshot():
    mutable_counters = [[name, 0] for name in SAFE_LLM_COUNTER_NAMES]

    diagnostics = SafeLlmLabelDiagnostics(
        label="extract_title_keywords",
        counters=mutable_counters,
    )
    mutable_counters[0][1] = 999
    mutable_counters.append(["RAW-USER-CONTENT", "RAW-COMPLETION"])

    assert type(diagnostics.counters) is tuple
    assert all(type(item) is tuple for item in diagnostics.counters)
    assert diagnostics.to_dict()["input_tokens"] == 0
    assert "RAW-USER-CONTENT" not in diagnostics.to_dict()


def test_safe_diagnostics_rejects_nested_subclass_with_overridden_serializer():
    class UnsafeLabelDiagnostics(SafeLlmLabelDiagnostics):
        def to_dict(self):
            return {"raw": "RAW-COMPLETION-SENTINEL"}

    unsafe = UnsafeLabelDiagnostics.from_counts("extract_title_keywords", {})

    with pytest.raises(TypeError, match="SafeLlmLabelDiagnostics"):
        SafeLlmDiagnostics(invalid_category="non_json", labels=(unsafe,))


def test_processing_error_rejects_safe_diagnostics_subclass_on_init_and_assignment():
    class UnsafeDiagnostics(SafeLlmDiagnostics):
        def to_dict(self):
            return {"raw": "RAW-COMPLETION-SENTINEL"}

    unsafe = UnsafeDiagnostics(invalid_category="non_json")

    with pytest.raises(TypeError, match="safe_diagnostics"):
        ProcessingError("safe", safe_diagnostics=unsafe)

    error = ProcessingError("safe")
    with pytest.raises(TypeError, match="safe_diagnostics"):
        error.safe_diagnostics = unsafe


def test_upstream_service_error_attributes():
    original = ValueError("connection refused")
    err = UpstreamServiceError("Pandoc", "timed out", original_error=original)
    assert isinstance(err, BibrError)
    assert err.service_name == "Pandoc"
    assert err.original_error is original
    assert "Pandoc" in str(err)
    assert "timed out" in str(err)


def test_upstream_service_error_without_original():
    err = UpstreamServiceError("LLM API", "rate limited")
    assert err.original_error is None
    assert "LLM API" in str(err)


def test_exceptions_are_catchable_as_bibr_error():
    """All custom exceptions should be catchable via the base class."""
    errors = [
        InputValidationError("test"),
        ProcessingError("test"),
        UpstreamServiceError("svc", "msg"),
    ]
    for err in errors:
        try:
            raise err
        except BibrError:
            pass  # Should be caught
