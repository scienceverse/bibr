"""Opt-in LLM trace capture (LLM_CAPTURE_TRACE) — Task 8.

Covers: the ``LlmTraceExport`` model shape, absence-by-default at both the
model and the pipeline (``_build_extraction``) level, the ``label`` join to
``extraction.usage.breakdown``, and the credential scrubber.
"""

from __future__ import annotations

import pytest

from bibr.export.models import ExtractionExport, LlmTraceExport
from tests.export.conftest import extraction_export


def test_trace_row_carries_sampling_params():
    """Params are mandatory: a completion recorded without them is untrustworthy
    as a training example — a good generation is indistinguishable from a lucky one."""
    row = LlmTraceExport(
        label="extract_core",
        provider="google",
        model="gemini-flash-lite",
        messages=[{"role": "user", "content": "hi"}],
        raw_completion='{"title": "x"}',
        parsed_ok=True,
        finish_reason="stop",
        params={"temperature": 0.0, "top_p": 1.0},
        attempt=1,
    )
    assert row.params["temperature"] == 0.0


def test_failed_attempt_records_the_error():
    row = LlmTraceExport(
        label="extract_core",
        messages=[],
        raw_completion="not json",
        parsed_ok=False,
        attempt=2,
        error="ValidationError: title missing",
    )
    assert row.parsed_ok is False
    assert "title missing" in row.error


def test_row_rejects_unknown_fields():
    """model_config = _STRICT everywhere: extra="forbid"."""
    with pytest.raises(Exception):  # noqa: PT011, B017 — pydantic ValidationError
        LlmTraceExport(label="x", parsed_ok=True, bogus_field=1)


def test_trace_labels_join_to_usage_breakdown_labels():
    """Task 3 changed the usage bucket key to the ``(label, provider, model)``
    triple (see tests/export/test_usage_export.py) — the brief's original
    example used a bare-string label key, which no longer matches
    ``build_usage_export``'s contract. Fixed to the real triple-key shape."""
    from bibr.export.usage import build_usage_export

    labels = {
        ("extract_core", "google", "m"): {
            "calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "cached_input_tokens": 0,
        }
    }
    usage = build_usage_export(labels)
    trace = [LlmTraceExport(label="extract_core", messages=[], parsed_ok=True, attempt=1)]
    assert {r["label"] for r in usage["breakdown"]} == {t.label for t in trace}


# ── absence-by-default ─────────────────────────────────────────────────


def test_trace_omitted_entirely_when_not_supplied():
    dumped = extraction_export().model_dump()
    assert "trace" not in dumped


def test_trace_omitted_when_explicitly_none():
    dumped = extraction_export(trace=None).model_dump()
    assert "trace" not in dumped


def test_trace_present_when_rows_supplied():
    dumped = extraction_export(
        trace=[LlmTraceExport(label="extract_core", parsed_ok=True)]
    ).model_dump()
    assert dumped["trace"][0]["label"] == "extract_core"


def test_settings_capture_trace_defaults_to_false():
    from bibr.config import Settings

    assert Settings.llm.capture_trace is False


def test_llm_capture_trace_env_var_enables_it(monkeypatch):
    from bibr.config import GlobalSettings

    monkeypatch.setenv("LLM_CAPTURE_TRACE", "true")
    # GlobalSettings (pydantic-settings) reads the environment at construction
    # time — no module reload needed, and reloading bibr.config would rebind
    # the Settings class for every already-imported module, corrupting
    # isinstance checks and monkeypatched state for the rest of the suite.
    assert GlobalSettings().llm.capture_trace is True


# ── _build_extraction wiring (bibr/pipeline/stages/export.py) ──────────


def _paper_with_trace(trace_rows):
    from pathlib import Path
    from unittest.mock import MagicMock

    from bibr.config import Settings
    from bibr.input.file import InputFile
    from bibr.models import PaperMetadata
    from bibr.paper import Paper
    from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.state import FileState

    contents = PaperContents(
        sentences=[PaperSentence(1, "Some text.", 1, 1, page_number=1)],
        sections=[
            PaperSection(0, "Root", 0, None, CanonicalSection.TITLE),
            PaperSection(1, "Title", 1, 0, CanonicalSection.TITLE),
        ],
        tables=[],
        links=[],
        sections_text={},
    )
    input_file = InputFile(path="paper.pdf")
    input_file.file_hash = "a" * 16
    paper = Paper(
        input_file=input_file,
        contents=contents,
        metadata=PaperMetadata(doi="", title="Paper"),
    )
    paper.llm_trace = trace_rows
    state = FileState(path=Path("paper.pdf"), paper=paper)
    ctx = PipelineContext(
        file_states=[state],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(no_llm=True),
        settings=Settings,
    )
    return ctx, paper


def test_paper_llm_trace_defaults_to_empty_list():
    from bibr.paper import Paper

    assert Paper.__dataclass_fields__["llm_trace"].default_factory() == []


def test_build_extraction_omits_trace_when_paper_llm_trace_is_empty():
    from bibr.pipeline.stages.export import _build_extraction

    ctx, paper = _paper_with_trace([])
    extraction = _build_extraction(ctx, paper)
    assert extraction["trace"] is None


def test_build_extraction_carries_trace_rows_through():
    from bibr.pipeline.stages.export import _build_extraction

    row = {
        "label": "extract_core",
        "provider": "google",
        "model": "gemini-flash-lite",
        "messages": [{"role": "user", "content": "hi"}],
        "raw_completion": '{"title": "x"}',
        "parsed_ok": True,
        "finish_reason": "stop",
        "params": {"temperature": 0.0},
        "attempt": 1,
        "error": None,
    }
    ctx, paper = _paper_with_trace([row])
    extraction = _build_extraction(ctx, paper)
    assert extraction["trace"] == [row]

    # Round-trips through the pydantic model too.
    validated = ExtractionExport(
        producer={"name": "bibr", "version": "0.0.0-test"},
        completed_at="2026-07-24T10:00:00Z",
        settings={
            "ref_seg": "geom",
            "ref_parse": "ner",
            "crossref_enrich": False,
            "consolidate": "off",
        },
        trace=extraction["trace"],
    )
    assert validated.model_dump()["trace"][0]["label"] == "extract_core"


def test_default_config_export_has_no_trace_key():
    """End-to-end: a Paper exported through the normal path (capture off,
    nothing captured) must not surface a ``trace`` key in the JSON at all."""
    from tests.export.conftest import extraction_block

    ctx, paper = _paper_with_trace([])
    paper.extraction = extraction_block()
    payload = paper.export_to_json()
    assert "trace" not in payload["extraction"]


# ── secret scrubbing ────────────────────────────────────────────────────


_SYNTHETIC_CREDENTIALS = [
    "sk-livekey123",
    "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890",
    "AIzaSyFAKEKEY",
    "AIza" + "SyntheticCredentialValue" * 2,
    "sv_1a2b3c4d5e6f7g8h9i0j",
    "gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234567890",
]


@pytest.mark.parametrize("secret", _SYNTHETIC_CREDENTIALS)
def test_captured_prompts_are_scrubbed_of_credentials(secret):
    from bibr.clients.llm import scrub_trace_text

    assert secret not in scrub_trace_text(f"use api key {secret} now")


def test_scrubber_redacts_multiple_secrets_in_one_blob():
    from bibr.clients.llm import scrub_trace_text

    secrets = ["sk-" + "test-value", "AIza" + "TestValue", "sv_" + "test-value"]
    text = " ".join(f"key{i}={value}" for i, value in enumerate(secrets))
    scrubbed = scrub_trace_text(text)
    for secret in secrets:
        assert secret not in scrubbed
    assert scrubbed.count("[REDACTED]") == 3


def test_scrubber_leaves_ordinary_text_untouched():
    from bibr.clients.llm import scrub_trace_text

    text = "Extract the title, authors, and DOI from this paper about skin-cancer."
    assert scrub_trace_text(text) == text


def test_scrubber_known_gap_aws_style_keys_are_not_caught():
    """Documents a deliberate limitation: the regex covers named provider
    prefixes (sk-/AIza/sv_/gsk_) actually used by this codebase's providers,
    not every credential shape in existence. An AWS-style access key ID
    (AKIA...) is NOT redacted — this is a known gap, not a silent failure,
    because bibr never issues or accepts AWS keys on this path today."""
    from bibr.clients.llm import scrub_trace_text

    aws_key = "AKIAIOSFODNN7EXAMPLE"
    assert scrub_trace_text(f"aws_key={aws_key}") == f"aws_key={aws_key}"
