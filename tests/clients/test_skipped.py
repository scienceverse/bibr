"""The fast-mode client: falsy, quiet where a run only reads it, and a refusal to every call."""

import inspect

import pytest

from bibr.clients.llm_protocol import LlmClient
from bibr.clients.skipped import (
    LlmSkipped,
    SkippedLlmClient,
    is_skipped,
    note_skipped,
)
from bibr.paper_contents import PaperContents

_CALLS = [
    "extract_core_metadata",
    "extract_core_metadata_merged",
    "extract_title_keywords",
    "extract_authors",
    "extract_paper_classification",
    "label_paper_type",
    "extract_references",
    "extract_references_chunk",
    "segment_references",
    "extract_research_integrity",
    "resolve_citations",
    "extract_equations",
    "invoke_structured",
]


def test_the_client_is_falsy_so_the_existing_gates_skip():
    assert not SkippedLlmClient()
    assert is_skipped(SkippedLlmClient())
    assert not is_skipped(None)
    assert not is_skipped(object())


@pytest.mark.parametrize("name", _CALLS)
async def test_every_call_is_refused_by_name(name):
    method = getattr(SkippedLlmClient(), name)
    with pytest.raises(LlmSkipped, match=name):
        await method("text", file_hash="h", anything=1)


def test_a_refusal_is_a_runtime_error_so_the_sites_catch_it_as_a_failure():
    assert issubclass(LlmSkipped, RuntimeError)


def test_every_call_the_client_protocol_names_is_here():
    # A call added to the protocol without a refusal here would be an
    # AttributeError at the site, not a visible failure code.
    protocol = [name for name in vars(LlmClient) if not name.startswith("_")]
    assert protocol
    client = SkippedLlmClient()
    for name in protocol:
        assert hasattr(client, name), name


def test_what_a_run_reads_off_the_client_is_empty():
    client = SkippedLlmClient()
    assert client.usage == {}
    assert client.limiter is None
    assert client.json_mode_reroll_is_distinct() is False
    assert client.usage_for_file("h") == {}
    assert client.usage_pop_file("h") == {}
    assert client.usage_labels_pop_file("h") == {}
    assert client.protocol_hashes_pop_file("h") == {}
    assert client.traces_pop_file("h") == []
    client.record_reference_usage({"m": {"calls": 1}})
    assert client.usage == {}


async def test_the_lifecycle_calls_do_nothing():
    client = SkippedLlmClient()
    assert await client.ensure_server_context() is None
    assert await client.close() is None


def test_only_the_model_calls_are_coroutines_beside_the_lifecycle():
    # A stage that reads usage at the end of a run must not trip over the stand-in.
    coroutines = {
        name
        for name, member in inspect.getmembers(SkippedLlmClient, inspect.iscoroutinefunction)
        if not name.startswith("_")
    }
    assert coroutines == {*_CALLS, "ensure_server_context", "close"}


def test_note_skipped_adds_up_and_ignores_zero():
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    note_skipped(contents, "citations", 0)
    assert contents.llm_skipped == {}
    note_skipped(contents, "citations", 2)
    note_skipped(contents, "citations")
    note_skipped(contents, "integrity")
    assert contents.llm_skipped == {"citations": 3, "integrity": 1}
