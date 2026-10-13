import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from bibr.api import _refs_kwargs
from bibr.clients import ref_llm
from bibr.config import GlobalSettings, compute_behavior_fingerprint
from bibr.exceptions import ConfigurationError
from bibr.extract.ref_extractor import ReferenceExtractor, _resolve_ref_strategies
from bibr.models import PaperReference
from bibr.paper_contents import PaperContents


def record(**fields):
    return dict.fromkeys(ref_llm.FIELDS) | fields


def settings(**overrides):
    return GlobalSettings(
        REF_LLM_BASE_URL="http://reference.test/v1",
        REF_LLM_MODEL="trained-ref",
        **overrides,
    )


def server(monkeypatch, handler):
    factory = httpx.AsyncClient
    monkeypatch.setattr(
        ref_llm.httpx,
        "AsyncClient",
        lambda **kw: factory(
            transport=httpx.MockTransport(handler),
            **kw,
        ),
    )


def reply(request, fields, finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"content": json.dumps(fields)},
                }
            ]
        },
    )


@pytest.mark.parametrize(
    "fields",
    [
        record(title="Invented work"),
        record(title="Printed work", year=True),
        record(title="Printed work", year="2005"),
        record(title="Printed work", year=2006),
        record(title="Printed work", first_page="25", last_page="15"),
        record(title="Printed work", doi="bad/doi"),
        {"title": "Printed work"},
        record(title=["Printed work"]),
        record(),
        record(title="Printed work", extra="unexpected"),
    ],
)
def test_declines_unsupported_or_malformed_records(fields):
    assert (
        ref_llm.validate_record(json.dumps(fields), "Printed work. 2005. 25-15. bad/doi.") is None
    )


def test_unicode_and_abbreviated_pages_preserved():
    fields = record(
        authors="García, A.",
        title="Título español",
        year=2005,
        year_suffix="a",
        first_page="1234",
        last_page="56",
    )
    assert (
        ref_llm.validate_record(json.dumps(fields), "García, A. 2005a. Título español. 1234–56.")
        == fields
    )


async def test_dedicated_contract_alignment_and_truncation(monkeypatch):
    calls = []

    async def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        title = payload["messages"][1]["content"]
        if title == "Slow work":
            await asyncio.sleep(0.01)
        return reply(request, record(title=title), "length" if title == "Truncated" else "stop")

    server(monkeypatch, handler)
    result = await ref_llm.parse_batch(
        ["Slow work", "Fast work", "Truncated"],
        settings(
            REF_LLM_API_KEY="private-key",
        ),
    )
    assert [r["title"] if r else None for r in result] == ["Slow work", "Fast work", None]
    assert all(c["model"] == "trained-ref" and c["temperature"] == 0 for c in calls)
    assert calls[0]["messages"][0]["content"] == ref_llm.SYSTEM_PROMPT
    assert calls[0]["response_format"]["json_schema"]["schema"] == ref_llm.json_schema()
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_total_deadline_retains_completed_entries_and_cancels(monkeypatch):
    cancelled = asyncio.Event()

    async def handler(request):
        title = json.loads(request.content)["messages"][1]["content"]
        if title == "Pending":
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()
        return reply(request, record(title=title))

    server(monkeypatch, handler)
    result = await ref_llm.parse_batch(
        ["Complete", "Pending"],
        settings(
            REF_LLM_TOTAL_TIMEOUT_SECONDS=0.03,
        ),
    )
    assert result == [record(title="Complete"), None]
    assert cancelled.is_set()


async def test_outer_cancellation_propagates(monkeypatch):
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        await asyncio.sleep(10)

    server(monkeypatch, handler)
    task = asyncio.create_task(ref_llm.parse_batch(["Pending"], settings()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_oversized_inputs_and_http_errors_keep_slots(monkeypatch):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(503)

    server(monkeypatch, handler)
    assert await ref_llm.parse_batch(
        ["123456", "", "Short"],
        settings(
            REF_LLM_MAX_INPUT_CHARS=5,
        ),
    ) == [None, None, None]
    assert len(calls) == 1


async def test_unconfigured_endpoint_fails_explicitly():
    with pytest.raises(ConfigurationError, match="REF_LLM_BASE_URL"):
        await ref_llm.parse_batch(["Work"], GlobalSettings())


def reference(title):
    return PaperReference(
        bib_id=1, title=title, authors=None, year=None, container=None, first_page=None, volume=None
    )


async def test_pipeline_recovers_only_failed_positions(monkeypatch):
    async def parse_batch(strings, config, **kwargs):
        assert strings == ["First", "Second", "Third"]
        return [record(title="First"), None, record(title="Third", year_suffix=None)]

    monkeypatch.setattr(ref_llm, "parse_batch", parse_batch)
    contents = Mock(spec=PaperContents)
    contents.processing_warnings = []
    extractor = ReferenceExtractor(contents, settings=settings(), llm_client=Mock())
    fallback = Mock(return_value=[reference("Second")])
    monkeypatch.setattr(extractor, "_parse_references_ner_aligned", fallback)
    refs = await extractor._parse_references_ref_llm(["1. First", "2. Second", "3. Third"])
    assert [(ref.bib_id, ref.title) for ref in refs] == [(1, "First"), (2, "Second"), (3, "Third")]
    fallback.assert_called_once_with(["2. Second"])
    assert contents.processing_warnings[0].code == "REF_PARSE_NER_FALLBACK"


async def test_failed_ner_recovery_retains_model_records(monkeypatch):
    async def parse_batch(strings, config, **kwargs):
        return [None, record(title="Second")]

    monkeypatch.setattr(ref_llm, "parse_batch", parse_batch)
    contents = Mock(spec=PaperContents)
    contents.processing_warnings = []
    extractor = ReferenceExtractor(contents, settings=settings(), llm_client=Mock())
    monkeypatch.setattr(extractor, "_parse_references_ner_aligned", Mock(side_effect=RuntimeError))
    refs = await extractor._parse_references_ref_llm(["First", "Second"])
    assert [ref.title for ref in refs] == ["Second"]
    assert contents.processing_warnings[-1].code == "REF_PARSE_LOST"


def test_public_route_defaults_and_cache_identity():
    assert _refs_kwargs("ref-llm") == {"ref_parse_strategy": "ref-llm"}
    assert _resolve_ref_strategies(
        None, None, settings=GlobalSettings(REF_SEG_STRATEGY=None, REF_PARSE_STRATEGY=None)
    ) == ("geom", "ner")
    assert GlobalSettings.model_fields["REF_PARSE_STRATEGY"].default == "ner"
    a = settings(REF_LLM_API_KEY="secret-one")
    b = settings(REF_LLM_API_KEY="secret-two")
    assert compute_behavior_fingerprint(a) == compute_behavior_fingerprint(b)
    assert "secret-one" not in repr(a) and "secret-one" not in str(a.model_dump())
    b.REF_LLM_MODEL = "different-adapter"
    assert compute_behavior_fingerprint(a) != compute_behavior_fingerprint(b)


async def test_reference_usage_has_own_model_and_file_identity(monkeypatch):
    from bibr.clients.llm import LLMClient, usage_file_context

    config = settings(llm={"track_usage": True, "model": "metadata-model"})
    client = LLMClient(settings=config)

    async def handler(request):
        response = reply(request, record(title="Work"), finish="length")
        payload = response.json()
        payload["usage"] = {
            "prompt_tokens": 35,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 5},
        }
        return httpx.Response(200, json=payload)

    server(monkeypatch, handler)
    with usage_file_context("file-a"):
        result = await ref_llm.parse_batch(
            ["Work"], config, usage_recorder=client.record_reference_usage
        )
    assert result == [None]  # failed completions still consume tokens
    assert client.usage_for_file("file-a") == {
        "trained-ref": {
            "input_tokens": 35,
            "output_tokens": 7,
            "total_tokens": 42,
            "cached_input_tokens": 5,
        }
    }
    assert list(client.usage_labels_pop_file("file-a")) == [
        ("parse_reference_finetuned", "openai", "trained-ref")
    ]
    assert client.usage_for_file("file-b") == {}
    assert client._settings.llm.model == "metadata-model"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {"choices": [1]},
        {"choices": {"one": 1}},
        {"choices": [{"finish_reason": "stop", "message": None}]},
    ],
)
async def test_malformed_server_envelope_declines_entry(monkeypatch, payload):
    async def handler(request):
        return httpx.Response(
            200, content=json.dumps(payload), headers={"content-type": "application/json"}
        )

    server(monkeypatch, handler)
    assert await ref_llm.parse_batch(["Work"], settings()) == [None]


async def test_real_extraction_route_preserves_suffix_and_receipt(monkeypatch):
    import pandas as pd

    text = "García, A. (2005a). Título español. Revista 12:1234-56."
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    contents.native_ref_strings = [text]
    extractor = ReferenceExtractor(
        contents, settings=settings(), parse_strategy="ref-llm", llm_client=Mock()
    )

    async def parse_batch(strings, config, **kwargs):
        assert strings == [text]
        return [
            record(
                authors="García, A.",
                title="Título español",
                year=2005,
                year_suffix="a",
                volume="12",
                first_page="1234",
                last_page="56",
            )
        ]

    monkeypatch.setattr(ref_llm, "parse_batch", parse_batch)
    refs = await extractor.extract(pd.DataFrame({"text": [text]}))
    assert len(refs) == 1 and refs[0].year_suffix == "a"
    assert refs[0].last_page == "1256"
    assert contents.reference_yield_receipt.parsed_count == 1
    assert "ref_llm_parser_selected" in contents.reference_yield_receipt.reason_flags


async def test_notes_keep_source_owner_after_an_unparsed_entry(monkeypatch):
    from bibr.extract.footnote_citations import NoteCitation, NoteCitations

    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    extractor = ReferenceExtractor(
        contents, settings=settings(), parse_strategy="ref-llm", llm_client=Mock()
    )
    found = NoteCitations(
        (NoteCitation("First work", 40), NoteCitation("Second work", 60)),
        notes=2,
        citing_notes=2,
        repeats=0,
    )

    async def parse_batch(strings, config, **kwargs):
        return [None, record(title="Second work")]

    monkeypatch.setattr(ref_llm, "parse_batch", parse_batch)
    monkeypatch.setattr(extractor, "_parse_references_ner_aligned", Mock(return_value=[None]))
    refs = await extractor.extract_from_notes(found, [])
    assert [(ref.title, ref.text_id) for ref in refs] == [("Second work", 60)]
