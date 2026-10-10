"""Audit regressions for the serve error, MCP memory and timeout paths.

- A stage failure built from ``str(exc)`` (model-cache paths, library
  internals, upstream text) reaches the 422 body and the job error only as its
  stage and code; the server log keeps the detail.
- The MCP paper stores are bounded server-wide, not only per session.
- A caller-chosen file name cannot split or forge log lines.
- ``PIPELINE_TIMEOUT`` bounds the whole request, waits included.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import json
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("litserve")

from fastapi import HTTPException  # noqa: E402

# predict() imports this lazily; a cold import (over a second on a loaded
# machine) inside a test's short PIPELINE_TIMEOUT would spend the deadline.
import bibr.extract.ref_extractor  # noqa: E402, F401
from bibr.config import GlobalSettings  # noqa: E402
from bibr.exceptions import (  # noqa: E402
    ConfigurationError,
    ProcessingError,
    SafeLlmDiagnostics,
    UpstreamServiceError,
)
from bibr.pipeline.context import RunConfig  # noqa: E402
from bibr.pipeline.pipeline import Pipeline  # noqa: E402
from bibr.serve.deployments import pipeline as serve_pipeline  # noqa: E402
from bibr.serve.deployments.pipeline import BibrPipelineAPI, _KeyedAsyncLockPool  # noqa: E402

FIXTURE = Path(__file__).parent.parent / "fixtures" / "inspect_full_export.json"

# ---------------------------------------------------------------------------
# Client-safe processing errors
# ---------------------------------------------------------------------------

_CACHE_PATH = "/app/.hf_cache/models--PaddlePaddle--PP-DocLayoutV3/model.safetensors"


class _LayoutInitFailure:
    """Records a failure the way the layout stage does: ``f"...: {exc}"``."""

    name = "layout"

    async def run(self, ctx):
        try:
            raise OSError(f"[Errno 2] No such file or directory: '{_CACHE_PATH}'")
        except OSError as exc:
            for fs in ctx.file_states:
                fs.set_error(
                    f"Layout initialization failed: {exc}",
                    code="layout_failed",
                    stage=self.name,
                    exc=exc,
                )


def _pipeline(stages) -> Pipeline:
    return Pipeline(
        stages=stages,
        resources=MagicMock(),
        config=RunConfig(),
        settings=GlobalSettings(),
    )


async def _client_failure(pipeline: Pipeline, path: str, content: bytes | None = None) -> dict:
    with pytest.raises(ProcessingError) as raised:
        await pipeline.process_file(path, content=content)
    return BibrPipelineAPI._translate_error(path, raised.value)


async def test_stage_exception_text_stays_out_of_the_client_error(caplog):
    failure = await _client_failure(_pipeline([_LayoutInitFailure()]), "paper.pdf")

    assert failure["error_kind"] == "processing"
    assert failure["error_code"] == "layout_failed"
    assert failure["error"] == "Processing failed in layout (layout_failed)"
    # The operator still reads what went wrong.
    assert _CACHE_PATH in caplog.text


async def test_typed_parser_error_quoting_library_text_names_only_its_stage():
    """``DocxParser`` raises ``ProcessingError(f"Failed to open DOCX: {exc}")``
    from the zipfile error; the client learns the stage, not the library text."""
    from bibr.pipeline.stages.docx import DocxHandlingStage

    failure = await _client_failure(
        _pipeline([DocxHandlingStage()]), "paper.docx", content=b"not a zip archive"
    )

    assert failure["error_kind"] == "processing"
    assert failure["error"] == "Processing failed in docx"


@pytest.mark.parametrize(
    ("message", "code", "stage"),
    [
        ("File is password-protected", "encrypted_file", "validate"),
        ("Could not render any pages from this PDF", "layout_failed", "layout"),
    ],
)
async def test_failure_messages_bibr_wrote_itself_are_kept(message, code, stage):
    class _Stage:
        name = stage

        async def run(self, ctx):
            for fs in ctx.file_states:
                fs.set_error(message, code=code, stage=stage)

    failure = await _client_failure(_pipeline([_Stage()]), "paper.pdf")

    assert failure["error"] == message
    assert failure["error_code"] == code


async def test_configuration_error_beneath_a_stage_failure_stays_in_the_log(caplog):
    """A ConfigurationError is bibr's text, but written for the operator: it
    names the server's model bundle paths."""

    class _LayoutConfigFailure:
        name = "layout"

        async def run(self, ctx):
            exc = ConfigurationError(f"ONNX bundle {_CACHE_PATH} is missing model.onnx")
            for fs in ctx.file_states:
                fs.set_error(
                    f"Layout initialization failed: {exc}",
                    code="layout_failed",
                    stage=self.name,
                    exc=exc,
                )

    failure = await _client_failure(_pipeline([_LayoutConfigFailure()]), "paper.pdf")

    assert failure["error"] == "Processing failed in layout (layout_failed)"
    assert _CACHE_PATH in caplog.text


def test_invalid_output_error_keeps_its_message_over_its_diagnostic_cause():
    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
        safe_diagnostics=SafeLlmDiagnostics(invalid_category="non_json"),
    )
    error.__cause__ = RuntimeError("stands in for NuExtractInvalidOutput")

    failure = BibrPipelineAPI._translate_error("paper.pdf", error)

    assert failure["error"] == "LLM returned invalid structured output"
    assert failure["safe_diagnostics"]["invalid_category"] == "non_json"


def test_upstream_service_text_is_kept_with_endpoints_and_credentials_scrubbed():
    error = UpstreamServiceError(
        "ocr",
        "POST https://ocr.internal:30000/v1 failed with Bearer abcdefgh12345678 "
        "and key sk-ant-0123456789abcdefghij",
    )

    failure = BibrPipelineAPI._translate_error("paper.pdf", error)

    assert failure["error_kind"] == "upstream_service"
    assert failure["error"] == "Error in ocr: POST <url> failed with Bearer *** and key ***"


async def test_generic_failure_reaches_the_http_body_without_its_detail(tmp_path):
    api = BibrPipelineAPI(upload_root=tmp_path)
    api._cache = None
    api._cache_inited = True
    api._pipeline = _pipeline([_LayoutInitFailure()])

    output = await api.predict(_inputs("paper.pdf"))
    with pytest.raises(HTTPException) as raised:
        await api.encode_response(output)

    assert raised.value.status_code == 422
    assert raised.value.detail == {
        "message": "Processing failed in layout (layout_failed)",
        "error_code": "layout_failed",
    }


# ---------------------------------------------------------------------------
# File names in log lines
# ---------------------------------------------------------------------------


def _descriptor(tmp_path, filename: str, content: bytes = b"%PDF-1.4") -> dict:
    upload_id = uuid.uuid4().hex
    (tmp_path / upload_id).write_bytes(content)
    return {
        "upload_id": upload_id,
        "filename": filename,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_file_name_cannot_forge_a_log_line(caplog):
    forged = "a.pdf\n2026-10-06 ERROR bibr: forged\x1b[2K"

    BibrPipelineAPI._translate_error(forged, ProcessingError("failed", error_code="code"))

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "bibr.serve.deployments.pipeline"
    ]
    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert "\x1b" not in messages[0]
    assert "[a.pdf\\n2026-10-06 ERROR bibr: forged\\x1b[2K]" in messages[0]


async def test_decode_request_escapes_control_characters_in_the_file_name(tmp_path):
    """Every stage logs ``fs.path.name``: the name is made printable at the boundary."""
    api = BibrPipelineAPI(upload_root=tmp_path)

    out = await api.decode_request(_descriptor(tmp_path, "a\nb\tc d\x85.pdf"))

    assert out["filename"] == "a\\nb\\tc\\u2028d\\x85.pdf"


async def test_decode_request_keeps_ordinary_unicode_file_names(tmp_path):
    api = BibrPipelineAPI(upload_root=tmp_path)

    out = await api.decode_request(_descriptor(tmp_path, "Müller – Studie.pdf"))

    assert out["filename"] == "Müller – Studie.pdf"


# ---------------------------------------------------------------------------
# End-to-end PIPELINE_TIMEOUT
# ---------------------------------------------------------------------------


class _CountingPipeline:
    def __init__(self):
        self._config = RunConfig()
        self.calls = 0

    async def process_file(self, filename, paper_id, content, config, content_hash=None):  # noqa: ARG002
        self.calls += 1
        return {"paper_id": paper_id}


class _MissCache:
    async def get(self, key):  # noqa: ARG002
        return None

    async def set(self, key, value):  # noqa: ARG002
        return None

    async def delete(self, key):  # noqa: ARG002
        return None


class _ForeignLeaseCache(_MissCache):
    """Another worker owns the extraction lease and never publishes."""

    async def try_acquire_lease(self, key, *, ttl_seconds):  # noqa: ARG002
        return None

    async def lease_alive(self, key):  # noqa: ARG002
        return True


def _inputs(filename: str = "paper.pdf") -> dict:
    return {
        "filename": filename,
        "content": b"%PDF-1.4 same bytes",
        "start_page": None,
        "end_page": None,
        "include_figures": False,
        "include_regions": False,
        "consolidate": None,
    }


def _timed_api(tmp_path, *, timeout: float, cache=None, settings=None) -> BibrPipelineAPI:
    settings = settings or GlobalSettings()
    settings.pipeline.timeout = timeout
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
    api._cache = cache
    api._cache_inited = True
    api._pipeline = _CountingPipeline()
    api._inflight_sem = None
    return api


async def test_waiting_for_an_inflight_slot_counts_against_the_timeout(tmp_path):
    api = _timed_api(tmp_path, timeout=0.2)
    api._inflight_sem = asyncio.Semaphore(1)
    await api._inflight_sem.acquire()  # a run that never ends holds the only slot

    with pytest.raises(HTTPException) as raised:
        await asyncio.wait_for(api.predict(_inputs()), timeout=3)

    assert raised.value.status_code == 504
    assert api._pipeline.calls == 0


async def test_waiting_for_an_identical_extraction_counts_against_the_timeout(
    tmp_path, monkeypatch
):
    api = _timed_api(tmp_path, timeout=0.2, cache=_MissCache())
    api._cache_flights = _KeyedAsyncLockPool()
    monkeypatch.setattr(api, "_cache_key", lambda *args, **kwargs: "key")
    holding, release = asyncio.Event(), asyncio.Event()

    async def identical_request_in_flight():
        async with api._cache_flights.hold("key"):
            holding.set()
            await release.wait()

    other = asyncio.create_task(identical_request_in_flight())
    await holding.wait()
    try:
        with pytest.raises(HTTPException) as raised:
            await asyncio.wait_for(api.predict(_inputs()), timeout=3)
    finally:
        release.set()
        await other

    assert raised.value.status_code == 504
    assert api._pipeline.calls == 0
    # The timed-out waiter left no lock entry behind.
    assert api._cache_flights._entries == {}


@pytest.mark.parametrize("explicit_wait", [None, 60.0])
async def test_distributed_wait_ends_at_the_request_deadline(tmp_path, monkeypatch, explicit_wait):
    """No duplicate extraction starts once the request's time is spent waiting,
    and an explicit SINGLEFLIGHT_WAIT_SECONDS cannot outlast PIPELINE_TIMEOUT."""
    settings = GlobalSettings()
    settings.cache.distributed_singleflight = True
    settings.cache.singleflight_poll_interval_ms = 10
    if explicit_wait is None:
        settings.cache.model_fields_set.discard("singleflight_wait_seconds")
    else:
        settings.cache.singleflight_wait_seconds = explicit_wait
    api = _timed_api(tmp_path, timeout=0.3, cache=_ForeignLeaseCache(), settings=settings)
    outcomes: list[str] = []
    monkeypatch.setattr(
        serve_pipeline,
        "_emit_singleflight_metric",
        lambda outcome, **_kwargs: outcomes.append(outcome),
    )

    with pytest.raises(HTTPException) as raised:
        await asyncio.wait_for(api.predict(_inputs()), timeout=3)

    assert raised.value.status_code == 504
    assert api._pipeline.calls == 0
    # No fallback extraction was reported, since none ran.
    assert "timeout_fallback" not in outcomes


async def test_request_within_the_timeout_still_succeeds(tmp_path):
    api = _timed_api(tmp_path, timeout=60, cache=_MissCache())
    api._inflight_sem = asyncio.Semaphore(1)

    result = await api.predict(_inputs())

    assert result["success"] is True
    assert api._pipeline.calls == 1


# ---------------------------------------------------------------------------
# MCP paper memory bounded server-wide
# ---------------------------------------------------------------------------


@pytest.fixture
def serve_mcp():
    pytest.importorskip("mcp")
    import bibr.serve.mcp as serve_mcp

    return serve_mcp


class _Client:
    """Weakref-able stand-in for a session's InitializeRequestParams."""


def _ctx(client: _Client):
    return SimpleNamespace(session=SimpleNamespace(client_params=client))


def _paper(paper_id: str, chars: int) -> dict:
    return {"paper_id": paper_id, "text": [{"text": "x" * chars}]}


def _ids(store) -> list[str]:
    return [paper_id for paper_id, _entry in store.items()]


def test_default_bounds_cap_the_sessions_one_client_can_open(serve_mcp):
    stores = serve_mcp._SessionStores(16)
    clients = [_Client() for _ in range(200)]

    for index, client in enumerate(clients):
        stores.resolve(_ctx(client)).add(_paper(f"p{index}", 10), source=f"{index}.pdf")

    assert len(stores._stores) == 64  # the documented default
    assert len(_ids(stores.resolve(_ctx(clients[-1])))) == 1


def test_sessions_beyond_the_cap_drop_the_least_recently_used(serve_mcp):
    stores = serve_mcp._SessionStores(max_papers=16, max_sessions=3)
    clients = [_Client() for _ in range(20)]
    for index, client in enumerate(clients):
        store = stores.resolve(_ctx(client))
        for n in range(16):
            store.add(_paper(f"p{index}-{n}", 100), source=f"{index}-{n}.pdf")
    stores.resolve(_ctx(clients[17]))  # a call refreshes its session

    assert len(stores._stores) == 3
    retained = [
        entry.data for _ref, store in stores._stores.values() for _id, entry in store.items()
    ]
    assert len(retained) == 3 * 16
    assert stores.stored_bytes == sum(serve_mcp._export_size(data) for data in retained)
    # A new session evicts 18's store (least recently used), not 17's.
    stores.resolve(_ctx(clients[0]))
    assert len(_ids(stores.resolve(_ctx(clients[17])))) == 16
    assert _ids(stores.resolve(_ctx(clients[18]))) == []


def test_stored_bytes_beyond_the_budget_drop_the_least_recently_used_paper(serve_mcp):
    one = serve_mcp._export_size(_paper("a", 1000))
    stores = serve_mcp._SessionStores(max_papers=16, max_bytes=3 * one)
    first_client, second_client = _Client(), _Client()
    first, second = stores.resolve(_ctx(first_client)), stores.resolve(_ctx(second_client))
    first.add(_paper("a", 1000), source="a.pdf")
    first.add(_paper("b", 1000), source="b.pdf")
    second.add(_paper("c", 1000), source="c.pdf")
    first.get("a")  # a read makes "a" the most recently used

    second.add(_paper("d", 1000), source="d.pdf")

    assert _ids(first) == ["a"]
    assert _ids(second) == ["c", "d"]
    assert stores.stored_bytes == 3 * one


def test_a_paper_over_the_whole_budget_is_still_kept_alone(serve_mcp):
    stores = serve_mcp._SessionStores(max_papers=16, max_bytes=100)
    client = _Client()
    store = stores.resolve(_ctx(client))
    store.add(_paper("small", 10), source="small.pdf")

    store.add(_paper("big", 1000), source="big.pdf")

    assert _ids(store) == ["big"]


def test_accounting_follows_rechews_session_evictions_and_closed_sessions(serve_mcp):
    stores = serve_mcp._SessionStores(max_papers=2)
    client = _Client()
    store = stores.resolve(_ctx(client))
    for paper_id in ("a", "b", "c"):
        store.add(_paper(paper_id, 500), source=f"{paper_id}.pdf")
    store.add(_paper("c", 900), source="c.pdf")  # a re-chew replaces its entry

    assert _ids(store) == ["b", "c"]
    assert stores.stored_bytes == serve_mcp._export_size(_paper("b", 500)) + serve_mcp._export_size(
        _paper("c", 900)
    )

    del store, client
    gc.collect()
    assert stores._stores == {}
    assert stores.stored_bytes == 0


async def test_chew_paper_sessions_share_one_server_wide_bound(serve_mcp):
    from mcp import Client
    from mcp.client._memory import InMemoryTransport

    from bibr.serve.ingress import UploadStore

    class _Tracker:
        async def submit(self, descriptor, request_state=None, *, admission=None):  # noqa: ARG002
            return json.loads(FIXTURE.read_text())

    def _payload(result) -> dict:
        assert not result.is_error, [c.text for c in result.content]
        payload = result.structured_content
        return payload["result"] if set(payload) == {"result"} else payload

    upload_store = UploadStore.create(
        max_size=1_000_000, spool_memory_bytes=1024, stale_after_seconds=60
    )
    try:
        server = serve_mcp.build_serve_mcp(
            upload_store=upload_store,
            tracker=_Tracker(),
            max_papers_per_session=4,
            max_sessions=2,
        )
        upload = {"filename": "p.pdf", "content_base64": base64.b64encode(b"%PDF-1.4").decode()}
        async with AsyncExitStack() as stack:
            clients = [
                await stack.enter_async_context(Client(InMemoryTransport(server), mode="legacy"))
                for _ in range(3)
            ]
            paper_ids = [
                _payload(await client.call_tool("chew_paper", upload))["paper_id"]
                for client in clients
            ]

            newest = await clients[2].call_tool("get_paper_summary", {"paper_id": paper_ids[2]})
            oldest = await clients[0].call_tool("get_paper_summary", {"paper_id": paper_ids[0]})

        assert not newest.is_error
        assert oldest.is_error
        assert "unknown paper_id" in oldest.content[0].text
    finally:
        await upload_store.close()
