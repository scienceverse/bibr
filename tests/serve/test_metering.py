"""Tests for per-request usage metering (D2) — HTTP middleware + predict record."""

import json
import logging

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _metering_app(monkeypatch, *, enabled: bool = True) -> FastAPI:
    """Minimal app replicating serve.app's metering middleware wiring."""
    import time
    import uuid

    from bibr.config import Settings
    from bibr.serve.app import metering_logger
    from bibr.serve.jobs import _sanitize_request_id

    monkeypatch.setattr(Settings.metering, "enabled", enabled)
    metering_logger.setLevel(logging.INFO)

    app = FastAPI()

    @app.middleware("http")
    async def _metering(request, call_next):
        if not Settings.metering.enabled or request.url.path in ("/health", "/ready"):
            return await call_next(request)
        request_id = _sanitize_request_id(request.headers.get("x-request-id")) or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = request_id
        response.headers["x-bibr-duration-ms"] = str(duration_ms)
        metering_logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                }
            )
        )
        return response

    @app.get("/echo")
    async def echo():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


class TestMeteringMiddleware:
    def test_generates_request_id_and_headers(self, monkeypatch):
        client = TestClient(_metering_app(monkeypatch))
        resp = client.get("/echo")
        assert resp.status_code == 200
        rid = resp.headers.get("x-request-id")
        assert rid and len(rid) <= 64
        assert resp.headers.get("x-bibr-duration-ms") is not None

    def test_honors_incoming_request_id(self, monkeypatch):
        client = TestClient(_metering_app(monkeypatch))
        resp = client.get("/echo", headers={"x-request-id": "my-req-42"})
        assert resp.headers.get("x-request-id") == "my-req-42"

    def test_sanitizes_request_id(self, monkeypatch):
        client = TestClient(_metering_app(monkeypatch))
        resp = client.get("/echo", headers={"x-request-id": "bad/id with spaces!"})
        assert resp.headers.get("x-request-id") == "badidwithspaces"

    def test_long_request_id_truncated(self, monkeypatch):
        client = TestClient(_metering_app(monkeypatch))
        resp = client.get("/echo", headers={"x-request-id": "a" * 200})
        assert resp.headers.get("x-request-id") == "a" * 64

    def test_log_record_emitted(self, monkeypatch, caplog):
        client = TestClient(_metering_app(monkeypatch))
        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            client.get("/echo", headers={"x-request-id": "req1"})
        records = [json.loads(r.getMessage()) for r in caplog.records]
        matched = [r for r in records if r.get("request_id") == "req1"]
        assert matched, "no metering record emitted"
        rec = matched[0]
        assert rec["method"] == "GET"
        assert rec["path"] == "/echo"
        assert rec["status"] == 200
        assert "duration_ms" in rec

    def test_build_server_record_has_integer_inference_outstanding(self, monkeypatch, caplog):
        import asyncio

        pytest.importorskip("litserve")
        from bibr.config import Settings
        from bibr.serve.app import build_server

        monkeypatch.setattr(Settings.auth, "api_key", None)
        monkeypatch.setattr(Settings.metering, "enabled", True)
        server = build_server()
        try:
            with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
                response = TestClient(server.app).get(
                    "/openapi.json",
                    headers={"x-request-id": "inference-count"},
                )
            assert response.status_code == 200
            records = [json.loads(record.getMessage()) for record in caplog.records]
            matched = [
                record for record in records if record.get("request_id") == "inference-count"
            ]
            assert matched
            assert isinstance(matched[0]["inference_outstanding"], int)
        finally:
            asyncio.run(server.app.state.inference_tracker.close())
            asyncio.run(server.app.state.upload_store.close())

    def test_health_skipped(self, monkeypatch, caplog):
        client = TestClient(_metering_app(monkeypatch))
        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            resp = client.get("/health")
        assert resp.headers.get("x-request-id") is None
        assert not caplog.records

    def test_unhandled_error_still_emits_500_metering_record(self, monkeypatch, caplog):
        """An exploding route must not vanish from usage/error dashboards."""
        import asyncio

        pytest.importorskip("litserve")
        from bibr.config import Settings
        from bibr.serve.app import build_server

        monkeypatch.setattr(Settings.auth, "api_key", None)
        monkeypatch.setattr(Settings.metering, "enabled", True)
        server = build_server()

        @server.app.get("/boom")
        async def boom():  # pyright: ignore[reportUnusedFunction]
            raise RuntimeError("worker exploded")

        try:
            client = TestClient(server.app, raise_server_exceptions=False)
            with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
                response = client.get("/boom", headers={"x-request-id": "boom-1"})
            assert response.status_code == 500
            records = [json.loads(record.getMessage()) for record in caplog.records]
            matched = [r for r in records if r.get("request_id") == "boom-1"]
            assert matched, "no metering record for the unhandled 500"
            assert matched[0]["status"] == 500
            assert matched[0]["path"] == "/boom"
        finally:
            asyncio.run(server.app.state.inference_tracker.close())
            asyncio.run(server.app.state.upload_store.close())

    def test_disabled_via_settings(self, monkeypatch, caplog):
        client = TestClient(_metering_app(monkeypatch, enabled=False))
        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            resp = client.get("/echo")
        assert resp.headers.get("x-request-id") is None
        assert not caplog.records


class TestFileSink:
    def test_file_sink_attached_once(self, monkeypatch, tmp_path):
        from bibr.config import Settings
        from bibr.serve.app import _configure_metering_logging, metering_logger

        log_path = str(tmp_path / "metering.jsonl")
        monkeypatch.setattr(Settings.metering, "log_path", log_path)
        before = list(metering_logger.handlers)
        try:
            _configure_metering_logging(Settings)
            _configure_metering_logging(Settings)  # idempotent
            sinks = [
                h
                for h in metering_logger.handlers
                if getattr(h, "_bibr_metering_sink", None) == log_path
            ]
            assert len(sinks) == 1
            metering_logger.info(json.dumps({"event": "test"}))
            for h in sinks:
                h.flush()
            with open(log_path) as f:
                assert '"event": "test"' in f.read()
        finally:
            for h in list(metering_logger.handlers):
                if h not in before:
                    metering_logger.removeHandler(h)
                    h.close()


# --------------------------------------------------------------------------- #
# Predict-level extraction record
# --------------------------------------------------------------------------- #


class TestPredictMetering:
    def _new_api(self, tmp_path, *, settings=None):
        from bibr.serve.deployments.pipeline import BibrPipelineAPI

        api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
        api._cache = None
        api._cache_inited = True
        return api

    async def test_success_record_has_tokens_total(self, monkeypatch, caplog, tmp_path):
        from bibr.pipeline.context import RunConfig

        api = self._new_api(tmp_path)

        class _FakePipeline:
            _config = RunConfig()

            async def process_file(self, filename, paper_id=None, content=None, config=None):
                return {
                    "info": {},
                    "extraction": {
                        "usage": {
                            "totals": {
                                "calls": 3,
                                "input_tokens": 100,
                                "cached_input_tokens": 0,
                                "output_tokens": 23,
                                "total_tokens": 123,
                            },
                            "breakdown": [],
                        }
                    },
                }

        api._pipeline = _FakePipeline()

        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            out = await api.predict(
                {
                    "filename": "a.pdf",
                    "content": b"%PDF-1.4",
                    "start_page": None,
                    "end_page": None,
                    "include_figures": False,
                    "include_regions": False,
                    "consolidate": None,
                    "refs": None,
                    "ref_seg": None,
                }
            )
        assert out["success"] is True
        recs = [
            json.loads(r.getMessage()) for r in caplog.records if r.name == "bibr.serve.metering"
        ]
        extract = [r for r in recs if r.get("event") == "extract"]
        assert len(extract) == 1
        rec = extract[0]
        assert rec["success"] is True
        assert rec["cache_hit"] is False
        assert rec["error_kind"] is None
        assert rec["llm_tokens_total"] == 123
        assert rec["filename"] == "a.pdf"
        # v11 renamed the per-model ``llm_usage`` key to ``llm_usage_totals``
        # (flat totals). The old key must be gone so consumers fail on a
        # missing key rather than silently misreading a changed shape.
        assert "llm_usage" not in rec
        assert rec["llm_usage_totals"] == {
            "calls": 3,
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 23,
            "total_tokens": 123,
        }

    async def test_failure_record_has_error_kind(self, monkeypatch, caplog, tmp_path):
        from bibr.pipeline.context import RunConfig

        api = self._new_api(tmp_path)

        class _FakePipeline:
            _config = RunConfig()

            async def process_file(self, filename, paper_id=None, content=None, config=None):
                from bibr.exceptions import ProcessingError

                raise ProcessingError("boom")

        api._pipeline = _FakePipeline()

        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            out = await api.predict(
                {
                    "filename": "a.pdf",
                    "content": b"%PDF-1.4",
                    "start_page": None,
                    "end_page": None,
                    "include_figures": False,
                    "include_regions": False,
                    "consolidate": None,
                    "refs": None,
                    "ref_seg": None,
                }
            )
        assert out["success"] is False
        recs = [
            json.loads(r.getMessage())
            for r in caplog.records
            if r.name == "bibr.serve.metering" and json.loads(r.getMessage()).get("event")
        ]
        extract = [r for r in recs if r.get("event") == "extract"]
        assert len(extract) == 1
        assert extract[0]["success"] is False
        assert extract[0]["error_kind"] == "processing"
        assert extract[0]["llm_tokens_total"] is None

    async def test_invalid_output_failure_record_has_only_safe_terminal_diagnostics(
        self,
        monkeypatch,
        caplog,
        tmp_path,
    ):
        from bibr.exceptions import (
            ProcessingError,
            SafeLlmDiagnostics,
            SafeLlmLabelDiagnostics,
        )
        from bibr.pipeline.context import RunConfig

        raw_sentinel = "RAW-COMPLETION-SENTINEL"
        diagnostics = SafeLlmDiagnostics(
            invalid_category="non_json",
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            cached_input_tokens=3,
            labels=(
                SafeLlmLabelDiagnostics.from_counts(
                    "extract_title_keywords",
                    {
                        "attempts": 2,
                        "native_attempts": 1,
                        "instructor_attempts": 1,
                        "protocol_fallbacks": 1,
                        "native_invalid_outputs": 1,
                        "native_invalid_non_json": 1,
                    },
                ),
            ),
        )
        api = self._new_api(tmp_path)

        class _FakePipeline:
            _config = RunConfig()

            async def process_file(self, filename, paper_id=None, content=None, config=None):
                assert raw_sentinel
                raise ProcessingError(
                    "LLM returned invalid structured output",
                    error_code="llm_invalid_output",
                    safe_diagnostics=diagnostics,
                )

        api._pipeline = _FakePipeline()

        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            out = await api.predict(
                {
                    "filename": "a.pdf",
                    "content": b"%PDF-1.4",
                    "start_page": None,
                    "end_page": None,
                    "include_figures": False,
                    "include_regions": False,
                    "consolidate": None,
                    "refs": None,
                    "ref_seg": None,
                }
            )

        extract = [
            json.loads(record.getMessage())
            for record in caplog.records
            if record.name == "bibr.serve.metering"
            and json.loads(record.getMessage()).get("event") == "extract"
        ]
        assert out["error_code"] == "llm_invalid_output"
        assert len(extract) == 1
        assert extract[0]["error_code"] == "llm_invalid_output"
        assert extract[0]["llm_tokens_total"] == 18
        assert extract[0]["llm_failure_diagnostics"]["invalid_category"] == "non_json"
        assert (
            extract[0]["llm_failure_diagnostics"]["llm_usage_by_label"]["extract_title_keywords"][
                "native_invalid_non_json"
            ]
            == 1
        )
        assert raw_sentinel not in json.dumps(out)
        assert raw_sentinel not in json.dumps(extract[0])
        assert raw_sentinel not in caplog.text

    def test_failure_meter_rejects_diagnostics_subclass_serializer(self, caplog):
        from bibr.config import GlobalSettings
        from bibr.exceptions import SafeLlmDiagnostics
        from bibr.serve.deployments.pipeline import _emit_extract_metric

        class UnsafeDiagnostics(SafeLlmDiagnostics):
            def to_dict(self):
                return {"raw": "RAW-COMPLETION-SENTINEL"}

        settings = GlobalSettings()
        settings.metering.enabled = True
        unsafe = UnsafeDiagnostics(invalid_category="non_json", total_tokens=19)

        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            _emit_extract_metric(
                file_hash="hash",
                filename="paper.pdf",
                duration_ms=1,
                cache_hit=False,
                success=False,
                error_kind="processing",
                result_json=None,
                settings=settings,
                error_code="llm_invalid_output",
                safe_diagnostics=unsafe,
            )

        record = json.loads(caplog.records[-1].getMessage())
        assert record["llm_tokens_total"] is None
        assert record["llm_failure_diagnostics"] is None
        assert "RAW-COMPLETION-SENTINEL" not in caplog.text

    async def test_disabled_emits_nothing(self, monkeypatch, caplog, tmp_path):
        from bibr.config import GlobalSettings
        from bibr.pipeline.context import RunConfig

        settings = GlobalSettings()
        settings.metering.enabled = False
        api = self._new_api(tmp_path, settings=settings)

        class _FakePipeline:
            _config = RunConfig()

            async def process_file(self, filename, paper_id=None, content=None, config=None):
                return {"info": {}}

        api._pipeline = _FakePipeline()
        with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
            await api.predict(
                {
                    "filename": "a.pdf",
                    "content": b"%PDF-1.4",
                    "start_page": None,
                    "end_page": None,
                    "include_figures": False,
                    "include_regions": False,
                    "consolidate": None,
                    "refs": None,
                    "ref_seg": None,
                }
            )
        extract = [r for r in caplog.records if r.name == "bibr.serve.metering"]
        assert not extract


# --- Metering log rotation (audit M5: disk-exhaustion DoS) --------------------


def test_metering_file_handler_rotates(monkeypatch, tmp_path):
    """The metering sink sits outside the auth gate, so unauthenticated spam must
    not grow the log without bound — it uses a size-capped RotatingFileHandler."""
    from logging.handlers import RotatingFileHandler

    from bibr.config import Settings
    from bibr.serve.app import _configure_metering_logging, metering_logger

    log_file = tmp_path / "meter.jsonl"
    monkeypatch.setattr(Settings.metering, "log_path", str(log_file))
    monkeypatch.setattr(Settings.metering, "log_max_bytes", 4096)
    monkeypatch.setattr(Settings.metering, "log_backup_count", 2)

    # Clear any handler left by another test so we assert on a fresh install.
    for h in list(metering_logger.handlers):
        metering_logger.removeHandler(h)

    _configure_metering_logging(Settings)

    sinks = [h for h in metering_logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(sinks) == 1
    assert sinks[0].maxBytes == 4096
    assert sinks[0].backupCount == 2
    # Idempotent: a second call must not stack a duplicate sink.
    _configure_metering_logging(Settings)
    assert len([h for h in metering_logger.handlers if isinstance(h, RotatingFileHandler)]) == 1

    for h in list(metering_logger.handlers):
        metering_logger.removeHandler(h)
        h.close()


# --------------------------------------------------------------------------- #
# Extract-record linkage (serve-8): request_id / job_id ride the descriptor
# --------------------------------------------------------------------------- #


def _staged_descriptor(tmp_path, *, request_id=None, job_id=None):
    import hashlib

    content = b"%PDF-1.4 fake"
    upload_id = "0" * 32
    (tmp_path / upload_id).write_bytes(content)
    descriptor = {
        "upload_id": upload_id,
        "filename": "a.pdf",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if request_id is not None:
        descriptor["request_id"] = request_id
    if job_id is not None:
        descriptor["job_id"] = job_id
    return descriptor


@pytest.mark.parametrize(
    ("raw_request", "raw_job", "want_request", "want_job"),
    [
        ("req-1", "job-2", "req-1", "job-2"),
        ("bad/id!", "", "badid", None),
        (123, None, None, None),
    ],
)
async def test_decode_request_threads_linkage_ids(
    tmp_path, raw_request, raw_job, want_request, want_job
):
    """The handoff descriptor's linkage ids reach the pipeline inputs."""
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    api = BibrPipelineAPI(upload_root=tmp_path)
    api._cache = None
    api._cache_inited = True
    inputs = await api.decode_request(
        _staged_descriptor(tmp_path, request_id=raw_request, job_id=raw_job)
    )
    assert inputs["request_id"] == want_request
    assert inputs["job_id"] == want_job


async def test_decode_request_without_linkage_ids_gives_nulls(tmp_path):
    """Guard: descriptors written before the linkage change still decode."""
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    api = BibrPipelineAPI(upload_root=tmp_path)
    api._cache = None
    api._cache_inited = True
    inputs = await api.decode_request(_staged_descriptor(tmp_path))
    assert inputs["request_id"] is None
    assert inputs["job_id"] is None


async def test_extract_record_links_request_and_job_ids(tmp_path, caplog):
    """The extract metering record carries the ids the route stored."""
    from bibr.pipeline.context import RunConfig
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    api = BibrPipelineAPI(upload_root=tmp_path)
    api._cache = None
    api._cache_inited = True

    class _FakePipeline:
        _config = RunConfig()

        async def process_file(self, filename, paper_id=None, content=None, config=None):
            return {"info": {}, "extraction": {}}

    api._pipeline = _FakePipeline()

    with caplog.at_level(logging.INFO, logger="bibr.serve.metering"):
        out = await api.predict(
            {
                "filename": "a.pdf",
                "content": b"%PDF-1.4",
                "start_page": None,
                "end_page": None,
                "include_figures": False,
                "include_regions": False,
                "consolidate": None,
                "refs": None,
                "ref_seg": None,
                "request_id": "req-9",
                "job_id": "job-7",
            }
        )
    assert out["success"] is True
    recs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "bibr.serve.metering"]
    extract = [r for r in recs if r.get("event") == "extract"]
    assert len(extract) == 1
    assert extract[0]["request_id"] == "req-9"
    assert extract[0]["job_id"] == "job-7"
