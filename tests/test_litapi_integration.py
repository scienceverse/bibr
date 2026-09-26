"""Unit tests for BibrPipelineAPI request/response handling.

Covers ``decode_request`` parsing of disk-backed upload descriptors and
``encode_response`` error-code mapping — the glue that LitServe uses to
bridge HTTP and the inner pipeline.
"""

import hashlib
import logging
import uuid
from unittest import mock

import pytest

pytest.importorskip("litserve")


def _new_api(tmp_path, *, settings=None):
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    return BibrPipelineAPI(upload_root=tmp_path, settings=settings)


def _descriptor(tmp_path, content=b"%PDF-1.4", filename="a.pdf", **fields):
    upload_id = uuid.uuid4().hex
    (tmp_path / upload_id).write_bytes(content)
    return {
        "upload_id": upload_id,
        "filename": filename,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        **fields,
    }


class TestDecodeRequest:
    async def test_missing_descriptor_raises_sanitized_500(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request({})
        assert exc.value.status_code == 500
        assert exc.value.detail == "Upload handoff failed"

    async def test_empty_descriptor_filename_raises_sanitized_500(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, filename=""))
        assert exc.value.status_code == 500
        assert exc.value.detail == "Upload handoff failed"

    async def test_oversized_descriptor_raises_sanitized_500(self, tmp_path, monkeypatch):
        from fastapi import HTTPException

        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "max_file_size", 10)
        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, b"x" * 100, "big.pdf"))
        assert exc.value.status_code == 500
        assert exc.value.detail == "Upload handoff failed"

    async def test_file_at_exact_limit_is_accepted(self, tmp_path, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "max_file_size", 10)
        api = _new_api(tmp_path)

        result = await api.decode_request(_descriptor(tmp_path, b"x" * 10, "at-limit.pdf"))

        assert result["content"] == b"x" * 10

    async def test_parses_page_range_and_flags(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.decode_request(
            _descriptor(
                tmp_path,
                start_page="2",
                end_page="7",
                include_figures="true",
            )
        )
        assert out["filename"] == "a.pdf"
        assert out["content"] == b"%PDF-1.4"
        assert out["start_page"] == 2
        assert out["end_page"] == 7
        assert out["include_figures"] is True
        assert out["include_regions"] is False

    async def test_parses_include_regions_flag(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.decode_request(_descriptor(tmp_path, include_regions="true"))
        assert out["include_regions"] is True

    async def test_parses_consolidate_field(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.decode_request(_descriptor(tmp_path, consolidate="FILL"))
        assert out["consolidate"] == "fill"

    async def test_consolidate_absent_is_none(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.decode_request(_descriptor(tmp_path))
        assert out["consolidate"] is None

    async def test_invalid_consolidate_raises_400(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, consolidate="merge"))
        assert exc.value.status_code == 400
        assert "fill" in exc.value.detail and "replace" in exc.value.detail

    async def test_parses_refs_field(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.decode_request(_descriptor(tmp_path, refs="OFF"))
        assert out["refs"] == "off"

    async def test_parses_ref_seg_field(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.decode_request(_descriptor(tmp_path, ref_seg="Region"))
        assert out["ref_seg"] == "region"

    async def test_refs_and_ref_seg_absent_are_none(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.decode_request(_descriptor(tmp_path))
        assert out["refs"] is None
        assert out["ref_seg"] is None

    async def test_invalid_refs_raises_400(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, refs="bogus"))
        assert exc.value.status_code == 400
        assert "off" in exc.value.detail

    async def test_invalid_ref_seg_raises_400(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, ref_seg="bogus"))
        assert exc.value.status_code == 400
        assert "geom" in exc.value.detail

    async def test_rejects_inverted_page_range(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, b"%PDF", start_page="10", end_page="2"))
        assert exc.value.status_code == 400

    @pytest.mark.parametrize("field", ["start_page", "end_page"])
    async def test_rejects_malformed_page_number(self, tmp_path, field):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, b"%PDF", **{field: "not-an-integer"}))
        assert exc.value.status_code == 400
        assert field in exc.value.detail

    @pytest.mark.parametrize("field", ["include_figures", "include_regions"])
    async def test_rejects_malformed_boolean(self, tmp_path, field):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, b"%PDF", **{field: "definitely"}))
        assert exc.value.status_code == 400
        assert field in exc.value.detail

    async def test_overlong_descriptor_filename_is_sanitized(self, tmp_path):
        """The worker rejects descriptor fields that bypass ingress bounds."""
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.decode_request(_descriptor(tmp_path, filename="a" * 5000 + ".pdf"))
        assert exc.value.status_code == 500
        assert exc.value.detail == "Upload handoff failed"

    async def test_decode_reads_descriptor_and_deletes_owned_file(self, tmp_path):
        api = _new_api(tmp_path)
        request = _descriptor(tmp_path, content=b"%PDF-1.4")
        path = tmp_path / request["upload_id"]

        result = await api.decode_request(request)

        assert result["content"] == b"%PDF-1.4"
        assert result["content_hash"] == request["sha256"]
        assert not path.exists()

    async def test_digest_mismatch_is_sanitized_500_without_temporary_path(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        request = _descriptor(tmp_path)
        request["sha256"] = "0" * 64

        with pytest.raises(HTTPException) as exc:
            await api.decode_request(request)

        assert exc.value.status_code == 500
        assert exc.value.detail == "Upload handoff failed"
        assert str(tmp_path) not in exc.value.detail


class TestEncodeResponse:
    async def test_success_passes_paper_json_through(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.encode_response(
            {"success": True, "paper_json": {"paper_id": "abc"}, "error": None, "error_kind": None}
        )
        assert out == {"paper_id": "abc"}

    async def test_success_returns_paper_json_without_bibr_release(self, tmp_path):
        api = _new_api(tmp_path)
        out = await api.encode_response(
            {
                "success": True,
                "paper_json": {"paper_id": "abc", "info": {"schema_version": "10.3"}},
                "error": None,
                "error_kind": None,
            }
        )
        # The package version now lives in the export's ``extraction.bibr_version``
        # (which reflects the producing bibr, even on cache hits). The serve layer
        # no longer stamps a redundant ``info.bibr_release``.
        assert out["info"]["schema_version"] == "10.3"
        assert "bibr_release" not in out["info"]

    async def test_processing_error_maps_to_422(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.encode_response(
                {
                    "success": False,
                    "paper_json": None,
                    "error": "bad parse",
                    "error_kind": "processing",
                }
            )
        assert exc.value.status_code == 422

    async def test_stable_processing_code_emits_structured_422_detail(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.encode_response(
                {
                    "success": False,
                    "paper_json": None,
                    "error": "LLM returned invalid structured output",
                    "error_kind": "processing",
                    "error_code": "llm_invalid_output",
                }
            )

        assert exc.value.status_code == 422
        assert exc.value.detail == {
            "message": "LLM returned invalid structured output",
            "error_code": "llm_invalid_output",
        }

    async def test_processing_failure_without_code_keeps_legacy_string_detail(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.encode_response(
                {
                    "success": False,
                    "paper_json": None,
                    "error": "bad parse",
                    "error_kind": "processing",
                }
            )

        assert exc.value.detail == "bad parse"

    async def test_input_validation_maps_to_400(self, tmp_path):
        from fastapi import HTTPException

        api = _new_api(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await api.encode_response(
                {
                    "success": False,
                    "paper_json": None,
                    "error": "bad format",
                    "error_kind": "input_validation",
                }
            )
        assert exc.value.status_code == 400


class TestErrorTranslation:
    def test_unexpected_error_returns_generic_message(self, caplog, tmp_path):
        api = _new_api(tmp_path)
        out = api._translate_error("x.pdf", RuntimeError("/tmp/secret-path/abc not found"))
        assert out["error_kind"] == "unexpected"
        assert "/tmp/secret-path" not in out["error"]
        assert out["error"] == "Internal processing error"
        # Full exception still in logs
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "/tmp/secret-path" in joined

    def test_known_error_kinds_return_their_message(self, tmp_path):
        from bibr.exceptions import InputValidationError

        api = _new_api(tmp_path)
        out = api._translate_error("x.pdf", InputValidationError("bad mime type"))
        assert out["error_kind"] == "input_validation"
        assert out["error"] == "bad mime type"

    def test_processing_error_preserves_stable_code_and_safe_diagnostics(self, tmp_path):
        from bibr.exceptions import ProcessingError, SafeLlmDiagnostics

        diagnostics = SafeLlmDiagnostics(
            invalid_category="schema_invalid",
            total_tokens=19,
        )
        error = ProcessingError(
            "LLM returned invalid structured output",
            error_code="llm_invalid_output",
            safe_diagnostics=diagnostics,
        )

        out = _new_api(tmp_path)._translate_error("x.pdf", error)

        assert out["error_kind"] == "processing"
        assert out["error_code"] == "llm_invalid_output"
        assert out["safe_diagnostics"] == diagnostics.to_dict()

    @pytest.mark.parametrize(
        ("error_class", "kind", "code", "status"),
        [
            ("LlmTruncatedError", "processing", "llm_truncated", 422),
            ("LlmInvalidOutputError", "processing", "llm_invalid_output", 422),
            ("LlmTimeoutError", "upstream_service", "llm_timeout", 502),
            ("LlmServiceError", "upstream_service", "llm_failed", 502),
            ("LlmUnreachableError", "upstream_service", "llm_failed", 502),
            ("LlmRejectedError", "upstream_service", "llm_failed", 502),
            ("LlmCallError", "upstream_service", "llm_failed", 502),
        ],
    )
    async def test_typed_llm_failures_keep_their_code(
        self, tmp_path, error_class, kind, code, status
    ):
        """A truncated or invalid response is deterministic: 422, so clients
        and ``bibr batch --remote`` do not retry it as an outage."""
        from fastapi import HTTPException

        from bibr import exceptions

        error = getattr(exceptions, error_class)("Failed to extract references", cause="why")
        api = _new_api(tmp_path)
        out = api._translate_error("x.pdf", error)

        assert out["error_kind"] == kind
        assert out["error_code"] == code
        assert out["error"] == "Error in LLM: Failed to extract references (why)"
        with pytest.raises(HTTPException) as raised:
            await api.encode_response(out)
        assert raised.value.status_code == status
        assert raised.value.detail == {"message": out["error"], "error_code": code}


def test_real_fastapi_body_nests_structured_processing_detail(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    api = _new_api(tmp_path)
    app = FastAPI()

    @app.get("/failure")
    async def failure():
        return await api.encode_response(
            {
                "success": False,
                "paper_json": None,
                "error": "LLM returned invalid structured output",
                "error_kind": "processing",
                "error_code": "llm_invalid_output",
            }
        )

    response = TestClient(app).get("/failure")

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "message": "LLM returned invalid structured output",
            "error_code": "llm_invalid_output",
        }
    }


def test_owned_client_close_failure_cannot_mask_typed_error_over_real_http(caplog, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.exceptions import ProcessingError, SafeLlmDiagnostics
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.stages.post_parse import post_parse

    raw_sentinel = "RAW-CLOSE-FAILURE-SENTINEL"
    typed_error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
        safe_diagnostics=SafeLlmDiagnostics(invalid_category="non_json"),
    )
    owned_client = mock.MagicMock()
    owned_client._track_usage = False
    owned_client.usage_pop_file.return_value = {}
    owned_client.usage_labels_pop_file.return_value = {}
    owned_client.close = mock.AsyncMock(side_effect=RuntimeError(raw_sentinel))
    api = _new_api(tmp_path)
    api._cache = None
    api._cache_inited = True

    class _FakePipeline:
        _config = RunConfig()

        async def process_file(self, filename, paper_id=None, content=None, config=None):
            return await post_parse(
                contents=mock.MagicMock(),
                file_name=filename,
                file_hash=paper_id or "hash",
                llm_client=None,
                settings=api._settings,
            )

    api._pipeline = _FakePipeline()
    app = FastAPI()

    @app.post("/extract")
    async def extract():
        output = await api.predict(
            {
                "filename": "paper.pdf",
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
        return await api.encode_response(output)

    with (
        mock.patch("bibr.clients.llm.LLMClient", return_value=owned_client),
        mock.patch(
            "bibr.pipeline.stages.post_parse._classify_sections",
            mock.AsyncMock(side_effect=typed_error),
        ),
        caplog.at_level(logging.WARNING),
    ):
        response = TestClient(app).post("/extract")

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "message": "LLM returned invalid structured output",
            "error_code": "llm_invalid_output",
        }
    }
    owned_client.close.assert_awaited_once()
    assert raw_sentinel not in response.text
    assert raw_sentinel not in caplog.text


class TestCancellation:
    async def test_cancellation_propagates_as_cancellation(self, monkeypatch, tmp_path):
        """CancelledError must NOT be re-raised as a 504 timeout."""
        import asyncio

        from bibr.pipeline.context import RunConfig

        api = _new_api(tmp_path)
        api._cache = None
        api._cache_inited = True
        api._pipeline = mock.MagicMock()
        api._pipeline._config = RunConfig()

        async def cancelled(*a, **kw):
            raise asyncio.CancelledError()

        api._pipeline.process_file = cancelled

        with pytest.raises(asyncio.CancelledError):
            await api.predict(
                {
                    "filename": "x.pdf",
                    "content": b"%PDF-1.4",
                    "start_page": None,
                    "end_page": None,
                    "include_figures": False,
                    "include_regions": False,
                    "consolidate": None,
                }
            )


def test_cache_key_includes_consolidate():
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    base = BibrPipelineAPI._cache_key("abc", None, None, False, False, None)
    fill = BibrPipelineAPI._cache_key("abc", None, None, False, False, "fill")
    replace = BibrPipelineAPI._cache_key("abc", None, None, False, False, "replace")
    off = BibrPipelineAPI._cache_key("abc", None, None, False, False, "off")
    assert len({base, fill, replace}) == 3
    assert off == base  # "off" adds no segment


def test_cache_key_distinguishes_refs_and_ref_seg():
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    base = BibrPipelineAPI._cache_key("abc", None, None, False, False, None)
    refs_off = BibrPipelineAPI._cache_key("abc", None, None, False, False, None, refs="off")
    refs_llm = BibrPipelineAPI._cache_key("abc", None, None, False, False, None, refs="llm")
    seg_region = BibrPipelineAPI._cache_key("abc", None, None, False, False, None, ref_seg="region")
    # Every distinct effective (refs, ref_seg) must key to a distinct entry so
    # a refs=off pass can't return a cached full-reference result.
    assert len({base, refs_off, refs_llm, seg_region}) == 4


async def test_predict_passes_effective_consolidate_to_runconfig(tmp_path):
    from bibr.pipeline.context import RunConfig

    api = _new_api(tmp_path)
    api._cache = None
    api._cache_inited = True
    captured = {}

    class _FakePipeline:
        _config = RunConfig()

        async def process_file(self, filename, paper_id=None, content=None, config=None):
            captured["config"] = config
            return {"info": {}}

    api._pipeline = _FakePipeline()
    out = await api.predict(
        {
            "filename": "a.pdf",
            "content": b"%PDF-1.4",
            "start_page": None,
            "end_page": None,
            "include_figures": False,
            "include_regions": False,
            "consolidate": "replace",
            "refs": None,
            "ref_seg": None,
        }
    )
    assert captured["config"].consolidate == "replace"
    assert out["success"] is True


async def test_predict_passes_refs_and_ref_seg_to_runconfig(tmp_path):
    from bibr.pipeline.context import RunConfig

    api = _new_api(tmp_path)
    api._cache = None
    api._cache_inited = True
    captured = {}

    class _FakePipeline:
        _config = RunConfig()

        async def process_file(self, filename, paper_id=None, content=None, config=None):
            captured["config"] = config
            return {"info": {}}

    api._pipeline = _FakePipeline()
    await api.predict(
        {
            "filename": "a.pdf",
            "content": b"%PDF-1.4",
            "start_page": None,
            "end_page": None,
            "include_figures": False,
            "include_regions": False,
            "consolidate": None,
            "refs": "off",
            "ref_seg": "region",
        }
    )
    assert captured["config"].ref_parse_strategy == "off"
    assert captured["config"].ref_seg_strategy == "region"
