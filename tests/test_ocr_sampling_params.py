"""OCR request sampling params must not use a sub-top-1 nucleus (top_p).

Regression for rapid-mlx NUL corruption: a ``top_p`` below the top token's
probability makes rapid-mlx's nucleus sampler mask every candidate and fall
back to token id 0, which decodes to a literal ``\\x00`` — corrupting every
OCR'd region (shredded table HTML, fused headings). Greedy decoding is
expressed via ``top_k: 1``; ``top_p`` must not be sent at all.
"""

import pytest


def test_http_ocr_payload_has_no_top_p():
    pytest.importorskip("httpx")
    from bibr.local.ocr import HttpOcrClient

    client = HttpOcrClient(base_url="http://localhost:1", model="glm-ocr")
    payload = client._build_payload("aW1n", "Text Recognition:")

    assert "top_p" not in payload
    assert payload["top_k"] == 1


def test_paddle_http_payload_uses_only_paddle_profile_request_settings():
    pytest.importorskip("httpx")
    from bibr.local.ocr import PaddleHttpOcrClient

    client = PaddleHttpOcrClient(base_url="http://localhost:1")
    payload = client._build_payload("aW1n", "OCR:")

    assert {key: payload[key] for key in ("model", "max_tokens", "temperature")} == {
        "model": "paddle-ocr-vl-1.6",
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    assert "top_k" not in payload
    assert "repetition_penalty" not in payload


def test_paddle_http_table_payload_uses_table_output_budget():
    pytest.importorskip("httpx")
    from bibr.local.ocr import PaddleHttpOcrClient

    client = PaddleHttpOcrClient(base_url="http://localhost:1")
    payload = client._build_payload("aW1n", "Table Recognition:")

    assert payload["max_tokens"] == 4096


def test_glm_http_payload_retains_glm_profile_request_settings():
    pytest.importorskip("httpx")
    from bibr.local.ocr import HttpOcrClient

    client = HttpOcrClient(base_url="http://localhost:1", model="glm-ocr")
    payload = client._build_payload("aW1n", "Text Recognition:")

    assert {
        key: payload[key]
        for key in ("model", "max_tokens", "temperature", "top_k", "repetition_penalty")
    } == {
        "model": "glm-ocr",
        "max_tokens": 16384,
        "temperature": 0.01,
        "top_k": 1,
        "repetition_penalty": 1.1,
    }


def test_paddle_http_default_model_agrees_with_runtime_identity():
    from bibr.config import GlobalSettings
    from bibr.ocr.profiles import resolve_ocr_runtime_identity
    from bibr.pipeline.context import RunConfig

    identity = resolve_ocr_runtime_identity(RunConfig(ocr_backend="paddle-http"), GlobalSettings())

    assert identity.model == "paddle-ocr-vl-1.6"
    assert identity.profile == "paddle"


def test_serve_ocr_backend_payload_has_no_top_p():
    pytest.importorskip("httpx")
    import inspect

    from bibr.serve import ocr_backend

    src = inspect.getsource(ocr_backend)
    assert '"top_p"' not in src, "serve OCR backend must not send top_p (rapid-mlx NUL bug)"


def test_local_ocr_module_sampling_params_have_no_top_p():
    import inspect

    from bibr.local import ocr

    src = inspect.getsource(ocr)
    assert '"top_p"' not in src, f"{ocr.__name__} must not send top_p (rapid-mlx NUL bug)"
