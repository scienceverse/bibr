"""Pre-spawn port guard for managed local servers.

A stale local server left listening on a managed port must never be silently
reused for a different model: health polling alone cannot tell "our subprocess
came up" from "some leftover process answered". The guard probes ``/v1/models``
on the configured port before spawning and:

- reuses a listener that already serves the requested model,
- fails fast (naming both models) when it serves a different one,
- proceeds to spawn only when the port is genuinely free, and reports the
  occupied port when an unidentifiable listener holds it.
"""

import contextlib
import json
import socket
from unittest.mock import MagicMock

import pytest

from bibr.exceptions import UpstreamServiceError
from bibr.local.http_runtime import LocalHttpError, guard_managed_server_port


def _models_payload(ids):
    return json.dumps({"object": "list", "data": [{"id": i} for i in ids]}).encode()


def _request_fn_serving(ids, *, models_status=200):
    """request_bytes stand-in: /v1/models lists *ids*, everything else is 200."""

    def fake_request_bytes(url, **_kwargs):
        if url.endswith("/v1/models"):
            return models_status, "OK", _models_payload(ids)
        return 200, "OK", b'{"model_loaded": true}'

    return fake_request_bytes


def _request_fn_port_free(url, **_kwargs):
    raise LocalHttpError("connection refused")


@contextlib.contextmanager
def _held_port():
    """Bind an ephemeral port so the guard sees a genuinely occupied one."""
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        yield held.getsockname()[1]


class TestGuardHelper:
    def test_free_port_returns_false(self):
        assert (
            guard_managed_server_port(
                "ocr",
                base_url="http://localhost:8772",
                model="mlx-community/GLM-OCR-bf16",
                server_label="Rapid-MLX",
                request_fn=_request_fn_port_free,
            )
            is False
        )

    def test_matching_listener_is_reused(self):
        assert (
            guard_managed_server_port(
                "ocr",
                base_url="http://localhost:8772",
                model="mlx-community/GLM-OCR-bf16",
                server_label="Rapid-MLX",
                request_fn=_request_fn_serving(["mlx-community/GLM-OCR-bf16"]),
            )
            is True
        )

    def test_mismatched_listener_raises_naming_both_models(self):
        with pytest.raises(UpstreamServiceError) as excinfo:
            guard_managed_server_port(
                "ocr",
                base_url="http://localhost:8772",
                model="mlx-community/GLM-OCR-bf16",
                server_label="Rapid-MLX",
                request_fn=_request_fn_serving(["mlx-community/GLM-OCR-8bit"]),
            )
        msg = str(excinfo.value)
        assert "mlx-community/GLM-OCR-bf16" in msg
        assert "mlx-community/GLM-OCR-8bit" in msg
        assert "8772" in msg

    def test_unidentifiable_listener_names_the_occupied_port(self):
        # A listener without a usable /v1/models holds the port, so the managed
        # server could never bind it. Report the port conflict rather than
        # spawning a subprocess that is guaranteed to die (issue #82).
        def not_found(url, **_kwargs):
            return 404, "Not Found", b""

        with _held_port() as port:
            with pytest.raises(UpstreamServiceError) as excinfo:
                guard_managed_server_port(
                    "ocr",
                    base_url=f"http://127.0.0.1:{port}",
                    model="m",
                    server_label="Rapid-MLX",
                    request_fn=not_found,
                )
        message = str(excinfo.value)
        assert str(port) in message
        assert "404" in message

    def test_unparsable_models_body_names_the_occupied_port(self):
        def garbage(url, **_kwargs):
            return 200, "OK", b"not json"

        with _held_port() as port:
            with pytest.raises(UpstreamServiceError, match=str(port)):
                guard_managed_server_port(
                    "ocr",
                    base_url=f"http://127.0.0.1:{port}",
                    model="m",
                    server_label="Rapid-MLX",
                    request_fn=garbage,
                )

    def test_unidentifiable_probe_on_a_free_port_still_spawns(self):
        # An unusable /v1/models with nothing actually on the port (a stubbed or
        # proxied probe) must not be reported as a conflict — spawn normally.
        # Uses an ephemeral free port, never a production default: with a
        # leftover Rapid-MLX server on :8772 this asserted the wrong thing.
        def not_found(url, **_kwargs):
            return 404, "Not Found", b""

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        assert (
            guard_managed_server_port(
                "ocr",
                base_url=f"http://127.0.0.1:{port}",
                model="m",
                server_label="Rapid-MLX",
                request_fn=not_found,
            )
            is False
        )


class TestRapidMlxServerGuard:
    def test_fails_fast_on_stale_server_with_other_model(self, monkeypatch):
        from bibr.config import GlobalSettings
        from bibr.local import rapid_mlx as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(
            mod, "request_bytes", _request_fn_serving(["mlx-community/GLM-OCR-8bit"])
        )

        with pytest.raises(UpstreamServiceError, match="GLM-OCR-bf16"):
            mod.RapidMlxServer(
                model="mlx-community/GLM-OCR-bf16",
                served_model_name="mlx-community/GLM-OCR-bf16",
                port=8772,
                multimodal=True,
                settings=GlobalSettings(),
            )
        popen.assert_not_called()

    def test_reuses_stale_server_with_matching_model(self, monkeypatch):
        from bibr.config import GlobalSettings
        from bibr.local import rapid_mlx as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.shutil, "which", lambda exe: "/opt/rapid-mlx/bin/rapid-mlx")
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(
            mod, "request_bytes", _request_fn_serving(["mlx-community/GLM-OCR-8bit"])
        )

        server = mod.RapidMlxServer(
            model="mlx-community/GLM-OCR-8bit",
            served_model_name="mlx-community/GLM-OCR-8bit",
            port=8772,
            multimodal=True,
            settings=GlobalSettings(),
        )
        assert server.loaded is True
        assert server.base_url == "http://localhost:8772"
        server.shutdown()  # must not touch the process we don't own
        popen.assert_not_called()


class TestVllmMlxServerGuard:
    def test_fails_fast_on_stale_server_with_other_model(self, monkeypatch):
        from bibr.local import ocr as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(mod, "request_bytes", _request_fn_serving(["other/model"]))

        with pytest.raises(UpstreamServiceError, match="requested/model"):
            mod.VllmMlxServer(model="requested/model", port=8766)
        popen.assert_not_called()

    def test_reuses_stale_server_with_matching_model(self, monkeypatch):
        from bibr.local import ocr as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(mod, "request_bytes", _request_fn_serving(["requested/model"]))

        server = mod.VllmMlxServer(model="requested/model", port=8766)
        assert server.loaded is True
        assert server.base_url == "http://localhost:8766"
        server.shutdown()
        popen.assert_not_called()


class TestVllmLlmServerGuard:
    def test_fails_fast_on_stale_server_with_other_model(self, monkeypatch):
        from bibr.config import GlobalSettings
        from bibr.local import vllm_llm as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(mod, "request_bytes", _request_fn_serving(["other/model"]))

        with pytest.raises(UpstreamServiceError, match="requested/model"):
            mod.VllmLlmServer(model="requested/model", port=8768, settings=GlobalSettings())
        popen.assert_not_called()

    def test_reuses_stale_server_with_matching_model(self, monkeypatch):
        from bibr.config import GlobalSettings
        from bibr.local import vllm_llm as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(mod, "request_bytes", _request_fn_serving(["requested/model"]))

        server = mod.VllmLlmServer(model="requested/model", port=8768, settings=GlobalSettings())
        assert server.base_url == "http://localhost:8768"
        server.shutdown()
        popen.assert_not_called()


class TestLlamaCppServerGuard:
    def test_fails_fast_on_stale_server_with_other_model(self, monkeypatch):
        from bibr.local import llama_cpp as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(mod, "request_bytes", _request_fn_serving(["other/model"]))

        with pytest.raises(UpstreamServiceError, match="requested/model"):
            mod.LlamaCppServer(model="requested/model", port=8771, role="ocr")
        popen.assert_not_called()

    def test_reuses_stale_server_with_matching_model(self, monkeypatch):
        from bibr.local import llama_cpp as mod

        popen = MagicMock(side_effect=AssertionError("must not spawn"))
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        monkeypatch.setattr(mod, "request_bytes", _request_fn_serving(["requested/model"]))

        server = mod.LlamaCppServer(model="requested/model", port=8771, role="ocr")
        assert server.loaded is True
        assert server.base_url == "http://127.0.0.1:8771"
        server.shutdown()
        popen.assert_not_called()
