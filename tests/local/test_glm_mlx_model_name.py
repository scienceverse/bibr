"""Regression tests for the glm-mlx model-name mismatch bug.

``HttpOcrClient._send_request`` hardcoded ``"model": "glm-ocr"`` in its
payload. That alias only exists on external servers that were deliberately
started with a matching name (the documented ``glm-http`` workflow). The
managed ``vllm-mlx`` subprocess behind ``glm-mlx`` serves under the actual
HF repo id passed via ``--model`` (``python -m vllm_mlx.server`` has no
``--served-model-name`` flag), so every real request 404s.
"""

import pytest


def _fake_image():
    from PIL import Image

    return Image.new("RGB", (100, 100), color="white")


class TestHttpOcrClientModelName:
    async def test_recognize_defaults_to_glm_ocr_alias(self):
        """No override -> preserves the existing external-server convention."""
        pytest.importorskip("cv2")
        from bibr.local.ocr import HttpOcrClient

        client = HttpOcrClient(base_url="http://localhost:9999")
        captured = {}

        async def fake_post(url, *, json=None, **_kw):
            captured["json"] = json

            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self_inner):
                    return {"choices": [{"message": {"content": "hi"}}]}

            return _Resp()

        client._client.post = fake_post
        await client.recognize(_fake_image(), "Text Recognition:")
        assert captured["json"]["model"] == "glm-ocr"

    async def test_recognize_sends_configured_model_when_overridden(self):
        """A managed mlx server serves under its own repo id, not the alias."""
        pytest.importorskip("cv2")
        from bibr.local.ocr import HttpOcrClient

        client = HttpOcrClient(base_url="http://localhost:9999", model="mlx-community/GLM-OCR-6bit")
        captured = {}

        async def fake_post(url, *, json=None, **_kw):
            captured["json"] = json

            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self_inner):
                    return {"choices": [{"message": {"content": "hi"}}]}

            return _Resp()

        client._client.post = fake_post
        await client.recognize(_fake_image(), "Text Recognition:")
        assert captured["json"]["model"] == "mlx-community/GLM-OCR-6bit"


class TestVllmMlxOcrClientDisabled:
    def test_construction_refuses(self):
        """glm-mlx (vllm-mlx --mllm) is permanently disabled — NUL-corrupted
        text plus an uncapped vision-cache leak — so it can no longer wire a
        model into an HttpOcrClient at all; see test_backend_resolution.py."""
        from bibr.exceptions import UpstreamServiceError
        from bibr.local import ocr as mod

        with pytest.raises(UpstreamServiceError, match="glm-rapid-mlx"):
            mod.VllmMlxOcrClient()
