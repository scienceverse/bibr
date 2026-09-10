"""Readiness payload info-disclosure gating (audit L1)."""

import pytest

pytest.importorskip("fastapi")


def test_readiness_payload_hides_detail_when_not_authenticated():
    from bibr.config import Settings
    from bibr.serve.app import readiness_payload

    checks = {"ocr": "ok", "redis": "ok"}
    full = readiness_payload("ready", checks, Settings, include_detail=True)
    assert full["checks"] == checks
    assert "build_sha" in full

    minimal = readiness_payload("not_ready", checks, Settings, include_detail=False)
    assert minimal == {"status": "not_ready"}
    assert "build_sha" not in minimal
    assert "checks" not in minimal


def test_wildcard_cors_disables_credentials():
    """`*` origins + credentials would reflect any Origin — credentials off (L3)."""
    from bibr.serve.app import _safe_cors_credentials

    assert _safe_cors_credentials(["*"], True) is False
    assert _safe_cors_credentials(["https://app.example"], True) is True
    assert _safe_cors_credentials(["*"], False) is False
