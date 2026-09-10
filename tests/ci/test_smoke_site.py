from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "scripts" / "ci" / "smoke_site.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
TEST_SECRET = "test-secret"


def load_smoke_site() -> ModuleType:
    assert MODULE_PATH.is_file(), "site smoke helper has not been implemented"
    spec = importlib.util.spec_from_file_location("smoke_site", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("status", [401, 403])
def test_anonymous_auth_failure_is_a_valid_challenge(status: int) -> None:
    smoke = load_smoke_site()

    assert smoke.anonymous_is_challenged(status, None) is True


def test_cloudflare_access_redirect_is_a_valid_challenge() -> None:
    smoke = load_smoke_site()

    assert (
        smoke.anonymous_is_challenged(
            302,
            "https://bibr.cloudflareaccess.com/cdn-cgi/access/login/bibr.org?kid=example",
        )
        is True
    )


@pytest.mark.parametrize(
    ("status", "location"),
    [(200, None), (302, "https://example.org/elsewhere"), (500, None)],
)
def test_anonymous_unprotected_or_unrelated_response_fails(
    status: int, location: str | None
) -> None:
    smoke = load_smoke_site()

    assert smoke.anonymous_is_challenged(status, location) is False


def test_authorized_marker_must_equal_the_full_source_sha() -> None:
    smoke = load_smoke_site()

    smoke.verify_build_marker(f"{SHA}\n", SHA)


@pytest.mark.parametrize("body", ["wrong", f"prefix-{SHA}", SHA[:12]])
def test_authorized_marker_rejects_the_wrong_revision(body: str) -> None:
    smoke = load_smoke_site()

    with pytest.raises(ValueError, match="marker"):
        smoke.verify_build_marker(body, SHA)


def test_expected_revision_must_be_a_full_sha() -> None:
    smoke = load_smoke_site()

    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        smoke.verify_build_marker(SHA, "abc123")


def test_public_deployment_checks_revision_without_credentials(monkeypatch) -> None:
    smoke = load_smoke_site()
    calls = []

    def fetch(url, *, headers=None, follow_redirects):
        calls.append((url, headers, follow_redirects))
        return 200, None, f"{SHA}\n"

    monkeypatch.setattr(smoke, "_fetch", fetch)
    smoke.smoke("https://bibr.org", SHA, access="public")

    assert calls == [("https://bibr.org/.well-known/bibr-build", None, False)]


@pytest.mark.parametrize("status", [301, 302, 401, 403, 404, 500])
def test_public_deployment_rejects_redirects_and_unavailable_markers(monkeypatch, status) -> None:
    smoke = load_smoke_site()
    monkeypatch.setattr(smoke, "_fetch", lambda *args, **kwargs: (status, None, SHA))

    with pytest.raises(ValueError, match=f"anonymous request returned HTTP {status}"):
        smoke.smoke("https://bibr.org", SHA, access="public")


def test_public_deployment_rejects_stale_revision(monkeypatch) -> None:
    smoke = load_smoke_site()
    monkeypatch.setattr(smoke, "_fetch", lambda *args, **kwargs: (200, None, "f" * 40))

    with pytest.raises(ValueError, match="expected revision"):
        smoke.smoke("https://bibr.org", SHA, access="public")


def test_protected_deployment_checks_access_before_authorized_revision(monkeypatch) -> None:
    smoke = load_smoke_site()
    calls = []

    def fetch(url, *, headers=None, follow_redirects):
        calls.append((url, headers, follow_redirects))
        return (200, None, SHA) if headers else (403, None, "Access denied")

    monkeypatch.setattr(smoke, "_fetch", fetch)
    smoke.smoke(
        "https://preview.example.org",
        SHA,
        access="protected",
        client_id="test-id",
        client_secret=TEST_SECRET,
    )

    assert calls == [
        ("https://preview.example.org/.well-known/bibr-build", None, False),
        (
            "https://preview.example.org/.well-known/bibr-build",
            {"CF-Access-Client-Id": "test-id", "CF-Access-Client-Secret": TEST_SECRET},
            True,
        ),
    ]


def test_protected_deployment_rejects_public_access(monkeypatch) -> None:
    smoke = load_smoke_site()
    monkeypatch.setattr(smoke, "_fetch", lambda *args, **kwargs: (200, None, SHA))

    with pytest.raises(ValueError, match="anonymous request was not challenged"):
        smoke.smoke(
            "https://preview.example.org",
            SHA,
            access="protected",
            client_id="test-id",
            client_secret=TEST_SECRET,
        )


@pytest.mark.parametrize("access, expected_status", [("public", 0), ("protected", 2)])
def test_cli_only_requires_access_credentials_for_protected_sites(
    monkeypatch, access, expected_status
) -> None:
    smoke = load_smoke_site()
    monkeypatch.delenv("CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(smoke, "_fetch", lambda *args, **kwargs: (200, None, SHA))
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke_site.py", f"--access={access}", "--url=https://bibr.org", f"--expected-sha={SHA}"],
    )

    assert smoke.main() == expected_status


@pytest.mark.parametrize("authorized", [False, True])
def test_http_probe_identifies_the_client_and_preserves_auth_headers(
    monkeypatch, authorized
) -> None:
    smoke = load_smoke_site()
    url = "https://preview.example.pages.dev/.well-known/bibr-build"
    headers = (
        {"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": TEST_SECRET} if authorized else {}
    )
    captured = []

    def open_request(request, timeout):
        assert timeout == 20
        captured.append({key.lower(): value for key, value in request.header_items()})
        return nullcontext(
            SimpleNamespace(status=200, headers={}, url=url, read=lambda: SHA.encode())
        )

    monkeypatch.setattr(smoke, "build_opener", lambda *args: SimpleNamespace(open=open_request))

    assert smoke._fetch(url, headers=headers, follow_redirects=authorized) == (200, None, SHA)
    assert captured == [
        {"user-agent": "bibr-ci-smoke/1.0", **{k.lower(): v for k, v in headers.items()}}
    ]


@pytest.mark.parametrize("follow_redirects", [False, True])
def test_browser_signature_block_is_not_reported_as_access_authentication(
    monkeypatch, follow_redirects
) -> None:
    smoke = load_smoke_site()
    url = "https://preview.example.pages.dev/.well-known/bibr-build"

    def open_request(request, timeout):
        raise HTTPError(url, 403, "Forbidden", {}, BytesIO(b"error code: 1010\n"))

    monkeypatch.setattr(smoke, "build_opener", lambda *args: SimpleNamespace(open=open_request))

    with pytest.raises(ValueError, match="browser signature.*1010.*before Access"):
        smoke._fetch(url, follow_redirects=follow_redirects)
