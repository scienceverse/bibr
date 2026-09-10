import pytest

from bibr.ocr.http_security import ocr_request_headers


def test_https_remote_ocr_gets_bearer_header():
    assert ocr_request_headers("https://ocr.example.com", "secret", allow_insecure_http=False) == {
        "Authorization": "Bearer secret"
    }


@pytest.mark.parametrize(
    "url", ["http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080"]
)
def test_loopback_http_is_allowed(url):
    assert ocr_request_headers(url, None, allow_insecure_http=False) == {}


def test_remote_plain_http_is_rejected_by_default():
    with pytest.raises(ValueError, match="insecure HTTP"):
        ocr_request_headers("http://ocr.example.com", None, allow_insecure_http=False)


def test_private_network_http_requires_explicit_override():
    assert ocr_request_headers("http://bibr-ocr:8080", "secret", allow_insecure_http=True) == {
        "Authorization": "Bearer secret"
    }


@pytest.mark.parametrize("url", ["ocr.example.com", "ftp://ocr.example.com/model"])
def test_invalid_ocr_url_is_rejected(url):
    with pytest.raises(ValueError, match="absolute"):
        ocr_request_headers(url, None, allow_insecure_http=False)


class TestNormalizeOcrBaseUrl:
    """bibr appends ``/v1/...`` itself; a configured ``/v1`` suffix must not double up."""

    def test_strips_trailing_v1_with_a_warning(self, caplog):
        from bibr.ocr.http_security import normalize_ocr_base_url

        with caplog.at_level("WARNING", logger="bibr.ocr.http_security"):
            url = normalize_ocr_base_url("https://ocr.example.internal/v1")
        assert url == "https://ocr.example.internal"
        assert any("/v1" in r.getMessage() for r in caplog.records)

    def test_handles_trailing_slashes_and_case(self):
        from bibr.ocr.http_security import normalize_ocr_base_url

        assert normalize_ocr_base_url("https://ocr.example.internal/v1/") == (
            "https://ocr.example.internal"
        )
        assert normalize_ocr_base_url("http://localhost:8080/") == "http://localhost:8080"
        assert normalize_ocr_base_url("http://host/api/V1") == "http://host/api"

    def test_leaves_other_paths_alone(self):
        from bibr.ocr.http_security import normalize_ocr_base_url

        assert normalize_ocr_base_url("http://host/v10") == "http://host/v10"
        assert normalize_ocr_base_url("http://host/v1x") == "http://host/v1x"
        assert normalize_ocr_base_url("http://host:8080") == "http://host:8080"
