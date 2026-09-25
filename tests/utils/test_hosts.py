"""Host classification and the plaintext-credential policy (x-security-9)."""

import pytest

from bibr.utils.hosts import is_loopback_host, is_private_network_host, refuse_public_plaintext


@pytest.mark.parametrize(
    "host",
    ["localhost", "LOCALHOST.", "bibr.localhost", "127.0.0.1", "127.8.9.10", "::1", "[::1]"],
)
def test_loopback_hosts(host):
    assert is_loopback_host(host) is True


@pytest.mark.parametrize("host", ["evil.example", "localhost.evil.example", "10.0.0.5", "::"])
def test_non_loopback_hosts(host):
    assert is_loopback_host(host) is False


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "gpu-box",  # single label: hosts file / local search domain
        "fridge",
        "10.1.2.3",
        "192.168.1.20",
        "172.16.0.9",
        "100.113.200.117",  # tailnet (100.64.0.0/10)
        "fd7a:115c:a1e0::1",  # unique-local IPv6
        "169.254.10.1",
        "ocr.internal",
        "nas.local",
        "box.lan",
        "printer.home.arpa",
    ],
)
def test_private_network_hosts(host):
    assert is_private_network_host(host) is True


@pytest.mark.parametrize("host", ["bibr.example.org", "api.openai.com", "8.8.8.8", ""])
def test_public_hosts(host):
    assert is_private_network_host(host) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://bibr.example.org",
        "http://127.0.0.1:8001/v1",
        "http://gpu-box:8000",
        "http://100.113.200.117:30000/v1",
        "http://ocr.internal:8080",
    ],
)
def test_refuse_public_plaintext_allows_tls_and_private_networks(url):
    refuse_public_plaintext(url, credential="token", opt_out="set X=true")


def test_refuse_public_plaintext_refuses_a_public_http_host():
    with pytest.raises(ValueError) as exc:
        refuse_public_plaintext(
            "http://bibr.example.org:8000", credential="serve bearer token", opt_out="pass --x"
        )
    assert str(exc.value) == (
        "Refusing to send the serve bearer token over plain HTTP to the public host "
        "'bibr.example.org'. Use an https:// URL, or pass --x if the network path is trusted."
    )
