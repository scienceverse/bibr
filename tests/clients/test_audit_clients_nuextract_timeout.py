"""The NuExtract native client's request timeout."""

from __future__ import annotations

from bibr.config import GlobalSettings


def test_nuextract_native_client_connects_within_the_sdk_connect_timeout():
    # A blackholed endpoint must fail at connect like every other provider
    # client (bibr.clients.providers.base.bound_sdk_client), not after the
    # whole request limit.
    from bibr.clients.nuextract import NuExtractNativeBackend

    settings = GlobalSettings(
        llm={
            "provider": "openai",
            "model": "m",
            "base_url": "http://127.0.0.1:8000/v1",
            "timeout_seconds": 60,
        }
    )
    client = NuExtractNativeBackend(settings=settings)._get_client()

    assert client.timeout.connect == 5.0
    assert client.timeout.read == 120.0
