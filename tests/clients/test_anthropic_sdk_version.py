"""The Anthropic adapter refuses the anthropic SDK from 1.0 on, with the fix in the error.

anthropic 1.0 dropped ``temperature`` from ``messages.create`` and moved to
httpx2, so every call through the adapter failed: with a ``TypeError`` when it
carried a temperature, as an unexplained "Connection error." otherwise. The
``batch`` extra pins the SDK below 1.0.
"""

import sys
from types import SimpleNamespace

import pytest

from bibr.clients.providers.anthropic import AnthropicProvider
from bibr.config import GlobalSettings


def _provider():
    settings = GlobalSettings(
        llm={"provider": "anthropic", "model": "claude-haiku-5-5", "api_key": "sk-ant-test-0000"}
    )
    return AnthropicProvider(settings=settings)


def _never_built(*_args, **_kwargs):
    pytest.fail("the SDK client was built")


@pytest.mark.parametrize("version", ["1.0.0", "1.13.0", "2.0.0a1"])
def test_sdk_from_1_0_is_refused_before_a_client_is_built(monkeypatch, version):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(__version__=version))
    monkeypatch.setattr("instructor.from_provider", _never_built)

    with pytest.raises(ValueError, match=r"below 1\.0") as excinfo:
        _provider().build_client()

    assert version in str(excinfo.value)
    assert 'pip install "anthropic>=0.40,<1"' in str(excinfo.value)


@pytest.mark.parametrize("version", ["0.40.0", "0.118.0", None])
def test_sdk_below_1_0_builds_the_client(monkeypatch, version):
    """``None``: a stand-in module without a version is let through."""
    raw = SimpleNamespace(max_retries=2, timeout=600.0)
    module = SimpleNamespace() if version is None else SimpleNamespace(__version__=version)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setattr("instructor.from_provider", lambda *a, **k: SimpleNamespace(client=raw))

    _provider().build_client()

    assert raw.max_retries == 0
