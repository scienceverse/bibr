"""Serve bounds in-flight LLM concurrency by default.

The library default (``LLM_MAX_CONCURRENCY=0`` → unlimited) suits CLI/cloud
batch, but in serve a single high-fan-out paper can fire dozens of concurrent
LLM calls and self-induce provider 503 storms (the 503-retry path only absorbs
the stragglers). Serve therefore caps concurrency unless the operator set it
explicitly — e.g. ``=1`` for a local single-device LLM server, which must be
honored.
"""

import pytest

pytest.importorskip("litserve")

from bibr.serve.deployments.pipeline import (  # noqa: E402
    SERVE_DEFAULT_LLM_CONCURRENCY,
    _apply_serve_llm_concurrency_default,
)


class _FakeLlm:
    def __init__(self, *, explicit: bool, value: int = 0):
        self.max_concurrency = value
        self.model_fields_set = {"max_concurrency"} if explicit else set()


class _FakeSettings:
    def __init__(self, llm):
        self.llm = llm


def test_default_applied_when_operator_left_it_unset():
    s = _FakeSettings(_FakeLlm(explicit=False))
    applied = _apply_serve_llm_concurrency_default(s)
    assert applied is True
    assert SERVE_DEFAULT_LLM_CONCURRENCY > 0
    assert s.llm.max_concurrency == SERVE_DEFAULT_LLM_CONCURRENCY


def test_explicit_value_is_respected():
    """An operator who set LLM_MAX_CONCURRENCY (e.g. =1 for a local LLM
    server) must not have it silently overridden."""
    s = _FakeSettings(_FakeLlm(explicit=True, value=1))
    applied = _apply_serve_llm_concurrency_default(s)
    assert applied is False
    assert s.llm.max_concurrency == 1
