"""Snapshot/restore the LLM provider registry per test.

Tests that call ``providers.register(_FakeProvider)`` mutate the module-level
``_PROVIDERS`` dict. Without this fixture the mutation would leak to other
tests in the same session.

The baseline is captured after ``bibr.clients.providers`` imports all bundled
adapters (which fires their ``@register`` side-effects once); restoring from
a pre-import snapshot would wipe the production providers for good.
"""

import pytest

from bibr.clients import providers

_BASELINE = dict(providers._PROVIDERS)


@pytest.fixture(autouse=True)
def _llm_provider_registry_snapshot():
    providers._PROVIDERS.clear()
    providers._PROVIDERS.update(_BASELINE)
    yield
    providers._PROVIDERS.clear()
    providers._PROVIDERS.update(_BASELINE)
