"""scripts/docs_macros.define_env must not leak the doc-builder's local

environment into published pages. It used to instantiate ``GlobalSettings()``,
which reads the builder's cwd `.env` / process environment — so a
developer's local ``LLM_PROVIDER``/``LLM_MODEL`` override would get baked
into the shipped `{{ default_llm_provider }}` / `{{ default_llm_model }}`
values. It must read the class defaults off ``LlmOptions`` directly instead.

scripts/ is not a package, so the module is loaded via its file path (same
pattern as tests/test_docs_ref_core.py).
"""

import importlib.util
from pathlib import Path

_MACROS_PATH = Path(__file__).parents[1] / "scripts" / "docs_macros.py"


def _load_macros():
    spec = importlib.util.spec_from_file_location("docs_macros", _MACROS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubEnv:
    """Minimal stand-in for the mkdocs-macros ``env`` object: just needs

    a ``.variables`` dict.
    """

    def __init__(self):
        self.variables = {}


def test_define_env_uses_class_defaults_not_local_env(monkeypatch):
    from bibr.config import LlmOptions

    # tests/conftest.py sets LLM_PROVIDER=google, which happens to equal the
    # class default — that coincidence would mask the bug. Override to values
    # that clash with the shipped defaults to prove the fix actually reads
    # class defaults rather than the environment.
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "some-local-dev-model")

    macros = _load_macros()
    env = _StubEnv()
    macros.define_env(env)

    expected_provider = LlmOptions.model_fields["provider"].default
    expected_model = LlmOptions.model_fields["model"].default

    assert env.variables["default_llm_provider"] == expected_provider
    assert env.variables["default_llm_model"] == expected_model
    assert env.variables["default_llm_provider"] != "anthropic"
    assert env.variables["default_llm_model"] != "some-local-dev-model"


def test_define_env_sets_schema_version():
    from bibr.export.json_export import _SCHEMA_VERSION

    macros = _load_macros()
    env = _StubEnv()
    macros.define_env(env)
    assert env.variables["schema_version"] == _SCHEMA_VERSION
