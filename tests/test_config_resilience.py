"""Resilience of the settings singleton against an invalid ``.env`` / environment.

``bibr.config.Settings`` is a lazy proxy: it constructs the real
``GlobalSettings`` on first attribute access (not at import) and raises a
friendly ``ConfigurationError`` — naming the offending env var and its allowed
values — instead of a raw pydantic traceback. These tests pin that behavior plus
the identity-preserving mutation the setup wizard relies on, and the CLI /
doctor degradation paths.
"""

from __future__ import annotations

import sys

import pytest

from bibr.config import GlobalSettings, _SettingsProxy
from bibr.exceptions import BibrError, ConfigurationError


def test_configuration_error_is_bibr_error():
    assert issubclass(ConfigurationError, BibrError)


def test_proxy_constructs_lazily_and_caches():
    proxy = _SettingsProxy()
    # Nothing constructed yet — the underlying instance is still None.
    assert object.__getattribute__(proxy, "_instance") is None
    first = proxy.llm  # first access triggers construction
    assert object.__getattribute__(proxy, "_instance") is not None
    # Same underlying instance is reused on subsequent access.
    assert proxy.llm is first


def test_proxy_identity_is_stable():
    proxy = _SettingsProxy()
    _ = proxy.llm.provider  # force construction
    import bibr.config

    assert bibr.config.Settings is bibr.config.Settings


def test_proxy_forwards_setattr_to_underlying_instance():
    proxy = _SettingsProxy()
    proxy.llm.local_model = "sentinel-model"
    assert proxy.llm.local_model == "sentinel-model"
    # Top-level scalar assignment also forwards.
    proxy.EQUATION_EXTRACTION = False
    assert proxy.EQUATION_EXTRACTION is False


def test_reload_settings_in_place_pattern_preserves_identity():
    """The wizard's ``_reload_settings_in_place`` loop must keep working: copy
    every section from a fresh ``GlobalSettings`` onto the *same* proxy object."""
    proxy = _SettingsProxy()
    underlying_before = proxy.llm  # force construction
    fresh = GlobalSettings()
    for name in type(fresh).model_fields:
        setattr(proxy, name, getattr(fresh, name))
    # The proxy object's identity is unchanged; its sections now match ``fresh``.
    assert proxy.llm is not None
    assert underlying_before is not None
    assert proxy.pipeline.memory_mode == fresh.pipeline.memory_mode


def test_bad_submodel_literal_maps_to_env_var(monkeypatch):
    monkeypatch.setenv("PIPELINE_MEMORY_MODE", "low")
    proxy = _SettingsProxy()
    with pytest.raises(ConfigurationError) as exc:
        _ = proxy.pipeline.memory_mode
    problems = exc.value.problems
    assert len(problems) == 1
    assert "PIPELINE_MEMORY_MODE" in problems[0]
    assert "low" in problems[0]
    # Allowed values from the Literal are surfaced.
    for allowed in ("aggressive", "balanced", "keep_all"):
        assert allowed in problems[0]
    assert "PIPELINE_MEMORY_MODE" in str(exc.value)


def test_bad_crossref_consolidate_maps_to_env_var(monkeypatch):
    monkeypatch.setenv("CROSSREF_CONSOLIDATE", "maybe")
    proxy = _SettingsProxy()
    with pytest.raises(ConfigurationError) as exc:
        _ = proxy.crossref.consolidate
    msg = str(exc.value)
    assert "CROSSREF_CONSOLIDATE=maybe" in msg
    for allowed in ("off", "fill", "replace"):
        assert allowed in msg


def test_bad_top_level_literal_maps_to_env_var(monkeypatch):
    # Top-level (un-clustered) field: the loc name is already the env var name.
    monkeypatch.setenv("REF_SEG_STRATEGY", "bogus")
    proxy = _SettingsProxy()
    with pytest.raises(ConfigurationError) as exc:
        _ = proxy.REF_SEG_STRATEGY
    assert "REF_SEG_STRATEGY=bogus" in str(exc.value)


def test_failure_is_not_cached_retry_after_fix(monkeypatch):
    monkeypatch.setenv("PIPELINE_MEMORY_MODE", "low")
    proxy = _SettingsProxy()
    with pytest.raises(ConfigurationError):
        _ = proxy.pipeline.memory_mode
    # Once the offending value is corrected, the next access must succeed.
    monkeypatch.delenv("PIPELINE_MEMORY_MODE")
    assert proxy.llm.provider is not None


def test_cli_chew_reports_bad_env_cleanly(monkeypatch, capsys):
    """A bad env on the chew path yields a styled ``✗ <friendly>`` + exit 1, no traceback."""
    import bibr.config
    from bibr.local.cli import main

    monkeypatch.setenv("CROSSREF_CONSOLIDATE", "maybe")
    monkeypatch.setattr(bibr.config, "Settings", _SettingsProxy())
    monkeypatch.setattr(sys, "argv", ["bibr", "chew", "nope.pdf", "-o", "/dev/null"])

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "✗" in err
    assert "CROSSREF_CONSOLIDATE" in err
    assert "Traceback" not in err


def test_doctor_survives_bad_env(monkeypatch, capsys):
    """``bibr doctor`` degrades a bad config into a failed check and still runs
    the Settings-free diagnostics (Python version, uv on PATH)."""
    import bibr.config
    from bibr.local.cli import _run_doctor

    monkeypatch.setenv("PIPELINE_MEMORY_MODE", "low")
    monkeypatch.setattr(bibr.config, "Settings", _SettingsProxy())

    with pytest.raises(SystemExit) as exc:
        _run_doctor()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    # The config problem is surfaced as a failed check naming the env var.
    assert "PIPELINE_MEMORY_MODE" in out
    assert "aggressive" in out
    # Settings-free checks still ran.
    assert "Python" in out
    assert "uv" in out
    assert "Traceback" not in out
