"""``bibr serve`` names the ``serve`` extra when its packages are missing.

Runs on a core install too: the check itself imports nothing from the extra.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from bibr.exceptions import ConfigurationError


def test_missing_serve_packages_stop_with_the_install_hint(monkeypatch):
    from bibr.serve import app as app_mod

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in ("fastapi", "python_multipart"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    with pytest.raises(ConfigurationError) as excinfo:
        app_mod.require_serve_extra()
    message = str(excinfo.value)
    assert "missing: fastapi, python-multipart" in message
    assert "uv tool install 'bibr[serve]'" in message


def test_the_cli_prints_the_hint_and_exits_1(monkeypatch, capsys):
    from bibr.local import cli
    from bibr.serve import app as app_mod

    def missing():
        raise ConfigurationError("bibr serve needs the optional 'serve' dependencies")

    monkeypatch.setattr(app_mod, "require_serve_extra", missing)
    monkeypatch.setattr(sys, "argv", ["bibr", "serve"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    assert "'serve' dependencies" in capsys.readouterr().err


def test_importing_the_serve_entry_point_needs_no_serve_package():
    """``bibr serve`` must reach the check: the module imports none of the extra."""
    import subprocess

    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in ('fastapi', 'starlette', 'uvicorn', 'python_multipart'):\n"
        "        raise ImportError(name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import bibr.serve.app\n"
        "print('ok')\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "ok", result.stderr
