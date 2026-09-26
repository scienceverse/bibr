"""``bibr mcp`` reports missing credentials without a traceback.

The ``Chewer`` preflight raises ``ConfigurationError`` for a missing key,
which the shared handler converts to a one-line stderr message plus exit 1.
A ``ValueError`` escaping the server session itself keeps its traceback.
"""

import pytest

from bibr.local.cli import main


def _need_mcp():
    pytest.importorskip("mcp")


def test_mcp_missing_credentials_exits_1_without_traceback(monkeypatch, capsys):
    _need_mcp()
    import bibr.mcp_server
    from bibr.exceptions import ConfigurationError

    def no_creds(args):
        raise ConfigurationError(
            "Google API key required. Set LLM_API_KEY or GOOGLE_API_KEY environment variable."
        )

    monkeypatch.setattr(bibr.mcp_server, "run_mcp", no_creds)
    monkeypatch.setattr("sys.argv", ["bibr", "mcp"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Google API key required" in err
    assert "Traceback" not in err


def test_mcp_bibr_error_still_exits_1(monkeypatch, capsys):
    _need_mcp()
    import bibr.mcp_server
    from bibr.exceptions import BibrError

    def broken(args):
        raise BibrError("something known went wrong")

    monkeypatch.setattr(bibr.mcp_server, "run_mcp", broken)
    monkeypatch.setattr("sys.argv", ["bibr", "mcp"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert "something known went wrong" in capsys.readouterr().err


def test_mcp_serve_value_error_is_not_swallowed(monkeypatch):
    """A ValueError from the running session keeps its traceback."""

    _need_mcp()
    import bibr.mcp_server

    def serve_bug(args):
        raise ValueError("pydantic validation failed mid-session")

    monkeypatch.setattr(bibr.mcp_server, "run_mcp", serve_bug)
    monkeypatch.setattr("sys.argv", ["bibr", "mcp"])
    with pytest.raises(ValueError, match="mid-session"):
        main()


def test_chewer_preflight_wraps_missing_key_as_configuration_error():
    from bibr.api import _preflight_llm
    from bibr.config import GlobalSettings
    from bibr.exceptions import ConfigurationError

    settings = GlobalSettings(llm={"provider": "anthropic", "api_key": None})
    settings.ANTHROPIC_API_KEY = None
    with pytest.raises(ConfigurationError, match="API key required"):
        _preflight_llm(settings, {})
