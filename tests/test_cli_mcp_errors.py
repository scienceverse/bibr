"""``bibr mcp`` must not crash with a traceback when cloud credentials are missing.

The ``Chewer`` preflight raises the provider's plain ``ValueError`` for a
missing key; the mcp branch converts it to a one-line stderr message plus
exit 1, like the batch branch already does.
"""

import pytest

pytest.importorskip("mcp")

from bibr.local.cli import main


def test_mcp_missing_credentials_exits_1_without_traceback(monkeypatch, capsys):
    import bibr.mcp_server

    def no_creds(args):
        raise ValueError(
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
