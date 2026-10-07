"""Audit fixes for the stdio MCP server (``bibr mcp``).

File tools are confined to the allowed directories (``--allow-dir``, default
the working directory), ``save_paper`` never writes through a symlink,
``chew_url`` honours the ``MCP_*`` URL settings, ``load_paper`` reads only
bounded regular files, a malformed export never stays half-loaded, the server
starts without LLM credentials, and a re-chewed paper is not evicted first.

Driven through the MCP in-memory client, like ``test_mcp_server.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client as client_session  # noqa: E402

import bibr.api  # noqa: E402
import bibr.mcp_server  # noqa: E402
from bibr.mcp_server import _PaperStore, build_server, run_mcp  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "inspect_full_export.json"
FIXTURE_ID = "10.1234/example.5678"


def _payload(result):
    assert not result.is_error, [c.text for c in result.content]
    payload = result.structured_content
    assert payload is not None
    return payload["result"] if set(payload) == {"result"} else payload


def _error_text(result) -> str:
    assert result.is_error
    return result.content[0].text


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return path


@pytest.fixture
def outside(tmp_path) -> Path:
    path = tmp_path / "outside"
    path.mkdir()
    return path


@pytest.fixture
def chews(monkeypatch) -> list[Path]:
    """Record every path the pipeline is asked to extract."""
    seen: list[Path] = []
    data = json.loads(FIXTURE.read_text())

    async def fake_achew_file(self, path, *, paper_id=None, progress=None):
        seen.append(Path(path))
        return bibr.api.Result(data)

    monkeypatch.setattr(bibr.api.Chewer, "achew_file", fake_achew_file)
    return seen


async def _papers(client) -> list[str]:
    return [p["paper_id"] for p in _payload(await client.call_tool("list_papers", {}))]


# ---------------------------------------------------------------------------
# Filesystem scope
# ---------------------------------------------------------------------------


async def test_default_scope_is_the_working_directory(monkeypatch, root, outside):
    shutil.copy(FIXTURE, root / "paper.json")
    shutil.copy(FIXTURE, outside / "private.json")
    monkeypatch.chdir(root)
    async with client_session(build_server()) as client:
        assert _payload(await client.call_tool("load_paper", {"path": "paper.json"}))
        text = _error_text(
            await client.call_tool("load_paper", {"path": str(outside / "private.json")})
        )
        assert "outside the directories this server may access" in text
        assert str(root) in text
        assert "--allow-dir" in text
        assert await _papers(client) == [FIXTURE_ID]


async def test_allow_dir_admits_each_listed_directory(root, outside):
    shutil.copy(FIXTURE, outside / "paper.json")
    async with client_session(build_server(allowed_dirs=[root, outside])) as client:
        summary = _payload(
            await client.call_tool("load_paper", {"path": str(outside / "paper.json")})
        )
        assert summary["paper_id"] == FIXTURE_ID


async def test_chew_paper_refuses_files_outside_the_roots(chews, root, outside):
    secret = outside / "private.pdf"
    secret.write_bytes(b"%PDF-1.4 private")
    (root / "link.pdf").symlink_to(secret)
    inside = root / "paper.pdf"
    inside.write_bytes(b"%PDF-1.4 stub")
    async with client_session(build_server(allowed_dirs=[root])) as client:
        for path in (secret, root / "link.pdf", root / ".." / "outside" / "private.pdf"):
            text = _error_text(await client.call_tool("chew_paper", {"path": str(path)}))
            assert "outside the directories" in text
        assert chews == []

        _payload(await client.call_tool("chew_paper", {"path": str(inside)}))
        assert chews == [inside]


async def test_chew_paper_keeps_an_in_root_symlink_name(chews, root):
    blob = root / "objects" / "abc123"
    blob.parent.mkdir()
    blob.write_bytes(b"%PDF-1.4 stub")
    (root / "paper.pdf").symlink_to(blob)
    async with client_session(build_server(allowed_dirs=[root])) as client:
        summary = _payload(await client.call_tool("chew_paper", {"path": str(root / "paper.pdf")}))
        assert summary["source"] == str(root / "paper.pdf")
        assert chews == [root / "paper.pdf"]


async def test_load_paper_refuses_a_symlink_out_of_the_root(root, outside):
    shutil.copy(FIXTURE, outside / "private.json")
    (root / "link.json").symlink_to(outside / "private.json")
    async with client_session(build_server(allowed_dirs=[root])) as client:
        text = _error_text(await client.call_tool("load_paper", {"path": str(root / "link.json")}))
        assert "outside the directories" in text
        assert await _papers(client) == []


async def test_save_paper_refuses_paths_outside_the_roots(root, outside):
    async with client_session(build_server(allowed_dirs=[root, FIXTURE.parent])) as client:
        _payload(await client.call_tool("load_paper", {"path": str(FIXTURE)}))
        for path in (outside / "out.json", outside / "new" / "out.json"):
            text = _error_text(
                await client.call_tool("save_paper", {"paper_id": FIXTURE_ID, "path": str(path)})
            )
            assert "outside the directories" in text
        # A directory symlink inside the root does not lead out either.
        (root / "sub").symlink_to(outside, target_is_directory=True)
        text = _error_text(
            await client.call_tool(
                "save_paper", {"paper_id": FIXTURE_ID, "path": str(root / "sub" / "out.json")}
            )
        )
        assert "outside the directories" in text
    assert list(outside.iterdir()) == []


async def test_save_paper_never_writes_through_a_symlink(root, outside):
    victim = outside / "config.json"
    victim.write_text('{"keep": true}')
    (root / "link.json").symlink_to(victim)
    (root / "dangling.json").symlink_to(outside / "planted.json")
    (root / "inner.json").symlink_to(root / "real.json")
    (root / "real.json").write_text("{}")
    async with client_session(build_server(allowed_dirs=[root, FIXTURE.parent])) as client:
        _payload(await client.call_tool("load_paper", {"path": str(FIXTURE)}))
        for name in ("link.json", "dangling.json", "inner.json"):
            for overwrite in (False, True):
                text = _error_text(
                    await client.call_tool(
                        "save_paper",
                        {"paper_id": FIXTURE_ID, "path": str(root / name), "overwrite": overwrite},
                    )
                )
                assert "symbolic link" in text
    assert victim.read_text() == '{"keep": true}'
    assert not (outside / "planted.json").exists()
    assert (root / "real.json").read_text() == "{}"


async def test_save_paper_refuses_a_non_regular_target(root):
    (root / "folder.json").mkdir()
    async with client_session(build_server(allowed_dirs=[root, FIXTURE.parent])) as client:
        _payload(await client.call_tool("load_paper", {"path": str(FIXTURE)}))
        text = _error_text(
            await client.call_tool(
                "save_paper",
                {"paper_id": FIXTURE_ID, "path": str(root / "folder.json"), "overwrite": True},
            )
        )
        assert "not a regular file" in text


async def test_save_paper_writes_inside_the_root(monkeypatch, root):
    monkeypatch.chdir(root)
    async with client_session(build_server(allowed_dirs=[root, FIXTURE.parent])) as client:
        _payload(await client.call_tool("load_paper", {"path": str(FIXTURE)}))
        saved = _payload(
            await client.call_tool(
                "save_paper", {"paper_id": FIXTURE_ID, "path": "out/export.json", "compact": True}
            )
        )
        written = root / "out" / "export.json"
        assert saved["bytes"] == written.stat().st_size
        assert json.loads(written.read_text()) == json.loads(FIXTURE.read_text())


def test_cli_passes_repeated_allow_dir(monkeypatch):
    from bibr.local.cli.parser import _build_parser

    args = _build_parser().parse_args(["mcp", "--allow-dir", "~/papers", "--allow-dir", "data"])
    assert args.allow_dir == ["~/papers", "data"]

    captured = {}

    class FakeServer:
        def run(self, transport):
            captured["transport"] = transport

    def fake_build_server(*, refs=None, settings=None, allowed_dirs=None, **options):
        captured["allowed_dirs"] = allowed_dirs
        return FakeServer()

    monkeypatch.setattr("bibr.mcp_server.build_server", fake_build_server)
    assert run_mcp(args) == 0
    assert captured["allowed_dirs"] == ["~/papers", "data"]


def test_missing_allow_dir_is_a_one_line_cli_error(monkeypatch, capsys, tmp_path):
    from bibr.local.cli import main

    missing = tmp_path / "nope"
    monkeypatch.setattr("sys.argv", ["bibr", "mcp", "--allow-dir", str(missing)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "not an existing directory" in err
    assert "Traceback" not in err


def test_instructions_name_the_allowed_directories(root):
    server = build_server(allowed_dirs=[root])
    assert str(root) in server.instructions
    assert "--allow-dir" in server.instructions


@pytest.mark.parametrize("start", ["filesystem root", "home", "project"])
def test_warns_when_the_default_scope_is_everything(monkeypatch, caplog, root, start):
    monkeypatch.setenv("HOME", str(root))
    cwd = {"filesystem root": Path(root.anchor), "home": root, "project": root / "project"}[start]
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    with caplog.at_level(logging.WARNING, logger="bibr.mcp_server"):
        build_server()
        assert ("--allow-dir to limit them" in caplog.text) == (start != "project")
        caplog.clear()
        build_server(allowed_dirs=[cwd])  # an explicit choice is not warned about
        assert "--allow-dir" not in caplog.text


# ---------------------------------------------------------------------------
# chew_url honours MCP_CHEW_URL_ENABLED / MCP_URL_ALLOWED_HOSTS
# ---------------------------------------------------------------------------


def _settings(**mcp):
    from bibr.config import snapshot_settings

    settings = snapshot_settings()
    for key, value in mcp.items():
        setattr(settings.mcp, key, value)
    return settings


async def test_chew_url_disabled_by_setting(root):
    server = build_server(allowed_dirs=[root], settings=_settings(chew_url_enabled=False))
    # (The tmp root's name carries the test name, so leave it out.)
    assert "chew_url" not in server.instructions.replace(str(root), "")
    async with client_session(server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert "chew_url" not in tools
        assert {"chew_paper", "load_paper", "save_paper"} <= tools


async def test_chew_url_refuses_hosts_outside_the_allowlist(root):
    server = build_server(allowed_dirs=[root], settings=_settings(url_allowed_hosts=["arxiv.org"]))
    async with client_session(server) as client:
        text = _error_text(
            await client.call_tool("chew_url", {"url": "https://evil.example/a.pdf?d=secret"})
        )
        assert "allowlist" in text


async def test_chew_url_passes_the_allowlist_to_the_fetch(monkeypatch, chews, root):
    import bibr.utils.safe_fetch as safe_fetch

    seen = {}

    async def fake_fetch(url, *, max_size, allowed_hosts=None, **kwargs):
        seen["allowed_hosts"] = allowed_hosts
        return safe_fetch.FetchedFile(
            content=b"%PDF-1.4 stub",
            filename="1234.5678.pdf",
            content_type="application/pdf",
            final_url=url,
        )

    monkeypatch.setattr(safe_fetch, "fetch_url_safely", fake_fetch)
    server = build_server(allowed_dirs=[root], settings=_settings(url_allowed_hosts=["arxiv.org"]))
    async with client_session(server) as client:
        _payload(await client.call_tool("chew_url", {"url": "https://export.arxiv.org/pdf/1"}))
    assert seen["allowed_hosts"] == ["arxiv.org"]


# ---------------------------------------------------------------------------
# load_paper reads only bounded regular files
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
async def test_load_paper_refuses_a_fifo_without_hanging(root):
    fifo = root / "pipe.json"
    os.mkfifo(fifo)
    try:
        async with client_session(build_server(allowed_dirs=[root])) as client:
            result = await asyncio.wait_for(
                client.call_tool("load_paper", {"path": str(fifo)}), timeout=10
            )
            assert "not a regular file" in _error_text(result)
    finally:
        # Release a reader stuck in open() (the old behaviour) so the worker
        # thread can finish.
        try:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass


@pytest.mark.skipif(not os.path.exists("/dev/null"), reason="needs /dev/null")
async def test_load_paper_refuses_a_device_file():
    # /dev/null, not the /dev/zero of the report: a regression must fail the
    # test, not exhaust the machine's memory.
    async with client_session(build_server(allowed_dirs=["/dev"])) as client:
        text = _error_text(await client.call_tool("load_paper", {"path": "/dev/null"}))
        assert "not a regular file" in text


async def test_load_paper_caps_the_file_size(monkeypatch, root):
    size = FIXTURE.stat().st_size
    shutil.copy(FIXTURE, root / "paper.json")
    async with client_session(build_server(allowed_dirs=[root])) as client:
        monkeypatch.setattr(bibr.mcp_server, "_LOAD_MAX_BYTES", size - 1)
        text = _error_text(await client.call_tool("load_paper", {"path": str(root / "paper.json")}))
        assert "load_paper limit" in text
        assert await _papers(client) == []

        monkeypatch.setattr(bibr.mcp_server, "_LOAD_MAX_BYTES", size)
        _payload(await client.call_tool("load_paper", {"path": str(root / "paper.json")}))


# ---------------------------------------------------------------------------
# Malformed exports
# ---------------------------------------------------------------------------


async def test_load_paper_summarizes_a_non_dict_extraction(root):
    cases = {
        "str.json": {"text": [], "extraction": "x"},
        "list.json": {
            "text": [],
            "extraction": [1],
            "validation": {"errors": 1, "warnings": 0, "issues": []},
        },
    }
    for name, data in cases.items():
        (root / name).write_text(json.dumps(data))
    async with client_session(build_server(allowed_dirs=[root])) as client:
        for name in cases:
            summary = _payload(await client.call_tool("load_paper", {"path": str(root / name)}))
            assert summary["llm_usage"] is None
            assert summary["enrichment"] == "not present"
        assert sorted(await _papers(client)) == ["list", "str"]


async def test_a_summary_failure_leaves_nothing_registered(monkeypatch, root):
    def broken(*args, **kwargs):
        raise RuntimeError("summary bug")

    monkeypatch.setattr(bibr.mcp_server, "_summarize", broken)
    shutil.copy(FIXTURE, root / "paper.json")
    async with client_session(build_server(allowed_dirs=[root])) as client:
        result = await client.call_tool("load_paper", {"path": str(root / "paper.json")})
        assert result.is_error
        assert await _papers(client) == []


# ---------------------------------------------------------------------------
# Startup without LLM credentials
# ---------------------------------------------------------------------------


def _no_credentials():
    from bibr.config import GlobalSettings

    settings = GlobalSettings(llm={"provider": "anthropic", "api_key": None})
    settings.ANTHROPIC_API_KEY = None
    return settings


async def test_server_starts_and_queries_without_llm_credentials(monkeypatch, caplog, chews, root):
    import bibr.utils.safe_fetch as safe_fetch

    async def no_fetch(url, **kwargs):
        raise AssertionError("chew_url must fail before downloading")

    monkeypatch.setattr(safe_fetch, "fetch_url_safely", no_fetch)
    with caplog.at_level(logging.WARNING, logger="bibr.mcp_server"):
        server = build_server(allowed_dirs=[root, FIXTURE.parent], settings=_no_credentials())
    assert "extraction is unavailable" in caplog.text
    assert "API key required" in caplog.text
    paper = root / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 stub")
    async with client_session(server) as client:
        _payload(await client.call_tool("load_paper", {"path": str(FIXTURE)}))
        meta = _payload(await client.call_tool("get_metadata", {"paper_id": FIXTURE_ID}))
        assert meta["metadata"]["title"]

        for tool, args in (
            ("chew_paper", {"path": str(paper)}),
            ("chew_url", {"url": "https://arxiv.org/pdf/1"}),
        ):
            text = _error_text(await client.call_tool(tool, args))
            assert "extraction is unavailable" in text
            assert "API key required" in text
    assert chews == []


@pytest.mark.parametrize("credentials", [True, False])
def test_a_bad_pipeline_option_still_stops_the_server(root, credentials):
    settings = None if credentials else _no_credentials()
    with pytest.raises(ValueError, match="ref_seg must be"):
        build_server(allowed_dirs=[root], settings=settings, ref_seg="bogus")


# ---------------------------------------------------------------------------
# Store recency
# ---------------------------------------------------------------------------


def test_paper_store_rechew_moves_paper_to_newest():
    store = _PaperStore(max_papers=2)
    store.add({"paper_id": "one"}, source="one.pdf")
    store.add({"paper_id": "two"}, source="two.pdf")
    store.add({"paper_id": "one"}, source="one.pdf")  # re-chewed
    store.add({"paper_id": "three"}, source="three.pdf")
    assert [pid for pid, _ in store.items()] == ["one", "three"]
