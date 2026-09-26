"""MCP sweep fixes: error detail, save_paper guards (wizard-mcp-demo-9/-18).

Drives the tool functions through the MCP in-memory client — the same path
a real stdio client exercises. ``chew_paper``/``chew_url`` run against a
monkeypatched ``Chewer.achew_file``; ``save_paper`` runs against the
``inspect_full_export.json`` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client as client_session  # noqa: E402

import bibr.api  # noqa: E402
from bibr.mcp_server import build_server  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "inspect_full_export.json"


def _payload(result):
    assert not result.is_error, [c.text for c in result.content]
    payload = result.structured_content
    assert payload is not None
    return payload["result"] if set(payload) == {"result"} else payload


def _error_text(result) -> str:
    assert result.is_error
    return result.content[0].text


async def test_chew_paper_reports_non_bibr_failure_detail(monkeypatch, tmp_path):
    async def failing_achew_file(self, path, *, paper_id=None, progress=None):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(bibr.api.Chewer, "achew_file", failing_achew_file)
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 stub")

    server = build_server()
    async with client_session(server) as client:
        text = _error_text(await client.call_tool("chew_paper", {"path": str(paper)}))
        assert "extraction failed for paper.pdf" in text
        assert "ImportError" in text
        assert "torch" in text


async def test_chew_paper_error_scrubs_secrets(monkeypatch, tmp_path):
    async def leaking_achew_file(self, path, *, paper_id=None, progress=None):
        raise RuntimeError(
            "request failed for https://api.example.com/v1?key=secret-token-value "
            "with Bearer sk-test-key-placeholder-abcdef"
        )

    monkeypatch.setattr(bibr.api.Chewer, "achew_file", leaking_achew_file)
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 stub")

    server = build_server()
    async with client_session(server) as client:
        text = _error_text(await client.call_tool("chew_paper", {"path": str(paper)}))
        assert "RuntimeError" in text
        assert "secret-token-value" not in text
        assert "sk-test-key-placeholder-abcdef" not in text
        assert "***" in text
        # Only the file name travels with the error, never the full path.
        assert str(tmp_path) not in text


async def test_chew_url_reports_non_bibr_failure_without_url(monkeypatch):
    import bibr.utils.safe_fetch as safe_fetch

    async def fake_fetch(url, *, max_size, allowed_hosts=None, **kwargs):
        return safe_fetch.FetchedFile(
            content=b"%PDF-1.4 stub",
            filename="1234.5678.pdf",
            content_type="application/pdf",
            final_url=url,
        )

    async def failing_achew_file(self, path, *, paper_id=None, progress=None):
        raise KeyError("stage bug")

    monkeypatch.setattr(safe_fetch, "fetch_url_safely", fake_fetch)
    monkeypatch.setattr(bibr.api.Chewer, "achew_file", failing_achew_file)

    server = build_server()
    async with client_session(server) as client:
        text = _error_text(
            await client.call_tool("chew_url", {"url": "https://arxiv.org/pdf/1234.5678"})
        )
        assert "KeyError" in text
        assert "1234.5678.pdf" in text
        assert "https://arxiv.org/pdf/1234.5678" not in text


async def test_chew_url_bibr_error_reports_filename_without_url(monkeypatch):
    import bibr.utils.safe_fetch as safe_fetch
    from bibr.exceptions import BibrError

    async def fake_fetch(url, *, max_size, allowed_hosts=None, **kwargs):
        return safe_fetch.FetchedFile(
            content=b"%PDF-1.4 stub",
            filename="1234.5678.pdf",
            content_type="application/pdf",
            final_url=url,
        )

    async def failing_achew_file(self, path, *, paper_id=None, progress=None):
        raise BibrError("parse failed with key=secret-token-value")

    monkeypatch.setattr(safe_fetch, "fetch_url_safely", fake_fetch)
    monkeypatch.setattr(bibr.api.Chewer, "achew_file", failing_achew_file)

    server = build_server()
    async with client_session(server) as client:
        text = _error_text(
            await client.call_tool("chew_url", {"url": "https://arxiv.org/pdf/1234.5678"})
        )
        assert "extraction failed for 1234.5678.pdf" in text
        assert "https://arxiv.org/pdf/1234.5678" not in text
        assert "secret-token-value" not in text


async def test_save_paper_requires_json_and_refuses_overwrite(tmp_path):
    server = build_server()
    async with client_session(server) as client:
        summary = _payload(await client.call_tool("load_paper", {"path": str(FIXTURE)}))
        pid = summary["paper_id"]

        text = _error_text(
            await client.call_tool(
                "save_paper", {"paper_id": pid, "path": str(tmp_path / "export.txt")}
            )
        )
        assert ".json" in text
        assert not (tmp_path / "export.txt").exists()

        target = tmp_path / "export.json"
        saved = _payload(
            await client.call_tool("save_paper", {"paper_id": pid, "path": str(target)})
        )
        assert saved["path"] == str(target)
        first = target.read_bytes()

        text = _error_text(
            await client.call_tool("save_paper", {"paper_id": pid, "path": str(target)})
        )
        assert "overwrite" in text
        assert target.read_bytes() == first

        saved = _payload(
            await client.call_tool(
                "save_paper", {"paper_id": pid, "path": str(target), "overwrite": True}
            )
        )
        assert saved["path"] == str(target)
