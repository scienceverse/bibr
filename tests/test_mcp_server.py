"""Tests for the ``bibr mcp`` MCP server (``bibr/mcp_server.py``).

All tests drive the server through the MCP SDK's in-memory client/server
transport — the same code path a real stdio client exercises, minus the
process boundary. ``chew_paper`` is covered with a monkeypatched
``Chewer.achew_file`` (the real pipeline needs models and credentials);
everything else runs against the ``inspect_full_export.json`` fixture.
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client as client_session  # noqa: E402

import bibr.api  # noqa: E402
from bibr.mcp_server import build_server, run_mcp  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "inspect_full_export.json"
FIXTURE_ID = "10.1234/example.5678"


def _payload(result):
    """Unwrap a successful call_tool result into its structured payload."""
    assert not result.is_error, [c.text for c in result.content]
    payload = result.structured_content
    assert payload is not None
    # FastMCP wraps non-dict returns (lists) under a "result" key.
    return payload["result"] if set(payload) == {"result"} else payload


def _error_text(result) -> str:
    assert result.is_error
    return result.content[0].text


@asynccontextmanager
async def open_session(*, load: bool = False):
    """Fresh in-memory server + client; optionally pre-load the fixture.

    A context manager rather than an async fixture: the MCP memory transport
    is anyio-task-group based, and pytest-asyncio tears async generator
    fixtures down in a different task, which trips anyio's cancel-scope
    ownership check.
    """
    server = build_server()
    async with client_session(server) as client:
        if load:
            _payload(await client.call_tool("load_paper", {"path": str(FIXTURE)}))
        yield client


async def test_tool_listing():
    async with open_session(load=False) as session:
        tools = {t.name for t in (await session.list_tools()).tools}
        assert tools == {
            "chew_paper",
            "chew_url",
            "load_paper",
            "list_papers",
            "get_paper_summary",
            "get_metadata",
            "get_sections",
            "get_text",
            "search_text",
            "get_references",
            "get_reference_citations",
            "get_tables",
            "get_figures",
            "save_paper",
        }


async def test_load_paper_returns_inspect_style_summary():
    async with open_session(load=False) as session:
        summary = _payload(await session.call_tool("load_paper", {"path": str(FIXTURE)}))
        assert summary["paper_id"] == FIXTURE_ID
        assert summary["title"] == "Deep Learning for Something Great"
        assert summary["counts"] == {
            "sections": 3,
            "sentences": 5,
            "references": 2,
            "tables": 1,
            "figures": 1,
            "equations": 1,
        }
        assert summary["authors"].startswith("4 (")
        assert "references cited" in summary["in_text_citations"]
        assert "enriched" in summary["enrichment"]
        assert any(row["model"] == "gemini-2.5-flash" for row in summary["llm_usage"]["breakdown"])


async def test_load_paper_rejects_non_export(tmp_path):
    async with open_session(load=False) as session:
        bogus = tmp_path / "openapi.json"
        bogus.write_text(json.dumps({"info": {"title": "Some API"}, "paths": {}}))
        assert "does not look like a bibr export" in _error_text(
            await session.call_tool("load_paper", {"path": str(bogus)})
        )
        assert "cannot read" in _error_text(
            await session.call_tool("load_paper", {"path": str(tmp_path / "missing.json")})
        )
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert "not valid JSON" in _error_text(
            await session.call_tool("load_paper", {"path": str(broken)})
        )


async def test_unknown_paper_id_lists_loaded_papers():
    async with open_session(load=True) as loaded:
        text = _error_text(await loaded.call_tool("get_text", {"paper_id": "nope"}))
        assert "unknown paper_id 'nope'" in text
        assert FIXTURE_ID in text


async def test_list_papers():
    async with open_session(load=True) as loaded:
        papers = _payload(await loaded.call_tool("list_papers", {}))
        assert len(papers) == 1
        assert papers[0]["paper_id"] == FIXTURE_ID
        assert papers[0]["references"] == 2
        assert papers[0]["source"] == str(FIXTURE)


async def test_get_metadata():
    async with open_session(load=True) as loaded:
        meta = _payload(await loaded.call_tool("get_metadata", {"paper_id": FIXTURE_ID}))
        assert meta["metadata"]["title"] == "Deep Learning for Something Great"
        assert len(meta["authors"]) == 4
        assert meta["authors"][0]["family"] == "Doe"


async def test_get_sections_with_sentence_counts():
    async with open_session(load=True) as loaded:
        sections = _payload(await loaded.call_tool("get_sections", {"paper_id": FIXTURE_ID}))
        assert [s["section_id"] for s in sections] == [1, 2, 3]
        assert all(s["section_type"] for s in sections)
        assert sum(s["sentences"] for s in sections) == 5


async def test_get_text_filters_and_paginates():
    async with open_session(load=True) as loaded:
        everything = _payload(await loaded.call_tool("get_text", {"paper_id": FIXTURE_ID}))
        assert everything["total"] == 5
        assert len(everything["sentences"]) == 5
        assert {"text_id", "text", "section_id", "paragraph_id", "page_number"} == set(
            everything["sentences"][0]
        )

        section = _payload(
            await loaded.call_tool("get_text", {"paper_id": FIXTURE_ID, "section_id": 1})
        )
        assert section["total"] == 2
        assert all(s["section_id"] == 1 for s in section["sentences"])

        window = _payload(
            await loaded.call_tool("get_text", {"paper_id": FIXTURE_ID, "offset": 4, "limit": 3})
        )
        assert window["total"] == 5
        assert window["returned"] == 1


async def test_search_text():
    async with open_session(load=True) as loaded:
        needle = json.loads(FIXTURE.read_text())["text"][0]["text"].split()[0]
        found = _payload(
            await loaded.call_tool("search_text", {"paper_id": FIXTURE_ID, "query": needle.lower()})
        )
        assert found["total_matches"] >= 1
        assert needle.lower() in found["matches"][0]["text"].lower()
        assert "non-empty" in _error_text(
            await loaded.call_tool("search_text", {"paper_id": FIXTURE_ID, "query": ""})
        )


async def test_get_references_drops_empty_fields():
    async with open_session(load=True) as loaded:
        refs = _payload(await loaded.call_tool("get_references", {"paper_id": FIXTURE_ID}))
        assert refs["total"] == 2
        assert len(refs["references"]) == 2
        for row in refs["references"]:
            assert None not in row.values()
            assert "bib_id" in row


async def test_get_reference_citations():
    async with open_session(load=True) as loaded:
        linked = _payload(
            await loaded.call_tool("get_reference_citations", {"paper_id": FIXTURE_ID, "bib_id": 1})
        )
        assert linked["reference"]["bib_id"] == 1
        # Fixture has two in-text citations of bib 1 (text_id 3 and 4); the
        # table xref with the same id must not leak in.
        assert [c["text_id"] for c in linked["citations"]] == [3, 4]
        assert all(c["text"] for c in linked["citations"])
        assert "no reference with bib_id 99" in _error_text(
            await loaded.call_tool(
                "get_reference_citations", {"paper_id": FIXTURE_ID, "bib_id": 99}
            )
        )


async def test_get_tables_listing_and_detail():
    async with open_session(load=True) as loaded:
        listing = _payload(await loaded.call_tool("get_tables", {"paper_id": FIXTURE_ID}))
        assert len(listing["tables"]) == 1
        assert "html" not in listing["tables"][0]
        table_id = listing["tables"][0]["table_id"]
        detail = _payload(
            await loaded.call_tool("get_tables", {"paper_id": FIXTURE_ID, "table_id": table_id})
        )
        assert detail["table"]["table_id"] == table_id
        assert "html" in detail["table"]


async def test_get_figures_never_inline_image_data():
    async with open_session(load=True) as loaded:
        listing = _payload(await loaded.call_tool("get_figures", {"paper_id": FIXTURE_ID}))
        figure = listing["figures"][0]
        assert "image" not in figure
        assert isinstance(figure["has_image"], bool)
        detail = _payload(
            await loaded.call_tool(
                "get_figures", {"paper_id": FIXTURE_ID, "figure_id": figure["figure_id"]}
            )
        )
        assert "image" not in detail["figure"]


async def test_save_paper_round_trips(tmp_path):
    async with open_session(load=True) as loaded:
        out = tmp_path / "nested" / "export.json"
        saved = _payload(
            await loaded.call_tool("save_paper", {"paper_id": FIXTURE_ID, "path": str(out)})
        )
        assert saved["path"] == str(out)
        assert saved["bytes"] == out.stat().st_size
        assert json.loads(out.read_text()) == json.loads(FIXTURE.read_text())


async def test_same_id_different_source_gets_suffix(tmp_path):
    async with open_session(load=False) as session:
        copy = tmp_path / "copy.json"
        copy.write_text(FIXTURE.read_text())
        first = _payload(await session.call_tool("load_paper", {"path": str(FIXTURE)}))
        second = _payload(await session.call_tool("load_paper", {"path": str(copy)}))
        reload_first = _payload(await session.call_tool("load_paper", {"path": str(FIXTURE)}))
        assert first["paper_id"] == FIXTURE_ID
        assert second["paper_id"] == f"{FIXTURE_ID}-2"
        assert reload_first["paper_id"] == FIXTURE_ID
        assert len(_payload(await session.call_tool("list_papers", {}))) == 2


async def test_chew_paper_runs_warm_pipeline(monkeypatch, tmp_path):
    data = json.loads(FIXTURE.read_text())
    seen: dict[str, object] = {}

    async def fake_achew_file(self, path, *, paper_id=None, progress=None):
        seen["path"] = Path(path)
        seen["paper_id"] = paper_id
        seen["progress"] = progress
        return bibr.api.Result(data)

    monkeypatch.setattr(bibr.api.Chewer, "achew_file", fake_achew_file)
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 stub")

    server = build_server()
    async with client_session(server) as client:
        summary = _payload(
            await client.call_tool("chew_paper", {"path": str(paper), "paper_id": "my-id"})
        )
        assert summary["paper_id"] == "my-id"
        assert summary["source"] == str(paper)
        assert summary["counts"]["references"] == 2
        assert isinstance(summary["seconds"], float)
        assert seen["path"] == paper
        assert seen["paper_id"] == "my-id"
        # The MCP progress bridge is handed to the pipeline.
        assert hasattr(seen["progress"], "stage_start")

        # The chewed paper is registered for the query tools.
        meta = _payload(await client.call_tool("get_metadata", {"paper_id": "my-id"}))
        assert meta["metadata"]["title"] == "Deep Learning for Something Great"


async def test_chew_paper_input_errors(tmp_path):
    async with open_session(load=False) as session:
        assert "file not found" in _error_text(
            await session.call_tool("chew_paper", {"path": str(tmp_path / "gone.pdf")})
        )
        assert "is a directory" in _error_text(
            await session.call_tool("chew_paper", {"path": str(tmp_path)})
        )


async def test_chew_url_fetches_then_chews(monkeypatch):
    import bibr.utils.safe_fetch as safe_fetch

    data = json.loads(FIXTURE.read_text())
    seen: dict[str, object] = {}

    async def fake_fetch(url, *, max_size, allowed_hosts=None, **kwargs):
        seen["url"] = url
        seen["max_size"] = max_size
        return safe_fetch.FetchedFile(
            content=b"%PDF-1.4 stub",
            filename="1234.5678.pdf",
            content_type="application/pdf",
            final_url=url,
        )

    async def fake_achew_file(self, path, *, paper_id=None, progress=None):
        seen["chewed_path"] = Path(path)
        return bibr.api.Result(data)

    monkeypatch.setattr(safe_fetch, "fetch_url_safely", fake_fetch)
    monkeypatch.setattr(bibr.api.Chewer, "achew_file", fake_achew_file)

    async with open_session(load=False) as session:
        summary = _payload(
            await session.call_tool("chew_url", {"url": "https://arxiv.org/pdf/1234.5678"})
        )
        # The download lands in a temp dir under its derived filename, and the
        # registered source is the URL, not the temp path.
        assert seen["url"] == "https://arxiv.org/pdf/1234.5678"
        assert seen["chewed_path"].name == "1234.5678.pdf"
        assert summary["source"] == "https://arxiv.org/pdf/1234.5678"
        assert summary["paper_id"] == FIXTURE_ID
        assert not seen["chewed_path"].exists()  # temp dir cleaned up

        papers = _payload(await session.call_tool("list_papers", {}))
        assert papers[0]["source"] == "https://arxiv.org/pdf/1234.5678"


async def test_chew_url_refuses_unsafe_urls_before_any_network():
    async with open_session(load=False) as session:
        assert "only https" in _error_text(
            await session.call_tool("chew_url", {"url": "http://arxiv.org/pdf/1234.pdf"})
        )
        assert "non-public" in _error_text(
            await session.call_tool("chew_url", {"url": "https://169.254.169.254/latest/meta-data"})
        )


async def test_chew_paper_wraps_bibr_errors(monkeypatch, tmp_path):
    from bibr.exceptions import ProcessingError

    async def failing_achew_file(self, path, *, paper_id=None, progress=None):
        raise ProcessingError("OCR backend unreachable")

    monkeypatch.setattr(bibr.api.Chewer, "achew_file", failing_achew_file)
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 stub")

    server = build_server()
    async with client_session(server) as client:
        text = _error_text(await client.call_tool("chew_paper", {"path": str(paper)}))
        assert "extraction failed for paper.pdf" in text
        assert "OCR backend unreachable" in text


def test_run_mcp_translates_cli_flags(monkeypatch):
    captured: dict[str, object] = {}

    class FakeServer:
        def run(self, transport):
            captured["transport"] = transport

    def fake_build_server(*, refs=None, settings=None, **options):
        captured["refs"] = refs
        captured["options"] = options
        return FakeServer()

    monkeypatch.setattr("bibr.mcp_server.build_server", fake_build_server)
    args = argparse.Namespace(
        ocr="paddle",
        llm=None,
        memory=None,
        device=None,
        ocr_url=None,
        ocr_model=None,
        ocr_profile=None,
        no_crossref=True,
        no_equations=False,
        no_llm=True,
        figure_images=False,
        consolidate="fill",
        ref_seg=None,
        refs="ner",
        verbose=False,
    )
    assert run_mcp(args) == 0
    assert captured["transport"] == "stdio"
    assert captured["refs"] == "ner"
    assert captured["options"] == {
        "ocr": "paddle",
        "crossref": False,
        "no_llm": True,
        "consolidate": "fill",
    }


def test_run_mcp_maps_crossref_flag_to_forced_on(monkeypatch):
    """``bibr mcp --crossref`` forces enrichment on for the warm pipeline."""
    import argparse

    captured = {}

    class FakeServer:
        def run(self, transport):
            captured["transport"] = transport

    def fake_build_server(*, refs=None, settings=None, **options):
        captured["options"] = options
        return FakeServer()

    monkeypatch.setattr("bibr.mcp_server.build_server", fake_build_server)
    args = argparse.Namespace(crossref=True, no_crossref=False, verbose=False)
    assert run_mcp(args) == 0
    assert captured["options"] == {"crossref": True}

    # Neither flag → the option is absent, so CROSSREF_ENRICH decides.
    args = argparse.Namespace(crossref=False, no_crossref=False, verbose=False)
    assert run_mcp(args) == 0
    assert captured["options"] == {}


def test_cli_parser_accepts_mcp_subcommand():
    from bibr.local.cli.parser import _build_parser

    args = _build_parser().parse_args(["mcp", "--no-llm", "--refs", "ner"])
    assert args.command == "mcp"
    assert args.no_llm is True
    assert args.refs == "ner"
