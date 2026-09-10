#!/usr/bin/env python3
"""Smoke-test a remote bibr MCP endpoint (``bibr serve`` with ``MCP_ENABLED=true``).

Performs the streamable-HTTP handshake, lists the tools, and — when ``--chew``
names a file — uploads it through ``chew_paper`` and reads back
``get_metadata`` + the first page of ``get_references``, printing compact
summaries. Exit status is non-zero on any failure, so it doubles as a
readiness gate for scripted deployments.

    uv run python scripts/mcp_smoke.py --url http://127.0.0.1:8000/mcp \
        --token "$AUTH_API_KEY" --chew tests/fixtures/native_text_sample.pdf

Needs the ``mcp`` extra (``uv sync --extra mcp``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path


def _text(result) -> str:
    """Concatenate the text blocks of a CallToolResult."""
    return "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "text", None))


def _parse(result) -> dict | list | str:
    raw = _text(result)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


async def run(url: str, token: str, chew: Path | None, refs: str | None, timeout: float) -> int:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with (
        streamablehttp_client(url, headers=headers, timeout=timeout) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        print(f"server    : {init.serverInfo.name} {init.serverInfo.version}")
        tools = await session.list_tools()
        names = sorted(t.name for t in tools.tools)
        print(f"tools ({len(names)}): {', '.join(names)}")
        missing = {"chew_paper", "get_metadata", "get_references"} - set(names)
        if missing:
            print(f"FAIL: missing tools {sorted(missing)}", file=sys.stderr)
            return 1
        if chew is None:
            return 0

        data = chew.read_bytes()
        args: dict = {"filename": chew.name, "content_base64": base64.b64encode(data).decode()}
        if refs:
            args["refs"] = refs
        print(f"chew_paper: {chew.name} ({len(data)} bytes) ...", flush=True)
        t0 = time.monotonic()
        res = await session.call_tool("chew_paper", args)
        elapsed = time.monotonic() - t0
        if res.isError:
            print(
                f"FAIL: chew_paper error after {elapsed:.0f}s: {_text(res)[:600]}",
                file=sys.stderr,
            )
            return 1
        summary = _parse(res)
        if not isinstance(summary, dict) or "paper_id" not in summary:
            print(f"FAIL: unexpected chew_paper payload: {str(summary)[:400]}", file=sys.stderr)
            return 1
        paper_id = summary["paper_id"]
        print(f"chewed    : paper_id={paper_id} in {elapsed:.0f}s")
        print(f"summary   : {json.dumps(summary)[:500]}")

        meta = _parse(await session.call_tool("get_metadata", {"paper_id": paper_id}))
        if isinstance(meta, dict):
            info = meta.get("info") or meta
            print(f"title     : {str(info.get('title'))[:100]}")
            print(f"doi       : {info.get('doi')}")
            print(f"authors   : {len(meta.get('authors') or meta.get('author') or [])}")
        bib = _parse(await session.call_tool("get_references", {"paper_id": paper_id, "limit": 5}))
        if isinstance(bib, dict):
            print(
                f"references: total={bib.get('total', '?')} first={json.dumps(bib.get('references', bib.get('items', []))[:1])[:300]}"
            )
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="MCP endpoint, e.g. http://127.0.0.1:8000/mcp")
    ap.add_argument("--token", default="", help="bearer token (AUTH_API_KEY of the serve)")
    ap.add_argument("--chew", type=Path, help="PDF/DOCX to upload through chew_paper")
    ap.add_argument("--refs", choices=["ner", "llm", "llm-chunked", "off"], help="refs option")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request HTTP timeout (s)")
    args = ap.parse_args()
    if args.chew is not None and not args.chew.is_file():
        print(f"no such file: {args.chew}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(run(args.url, args.token, args.chew, args.refs, args.timeout))
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not raise
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
