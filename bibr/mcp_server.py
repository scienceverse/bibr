"""``bibr mcp`` — Model Context Protocol server for agentic extraction.

Exposes the extraction pipeline as MCP tools over stdio so agents (Claude
Code, Claude Desktop, any MCP client) can chew papers and query the results
without shelling out to the CLI or parsing whole export files.

A full export is far too large for a single tool result (sentence-level
text spans, table HTML, optionally base64 figure images), so the surface
follows a chew-once / query-granularly contract: ``chew_paper`` runs the
pipeline and returns only a compact summary; the ``get_*`` and
``search_text`` tools then read slices of the stored export on demand.
``load_paper`` admits exports produced earlier (CLI, HTTP API) without
re-processing, and ``save_paper`` persists a chewed result as JSON.

State is in-memory per server process: papers live in a dict keyed by paper
id and vanish on exit. The pipeline is a single warm :class:`bibr.api.Chewer`
(models load once, then every ``chew_paper`` call reuses them). Pipeline
options (OCR/LLM backend, reference strategy, ...) are fixed at server start
via ``bibr mcp`` flags — the same constraint :class:`~bibr.api.Chewer` has,
because they are pipeline-constructor arguments.

stdio discipline: the MCP transport owns the real stdout (JSON-RPC frames),
so ``chew_paper`` runs the pipeline under ``redirect_stdout(sys.stderr)`` —
a stray ``print`` in a backend would otherwise corrupt the frame stream. The
transport keeps its own reference to the original stream from server start,
so the redirect never touches it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Unpack

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from bibr.api import Chewer, ChewOptions
from bibr.exceptions import BibrError
from bibr.validation import payload_validation

__all__ = ["build_server", "run_mcp"]

_TEXT_LIMIT_MAX = 500
_REF_LIMIT_MAX = 200
_SEARCH_LIMIT_MAX = 100
_URL_MAX_BYTES = 100 * 1024 * 1024  # chew_url download cap (stdio server)

_INSTRUCTIONS = """\
bibr extracts structured data from scientific papers (PDF, DOCX, JATS XML, HTML, ePub):
metadata, authors, full text (sentence-level, linked to sections and pages), references,
in-text citations, tables, figures, and statistical expressions.

Start with chew_paper (a local file), chew_url (a public https:// URL), or load_paper
(registers an existing bibr JSON export without re-processing). Extraction runs the full
pipeline — the first call may take minutes while models load. All three return a compact
summary and a paper_id for the query tools: get_metadata, get_sections, get_text,
search_text, get_references, get_reference_citations, get_tables, get_figures. Full
exports are large, so query the slices you need instead of asking for everything;
save_paper writes the complete export JSON to disk.
"""


def _drop_empty(row: dict[str, Any]) -> dict[str, Any]:
    """Trim a row for tool output: drop keys whose value carries no signal."""
    return {k: v for k, v in row.items() if v not in (None, "", [])}


def _text_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in ("text_id", "text", "section_id", "paragraph_id", "page_number")
    }


def _summarize(paper_id: str, data: dict[str, Any], source: str) -> dict[str, Any]:
    """Compact chew/load summary — same signals as ``bibr inspect``, as JSON."""
    from bibr.local.cli import _format_validation_line
    from bibr.local.inspect import _authors_line, _citation_coverage, _enrichment_state

    info = data.get("metadata")
    if not isinstance(info, dict):
        info = {}

    def count(key: str) -> int | None:
        value = data.get(key)
        return len(value) if isinstance(value, list) else None

    if payload_validation(data) is not None:
        line = _format_validation_line(data)
        validation = line.strip() if line else "clean (0 errors, 0 warnings)"
    else:
        validation = "not present"

    llm_usage = (data.get("extraction") or {}).get("usage")
    return {
        "paper_id": paper_id,
        "source": source,
        "title": info.get("title"),
        "doi": info.get("doi"),
        "paper_type": info.get("paper_type"),
        "authors": _authors_line(data),
        "counts": {
            "sections": count("section"),
            "sentences": count("text"),
            "references": count("bib"),
            "tables": count("table"),
            "figures": count("figure"),
            "equations": count("eq"),
        },
        "in_text_citations": _citation_coverage(data),
        "enrichment": _enrichment_state(data),
        "validation": validation,
        "llm_usage": llm_usage if isinstance(llm_usage, dict) else None,
    }


class _Entry:
    __slots__ = ("data", "source")

    def __init__(self, data: dict[str, Any], source: str):
        self.data = data
        self.source = source


class _PaperStore:
    """In-memory session store: paper id → export dict + source path.

    ``max_papers`` bounds memory for long-lived multi-client deployments
    (serve): adding beyond it evicts the oldest entry. ``None`` (stdio) keeps
    everything for the process lifetime.
    """

    def __init__(self, max_papers: int | None = None) -> None:
        self._papers: dict[str, _Entry] = {}
        self._max_papers = max_papers

    def add(self, data: dict[str, Any], *, source: str, requested_id: str | None = None) -> str:
        base = str(requested_id or data.get("paper_id") or Path(source).stem or "paper")
        paper_id, n = base, 2
        # Re-chewing the same file overwrites its entry; a different file that
        # happens to carry the same id gets a "-2"/"-3" suffix instead.
        while paper_id in self._papers and self._papers[paper_id].source != source:
            paper_id = f"{base}-{n}"
            n += 1
        if self._max_papers is not None and paper_id not in self._papers:
            while len(self._papers) >= self._max_papers:
                self._papers.pop(next(iter(self._papers)))
        self._papers[paper_id] = _Entry(data, source)
        return paper_id

    def get(self, paper_id: str) -> _Entry:
        entry = self._papers.get(paper_id)
        if entry is None:
            known = ", ".join(self._papers) or "none (use chew_paper or load_paper first)"
            raise ToolError(f"unknown paper_id {paper_id!r}; loaded papers: {known}")
        return entry

    def rows(self, paper_id: str, key: str) -> list[dict[str, Any]]:
        value = self.get(paper_id).data.get(key)
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []

    def items(self) -> list[tuple[str, _Entry]]:
        return list(self._papers.items())


class _McpProgress:
    """Bridge the pipeline's sync ``ProgressTracker`` onto MCP notifications.

    Tracker methods can fire from worker threads (OCR executors), so every
    notification is scheduled onto the server loop with
    ``call_soon_threadsafe`` and dropped if the loop is gone. Stage totals
    use the full ``STAGES`` list, so a run whose plan skips stages simply
    ends below 100% — fine for a progress display. OCR region updates are
    throttled to whole-percent steps.
    """

    def __init__(self, ctx: Context, loop: asyncio.AbstractEventLoop):
        self._ctx = ctx
        self._loop = loop
        self._stages_done = 0
        self._ocr_total = 0
        self._ocr_done = 0
        self._ocr_last_pct = -1

    def _post(self, coro: Any) -> None:
        try:
            self._loop.call_soon_threadsafe(asyncio.ensure_future, coro)
        except RuntimeError:
            coro.close()

    def _report(self, progress: float, message: str) -> None:
        from bibr.pipeline.progress import STAGES

        self._post(self._ctx.report_progress(progress, len(STAGES), message))

    def stage_start(self, name: str, detail: str = "") -> None:
        from bibr.pipeline.progress import _stage_label

        label = _stage_label(name)
        if detail:
            label = f"{label} — {detail}"
        self._report(self._stages_done, label)

    def stage_end(self, name: str) -> None:  # noqa: ARG002 — tracker protocol
        self._stages_done += 1

    def ocr_start(self, total_regions: int) -> None:
        self._ocr_total = total_regions
        self._ocr_done = 0
        self._ocr_last_pct = -1

    def ocr_region_done(self) -> None:
        self._ocr_done += 1
        if not self._ocr_total:
            return
        pct = (100 * self._ocr_done) // self._ocr_total
        if pct == self._ocr_last_pct:
            return
        self._ocr_last_pct = pct
        self._report(
            self._stages_done + self._ocr_done / self._ocr_total,
            f"Reading text (OCR) — {self._ocr_done}/{self._ocr_total} regions",
        )

    def ocr_end(self) -> None:
        pass


def _register_query_tools(server: MCPServer, get_store: Callable[[Context], _PaperStore]) -> None:
    """Register the read-only query tools shared by every transport.

    ``get_store`` resolves the paper store for the calling client: a single
    shared store for the stdio server (one agent per process), a per-MCP-
    session store over serve HTTP (see ``bibr.serve.mcp``). The ``ctx``
    parameters are injected by MCPServer and hidden from the tool schemas.
    """

    @server.tool()
    async def list_papers(*, ctx: Context) -> list[dict[str, Any]]:
        """List papers loaded in this session, with id, title, DOI, and source."""
        out = []
        for pid, entry in get_store(ctx).items():
            info = entry.data.get("metadata")
            info = info if isinstance(info, dict) else {}
            bib = entry.data.get("bib")
            out.append(
                {
                    "paper_id": pid,
                    "title": info.get("title"),
                    "doi": info.get("doi"),
                    "source": entry.source,
                    "references": len(bib) if isinstance(bib, list) else None,
                }
            )
        return out

    @server.tool()
    async def get_paper_summary(paper_id: str, *, ctx: Context) -> dict[str, Any]:
        """The compact summary (title, counts, validation, LLM usage) for one paper."""
        entry = get_store(ctx).get(paper_id)
        return _summarize(paper_id, entry.data, entry.source)

    @server.tool()
    async def get_metadata(paper_id: str, *, ctx: Context) -> dict[str, Any]:
        """Paper-level metadata: the full metadata block (title, abstract, DOI, journal,
        paper type, research-integrity statements, ...), authors, affiliations, funding."""
        entry = get_store(ctx).get(paper_id)
        return {
            "paper_id": paper_id,
            "metadata": entry.data.get("metadata"),
            "authors": entry.data.get("author"),
            "affiliations": entry.data.get("affiliation"),
            "funding": entry.data.get("funding"),
        }

    @server.tool()
    async def get_sections(paper_id: str, *, ctx: Context) -> list[dict[str, Any]]:
        """Section headers with hierarchy (level, parent), IMRaD-style section_type,
        and per-section sentence counts. Use section_id with get_text."""
        from collections import Counter

        store = get_store(ctx)
        counts = Counter(t.get("section_id") for t in store.rows(paper_id, "text"))
        return [
            {
                "section_id": row.get("section_id"),
                "header": row.get("header"),
                "level": row.get("level"),
                "parent_section_id": row.get("parent_section_id"),
                "section_type": row.get("section_type"),
                "sentences": counts.get(row.get("section_id"), 0),
            }
            for row in store.rows(paper_id, "section")
        ]

    @server.tool()
    async def get_text(
        paper_id: str,
        section_id: int | None = None,
        page: int | None = None,
        offset: int = 0,
        limit: int = 200,
        *,
        ctx: Context,
    ) -> dict[str, Any]:
        """Body text as ordered sentence spans, each with text_id, section_id,
        paragraph_id, and page_number. Filter by section_id and/or page; paginate
        with offset/limit (limit is capped at 500)."""
        rows = get_store(ctx).rows(paper_id, "text")
        if section_id is not None:
            rows = [r for r in rows if r.get("section_id") == section_id]
        if page is not None:
            rows = [r for r in rows if r.get("page_number") == page]
        offset = max(0, offset)
        limit = max(1, min(limit, _TEXT_LIMIT_MAX))
        window = rows[offset : offset + limit]
        return {
            "paper_id": paper_id,
            "total": len(rows),
            "offset": offset,
            "returned": len(window),
            "sentences": [_text_row(r) for r in window],
        }

    @server.tool()
    async def search_text(
        paper_id: str, query: str, limit: int = 20, *, ctx: Context
    ) -> dict[str, Any]:
        """Case-insensitive substring search over the paper's sentences; each match
        carries its text_id/section_id/page_number for follow-up queries."""
        needle = query.lower()
        if not needle:
            raise ToolError("query must be non-empty")
        limit = max(1, min(limit, _SEARCH_LIMIT_MAX))
        matches = [
            r
            for r in get_store(ctx).rows(paper_id, "text")
            if needle in str(r.get("text", "")).lower()
        ]
        return {
            "paper_id": paper_id,
            "query": query,
            "total_matches": len(matches),
            "returned": min(limit, len(matches)),
            "matches": [_text_row(r) for r in matches[:limit]],
        }

    @server.tool()
    async def get_references(
        paper_id: str, offset: int = 0, limit: int = 50, *, ctx: Context
    ) -> dict[str, Any]:
        """Parsed bibliography entries, verbatim from the printed reference list
        (authors, title, year, DOI, container, ...). Paginate with offset/limit
        (limit capped at 200); empty fields are omitted from each row."""
        rows = get_store(ctx).rows(paper_id, "bib")
        offset = max(0, offset)
        limit = max(1, min(limit, _REF_LIMIT_MAX))
        window = rows[offset : offset + limit]
        return {
            "paper_id": paper_id,
            "total": len(rows),
            "offset": offset,
            "returned": len(window),
            "references": [_drop_empty(r) for r in window],
        }

    @server.tool()
    async def get_reference_citations(
        paper_id: str, bib_id: int, *, ctx: Context
    ) -> dict[str, Any]:
        """Where a reference is cited: every in-text citation of the given bib_id,
        with the citation marker and the full sentence (text_id, section, page) it
        appears in — extracted facts trace back to source sentences."""
        store = get_store(ctx)
        ref = next((r for r in store.rows(paper_id, "bib") if r.get("bib_id") == bib_id), None)
        if ref is None:
            raise ToolError(f"no reference with bib_id {bib_id} in paper {paper_id!r}")
        text_by_id = {t.get("text_id"): t for t in store.rows(paper_id, "text")}
        citations = []
        for x in store.rows(paper_id, "xref"):
            if x.get("xref_type") != "bib" or x.get("target_id") != bib_id:
                continue
            sentence = text_by_id.get(x.get("text_id")) or {}
            citations.append(
                {
                    "marker": x.get("contents"),
                    "text_id": x.get("text_id"),
                    "text": sentence.get("text"),
                    "section_id": sentence.get("section_id"),
                    "page_number": sentence.get("page_number"),
                }
            )
        return {"paper_id": paper_id, "reference": _drop_empty(ref), "citations": citations}

    @server.tool()
    async def get_tables(
        paper_id: str, table_id: int | None = None, *, ctx: Context
    ) -> dict[str, Any]:
        """Extracted tables. Without table_id: id/label/caption/page per table.
        With table_id: the full table including HTML markup and structured cells."""
        rows = get_store(ctx).rows(paper_id, "table")
        if table_id is None:
            return {
                "paper_id": paper_id,
                "tables": [
                    {
                        "table_id": r.get("table_id"),
                        "label": r.get("label"),
                        "caption": r.get("caption"),
                        "page_number": r.get("page_number"),
                        "section_id": r.get("section_id"),
                    }
                    for r in rows
                ],
            }
        row = next((r for r in rows if r.get("table_id") == table_id), None)
        if row is None:
            raise ToolError(f"no table with table_id {table_id} in paper {paper_id!r}")
        return {"paper_id": paper_id, "table": row}

    @server.tool()
    async def get_figures(
        paper_id: str, figure_id: int | None = None, *, ctx: Context
    ) -> dict[str, Any]:
        """Extracted figures (caption, page, section). Image data is never inlined —
        has_image says whether the export carries it; the full export JSON has the pixels."""

        def strip(row: dict[str, Any]) -> dict[str, Any]:
            slim = {k: v for k, v in row.items() if k != "image"}
            slim["has_image"] = bool(row.get("image"))
            return slim

        rows = get_store(ctx).rows(paper_id, "figure")
        if figure_id is None:
            return {"paper_id": paper_id, "figures": [strip(r) for r in rows]}
        row = next((r for r in rows if r.get("figure_id") == figure_id), None)
        if row is None:
            raise ToolError(f"no figure with figure_id {figure_id} in paper {paper_id!r}")
        return {"paper_id": paper_id, "figure": strip(row)}


def build_server(
    *,
    refs: str | bool | None = None,
    settings: Any = None,
    **options: Unpack[ChewOptions],
) -> MCPServer:
    """Build the bibr MCP server; options mirror :class:`bibr.api.Chewer`."""
    chewer = Chewer(refs=refs, settings=settings, **options)
    store = _PaperStore()
    chew_lock = asyncio.Lock()

    @asynccontextmanager
    async def _lifespan(_server: MCPServer):
        try:
            yield None
        finally:
            await chewer.aclose()

    server = MCPServer("bibr", instructions=_INSTRUCTIONS, lifespan=_lifespan)

    @server.tool()
    async def chew_paper(path: str, paper_id: str | None = None, *, ctx: Context) -> dict[str, Any]:
        """Extract a paper (PDF/DOCX/JATS XML/HTML/ePub) and return a compact summary.

        Runs the full bibr pipeline on one file and registers the result for
        the get_*/search_text tools under the returned paper_id. The first
        call loads models and can take minutes; later calls reuse the warm
        pipeline. Progress is reported via MCP progress notifications.
        """
        target = Path(path).expanduser()
        if target.is_dir():
            raise ToolError(f"{target} is a directory — chew one file at a time")
        if not target.is_file():
            raise ToolError(f"file not found: {target}")
        tracker = _McpProgress(ctx, asyncio.get_running_loop())
        started = time.monotonic()
        async with chew_lock:  # one paper at a time on the shared pipeline
            with redirect_stdout(sys.stderr):
                try:
                    result = await chewer.achew_file(target, paper_id=paper_id, progress=tracker)
                except BibrError as e:
                    raise ToolError(f"extraction failed for {target.name}: {e}") from e
        pid = store.add(result.data, source=str(target), requested_id=paper_id)
        summary = _summarize(pid, result.data, str(target))
        summary["seconds"] = round(time.monotonic() - started, 1)
        return summary

    @server.tool()
    async def chew_url(url: str, paper_id: str | None = None, *, ctx: Context) -> dict[str, Any]:
        """Download a paper from a public https:// URL and extract it.

        Same pipeline, summary, and registration as chew_paper. The download
        is SSRF-guarded (public HTTPS hosts only — even locally, a hostile
        link must not reach loopback services or cloud metadata endpoints)
        and capped at 100MB.
        """
        import tempfile

        from bibr.utils.safe_fetch import (
            FetchFailedError,
            FetchTooLargeError,
            UnsafeUrlError,
            fetch_url_safely,
        )

        started = time.monotonic()
        try:
            fetched = await fetch_url_safely(url, max_size=_URL_MAX_BYTES)
        except (UnsafeUrlError, FetchTooLargeError, FetchFailedError) as e:
            raise ToolError(str(e)) from None
        tracker = _McpProgress(ctx, asyncio.get_running_loop())
        with tempfile.TemporaryDirectory(prefix="bibr-mcp-url-") as tmp:
            target = Path(tmp) / fetched.filename
            target.write_bytes(fetched.content)
            async with chew_lock:  # one paper at a time on the shared pipeline
                with redirect_stdout(sys.stderr):
                    try:
                        result = await chewer.achew_file(
                            target, paper_id=paper_id, progress=tracker
                        )
                    except BibrError as e:
                        raise ToolError(f"extraction failed for {url}: {e}") from e
        pid = store.add(result.data, source=url, requested_id=paper_id)
        summary = _summarize(pid, result.data, url)
        summary["seconds"] = round(time.monotonic() - started, 1)
        return summary

    @server.tool()
    async def load_paper(path: str) -> dict[str, Any]:
        """Register an existing bibr export JSON (from `bibr chew`) for querying.

        No re-processing — reads the file, checks it looks like a bibr
        export, and returns the same summary shape as chew_paper.
        """
        from bibr.local.inspect import _looks_like_bibr_export

        src = Path(path).expanduser()
        try:
            raw = await asyncio.to_thread(src.read_text, encoding="utf-8")
        except OSError as e:
            raise ToolError(f"cannot read {src}: {e}") from e
        except UnicodeDecodeError:
            raise ToolError(f"{src} is not valid UTF-8 text (not a bibr export)") from None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ToolError(f"{src} is not valid JSON ({e.msg} at line {e.lineno})") from None
        if not _looks_like_bibr_export(data):
            raise ToolError(f"{src} does not look like a bibr export (no recognizable fields)")
        pid = store.add(data, source=str(src))
        return _summarize(pid, data, str(src))

    @server.tool()
    async def save_paper(paper_id: str, path: str, compact: bool = False) -> dict[str, Any]:
        """Write a paper's complete export JSON (schema-versioned, everything the
        query tools slice from) to the given path."""
        entry = store.get(paper_id)
        out = Path(path).expanduser()
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {"separators": (",", ":")} if compact else {"indent": 2}
            await asyncio.to_thread(
                out.write_text,
                json.dumps(entry.data, ensure_ascii=False, **kwargs),
                encoding="utf-8",
            )
        except OSError as e:
            raise ToolError(f"cannot write {out}: {e}") from e
        return {"paper_id": paper_id, "path": str(out), "bytes": out.stat().st_size}

    _register_query_tools(server, lambda ctx: store)  # noqa: ARG005 — one shared store
    return server


def run_mcp(args: argparse.Namespace) -> int:
    """CLI entry for ``bibr mcp``: build the server from flags, serve stdio."""
    options: ChewOptions = {}
    for opt in ("ocr", "llm", "memory", "device", "ocr_url", "ocr_model", "ocr_profile"):
        value = getattr(args, opt, None)
        if value:
            options[opt] = value
    if getattr(args, "crossref", False):
        options["crossref"] = True
    elif getattr(args, "no_crossref", False):
        options["crossref"] = False
    if getattr(args, "no_equations", False):
        options["equations"] = False
    if getattr(args, "no_llm", False):
        options["no_llm"] = True
    if getattr(args, "figure_images", False):
        options["figure_images"] = True
    if getattr(args, "consolidate", None):
        options["consolidate"] = args.consolidate
    if getattr(args, "ref_seg", None):
        options["ref_seg"] = args.ref_seg

    server = build_server(refs=getattr(args, "refs", None), **options)
    server.run(transport="stdio")
    return 0
