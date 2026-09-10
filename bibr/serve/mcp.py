"""MCP endpoint for ``bibr serve`` — streamable HTTP at ``/mcp``.

Mounts a Model Context Protocol server onto the serve FastAPI app so remote
agents get the same chew-then-query tool surface as ``bibr mcp`` (see
``bibr.mcp_server``), with three deployment-shaped differences:

- **Extraction rides the serve dispatch path.** ``chew_paper`` persists the
  upload into the shared :class:`~bibr.serve.ingress.UploadStore` and submits
  a descriptor through the :class:`~bibr.serve.ingress.InferenceDispatchTracker`
  — the exact route ``POST /papers/extract`` takes — so admission control,
  size caps, per-request options, and the LitServe workers' resident models
  all apply. No second pipeline, no extra VRAM.
- **Input is uploaded or fetched, never referenced.** A remote client's paths
  mean nothing on the server (and serving host-filesystem reads to
  bearer-token holders would be a new capability the REST API deliberately
  does not have), so ``chew_paper`` takes base64 file content and the stdio
  server's ``load_paper``/``save_paper`` filesystem tools are not registered.
  ``chew_url`` downloads a public HTTPS URL server-side under
  :mod:`bibr.utils.safe_fetch`'s SSRF policy — operators can restrict it to
  ``MCP_URL_ALLOWED_HOSTS`` or remove it with ``MCP_CHEW_URL_ENABLED=false``.
- **State is per MCP session, bounded.** Each client session gets its own
  paper store (no cross-client leakage through a shared bearer key), capped
  at ``MCP_MAX_PAPERS_PER_SESSION`` and dropped with the session via weak
  references.

Auth needs nothing new: the serve app's bearer middleware gates every path
outside ``PUBLIC_PATHS``, ``/mcp`` included — clients send the same
``Authorization: Bearer <AUTH_API_KEY>`` header as REST callers. Transport
DNS-rebinding protection stays at the SDK default (off), matching the REST
surface where bearer auth is the boundary.

The streamable-HTTP session manager must be *running* for the endpoint to
serve, and Starlette does not run mounted apps' lifespans — so the mount
composes ``session_manager.run()`` into the outer app's lifespan, the same
pattern as ``app._compose_lifespan_cleanup``.
"""

from __future__ import annotations

import base64
import binascii
import io
import time
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from bibr.mcp_server import _PaperStore, _register_query_tools, _summarize
from bibr.serve.admission import (
    UploadAdmission,
    UploadAdmissionError,
    UploadAdmissionGate,
    release_spool_slot,
)
from bibr.serve.ingress import (
    EmptyUploadError,
    InvalidUploadOptionError,
    UploadStorageError,
    UploadTooLargeError,
    _validate_upload_options,
)
from bibr.utils.safe_fetch import (
    FetchFailedError,
    FetchTooLargeError,
    UnsafeUrlError,
    fetch_url_safely,
)

if TYPE_CHECKING:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore

MCP_MOUNT_PATH = "/mcp"

_SERVE_INSTRUCTIONS = """\
bibr extracts structured data from scientific papers (PDF, DOCX, JATS XML, HTML, ePub):
metadata, authors, full text (sentence-level, linked to sections and pages), references,
in-text citations, tables, figures, and statistical expressions.

Start with chew_paper: upload one file as base64 and get back a compact summary plus a
paper_id for the query tools — get_metadata, get_sections, get_text, search_text,
get_references, get_reference_citations, get_tables, get_figures. Full exports are large,
so query the slices you need instead of asking for everything. Papers live in server
memory for your MCP session only (oldest evicted beyond a cap); re-chew after a
disconnect, or use the REST API (POST /papers/extract) when you want the full export
JSON as a file.
"""


class _SessionStores:
    """Per-MCP-session paper stores, dropped with the session.

    Keyed weakly on the ``ServerSession`` object, which lives exactly as long
    as the client's streamable-HTTP session — when the session terminates,
    its store (and every chewed export in it) becomes collectable.
    """

    def __init__(self, max_papers: int) -> None:
        self._stores: weakref.WeakKeyDictionary[Any, _PaperStore] = weakref.WeakKeyDictionary()
        self._max_papers = max_papers

    def resolve(self, ctx: Context) -> _PaperStore:
        session = ctx.session
        store = self._stores.get(session)
        if store is None:
            store = _PaperStore(max_papers=self._max_papers)
            self._stores[session] = store
        return store


def build_serve_mcp(
    *,
    upload_store: UploadStore,
    tracker: InferenceDispatchTracker,
    admission_gate: UploadAdmissionGate,
    max_papers_per_session: int,
    chew_url_enabled: bool = True,
    url_allowed_hosts: list[str] | None = None,
) -> FastMCP:
    """Build the serve-mounted MCP server around the shared dispatch machinery."""
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.datastructures import UploadFile

    instructions = _SERVE_INSTRUCTIONS
    if chew_url_enabled:
        instructions = instructions.replace(
            "Start with chew_paper:",
            "Start with chew_paper (upload) or chew_url (public https:// URL):",
        )

    stores = _SessionStores(max_papers_per_session)
    server = FastMCP(
        "bibr",
        instructions=instructions,
        # FastMCP auto-enables Host-header (DNS-rebinding) validation with a
        # localhost-only allowlist because its own `host` default is
        # 127.0.0.1 — but that setting is meaningless here (uvicorn binds via
        # `bibr serve`, FastMCP.run is never called) and would 421 every
        # reverse-proxied deployment. Match the REST surface instead: bearer
        # auth is the boundary, and validate_bind_auth already refuses
        # network-visible binds without a key.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    def _validated_options(
        start_page: int | None, end_page: int | None, refs: str | None, consolidate: str | None
    ) -> dict[str, str]:
        # MCP advertises physical page numbers; the REST/worker descriptor
        # uses zero-based indices. Validate in the caller's numbering first.
        for name, page in (("start_page", start_page), ("end_page", end_page)):
            if page is not None and page < 1:
                raise ToolError(f"{name} must be >= 1 (pages are 1-based)")
        raw_options = {
            "start_page": "" if start_page is None else str(start_page),
            "end_page": "" if end_page is None else str(end_page),
            "refs": refs or "",
            "consolidate": consolidate or "",
        }
        try:
            options = _validate_upload_options(raw_options)
        except InvalidUploadOptionError as e:
            raise ToolError(str(e)) from None
        for name in ("start_page", "end_page"):
            if name in options:
                options[name] = str(int(options[name]) - 1)
        return options

    @contextmanager
    def _admitted(ctx: Context):
        # Replace the HTTP pre-parse slot with tool-owned slots atomically.
        # SSE headers may already have released the HTTP slot; its release is
        # idempotent. Tool lifetime, unlike HTTP headers, spans the extraction.
        request = ctx.request_context.request
        if request is not None:
            release_spool_slot(request)
        try:
            with admission_gate.admit() as admission:
                yield admission
        except UploadAdmissionError as exc:
            raise ToolError(f"{exc}; retry shortly") from None

    async def _extract_and_register(
        *,
        filename: str,
        content: bytes,
        options: dict[str, str],
        source: str,
        ctx: Context,
        started: float,
        admission: UploadAdmission,
    ) -> dict[str, Any]:
        """Persist → dispatch → register: the shared tail of both chew tools."""
        try:
            stored = await upload_store.persist(
                UploadFile(file=io.BytesIO(content), filename=filename, size=len(content))
            )
        except EmptyUploadError as e:
            # "empty upload" (zero-byte content) or "empty filename".
            raise ToolError(f"invalid upload: {e}") from None
        except UploadTooLargeError:
            max_mib = upload_store.max_size / 1024 / 1024
            raise ToolError(f"file too large (>{max_mib:.0f}MB)") from None
        except UploadStorageError:
            raise ToolError("insufficient temporary storage on the server") from None

        admission.release_spool()
        from fastapi import HTTPException

        try:
            result = await tracker.submit(stored.to_descriptor(options), admission=admission)
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            raise ToolError(f"extraction failed for {source}: {detail}") from None
        if not isinstance(result, dict):
            raise ToolError("extraction returned an unexpected response shape")

        store = stores.resolve(ctx)
        pid = store.add(result, source=source)
        summary = _summarize(pid, result, source)
        summary["seconds"] = round(time.monotonic() - started, 1)
        return summary

    @server.tool()
    async def chew_paper(
        filename: str,
        content_base64: str,
        start_page: int | None = None,
        end_page: int | None = None,
        refs: str | None = None,
        consolidate: str | None = None,
        *,
        ctx: Context,
    ) -> dict[str, Any]:
        """Extract a paper and return a compact summary.

        Upload the file (PDF/DOCX/JATS XML/HTML/ePub) as base64 content; the
        extraction runs on the server's shared pipeline and the result is
        registered for the get_*/search_text tools under the returned
        paper_id. Optional per-call knobs are start_page/end_page (1-based,
        inclusive), refs (ner|llm|llm-chunked|off), and
        consolidate (fill|replace). Large papers can take minutes.
        """
        options = _validated_options(start_page, end_page, refs, consolidate)
        with _admitted(ctx) as admission:
            # Reject oversized encodings before allocating the decoded copy.
            if len(content_base64) > 4 * ((upload_store.max_size + 2) // 3):
                raise ToolError(f"file too large (>{upload_store.max_size / 1024 / 1024:.0f}MB)")
            try:
                content = base64.b64decode(content_base64, validate=True)
            except (binascii.Error, ValueError):
                raise ToolError("content_base64 is not valid base64") from None
            return await _extract_and_register(
                filename=filename,
                content=content,
                options=options,
                source=filename,
                ctx=ctx,
                started=time.monotonic(),
                admission=admission,
            )

    if chew_url_enabled:

        @server.tool()
        async def chew_url(
            url: str,
            start_page: int | None = None,
            end_page: int | None = None,
            refs: str | None = None,
            consolidate: str | None = None,
            *,
            ctx: Context,
        ) -> dict[str, Any]:
            """Download a paper from a public https:// URL and extract it.

            The server fetches the URL itself (SSRF-guarded: HTTPS to public
            hosts only, redirects re-validated, size-capped — an operator may
            additionally restrict the hosts), then runs the same extraction
            and registration as chew_paper, with the same optional per-call
            knobs (start_page/end_page are 1-based, inclusive). Prefer this
            over chew_paper for anything you would otherwise download just
            to re-upload.
            """
            options = _validated_options(start_page, end_page, refs, consolidate)
            with _admitted(ctx) as admission:
                started = time.monotonic()
                try:
                    fetched = await fetch_url_safely(
                        url,
                        max_size=upload_store.max_size,
                        allowed_hosts=url_allowed_hosts or None,
                    )
                except (UnsafeUrlError, FetchTooLargeError, FetchFailedError) as e:
                    raise ToolError(str(e)) from None
                return await _extract_and_register(
                    filename=fetched.filename,
                    content=fetched.content,
                    options=options,
                    source=url,
                    ctx=ctx,
                    started=started,
                    admission=admission,
                )

    _register_query_tools(server, stores.resolve)
    return server


class _McpBarePathRewrite:
    """Raw ASGI middleware: route ``/mcp`` exactly like ``/mcp/``.

    Starlette's router answers a bare ``/mcp`` with a 307 to ``/mcp/``
    (mounts only match with the trailing slash), and the MCP client
    correctly refuses to follow a redirect on POST. Clients configure
    ``https://host/mcp`` — normalize the path before routing instead of
    redirecting. Raw ASGI (not ``BaseHTTPMiddleware``): nothing to buffer,
    nothing added to the response path of every other route.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope.get("path") == MCP_MOUNT_PATH:
            scope = dict(scope, path=MCP_MOUNT_PATH + "/")
        await self.app(scope, receive, send)


def _compose_session_manager_lifespan(app, session_manager: StreamableHTTPSessionManager) -> None:
    """Run the MCP session manager for the outer app's lifetime.

    Starlette does not run a mounted sub-app's lifespan, and the streamable-
    HTTP transport refuses requests until its session manager's task group is
    running — so wrap the (possibly already-wrapped) outer lifespan.
    """
    from contextlib import asynccontextmanager

    inner = app.router.lifespan_context

    @asynccontextmanager
    async def _lifespan(lifespan_app):
        async with session_manager.run(), inner(lifespan_app):
            yield

    app.router.lifespan_context = _lifespan


def mount_mcp(
    app,
    settings,
    *,
    upload_store: UploadStore,
    tracker: InferenceDispatchTracker,
    admission_gate: UploadAdmissionGate,
) -> FastMCP:
    """Mount the MCP endpoint at ``/mcp`` on the serve app (``MCP_ENABLED``)."""
    from mcp.server.fastmcp.server import StreamableHTTPASGIApp

    server = build_serve_mcp(
        upload_store=upload_store,
        tracker=tracker,
        admission_gate=admission_gate,
        max_papers_per_session=settings.mcp.max_papers_per_session,
        chew_url_enabled=settings.mcp.chew_url_enabled,
        url_allowed_hosts=settings.mcp.url_allowed_hosts,
    )
    # Mount the raw ASGI handler, not server.streamable_http_app(): the
    # Starlette wrapper routes by path, and mounting a router under /mcp
    # makes a POST to /mcp answer with a 307 to /mcp/ — a redirect the MCP
    # client (correctly) refuses to follow. The handler itself is
    # path-agnostic, so /mcp and /mcp/ both hit the transport directly.
    # (streamable_http_app() is still what instantiates session_manager with
    # the FastMCP settings; the wrapper app it returns is discarded.)
    server.streamable_http_app()
    app.mount(MCP_MOUNT_PATH, StreamableHTTPASGIApp(server.session_manager))
    app.add_middleware(_McpBarePathRewrite)
    _compose_session_manager_lifespan(app, server.session_manager)
    return server
