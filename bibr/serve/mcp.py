"""MCP endpoint for ``bibr serve`` — streamable HTTP at ``/mcp``.

Mounts a Model Context Protocol server onto the serve FastAPI app so remote
agents get the same chew-then-query tool surface as ``bibr mcp`` (see
``bibr.mcp_server``), with three deployment-shaped differences:

- **Extraction rides the serve dispatch path.** ``chew_paper`` persists the
  upload into the shared :class:`~bibr.serve.ingress.UploadStore` and submits
  a descriptor through the :class:`~bibr.serve.ingress.InferenceDispatchTracker`
  — the exact route ``POST /papers/extract`` takes — so size caps,
  per-request options, and the LitServe workers' resident models all apply.
  No second pipeline, no extra VRAM. Admission control applies too, in two
  halves: the serve app holds a ``PIPELINE_MAX_ACTIVE_UPLOADS`` slot while a
  large ``/mcp`` body is received (the transport buffers JSON-RPC bodies
  whole), and both chew tools take a slot for the extraction itself, failing
  with a "server busy" tool error when none is free.
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
  references. Sessions idle for ``MCP_SESSION_IDLE_TIMEOUT_SECONDS`` are
  closed server-side, so a client that vanishes without ``DELETE`` cannot
  pin its papers for the process lifetime.

Auth needs nothing new: the serve app's bearer middleware gates every path
outside ``PUBLIC_PATHS``, ``/mcp`` included — clients send the same
``Authorization: Bearer <AUTH_API_KEY>`` header as REST callers. With a key,
the bearer token is the boundary and the transport's DNS-rebinding check is
off, so proxies may use any Host. Without one (a loopback-only server), the
check is on and admits only loopback Host and Origin headers, like the serve
app's own gate (:func:`bibr.serve.auth.check_keyless_request`).

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
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from bibr.mcp_server import _PaperStore, _register_query_tools, _summarize
from bibr.serve.admission import UploadAdmission, UploadAdmissionError, base64_envelope
from bibr.serve.auth import KEYLESS_HOSTS
from bibr.serve.ingress import (
    EmptyUploadError,
    InvalidUploadOptionError,
    UploadStorageError,
    UploadTooLargeError,
    _validate_upload_options,
)
from bibr.serve.paths import MCP_MOUNT_PATH
from bibr.utils.safe_fetch import (
    FetchFailedError,
    FetchTooLargeError,
    UnsafeUrlError,
    fetch_url_safely,
)

if TYPE_CHECKING:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from bibr.serve.admission import UploadAdmissionGate
    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore

__all__ = ["MCP_MOUNT_PATH", "build_serve_mcp", "mount_mcp"]

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
    """Stores follow the initialized client's identity, not the per-request proxy.

    SDK v2 creates a fresh ServerSession for each call. Its public client_params
    property refers to the connection's InitializeRequestParams throughout the
    session. Those Pydantic objects are unhashable, so use identity keys with
    weak references and release the store when the connection is collected.
    All callers run on the server event loop.
    """

    def __init__(self, max_papers: int) -> None:
        self._stores: dict[int, tuple[weakref.ReferenceType, _PaperStore]] = {}
        self._max_papers = max_papers

    def resolve(self, ctx: Context) -> _PaperStore:
        client = ctx.session.client_params
        if client is None:
            raise ToolError("initialize a stateful MCP session before using paper tools")
        key = id(client)
        existing = self._stores.get(key)
        if existing is not None and existing[0]() is client:
            return existing[1]
        store = _PaperStore(max_papers=self._max_papers)
        self._stores[key] = (weakref.ref(client, lambda _ref: self._stores.pop(key, None)), store)
        return store


async def _require_session_protocol(ctx, call_next):
    """Negotiate the initialized protocol needed by the chew/query contract.

    SDK v2 supports both 2025 sessions and the sessionless 2026 protocol.
    Returning method-not-found for discovery makes auto clients negotiate an
    initialized session. Explicit sessionless calls cannot access stored papers.
    """
    from mcp import MCPError
    from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS

    if ctx.method == "server/discover":
        raise MCPError(code=-32601, message="bibr paper tools require an initialized session")
    if ctx.protocol_version not in HANDSHAKE_PROTOCOL_VERSIONS and ctx.method == "tools/call":
        raise MCPError(code=-32600, message="use the initialized 2025-11-25 MCP protocol")
    return await call_next(ctx)


def build_serve_mcp(
    *,
    upload_store: UploadStore,
    tracker: InferenceDispatchTracker,
    max_papers_per_session: int,
    chew_url_enabled: bool = True,
    url_allowed_hosts: list[str] | None = None,
    admission_gate: UploadAdmissionGate | None = None,
) -> MCPServer:
    """Build the serve-mounted MCP server around the shared dispatch machinery.

    ``admission_gate`` is the serve app's upload gate; each chew tool takes the
    same pair of slots ``POST /papers/extract`` does — a spool slot while the
    bytes are materialised and persisted, and an inflight slot for the whole
    run. ``None`` (stdio-style tests) skips the gate.
    """
    from starlette.datastructures import UploadFile

    instructions = _SERVE_INSTRUCTIONS
    if chew_url_enabled:
        instructions = instructions.replace(
            "Start with chew_paper:",
            "Start with chew_paper (upload) or chew_url (public https:// URL):",
        )

    stores = _SessionStores(max_papers_per_session)
    server = MCPServer(
        "bibr",
        instructions=instructions,
        middleware=[_require_session_protocol],
    )

    def _validated_options(
        start_page: int | None,
        end_page: int | None,
        refs: str | None,
        consolidate: str | None,
        crossref: bool | None = None,
    ) -> dict[str, str]:
        raw_options = {
            "start_page": "" if start_page is None else str(start_page),
            "end_page": "" if end_page is None else str(end_page),
            "refs": refs or "",
            "consolidate": consolidate or "",
            # Tri-state like the form field: absent → "" → dropped → server default.
            "crossref": "" if crossref is None else ("true" if crossref else "false"),
        }
        try:
            return _validate_upload_options(raw_options)
        except InvalidUploadOptionError as e:
            raise ToolError(str(e)) from None

    @asynccontextmanager
    async def _admission_slots():
        """Own upload slots until persistence, then hand inference to dispatch."""
        if admission_gate is None:
            yield None
            return
        try:
            with admission_gate.admit() as admission:
                yield admission
        except UploadAdmissionError as exc:
            raise ToolError(f"server busy: {str(exc).lower()}, retry in a moment") from None

    async def _extract_and_register(
        *,
        filename: str,
        content: bytes,
        options: dict[str, str],
        source: str,
        ctx: Context,
        started: float,
        admission: UploadAdmission | None,
    ) -> dict[str, Any]:
        """Persist → dispatch → register: the shared tail of both chew tools."""
        max_mib = upload_store.max_size / 1024 / 1024
        if len(content) > upload_store.max_size:
            # The bytes are already resident; refuse before spooling them to disk.
            raise ToolError(f"file too large (>{max_mib:.0f}MB)")
        try:
            stored = await upload_store.persist(
                UploadFile(file=io.BytesIO(content), filename=filename, size=len(content))
            )
        except EmptyUploadError as e:
            # "empty upload" (zero-byte content) or "empty filename".
            raise ToolError(f"invalid upload: {e}") from None
        except UploadTooLargeError:
            raise ToolError(f"file too large (>{max_mib:.0f}MB)") from None
        except UploadStorageError:
            raise ToolError("insufficient temporary storage on the server") from None

        # Upload is on disk, so this call no longer occupies what the spool
        # counter bounds. Hand that slot back before the pipeline runs.
        if admission is not None:
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
        crossref: bool | None = None,
        *,
        ctx: Context,
    ) -> dict[str, Any]:
        """Extract a paper and return a compact summary.

        Upload the file (PDF/DOCX/JATS XML/HTML/ePub) as base64 content; the
        extraction runs on the server's shared pipeline and the result is
        registered for the get_*/search_text tools under the returned
        paper_id. Optional per-call knobs match POST /papers/extract:
        start_page/end_page (0-indexed, inclusive — the first page is 0),
        refs (ner|llm|llm-chunked|off), consolidate (fill|replace), and
        crossref (true runs Crossref/resolver reference enrichment, false skips
        it; omit for the server default, which is off). Large papers can take
        minutes.
        """
        options = _validated_options(start_page, end_page, refs, consolidate, crossref)
        async with _admission_slots() as admission:
            if len(content_base64) > base64_envelope(upload_store.max_size):
                max_mib = upload_store.max_size / 1024 / 1024
                raise ToolError(f"file too large (>{max_mib:.0f}MB)")
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
            crossref: bool | None = None,
            *,
            ctx: Context,
        ) -> dict[str, Any]:
            """Download a paper from a public https:// URL and extract it.

            The server fetches the URL itself (SSRF-guarded: HTTPS to public
            hosts only, redirects re-validated, size-capped — an operator may
            additionally restrict the hosts), then runs the same extraction
            and registration as chew_paper, with the same optional per-call
            knobs (start_page/end_page are 0-indexed and inclusive). Prefer
            this over chew_paper for anything you would otherwise download
            just to re-upload.
            """
            options = _validated_options(start_page, end_page, refs, consolidate, crossref)
            started = time.monotonic()
            async with _admission_slots() as admission:
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
    admission_gate: UploadAdmissionGate | None = None,
) -> MCPServer:
    """Mount the MCP endpoint at ``/mcp`` on the serve app (``MCP_ENABLED``)."""
    from mcp.server.transport_security import TransportSecuritySettings

    server = build_serve_mcp(
        upload_store=upload_store,
        tracker=tracker,
        max_papers_per_session=settings.mcp.max_papers_per_session,
        chew_url_enabled=settings.mcp.chew_url_enabled,
        url_allowed_hosts=settings.mcp.url_allowed_hosts,
        admission_gate=admission_gate,
    )
    idle_timeout = float(settings.mcp.session_idle_timeout_seconds)
    # The SDK's default 4 MiB JSON cap is smaller than bibr's upload limit.
    # Account for base64 expansion plus bounded JSON/options overhead.
    body_limit = 4 * ((upload_store.max_size + 2) // 3) + 64 * 1024
    if settings.auth.api_key:
        # Bearer auth is the deployment boundary; any Host may front it.
        transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    else:
        # No key: loopback is the only boundary, and a DNS-rebinding page would
        # otherwise reach it under its own host name. Same names and origins as
        # the serve app's own gate (check_keyless_request).
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[host for name in KEYLESS_HOSTS for host in (name, f"{name}:*")],
            allowed_origins=[
                *(
                    origin
                    for name in KEYLESS_HOSTS
                    for scheme in ("http", "https")
                    for origin in (f"{scheme}://{name}", f"{scheme}://{name}:*")
                ),
                *(origin for origin in settings.cors.origins if origin != "*"),
            ],
        )
    mounted = server.streamable_http_app(
        streamable_http_path="/",
        session_idle_timeout=idle_timeout if idle_timeout > 0 else None,
        max_request_body_size=body_limit,
        transport_security=transport_security,
    )
    session_manager = server.session_manager
    app.mount(MCP_MOUNT_PATH, mounted)
    app.add_middleware(_McpBarePathRewrite)
    _compose_session_manager_lifespan(app, session_manager)
    return server
