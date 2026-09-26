# MCP server

`bibr mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio, so agents (Claude Code, Claude Desktop, any MCP client)
can extract papers and query the results as tools — no shelling out to the
CLI, no parsing whole export files into the agent's context.

One warm pipeline serves the whole session: it is created on the first
extraction call and reused by later calls. Model residency follows the
configured memory mode, just as it does with `bibr.Chewer`.

## Setup

The server uses the MCP Python SDK v2 (minimum 2.2.0) and needs the `mcp` extra:

```bash
uv sync --extra mcp        # add MCP to a source checkout
```

Keep the extras required by your selected OCR/runtime as well; for example,
`uv sync --extra all --extra vllm` includes MCP, ML models, and the CUDA vLLM
runtime. See [Installation](../getting-started/install.md) for platform
choices. Loading and querying saved exports does not load extraction models.

Register it with your client — for Claude Code, from your project directory:

```bash
claude mcp add bibr -- uv run bibr mcp
```

or in JSON client configs (Claude Desktop and most others):

```json
{
  "mcpServers": {
    "bibr": {
      "command": "uv",
      "args": ["--directory", "/path/to/bibr", "run", "bibr", "mcp"]
    }
  }
}
```

Pipeline options are fixed at server start (they configure the shared warm
pipeline, exactly like `bibr.Chewer`): pass a subset of the `bibr chew`
flags after `bibr mcp`, e.g. `bibr mcp --refs ner --crossref` (enrichment is
off unless `--crossref` or `CROSSREF_ENRICH=true` turns it on). Everything
else (provider keys, model choice) comes from your `.env` /
[settings](../reference/settings.md) as usual.

The HTTP endpoint keeps papers within an initialized MCP session and negotiates
protocol `2025-11-25`. SDK v2 clients in automatic mode negotiate this protocol;
clients configured for the sessionless `2026-07-28` protocol must enable automatic
or legacy negotiation. The stdio server also supports the newer protocol because
its process already belongs to a single client.

Exports and query results use schema v12. `get_metadata` returns its paper fields
under `metadata`; summaries expose LLM `usage` as `totals` and `breakdown` within
the existing `llm_usage` summary field.

## The tool surface

A full export is far too large for a single tool result (sentence-level
spans, table HTML, optionally base64 figures), so the tools follow a
**chew once, query granularly** contract: extraction returns only a compact
summary plus a `paper_id`, and the query tools read slices on demand.

| Tool | What it returns |
|---|---|
| `chew_paper(path, paper_id?)` | Runs the pipeline on one file; compact summary (title, DOI, counts, validation, LLM token usage) + `paper_id` |
| `chew_url(url, paper_id?)` | Downloads a public `https://` URL (SSRF-guarded — see below) and runs the same extraction |
| `load_paper(path)` | Registers an existing bibr export JSON without re-processing; same summary |
| `list_papers()` | Papers loaded this session (id, title, DOI, source) |
| `get_paper_summary(paper_id)` | The compact summary again |
| `get_metadata(paper_id)` | Full metadata block (title, abstract, journal, paper type, integrity statements, …), authors, affiliations, funding |
| `get_sections(paper_id)` | Section hierarchy with IMRaD-style types and per-section sentence counts |
| `get_text(paper_id, section_id?, page?, offset?, limit?)` | Sentence spans with `text_id`/`section_id`/`page_number`, filtered and paginated |
| `search_text(paper_id, query, limit?)` | Case-insensitive substring search over sentences |
| `get_references(paper_id, offset?, limit?)` | Parsed bibliography entries, paginated, empty fields omitted |
| `get_reference_citations(paper_id, bib_id)` | Every in-text citation of one reference, with the full source sentence and page |
| `get_tables(paper_id, table_id?)` | Printed label and caption per table; full HTML + cells for one `table_id` |
| `get_figures(paper_id, figure_id?)` | Captions and pages; the image is replaced by `has_image` |
| `save_paper(paper_id, path, compact?)` | Writes the complete export JSON to disk |

This maps directly onto bibr's auditability contract: an agent can pull a
claim from `get_references`, then `get_reference_citations` to see the exact
sentences and pages that cite it.

Stdio results live in server memory for the process lifetime, without a paper
count cap. `save_paper` persists one, and `load_paper` brings saved exports
(or `bibr chew` output) into a later session. Always use the returned
`paper_id`: if different sources have the same ID, the store adds a suffix.

`get_text` defaults to 200 rows and caps requests at 500; `offset` is
zero-based and `page` matches the exported 1-based PDF page number.
`get_tables(table_id=...)` returns the complete table, including every
page's HTML for a table continued across pages. `get_figures` replaces the
figure image with `has_image`, so no image data reaches a tool response.

## URL downloads (`chew_url`)

`chew_url` lets an agent go straight from a link (arXiv, publisher, data
repository) to queryable extraction. Because fetching a caller-supplied URL
is the canonical SSRF surface — even locally, a hostile link must not reach
loopback services or a cloud VM's metadata endpoint — the download is
policy-gated by `bibr.utils.safe_fetch`:

- HTTPS only, port 443 only, no credentials in the URL.
- Every DNS answer must be a public unicast address (private, loopback,
  link-local, CGNAT, multicast, and IPv4-mapped tricks are refused), and the
  connection is **pinned to the validated IP** — TLS SNI and certificate
  verification still use the hostname — so a DNS-rebinding race can't
  redirect the connection after validation.
- Redirects are followed manually (bounded) and every hop re-validated, so
  a public URL can't 302 into an internal network or downgrade to HTTP.
- Downloads are size-capped (100MB on the stdio server; the serve upload
  limit remotely) under a wall-clock deadline.

## Long extractions

A PDF through OCR + LLM can take minutes, and the first call also loads
models. The server streams MCP progress notifications (stage transitions,
per-region OCR progress) while it works; if your client enforces a tool
timeout, raise it for `chew_paper` (in Claude Code:
`MCP_TOOL_TIMEOUT=600000`). One paper is processed at a time on the shared
pipeline; concurrent `chew_paper` calls queue.

## Remote MCP on `bibr serve`

A [self-hosted deployment](deployment.md) can expose the same tool surface
over the network: set `MCP_ENABLED=true` (with the `mcp` extra installed)
and `bibr serve` mounts a streamable-HTTP MCP endpoint at `/mcp`, e.g. for
Claude Code:

```bash
claude mcp add --transport http bibr https://bibr.example.org/mcp \
  --header "Authorization: Bearer $AUTH_API_KEY"
```

It is gated by the same bearer auth as every REST route (`AUTH_API_KEY`;
a keyless loopback server answers only loopback `Host` and `Origin` headers,
see [Authentication](deployment.md#authentication)),
and extraction rides the regular serve dispatch — the LitServe workers'
resident models, admission control, and upload size caps all apply; no
second pipeline is loaded. Concretely: `chew_paper` and `chew_url` accept
files up to `PIPELINE_MAX_FILE_SIZE` (50 MiB), and because `chew_paper`
carries its file base64-encoded inside the JSON-RPC body, enabling MCP raises
the outer request-body cap to fit a full-size file in that form (about
68 MiB by default) — a larger body is refused with a `413` that says so. An
upload counts against `PIPELINE_MAX_ACTIVE_UPLOADS` both while its body is
received (HTTP `429`) and while it is extracted (a `server busy` tool
error), exactly like a `POST /papers/extract` request.

Three differences from the stdio server:

- **`chew_paper` takes an upload, not a path** — `filename` plus base64
  `content_base64`, because client paths don't exist on the server (and the
  server never reads its own filesystem for clients). Per-call options match
  `POST /papers/extract`: `start_page`/`end_page`, `refs`, `consolidate`,
  and `crossref` (`true`/`false`; omit to follow the server's
  `CROSSREF_ENRICH`, which is off by default).
  Both MCP and REST use zero-based, inclusive page indices (`0` is the first page).
  `chew_url` avoids the upload entirely: the server downloads a public
  `https://` URL itself under the SSRF policy above, capped at the serve
  upload limit. Operators can pin it to specific hosts
  (`MCP_URL_ALLOWED_HOSTS=arxiv.org,zenodo.org` — subdomains included) or
  remove the tool with `MCP_CHEW_URL_ENABLED=false`.
- **No `load_paper` / `save_paper`** — both are host-filesystem tools; use
  the REST API when you want the full export JSON as a file.
- **Papers are per-session and bounded** — each MCP client session gets its
  own in-memory store, capped at `MCP_MAX_PAPERS_PER_SESSION` (default 16,
  oldest evicted) and dropped when the session ends. A session that goes
  quiet for `MCP_SESSION_IDLE_TIMEOUT_SECONDS` (default 1800) is closed by
  the server and its papers are dropped, so clients that disconnect without
  `DELETE` do not pin memory. Re-chew after a disconnect.
