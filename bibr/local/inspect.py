"""``bibr inspect`` — human-readable summary of a bibr JSON export.

Reads a single extraction-output JSON file (any v10.x export from
``bibr.export.json_export.export_paper_to_json``) and prints title/authors/
DOI/paper-type, structure counts (sections/sentences/tables/figures/
equations), reference stats (bib count, in-text citation coverage,
enrichment state), the output-validation summary, and per-model LLM token
usage.

Degradation contract: any missing block renders as ``—`` (scalars) or "not
present" (blocks/lists) and the command exits 0 — a file that legitimately
lacks a block (e.g. ``--refs off``, ``--no-llm``) is not an error. A file that
isn't valid UTF-8/JSON, or is JSON but clearly not a bibr export (no
``info``-with-version-fields and none of the bibr-only top-level blocks —
see ``_looks_like_bibr_export``), exits 1 with a single clear error line and
never a traceback.

Field names below are read verbatim from the export builder
(``bibr/export/json_export.py``) — see the module docstring lines cited in
each helper rather than re-deriving them here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Reused verbatim from Task 5 (bibr/local/cli.py) rather than forked: these
# helpers are already hardened against malformed ``validation`` payloads
# (non-int counts coerce to 0, non-dict issues are skipped, missing keys
# render "?") — exactly the crash-safety this command's degrade contract
# needs. cli.py is a light import (argparse/json/stdlib + a couple of lazy
# in-function imports), so importing it here doesn't pull in heavy deps.
from bibr.local.cli import _format_validation_line


def _looks_like_bibr_export(data: Any) -> bool:
    """A top-level ``info`` *dict* alone is too loose a gate — plenty of
    unrelated JSON (an OpenAPI spec being the canonical example) also has a
    top-level ``info`` object. Require either an ``info`` that carries one of
    the version/format fields every bibr export has had since early v10.x
    (``schema_version``, ``bibr_version``, ``input_format`` —
    json_export.py:122-160, ``InfoExport``), or a top-level block that only a
    bibr export would ever have (``author``, ``bib``, ``xref``, ``text``,
    ``section``, ``validation``, ``processing_warnings``, ``llm_usage`` —
    json_export.py:438-, ``PaperExport``). This keeps old pre-v10.3 exports
    and the hand-built degraded fixture (``info`` + ``author`` only)
    accepted, while rejecting generic ``info``-shaped JSON."""
    if not isinstance(data, dict):
        return False
    info = data.get("info")
    if isinstance(info, dict) and any(
        key in info for key in ("schema_version", "bibr_version", "input_format")
    ):
        return True
    return any(
        key in data
        for key in (
            "author",
            "bib",
            "xref",
            "text",
            "section",
            "validation",
            "processing_warnings",
            "llm_usage",
        )
    )


def _scalar(info: dict, key: str) -> str:
    val = info.get(key)
    if val in (None, ""):
        return "—"
    return str(val)


def _count_or_dash(data: dict, key: str) -> str:
    """``len(data[key])`` when present as a list, else "—" — distinguishes a
    block that's entirely absent from one that's present-but-empty (a real
    zero, e.g. a short paper with no tables)."""
    val = data.get(key)
    if isinstance(val, list):
        return str(len(val))
    return "—"


def _authors_line(data: dict) -> str:
    """``author`` (json_export.py:452-455, ``AuthorExport`` at :166-177):
    count plus the first three as "Given Family", folding any remainder into
    a "+N more" marker."""
    authors = data.get("author")
    if not isinstance(authors, list):
        return "not present"
    if not authors:
        return "0"
    names = []
    for a in authors[:3]:
        if isinstance(a, dict):
            given = (a.get("given") or "").strip()
            family = (a.get("family") or "").strip()
            full = f"{given} {family}".strip() or "?"
        else:
            full = "?"
        names.append(full)
    more = len(authors) - 3
    suffix = f", … +{more} more" if more > 0 else ""
    return f"{len(authors)} ({', '.join(names)}{suffix})"


def _citation_coverage(data: dict) -> str:
    """In-text citation coverage: NOT a "matched/total over the xref list"
    rate — that field-shape looks meaningful (``xref_id`` is nullable
    per json_export.py:341-348, ``XrefExport``) but is empirically vacuous:
    every code path that ever constructs a ``xref_type == "bib"`` entry
    (``bibr/structure/citation_linker.py``, ``bibr/structure/
    citation_matcher.py``) only appends it *after* a match is already found —
    an unresolved in-text citation candidate is dropped, never stored with a
    null ``xref_id``. So "matched/total" over that list reads ~100% on every
    real export and carries no signal.

    What genuinely varies per paper: how many of the *references themselves*
    got cited at least once in the body text. N = count of ``xref`` entries
    with ``xref_type == "bib"`` (in-text citation occurrences); M = count of
    distinct non-null ``xref_id`` values among those (references cited at
    least once); K = ``len(bib)`` (total references). Reported as
    "N linked → M of K references cited (pct%)", where the percentage is
    only ever attached to the M/K coverage figure, never to N."""
    xref = data.get("xref")
    if not isinstance(xref, list) or not xref:
        return "not present"
    bib_xrefs = [x for x in xref if isinstance(x, dict) and x.get("xref_type") == "bib"]
    n = len(bib_xrefs)
    if n == 0:
        return "not present"
    cited_ids = {x.get("xref_id") for x in bib_xrefs if x.get("xref_id") is not None}
    m = len(cited_ids)
    bib = data.get("bib")
    k = len(bib) if isinstance(bib, list) else 0
    if k > 0:
        pct = round(100 * m / k)
        return f"{n} linked → {m} of {k} references cited ({pct}%)"
    return f"{n} linked → {m} of {k} references cited"


def _as_nonneg_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(value, 0)
    return 0


def _enrichment_state(data: dict) -> str:
    """Prefer the dedicated ``enrichment`` block (json_export.py:426-436,
    ``EnrichmentExport``: ``complete``/``refs_enriched``/``refs_total`` —
    null when enrichment never ran). When absent, fall back to counting
    unique ``bib_id``s covered by ``bib_match`` (json_export.py:260-285,
    flattened enrichment matches — one row per external-service hit) against
    the ``bib`` count."""
    enrichment = data.get("enrichment")
    if isinstance(enrichment, dict):
        complete = bool(enrichment.get("complete"))
        enriched = _as_nonneg_int(enrichment.get("refs_enriched"))
        total = _as_nonneg_int(enrichment.get("refs_total"))
        state = "complete" if complete else "partial"
        return f"{state} ({enriched}/{total} refs enriched)"

    bib_match = data.get("bib_match")
    bib = data.get("bib")
    if isinstance(bib_match, list) and bib_match and isinstance(bib, list) and bib:
        matched_ids = {
            m.get("bib_id")
            for m in bib_match
            if isinstance(m, dict) and m.get("bib_id") is not None
        }
        pct = round(100 * len(matched_ids) / len(bib))
        return f"{len(matched_ids)}/{len(bib)} refs matched ({pct}%, from bib_match — no enrichment block)"

    return "not present"


def _llm_usage_lines(data: dict) -> list[str]:
    """Per-model token counts from ``llm_usage`` (a
    ``dict[model_name, {input_tokens, output_tokens, total_tokens}]``,
    json_export.py:524-530). No cost estimate: there is no $/token price
    table anywhere in the codebase, and the task brief is explicit that one
    must never be invented — tokens only."""
    usage = data.get("llm_usage")
    if not isinstance(usage, dict) or not usage:
        return ["  not present"]
    lines = []
    for model, counts in usage.items():
        if not isinstance(counts, dict):
            continue
        inp = _as_nonneg_int(counts.get("input_tokens"))
        out = _as_nonneg_int(counts.get("output_tokens"))
        total = _as_nonneg_int(counts.get("total_tokens"))
        lines.append(f"  {model}: in={inp:,} out={out:,} total={total:,}")
    return lines or ["  not present"]


def _validation_lines(data: dict) -> list[str]:
    """Reuses Task 5's ``_format_validation_line`` (bibr/local/cli.py) —
    same rendering, same hardening against malformed ``errors``/``warnings``/
    ``issues`` shapes."""
    validation = data.get("validation")
    if not isinstance(validation, dict):
        return ["  not present"]
    line = _format_validation_line(data)
    if line is None:
        return ["  clean (0 errors, 0 warnings)"]
    return [f"  {line.strip()}"]


def _print_report(console, data: dict, source: str) -> None:
    info = data.get("info")
    if not isinstance(info, dict):
        info = {}

    console.print(f"[bold]bibr inspect[/bold] — {source}\n")

    console.print(f"Title: {_scalar(info, 'title')}")
    console.print(f"Authors: {_authors_line(data)}")
    console.print(f"DOI: {_scalar(info, 'doi')}")
    console.print(f"Paper type: {_scalar(info, 'paper_type')}")
    console.print()

    console.print(f"Sections: {_count_or_dash(data, 'section')}")
    console.print(f"Sentences: {_count_or_dash(data, 'text')}")
    console.print(
        f"Tables: {_count_or_dash(data, 'table')}   "
        f"Figures: {_count_or_dash(data, 'figure')}   "
        f"Equations: {_count_or_dash(data, 'eq')}"
    )
    console.print()

    console.print("[bold]References[/bold]")
    console.print(f"  Bibliography entries: {_count_or_dash(data, 'bib')}")
    console.print(f"  In-text citations: {_citation_coverage(data)}")
    console.print(f"  Enrichment: {_enrichment_state(data)}")
    console.print()

    console.print("[bold]Validation[/bold]")
    for line in _validation_lines(data):
        console.print(line)
    console.print()

    console.print("[bold]LLM usage[/bold]")
    for line in _llm_usage_lines(data):
        console.print(line)


def run_inspect(json_file: str) -> int:
    """Print the report for *json_file* to stdout; return the process exit
    code (0 on success, 1 on a read/parse/shape error). Never raises — every
    error path is caught and reported as a single clear stderr line."""
    import sys

    from rich.console import Console

    path = Path(json_file)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"bibr inspect: cannot read {json_file}: {e}", file=sys.stderr)
        return 1
    except UnicodeDecodeError:
        # Binary/non-UTF-8 input (e.g. the wrong file entirely) — a
        # UnicodeDecodeError is a ValueError subclass, not an OSError, so it
        # needs its own clause; without it, this crashes with a raw
        # traceback instead of the required clean exit-1 message.
        print(
            f"bibr inspect: {json_file} is not valid UTF-8 text (not a bibr export)",
            file=sys.stderr,
        )
        return 1

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(
            f"bibr inspect: {json_file} is not valid JSON ({e.msg} at line {e.lineno})",
            file=sys.stderr,
        )
        return 1

    if not _looks_like_bibr_export(data):
        print(
            f"bibr inspect: {json_file} does not look like a bibr export "
            "(no recognizable bibr fields)",
            file=sys.stderr,
        )
        return 1

    console = Console(width=120)
    try:
        _print_report(console, data, json_file)
    except Exception as e:  # noqa: BLE001 — a rendering bug must not traceback on the user
        print(f"bibr inspect: internal error rendering {json_file}: {e}", file=sys.stderr)
        return 1
    return 0
