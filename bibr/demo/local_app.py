"""Local Gradio Demo for bibr

Standalone demo that runs the full bibr pipeline in-process.
No external API server required — just an LLM API key and OCR backend.
"""

import asyncio
import copy
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from html import escape as _esc
from typing import Any

import gradio as gr

from bibr.input.supported_files import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE_MB = int(os.environ.get("DEMO_MAX_FILE_SIZE_MB", "10"))
_MAX_FILE_SIZE_BYTES = _MAX_FILE_SIZE_MB * 1024 * 1024

_ALLOWED_EXTENSIONS: list[str] = sorted(SUPPORTED_EXTENSIONS)

_TABLE_SCROLL_CSS = """\
/* Scroll shadow indicators for dataframe tables */
.table-wrap {
    scrollbar-width: thin;
}
.table-wrap.can-scroll-right {
    box-shadow: inset -16px 0 12px -12px rgba(0, 0, 0, 0.12);
}
.table-wrap.can-scroll-left {
    box-shadow: inset 16px 0 12px -12px rgba(0, 0, 0, 0.12);
}
.table-wrap.can-scroll-left.can-scroll-right {
    box-shadow: inset 16px 0 12px -12px rgba(0, 0, 0, 0.12),
                inset -16px 0 12px -12px rgba(0, 0, 0, 0.12);
}
/* Always show a visible scrollbar on WebKit browsers */
.table-wrap::-webkit-scrollbar {
    height: 8px;
    -webkit-appearance: none;
}
.table-wrap::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.04);
    border-radius: 4px;
}
.table-wrap::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.18);
    border-radius: 4px;
}
.table-wrap::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.28);
}
"""

_TABLE_SCROLL_JS = """\
() => {
    function initScrollHints() {
        document.querySelectorAll('.table-wrap').forEach(el => {
            if (el.dataset.scrollInit) return;
            el.dataset.scrollInit = '1';
            const update = () => {
                const canRight = el.scrollWidth - el.scrollLeft - el.clientWidth > 1;
                const canLeft = el.scrollLeft > 1;
                el.classList.toggle('can-scroll-right', canRight);
                el.classList.toggle('can-scroll-left', canLeft);
            };
            el.addEventListener('scroll', update, {passive: true});
            new ResizeObserver(update).observe(el);
            update();
        });
    }
    new MutationObserver(initScrollHints).observe(
        document.body, {childList: true, subtree: true}
    );
    initScrollHints();
}
"""

_HEADER_MD = """\
# bibr 🦫

Upload a scientific paper and bibr will extract its metadata: title, authors, sections, \
references, tables, equations, and more.

Accepts {extensions} files.
""".format(extensions=" and ".join(f"`{ext}`" for ext in _ALLOWED_EXTENSIONS))


def _check_file_size(file_path: str) -> None:
    """Raise gr.Error if the file exceeds the size limit."""
    size = os.path.getsize(file_path)
    if size > _MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        raise gr.Error(
            f"This file is {size_mb:.1f} MB — the limit is {_MAX_FILE_SIZE_MB} MB. "
            f"Try a shorter paper or use the CLI for larger files."
        )


def _compute_section_levels(sections: list[dict]) -> None:
    """Compute level field for sections from parent_section_id tree (in-place)."""
    by_id = {s["section_id"]: s for s in sections}
    for s in sections:
        level = 1
        current = s
        seen = {s["section_id"]}
        while current.get("parent_section_id") and current["parent_section_id"] in by_id:
            pid = current["parent_section_id"]
            if pid in seen:
                break  # guard against cycles
            seen.add(pid)
            level += 1
            current = by_id[pid]
        s["level"] = level


def _parse_json_response(paper_json: dict) -> dict:
    """Normalize a JSON API response into the dict structure the builder functions expect."""
    metadata = paper_json.get("metadata", {})
    authors = paper_json.get("author", [])
    text = paper_json.get("text", [])
    sections = paper_json.get("section", [])
    urls = paper_json.get("url", [])
    bib = paper_json.get("bib", [])
    xrefs = paper_json.get("xref", [])
    figs = paper_json.get("figure", [])
    tbls = paper_json.get("table", [])
    footnotes = paper_json.get("footnote", [])
    equations = paper_json.get("eq", [])

    # v12 keeps processing facts under ``extraction.diagnostics`` and the
    # affiliations in their own table; fold them back into the display rows.
    diagnostics = (paper_json.get("extraction") or {}).get("diagnostics") or {}
    metadata = {**metadata, **(diagnostics.get("paper_classification") or {})}
    scores = {
        row.get("section_id"): row.get("score")
        for row in diagnostics.get("section_classification") or []
    }
    if scores:
        sections = [
            {**s, "classification_score": scores.get(s.get("section_id"))} for s in sections
        ]
    affiliations = paper_json.get("affiliation") or []
    if affiliations:
        authors = [
            {
                **a,
                "affiliation": "; ".join(
                    row.get("text") or ""
                    for row in affiliations
                    if a.get("author_id") in (row.get("author_ids") or [])
                ),
            }
            for a in authors
        ]

    # Compute section levels from parent_section_id tree
    if sections:
        _compute_section_levels(sections)

    # bib_match is a top-level array in v10+; fall back to nested bib.match for v9 compat
    bib_matches = paper_json.get("bib_match", [])
    if not bib_matches:
        for b in bib:
            match = b.get("match", {}) or {}
            for source_name, m in match.items():
                if m is not None:
                    bib_matches.append({"bib_id": b["bib_id"], "service": source_name, **m})

    return {
        "metadata": metadata,
        "authors": authors,
        "text": text,
        "sections": sections,
        "bib": bib,
        "bib_matches": bib_matches,
        "xrefs": xrefs,
        "urls": urls,
        "table": tbls,
        "eq": equations,
        "figure": figs,
        "footnote": footnotes,
    }


def _write_json_file(paper_json: dict, suffix: str = "") -> str:
    """Serialize the full bibr JSON to a temp file, named after the DOI/title.

    Returns the path so a DownloadButton can serve it; the basename becomes the
    downloaded filename. ``suffix`` disambiguates variants (e.g. no-images).
    """
    metadata = paper_json.get("metadata", {}) or {}
    slug = metadata.get("doi") or metadata.get("title") or "bibr"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(slug)).strip("_")[:60] or "bibr"
    tmpdir = tempfile.mkdtemp(prefix="bibr_json_")
    path = os.path.join(tmpdir, f"{slug}{suffix}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(paper_json, fh, indent=2, ensure_ascii=False)
    return path


def _strip_figure_images(paper_json: dict) -> dict:
    """Return a shallow copy of the bibr JSON with base64 figure images dropped.

    Keeps the JSON tree viewer readable and backs the "no images" download.
    """
    figures = paper_json.get("figure")
    if not figures:
        return paper_json
    stripped = dict(paper_json)
    stripped["figure"] = [
        {**f, "image": None} if isinstance(f, dict) and f.get("image") else f for f in figures
    ]
    return stripped


def _build_summary_md(result: dict) -> str:
    """Build a markdown summary card from the API result dict."""
    meta = result.get("metadata", {})
    lines = []
    lines.append(f"### {meta.get('title') or '(untitled)'}")
    if meta.get("doi"):
        lines.append(f"**DOI:** `{meta['doi']}`")
    if meta.get("paper_type"):
        conf = (
            f" ({meta['paper_type_confidence']:.2f})" if meta.get("paper_type_confidence") else ""
        )
        lines.append(f"**Paper type:** {meta['paper_type']}{conf}")
    if meta.get("oecd_l1"):
        domain = meta["oecd_l1"]
        if meta.get("oecd_l2"):
            domain += f" > {meta['oecd_l2']}"
        conf = f" ({meta['oecd_confidence']:.2f})" if meta.get("oecd_confidence") else ""
        lines.append(f"**OECD domain:** {domain}{conf}")
    if meta.get("keywords"):
        lines.append(f"**Keywords:** {', '.join(meta['keywords'])}")
    authors = result.get("authors", [])
    refs = result.get("bib", [])
    sections = result.get("sections", [])
    sentences = result.get("text", [])
    tables = result.get("table", [])
    figures = result.get("figure", [])
    bib_matches = result.get("bib_matches", [])
    lines.append(f"**Authors:** {len(authors)} | **References:** {len(refs)}")
    if sections:
        lines.append(
            f"**Sections:** {len(sections)} | "
            f"**Tables:** {len(tables)} | "
            f"**Sentences:** {len(sentences)}"
        )
        extras = []
        equations = result.get("eq", [])
        links = result.get("urls", [])
        if figures:
            extras.append(f"**Figures:** {len(figures)}")
        if bib_matches:
            extras.append(f"**Crossref matches:** {len(bib_matches)}")
        if equations:
            extras.append(f"**Equations:** {len(equations)}")
        if links:
            extras.append(f"**Links:** {len(links)}")
        if extras:
            lines.append(" | ".join(extras))
    return "\n\n".join(lines)


def _build_authors_data(result: dict) -> list[list]:
    """Build authors table rows."""
    return [
        [
            a.get("given") or "",
            # A group author's whole name is ``literal``.
            a.get("family") or a.get("literal") or "",
            a.get("affiliation") or "",
            a.get("orcid", ""),
            a.get("email", ""),
            "Yes" if a.get("corresponding") else "",
        ]
        for a in result.get("authors", [])
    ]


def _build_sections_data(result: dict) -> list[list]:
    """Build sections table rows."""
    return [
        [
            s.get("header") or "",
            s.get("level", 0),
            s.get("section_type", ""),
            round(s.get("classification_score", 0) or 0, 3),
        ]
        for s in result.get("sections", [])
        if s.get("level", 0) > 0  # skip root
    ]


def _build_references_data(result: dict) -> list[list]:
    """Build references table rows."""
    # Build bib_id -> match lookup for display
    match_by_bib = {}
    for m in result.get("bib_matches", []):
        match_by_bib[m.get("bib_id")] = m
    return [
        [
            r.get("bib_id", ""),
            r.get("authors", ""),
            r.get("title", ""),
            r.get("year", ""),
            r.get("container", ""),
            r.get("doi", ""),
            r.get("bib_type", ""),
            "Y" if r.get("bib_id") in match_by_bib else "",
            f"{match_by_bib[r.get('bib_id')]['score']:.2f}"
            if r.get("bib_id") in match_by_bib
            and match_by_bib[r.get("bib_id")].get("score") is not None
            else "",
        ]
        for r in result.get("bib", [])
    ]


def _format_bib_authors(authors) -> str:
    """Format bib_match authors (list of {given, family} dicts) into a readable string."""
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    parts = []
    for a in authors:
        if isinstance(a, dict):
            name = " ".join(filter(None, [a.get("given"), a.get("family")])) or a.get("literal")
            if name:
                parts.append(name)
        else:
            parts.append(str(a))
    return ", ".join(parts)


def _build_bib_matches_data(result: dict) -> list[list]:
    """Build bib_matches table rows."""
    return [
        [
            m.get("bib_id", ""),
            m.get("service", "") or m.get("source", ""),
            round(m.get("score", 0) or 0, 2),
            m.get("title", ""),
            _format_bib_authors(m.get("author")),
            m.get("year", ""),
            m.get("container", ""),
            m.get("doi", ""),
            m.get("bib_type", ""),
        ]
        for m in result.get("bib_matches", [])
    ]


def _build_text_html(result: dict) -> str:
    """Render sentences grouped by section in a scrollable HTML container."""
    sentences = result.get("text", [])
    sections = result.get("sections", [])
    if not sentences:
        return "<p><em>No text was found in this paper.</em></p>"

    # Captions show with their figure or table; footnotes go after the body.
    captions = {row.get("text_id") for key in ("figure", "table") for row in result.get(key) or []}
    notes = {row.get("text_id") for row in result.get("footnote") or []}
    by_section: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for sent in sentences:
        if sent.get("text_id") in captions or sent.get("text_id") in notes:
            continue
        by_section[sent["section_id"]][sent["paragraph_id"]].append(sent["text"])

    section_map = {s.get("section_id", i): s for i, s in enumerate(sections)}

    parts = []
    # section_id is None for root / front-matter sentences (the schema
    # remaps section 0 → null). Sort those first and never compare None to int.
    for sec_id, paragraphs in sorted(by_section.items(), key=lambda kv: (kv[0] is not None, kv[0])):
        sec = section_map.get(sec_id, {})
        level = sec.get("level", 0)
        if level > 0:
            tag = f"h{min(level + 2, 6)}"
            header = _esc(sec.get("header") or "")
            badge = ""
            sec_type = sec.get("section_type", "")
            if sec_type and sec_type != "unknown":
                badge = (
                    f' <span style="font-size:0.7em; color:#666; font-weight:normal;">'
                    f"[{_esc(sec_type)}]</span>"
                )
            parts.append(f"<{tag}>{header}{badge}</{tag}>")

        for _pid, sents in sorted(paragraphs.items()):
            joined = "  ".join(_esc(s) for s in sents)
            parts.append(f"<p>{joined}</p>")

    note_rows = [sent for sent in sentences if sent.get("text_id") in notes]
    if note_rows:
        parts.append("<h4>Footnotes</h4>")
        parts.extend(f"<p>{_esc(sent['text'])}</p>" for sent in note_rows)

    body = "\n".join(parts)
    return (
        '<div style="max-height:500px; overflow-y:auto; padding:0 1em; '
        'line-height:1.6; font-size:0.95em;">'
        f"{body}</div>"
    )


def _build_tables_html(result: dict) -> str:
    """Render extracted tables as HTML."""
    tables = result.get("table", [])
    if not tables:
        return "<p><em>No tables were found.</em></p>"

    parts = []
    for tbl in tables:
        parts.append(f"<p><strong>Table {_esc(str(tbl.get('table_id', '')))}</strong></p>")

        contents = tbl.get("contents")
        if contents and isinstance(contents, list) and len(contents) > 0:
            headers = contents[0]
            rows = contents[1:]
            html = "<table><thead><tr>"
            html += "".join(f"<th>{_esc(str(h))}</th>" for h in headers)
            html += "</tr></thead><tbody>"
            for row in rows:
                html += "<tr>" + "".join(f"<td>{_esc(str(c))}</td>" for c in row) + "</tr>"
            html += "</tbody></table>"
            parts.append(html)
        else:
            parts.append("<p><em>(no table data)</em></p>")
        parts.append("<hr>")

    return "\n".join(parts)


def _build_xrefs_data(result: dict) -> list[list]:
    """Build cross-references table rows."""
    rows = []
    for x in result.get("xrefs", []):
        rows.append(
            [
                x.get("xref_type", ""),
                x.get("contents", ""),
                x.get("text_id", ""),
                x.get("target_id", ""),
            ]
        )
    return rows


def _build_equations_data(result: dict) -> list[list]:
    """Build equations table rows."""
    return [
        [
            e.get("grp_id", ""),
            e.get("lhs", ""),
            e.get("df") or "",
            e.get("comp", ""),
            e.get("rhs", ""),
            e.get("text_id", ""),
        ]
        for e in result.get("eq", [])
    ]


def _build_links_data(result: dict) -> list[list]:
    """Build URL links table rows."""
    return [
        [
            lnk.get("href", ""),
            lnk.get("link_text", ""),
            lnk.get("text_id", ""),
        ]
        for lnk in result.get("urls", [])
    ]


def _build_figures_html(result: dict) -> str:
    """Render extracted figures as HTML with inline base64 images."""
    figures = result.get("figure", [])
    if not figures:
        return "<p><em>No figures were found.</em></p>"

    parts = []
    for f in figures:
        parts.append(f"<p><strong>Figure {_esc(str(f.get('figure_id', '')))}</strong></p>")
        img = f.get("image")
        if img:
            # v12 exports the image as a data URI that names its media type.
            parts.append(f'<img src="{_esc(img)}" style="max-width:100%; height:auto;" />')
        caption = f.get("caption")
        if caption:
            parts.append(f"<p><em>{_esc(caption)}</em></p>")
        parts.append("<hr>")
    return "\n".join(parts)


_STAGE_PROGRESS: dict[str, tuple[float, str]] = {
    "validate": (0.05, "Checking file..."),
    "docx": (0.10, "Reading DOCX..."),
    "jats": (0.10, "Reading JATS XML..."),
    "html": (0.10, "Reading HTML/ePub..."),
    "layout": (0.15, "Analyzing page layout..."),
    "ocr": (0.35, "Reading text (OCR)..."),
    "parse": (0.55, "Structuring content..."),
    "extract": (0.70, "Extracting metadata..."),
    "enrich": (0.85, "Looking up references on Crossref..."),
    "export": (0.95, "Writing output..."),
}


class _GradioProgress:
    """Bridges bibr ProgressTracker protocol to Gradio progress bar."""

    def __init__(self, progress: gr.Progress):
        self._progress = progress

    def stage_start(self, name: str, detail: str = "") -> None:  # noqa: ARG002
        frac, desc = _STAGE_PROGRESS.get(name, (0.5, "Processing..."))
        self._progress(frac, desc=desc)

    def stage_end(self, name: str) -> None:
        pass

    def ocr_start(self, total_regions: int) -> None:
        pass

    def ocr_region_done(self) -> None:
        pass

    def ocr_end(self) -> None:
        pass


def _build_status_md(
    ocr_backend: str,
    llm_provider: str,
    llm_model: str,
    refs_strategy: str = "llm",
    llm_backend: str | None = None,
) -> str:
    """Build a compact status line showing active backend configuration."""
    backend = f" `{llm_backend}`" if llm_backend else ""
    return (
        f"**OCR** `{ocr_backend}` · **LLM**{backend} `{llm_provider}/{llm_model}` "
        f"· **Refs** `{refs_strategy}` "
    )


async def _replace_demo_pipeline_with_preset(
    name: str,
    *,
    manager,
    settings,
    pipeline_state: dict[str, Any],
    pipeline_factory,
    normalize_ocr_backend,
    refs: str | None,
) -> str:
    """Apply one preset, atomically replace the demo pipeline, and close the old one."""
    settings_snapshot = copy.deepcopy(vars(settings))
    try:
        unknown = manager.apply_to_settings(name, settings)
        if unknown:
            logger.warning("Preset %s has unknown settings: %s", name, ", ".join(sorted(unknown)))

        new_ocr = normalize_ocr_backend(settings.ocr.backend)
        new_llm = pipeline_state["llm_backend"] or settings.llm.backend
        replacement = pipeline_factory(
            memory_mode=pipeline_state["memory_mode"],
            ocr_backend=new_ocr,
            llm_backend=new_llm,
            ref_parse_strategy=refs,
        )
    except Exception:
        # apply_to_settings mutates the singleton incrementally. If validation
        # or replacement construction fails, keep the old pipeline and restore
        # its matching configuration instead of leaving a half-applied preset.
        vars(settings).clear()
        vars(settings).update(settings_snapshot)
        raise

    old_pipeline = pipeline_state["pipeline"]
    pipeline_state["pipeline"] = replacement
    pipeline_state["ocr_backend"] = new_ocr

    try:
        await old_pipeline.aclose()
    except Exception:  # noqa: BLE001 — replacement is live; cleanup stays best-effort
        logger.warning("Previous demo pipeline cleanup failed", exc_info=True)

    effective_refs = refs or settings.REF_PARSE_STRATEGY or "ner"
    effective_backend = getattr(replacement, "llm_backend", new_llm)
    return _build_status_md(
        new_ocr,
        settings.llm.provider,
        settings.llm.model,
        effective_refs,
        effective_backend,
    )


def create_local_demo(
    ocr_backend: str | None = None,
    memory_mode: str | None = None,
    llm_backend: str | None = None,
    presets_enabled: bool = False,
    refs: str | None = None,
) -> gr.Blocks:
    """Create a Gradio demo that runs the bibr pipeline locally."""
    from bibr.config import Settings
    from bibr.local.cli import normalize_ocr_backend
    from bibr.local.pipeline import LocalPipeline

    ocr_backend = normalize_ocr_backend(ocr_backend)

    if "ocr" not in Settings.cache.model_fields_set:
        # Demo users repeatedly re-run the same PDF while poking at the UI;
        # skipping OCR inference on repeat runs is a better default here than
        # in the batch pipeline, where silently caching results is unwanted.
        Settings.cache.ocr = True
        logger.info("Local demo — enabling OCR disk cache (CACHE_OCR) by default")

    # The reference-parse strategy rides the pipeline's per-run config
    # (mirrors `bibr chew --refs`) instead of mutating the process-global
    # Settings; the status line shows the effective value.
    effective_refs = refs or Settings.REF_PARSE_STRATEGY or "ner"

    # Figure base64 images are only emitted when FIGURE_IMAGES is on; without
    # them a "no images" download would be identical to the full one, so we
    # only offer the second button when figures are actually present.
    figures_enabled = bool(Settings.FIGURE_IMAGES)

    pipeline = LocalPipeline(
        memory_mode=memory_mode,
        ocr_backend=ocr_backend,
        llm_backend=llm_backend,
        ref_parse_strategy=refs,
    )
    pipeline_state: dict[str, Any] = {
        "pipeline": pipeline,
        "ocr_backend": ocr_backend,
        "memory_mode": memory_mode,
        "llm_backend": llm_backend,
    }
    pipeline_lock = asyncio.Lock()
    logger.info(
        "Local pipeline ready (ocr=%s, llm=%s, memory=%s)",
        ocr_backend,
        llm_backend or Settings.llm.backend,
        getattr(pipeline, "memory_mode", memory_mode),
    )

    status_md = _build_status_md(
        ocr_backend,
        Settings.llm.provider,
        Settings.llm.model,
        effective_refs,
        getattr(pipeline_state["pipeline"], "llm_backend", llm_backend or Settings.llm.backend),
    )

    with gr.Blocks(title="bibr 🦫 Demo") as demo:
        gr.Markdown(_HEADER_MD)
        status_display = gr.Markdown(status_md)

        if presets_enabled:
            from pathlib import Path

            from bibr.presets import PresetManager

            manager = PresetManager()
            preset_names = manager.list_presets()

            if preset_names:
                active = manager.get_active(Path.cwd() / ".env")
                preset_dropdown = gr.Dropdown(
                    choices=["(current .env)"] + preset_names,
                    value=active if active in preset_names else "(current .env)",
                    label="Configuration Preset",
                    interactive=True,
                )

                async def _switch_preset(name):
                    if name == "(current .env)" or not name:
                        return status_md
                    try:
                        async with pipeline_lock:
                            return await _replace_demo_pipeline_with_preset(
                                name,
                                manager=manager,
                                settings=Settings,
                                pipeline_state=pipeline_state,
                                pipeline_factory=LocalPipeline,
                                normalize_ocr_backend=normalize_ocr_backend,
                                refs=refs,
                            )
                    except Exception as e:
                        raise gr.Error(f"Failed to apply preset: {e}") from e

                preset_dropdown.change(
                    fn=_switch_preset,
                    inputs=[preset_dropdown],
                    outputs=[status_display],
                )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload a paper",
                    file_count="single",
                    file_types=_ALLOWED_EXTENSIONS,
                )
                run_btn = gr.Button("Extract Metadata", variant="primary", size="lg")

            with gr.Column(scale=2):
                summary_output = gr.Markdown(
                    value="*Results will appear here after processing.*",
                    label="Summary",
                )
                with gr.Row():
                    download_btn = gr.DownloadButton(
                        "⬇ Download bibr JSON",
                        visible=False,
                        size="sm",
                    )
                    download_btn_noimg = (
                        gr.DownloadButton(
                            "⬇ Download JSON (no images)",
                            visible=False,
                            size="sm",
                        )
                        if figures_enabled
                        else None
                    )
                with gr.Accordion("View bibr JSON", open=False):
                    json_view = gr.JSON(label="bibr JSON")

        with gr.Tabs():
            with gr.TabItem("Text"):
                text_html = gr.HTML(
                    value="",
                    label="Paper Text",
                )
            with gr.TabItem("Authors"):
                authors_table = gr.Dataframe(
                    headers=[
                        "Given Name",
                        "Family Name",
                        "Affiliation",
                        "ORCID",
                        "Email",
                        "Corresponding",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Authors",
                )
            with gr.TabItem("Sections"):
                sections_table = gr.Dataframe(
                    headers=["Title", "Level", "Canonical Section", "Score"],
                    datatype=["str", "number", "str", "number"],
                    interactive=False,
                    label="Sections",
                )
            with gr.TabItem("Bibliography"):
                refs_table = gr.Dataframe(
                    headers=[
                        "#",
                        "Authors",
                        "Title",
                        "Year",
                        "Container",
                        "DOI",
                        "Type",
                        "Match",
                        "Score",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    interactive=False,
                    label="Bibliography",
                )
            with gr.TabItem("Bib Matches"):
                bib_matches_table = gr.Dataframe(
                    headers=[
                        "Bib #",
                        "Service",
                        "Score",
                        "Title",
                        "Authors",
                        "Year",
                        "Container",
                        "DOI",
                        "Type",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "number",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    interactive=False,
                    label="Crossref Matches",
                )
            with gr.TabItem("Xrefs"):
                xrefs_table = gr.Dataframe(
                    headers=["Type", "Reference Text", "Sentence ID", "Target ID"],
                    datatype=["str", "str", "number", "number"],
                    interactive=False,
                    label="Cross-References",
                )
            with gr.TabItem("Tables"):
                tables_html = gr.HTML(
                    value="",
                    label="Extracted Tables",
                )
            with gr.TabItem("Figures"):
                figures_html = gr.HTML(
                    value="",
                    label="Figures",
                )
            with gr.TabItem("Equations"):
                equations_table = gr.Dataframe(
                    headers=["Group", "LHS", "df", "Operator", "RHS", "Sentence ID"],
                    datatype=["number", "str", "str", "str", "str", "number"],
                    interactive=False,
                    label="Equations",
                )
            with gr.TabItem("Links"):
                links_table = gr.Dataframe(
                    headers=["URL", "Link Text", "Sentence ID"],
                    datatype=["str", "str", "number"],
                    interactive=False,
                    label="URL Links",
                )

        async def _process_file(file_path, progress=gr.Progress()):
            if file_path is None:
                raise gr.Error("Drop a PDF or DOCX file above, then click Extract Metadata.")
            _check_file_size(file_path)

            progress(0.0, desc="Reading file...")

            tracker = _GradioProgress(progress)
            async with pipeline_lock:
                paper_json = await pipeline_state["pipeline"].process_file(
                    file_path, progress=tracker
                )

            progress(0.98, desc="Parsing response...")
            result = _parse_json_response(paper_json)
            # The JSON tree viewer shows the image-free copy (base64 blobs are
            # unreadable and heavy in-browser); downloads carry the real bytes.
            stripped_json = _strip_figure_images(paper_json)
            download_updates = [gr.DownloadButton(value=_write_json_file(paper_json), visible=True)]
            if figures_enabled:
                download_updates.append(
                    gr.DownloadButton(
                        value=_write_json_file(stripped_json, suffix=".no-images"),
                        visible=True,
                    )
                )
            progress(1.0, desc="Done!")

            return (
                _build_summary_md(result),
                _build_text_html(result),
                _build_authors_data(result),
                _build_sections_data(result),
                _build_references_data(result),
                _build_bib_matches_data(result),
                _build_xrefs_data(result),
                _build_tables_html(result),
                _build_figures_html(result),
                _build_equations_data(result),
                _build_links_data(result),
                *download_updates,
                stripped_json,
            )

        download_outputs = [download_btn]
        if download_btn_noimg is not None:
            download_outputs.append(download_btn_noimg)

        run_btn.click(
            fn=_process_file,
            inputs=[file_input],
            outputs=[
                summary_output,
                text_html,
                authors_table,
                sections_table,
                refs_table,
                bib_matches_table,
                xrefs_table,
                tables_html,
                figures_html,
                equations_table,
                links_table,
                *download_outputs,
                json_view,
            ],
            api_name=False,
        )

    return demo
