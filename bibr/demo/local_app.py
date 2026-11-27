"""Local Gradio demo for bibr.

Runs the bibr pipeline in-process: no separate API server. With nothing
configured it runs in fast mode (local models only, no LLM, PDFs read from
their text layer); ``bibr demo --help`` lists the options.
"""

import asyncio
import copy
import csv
import io
import json
import logging
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from html import escape as _esc
from importlib import resources
from typing import Any
from urllib.parse import quote

import gradio as gr

from bibr.input.supported_files import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FILE_SIZE_MB = 10

# Uploaded papers and the downloads made from them are deleted once they are
# this old, and all of them when the server stops.
_DEFAULT_CACHE_TTL_SECONDS = 3600
# Gradio looks for files to delete this often (or every TTL, when that is
# shorter), so a file lives at most its TTL plus this long.
_CACHE_SWEEP_SECONDS = 300

_ALLOWED_EXTENSIONS: list[str] = sorted(SUPPORTED_EXTENSIONS)

# The formats as a reader names them, for the page and its messages.
_FORMATS_TEXT = "PDF, Word (.docx), JATS XML, HTML or ePub"

_ISSUES_URL = "https://github.com/scienceverse/bibr/issues"
_DEMO_GUIDE_URL = "https://bibr.org/guides/demo/"

# Theme variables only, so the page reads in light and dark mode alike. Secondary
# text is the body colour at reduced strength: Gradio's own subdued colour is too
# faint on the light theme (about 2.5:1 against the page) for text that has to be read.
DEMO_CSS = """\
.gradio-container { --bibr-muted: color-mix(in srgb, var(--body-text-color) 72%, transparent); }
.bibr-header h1 { margin-bottom: 0.2em; }
.bibr-header p { margin: 0.2em 0; }
.bibr-status p { margin: 0; }
.bibr-notice { padding: 0.65em 0.9em; border-radius: var(--radius-md);
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary); margin: 0.25em 0; }
.bibr-notice p { margin: 0; font-size: var(--text-sm); color: var(--body-text-color); }
.bibr-mode { margin: 0.2em 0; }
.bibr-hint p { margin: 0.4em 0 0; font-size: var(--text-sm);
    color: var(--bibr-muted); }
.bibr-placeholder { padding: 1.2em 1em; border: 1px dashed var(--border-color-primary);
    border-radius: var(--radius-lg); color: var(--bibr-muted); min-height: 5.5em; }
.bibr-placeholder p { margin: 0; }
.bibr-card { padding: 0.2em 0.1em 0.4em; overflow-wrap: anywhere; }
.bibr-card h2 { margin: 0 0 0.35em; font-size: 1.45em; line-height: 1.3; }
.bibr-card p { margin: 0.3em 0; }
.bibr-card .bibr-authors { font-size: 1.05em; }
.bibr-card .bibr-meta, .bibr-card .bibr-small { color: var(--bibr-muted); }
.bibr-card .bibr-small { font-size: var(--text-sm); }
.bibr-card a { color: var(--link-text-color); }
.bibr-abstract { margin-top: 0.6em; line-height: 1.55; position: relative; }
.bibr-abstract .bibr-clamp { display: -webkit-box; -webkit-box-orient: vertical;
    -webkit-line-clamp: 6; line-clamp: 6; overflow: hidden; }
.bibr-abstract input.bibr-more { position: absolute; opacity: 0; width: 1px; height: 1px; }
.bibr-abstract input.bibr-more:checked ~ .bibr-clamp { -webkit-line-clamp: unset;
    line-clamp: unset; display: block; }
.bibr-abstract label.bibr-more-label { cursor: pointer; color: var(--link-text-color);
    font-size: var(--text-sm); }
.bibr-abstract input.bibr-more:checked ~ label .bibr-show { display: none; }
.bibr-abstract input.bibr-more:not(:checked) ~ label .bibr-hide { display: none; }
.bibr-abstract input.bibr-more:focus-visible ~ label { outline: 2px solid var(--color-accent); }
.bibr-chips { display: flex; flex-wrap: wrap; gap: 0.4em; margin-top: 0.7em; }
.bibr-chip { padding: 0.1em 0.65em; border-radius: 999px; font-size: var(--text-sm);
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary); }
.bibr-error { padding: 0.9em 1.1em; border-radius: var(--radius-lg);
    border: 1px solid var(--error-border-color); background: var(--error-background-fill);
    color: var(--body-text-color); }
.bibr-error h3 { margin: 0 0 0.4em; color: var(--error-text-color); }
.bibr-error p { margin: 0.35em 0; }
.bibr-error code { white-space: nowrap; }
.bibr-card code { overflow-wrap: anywhere; }
.bibr-error a { padding: 0; margin: 0; }
.bibr-scroll { max-height: 70vh; overflow: auto; padding: 0 0.6em; line-height: 1.6; }
.bibr-scroll h3, .bibr-scroll h4, .bibr-scroll h5, .bibr-scroll h6 { margin: 1em 0 0.3em; }
.bibr-caption { font-style: italic; color: var(--bibr-muted); }
.bibr-xscroll { overflow-x: auto; margin-bottom: 0.4em; }
.bibr-xscroll table { border-collapse: collapse; font-size: var(--text-sm); }
.bibr-xscroll th, .bibr-xscroll td { border: 1px solid var(--border-color-primary);
    padding: 0.25em 0.5em; vertical-align: top; text-align: left; }
.bibr-xscroll th { background: var(--background-fill-secondary); }
.bibr-tableblock { margin: 0 0 1.4em; }
.bibr-tableblock figcaption { margin-bottom: 0.4em; }
.bibr-figure img { max-width: 100%; height: auto; }
.bibr-empty p { color: var(--bibr-muted); font-style: italic; }
.bibr-downloads { align-items: center; gap: 0.5em; }
.bibr-table, .bibr-table table, .bibr-table th, .bibr-table td, .bibr-table span,
.bibr-table input { font-family: var(--font); }
/* Column widths are in px: on a narrow screen a table scrolls inside its box.
   Gradio draws the visible cells as divs; a long DOI or URL breaks anywhere. */
.bibr-table .md, .bibr-table .md p { white-space: normal; }
.bibr-table a { word-break: break-all; }
@media (max-width: 640px) {
    /* More tab names fit before the overflow menu. */
    .tab-wrapper .tab-container button { padding-left: 8px; padding-right: 8px; }
}
/* Gradio keeps the divider of the API link this demo does not show. */
footer .show-api-divider { display: none; }
/* Tables scroll inside their own box; a shadow shows there is more to see. */
.table-wrap { scrollbar-width: thin; }
.table-wrap.can-scroll-right {
    box-shadow: inset -16px 0 12px -12px
        color-mix(in srgb, var(--body-text-color) 22%, transparent); }
.table-wrap.can-scroll-left {
    box-shadow: inset 16px 0 12px -12px
        color-mix(in srgb, var(--body-text-color) 22%, transparent); }
.table-wrap.can-scroll-left.can-scroll-right {
    box-shadow: inset 16px 0 12px -12px
            color-mix(in srgb, var(--body-text-color) 22%, transparent),
        inset -16px 0 12px -12px
            color-mix(in srgb, var(--body-text-color) 22%, transparent); }
"""

DEMO_JS = """\
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

Turn a research paper into structured data: title, authors, abstract, full text, \
references and tables.
"""

_PLACEHOLDER_HTML = (
    '<div class="bibr-placeholder">'
    "<p>Choose a paper above and click <strong>Extract</strong>, or try the sample paper. "
    "The title, authors and abstract show here, with downloads, and the references, "
    "full text and tables in tabs below them.</p></div>"
)

_READY_HTML = (
    '<div class="bibr-placeholder"><p>Ready. Click <strong>Extract</strong> to read '
    "this paper.</p></div>"
)


def _max_file_size_mb() -> int:
    """The upload limit in MB, from ``DEMO_MAX_FILE_SIZE_MB`` (default 10).

    Any value that is not a positive whole number raises ValueError: ``0`` or
    a negative limit would refuse every upload.
    """
    raw = os.environ.get("DEMO_MAX_FILE_SIZE_MB", str(_DEFAULT_MAX_FILE_SIZE_MB))
    try:
        limit = int(raw)
        if limit <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"DEMO_MAX_FILE_SIZE_MB must be a positive whole number of MB; got {raw!r}"
        ) from None
    return limit


class _ShownError(Exception):
    """A failure the page explains: a title, then paragraphs of HTML (already escaped)."""

    def __init__(self, title: str, *paragraphs: str):
        super().__init__(title)
        self.heading = title
        self.paragraphs = paragraphs


def _check_file_size(file_path: str) -> None:
    """Raise :class:`_ShownError` if the file exceeds the size limit."""
    limit_mb = _max_file_size_mb()
    size = os.path.getsize(file_path)
    if size > limit_mb * 1024 * 1024:
        size_mb = size / (1024 * 1024)
        raise _ShownError(
            "This file is too large for the demo",
            f"It is {size_mb:.1f} MB, and this demo takes files up to {limit_mb} MB.",
            "Try a smaller version of the paper (without supplements, for example), or "
            "install bibr and run <code>bibr chew</code> on it on your own computer.",
        )


def _slug(paper_json: dict) -> str:
    """A file name stem for the downloads: the DOI, else the title, cut at a word."""
    metadata = paper_json.get("metadata", {}) or {}
    raw = metadata.get("doi") or metadata.get("title") or ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw)).strip("_.")
    if len(slug) > 60:
        cut = slug[:61]
        # Cut at the last word boundary that keeps a useful stem.
        boundary = max(cut.rfind("_"), cut.rfind("-"))
        slug = cut[:boundary] if boundary >= 20 else slug[:60]
        slug = slug.rstrip("_.-")
    return slug or "bibr"


def _export_folder() -> str:
    """A new folder inside Gradio's temp folder for one paper's downloads.

    Inside it, Gradio serves the files in place and deletes them with the
    uploads (``DEMO_CACHE_TTL_SECONDS``). A file outside it would be copied in,
    and the original never deleted. Gradio removes only the files, not their
    empty folder; the web demo guide describes a sweep for a hosted demo.
    """
    from gradio.utils import get_upload_folder

    upload_folder = get_upload_folder()
    os.makedirs(upload_folder, exist_ok=True)
    return tempfile.mkdtemp(prefix="bibr_json_", dir=upload_folder)


def _write_json_file(paper_json: dict, suffix: str = "", folder: str | None = None) -> str:
    """Serialize the full bibr JSON to a file named after the DOI or title.

    Returns the path so a DownloadButton can serve it; the basename becomes the
    downloaded filename. ``suffix`` disambiguates variants (e.g. no-images).
    """
    folder = folder or _export_folder()
    path = os.path.join(folder, f"{_slug(paper_json)}{suffix}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(paper_json, fh, indent=2, ensure_ascii=False)
    return path


def _write_references_file(paper_json: dict, fmt: str, folder: str) -> str | None:
    """The paper's references as BibTeX (``bibtex``) or RIS (``ris``); None when it has none."""
    from bibr.export.citations import render_references

    text, count = render_references([paper_json], fmt)
    if not count:
        return None
    path = os.path.join(folder, f"{_slug(paper_json)}.{'bib' if fmt == 'bibtex' else 'ris'}")
    # newline="" keeps RIS's CRLF line endings as they are on every platform.
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path


def _write_data_tables_zip(paper_json: dict, folder: str) -> str:
    """The paper as bibr's data tables (one Parquet file per table), zipped."""
    from bibr.export.tables import write_tables

    tables_dir = tempfile.mkdtemp(prefix="tables_", dir=folder)
    report = write_tables([paper_json], tables_dir)
    path = os.path.join(folder, f"{_slug(paper_json)}-data-tables.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for table_path in report.files.values():
            zf.write(table_path, arcname=os.path.basename(table_path))
    for table_path in report.files.values():
        os.remove(table_path)
    os.rmdir(tables_dir)
    return path


def _write_tables_csv_zip(paper_json: dict, folder: str) -> str | None:
    """Each table printed in the paper as a CSV file, zipped; None when it has none."""
    files: list[tuple[str, str]] = []
    for position, table in enumerate(paper_json.get("table") or [], start=1):
        contents = table.get("contents")
        if not contents or not isinstance(contents, list):
            continue
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        for row in contents:
            writer.writerow(["" if cell is None else str(cell) for cell in row])
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(table.get("label") or position))
        files.append((f"table_{label}.csv", buffer.getvalue()))
    if not files:
        return None
    path = os.path.join(folder, f"{_slug(paper_json)}-tables-csv.zip")
    seen: set[str] = set()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for position, (name, text) in enumerate(files, start=1):
            if name in seen:
                name = f"{name[:-4]}_{position}.csv"
            seen.add(name)
            zf.writestr(name, text)
    return path


def _write_downloads(paper_json: dict, stripped_json: dict, with_no_images: bool) -> dict:
    """Write every download for one paper into one folder; a path, or None, per kind.

    A download that fails is left out (and logged) rather than failing the run.
    """
    folder = _export_folder()
    paths: dict[str, str | None] = {"json": _write_json_file(paper_json, folder=folder)}
    if with_no_images:
        paths["json_no_images"] = _write_json_file(
            stripped_json, suffix=".no-images", folder=folder
        )
    writers = {
        "bibtex": lambda: _write_references_file(paper_json, "bibtex", folder),
        "ris": lambda: _write_references_file(paper_json, "ris", folder),
        "data_tables": lambda: _write_data_tables_zip(paper_json, folder),
        "tables_csv": lambda: _write_tables_csv_zip(paper_json, folder),
    }
    for kind, write in writers.items():
        try:
            paths[kind] = write()
        except Exception:  # noqa: BLE001 — one missing download must not hide the results
            logger.warning("Could not write the %s download", kind, exc_info=True)
            paths[kind] = None
    return paths


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


def _sample_paper_bytes() -> bytes:
    """The synthetic sample paper that ships with bibr (``bibr/data/sample_paper.pdf``)."""
    return resources.files("bibr").joinpath("data", "sample_paper.pdf").read_bytes()


def _copy_sample_paper() -> str:
    """Copy the sample paper into Gradio's temp folder, which Gradio may serve from."""
    from gradio.utils import get_upload_folder

    upload_folder = get_upload_folder()
    os.makedirs(upload_folder, exist_ok=True)
    folder = tempfile.mkdtemp(prefix="bibr_sample_", dir=upload_folder)
    path = os.path.join(folder, "sample_paper.pdf")
    with open(path, "wb") as fh:
        fh.write(_sample_paper_bytes())
    return path


# ``:`` and ``@`` stop GFM autolinks (bare URLs and email addresses), ``$``
# stops gr.Markdown's ``$$`` LaTeX delimiter.
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~:@$])")


def _md_text(value: object) -> str:
    """Show extracted text literally in Markdown: no HTML, links, images or emphasis.

    Titles and keywords come from the uploaded document, so a crafted PDF could
    otherwise make the viewer's browser load an outside image or follow a link.
    Quotes are left as they are: the ``#`` of an entity such as ``&#x27;``
    would be escaped, and the entity shown literally.
    """
    flat = " ".join(str(value).split())
    return _MD_SPECIAL.sub(r"\\\1", _esc(flat, quote=False))


def _flat(value: object) -> str:
    """Text from the document on one line, HTML-escaped."""
    return _esc(" ".join(str(value).split()))


def _doi_url(doi: str) -> str:
    """The doi.org link for a DOI; every character that could leave the path is encoded."""
    return "https://doi.org/" + quote(doi.strip(), safe="/")


def _doi_link_html(doi: str) -> str:
    return (
        f'<a href="{_esc(_doi_url(doi))}" target="_blank" rel="noopener noreferrer">'
        f"{_flat(doi)}</a>"
    )


def _doi_link_md(doi: object) -> str:
    """A DOI as a Markdown link for a table cell; empty when there is none."""
    if not doi:
        return ""
    return f"[{_md_text(doi)}]({_doi_url(str(doi))})"


def _plural(count: int, word: str, plural: str | None = None) -> str:
    return f"{count} {word if count == 1 else plural or word + 's'}"


def _author_name(author: dict) -> str:
    # A group author's whole name is ``literal``.
    return " ".join(filter(None, [author.get("given"), author.get("family")])) or (
        author.get("literal") or ""
    )


def _year_of(metadata: dict) -> str:
    """The publication year from ``published_date`` (ISO), else from the printed date."""
    for value in (metadata.get("published_date"), metadata.get("published")):
        match = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", str(value or ""))
        if match:
            return match.group(1)
    return ""


def _venue_parts(metadata: dict) -> list[str]:
    """Journal, year, volume(issue) and pages, as escaped HTML parts."""
    parts = []
    if metadata.get("journal"):
        parts.append(f"<em>{_flat(metadata['journal'])}</em>")
    year = _year_of(metadata)
    if year:
        parts.append(year)
    volume, issue = metadata.get("volume"), metadata.get("issue")
    if volume:
        parts.append(f"vol. {_flat(volume)}" + (f" ({_flat(issue)})" if issue else ""))
    elif issue:
        parts.append(f"issue {_flat(issue)}")
    first, last = metadata.get("first_page"), metadata.get("last_page")
    if first and last and first != last:
        parts.append(f"pp. {_flat(first)}–{_flat(last)}")
    elif first:
        parts.append(f"p. {_flat(first)}")
    return parts


def _build_paper_card_html(result: dict) -> str:
    """The result header: what a reader looks for first, then small counts.

    Every value from the document is escaped (``_esc``); the DOI link is built
    from an encoded DOI, so it can only point at doi.org.
    """
    meta = result.get("metadata", {}) or {}
    parts = ['<div class="bibr-card">']
    parts.append(f"<h2>{_flat(meta.get('title') or '(No title found)')}</h2>")

    authors = result.get("authors", []) or []
    names = []
    for author in authors:
        name = _author_name(author)
        if not name:
            continue
        mark = "<sup>*</sup>" if author.get("corresponding") else ""
        names.append(f"{_flat(name)}{mark}")
    if names:
        note = ""
        if any(a.get("corresponding") for a in authors):
            note = ' <span class="bibr-small">(* corresponding author)</span>'
        parts.append(f'<p class="bibr-authors">{", ".join(names)}{note}</p>')

    venue = _venue_parts(meta)
    if meta.get("doi"):
        venue.append(f"DOI {_doi_link_html(str(meta['doi']))}")
    if venue:
        parts.append(f'<p class="bibr-meta">{" · ".join(venue)}</p>')

    if meta.get("keywords"):
        keywords = "; ".join(_flat(k) for k in meta["keywords"])
        parts.append(f'<p class="bibr-small"><strong>Keywords:</strong> {keywords}</p>')

    abstract = meta.get("abstract")
    if abstract:
        text = _flat(abstract)
        if len(abstract) > 900:
            # A checkbox and a label open the rest without any script.
            parts.append(
                '<div class="bibr-abstract"><input type="checkbox" class="bibr-more" '
                'id="bibr-abstract-more">'
                f'<p class="bibr-clamp"><strong>Abstract.</strong> {text}</p>'
                '<label class="bibr-more-label" for="bibr-abstract-more">'
                '<span class="bibr-show">Show the whole abstract</span>'
                '<span class="bibr-hide">Show less</span></label></div>'
            )
        else:
            parts.append(
                f'<div class="bibr-abstract"><p><strong>Abstract.</strong> {text}</p></div>'
            )

    classification = []
    if meta.get("paper_type"):
        classification.append(f"Paper type: {_flat(meta['paper_type'])}")
    if meta.get("oecd_l1"):
        field = _flat(meta["oecd_l1"])
        if meta.get("oecd_l2"):
            field += f" › {_flat(meta['oecd_l2'])}"
        classification.append(f"Field: {field}")
    if classification:
        parts.append(f'<p class="bibr-small">{" · ".join(classification)}</p>')

    counts = [
        (len(authors), "author", None),
        (len(result.get("bib", []) or []), "reference", None),
        (
            len([s for s in result.get("sections", []) or [] if s.get("level", 0) > 0]),
            "section",
            None,
        ),
        (len(result.get("table", []) or []), "table", None),
        (len(result.get("figure", []) or []), "figure", None),
        (len(result.get("eq", []) or []), "statistic", None),
        (len(result.get("xrefs", []) or []), "in-text citation", None),
        (len(result.get("urls", []) or []), "link", None),
    ]
    chips = [
        f'<span class="bibr-chip">{_plural(n, word, plural)}</span>'
        for n, word, plural in counts
        if n
    ]
    if chips:
        parts.append(f'<div class="bibr-chips">{"".join(chips)}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def _error_html(error: _ShownError) -> str:
    paragraphs = "".join(f"<p>{p}</p>" for p in error.paragraphs)
    return f'<div class="bibr-error" role="alert"><h3>{_esc(error.heading)}</h3>{paragraphs}</div>'


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


def _cache_lifetime() -> tuple[int, int] | None:
    """Gradio ``delete_cache`` (frequency, age) from ``DEMO_CACHE_TTL_SECONDS``.

    Files are deleted once they are ``ttl`` seconds old, checked every
    ``_CACHE_SWEEP_SECONDS`` (or every ``ttl`` when shorter). ``0`` keeps them
    and returns None. Any other value that is not a positive whole number of
    seconds raises ValueError, so a typo cannot silently keep every file.
    """
    raw = os.environ.get("DEMO_CACHE_TTL_SECONDS", str(_DEFAULT_CACHE_TTL_SECONDS))
    try:
        ttl = int(raw)
        if ttl < 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            "DEMO_CACHE_TTL_SECONDS must be a whole number of seconds, "
            f"or 0 to keep uploads and downloads; got {raw!r}"
        ) from None
    if ttl == 0:
        return None
    return (min(ttl, _CACHE_SWEEP_SECONDS), ttl)


def _explain_failure(exc: BaseException, *, file_name: str, text_layer_only: bool) -> _ShownError:
    """What a failed run says in the browser: what happened, then what to do next.

    Written for someone at a browser, who may not have started the demo: a
    command-line option is offered only as what to restart the demo with.
    bibr's own message, which can name paths and command-line flags, stays in
    the demo's log.
    """
    from bibr.exceptions import InputValidationError, ProcessingError, UpstreamServiceError

    name = f"<strong>{_flat(file_name)}</strong>"
    code = getattr(exc, "error_code", None)
    if isinstance(exc, ProcessingError) and code == "scan_needs_ocr":
        how = (
            "This demo reads the text layer only, without OCR (optical character "
            "recognition, which reads text from an image of a page)."
            if text_layer_only
            else "The OCR this demo uses did not read it."
        )
        return _ShownError(
            "This PDF is a scan, so the demo can't read it",
            f"{name} has no usable text layer: its pages are pictures of text. {how}",
            "Try a copy of the paper from the publisher's site, which usually has a text "
            "layer. If you started this demo yourself, restart it with an OCR backend, for "
            "example <code>bibr demo --ocr anthropic</code> (a cloud model that needs an "
            "Anthropic API key; the page images leave the machine the demo runs on). "
            f'<a href="{_DEMO_GUIDE_URL}#scanned-pdfs" target="_blank" '
            'rel="noopener noreferrer">The web demo guide</a> lists the other options, and '
            'the hosted demo at <a href="https://try.bibr.org" target="_blank" '
            'rel="noopener noreferrer">try.bibr.org</a> reads scans too.',
        )
    if code == "corrupted_file":
        return _ShownError(
            "bibr couldn't open this file",
            f"{name} looks damaged, or it is not really the kind of file its name says "
            "(for example, another file renamed to .pdf).",
            "Check that it opens on your computer, then download it again or try another "
            f"copy. The demo reads {_FORMATS_TEXT} files.",
        )
    if code == "encrypted_file":
        return _ShownError(
            "This file is password-protected",
            f"bibr can't read {name} because it is encrypted.",
            "Save a copy without the password (most PDF readers can), then upload that.",
        )
    if code == "unsupported_format":
        return _ShownError(
            "bibr can't read this kind of file",
            f"{name} is not a kind of file bibr reads, or its content is not what its name "
            "says (a .pdf that is not a PDF, for example).",
            f"Upload a {_FORMATS_TEXT} file.",
        )
    if getattr(exc, "outage", False) or isinstance(exc, UpstreamServiceError):
        return _ShownError(
            "A service the demo needs didn't respond",
            f"Reading {name} stopped because a model or service the demo uses was "
            "unavailable. This is not a problem with your paper.",
            "Try again in a minute. If it keeps happening, the details are in the demo's log.",
        )
    if code in ("ocr_failed", "ocr_mostly_failed"):
        return _ShownError(
            "Reading the text with OCR failed",
            f"The OCR backend returned too little text for {name}.",
            "Try again. If it keeps failing, whoever runs the demo can check the OCR "
            "backend with <code>bibr doctor</code>.",
        )
    if isinstance(exc, (ProcessingError, InputValidationError)):
        return _ShownError(
            "bibr couldn't finish reading this paper",
            f"Something in {name} stopped the extraction.",
            "Try another copy or another paper. The details are in the demo's log; if "
            "the same file keeps failing, please "
            f'<a href="{_ISSUES_URL}" target="_blank" rel="noopener noreferrer">report it</a>.',
        )
    return _ShownError(
        "Something went wrong",
        f"The demo hit an unexpected error while reading {name}. This is most likely a "
        "bug in bibr, not a problem with your file.",
        "Try again, or try another paper. The details are in the demo's log; please "
        f'<a href="{_ISSUES_URL}" target="_blank" rel="noopener noreferrer">report it</a> '
        "if it keeps happening.",
    )


# Plain-language notes for the warnings a reader can act on. Others stay in the
# JSON (extraction.warnings); the notes say how many.
_WARNING_NOTES: dict[str, str] = {
    "TEXT_LAYER_TABLE_REBUILT": "Tables were rebuilt from the positions of the words in the "
    "PDF, not read by an OCR model, so merged or multi-line cells may be split.",
    "TEXT_LAYER_REGION_UNREAD": "Some parts of the pages (often text inside figures) are "
    "drawn as images, so they are empty here. OCR can read them.",
    "TEXT_LAYER_PAGE_UNUSABLE": "Some pages have no usable text layer (they may be "
    "scanned), so their text is missing. OCR can read them.",
    "TEXT_LAYER_GLYPHS_UNMAPPED": "Some characters in the PDF have no text equivalent and "
    "show as �; a letter or symbol may be missing there.",
    "TEXT_LAYER_PRIVATE_USE_DROPPED": "Some symbols in the PDF are not standard characters "
    "and were left out.",
    "TEXT_LAYER_FORMULA_TEXT": "Formulas were read as plain text, so sub- and superscripts "
    "may be lost.",
    "TEXT_LAYER_OCR_LAYER_READ": "Scanned pages were read from the text the PDF already "
    "carries, which is often less accurate than OCR.",
    "TEXT_LAYER_READ_ERROR": "bibr failed to read some of the PDF text, so it is missing. "
    "This is a bug in bibr; please report it.",
    "OCR_BACKEND_UNAVAILABLE": "No OCR could run, so the PDF was read from its text layer "
    "only and any scanned pages are missing.",
    "OCR_PAGE_FAILED": "OCR failed on part of the paper, so some text is missing.",
    "OCR_REGION_FAILED": "OCR failed on part of the paper, so some text is missing.",
    "LOW_TEXT_QUALITY": "The text looks garbled in places; check the results against the paper.",
    "AUTHORS_EMPTY": "No authors were found.",
    "REF_SECTION_NOT_FOUND": "No reference list was found.",
    "REF_UNDER_EXTRACTION_SUSPECTED": "Some references may be missing from the list.",
    "PAPER_CLASSIFICATION_FAILED": "The paper type and field could not be worked out.",
    "CROSSREF_ENRICHMENT_TIMEOUT": "The Crossref lookup did not finish, so some references "
    "have no Crossref match.",
    "CROSSREF_ENRICHMENT_FAILED": "The Crossref lookup failed, so references have no "
    "Crossref match.",
    "DOCX_TABLE_DROPPED": "Some tables in the Word file could not be read and were left out.",
}

# Validation issues that mean "check this field against the paper".
_CHECK_FIELDS: dict[str, str] = {
    "VAL_TITLE_UNGROUNDED": "the title",
    "VAL_TITLE_GENERIC": "the title",
    "VAL_AUTHOR_MISSING": "the authors",
    "VAL_AUTHOR_UNGROUNDED": "the authors",
    "VAL_AUTHOR_OUTLIER": "the authors",
    "VAL_AUTHOR_FABRICATED": "the authors",
    "VAL_AUTHOR_UNCHECKED": "the authors",
    "VAL_AUTHOR_BLANK": "the authors",
    "VAL_AFFILIATION_UNGROUNDED": "the affiliations",
    "VAL_ABSTRACT_SUSPECT": "the abstract",
    "VAL_DOI_AMBIGUOUS": "the DOI",
    "VAL_REFERENCES_INCOMPLETE": "the references",
    "VAL_REF_LOW_YIELD": "the references",
    "VAL_REF_COUNT_MISMATCH": "the references",
}


def _fast_mode_note(message: str) -> str:
    """The fast-mode note, keeping the list of what was left undone."""
    match = re.search(r"Not produced:\s*(.+?)\.?\s*$", message or "")
    if match:
        left = match.group(1)
        # bibr's wording for two of the items, in a reader's words.
        left = re.sub(
            r"(\d+) ambiguous (citations?) left unlinked",
            r"links for \1 ambiguous in-text \2",
            left,
        )
        left = re.sub(
            r"statistics in (\d+) (sentences?) left to the regex pass",
            r"a closer check of the statistics in \1 \2",
            left,
        )
        return (
            "This run used local models only (fast mode, no AI language model), so these "
            f"were left out: {_md_text(left)}."
        )
    return (
        "This run used local models only (fast mode, no AI language model), so some "
        "details a language model adds are missing."
    )


def _build_notes_md(paper_json: dict) -> str:
    """Plain-language notes on one run: what it did not do, and what to double-check."""
    extraction = paper_json.get("extraction") or {}
    notes: list[str] = []
    seen: set[str] = set()
    fields: list[str] = []
    other = 0
    for warning in extraction.get("warnings") or []:
        code = (warning or {}).get("code") or ""
        if code in seen:
            continue
        if code == "LLM_SKIPPED":
            notes.append(_fast_mode_note(warning.get("message") or ""))
        elif code == "FRONT_TAGGER_FLAGGED":
            # The title page was hard to read: said once, with the fields below.
            fields.extend(f for f in ("the authors", "the affiliations") if f not in fields)
        elif code in _WARNING_NOTES:
            notes.append(_WARNING_NOTES[code])
        else:
            other += 1
            continue
        seen.add(code)
    for issue in (extraction.get("validation") or {}).get("issues") or []:
        field = _CHECK_FIELDS.get((issue or {}).get("code") or "")
        if field and field not in fields:
            fields.append(field)
    if fields:
        listed = ", ".join(fields[:-1]) + (" and " if len(fields) > 1 else "") + fields[-1]
        notes.append(f"bibr is less sure of {listed}; check them against the paper.")
    lines = [f"- {note}" for note in notes]
    if other:
        lines.append(
            f"- {_plural(other, 'more technical note')} "
            f"{'is' if other == 1 else 'are'} in the JSON download "
            "(`extraction.warnings`)."
        )
    return "\n".join(lines)


def _build_authors_data(result: dict) -> list[list]:
    """Authors table rows: name, affiliation, ORCID, email, corresponding."""
    return [
        [
            _author_name(a),
            a.get("affiliation") or "",
            a.get("orcid") or "",
            a.get("email") or "",
            "Yes" if a.get("corresponding") else "",
        ]
        for a in result.get("authors", [])
    ]


def _build_sections_data(result: dict) -> list[list]:
    """Sections table rows: heading, level, and the kind of section it was read as."""
    return [
        [
            s.get("header") or "",
            s.get("level", 0),
            "" if s.get("section_type") in (None, "", "unknown") else s["section_type"],
        ]
        for s in result.get("sections", [])
        if s.get("level", 0) > 0  # skip root
    ]


def _build_references_data(result: dict) -> list[list]:
    """References table rows: #, title, authors, year, venue, DOI (a doi.org link)."""
    return [
        [
            r.get("bib_id", ""),
            r.get("title") or "",
            r.get("authors") or "",
            str(r.get("year") or "") + (r.get("year_suffix") or ""),
            r.get("container") or r.get("publisher") or "",
            _doi_link_md(r.get("doi")),
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
    """Crossref (or other service) match rows for the references."""
    return [
        [
            m.get("bib_id", ""),
            m.get("title") or "",
            _format_bib_authors(m.get("author")),
            m.get("year") or "",
            m.get("container") or "",
            _doi_link_md(m.get("doi")),
            round(m.get("score", 0) or 0, 2),
        ]
        for m in result.get("bib_matches", [])
    ]


def _build_text_html(result: dict) -> str:
    """Render sentences grouped by section in a scrollable HTML container.

    Figure and table captions show where they are in the text, set apart;
    footnotes go after the body.
    """
    sentences = result.get("text", [])
    sections = result.get("sections", [])
    if not sentences:
        return '<div class="bibr-empty"><p>No text was found in this paper.</p></div>'

    # A caption sentence may sit in no section (the root); show it in the
    # section of its figure or table instead of before everything else.
    caption_section = {
        row.get("text_id"): row.get("section_id")
        for key in ("figure", "table")
        for row in result.get(key) or []
        if row.get("text_id") is not None
    }
    notes = {row.get("text_id") for row in result.get("footnote") or []}
    by_section: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for sent in sentences:
        text_id = sent.get("text_id")
        if text_id in notes:
            continue
        section_id = sent["section_id"]
        if text_id in caption_section and caption_section[text_id] is not None:
            section_id = caption_section[text_id]
        by_section[section_id][sent["paragraph_id"]].append(sent)

    section_map = {s.get("section_id", i): s for i, s in enumerate(sections)}

    parts = []
    # section_id is None for root / front-matter sentences (the schema
    # remaps section 0 → null). Sort those first and never compare None to int.
    for sec_id, paragraphs in sorted(by_section.items(), key=lambda kv: (kv[0] is not None, kv[0])):
        sec = section_map.get(sec_id, {})
        level = sec.get("level", 0)
        if level > 0:
            tag = f"h{min(level + 2, 6)}"
            parts.append(f"<{tag}>{_esc(sec.get('header') or '')}</{tag}>")

        for _pid, sents in sorted(paragraphs.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            caption = all(s.get("text_id") in caption_section for s in sents)
            joined = "  ".join(_esc(s["text"]) for s in sents)
            parts.append(f'<p class="bibr-caption">{joined}</p>' if caption else f"<p>{joined}</p>")

    note_rows = [sent for sent in sentences if sent.get("text_id") in notes]
    if note_rows:
        parts.append("<h4>Footnotes</h4>")
        parts.extend(f"<p>{_esc(sent['text'])}</p>" for sent in note_rows)

    return f'<div class="bibr-scroll">{chr(10).join(parts)}</div>'


def _build_tables_html(result: dict) -> str:
    """Render extracted tables as HTML, each with its caption, in one scrolling box."""
    tables = result.get("table", [])
    if not tables:
        return '<div class="bibr-empty"><p>No tables were found in this paper.</p></div>'

    parts = ['<div class="bibr-scroll">']
    for tbl in tables:
        # The printed label ("3.1", "S2"); the id is only the table's position.
        name = tbl.get("label") or tbl.get("table_id", "")
        caption = tbl.get("caption")
        if caption:
            heading = _flat(caption)
            # A caption printed with its own label ("Table 2", "Tabel 2:") keeps it.
            own_label = re.match(
                rf"\s*(table|tab\.?)\b|\s*\S+\s*{re.escape(str(name))}\b",
                str(caption),
                re.IGNORECASE,
            )
            if not own_label:
                heading = f"<strong>Table {_esc(str(name))}.</strong> {heading}"
        else:
            heading = f"<strong>Table {_esc(str(name))}</strong>"
        parts.append(f'<figure class="bibr-tableblock"><figcaption>{heading}</figcaption>')

        contents = tbl.get("contents")
        if contents and isinstance(contents, list) and len(contents) > 0:
            headers = contents[0]
            rows = contents[1:]
            html = '<div class="bibr-xscroll"><table><thead><tr>'
            html += "".join(f"<th>{_esc(str(h))}</th>" for h in headers)
            html += "</tr></thead><tbody>"
            for row in rows:
                html += "<tr>" + "".join(f"<td>{_esc(str(c))}</td>" for c in row) + "</tr>"
            html += "</tbody></table></div>"
            parts.append(html)
        else:
            parts.append('<p class="bibr-caption">The cells of this table were not read.</p>')
        parts.append("</figure>")
    parts.append("</div>")
    return "\n".join(parts)


_XREF_KINDS = {
    "bib": "reference",
    "table": "table",
    "figure": "figure",
    "foot": "footnote",
    "supplementary": "supplement",
    "equation": "equation",
    "section": "section",
}


def _build_xrefs_data(result: dict) -> list[list]:
    """In-text citation rows: as printed, what it points to, and its kind."""
    bib = {b.get("bib_id"): b for b in result.get("bib", [])}
    labels = {
        kind: {row.get(f"{kind}_id"): row.get("label") for row in result.get(kind) or []}
        for kind in ("figure", "table")
    }
    rows = []
    for x in result.get("xrefs", []):
        kind = x.get("xref_type", "")
        target = x.get("target_id")
        points_to = ""
        if kind == "bib" and target in bib:
            ref = bib[target]
            points_to = ref.get("title") or ref.get("authors") or f"Reference {target}"
        elif kind in labels and target in labels[kind]:
            points_to = f"{kind.capitalize()} {labels[kind][target] or target}"
        rows.append([x.get("contents") or "", points_to, _XREF_KINDS.get(kind, kind)])
    return rows


def _build_equations_data(result: dict) -> list[list]:
    """Statistics rows, one per sentence: its statistics as printed, then the sentence.

    Statistics reported together (one test's t, p and d) are joined with commas,
    separate groups with semicolons.
    """
    sentences = {t.get("text_id"): t.get("text") for t in result.get("text", [])}
    by_sentence: dict[Any, dict[Any, list[str]]] = {}
    for e in result.get("eq", []):
        statistic = e.get("verbatim")
        if not statistic:
            df = f"({e['df']})" if e.get("df") else ""
            statistic = f"{e.get('lhs', '')}{df} {e.get('comp', '')} {e.get('rhs', '')}".strip()
        groups = by_sentence.setdefault(e.get("text_id"), {})
        groups.setdefault(e.get("grp_id"), []).append(statistic)
    return [
        ["; ".join(", ".join(group) for group in groups.values()), sentences.get(text_id) or ""]
        for text_id, groups in by_sentence.items()
    ]


def _build_links_data(result: dict) -> list[list]:
    """Link rows: the address and the text it is attached to (shown, not linked)."""
    return [[lnk.get("href") or "", lnk.get("link_text") or ""] for lnk in result.get("urls", [])]


def _build_figures_html(result: dict, *, images_enabled: bool) -> str:
    """List the figures with their captions, and their images when extracted."""
    figures = result.get("figure", [])
    if not figures:
        return '<div class="bibr-empty"><p>No figures were found in this paper.</p></div>'

    parts = []
    if not images_enabled and not any(f.get("image") for f in figures):
        parts.append(
            '<p class="bibr-caption">This demo lists the figures and their captions; it '
            "does not extract the images.</p>"
        )
    seen: set[tuple] = set()
    unnamed = 0
    for f in figures:
        label, caption, img = f.get("label"), f.get("caption"), f.get("image")
        if not img and not label and not caption:
            unnamed += 1
            continue
        key = (label, " ".join(str(caption or "").split()))
        if not img and key in seen:
            continue  # the same figure found twice, with nothing new to show
        seen.add(key)
        name = label or f.get("figure_id", "")
        parts.append('<figure class="bibr-figure">')
        if img:
            # v12 exports the image as a data URI that names its media type.
            alt = _flat(caption or f"Figure {name}")
            parts.append(f'<img src="{_esc(img)}" alt="{alt}" />')
        text = _flat(caption) if caption else f"<strong>Figure {_esc(str(name))}</strong>"
        parts.append(f"<figcaption>{text}</figcaption></figure>")
    if unnamed:
        parts.append(
            f'<p class="bibr-caption">{_plural(unnamed, "more figure")} with no label or '
            "caption.</p>"
        )
    return f'<div class="bibr-scroll">{chr(10).join(parts)}</div>'


_STAGE_PROGRESS: dict[str, tuple[float, str]] = {
    "validate": (0.05, "Checking the file…"),
    "docx": (0.10, "Reading the Word file…"),
    "jats": (0.10, "Reading the JATS XML…"),
    "html": (0.10, "Reading the HTML or ePub…"),
    "layout": (0.15, "Finding the page layout…"),
    "native_text": (0.30, "Reading the PDF text…"),
    "ocr": (0.35, "Reading the text with OCR…"),
    "parse": (0.55, "Finding sections and references…"),
    "extract": (0.70, "Extracting the title, authors and references…"),
    "enrich": (0.85, "Looking up references on Crossref…"),
    "export": (0.95, "Writing the results…"),
}

_FIRST_RUN_NOTE = " (first paper: loading models, so this one takes longer)"
_FIRST_RUN_DOWNLOAD_NOTE = (
    " (first paper: downloading and loading models, so this one takes longer)"
)


class _GradioProgress:
    """Bridges bibr's ProgressTracker protocol to the Gradio progress bar.

    *text_layer_only*: the run reads the PDF text layer, so the OCR stage is
    labelled as reading the PDF text. *first_run*: the models load (and, when
    *downloading* is true, download) during this run, so every label says so.
    """

    def __init__(
        self,
        progress: gr.Progress,
        *,
        text_layer_only: bool = False,
        first_run: bool = False,
        downloading: bool = False,
    ):
        self._progress = progress
        self._text_layer_only = text_layer_only
        if not first_run:
            self._suffix = ""
        elif downloading:
            self._suffix = _FIRST_RUN_DOWNLOAD_NOTE
        else:
            self._suffix = _FIRST_RUN_NOTE

    def stage_start(self, name: str, detail: str = "") -> None:  # noqa: ARG002
        if name not in _STAGE_PROGRESS:
            return  # an internal step: keep the last label
        frac, desc = _STAGE_PROGRESS[name]
        if name == "ocr" and self._text_layer_only:
            desc = _STAGE_PROGRESS["native_text"][1]
        self._progress(frac, desc=desc + self._suffix)

    def stage_end(self, name: str) -> None:
        pass

    def ocr_start(self, total_regions: int) -> None:
        pass

    def ocr_region_done(self) -> None:
        pass

    def ocr_end(self) -> None:
        pass


def _effective_refs(refs: str | None, settings, mode: str) -> str:
    """The reference-parse strategy a run in ``mode`` uses, as the status shows it."""
    from bibr.run_mode import LLM_PARSE_STRATEGIES

    strategy = refs or settings.REF_PARSE_STRATEGY or "ner"
    # Fast mode makes no LLM call, so a setting that names one runs as the local parser.
    return "ner" if mode == "fast" and strategy in LLM_PARSE_STRATEGIES else strategy


_LOCAL_LLM_BACKENDS = ("local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster")


def _build_status_summary(ocr_backend: str | None, mode: str, llm_backend: str | None) -> str:
    """One plain sentence on how this demo reads papers, for under the header."""
    if mode == "fast":
        how = "**Fast mode:** local models only, no AI language model"
    elif llm_backend in _LOCAL_LLM_BACKENDS:
        how = "**Full mode:** local models, plus a language model running on this machine"
    else:
        how = (
            "**Full mode:** local models, plus a cloud language model for what they are "
            "unsure of (parts of the paper's text are sent to it)"
        )
    if ocr_backend == "text-layer":
        reading = "PDFs are read from their text layer, so scanned PDFs can't be read"
    elif ocr_backend:
        reading = "scanned pages are read with OCR"
    else:
        reading = ""
    return f"{how}. {reading[:1].upper() + reading[1:]}." if reading else f"{how}."


def _build_status_md(
    ocr_backend: str,
    llm_provider: str,
    llm_model: str,
    refs_strategy: str = "llm",
    llm_backend: str | None = None,
    mode: str = "full",
) -> str:
    """The settings this demo runs with, for the "Settings in use" panel."""
    if mode == "fast":
        mode_line = "fast (local models only, no LLM call)"
        llm = "none in fast mode"
    else:
        mode_line = "full (local models, plus an LLM for what they are unsure of)"
        backend = f" via `{llm_backend}`" if llm_backend else ""
        llm = f"`{llm_provider}/{llm_model}`{backend}"
    if ocr_backend == "text-layer":
        reading = "the PDF text layer (`text-layer`), no OCR"
    elif not ocr_backend:
        reading = "OCR with the default backend for this machine"
    else:
        reading = f"OCR with `{ocr_backend}`"
    refs = "the local model" if refs_strategy == "ner" else "the LLM"
    return "\n".join(
        [
            f"- **Mode:** {mode_line}",
            f"- **PDF text:** {reading}",
            f"- **Language model (LLM):** {llm}",
            f"- **Reference parsing:** {refs} (`{refs_strategy}`)",
        ]
    )


def _pending_demo_downloads(
    settings,
    *,
    ocr_backend: str | None,
    llm_backend: str | None,
    refs: str | None,
    mode: str,
) -> list[tuple[str, str, str]]:
    """Models not yet in the local Hugging Face cache for the demo's current mode.

    Offline cache inspection only (never fetches anything); errors degrade to
    an empty list because the notice is advisory.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from bibr.local.cli.dry_run import pending_downloads

    resolved_llm = "cloud" if mode == "fast" else (llm_backend or settings.llm.backend)
    config = SimpleNamespace(
        ref_seg=None,
        refs=refs,
        mode=mode,
        no_llm=False,
        uses_llm=mode != "fast",
        llm_backend=resolved_llm,
        ocr_backend=ocr_backend or "text-layer",
        ocr_model=None,
        ocr_profile=None,
    )
    try:
        return pending_downloads(config, [Path("paper.pdf")])  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — advisory notice must never break the demo
        return []


def _build_first_run_notice(pending: list[tuple[str, str, str]], *, warm: bool = False) -> str:
    """Upfront notice explaining why the first paper takes longer, or empty once warm."""
    if warm:
        return ""
    if pending:
        from bibr.local.cli.dry_run import _download_total
        from bibr.utils.hf_cache import hub_offline

        if hub_offline():
            count = len(pending)
            noun = "model is" if count == 1 else "models are"
            return (
                f"**First paper notice:** {count} {noun} not in the local cache and "
                "`HF_HUB_OFFLINE` is set, so nothing can be downloaded. Run once online to fetch."
            )
        total = _download_total([status for *_, status in pending])
        return (
            f"**First paper takes longer:** the first run downloads {total} from "
            "Hugging Face and loads them into memory. They are cached locally, so "
            "later papers run much faster."
        )
    return (
        "**First paper takes longer:** models load into memory on the first run; "
        "later papers reuse them."
    )


def _mode_toggle_state(
    settings,
    *,
    llm_backend: str | None,
    current_mode: str,
    locked_fast: bool,
) -> tuple[bool, str]:
    """Return ``(interactive, info_text)`` for the demo's Fast/Full mode toggle."""
    from bibr.run_mode import llm_configured

    if locked_fast:
        return (
            False,
            "Started with --fast (no LLM calls). Restart without --fast to switch modes.",
        )
    if not llm_configured(settings, llm_backend) and current_mode == "fast":
        return (
            False,
            "Full mode needs an LLM — run `bibr setup` or set an API key to enable it.",
        )
    return (
        True,
        "Fast uses local models only; Full also asks the configured LLM for uncertain fields.",
    )


async def _switch_demo_mode(
    target_mode: str,
    *,
    settings,
    pipeline_state: dict[str, Any],
    pipeline_factory,
    refs: str | None,
) -> str:
    """Switch the demo between ``fast`` and ``full`` mode, atomically replacing the pipeline."""
    from bibr.run_mode import LLM_PARSE_STRATEGIES, llm_configured

    if target_mode not in ("fast", "full"):
        raise ValueError(f"Unknown run mode {target_mode!r}. Expected 'fast' or 'full'.")
    if target_mode == "full" and pipeline_state.get("locked_fast"):
        raise ValueError("This demo was started with --fast, so Full mode is disabled.")
    if (
        target_mode == "full"
        and not llm_configured(settings, pipeline_state.get("llm_backend"))
        and pipeline_state.get("explicit_mode") != "full"
    ):
        raise ValueError(
            "Full mode needs an LLM — run `bibr setup` or set an API key to enable it."
        )

    current_mode = getattr(pipeline_state["pipeline"], "mode", None)
    if current_mode == target_mode:
        effective_ocr = pipeline_state["ocr_backend"]
        effective_backend = getattr(
            pipeline_state["pipeline"],
            "llm_backend",
            pipeline_state.get("llm_backend") or settings.llm.backend,
        )
        return _build_status_md(
            effective_ocr,
            settings.llm.provider,
            settings.llm.model,
            _effective_refs(refs, settings, target_mode),
            effective_backend,
            target_mode,
        )

    base_ocr = pipeline_state.get("base_ocr_backend", pipeline_state["ocr_backend"])
    new_llm = (
        None
        if target_mode == "fast"
        else (pipeline_state.get("llm_backend") or settings.llm.backend)
    )
    effective_ref_arg = None if target_mode == "fast" and refs in LLM_PARSE_STRATEGIES else refs
    replacement = pipeline_factory(
        memory_mode=pipeline_state["memory_mode"],
        ocr_backend=base_ocr,
        llm_backend=new_llm,
        ref_parse_strategy=effective_ref_arg,
        mode=target_mode,
    )

    old_pipeline = pipeline_state["pipeline"]
    effective_ocr = getattr(replacement, "ocr_backend", None) or base_ocr
    pipeline_state["pipeline"] = replacement
    pipeline_state["ocr_backend"] = effective_ocr
    pipeline_state["warm"] = False

    try:
        await old_pipeline.aclose()
    except Exception:  # noqa: BLE001 — replacement is live; cleanup stays best-effort
        logger.warning("Previous demo pipeline cleanup failed", exc_info=True)

    new_mode = getattr(replacement, "mode", target_mode)
    effective_backend = getattr(replacement, "llm_backend", new_llm)
    return _build_status_md(
        effective_ocr,
        settings.llm.provider,
        settings.llm.model,
        _effective_refs(refs, settings, new_mode),
        effective_backend,
        new_mode,
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
    from bibr.ocr.registry import ocr_route_chosen

    def apply() -> str | None:
        unknown = manager.apply_to_settings(name, settings)
        if unknown:
            logger.warning("Preset %s has unknown settings: %s", name, ", ".join(sorted(unknown)))
        # A preset that names no OCR route leaves it unchosen, so a fast run
        # reads the PDF text layer as it does without a preset.
        return normalize_ocr_backend(settings.ocr.backend) if ocr_route_chosen(settings) else None

    return await _swap_demo_pipeline(
        apply,
        settings=settings,
        pipeline_state=pipeline_state,
        pipeline_factory=pipeline_factory,
        refs=refs,
    )


async def _restore_demo_pipeline(
    initial: dict[str, Any],
    *,
    settings,
    pipeline_state: dict[str, Any],
    pipeline_factory,
    refs: str | None,
) -> str:
    """Put back the settings the demo started with and rebuild its pipeline on them.

    *initial* holds the settings (``vars`` of the singleton) and the OCR backend
    the demo was started with (``None`` when it chose no OCR route), both
    captured when the demo was created. "(current .env)" after a preset must
    undo the preset, not only the status line.
    """

    def apply() -> str | None:
        vars(settings).clear()
        vars(settings).update(copy.deepcopy(initial["settings"]))
        return initial["ocr_backend"]

    return await _swap_demo_pipeline(
        apply,
        settings=settings,
        pipeline_state=pipeline_state,
        pipeline_factory=pipeline_factory,
        refs=refs,
    )


async def _swap_demo_pipeline(
    apply,
    *,
    settings,
    pipeline_state: dict[str, Any],
    pipeline_factory,
    refs: str | None,
) -> str:
    """Change *settings* with *apply*, replace the demo pipeline, and close the old one.

    *apply* mutates *settings* and returns the OCR backend for the new pipeline,
    or ``None`` when no OCR route is chosen (the pipeline then picks one).
    """
    settings_snapshot = copy.deepcopy(vars(settings))
    try:
        new_ocr = apply()
        new_llm = pipeline_state["llm_backend"] or settings.llm.backend
        replacement = pipeline_factory(
            memory_mode=pipeline_state["memory_mode"],
            ocr_backend=new_ocr,
            llm_backend=new_llm,
            ref_parse_strategy=refs,
            # Only an explicit --fast / --full is carried over; otherwise the
            # replacement resolves it afresh from the preset's settings.
            **({"mode": pipeline_state["mode"]} if pipeline_state.get("mode") else {}),
        )
    except Exception:
        # apply_to_settings mutates the singleton incrementally. If validation
        # or replacement construction fails, keep the old pipeline and restore
        # its matching configuration instead of leaving a half-applied preset.
        vars(settings).clear()
        vars(settings).update(settings_snapshot)
        raise

    old_pipeline = pipeline_state["pipeline"]
    # The route the pipeline resolved: the text layer for a fast run with no OCR
    # route chosen, else the chosen or platform-default backend.
    effective_ocr = getattr(replacement, "ocr_backend", None) or new_ocr
    pipeline_state["pipeline"] = replacement
    pipeline_state["ocr_backend"] = effective_ocr

    try:
        await old_pipeline.aclose()
    except Exception:  # noqa: BLE001 — replacement is live; cleanup stays best-effort
        logger.warning("Previous demo pipeline cleanup failed", exc_info=True)

    new_mode = getattr(replacement, "mode", "full")
    effective_backend = getattr(replacement, "llm_backend", new_llm)
    return _build_status_md(
        effective_ocr,
        settings.llm.provider,
        settings.llm.model,
        _effective_refs(refs, settings, new_mode),
        effective_backend,
        new_mode,
    )


def create_local_demo(
    ocr_backend: str | None = None,
    memory_mode: str | None = None,
    llm_backend: str | None = None,
    presets_enabled: bool = False,
    refs: str | None = None,
    mode: str | None = None,
) -> gr.Blocks:
    """Create a Gradio demo that runs the bibr pipeline locally.

    ``mode`` is ``"fast"`` (no LLM call, no key), ``"full"`` (the configured LLM)
    or ``None``/``"auto"`` (``PIPELINE_MODE``, which is fast when no LLM is configured).
    """
    from bibr.config import Settings
    from bibr.local.cli import normalize_ocr_backend
    from bibr.local.pipeline import LocalPipeline
    from bibr.run_mode import LLM_PARSE_STRATEGIES, resolve_run_mode

    # Read first, so a bad DEMO_CACHE_TTL_SECONDS or DEMO_MAX_FILE_SIZE_MB
    # fails before models load.
    cache_lifetime = _cache_lifetime()
    _max_file_size_mb()
    # No --ocr stays unset: the pipeline then reads the PDF text layer in fast
    # mode unless OCR_BACKEND, OCR_MODEL or OCR_PROFILE names a route, and takes
    # the platform default OCR backend otherwise.
    ocr_backend = normalize_ocr_backend(ocr_backend) if ocr_backend else None

    if cache_lifetime is None and "ocr" not in Settings.cache.model_fields_set:
        # Demo users repeatedly re-run the same PDF while poking at the UI;
        # skipping OCR inference on repeat runs is a better default here than
        # in the batch pipeline, where silently caching results is unwanted.
        # Only when the demo keeps its files: the cache keeps each paper's OCR
        # text with no expiry, so a demo that deletes uploads leaves it off.
        Settings.cache.ocr = True
        logger.info("Local demo — enabling OCR disk cache (CACHE_OCR) by default")

    # Like `bibr chew`: --llm or --refs llm under auto ask for an LLM (full mode),
    # and in an explicit fast mode they are refused.
    llm_flags = [f"--llm {llm_backend}"] if llm_backend else []
    if refs in LLM_PARSE_STRATEGIES:
        llm_flags.append(f"--refs {refs}")
    run_mode = resolve_run_mode(
        Settings, explicit=mode, llm_backend=llm_backend, llm_flags=llm_flags
    )

    # The reference-parse strategy rides the pipeline's per-run config
    # (mirrors `bibr chew --refs`) instead of mutating the process-global
    # Settings; the status line shows the effective value.
    effective_refs = _effective_refs(refs, Settings, run_mode)

    # Figure base64 images are only emitted when FIGURE_IMAGES is on; without
    # them a "no images" download would be identical to the full one, so we
    # only offer the second button when figures are actually present.
    figures_enabled = bool(Settings.FIGURE_IMAGES)

    pipeline = LocalPipeline(
        memory_mode=memory_mode,
        ocr_backend=ocr_backend,
        llm_backend=llm_backend,
        ref_parse_strategy=refs,
        mode=run_mode,
    )
    # What the pipeline resolved, for the log and the status line.
    effective_ocr = getattr(pipeline, "ocr_backend", None) or ocr_backend
    locked_fast = mode == "fast"
    pipeline_state: dict[str, Any] = {
        "pipeline": pipeline,
        "ocr_backend": effective_ocr,
        "base_ocr_backend": ocr_backend,
        "memory_mode": memory_mode,
        "llm_backend": llm_backend,
        # What the user asked for, not what it resolved to: a preset that
        # configures an LLM turns an automatic demo into a full one.
        "mode": None if mode == "auto" else mode,
        "explicit_mode": None if mode == "auto" else mode,
        "locked_fast": locked_fast,
    }
    pipeline_lock = asyncio.Lock()
    logger.info(
        "Local pipeline ready (mode=%s, ocr=%s, llm=%s, memory=%s)",
        run_mode,
        effective_ocr,
        "none" if run_mode == "fast" else llm_backend or Settings.llm.backend,
        getattr(pipeline, "memory_mode", memory_mode),
    )
    status_md = _build_status_md(
        effective_ocr,
        Settings.llm.provider,
        Settings.llm.model,
        effective_refs,
        getattr(pipeline_state["pipeline"], "llm_backend", llm_backend or Settings.llm.backend),
        run_mode,
    )

    effective_llm_backend = getattr(
        pipeline_state["pipeline"], "llm_backend", llm_backend or Settings.llm.backend
    )
    status_summary = _build_status_summary(effective_ocr, run_mode, effective_llm_backend)
    pipeline_state["loop"] = None  # the event loop the runs use, once one has run
    pipeline_state["warm"] = False  # models load during the first run
    limit_mb = _max_file_size_mb()

    def _current_mode() -> str:
        return getattr(pipeline_state["pipeline"], "mode", run_mode)

    def _current_pending() -> list[tuple[str, str, str]]:
        current = pipeline_state["pipeline"]
        cur_mode = getattr(current, "mode", run_mode)
        cur_llm = getattr(current, "llm_backend", llm_backend or Settings.llm.backend)
        return _pending_demo_downloads(
            Settings,
            ocr_backend=pipeline_state["ocr_backend"],
            llm_backend=cur_llm,
            refs=refs,
            mode=cur_mode,
        )

    def _notice_text() -> str:
        if pipeline_state["warm"]:
            return ""
        return _build_first_run_notice(_current_pending(), warm=False)

    def _notice_update():
        text = _notice_text()
        return gr.Markdown(value=text, visible=bool(text))

    initial_notice = _notice_text()
    can_toggle, mode_info = _mode_toggle_state(
        Settings,
        llm_backend=llm_backend,
        current_mode=run_mode,
        locked_fast=locked_fast,
    )

    with gr.Blocks(title="bibr demo", delete_cache=cache_lifetime) as demo:
        gr.Markdown(_HEADER_MD, elem_classes="bibr-header")
        status_line = gr.Markdown(status_summary, elem_classes="bibr-status")
        first_run_notice = gr.Markdown(
            initial_notice,
            visible=bool(initial_notice),
            elem_classes="bibr-notice",
        )
        mode_toggle = gr.Radio(
            choices=[
                ("Fast (local models only)", "fast"),
                ("Full (with LLM)", "full"),
            ],
            value=run_mode,
            label="Extraction mode",
            info=mode_info,
            interactive=can_toggle,
            elem_classes="bibr-mode",
        )

        def _summary_now() -> str:
            current = pipeline_state["pipeline"]
            return _build_status_summary(
                pipeline_state["ocr_backend"],
                getattr(current, "mode", run_mode),
                getattr(current, "llm_backend", None),
            )

        def _mode_toggle_update():
            cur_mode = _current_mode()
            interactive, info = _mode_toggle_state(
                Settings,
                llm_backend=pipeline_state["llm_backend"],
                current_mode=cur_mode,
                locked_fast=pipeline_state["locked_fast"],
            )
            return gr.Radio(value=cur_mode, interactive=interactive, info=info)

        with gr.Accordion("Settings in use", open=False):
            status_display = gr.Markdown(status_md)
            gr.Markdown(
                "These are set when the demo starts; `bibr demo --help` lists the options.",
                elem_classes="bibr-hint",
            )

            if presets_enabled:
                from bibr.presets import PresetManager, effective_env_file

                manager = PresetManager()
                preset_names = manager.list_presets()

                if not preset_names:
                    logger.warning("--presets: no saved presets to switch between")
                    gr.Markdown(
                        "No presets are saved, so there is nothing to switch to. Save one "
                        "with `bibr preset save NAME` and restart the demo.",
                        elem_classes="bibr-hint",
                    )
                else:
                    env_file = effective_env_file()
                    active = manager.get_active(env_file) if env_file is not None else None
                    # What "(current .env)" goes back to after a preset.
                    initial = {
                        "settings": copy.deepcopy(vars(Settings)),
                        "ocr_backend": ocr_backend,
                    }
                    preset_dropdown = gr.Dropdown(
                        choices=["(current .env)"] + preset_names,
                        value=active if active in preset_names else "(current .env)",
                        label="Configuration preset",
                        interactive=True,
                    )

                    async def _switch_preset(name):
                        try:
                            async with pipeline_lock:
                                if name == "(current .env)" or not name:
                                    if pipeline_state.get("preset") is None:
                                        return status_md
                                    status = await _restore_demo_pipeline(
                                        initial,
                                        settings=Settings,
                                        pipeline_state=pipeline_state,
                                        pipeline_factory=LocalPipeline,
                                        refs=refs,
                                    )
                                    pipeline_state["preset"] = None
                                    pipeline_state["base_ocr_backend"] = initial["ocr_backend"]
                                    pipeline_state["warm"] = False
                                    return status
                                status = await _replace_demo_pipeline_with_preset(
                                    name,
                                    manager=manager,
                                    settings=Settings,
                                    pipeline_state=pipeline_state,
                                    pipeline_factory=LocalPipeline,
                                    normalize_ocr_backend=normalize_ocr_backend,
                                    refs=refs,
                                )
                                pipeline_state["preset"] = name
                                from bibr.ocr.registry import ocr_route_chosen

                                pipeline_state["base_ocr_backend"] = (
                                    normalize_ocr_backend(Settings.ocr.backend)
                                    if ocr_route_chosen(Settings)
                                    else None
                                )
                                pipeline_state["warm"] = False
                                return status
                        except Exception as e:
                            raise gr.Error(f"Failed to apply preset: {e}") from e

                    # The details, then the summary, mode toggle, and cold-start notice.
                    preset_dropdown.change(
                        fn=_switch_preset,
                        inputs=[preset_dropdown],
                        outputs=[status_display],
                    ).then(
                        fn=lambda: (_summary_now(), _mode_toggle_update(), _notice_update()),
                        outputs=[status_line, mode_toggle, first_run_notice],
                        show_progress="hidden",
                    )

        async def _on_mode_change(selected_mode):
            try:
                async with pipeline_lock:
                    status = await _switch_demo_mode(
                        selected_mode,
                        settings=Settings,
                        pipeline_state=pipeline_state,
                        pipeline_factory=LocalPipeline,
                        refs=refs,
                    )
            except Exception as e:
                raise gr.Error(f"Failed to switch mode: {e}") from e
            return _summary_now(), status, _notice_update(), _mode_toggle_update()

        mode_toggle.input(
            fn=_on_mode_change,
            inputs=[mode_toggle],
            outputs=[status_line, status_display, first_run_notice, mode_toggle],
            show_progress="hidden",
            api_name=False,
        )

        with gr.Row():
            with gr.Column(scale=2, min_width=280):
                file_input = gr.File(
                    label=f"Your paper: {_FORMATS_TEXT}, up to {limit_mb} MB",
                    file_count="single",
                    file_types=_ALLOWED_EXTENSIONS,
                )
            with gr.Column(scale=1, min_width=220):
                run_btn = gr.Button("Extract", variant="primary", size="lg")
                sample_btn = gr.Button("Try a sample paper", variant="secondary")
                gr.Markdown(
                    "No paper at hand? The sample is a short, made-up paper that comes with bibr.",
                    elem_classes="bibr-hint",
                )

        result_card = gr.HTML(_PLACEHOLDER_HTML, elem_classes="bibr-result")
        with gr.Row(visible=False, elem_classes="bibr-downloads") as downloads_row:
            dl_json = gr.DownloadButton("Download JSON", size="md", variant="primary")
            dl_json_noimg = (
                gr.DownloadButton("JSON without images", size="md") if figures_enabled else None
            )
            dl_bib = gr.DownloadButton("BibTeX", size="md")
            dl_ris = gr.DownloadButton("RIS", size="md")
            dl_data = gr.DownloadButton("Data tables (Parquet, .zip)", size="md")
        with gr.Accordion("Notes on this extraction", open=True, visible=False) as notes_box:
            notes_md = gr.Markdown()

        with gr.Tabs(visible=False) as result_tabs:
            with gr.Tab("References"):
                refs_empty = gr.Markdown(
                    "No references were found in this paper.",
                    visible=False,
                    elem_classes="bibr-empty",
                )
                refs_table = gr.Dataframe(
                    headers=["#", "Title", "Authors", "Year", "Venue", "DOI"],
                    datatype=["number", "str", "str", "str", "str", "markdown"],
                    column_widths=["48px", "380px", "250px", "64px", "180px", "200px"],
                    wrap=True,
                    max_height=600,
                    show_search="search",
                    interactive=False,
                    show_label=False,
                    elem_classes="bibr-table",
                )
                with gr.Accordion("Crossref matches", open=False, visible=False) as matches_box:
                    matches_table = gr.Dataframe(
                        headers=["#", "Title", "Authors", "Year", "Venue", "DOI", "Score"],
                        datatype=["number", "str", "str", "str", "str", "markdown", "number"],
                        column_widths=["48px", "330px", "220px", "64px", "160px", "200px", "72px"],
                        wrap=True,
                        max_height=500,
                        interactive=False,
                        show_label=False,
                        elem_classes="bibr-table",
                    )
            with gr.Tab("Full text"):
                text_html = gr.HTML()
            with gr.Tab("Tables & figures"):
                with gr.Row():
                    dl_tables_csv = gr.DownloadButton(
                        "Download these tables as CSV (.zip)",
                        size="sm",
                        visible=False,
                        scale=0,
                        min_width=300,
                    )
                tables_html = gr.HTML()
                gr.Markdown("### Figures")
                figures_html = gr.HTML()
            with gr.Tab("Authors & sections"):
                gr.Markdown("### Authors")
                authors_empty = gr.Markdown(
                    "No authors were found.", visible=False, elem_classes="bibr-empty"
                )
                authors_table = gr.Dataframe(
                    headers=["Name", "Affiliation", "ORCID", "Email", "Corresponding author"],
                    datatype=["str"] * 5,
                    column_widths=["200px", "400px", "170px", "220px", "130px"],
                    wrap=True,
                    interactive=False,
                    show_label=False,
                    elem_classes="bibr-table",
                )
                gr.Markdown("### Sections")
                sections_empty = gr.Markdown(
                    "No section headings were found.", visible=False, elem_classes="bibr-empty"
                )
                sections_table = gr.Dataframe(
                    headers=["Heading", "Level", "Read as"],
                    datatype=["str", "number", "str"],
                    column_widths=["760px", "80px", "240px"],
                    wrap=True,
                    interactive=False,
                    show_label=False,
                    elem_classes="bibr-table",
                )
            with gr.Tab("More"):
                gr.Markdown(
                    "### In-text citations\nWhere the text cites a reference, figure or table."
                )
                xrefs_empty = gr.Markdown(
                    "No in-text citations were found.", visible=False, elem_classes="bibr-empty"
                )
                xrefs_table = gr.Dataframe(
                    headers=["In the text", "Refers to", "Kind"],
                    datatype=["str", "str", "str"],
                    column_widths=["320px", "640px", "140px"],
                    wrap=True,
                    interactive=False,
                    show_label=False,
                    elem_classes="bibr-table",
                )
                gr.Markdown(
                    "### Statistics\nTest results reported in the text, such as t(28) = 2.10."
                )
                eq_empty = gr.Markdown(
                    "No statistics were found.", visible=False, elem_classes="bibr-empty"
                )
                equations_table = gr.Dataframe(
                    headers=["Statistics", "Sentence"],
                    datatype=["str", "str"],
                    column_widths=["320px", "780px"],
                    wrap=True,
                    interactive=False,
                    show_label=False,
                    elem_classes="bibr-table",
                )
                gr.Markdown("### Links")
                links_empty = gr.Markdown(
                    "No links were found.", visible=False, elem_classes="bibr-empty"
                )
                links_table = gr.Dataframe(
                    headers=["Address", "Link text"],
                    datatype=["str", "str"],
                    column_widths=["640px", "460px"],
                    wrap=True,
                    interactive=False,
                    show_label=False,
                    elem_classes="bibr-table",
                )
                with gr.Accordion("The full bibr JSON", open=False):
                    json_view = gr.JSON(show_label=False)

        downloads = {
            "json": dl_json,
            "json_no_images": dl_json_noimg,
            "bibtex": dl_bib,
            "ris": dl_ris,
            "data_tables": dl_data,
            "tables_csv": dl_tables_csv,
        }
        download_buttons = [b for b in downloads.values() if b is not None]
        result_outputs = [
            first_run_notice,
            result_card,
            downloads_row,
            *download_buttons,
            notes_box,
            notes_md,
            result_tabs,
            refs_empty,
            refs_table,
            matches_box,
            matches_table,
            text_html,
            tables_html,
            figures_html,
            authors_empty,
            authors_table,
            sections_empty,
            sections_table,
            xrefs_empty,
            xrefs_table,
            eq_empty,
            equations_table,
            links_empty,
            links_table,
            json_view,
        ]

        def _cleared(card_html: str) -> dict:
            """Hide the last paper's results: they must not sit beside another file."""
            return {
                first_run_notice: _notice_update(),
                result_card: card_html,
                downloads_row: gr.Row(visible=False),
                notes_box: gr.Accordion(visible=False),
                result_tabs: gr.Tabs(visible=False),
                **{
                    button: gr.DownloadButton(value=None, visible=False)
                    for button in download_buttons
                },
            }

        def _table(empty, table, rows) -> dict:
            return {
                empty: gr.Markdown(visible=not rows),
                table: gr.Dataframe(rows, visible=bool(rows)),
            }

        def _results(paper_json: dict, paths: dict) -> dict:
            result = _parse_json_response(paper_json)
            notes = _build_notes_md(paper_json)
            matches = _build_bib_matches_data(result)
            updates = {
                first_run_notice: _notice_update(),
                result_card: _build_paper_card_html(result),
                downloads_row: gr.Row(visible=True),
                notes_box: gr.Accordion(visible=bool(notes), open=True),
                notes_md: notes,
                result_tabs: gr.Tabs(visible=True),
                matches_box: gr.Accordion(
                    label=f"Crossref matches ({len(matches)})", visible=bool(matches)
                ),
                matches_table: matches,
                text_html: _build_text_html(result),
                tables_html: _build_tables_html(result),
                figures_html: _build_figures_html(result, images_enabled=figures_enabled),
                # The tree viewer shows the image-free copy (base64 blobs are
                # unreadable and heavy in a browser); the JSON download keeps them.
                json_view: _strip_figure_images(paper_json),
            }
            updates.update(_table(refs_empty, refs_table, _build_references_data(result)))
            updates.update(_table(authors_empty, authors_table, _build_authors_data(result)))
            updates.update(_table(sections_empty, sections_table, _build_sections_data(result)))
            updates.update(_table(xrefs_empty, xrefs_table, _build_xrefs_data(result)))
            updates.update(_table(eq_empty, equations_table, _build_equations_data(result)))
            updates.update(_table(links_empty, links_table, _build_links_data(result)))
            for kind, button in downloads.items():
                if button is not None:
                    path = paths.get(kind)
                    updates[button] = gr.DownloadButton(value=path, visible=bool(path))
            return updates

        async def _process_file(file_path, progress=gr.Progress()):
            if file_path is None:
                return _cleared(
                    _error_html(
                        _ShownError(
                            "Choose a paper first",
                            "Drop a file in the box above, or click <strong>Try a sample "
                            "paper</strong>.",
                        )
                    )
                )
            file_name = os.path.basename(file_path)
            pipeline_state["loop"] = asyncio.get_running_loop()
            try:
                _check_file_size(file_path)
                progress(0.0, desc="Starting…")
                first_run = not pipeline_state["warm"]
                downloading = bool(_current_pending()) if first_run else False
                tracker = _GradioProgress(
                    progress,
                    text_layer_only=pipeline_state["ocr_backend"] == "text-layer",
                    first_run=first_run,
                    downloading=downloading,
                )
                async with pipeline_lock:
                    try:
                        paper_json = await pipeline_state["pipeline"].process_file(
                            file_path, progress=tracker
                        )
                    except Exception as exc:
                        # A file refused before any model loaded leaves them unloaded.
                        if getattr(exc, "failed_stage", None) != "validate":
                            pipeline_state["warm"] = True
                        raise
                    pipeline_state["warm"] = True
                progress(0.97, desc="Preparing the downloads…")
                stripped_json = _strip_figure_images(paper_json)
                # Serializing an export takes tens of ms (more with figure images):
                # off the loop that serves every other demo user.
                paths = await asyncio.to_thread(
                    _write_downloads, paper_json, stripped_json, figures_enabled
                )
                return _results(paper_json, paths)
            except _ShownError as shown:
                return _cleared(_error_html(shown))
            except Exception as exc:  # noqa: BLE001 — every failure is explained on the page
                from bibr.exceptions import BibrError

                if isinstance(exc, BibrError):
                    # The stage that failed has usually logged it already.
                    logger.info("%s: %s", file_name, exc)
                else:
                    logger.exception("The demo failed on %s", file_name)
                return _cleared(
                    _error_html(
                        _explain_failure(
                            exc,
                            file_name=file_name,
                            text_layer_only=pipeline_state["ocr_backend"] == "text-layer",
                        )
                    )
                )

        async def _aclose_pipeline() -> None:
            await pipeline_state["pipeline"].aclose()

        def _close_pipeline(timeout: float = 120.0) -> None:
            """Close the pipeline on the event loop its runs used.

            Its models and batchers belong to that loop; closing them from a
            new one fails. Before any run, a new loop does.
            """
            loop = pipeline_state.get("loop")
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(_aclose_pipeline(), loop).result(timeout)
            else:
                asyncio.run(_aclose_pipeline())

        # For whoever stops the demo (``bibr demo`` once Gradio returns): the
        # managed OCR/LLM servers run in their own session and would otherwise
        # outlive the process.
        demo.aclose_pipeline = _aclose_pipeline
        demo.close_pipeline = _close_pipeline
        # What ``bibr demo`` prints when it starts.
        demo.status_summary = status_summary
        demo.first_run_notice = initial_notice

        run_btn.click(
            fn=_process_file,
            inputs=[file_input],
            outputs=result_outputs,
            show_progress_on=[result_card],
            api_name=False,
        )
        sample_btn.click(
            fn=_copy_sample_paper,
            outputs=[file_input],
            show_progress="hidden",
            api_name=False,
        ).then(
            fn=_process_file,
            inputs=[file_input],
            outputs=result_outputs,
            show_progress_on=[result_card],
            api_name=False,
        )
        # A new or removed file hides the last paper's results.
        file_input.upload(
            fn=lambda: _cleared(_READY_HTML),
            outputs=result_outputs,
            show_progress="hidden",
            api_name=False,
        )
        file_input.clear(
            fn=lambda: _cleared(_PLACEHOLDER_HTML),
            outputs=result_outputs,
            show_progress="hidden",
            api_name=False,
        )

    return demo
