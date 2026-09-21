"""Keep public documentation limited to reviewed pages and assets."""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlsplit

# Adding a downloadable document or dataset requires an explicit review here.
PUBLIC_ASSETS = frozenset(
    {
        "robots.txt",
        "assets/logo.svg",
        "assets/readme-banner.png",
        "reference/paper.schema.json",
        "schema/bibr-export-v11.schema.json",
        "schema/bibr-export-v11-reader.schema.json",
        "schema/bibr-export-v10.schema.json",
    }
)
PRIVATE_CONTENT_RULES = {
    "private repository reference": re.compile(r"\bbibr-(?:validation|training)\b", re.I),
    "internal corpus identifier": re.compile(
        r"\b(?:val\d+|psych\d+|dgmdpi\w*\d\w*|gold_opus\w*|bench_data|corpus_store)\b", re.I
    ),
    "local operator path": re.compile(r"/(?:home|Users)/[^/\s<>]+/"),
    "engineering record path": re.compile(r"\b(?:docs/)?(?:reports|superpowers)/"),
    "evaluation payload path": re.compile(r"\b(?:benchmarks|evaluation)/(?:results|data|gold)/"),
    "source corpus record": re.compile(r"\bW\d{6,}\b"),
}


def nav_paths(nav: list | dict | str) -> set[str]:
    """Collect local page sources from MkDocs' nested navigation."""
    if isinstance(nav, str):
        return {nav.split("#", 1)[0]} if not urlsplit(nav).scheme else set()
    values = nav.values() if isinstance(nav, dict) else nav
    return {path for value in values for path in nav_paths(value)}


def private_content_findings(content: str) -> list[str]:
    """Detect internal references in Markdown, rendered HTML, and JSON assets."""
    # Syntax highlighting can split a path or identifier across HTML spans.
    normalized = html.unescape(re.sub(r"<[^>]*>", "", content))
    return [label for label, pattern in PRIVATE_CONTENT_RULES.items() if pattern.search(normalized)]


def on_pre_build(config):
    """Apply the same boundary to public repository landing documents."""
    root = Path(config["docs_dir"]).resolve().parent
    findings = []
    for name in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LIMITATIONS.md", "LLM_POLICY.md"):
        path = root / name
        for label in private_content_findings(path.read_text(encoding="utf-8")):
            findings.append(f"{name}: {label}")
    if findings:
        raise ValueError(
            "Internal material in public repository documents:\n" + "\n".join(findings)
        )


def on_files(files, config):
    """Reject unreviewed source pages and payloads before MkDocs copies them."""
    pages = nav_paths(config["nav"])
    docs_dir = Path(config["docs_dir"]).resolve()
    for file in files:
        if file.inclusion.is_excluded():
            continue
        if file.is_documentation_page():
            if file.src_uri not in pages:
                raise ValueError(f"Public documentation page is absent from nav: {file.src_uri}")
        elif (file.src_dir is None or Path(file.src_dir).resolve() == docs_dir) and (
            file.src_uri not in PUBLIC_ASSETS
        ):
            raise ValueError(f"Unreviewed public documentation asset: {file.src_uri}")
    return files


def on_post_build(config):
    """Check generated references and search data as well as handwritten pages."""
    site_dir = Path(config["site_dir"])
    findings = []
    for path in sorted(site_dir.rglob("*")):
        if path.suffix not in {".html", ".json", ".xml", ".txt"} or not path.is_file():
            continue
        for label in private_content_findings(path.read_text(encoding="utf-8")):
            findings.append(f"{path.relative_to(site_dir)}: {label}")
    if findings:
        raise ValueError("Internal material in public documentation:\n" + "\n".join(findings))
