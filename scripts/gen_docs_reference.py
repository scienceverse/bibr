"""mkdocs-gen-files entry: write generated reference pages at build time."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mkdocs_gen_files  # noqa: E402
from docs_ref_core import render_cli_md, render_schema_md, render_settings_md  # noqa: E402

from bibr.export.schema_artifact import build_export_schema  # noqa: E402

for path, render in [
    ("reference/settings.md", render_settings_md),
    ("reference/cli.md", render_cli_md),
    ("reference/schema.md", render_schema_md),
]:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(render())
    # These pages have no source file under docs/ — point Material's edit
    # button at the generator script instead (path is relative to docs_dir,
    # so "../" climbs out to the repo root: edit/main/scripts/gen_docs_reference.py).
    mkdocs_gen_files.set_edit_path(path, "../scripts/gen_docs_reference.py")

with mkdocs_gen_files.open("reference/paper.schema.json", "w") as f:
    json.dump(build_export_schema(), f, indent=2)
    f.write("\n")

# Keep repository and site policy pages identical, adjusting only Markdown
# presentation and links for their new location at the docs root.
root_pages = {"LIMITATIONS.md": "limitations.md", "LLM_POLICY.md": "llm-use.md"}
repo_root = Path(__file__).resolve().parents[1]
for source, target in root_pages.items():
    content = (repo_root / source).read_text(encoding="utf-8")
    content = content.replace(
        "> [!NOTE]\n> **Work in progress:** This section needs further work and will be revised.",
        '!!! note "Work in progress"\n    This section needs further work and will be revised.',
    )

    def site_link(match: re.Match[str]) -> str:
        url = match.group(1)
        if url.startswith("docs/"):
            url = url.removeprefix("docs/")
        elif url in root_pages:
            url = root_pages[url]
        return f"]({url})"

    content = re.sub(r"\]\(([^)]+)\)", site_link, content)
    with mkdocs_gen_files.open(target, "w") as f:
        f.write(content)
    mkdocs_gen_files.set_edit_path(target, f"../{source}")
