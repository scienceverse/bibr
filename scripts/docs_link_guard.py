"""mkdocs hook: fail strict builds when a page links to an excluded doc.

mkdocs' own link validation deliberately caps the "link points at an excluded
page" warning at INFO, regardless of `validation.links.not_found` in
mkdocs.yml (see `mkdocs.structure.pages._RelativePathTreeprocessor.convert_url`:
`warning_level = min(logging.INFO, self.config.validation.links.not_found)`
whenever the *target* file is excluded). `mkdocs build --strict` only counts
WARNING-and-above records, so that class of bug never trips `--strict` no
matter how the `validation:` block is configured — this is exactly how
`docs/tester-guide.md` 404'd in production despite the site building "clean"
(see docs/index.md and docs/getting-started/quickstart.md, which both linked
to it while it sat in `exclude_docs`).

This hook re-does that specific check at WARNING level instead, so it lands
in the same `mkdocs` logger hierarchy that strict mode's CountHandler watches.
It only looks at plain inline Markdown links (`[text](path)`); it isn't a
full Markdown link parser, but it's enough to catch the case that matters:
someone adds a page to `exclude_docs` (or `nav: exclude_docs`-equivalent)
while docs still link to it.
"""

from __future__ import annotations

import posixpath
import re

from mkdocs.plugins import get_plugin_logger

log = get_plugin_logger(__name__)

# Matches `[text](target)` and `[text](target "title")`, capturing `target`.
_INLINE_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?([^)\s>]+)[^)]*\)")


def on_files(files, config):
    excluded_uris = {f.src_uri for f in files if f.inclusion.is_excluded()}
    if not excluded_uris:
        return files

    for file in files.documentation_pages():
        if file.inclusion.is_excluded():
            continue  # mkdocs already treats these as fully out-of-band

        try:
            text = file.content_string
        except OSError:
            continue

        src_dir = posixpath.dirname(file.src_uri)
        for raw_target in _INLINE_LINK_RE.findall(text):
            target = raw_target.split("#", 1)[0].split("?", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "/")):
                continue
            resolved = posixpath.normpath(posixpath.join(src_dir, target))
            if resolved in excluded_uris:
                log.warning(
                    f"Doc file '{file.src_uri}' links to '{raw_target}', which "
                    f"resolves to '{resolved}' — a page excluded via `exclude_docs`. "
                    "Stop excluding it, or remove/redirect the link."
                )

    return files
