"""Links a paper prints inside its tables, and links repaired from the PDF's own targets.

The ``url`` list is read from sentences, and a table's cells are no sentence:
a data-availability table, a funder table or a table of sources printed its
links where nothing looked. :func:`add_table_links` reads them from two places:

* the text of each cell, with the rules sentences are read with
  (:func:`bibr.utils.text.find_text_links`);
* for a PDF, the link annotations that lie on the table. A cell wraps a long
  link over several lines and the table reader keeps the lines apart or drops
  an underscore, so the cell text holds a piece of the link or nothing
  linkable at all; the annotation names the whole target.

A link of a table has no sentence: its ``text_id`` is None and ``table_id``
names the table. Each target is listed once per table.

:func:`repair_links_from_targets` uses the same annotations on the links read
from sentences. A link printed over a line break can come out with a hyphen
that is not in it ("readthe-docs.io"), without one that is, or cut at the break
("…/cata" for "…/catalog/50"). When exactly one of the PDF's targets is the
printed link but for hyphens, underscores and spaces, or continues it with
what the sentence prints next, the link takes the target's spelling and keeps
the printed form as its ``link_text``.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from bibr.paper_contents import PaperURLLink
from bibr.utils.text import find_text_links

if TYPE_CHECKING:
    from bibr.paper_contents import PaperContents, PaperTable

_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)
# Characters a table reader leaves in a cell for a line break it could not
# read: Unicode non-characters and the soft hyphen.
_CELL_NOISE = dict.fromkeys(map(ord, "￾￿­"))
# A DOI prefix whose dot the table reader set apart ("10 . 35802/227093").
_SPACED_DOI_PREFIX_RE = re.compile(r"(?<![\w.])10 \. (?=\d{4,9}/)")
# What a line break adds to or takes from a printed link.
_BREAK_CHARS_RE = re.compile(r"[\s\-_­‐‑]+")
# A link shorter than this, squeezed, is too short to be told apart by spelling.
_MIN_SQUEEZED = 12


def _target(uri: object) -> str | None:
    """An http(s) link annotation's target, or None for any other annotation."""
    if not isinstance(uri, str):
        return None
    text = html.unescape(uri.strip())
    if not _HTTP_RE.match(text) or any(ch.isspace() for ch in text):
        return None
    return text


def _key(url: str) -> str:
    """*url* as compared: no scheme, ``www.`` or trailing "/", percent-decoded, case folded."""
    text = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", "".join(url.split()), flags=re.IGNORECASE)
    if text[:4].lower() == "www.":
        text = text[4:]
    return unquote(text).rstrip("/.").casefold()


def _squeeze(text: str) -> str:
    return _BREAK_CHARS_RE.sub("", text)


def _cell_texts(table: PaperTable) -> Iterator[str]:
    rows = [row for part in table.parts for row in part.contents] or table.contents
    for row in rows:
        for cell in row:
            text = str(cell).translate(_CELL_NOISE)
            if "/" in text or "." in text:
                yield _SPACED_DOI_PREFIX_RE.sub("10.", text)


def _boxes(table: PaperTable) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes = [(p.page_number, p.bbox) for p in table.parts if p.page_number and p.bbox]
    if not boxes:
        boxes = [(p.page_no, p.bbox) for p in table.provenance if p.bbox]
    return boxes


def _targets_on(table: PaperTable, uri_links: Iterable[Mapping[str, Any]]) -> list[str]:
    """Targets of the link annotations whose centre lies on *table*, in reading order."""
    boxes = _boxes(table)
    if not boxes:
        return []
    found: list[tuple[int, float, float, str]] = []
    for link in uri_links:
        target = _target(link.get("uri"))
        bbox = link.get("bbox")
        if target is None or not bbox or len(bbox) != 4:
            continue
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        for page, box in boxes:
            if link.get("page") == page and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                found.append((page, y, x, target))
                break
    return [target for _page, _y, _x, target in sorted(found)]


def add_table_links(contents: PaperContents, *, bare: bool = True) -> None:
    """Add the links printed in the cells of *contents*' tables to its links.

    *bare* is passed on to :func:`find_text_links`: False where only a URL
    printed in full is read as a link from text (HTML, DOCX and JATS input).
    Calling it again adds nothing.
    """
    if not contents.tables:
        return
    seen = {(link.table_id, _key(link.url)) for link in contents.links if link.table_id is not None}
    uri_links = contents.pdf_uri_links or []
    added = False
    for table in contents.tables:
        targets = _targets_on(table, uri_links) if uri_links else []
        squeezed_targets = [_squeeze(_key(target)) for target in targets]
        links = [PaperURLLink(url=target, section_id=table.section_id) for target in targets]
        for text in _cell_texts(table):
            for found in find_text_links(text, bare=bare):
                squeezed = _squeeze(_key(found.url))
                # A target on the table that is this link, or continues it,
                # is the link as the paper means it.
                if any(target.startswith(squeezed) for target in squeezed_targets):
                    continue
                links.append(
                    PaperURLLink(
                        url=found.url,
                        section_id=table.section_id,
                        link_text=found.link_text,
                        completed=found.link_text is not None,
                    )
                )
        for link in links:
            key = (table.table_id, _key(link.url))
            if key in seen:
                continue
            seen.add(key)
            link.table_id = table.table_id
            contents.links.append(link)
            added = True
    if added:
        contents.invalidate_text_caches()


def repair_links_from_targets(contents: PaperContents) -> None:
    """Respell sentence links a line break damaged, from the PDF's link targets."""
    by_squeezed: dict[str, set[str]] = defaultdict(set)
    by_key: dict[str, str] = {}
    for link in contents.pdf_uri_links or []:
        target = _target(link.get("uri"))
        if target is None:
            continue
        key = _key(target)
        by_key.setdefault(key, target)
        by_squeezed[_squeeze(key)].add(key)
    if not by_key:
        return
    sentences: dict[int, str] | None = None
    changed = False
    for link in contents.links:
        if link.table_id is not None or link.text_id is None:
            continue
        key = _key(link.url)
        if key in by_key:
            continue
        squeezed = _squeeze(key)
        if len(squeezed) < _MIN_SQUEEZED:
            continue
        same = by_squeezed.get(squeezed)
        if same is not None:
            if len(same) != 1:
                continue
            repaired = by_key[next(iter(same))]
        else:
            longer = {
                target_key
                for target_squeezed, keys in by_squeezed.items()
                if target_squeezed.startswith(squeezed)
                for target_key in keys
            }
            if len(longer) != 1:
                continue
            if sentences is None:
                sentences = {s.text_id: _squeeze(s.text).casefold() for s in contents.sentences}
            printed = sentences.get(link.text_id, "")
            target_key = next(iter(longer))
            whole = _squeeze(target_key)
            # Every place the sentence prints the link, it prints the rest of
            # the target after it.
            if not printed.count(whole) or printed.count(whole) != printed.count(squeezed):
                continue
            repaired = by_key[target_key]
        if link.link_text is None:
            link.link_text = link.url
        link.url = repaired
        changed = True
    if changed:
        contents.invalidate_text_caches()
