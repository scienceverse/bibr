"""Convert GROBID TEI XML into bibr export JSON, so GROBID is scored like bibr.

``evaluation.evaluate`` scores bibr export files. This module writes each
GROBID full-text TEI document into the same export schema (12.x), with GROBID
recorded as ``extraction.producer`` and this converter as
``extraction.converter``, so a directory of GROBID output is scored by the
unchanged evaluator, exactly as a directory of bibr exports is::

    uv run python -m evaluation.grobid_tei --tei-dir grobid-tei/ --out grobid-json/
    uv run python -m evaluation.evaluate --results-dir grobid-json/ \\
        --gold-dirs /path/to/gold --expected-ids grobid-tei/manifest.json

The comparison is only fair if everything GROBID emitted that the evaluator
can credit survives the conversion. The mapping, element by element:

* Title: the ``titleStmt`` title, else the header ``analytic`` title.
* DOI: the header ``biblStruct`` DOI, then its ``analytic``/``monogr`` DOIs.
* Authors: every header ``author`` with a ``persName`` (all ``forename``
  elements, in order, as ``given``; ``surname`` as ``family``; ``genName`` as
  ``suffix``) or an ``orgName`` (a group author, ``literal``), with ``email``,
  the ORCID ``idno`` and ``role="corresp"``. An ``author`` without a name is an
  affiliation GROBID could not attach; its affiliation is kept, unlinked.
* Affiliations: the verbatim ``raw_affiliation`` note when GROBID emits it
  (``includeRawAffiliations=1``), else the structured parts in printed order.
* Abstract: every heading and paragraph of ``profileDesc/abstract``, joined
  with spaces; keywords: the ``term`` elements (or the unsplit text).
* References, one ``bib`` row per ``listBibl/biblStruct``:
  the ``analytic`` title is the cited work's title and the ``monogr`` title
  its container (journal first, then book or proceedings, then series); with
  no ``analytic`` title, the ``monogr`` book title is the work's title.
  Authors are the ``analytic`` authors, else the ``monogr`` authors, rendered
  family name first like a printed list (``"Family, Given, Family, Given"``,
  group names too; a name part with no letter or digit is dropped); editors
  go to ``editors``, never to ``authors``. Volume, issue and pages come from
  ``biblScope`` (``from``/``to`` attributes or text; a printed range is split
  at its dash), the year from ``date/@when``. The DOI is the first
  ``idno[@type="DOI"]`` at any level; failing that, an untyped ``idno``
  holding a bare DOI or a ``doi.org`` link. With ``includeRawCitations=1``
  each printed entry becomes a ``references`` text row that its ``bib`` row
  points at.
* Body: every division becomes a section (``unknown`` type, since GROBID
  does not classify body sections; back-matter divisions keep GROBID's
  acknowledgement, funding, availability, conflict, contribution and annex
  roles), each paragraph a text row; figure and table captions and footnotes
  follow as rows of their own.

Every DOI is written bare and lowercase, as the schema requires. A value GROBID
typed as a DOI but wrapped in other text (``https://doi.org.10.1000/x``,
``ISSN…DOI10.1000/x``) is reduced to the DOI inside it; any DOI or ORCID that
cannot be conformed is dropped and named in ``extraction.warnings``, so every
loss is visible per paper.

Not converted, because the evaluator reads none of them: the in-text citation
and hyperlink tables (``xref``, ``url``), statistical expressions (``eq``),
funders, and the paper's own imprint (journal, volume, dates).

A TEI file that is empty, not XML or not TEI is a failed paper: no prediction is
written, and the run summary lists it. Pass the runner's ``manifest.json`` (or
this converter's ``--summary`` file) to the evaluator's ``--expected-ids`` so a
failed paper stays in the denominator, as a crashed bibr paper does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from lxml import etree

from bibr.export.models import _SCHEMA_VERSION, DOI_PATTERN, ISO_DATE_PATTERN, PaperExport
from bibr.input.xml_entities import parse_xml
from bibr.models import canonicalize_orcid
from bibr.utils.text import collapse_ws, normalize_doi

CONVERTER_NAME = "bibr-grobid-tei"
# Bump whenever a mapping rule changes, so converted runs can be told apart.
CONVERTER_VERSION = "1.0"
# A converter's own warning codes start with its name (see WarningExport).
WARNING_PREFIX = "BIBR_GROBID_TEI_"

# File layout shared with ``evaluation.grobid_run``.
TEI_SUFFIX = ".grobid.tei.xml"
FAILED_SUFFIX = ".grobid.failed.json"
MANIFEST_NAME = "manifest.json"
# Recognised TEI names, longest first: <paper_id>.grobid.tei.xml, .tei.xml, .xml.
_TEI_SUFFIXES = (TEI_SUFFIX, ".tei.xml", ".xml")

TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": TEI_NS}
_TEI = f"{{{TEI_NS}}}"

_DOI_RE = re.compile(DOI_PATTERN)
_ISO_DATE_RE = re.compile(ISO_DATE_PATTERN)
_DOI_INSIDE_RE = re.compile(r"10\.\d{4,9}/\S+")
_YEAR_RE = re.compile(r"^\d{4}")
_NAME_CHAR_RE = re.compile(r"[^\W_]")
# The evaluator's own page-range split (validation_metrics._PAGE_RANGE_RE).
_PAGE_RANGE_RE = re.compile(r"^\s*(?P<first>.+?)\s*[-–—]\s*(?P<last>.+?)\s*$")

# GROBID's back-matter division types -> the export's section_type vocabulary.
_BACK_SECTION_TYPES = {
    "acknowledgement": "acknowledgment",
    "funding": "funding",
    "availability": "data_availability",
    "conflict": "coi",
    "contribution": "author_contributions",
    "annex": "appendix",
}


class TeiError(ValueError):
    """A TEI file that cannot be converted: empty, not XML, not TEI, or not
    from the PDF the run recorded for it."""


# ---------------------------------------------------------------------------
# Small TEI helpers
# ---------------------------------------------------------------------------


def _text(node: Any) -> str | None:
    """Whitespace-collapsed text of an element and its descendants, or None."""
    if node is None:
        return None
    return collapse_ws("".join(node.itertext())) or None


def _text_without_labels(node: Any) -> str | None:
    """Text of *node* minus its ``<label>`` children (the printed marker)."""
    if node is None:
        return None
    parts = [node.text or ""]
    for child in node:
        if isinstance(child.tag, str) and child.tag != f"{_TEI}label":
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return collapse_ws("".join(parts)) or None


def _texts(nodes: Iterable[Any]) -> list[str]:
    return [value for value in (_text(node) for node in nodes) if value]


def _find(node: Any, path: str) -> Any:
    return None if node is None else node.find(path, _NS)


def _findall(node: Any, path: str) -> list[Any]:
    return [] if node is None else node.findall(path, _NS)


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": f"{WARNING_PREFIX}{code}", "message": message}


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def conform_doi(value: str | None, *, search: bool = False) -> str | None:
    """*value* as the bare, lowercase DOI the export schema requires, or None.

    Uses the normalizer bibr's own exporter applies (resolver prefixes,
    trailing punctuation). With *search*, a value that is not a DOI on its own
    is reduced to the ``10.xxxx/...`` DOI inside it — only for a value GROBID
    itself typed as a DOI.
    """
    if not value or not value.strip():
        return None
    bare = normalize_doi(value)
    if bare is None and search and (match := _DOI_INSIDE_RE.search(value)):
        bare = normalize_doi(match.group(0))
    if bare is None:
        return None
    bare = bare.lower()
    return bare if _DOI_RE.fullmatch(bare) else None


def _doi_url(target: str | None) -> str | None:
    """The DOI of a ``doi.org`` link (percent-decoded), else None."""
    if not target or "doi.org/" not in target.lower():
        return None
    return conform_doi(unquote(target))


def _first_doi(scopes: Sequence[Any], warnings: list[dict[str, str]], where: str) -> str | None:
    """The first conformable DOI GROBID emitted in *scopes*, in priority order.

    Typed DOIs first (direct children of each scope, in order), then an untyped
    ``idno`` that is a bare DOI, then a ``doi.org`` pointer. A typed DOI that
    cannot be conformed is reported, never silently dropped.
    """
    typed = [idno for scope in scopes for idno in _findall(scope, "tei:idno[@type='DOI']")]
    for idno in typed:
        if doi := conform_doi(_text(idno), search=True):
            return doi
    for idno in typed:
        warnings.append(_warning("DOI_DROPPED", f"{where}: {_text(idno)!r} is not a DOI"))
    for scope in scopes:
        for idno in _findall(scope, "tei:idno"):
            if idno.get("type") is None and (doi := conform_doi(_text(idno))):
                return doi
        for ptr in _findall(scope, "tei:ptr"):
            if doi := _doi_url(ptr.get("target")):
                return doi
    return None


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Name:
    given: str | None = None
    family: str | None = None
    suffix: str | None = None
    literal: str | None = None

    def rendered(self) -> str:
        """``Family, Given`` (or the single part present) for a reference string."""
        if self.literal:
            return self.literal
        name = ", ".join(part for part in (self.family, self.given) if part)
        return f"{name}, {self.suffix}" if self.suffix else name


def _parts(nodes: Iterable[Any]) -> str | None:
    """Name parts joined by spaces, minus stray punctuation GROBID tags as a name.

    A part with no letter or digit (a lone ``.``, ``;`` or ``*`` tagged as a
    forename) carries no name; kept, it would become a reference's first-author
    token and block matching on it.
    """
    return " ".join(text for text in _texts(nodes) if _NAME_CHAR_RE.search(text)) or None


def _name(person: Any) -> _Name | None:
    """Name of a TEI ``<author>``/``<editor>``: a person, a group, or None."""
    pers = _find(person, "tei:persName")
    if pers is not None:
        given = _parts(_findall(pers, "tei:forename"))
        family = _parts(_findall(pers, "tei:surname"))
        suffix = _parts(_findall(pers, "tei:genName"))
        if given or family:
            return _Name(given=given, family=family, suffix=suffix)
        if literal := _parts([pers]):
            return _Name(literal=literal)
    if literal := _parts(_findall(person, "tei:orgName")[:1]):
        return _Name(literal=literal)
    return None


def _render_names(people: Iterable[Any]) -> str | None:
    """Reference authors or editors as one ``"Family, Given, Family, Given"`` string.

    Family name first and commas only between names, like a printed APA list:
    the evaluator takes a reference's first author as the string's first token,
    up to a comma or space. Group names GROBID files under a person's
    affiliation (``orgName[@type="collaboration"]``, "…, & EASE Team") follow
    the people, once each.
    """
    names: list[str] = []
    groups: list[str] = []
    for person in people:
        if (name := _name(person)) is not None:
            names.append(name.rendered())
        for org in _findall(person, "tei:affiliation/tei:orgName[@type='collaboration']"):
            if (group := _text(org)) and group not in groups:
                groups.append(group)
    return ", ".join(names + groups) or None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _affiliation(aff: Any) -> dict[str, Any] | None:
    """One affiliation: verbatim text plus GROBID's parsed parts."""
    raw = _text_without_labels(_find(aff, "tei:note[@type='raw_affiliation']"))
    parts: list[str] = []
    for child in aff:
        if child.tag == f"{_TEI}orgName" and (value := _text(child)):
            parts.append(value)
        elif child.tag == f"{_TEI}address":
            parts.extend(_texts(child))
    text = raw or ", ".join(parts)
    if not text:
        return None
    departments = _texts(_findall(aff, "tei:orgName[@type='department']"))
    departments += _texts(_findall(aff, "tei:orgName[@type='laboratory']"))
    return {
        "text": text,
        "institution": "; ".join(_texts(_findall(aff, "tei:orgName[@type='institution']"))) or None,
        "department": "; ".join(departments) or None,
        "city": _text(_find(aff, "tei:address/tei:settlement")),
        "country": _text(_find(aff, "tei:address/tei:country")),
    }


def _header(
    header: Any, warnings: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Any]:
    """(metadata, authors, affiliations, abstract element) from ``teiHeader``."""
    bibl = _find(header, "tei:fileDesc/tei:sourceDesc/tei:biblStruct")
    analytic = _find(bibl, "tei:analytic")
    title = _text(_find(header, "tei:fileDesc/tei:titleStmt/tei:title[@type='main']"))
    title = title or _text(_find(header, "tei:fileDesc/tei:titleStmt/tei:title"))
    title = title or _text(_find(analytic, "tei:title"))
    doi = _first_doi([bibl, analytic, _find(bibl, "tei:monogr")], warnings, "header")

    authors: list[dict[str, Any]] = []
    affiliations: dict[str, dict[str, Any]] = {}
    for person in _findall(analytic, "tei:author"):
        name = _name(person)
        author_id = len(authors) + 1
        if name is not None:
            orcid_value = _text(_find(person, "tei:idno[@type='ORCID']"))
            orcid = canonicalize_orcid(orcid_value)
            if orcid_value and orcid is None:
                warnings.append(_warning("ORCID_DROPPED", f"{orcid_value!r} is not an ORCID iD"))
            authors.append(
                {
                    "author_id": author_id,
                    "given": name.given,
                    "family": name.family,
                    "suffix": name.suffix,
                    "literal": name.literal,
                    "email": _text(_find(person, "tei:email")),
                    "corresponding": person.get("role") == "corresp",
                    "orcid": orcid,
                }
            )
        for aff in _findall(person, "tei:affiliation"):
            parsed = _affiliation(aff)
            key = aff.get("key") or (parsed["text"] if parsed else None)
            if key is None or (parsed is None and key not in affiliations):
                continue  # an empty affiliation GROBID never filled in
            row = affiliations.setdefault(key, {**(parsed or {}), "author_ids": []})
            if name is not None and author_id not in row["author_ids"]:
                row["author_ids"].append(author_id)

    profile = _find(header, "tei:profileDesc")
    keywords = _texts(_findall(profile, "tei:textClass/tei:keywords/tei:term"))
    if not keywords:
        keywords = _texts(_findall(profile, "tei:textClass/tei:keywords"))
    abstract = _find(profile, "tei:abstract")
    metadata = {
        "title": title,
        "abstract": " ".join(_abstract_texts(abstract)) or None,
        "keywords": keywords,
        "doi": doi,
    }
    affiliation_rows = [
        {"affiliation_id": position, **row}
        for position, row in enumerate(affiliations.values(), start=1)
    ]
    return metadata, authors, affiliation_rows, abstract


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _pages(scope: Any) -> tuple[str | None, str | None]:
    """(first, last) page from ``biblScope[@unit='page']``: attributes, else text."""
    if scope is None:
        return None, None
    first = collapse_ws(scope.get("from") or "") or None
    last = collapse_ws(scope.get("to") or "") or None
    if first or last:
        return first, last
    text = _text(scope)
    if text and (match := _PAGE_RANGE_RE.fullmatch(text)):
        return match.group("first"), match.group("last")
    return text, None


def _scope(imprint: Any, unit: str) -> str | None:
    scope = _find(imprint, f"tei:biblScope[@unit='{unit}']")
    if scope is None:
        return None
    if scope.get("from") or scope.get("to"):
        return "-".join(v for v in (scope.get("from"), scope.get("to")) if v)
    return _text(scope)


def _title(node: Any, *levels: str | None) -> str | None:
    """First non-empty ``title`` of *node* at the given ``@level`` values, in order.

    ``None`` stands for a title without a level. Within a level, an abbreviated
    title (``type="abbrev"``) is taken only when there is no full one.
    """
    titles = [t for t in _findall(node, "tei:title") if t.get("type") != "abbrev"]
    titles += [t for t in _findall(node, "tei:title") if t.get("type") == "abbrev"]
    for level in levels:
        for title in titles:
            if title.get("level") == level and (value := _text(title)):
                return value
    return None


def _reference(bibl: Any, warnings: list[dict[str, str]], bib_id: int) -> dict[str, Any]:
    """One ``bib`` row from a ``listBibl/biblStruct``."""
    analytic = _find(bibl, "tei:analytic")
    monogr = _find(bibl, "tei:monogr")
    imprint = _find(monogr, "tei:imprint")

    article_title = _title(analytic, "a", None, "m", "j")
    if article_title:
        title = article_title
        container = _title(monogr, "j", "m", None, "s")
    else:
        # No article-level title: the monograph (book, report, thesis) is the work.
        title = _title(monogr, "m", None)
        container = _title(monogr, "j", "s")

    authors = _findall(analytic, "tei:author") or _findall(monogr, "tei:author")
    editors = _findall(analytic, "tei:editor") + _findall(monogr, "tei:editor")

    date = _find(imprint, "tei:date[@type='published'][@when]")
    date = date if date is not None else _find(bibl, ".//tei:date[@when]")
    when = collapse_ws(date.get("when") or "") if date is not None else ""
    first_page, last_page = _pages(_find(imprint, "tei:biblScope[@unit='page']"))
    notes = [
        text
        for note in _findall(bibl, "tei:note")
        if note.get("type") is None and (text := _text(note))
    ]
    ptr = next((p.get("target") for p in bibl.iterfind(".//tei:ptr", _NS) if p.get("target")), None)
    return {
        "bib_id": bib_id,
        "doi": _first_doi([analytic, monogr, bibl], warnings, f"reference {bib_id}"),
        "title": title,
        "authors": _render_names(authors),
        "editors": _render_names(editors),
        "publisher": _text(_find(imprint, "tei:publisher")),
        "year": int(when[:4]) if _YEAR_RE.match(when) else None,
        "published_date": when if _ISO_DATE_RE.fullmatch(when) else None,
        "container": container,
        "volume": _scope(imprint, "volume"),
        "issue": _scope(imprint, "issue"),
        "first_page": first_page,
        "last_page": last_page,
        "url": ptr,
        "arxiv": _text(_find(bibl, ".//tei:idno[@type='arXiv']")),
        "pmid": _text(_find(bibl, ".//tei:idno[@type='PMID']")),
        "series": _title(monogr, "s"),
        "note": "; ".join(notes) or None,
    }


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _blocks(div: Any) -> list[tuple[str, str, Any]]:
    """(kind, text, element) for a division's content, in document order.

    ``kind`` is ``head``, ``p``, ``formula`` or ``div`` (a nested division,
    with empty text). Text GROBID leaves outside any paragraph, such as a
    citation ``<ref>`` or a ``<note>`` directly under the division, becomes a
    ``p`` block of its own (element ``None``), so none of it is lost. Figures
    are skipped here: their captions become rows of their own.
    """
    out: list[tuple[str, str, Any]] = []
    if div is None:
        return out
    loose = [div.text or ""]

    def flush() -> None:
        if text := collapse_ws("".join(loose)):
            out.append(("p", text, None))
        loose.clear()

    for child in div:
        kind = child.tag.removeprefix(_TEI) if isinstance(child.tag, str) else None
        if kind in ("head", "p", "formula", "div"):
            flush()
            if kind == "div":
                out.append(("div", "", child))
            elif text := _text(child):
                out.append((kind, text, child))
        elif kind is not None and kind != "figure":
            loose.append("".join(child.itertext()))
        loose.append(child.tail or "")
    flush()
    return out


def _abstract_texts(abstract: Any) -> list[str]:
    """Every heading and paragraph of the abstract, nested divisions flattened."""
    texts: list[str] = []
    for kind, text, node in _blocks(abstract):
        texts.extend(_abstract_texts(node) if kind == "div" else [text])
    return texts


def _level(head: Any) -> int:
    """Heading depth from GROBID's section number (``n="2.1."`` -> 2)."""
    number = (head.get("n") or "").strip().strip(".") if head is not None else ""
    return len(number.split(".")) if re.fullmatch(r"\w+(\.\w+)*", number) else 1


@dataclass
class _Document:
    """Accumulates the text, section and caption tables in reading order."""

    text: list[dict[str, Any]] = field(default_factory=list)
    section: list[dict[str, Any]] = field(default_factory=list)
    figure: list[dict[str, Any]] = field(default_factory=list)
    table: list[dict[str, Any]] = field(default_factory=list)
    footnote: list[dict[str, Any]] = field(default_factory=list)
    _paragraphs: int = 0
    # Sections before this index belong to an earlier part (abstract, body,
    # back matter) and cannot be a parent.
    part_start: int = 0

    def add_section(self, header: str | None, level: int, section_type: str) -> int:
        candidates = self.section[self.part_start :]
        parent = next((s for s in reversed(candidates) if s["level"] < level), None)
        section_id = len(self.section) + 1
        self.section.append(
            {
                "section_id": section_id,
                "header": header,
                "level": level,
                "parent_section_id": parent["section_id"] if parent else None,
                "section_type": section_type,
            }
        )
        return section_id

    def add_text(
        self,
        text: str,
        section_id: int | None,
        formatted: str | None = None,
        paragraph_id: int | None = None,
    ) -> int:
        """Append a text row; without *paragraph_id* it starts a paragraph of its own."""
        if paragraph_id is None:
            self._paragraphs += 1
            paragraph_id = self._paragraphs
        text_id = len(self.text) + 1
        self.text.append(
            {
                "text": text,
                "text_id": text_id,
                "paragraph_id": paragraph_id,
                "section_id": section_id,
                "page_number": None,
                "formatted": formatted,
            }
        )
        return text_id

    def add_division(self, div: Any, section_type: str, outer_level: int = 0) -> None:
        """A division as a section: its heading, then paragraphs and display equations.

        A paragraph GROBID split into sentences (``segmentSentences=1``) gives
        one row per ``<s>``, sharing a paragraph id. A nested division is a
        subsection; a second heading is kept as a paragraph.
        """
        blocks = _blocks(div)
        heading = next((i for i, block in enumerate(blocks) if block[0] == "head"), None)
        head = blocks[heading][2] if heading is not None else None
        level = max(_level(head), outer_level + 1)
        section_id = self.add_section(_text(head), level, section_type)
        for index, (kind, text, node) in enumerate(blocks):
            if index == heading:
                continue
            if kind == "div":
                self.add_division(node, section_type, level)
            elif kind == "formula":
                # bibr writes a display equation as a placeholder row, the same way.
                self.add_text("[equation]", section_id, formatted=text)
            else:
                self._paragraphs += 1
                sentences = _texts(_findall(node, "tei:s")) if kind == "p" else []
                for sentence in sentences or [text]:
                    self.add_text(sentence, section_id, paragraph_id=self._paragraphs)

    def add_float(self, figure: Any) -> None:
        """A figure or table: its caption becomes a text row after the body."""
        label = _text(_find(figure, "tei:label"))
        caption = _text(_find(figure, "tei:figDesc")) or _text(_find(figure, "tei:head"))
        text_id = self.add_text(caption, None) if caption else None
        row = {
            "label": re.sub(r"\s+", "", label) if label else None,
            "section_id": None,
            "text_id": text_id,
            "caption": caption,
            "page_number": None,
        }
        if figure.get("type") == "table":
            cells = [
                [_text(cell) or "" for cell in _findall(r, "tei:cell")]
                for r in _findall(figure, "tei:table/tei:row")
            ]
            self.table.append({"table_id": len(self.table) + 1, "contents": cells, **row})
        else:
            self.figure.append({"figure_id": len(self.figure) + 1, **row})


def _document(root: Any, abstract: Any, references: list[Any]) -> tuple[_Document, list[int]]:
    """The text tables, plus the text row of each printed reference (or 0)."""
    doc = _Document()
    if texts := _abstract_texts(abstract):
        # The same blocks as metadata.abstract; GROBID prints no "Abstract" heading.
        section_id = doc.add_section(None, 1, "abstract")
        for text in texts:
            doc.add_text(text, section_id)
    text_el = _find(root, "tei:text")
    doc.part_start = len(doc.section)
    for div in _findall(text_el, "tei:body/tei:div"):
        doc.add_division(div, "unknown")
    for wrapper in _findall(text_el, "tei:back/tei:div"):
        kind = wrapper.get("type")
        if kind == "references":
            continue
        doc.part_start = len(doc.section)
        section_type = _BACK_SECTION_TYPES.get(kind or "", "unknown")
        for div in _findall(wrapper, "tei:div") or [wrapper]:
            doc.add_division(div, section_type)
    doc.part_start = len(doc.section)

    reference_rows = []
    raw = [_text(_find(bibl, "tei:note[@type='raw_reference']")) for bibl in references]
    section_id = doc.add_section(None, 1, "references") if any(raw) else None
    for printed in raw:
        reference_rows.append(doc.add_text(printed, section_id) if printed else 0)

    for figure in text_el.iterfind(".//tei:figure", _NS) if text_el is not None else []:
        doc.add_float(figure)
    for note in _findall(text_el, "tei:body/tei:note[@place='foot']"):
        if printed := " ".join(_texts(_findall(note, "tei:p"))) or _text(note):
            text_id = doc.add_text(printed, None)
            doc.footnote.append(
                {
                    "footnote_id": len(doc.footnote) + 1,
                    "label": collapse_ws(note.get("n") or "") or None,
                    "text_id": text_id,
                }
            )
    return doc, reference_rows


# ---------------------------------------------------------------------------
# Whole document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Producer:
    """The GROBID build and request parameters a TEI file records."""

    version: str
    revision: str | None
    parameters: str | None
    completed_at: str | None


def read_tei(data: bytes) -> Any:
    """Parse TEI bytes with bibr's hardened XML parser; ``TeiError`` if unusable."""
    if not data.strip():
        raise TeiError("empty file")
    try:
        root = parse_xml(data)
    except etree.XMLSyntaxError as error:
        raise TeiError(f"not well-formed XML: {error}") from error
    if root.tag != f"{_TEI}TEI":
        raise TeiError(f"root element is {root.tag!r}, not TEI")
    return root


def _utc_timestamp(value: str | None) -> str | None:
    """GROBID's ``application/@when`` (``2026-06-19T14:51+0000``) as UTC ISO 8601."""
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            moment = datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
        return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def producer_of(root: Any) -> Producer:
    """GROBID version, revision, parameters and completion time from ``appInfo``."""
    app = root.find(".//tei:encodingDesc/tei:appInfo/tei:application[@ident='GROBID']", _NS)
    labels = {label.get("type"): _text(label) for label in _findall(app, "tei:label")}
    return Producer(
        version=(app.get("version") if app is not None else None) or "unknown",
        revision=labels.get("revision"),
        parameters=labels.get("parameters"),
        completed_at=_utc_timestamp(app.get("when") if app is not None else None),
    )


def tei_md5(root: Any) -> str | None:
    """MD5 of the input PDF, as GROBID records it in the header."""
    value = _text(root.find(".//tei:sourceDesc/tei:biblStruct/tei:idno[@type='MD5']", _NS))
    return value.lower() if value else None


def tei_to_export(
    root: Any,
    *,
    source: dict[str, Any],
    converter_build_sha: str | None = None,
    warnings: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    """The export-schema payload for one parsed GROBID TEI document.

    *source* is the export's ``source`` block (``file_name``, ``sha256``,
    ``input_format``); ``paper_id`` is its file name without the extension.
    The payload is validated against the strict export model, so it is
    exactly what the evaluator reads from a bibr export.
    """
    notes = list(warnings)
    metadata, authors, affiliations, abstract = _header(_find(root, "tei:teiHeader"), notes)
    references = root.findall(".//tei:text//tei:listBibl/tei:biblStruct", _NS)
    bib = [_reference(bibl, notes, position) for position, bibl in enumerate(references, 1)]
    doc, reference_rows = _document(root, abstract, references)
    for row, text_id in zip(bib, reference_rows, strict=True):
        row["text_id"] = text_id or None

    producer = producer_of(root)
    revision = producer.revision if producer.revision != producer.version else None
    payload = {
        "paper_id": Path(source["file_name"]).stem,
        "schema_version": _SCHEMA_VERSION,
        "source": source,
        "metadata": metadata,
        "author": authors,
        "affiliation": affiliations,
        "funding": [],
        "text": doc.text,
        "section": doc.section,
        "url": [],
        "bib": bib,
        "xref": [],
        "figure": doc.figure,
        "table": doc.table,
        "footnote": doc.footnote,
        "eq": [],
        "extraction": {
            "producer": {"name": "grobid", "version": producer.version, "build_sha": revision},
            "converter": {
                "name": CONVERTER_NAME,
                "version": CONVERTER_VERSION,
                "build_sha": converter_build_sha,
            },
            "completed_at": producer.completed_at or _now(),
            "ocr": None,
            "llm": None,
            "warnings": notes,
        },
    }
    return PaperExport.model_validate(payload).model_dump(by_alias=True)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------


def paper_id_for(path: Path) -> str:
    """The paper id a TEI file name carries: its name minus the TEI suffix."""
    name = path.name
    for suffix in _TEI_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _digests(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), hashlib.md5(data, usedforsecurity=False).hexdigest()


@dataclass
class ConversionSummary:
    """What a directory conversion produced, for the operator and ``--expected-ids``."""

    converted: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    # TEI files the runner's manifest does not list (another run's leftovers).
    ignored: list[str] = field(default_factory=list)
    producers: Counter[tuple[str, str | None]] = field(default_factory=Counter)
    warnings: Counter[str] = field(default_factory=Counter)

    def as_dict(self, converter_build_sha: str | None) -> dict[str, Any]:
        return {
            "converter": {
                "name": CONVERTER_NAME,
                "version": CONVERTER_VERSION,
                "build_sha": converter_build_sha,
            },
            # Every paper the TEI directory accounts for, converted or failed:
            # usable as the evaluator's --expected-ids.
            "ids": sorted({*self.converted, *self.failed}),
            "converted": len(self.converted),
            "failed": [
                {"paper_id": paper_id, "reason": reason}
                for paper_id, reason in sorted(self.failed.items())
            ],
            "ignored": sorted(self.ignored),
            "producers": [
                {"grobid_version": version, "parameters": parameters, "papers": count}
                for (version, parameters), count in sorted(
                    self.producers.items(), key=lambda item: (-item[1], str(item[0]))
                )
            ],
            "warnings": dict(sorted(self.warnings.items())),
        }


def _manifest(tei_dir: Path) -> dict[str, dict[str, Any]]:
    """The runner's per-paper records, keyed by paper id; empty without a manifest."""
    path = tei_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    papers = json.loads(path.read_text(encoding="utf-8")).get("papers") or []
    return {str(entry["paper_id"]): entry for entry in papers if entry.get("paper_id")}


def _find_pdf(paper_id: str, pdf_dirs: Sequence[Path]) -> Path | None:
    for directory in pdf_dirs:
        for suffix in (".pdf", ".PDF"):
            if (candidate := directory / f"{paper_id}{suffix}").is_file():
                return candidate
    return None


def _source(
    paper_id: str,
    record: dict[str, Any] | None,
    pdf_dirs: Sequence[Path],
    md5_in_tei: str | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """The export ``source`` block: the PDF GROBID read, from the runner or a PDF dir.

    Raises ``TeiError`` when the TEI's recorded MD5 shows it came from another PDF.
    Without either, the PDF name is inferred from the TEI name and flagged.
    """
    if record is not None and record.get("pdf"):
        sha256, md5 = record.get("sha256"), record.get("md5")
        file_name = str(record["pdf"])
    elif (pdf := _find_pdf(paper_id, pdf_dirs)) is not None:
        (sha256, md5), file_name = _digests(pdf), pdf.name
    else:
        source = {"file_name": f"{paper_id}.pdf", "sha256": None, "input_format": "pdf"}
        message = f"PDF not available; file name {paper_id}.pdf inferred from the TEI name"
        return source, [_warning("SOURCE_INFERRED", message)]
    if md5 and md5_in_tei and md5.lower() != md5_in_tei:
        raise TeiError(f"TEI records PDF MD5 {md5_in_tei}, but {file_name} has MD5 {md5}")
    return {"file_name": file_name, "sha256": sha256, "input_format": "pdf"}, []


def convert_directory(
    tei_dir: Path,
    out_dir: Path,
    *,
    pdf_dirs: Sequence[Path] = (),
    converter_build_sha: str | None = None,
) -> ConversionSummary:
    """Convert every TEI file in *tei_dir* into ``<out_dir>/<paper_id>.json``.

    A paper the runner recorded as failed, or whose TEI is unusable, gets no
    prediction file; the summary lists it with the reason.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.json")):
        raise FileExistsError(f"{out_dir} already holds JSON files; convert into an empty dir")
    records = _manifest(tei_dir)
    summary = ConversionSummary()
    tei_files: dict[str, Path] = {}
    for path in sorted(tei_dir.iterdir()):
        if path.is_file() and path.name.lower().endswith(".xml"):
            paper_id = paper_id_for(path)
            if paper_id in tei_files:
                raise ValueError(f"{tei_files[paper_id].name} and {path.name} both hold {paper_id}")
            tei_files[paper_id] = path
    if records:
        summary.ignored = sorted(set(tei_files) - set(records))
        for paper_id in summary.ignored:
            del tei_files[paper_id]
    for paper_id, record in records.items():
        if record.get("status") != "ok":
            summary.failed[paper_id] = f"GROBID run failed: {record.get('error') or 'unknown'}"
        elif paper_id not in tei_files:
            summary.failed[paper_id] = "the runner recorded a TEI file that is missing"
    if not records:
        for path in sorted(tei_dir.glob(f"*{FAILED_SUFFIX}")):
            paper_id = path.name[: -len(FAILED_SUFFIX)]
            if paper_id not in tei_files:
                summary.failed[paper_id] = "GROBID run failed (failure marker)"

    for paper_id, path in tei_files.items():
        if paper_id in summary.failed:
            continue
        try:
            root = read_tei(path.read_bytes())
            source, notes = _source(paper_id, records.get(paper_id), pdf_dirs, tei_md5(root))
            payload = tei_to_export(
                root, source=source, converter_build_sha=converter_build_sha, warnings=notes
            )
        except TeiError as error:
            summary.failed[paper_id] = str(error)
            continue
        target = out_dir / f"{payload['paper_id']}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        producer = producer_of(root)
        summary.converted.append(payload["paper_id"])
        summary.producers[(producer.version, producer.parameters)] += 1
        summary.warnings.update(w["code"] for w in payload["extraction"]["warnings"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tei-dir", type=Path, required=True, help="GROBID TEI files")
    parser.add_argument("--out", type=Path, required=True, help="Empty directory for the JSON")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        nargs="+",
        default=[],
        help="Directories holding the PDFs, for their name and SHA-256 when the TEI "
        "directory has no runner manifest (<paper_id>.pdf).",
    )
    parser.add_argument(
        "--summary", type=Path, help="Write the conversion summary JSON here (not in --out)"
    )
    args = parser.parse_args(argv)

    from evaluation.evaluate import bibr_commit

    build_sha = bibr_commit()
    summary = convert_directory(
        args.tei_dir, args.out, pdf_dirs=args.pdf_dir, converter_build_sha=build_sha
    )
    report = summary.as_dict(build_sha)
    print(f"Converted {report['converted']} TEI file(s) into {args.out}")
    for producer in report["producers"]:
        print(f"  GROBID {producer['grobid_version']} ({producer['papers']} papers)")
        print(f"    parameters: {producer['parameters']}")
    if len(report["producers"]) > 1:
        print("WARNING: the TEI files come from more than one GROBID version or parameter set")
    for failure in report["failed"]:
        print(f"  failed: {failure['paper_id']}: {failure['reason']}")
    for paper_id in report["ignored"]:
        print(f"  ignored (not in the run manifest): {paper_id}")
    for code, count in report["warnings"].items():
        print(f"  {code}: {count}")
    if report["failed"]:
        print(
            f"{len(report['failed'])} paper(s) have no prediction. Score with "
            "--expected-ids (the runner's manifest.json or --summary) so they count."
        )
    if args.summary:
        args.summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["converted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
