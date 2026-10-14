"""Reference lists for reference managers: CSL-JSON, BibTeX and RIS.

Built from an export's ``bib`` rows (the references as printed), its
``bib_match`` rows (the registry records they were matched to) and its
``bib_check`` rows (whether each reference agrees with its record).

Which values are written:

- By default (``prefer="printed"``) every field is the reference's printed
  value; a field the reference does not print is filled from the matched
  record. ``prefer="record"`` takes the record's values over the printed ones.
- A record is used only when the reference's check status is ``matched``.
  A reference whose record disagrees with it (``fields_disagree``), whose
  printed DOI does not exist (``doi_unregistered``), or that was not found or
  not checked is written as printed. An export without ``bib_check`` rows
  (enrichment off, or written before export schema 12.3) is checked here
  first when it has ``bib_match`` rows.
- Names come from the matched record, which lists them as family and given
  names; otherwise the printed list is split (:mod:`bibr.export.names`). An
  organization becomes one name, and a list that cannot be split is written
  as one name rather than split wrongly. "et al." becomes "and others" in
  BibTeX and is left out of CSL-JSON and RIS.
- The type is the record's when matched, else the printed one; an untyped
  reference with a container and a volume, issue or page is a journal
  article.
- Pages: an abbreviated last page is written in full ("1267-89" as
  1267-1289), and an article number ("e0123456") as the first page only.
- A note says what the check found when it is not ``matched`` (and not
  ``unchecked``) and names any retraction or withdrawal notice of the record;
  ``check_note=False`` leaves it out. The printed note comes first.

The export itself is never changed.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from bibr.export.names import (
    PrintedName,
    after_repeated_author_rule,
    is_repeated_author_rule,
    split_printed_names,
)

__all__ = [
    "CONTENT_TYPES",
    "EXTENSIONS",
    "FORMATS",
    "Prefer",
    "ReferenceFormat",
    "ReferencesReport",
    "expand_page_range",
    "format_for_path",
    "normalize_format",
    "render_references",
    "to_bibtex",
    "to_csl_json",
    "to_ris",
    "write_references",
]

ReferenceFormat = Literal["csl-json", "bibtex", "ris"]
Prefer = Literal["printed", "record"]
FORMATS: tuple[str, ...] = get_args(ReferenceFormat)
CONTENT_TYPES: dict[str, str] = {
    "csl-json": "application/vnd.citationstyles.csl+json",
    "bibtex": "application/x-bibtex",
    "ris": "application/x-research-info-systems",
}
EXTENSIONS: dict[str, str] = {"csl-json": ".json", "bibtex": ".bib", "ris": ".ris"}
_FORMAT_ALIASES = {
    "csl-json": "csl-json",
    "csl_json": "csl-json",
    "csljson": "csl-json",
    "csl": "csl-json",
    "json": "csl-json",
    "bibtex": "bibtex",
    "bib": "bibtex",
    "ris": "ris",
}

# bib_type -> (CSL type, BibTeX entry type, RIS type).
_TYPES: dict[str, tuple[str, str, str]] = {
    "journal_article": ("article-journal", "article", "JOUR"),
    "book": ("book", "book", "BOOK"),
    "book_chapter": ("chapter", "incollection", "CHAP"),
    "conference_paper": ("paper-conference", "inproceedings", "CPAPER"),
    "report": ("report", "techreport", "RPRT"),
    "thesis": ("thesis", "phdthesis", "THES"),
    "dataset": ("dataset", "misc", "DATA"),
    "software": ("software", "misc", "COMP"),
    "preprint": ("article", "misc", "UNPB"),
    "other": ("document", "misc", "GEN"),
}
_SERVICE_NAMES = {
    "crossref": "Crossref",
    "openalex": "OpenAlex",
    "datacite": "DataCite",
    "doi.org": "doi.org",
    "openlibrary": "Open Library",
    "ror": "ROR",
}
# Notice type -> (noun, what it did to the work).
_NOTICE_WORDS = {
    "retraction": ("retraction", "retracted"),
    "partial_retraction": ("partial retraction", "partly retracted"),
    "withdrawal": ("withdrawal", "withdrawn"),
    "removal": ("removal", "removed"),
}
_FIELD_WORDS = {"first_page": "pages", "last_page": "pages", "container": "journal"}
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
# Markup a registry keeps in titles: <i>, </sub>, <mml:math ...>.
_TAGS = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>")
_ISO_DATE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_LOCATOR = re.compile(r"^([^\W\d_]*)(\d+)$")
# Characters NFKD does not reduce to ASCII, for cite keys.
_SPECIAL = str.maketrans(
    {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l", "ı": "i", "þ": "th"}
    | {"Æ": "AE", "Œ": "OE", "Ø": "O", "Đ": "D", "Ł": "L", "Þ": "Th"}
)
# BibTeX balances braces whether or not a backslash precedes them, so a literal
# brace is written as a command.
_BIBTEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "{": r"\textbraceleft{}",
    "}": r"\textbraceright{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


_FROM_RECORD = object()


@dataclass
class _Entry:
    """One reference, its values chosen, ready for any of the writers."""

    bib_type: str
    title: str | None = None
    authors: list[PrintedName] = field(default_factory=list)
    authors_et_al: bool = False
    editors: list[PrintedName] = field(default_factory=list)
    editors_et_al: bool = False
    container: str | None = None
    series: str | None = None
    publisher: str | None = None
    edition: str | None = None
    version: str | None = None
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None
    year: int | None = None
    month: int | None = None
    day: int | None = None
    in_press: bool = False
    doi: str | None = None
    url: str | None = None
    pmid: str | None = None
    arxiv: str | None = None
    accessed: str | None = None
    notes: list[str] = field(default_factory=list)
    key: str = ""


@dataclass(frozen=True)
class ReferencesReport:
    """What :func:`write_references` wrote."""

    path: Path
    format: str
    references: int
    papers: int
    skipped: tuple[tuple[str, str], ...] = ()  # (source, reason)


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


def normalize_format(value: str) -> str:
    """The format *value* names (``csl-json``, ``bibtex``, ``ris``; ``bib``,
    ``csl`` and ``json`` are accepted too)."""
    fmt = _FORMAT_ALIASES.get(str(value).strip().lower())
    if fmt is None:
        raise ValueError(f"unknown reference format {value!r}; use one of {', '.join(FORMATS)}")
    return fmt


def format_for_path(path: str | os.PathLike[str]) -> str | None:
    """The format a file name's extension implies (.bib, .ris, .json), or None."""
    suffix = Path(path).suffix.lower()
    return {".bib": "bibtex", ".bibtex": "bibtex", ".ris": "ris", ".json": "csl-json"}.get(suffix)


def _check_prefer(prefer: str) -> str:
    if prefer not in get_args(Prefer):
        raise ValueError(f"prefer must be 'printed' or 'record', not {prefer!r}")
    return prefer


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    """A value as one line of text; None for nothing."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _record_text(value: Any) -> str | None:
    """A record value: markup (``<i>``) a registry keeps in titles removed."""
    text = _text(value)
    return _text(_TAGS.sub("", text)) if text else None


def _record_locator(value: Any) -> str | None:
    """A record's volume, issue or page; its placeholder "0" is no value."""
    text = _record_text(value)
    return None if text in ("0", "00") else text


def _bare_doi(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    return _DOI_PREFIX.sub("", text).strip().lower() or None


def _date_parts(value: Any) -> tuple[int, int | None, int | None] | None:
    """``(year, month, day)`` of an ISO date or a year; None if neither."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value, None, None) if 0 < value < 10000 else None
    text = _text(value)
    match = _ISO_DATE.match(text or "")
    if not match:
        return None
    year, month, day = (int(g) if g else None for g in match.groups())
    if month is not None and not 1 <= month <= 12:
        month = day = None
    if day is not None and not 1 <= day <= 31:
        day = None
    return year, month, day  # type: ignore[return-value]


def _date_of(row: Mapping[str, Any]) -> tuple[int, int | None, int | None] | None:
    return _date_parts(row.get("published_date")) or _date_parts(row.get("year"))


def expand_page_range(first: Any, last: Any) -> tuple[str | None, str | None]:
    """``(first, last)`` pages with an abbreviated last page written in full:
    ("1267", "89") is ("1267", "1289"), ("S12", "19") is ("S12", "S19"). An
    article number ("e0123456") has no last page; a last page equal to the
    first is dropped, and one that does not read as a range after the first
    page ("xii"-"xv", "120"-"11") is kept as printed."""
    first_text, last_text = _text(first), _text(last)
    if not first_text:
        return None, None
    if not last_text or last_text == first_text:
        return first_text, None
    m_first, m_last = _LOCATOR.match(first_text), _LOCATOR.match(last_text)
    if not m_first or not m_last:
        return first_text, last_text
    prefix, digits = m_first.groups()
    last_prefix, last_digits = m_last.groups()
    if last_prefix and last_prefix != prefix:
        return first_text, last_text
    if len(last_digits) < len(digits):
        last_digits = digits[: len(digits) - len(last_digits)] + last_digits
    if int(last_digits) < int(digits):
        # Not a range read the usual way ("120-11"): kept as printed.
        return first_text, last_text
    return first_text, prefix + last_digits


# ---------------------------------------------------------------------------
# Building entries
# ---------------------------------------------------------------------------


def _export_dict(source: Any) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    data = getattr(source, "data", None)
    if isinstance(data, Mapping):
        return data
    raise TypeError(f"expected a bibr export dict or Result, not {type(source).__name__}")


def _rows(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rows = data.get(key)
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _check_rows(data: Mapping[str, Any]) -> dict[Any, Mapping[str, Any]]:
    """``bib_check`` rows by ``bib_id``. An export without them that has
    ``bib_match`` rows is checked on a copy first."""
    rows = _rows(data, "bib_check")
    if not rows and _rows(data, "bib_match"):
        from bibr.enrich.check import check_bibs

        checked = dict(data)  # check_bibs only sets checked["bib_check"]
        check_bibs(checked)
        rows = _rows(checked, "bib_check")
    return {row.get("bib_id"): row for row in rows}


def _score(match: Mapping[str, Any]) -> float:
    score = match.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return 0.0
    return score / 100 if score > 1 else float(score)


def _matched_record(
    check: Mapping[str, Any] | None, matches: list[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """The record a ``matched`` reference was checked against; None otherwise."""
    if check is None or check.get("status") != "matched" or not matches:
        return None
    service = check.get("service")
    candidates = [m for m in matches if m.get("service") == service] or matches
    return max(candidates, key=_score)


def _record_names(people: Any) -> list[PrintedName]:
    names: list[PrintedName] = []
    for person in people if isinstance(people, list) else []:
        if not isinstance(person, Mapping):
            continue
        family, given = _record_text(person.get("family")), _record_text(person.get("given"))
        literal = _record_text(person.get("literal"))
        suffix = _record_text(person.get("suffix"))
        if family and family.lower() in ("and", "others", "et al", "et al."):
            continue
        if family or given:
            names.append(PrintedName(family=family, given=given, suffix=suffix))
        elif literal:
            names.append(PrintedName(literal=literal))
    return names


def _printed_names(
    text: Any, previous: tuple[list[PrintedName], bool], *, editors: bool = False
) -> tuple[list[PrintedName], bool]:
    """Split a printed name list; a rule ("——") repeats the previous
    reference's authors, with any names printed after it."""
    printed = _text(text)
    if not printed:
        return [], False
    if not editors and is_repeated_author_rule(printed):
        rest = after_repeated_author_rule(printed)
        if not rest:
            return list(previous[0]), previous[1]
        names, et_al = split_printed_names(rest)
        return [*previous[0][:1], *names], et_al
    return split_printed_names(printed, editors=editors)


def _entry_type(bib: Mapping[str, Any], record: Mapping[str, Any] | None) -> str:
    printed = str(bib["bib_type"]) if bib.get("bib_type") in _TYPES else None
    recorded = str(record["bib_type"]) if record and record.get("bib_type") in _TYPES else None
    # The registry's catch-all "other" does not replace a printed type.
    if recorded and not (recorded == "other" and printed):
        return recorded
    if printed:
        return printed
    has_locator = any(_text(bib.get(k)) for k in ("volume", "issue", "first_page"))
    return "journal_article" if _text(bib.get("container")) and has_locator else "other"


def _check_notes(check: Mapping[str, Any] | None) -> list[str]:
    if not check:
        return []
    service = _SERVICE_NAMES.get(str(check.get("service")), "registry")
    status = check.get("status")
    notes: list[str] = []
    if status == "fields_disagree":
        fields: list[str] = []
        for note in check.get("notes") or []:
            name = note.get("field") if isinstance(note, Mapping) else None
            if isinstance(name, str):
                word = _FIELD_WORDS.get(name, name.replace("_", " "))
                if word not in fields:
                    fields.append(word)
        record = f"the {service} record" if check.get("service") else "the record its DOI names"
        detail = f" ({', '.join(fields)})" if fields else ""
        notes.append(f"bibr reference check: differs from {record}{detail}")
    elif status == "doi_unregistered":
        notes.append("bibr reference check: doi.org says the printed DOI does not exist")
    elif status == "not_found":
        notes.append("bibr reference check: not found in Crossref")
    retraction = check.get("retraction")
    if isinstance(retraction, Mapping):
        noun, done = _NOTICE_WORDS.get(str(retraction.get("type")), _NOTICE_WORDS["retraction"])
        notice = f"a {noun} notice"
        if _text(retraction.get("date")):
            notice += f" dated {_text(retraction.get('date'))}"
        if _bare_doi(retraction.get("doi")):
            notice += f", https://doi.org/{_bare_doi(retraction.get('doi'))}"
        if status == "matched":
            notes.append(f"bibr reference check: {done} (the {service} record lists {notice})")
        else:
            # The record may be another work than the one cited.
            notes.append(
                f"bibr reference check: the {service} record matched to this reference "
                f"lists {notice}"
            )
    return notes


def _build_entry(
    bib: Mapping[str, Any],
    check: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
    *,
    prefer: str,
    check_note: bool,
    previous: tuple[list[PrintedName], bool],
) -> _Entry:
    take_record = prefer == "record"

    def pick(name: str, recorded: Any = _FROM_RECORD) -> str | None:
        printed = _text(bib.get(name))
        if recorded is _FROM_RECORD:
            recorded = _record_text(record.get(name)) if record is not None else None
        if take_record and recorded:
            return str(recorded)
        return printed or (str(recorded) if recorded else None)

    printed_doi = _bare_doi(bib.get("doi"))
    record_doi = _bare_doi(record.get("doi")) if record else None
    entry = _Entry(
        bib_type=_entry_type(bib, record),
        doi=(record_doi or printed_doi) if take_record else (printed_doi or record_doi),
    )
    entry.title = pick("title")
    entry.container = pick("container")
    entry.publisher = pick("publisher")
    entry.edition = pick("edition")
    entry.version = pick("version")
    entry.volume = pick("volume", _record_locator(record.get("volume")) if record else None)
    entry.issue = pick("issue", _record_locator(record.get("issue")) if record else None)
    entry.series = _text(bib.get("series"))
    entry.pmid = _text(bib.get("pmid"))
    entry.arxiv = _text(bib.get("arxiv"))
    entry.accessed = _text(bib.get("access_date"))

    # A record URL that only resolves the DOI adds nothing.
    record_url = _record_text(record.get("url")) if record else None
    if record_url and re.match(r"^https?://(?:dx\.)?doi\.org/", record_url, re.IGNORECASE):
        record_url = None
    entry.url = pick("url", record_url)

    # Pages and dates are taken whole from one side, never mixed.
    printed_pages = expand_page_range(bib.get("first_page"), bib.get("last_page"))
    record_pages: tuple[str | None, str | None] = (None, None)
    if record is not None:
        record_pages = expand_page_range(
            _record_locator(record.get("first_page")), _record_locator(record.get("last_page"))
        )
    if take_record and record_pages[0]:
        entry.first_page, entry.last_page = record_pages
    else:
        entry.first_page, entry.last_page = printed_pages if printed_pages[0] else record_pages

    printed_date = _date_of(bib)
    record_date = _date_of(record) if record is not None else None
    date = (record_date or printed_date) if take_record else (printed_date or record_date)
    if date:
        entry.year, entry.month, entry.day = date
    # In press, unless the record dated the work.
    entry.in_press = bool(bib.get("is_in_press")) and (date is None or date == printed_date)

    record_authors = _record_names(record.get("author")) if record else []
    if record_authors:
        entry.authors = record_authors
    else:
        entry.authors, entry.authors_et_al = _printed_names(bib.get("authors"), previous)
    record_editors = _record_names(record.get("editor")) if record else []
    if record_editors:
        entry.editors = record_editors
    else:
        entry.editors, entry.editors_et_al = _printed_names(
            bib.get("editors"), ([], False), editors=True
        )

    printed_note = _text(bib.get("note"))
    if printed_note:
        entry.notes.append(printed_note)
    if check_note:
        entry.notes.extend(_check_notes(check))
    return entry


def _entries(data: Mapping[str, Any], *, prefer: str, check_note: bool) -> list[_Entry]:
    checks = _check_rows(data)
    matches_by_bib: dict[Any, list[Mapping[str, Any]]] = {}
    for match in _rows(data, "bib_match"):
        matches_by_bib.setdefault(match.get("bib_id"), []).append(match)
    entries: list[_Entry] = []
    previous: tuple[list[PrintedName], bool] = ([], False)
    for bib in _rows(data, "bib"):
        check = checks.get(bib.get("bib_id"))
        record = _matched_record(check, matches_by_bib.get(bib.get("bib_id"), []))
        entry = _build_entry(
            bib, check, record, prefer=prefer, check_note=check_note, previous=previous
        )
        previous = (entry.authors, entry.authors_et_al)
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Cite keys
# ---------------------------------------------------------------------------


def _key_part(value: str | None) -> str:
    folded = unicodedata.normalize("NFKD", (value or "").translate(_SPECIAL))
    return "".join(c for c in folded if c.isascii() and c.isalnum())


def _collision_suffix(index: int) -> str:
    """'' for the first use of a key, then a, b, ..., z, aa, ab, ..."""
    suffix = ""
    while index > 0:
        index -= 1
        suffix = chr(ord("a") + index % 26) + suffix
        index //= 26
    return suffix


def _assign_keys(entries: list[_Entry]) -> None:
    """Deterministic, unique cite keys: first author's family name and the
    year ("Smith2020"), then "Smith2020a", "Smith2020b" for the next ones."""
    next_suffix: dict[str, int] = {}
    emitted: set[str] = set()
    for entry in entries:
        lead = entry.authors[0] if entry.authors else None
        name = (lead.family or lead.literal or lead.given) if lead else None
        base_name = _key_part(name)[:40] or "ref"
        when = str(entry.year) if entry.year else ("inpress" if entry.in_press else "nd")
        base = f"{base_name}{when}"
        collision = next_suffix.get(base, 0)
        key = base + _collision_suffix(collision)
        while key in emitted:
            collision += 1
            key = base + _collision_suffix(collision)
        next_suffix[base] = collision + 1
        emitted.add(key)
        entry.key = key


def _all_entries(
    exports: Iterable[Mapping[str, Any]], *, prefer: str, check_note: bool
) -> list[_Entry]:
    _check_prefer(prefer)
    entries: list[_Entry] = []
    for data in exports:
        entries.extend(_entries(data, prefer=prefer, check_note=check_note))
    _assign_keys(entries)
    return entries


# ---------------------------------------------------------------------------
# CSL-JSON
# ---------------------------------------------------------------------------


def _csl_date(entry: _Entry) -> dict[str, Any] | None:
    if entry.year is None:
        return None
    parts = [entry.year]
    if entry.month:
        parts.append(entry.month)
        if entry.day:
            parts.append(entry.day)
    return {"date-parts": [parts]}


def _csl_item(entry: _Entry) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": entry.key,
        "citation-key": entry.key,
        "type": _TYPES[entry.bib_type][0],
    }

    def put(name: str, value: Any) -> None:
        if value not in (None, "", []):
            item[name] = value

    put("title", entry.title)
    put("author", [n.as_dict() for n in entry.authors])
    put("editor", [n.as_dict() for n in entry.editors])
    put("container-title", entry.container)
    put("collection-title", entry.series)
    put("publisher", entry.publisher)
    put("edition", entry.edition)
    put("version", entry.version)
    put("volume", entry.volume)
    put("issue", entry.issue)
    if entry.first_page:
        put(
            "page", f"{entry.first_page}-{entry.last_page}" if entry.last_page else entry.first_page
        )
    put("issued", _csl_date(entry))
    if entry.in_press:
        item["status"] = "in press"
    put("DOI", entry.doi)
    put("URL", entry.url)
    put("PMID", entry.pmid)
    if entry.accessed:
        accessed = _date_parts(entry.accessed)
        if accessed:
            parts = [p for p in accessed if p is not None]
            item["accessed"] = {"date-parts": [parts]}
        else:
            item["accessed"] = {"raw": entry.accessed}
    notes = list(entry.notes)
    if entry.arxiv:
        notes.append(f"arXiv: {entry.arxiv}")
    put("note", "\n".join(notes))
    return item


def to_csl_json(
    data: Any, *, prefer: Prefer = "printed", check_note: bool = True
) -> list[dict[str, Any]]:
    """The export's references as CSL-JSON items (a list of dicts, ready for
    :func:`json.dump`; Zotero, Mendeley, Pandoc and citeproc read it)."""
    entries = _all_entries([_export_dict(data)], prefer=prefer, check_note=check_note)
    return [_csl_item(e) for e in entries]


# ---------------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------------


def _bibtex_escape(value: str) -> str:
    return "".join(_BIBTEX_SPECIALS.get(c, c) for c in " ".join(value.split()))


def _bibtex_verbatim(value: str) -> str:
    """A URL or DOI: kept as it is, but a brace or backslash would end the
    field, so those are percent-encoded."""
    value = "".join(value.split())
    return value.replace("\\", "%5C").replace("{", "%7B").replace("}", "%7D")


_BIBTEX_AND = re.compile(r"\band\b|,", re.IGNORECASE)


def _bibtex_name(name: PrintedName) -> str:
    if name.literal or not name.family:
        return "{" + _bibtex_escape(name.literal or name.given or "") + "}"

    def part(text: str) -> str:
        escaped = _bibtex_escape(text)
        # "and" or a comma inside a part would split the name list or the name.
        return "{" + escaped + "}" if _BIBTEX_AND.search(text) else escaped

    family = part(name.family)
    if " " in name.family and not name.given and not name.suffix:
        family = "{" + _bibtex_escape(name.family) + "}"
    pieces = [family]
    if name.suffix:
        pieces.append(part(name.suffix))
    if name.given:
        pieces.append(part(name.given))
    elif name.suffix:
        pieces.append("")
    return ", ".join(pieces)


def _bibtex_names(names: list[PrintedName], et_al: bool) -> str | None:
    rendered = [_bibtex_name(n) for n in names]
    rendered = [r for r in rendered if r and r != "{}"]
    if not rendered:
        return None
    if et_al:
        rendered.append("others")
    return " and ".join(rendered)


def _bibtex_entry(entry: _Entry) -> str:
    kind = _TYPES[entry.bib_type][1]
    fields: list[tuple[str, str]] = []  # (name, rendered value incl. braces)

    def put(name: str, value: str | None, *, verbatim: bool = False) -> None:
        if value:
            body = _bibtex_verbatim(value) if verbatim else _bibtex_escape(value)
            fields.append((name, "{" + body + "}"))

    def put_names(name: str, value: str | None) -> None:
        if value:
            fields.append((name, "{" + value + "}"))

    put_names("author", _bibtex_names(entry.authors, entry.authors_et_al))
    put_names("editor", _bibtex_names(entry.editors, entry.editors_et_al))
    put("title", entry.title)
    container, series, publisher = entry.container, entry.series, entry.publisher
    if entry.bib_type == "journal_article":
        put("journal", container)
    elif entry.bib_type in ("book_chapter", "conference_paper"):
        put("booktitle", container)
    elif entry.bib_type == "book":
        series = series or container
    elif entry.bib_type == "report":
        put("institution", publisher or container)
        if publisher:
            series = series or container
        publisher = None
    elif entry.bib_type == "thesis":
        put("school", publisher or container)
        publisher = None
    else:
        put("howpublished", container)
    put("series", series)
    put("edition", entry.edition)
    put("publisher", publisher)
    put("year", str(entry.year) if entry.year else None)
    if entry.month:
        fields.append(("month", _MONTHS[entry.month - 1]))
    put("volume", entry.volume)
    put("number", entry.issue)
    if entry.first_page:
        last = entry.last_page
        put("pages", f"{entry.first_page}--{last}" if last else entry.first_page)
    put("doi", entry.doi, verbatim=True)
    put("url", entry.url, verbatim=True)
    if entry.accessed and _ISO_DATE.match(entry.accessed):
        put("urldate", entry.accessed)
    put("pmid", entry.pmid)
    if entry.arxiv:
        put("eprint", entry.arxiv, verbatim=True)
        put("eprinttype", "arxiv")
        put("archiveprefix", "arXiv")
    put("version", entry.version)
    if entry.in_press:
        put("pubstate", "inpress")
    put("note", "; ".join(entry.notes))
    body = ",\n".join(f"  {name} = {value}" for name, value in fields)
    return f"@{kind}{{{entry.key},\n{body}\n}}" if body else f"@{kind}{{{entry.key}\n}}"


def to_bibtex(data: Any, *, prefer: Prefer = "printed", check_note: bool = True) -> str:
    """The export's references as a BibTeX file's text."""
    entries = _all_entries([_export_dict(data)], prefer=prefer, check_note=check_note)
    return _render_bibtex(entries)


def _render_bibtex(entries: list[_Entry]) -> str:
    return "\n\n".join(_bibtex_entry(e) for e in entries) + ("\n" if entries else "")


# ---------------------------------------------------------------------------
# RIS
# ---------------------------------------------------------------------------


def _ris_name(name: PrintedName) -> str | None:
    if name.literal or not name.family:
        return _text(name.literal or name.given)
    parts = [name.family, name.given or ""]
    if name.suffix:
        parts.append(name.suffix)
    return ", ".join(parts).rstrip(", ")


def _ris_record(entry: _Entry) -> list[str]:
    lines = [f"TY  - {_TYPES[entry.bib_type][2]}"]

    def put(tag: str, value: str | None) -> None:
        text = _text(value)
        if text:
            lines.append(f"{tag}  - {text}")

    for name in entry.authors:
        put("AU", _ris_name(name))
    for name in entry.editors:
        put("ED", _ris_name(name))
    put("TI", entry.title)
    put("T2", entry.container)
    put("T3", entry.series)
    put("PY", str(entry.year) if entry.year else None)
    if entry.year and entry.month:
        day = f"/{entry.day:02d}" if entry.day else ""
        put("DA", f"{entry.year}/{entry.month:02d}{day}")
    put("VL", entry.volume)
    put("IS", entry.issue)
    put("SP", entry.first_page)
    put("EP", entry.last_page)
    put("ET", entry.edition)
    put("PB", entry.publisher)
    put("DO", entry.doi)
    put("UR", entry.url)
    notes = list(entry.notes)
    if entry.in_press:
        notes.append("In press")
    if entry.pmid:
        notes.append(f"PMID: {entry.pmid}")
    if entry.arxiv:
        notes.append(f"arXiv: {entry.arxiv}")
    for note in notes:
        put("N1", note)
    lines.append("ER  - ")
    return lines


def to_ris(data: Any, *, prefer: Prefer = "printed", check_note: bool = True) -> str:
    """The export's references as an RIS file's text (CRLF line endings)."""
    entries = _all_entries([_export_dict(data)], prefer=prefer, check_note=check_note)
    return _render_ris(entries)


def _render_ris(entries: list[_Entry]) -> str:
    return "".join("\r\n".join(_ris_record(e)) + "\r\n\r\n" for e in entries)


# ---------------------------------------------------------------------------
# Many exports, files
# ---------------------------------------------------------------------------


def render_references(
    exports: Iterable[Any],
    format: str,
    *,
    prefer: Prefer = "printed",
    check_note: bool = True,
) -> tuple[str, int]:
    """``(text, references)``: the references of every export in *exports*
    (dicts or Results) as one file of *format*, with cite keys unique across
    them."""
    fmt = normalize_format(format)
    dicts = [_export_dict(e) for e in exports]
    entries = _all_entries(dicts, prefer=prefer, check_note=check_note)
    if fmt == "csl-json":
        items = [_csl_item(e) for e in entries]
        return json.dumps(items, ensure_ascii=False, indent=2) + "\n", len(entries)
    if fmt == "bibtex":
        return _render_bibtex(entries), len(entries)
    return _render_ris(entries), len(entries)


def _sources(sources: Any) -> list[Any]:
    """*sources* as a list: one export (dict, Result, path) or many."""
    if isinstance(sources, (Mapping, str, os.PathLike)) or hasattr(sources, "data"):
        return [sources]
    return list(sources)


def write_references(
    sources: Any,
    path: str | os.PathLike[str],
    *,
    format: str | None = None,
    prefer: Prefer = "printed",
    check_note: bool = True,
) -> ReferencesReport:
    """Write the references of *sources* to *path* as CSL-JSON, BibTeX or RIS.

    *sources* is one export or many: dicts, :class:`bibr.Result` objects or
    paths to export JSON files. Files that are not bibr exports of this major
    version, and the :class:`bibr.ChewFailure` slots of a batch, are skipped
    and listed in the report. *format* is ``"csl-json"``, ``"bibtex"`` or
    ``"ris"``; left out, the extension of *path* decides (.json, .bib, .ris).
    Cite keys are unique across the file. *prefer* and *check_note* are as
    for :func:`to_csl_json`. The file is replaced whole, or left as it was
    when writing fails.
    """
    from bibr.export.tables import _payloads

    target = Path(path)
    fmt = normalize_format(format) if format else format_for_path(target)
    if fmt is None:
        raise ValueError(
            f"cannot tell the format from {target.name!r}; pass format= "
            f"({', '.join(FORMATS)}) or use a .json, .bib or .ris file name"
        )
    _check_prefer(prefer)
    exports: list[Mapping[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for label, payload, reason in _payloads(_sources(sources)):
        if payload is None:
            skipped.append((label, reason or "skipped"))
        else:
            exports.append(payload)
    text, count = render_references(exports, fmt, prefer=prefer, check_note=check_note)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Created like any new file (mode 0o666 less the umask), then moved over the target.
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        # newline="" keeps RIS's CRLF line endings as they are on every platform.
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return ReferencesReport(
        path=target, format=fmt, references=count, papers=len(exports), skipped=tuple(skipped)
    )
