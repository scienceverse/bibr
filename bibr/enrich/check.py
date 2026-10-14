"""Check each reference against the registry record bibr matched to it.

For every ``bib`` row, :func:`check_bibs` writes one ``bib_check`` row:

- ``matched``: a record was found, and what the reference printed agrees
  with it (at most one minor difference).
- ``fields_disagree``: a record was found, but the reference prints a
  different title, first author, year or DOI (a major difference), or
  several minor ones. A printed DOI that names another work lands here.
- ``doi_unregistered``: doi.org says the printed DOI does not exist.
- ``not_found``: every lookup answered, and Crossref has no record of the
  work. That is not evidence the work does not exist: books, reports,
  theses, preprints, software and non-English venues are often missing
  from Crossref.
- ``unchecked``: nothing was compared. The DOI is registered with another
  agency (DataCite, mEDRA), a lookup failed or ran out of time, or the
  reference gave nothing to look up.

A matched record that has been retracted or withdrawn gets ``retraction``,
whatever its status. Each difference is a note ``{field, printed, record,
severity}``; ``printed`` is the reference's value and ``record`` the
registry's, as they are.

The comparators follow the Scienceverse reference checker: a title is
compared after folding case, diacritics and punctuation, and a record title
printed verbatim inside the reference counts as agreement when it is long
enough to identify a work (or the DOI or volume and first page already
identify it); the record's first author must be named, and its other
authors too unless the reference says "et al."; the year may differ by one,
or by two when the reference is the later (print after online-first); a
journal abbreviation agrees with the full name it abbreviates (ISO 4). The
volume, issue and page comparisons are :mod:`bibr.enrich.verify`'s.

Runs on the export-dict shape after :func:`bibr.enrich.verify.verify_bibs`,
whose ``extraction.diagnostics.verification`` rows it reads.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from bibr.enrich.schemas import strip_notice_prefix

CHECK_STATUSES = ("matched", "fields_disagree", "doi_unregistered", "not_found", "unchecked")

# A title this similar (0-100) or more agrees; below the minor bound it is
# noted, and below the major bound it names another work.
TITLE_MAJOR_BELOW = 70
TITLE_MINOR_BELOW = 92
# A record title shorter than this (letters and digits) does not identify a
# work on its own ("Power Posing"), so it is not credited for being printed
# inside the reference unless the DOI or the locator already identifies it.
CONTAINMENT_MIN_CHARS = 20
# Printed and record years this far apart agree.
YEAR_TOLERANCE = 1
# Record authors compared with the printed author list.
MAX_AUTHORS = 6

# Notice types that withdraw the work itself.
_RETRACTION_TYPES = ("retraction", "partial_retraction", "withdrawal", "removal")
# Record types whose year is that of the registered edition (see
# bibr.enrich.references._EDITION_DATED_TYPES): a year difference is minor.
_EDITION_DATED_BIB_TYPES = frozenset({"book", "book_chapter"})
# Locator fields: a disagreement is minor. A different issue alone is common
# (combined issues, supplements), so issue notes do not count towards the
# two minor notes that make a reference disagree.
_LOCATOR_FIELDS = ("volume", "issue", "first_page", "last_page")
_UNCOUNTED_FIELDS = frozenset({"issue"})

_ET_AL = re.compile(r"\bet\.?\s*al\b", re.IGNORECASE)
# Where a subtitle starts: a colon, a question or exclamation mark followed by
# more title, or a spaced dash.
_SUBTITLE_SPLIT = re.compile(r"\s*:\s+|(?<=[?!])\s+|\s+[-–—]\s+")
_TAGS = re.compile(r"<sup>.*?</sup>|</?[a-z]+>", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]")
# Characters NFKD does not decompose to ASCII.
_SPECIAL = str.maketrans(
    {
        "ß": "ss",
        "æ": "ae",
        "œ": "oe",
        "ø": "o",
        "đ": "d",
        "ł": "l",
        "ı": "i",
        "Æ": "AE",
        "Œ": "OE",
        "Ø": "O",
        "Đ": "D",
        "Ł": "L",
    }
)
_JOURNAL_STOP = frozenset({"of", "and", "the", "for", "in", "on", "a", "an", "de"})
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------


def fold_diacritics(text: str) -> str:
    text = (text or "").translate(_SPECIAL)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def norm_squash(text: str | None) -> str:
    """Lowercase letters and digits only, diacritics folded and tags dropped:
    dashes, colons, line-break splits and footnote markers disappear."""
    text = _TAGS.sub("", (text or "").lower())
    return _NON_ALNUM.sub("", fold_diacritics(text))


def _words(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", fold_diacritics((text or "").lower())))


def title_similarity(printed: str | None, record: str | None) -> float | None:
    """0-100, or ``None`` when either side is empty."""
    a, b = norm_squash(printed), norm_squash(record)
    if not a or not b:
        return None
    return float(max(fuzz.ratio(a, b), fuzz.token_sort_ratio(_words(printed), _words(record))))


def containment_usable(squashed_record_title: str, identified: bool) -> bool:
    """Is the record title long enough to identify a work by being printed in
    the reference, or is the work identified another way?"""
    return len(squashed_record_title) >= CONTAINMENT_MIN_CHARS or identified


def main_title(title: str | None) -> str | None:
    """The title before its subtitle ("Power posing" of "Power posing: Brief
    nonverbal displays ..."), or ``None`` when it has no subtitle."""
    head = _SUBTITLE_SPLIT.split(title or "", maxsplit=1)[0].strip()
    return head if head and head != (title or "").strip() else None


def title_score(
    printed: str | None, record: str | None, reference: str, *, identified: bool
) -> float | None:
    """How well the printed title agrees with the record's (0-100).

    The record title printed verbatim in the reference scores 100: the parser
    cut the title short or ran it into the venue, which is extraction noise,
    not a different work. A title with its subtitle agrees with the same
    title without it: Crossref deposits many subtitles apart from the title,
    and references drop them. ``None`` when there is nothing to compare.
    """
    if not record:
        return None
    record = strip_notice_prefix(record)
    squashed = norm_squash(record)
    printed_whole = squashed in norm_squash(reference) or squashed in norm_squash(printed)
    if squashed and containment_usable(squashed, identified) and printed_whole:
        return 100.0
    pairs = ((printed, record), (main_title(printed), record), (printed, main_title(record)))
    scores = [title_similarity(a, b) for a, b in pairs if a and b]
    scores = [s for s in scores if s is not None]
    return max(scores) if scores else None


def first_author_in_text(record_names: list[str], text: str) -> bool | None:
    """Is the record's first author named in *text*? ``None`` when that cannot
    be told: a name with no Latin letters squashes to "", which every text
    "contains"."""
    names = [n for n in record_names if norm_squash(n)]
    if not names:
        return None
    return norm_squash(names[0]) in norm_squash(text)


def authors_match(record_names: list[str], text: str, max_authors: int = MAX_AUTHORS) -> bool:
    """Are the record's first *max_authors* authors all named in *text*?"""
    names = [n for n in record_names if norm_squash(n)][:max_authors]
    squashed = norm_squash(text)
    return all(norm_squash(n) in squashed for n in names)


def year_disagrees(printed: int | None, record: int | None) -> bool:
    """More than a year apart, except a reference dated exactly two years after
    the record: the print-issue year of a record indexed under its
    online-first year."""
    if printed is None or record is None:
        return False
    delta = printed - record
    if abs(delta) <= YEAR_TOLERANCE:
        return False
    return delta != YEAR_TOLERANCE + 1


def _journal_tokens(text: str) -> list[str]:
    text = fold_diacritics((text or "").lower()).replace("&amp;", " and ").replace("&", " and ")
    return [t for t in re.split(r"[^a-z]+", text) if t and t not in _JOURNAL_STOP]


def _token_abbreviates(short: str, long: str) -> bool:
    """Could *short* be the ISO 4 abbreviation of *long*: a truncation ("Biol."),
    or a contraction keeping the first letter and the order of the rest
    ("Natl.")? A contraction needs three letters, else an initial would match
    any word."""
    if long.startswith(short):
        return True
    if len(short) < 3 or short[0] != long[0]:
        return False
    rest = iter(long)
    return all(ch in rest for ch in short)


def journal_coherent(a: str | None, b: str | None) -> bool:
    """Do the (possibly abbreviated) words of one name match, in order, words
    of the other? An empty side is no disagreement."""
    ta, tb = _journal_tokens(a or ""), _journal_tokens(b or "")
    if not ta or not tb:
        return True
    if len(ta) > len(tb):
        ta, tb = tb, ta
    j = 0
    for token in ta:
        while j < len(tb):
            candidate = tb[j]
            j += 1
            if _token_abbreviates(token, candidate) or _token_abbreviates(candidate, token):
                break
        else:
            return False
    return True


def _norm_number(value: object) -> str | None:
    text = re.sub(r"\s+", "", str(value)).lower() if value is not None else ""
    return (text.lstrip("0") or "0") if text else None


def locator_identifies(bib: dict, record: dict) -> bool:
    """Volume and first page both printed, both on the record, and equal: close
    to a primary key for a journal article."""
    pairs = [
        (_norm_number(bib.get(f)), _norm_number(record.get(f))) for f in ("volume", "first_page")
    ]
    return all(a is not None and a == b for a, b in pairs)


def _bare_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _DOI_PREFIX.sub("", value.strip()).rstrip(".").casefold()


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


@dataclass
class CheckReport:
    """Tally of one payload's statuses."""

    statuses: Counter = field(default_factory=Counter)
    retracted: int = 0


def _note(field_name: str, printed, record, severity: str) -> dict:
    def text(value) -> str | None:
        if value is None or value == "":
            return None
        return str(value)

    return {
        "field": field_name,
        "printed": text(printed),
        "record": text(record),
        "severity": severity,
    }


def _record_names(match: dict) -> list[str]:
    names = []
    for person in match.get("author") or []:
        if isinstance(person, dict):
            name = person.get("family") or person.get("literal")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def _score(match: dict) -> float:
    score = match.get("score")
    if not isinstance(score, (int, float)):
        return 0.0
    return score / 100 if score > 1 else float(score)


def _record_for(matches: list[dict]) -> dict | None:
    """The record to check against: the highest-precedence service (Crossref),
    then the best score."""
    from bibr.enrich.consolidate import SERVICE_PRECEDENCE

    if not matches:
        return None
    rank = {s: i for i, s in enumerate(SERVICE_PRECEDENCE)}
    return sorted(matches, key=lambda m: (rank.get(m.get("service"), len(rank)), -_score(m)))[0]


def _retraction(match: dict) -> dict | None:
    for notice in match.get("updated_by") or []:
        if isinstance(notice, dict) and notice.get("type") in _RETRACTION_TYPES:
            return dict(notice)
    return None


def _field_notes(bib: dict, match: dict, reference: str, verification: dict | None) -> list[dict]:
    notes: list[dict] = []
    printed_doi, record_doi = _bare_doi(bib.get("doi")), _bare_doi(match.get("doi"))
    by_doi = printed_doi is not None and printed_doi == record_doi
    identified = by_doi or locator_identifies(bib, match)

    # A printed DOI that is not the record's names another work.
    if printed_doi is not None and record_doi is not None and printed_doi != record_doi:
        notes.append(_note("doi", bib.get("doi"), match.get("doi"), "major"))

    title = bib.get("title")
    if title:
        score = title_score(title, match.get("title"), reference, identified=identified)
        if score is not None and score < TITLE_MINOR_BELOW:
            severity = "major" if score < TITLE_MAJOR_BELOW else "minor"
            notes.append(_note("title", title, match.get("title"), severity))

    names = _record_names(match)
    printed_authors = bib.get("authors") or ""
    author_text = printed_authors or reference
    if names and author_text:
        first = first_author_in_text(names, author_text)
        if first is False:
            notes.append(_note("author", printed_authors, names[0], "major"))
        elif first is True:
            # "et al." can hide the record's other authors only when it has
            # more than two: a one- or two-author work is never cut short.
            excused = bool(_ET_AL.search(author_text)) and len(names) > 2
            if not excused and not authors_match(names, author_text):
                notes.append(
                    _note("author", printed_authors, ", ".join(names[:MAX_AUTHORS]), "minor")
                )

    year, record_year = bib.get("year"), match.get("year")
    if isinstance(year, int) and isinstance(record_year, int) and year_disagrees(year, record_year):
        edition_dated = (
            match.get("bib_type") in _EDITION_DATED_BIB_TYPES
            or bib.get("bib_type") in _EDITION_DATED_BIB_TYPES
        )
        notes.append(_note("year", year, record_year, "minor" if edition_dated else "major"))

    container, record_container = bib.get("container"), match.get("container")
    if container and record_container and not journal_coherent(container, record_container):
        notes.append(_note("container", container, record_container, "minor"))

    notes.extend(_locator_notes(bib, match, verification))
    return notes


def _locator_notes(bib: dict, match: dict, verification: dict | None) -> list[dict]:
    """Volume, issue and pages. With a verification row, its outcome decides:
    only ``disagree`` is a difference (``filled`` and ``corrected`` took the
    record's value from the printed string). Without one (the match scored
    below verification's bar, or verification is off), the values compare
    directly."""
    from bibr.enrich.verify import _same

    notes = []
    for field_name in _LOCATOR_FIELDS:
        printed, record = bib.get(field_name), match.get(field_name)
        if printed in (None, "") or record in (None, ""):
            continue
        if verification is not None and field_name in verification:
            differs = verification[field_name] == "disagree"
        else:
            differs = not _same(printed, str(record).strip())
        if differs:
            notes.append(_note(field_name, printed, record, "minor"))
    return notes


def _status_of_notes(notes: list[dict]) -> str:
    """Any major difference, or two minor ones, and the reference disagrees."""
    counted = [n for n in notes if n["field"] not in _UNCOUNTED_FIELDS]
    if any(n["severity"] == "major" for n in counted) or len(counted) >= 2:
        return "fields_disagree"
    return "matched"


def _unmatched(bib: dict, lookup: dict | None) -> tuple[str, list[dict]]:
    outcome = (lookup or {}).get("outcome")
    if outcome == "doi_404":
        agency = (lookup or {}).get("registration_agency")
        if agency == "none":
            return "doi_unregistered", [_note("doi", bib.get("doi"), None, "major")]
        if agency is not None and agency != "crossref":
            # Registered elsewhere (DataCite, mEDRA): bibr checks Crossref only.
            return "unchecked", []
        return "not_found", []
    if outcome == "doi_rejected":
        # The printed DOI resolves, but to a work dated more than a year away.
        return "fields_disagree", [_note("doi", bib.get("doi"), None, "major")]
    if outcome == "clean_miss":
        return "not_found", []
    return "unchecked", []


def check_bib(
    bib: dict,
    matches: list[dict],
    lookup: dict | None,
    reference: str = "",
    verification: dict | None = None,
) -> dict:
    """One ``bib_check`` row for *bib*."""
    match = _record_for(matches)
    if match is None:
        status, notes = _unmatched(bib, lookup)
        return {
            "bib_id": bib["bib_id"],
            "status": status,
            "service": None,
            "notes": notes,
            "retraction": None,
        }
    if verification is not None and verification.get("service") != match.get("service"):
        # Verification checked another record (this one scored below its bar).
        verification = None
    notes = _field_notes(bib, match, reference, verification)
    return {
        "bib_id": bib["bib_id"],
        "status": _status_of_notes(notes),
        "service": match.get("service"),
        "notes": notes,
        "retraction": _retraction(match),
    }


def check_bibs(data: dict) -> CheckReport:
    """Write ``data["bib_check"]``: one row per ``bib`` row (see the module
    docstring). Reads ``bib_match``, ``extraction.enrichment.lookups``,
    ``extraction.diagnostics.verification`` and the reference strings in
    ``text``."""
    extraction = data.get("extraction") if isinstance(data.get("extraction"), dict) else {}
    enrichment = extraction.get("enrichment") if isinstance(extraction, dict) else None
    lookups = {
        row.get("bib_id"): row
        for row in (enrichment or {}).get("lookups") or []
        if isinstance(row, dict)
    }
    diagnostics = extraction.get("diagnostics") if isinstance(extraction, dict) else None
    verification = {
        row.get("bib_id"): row
        for row in (diagnostics or {}).get("verification") or []
        if isinstance(row, dict)
    }
    text_by_id = {
        row.get("text_id"): row.get("text") or ""
        for row in data.get("text") or []
        if isinstance(row, dict)
    }
    matches_by_bib: dict[object, list[dict]] = {}
    for row in data.get("bib_match") or []:
        if isinstance(row, dict):
            matches_by_bib.setdefault(row.get("bib_id"), []).append(row)

    report = CheckReport()
    rows = []
    for bib in data.get("bib") or []:
        if not isinstance(bib, dict) or bib.get("bib_id") is None:
            continue
        row = check_bib(
            bib,
            matches_by_bib.get(bib["bib_id"], []),
            lookups.get(bib["bib_id"]),
            text_by_id.get(bib.get("text_id"), ""),
            verification.get(bib["bib_id"]),
        )
        report.statuses[row["status"]] += 1
        report.retracted += row["retraction"] is not None
        rows.append(row)
    data["bib_check"] = rows
    return report
