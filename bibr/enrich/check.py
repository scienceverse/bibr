"""Check each reference against the registry record bibr matched to it.

For every ``bib`` row, :func:`check_bibs` writes one ``bib_check`` row:

- ``matched``: a record was found, and what the reference printed agrees
  with it (no major difference, and at most one minor one besides the issue).
- ``fields_disagree``: a record was found, but the reference prints a
  different title, first author, year or DOI (a major difference), or
  several minor ones. A printed DOI that names another work lands here, and
  so does one whose record enrichment did not accept because it is dated
  more than a year from the printed year (lookup outcome ``doi_rejected``):
  its notes say whether the title differs too (the DOI names another work)
  or only the year.
- ``doi_unregistered``: doi.org says the printed DOI does not exist.
- ``not_found``: every lookup answered, and Crossref has no record of the
  work. That is not evidence the work does not exist: books, reports,
  theses, preprints, software and non-English venues are often missing
  from Crossref.
- ``unchecked``: nothing was compared. The DOI is registered with another
  agency (DataCite, mEDRA), Crossref has no record of a printed DOI that
  doi.org did not call unregistered (the work was not searched for by
  title then), a lookup failed or ran out of time, or the reference gave
  nothing to look up.

A matched record that has been retracted or withdrawn gets ``retraction``,
whatever its status. Each difference is a note ``{field, printed, record,
severity}``; ``printed`` is the reference's value and ``record`` the
registry's, as they are.

How fields are compared: a title is compared after folding case, diacritics
and punctuation, and a record title printed verbatim inside the reference
counts as agreement when it is long enough to identify a work (or the DOI or
volume and first page already identify it); the record's first author must
be named, and its other authors too unless the reference says "et al."; the
year may differ by one, or by two when the reference is the later (print
after online-first); a journal abbreviation agrees with the full name it
abbreviates (ISO 4). The volume, issue and page comparisons are
:mod:`bibr.enrich.verify`'s.

On top of that, differences that live Crossref answers for printed
references showed to be noise, not citing errors, are let through: a
subtitle one side leaves out, one word of a compound family name, an
acronym for a journal ("BMJ"), a page printed without its supplement or
electronic prefix ("19" for "S19"), and a record's placeholder "0" volume or
page. A rule ("——") stands for the previous reference's first author, the
reference string counts where the parsed author list lost a name, and a page
range is one difference. A surname spelled slightly differently ("Dehane"
for Dehaene) and an organization the record lists as its first author are
minor notes rather than a missing author.

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
# A printed word this similar (0-100) to a record surname of four or more
# letters is that name misspelled or misread, not another author.
NAME_CLOSE_MIN = 80

# Notice types that withdraw the work itself.
_RETRACTION_TYPES = ("retraction", "partial_retraction", "withdrawal", "removal")
# Record types whose year is that of the registered edition, often a reprint
# or e-book long after the printed one (Cohen's 1988 "Statistical Power
# Analysis" is registered as its 2013 edition): a year difference is minor.
_EDITION_DATED_BIB_TYPES = frozenset({"book", "book_chapter"})
# Locator fields: a disagreement is minor. A different issue alone is common
# (combined issues, supplements), so issue notes do not count towards the
# two minor differences that make a reference disagree; and a page range is
# one difference, however many of its ends differ (an online-first article
# cited by its own pages "1-14").
_LOCATOR_FIELDS = ("volume", "issue", "first_page", "last_page")
_UNCOUNTED_FIELDS = frozenset({"issue"})
_DIFFERENCE_OF = {"first_page": "pages", "last_page": "pages"}

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
# Words of a compound family name that a reference does not print alone.
_NAME_PARTICLES = frozenset(
    {"van", "von", "der", "den", "del", "della", "dos", "das", "ten", "ter", "bin", "ibn"}
)
# A supplement or electronic-locator prefix on a page: "S19", "e1".
_LOCATOR_PREFIX = re.compile(r"^[a-z]{1,2}(?=[0-9])")
# A reference that opens with a rule ("——, R. Conti, ...", "———. 1987.")
# repeats the previous reference's first author, after any number label.
_REPEATED_AUTHOR = re.compile(r"^\s*(?:\[\d+\]|\d+\.)?\s*[-\u2010-\u2015_]{2,}")
# Words a registry record sometimes holds as an author's family name.
_NOT_NAMES = frozenset({"and", "et", "al", "etal", "others"})
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


def _name_keys(name: str) -> list[str]:
    """The squashed name and, for a compound family name ("Venkateswar
    Reddy", "van der Berg"), each of its words other than particles: a
    reference may print any one of them."""
    whole = norm_squash(name)
    words = [norm_squash(w) for w in re.split(r"[\s\-‐]+", fold_diacritics(name or ""))]
    words = [w for w in words if len(w) >= 3 and w not in _NAME_PARTICLES]
    return [whole, *(w for w in words if w != whole)] if whole else []


def name_state(name: str, text: str) -> str | None:
    """Is *name* in *text*: ``named``, ``misspelled`` (a printed word close to
    it, as "Dehane" for Dehaene or "Irtel" for an OCR-damaged Birtel), or
    ``absent``? ``None`` when that cannot be told: a name with no Latin
    letters squashes to "", which every text "contains"."""
    keys = _name_keys(name)
    if not keys:
        return None
    squashed = norm_squash(text)
    if any(key in squashed for key in keys):
        return "named"
    words = re.findall(r"[a-z]+", fold_diacritics((text or "").lower()))
    for key in keys:
        if len(key) >= 4 and any(
            len(w) >= 3 and fuzz.ratio(w, key) >= NAME_CLOSE_MIN for w in words
        ):
            return "misspelled"
    return "absent"


def first_author_in_text(record_names: list[str], text: str) -> bool | None:
    """Is the record's first author named in *text*, if misspelled? ``None``
    when that cannot be told (see :func:`name_state`)."""
    names = [n for n in record_names if norm_squash(n)]
    if not names:
        return None
    return name_state(names[0], text) != "absent"


def authors_match(record_names: list[str], text: str, max_authors: int = MAX_AUTHORS) -> bool:
    """Are the record's first *max_authors* authors all named in *text*, if
    misspelled?"""
    names = [n for n in record_names if norm_squash(n)][:max_authors]
    return all(name_state(n, text) != "absent" for n in names)


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
    of the other, or is one the other's acronym ("BMJ", "PNAS")? An empty
    side is no disagreement."""
    ta, tb = _journal_tokens(a or ""), _journal_tokens(b or "")
    if not ta or not tb:
        return True
    if len(ta) > len(tb):
        ta, tb = tb, ta
    if len(ta) == 1 and len(tb) > 1 and ta[0] == "".join(t[0] for t in tb):
        return True
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


def _same_number(printed: object, record: object) -> bool:
    """Equal once a letter prefix is dropped: a supplement page ("S19") or an
    electronic locator ("e1") printed as its number."""

    def bare(value: object) -> str | None:
        text = _norm_number(value)
        return _norm_number(_LOCATOR_PREFIX.sub("", text)) if text else None

    a, b = bare(printed), bare(record)
    return a is not None and a == b


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


@dataclass(frozen=True)
class _Author:
    name: str  # family name, or an organization's whole name
    given: str = ""
    organization: bool = False


def _record_authors(match: dict) -> list[_Author]:
    """The record's authors that can be looked for in a reference."""
    authors = []
    for person in match.get("author") or []:
        if not isinstance(person, dict):
            continue
        family, literal = person.get("family"), person.get("literal")
        name = family or literal
        if not isinstance(name, str) or norm_squash(name) in ("", *_NOT_NAMES):
            continue
        given = person.get("given") if isinstance(person.get("given"), str) else ""
        authors.append(_Author(name.strip(), given.strip(), organization=not family))
    return authors


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


def repeats_previous_author(bib: dict, reference: str) -> bool:
    """Does the reference stand its first author in with a rule ("——")?"""
    return any(_REPEATED_AUTHOR.match(text or "") for text in (bib.get("authors"), reference))


def _field_notes(
    bib: dict,
    match: dict,
    reference: str,
    verification: dict | None,
    previous_authors: str = "",
) -> list[dict]:
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

    authors = _record_authors(match)
    printed_authors = bib.get("authors") or ""
    # The reference string is what was printed; the parsed author list can
    # have lost or mangled a name the string still holds.
    author_text = " ".join(t for t in (printed_authors, reference) if t)
    # A rule for the first author names the previous reference's.
    first_text = previous_authors if repeats_previous_author(bib, reference) else author_text
    if authors and author_text:
        lead = authors[0]
        first = name_state(lead.name, first_text) if first_text else "named"
        if first == "absent" and lead.given and name_state(lead.given, first_text) == "named":
            # The record swapped given and family name ("Isabelle" for Carrard).
            first = "misspelled"
        if first == "absent":
            # A record sometimes lists an institution where its authors
            # belong ("MGIMO University"): an organization missing from the
            # reference is weaker evidence than a person.
            severity = "minor" if lead.organization else "major"
            notes.append(_note("author", printed_authors, lead.name, severity))
        else:
            # "et al." can hide the record's other authors only when it has
            # more than two: a one- or two-author work is never cut short.
            excused = bool(_ET_AL.search(author_text)) and len(authors) > 2
            every_name = author_text if first_text == author_text else f"{first_text} {author_text}"
            people = [a.name for a in authors[1:] if not a.organization]
            if not excused and not authors_match(people, every_name, MAX_AUTHORS - 1):
                record_names = ", ".join(a.name for a in authors[:MAX_AUTHORS])
                notes.append(_note("author", printed_authors, record_names, "minor"))
            elif first == "misspelled":
                notes.append(_note("author", printed_authors, lead.name, "minor"))

    year, record_year = bib.get("year"), match.get("year")
    if isinstance(year, int) and isinstance(record_year, int) and year_disagrees(year, record_year):
        severity = "minor" if _edition_dated(bib, match.get("bib_type")) else "major"
        notes.append(_note("year", year, record_year, severity))

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
        # Some records hold "0" for a volume or page they do not know.
        if printed in (None, "") or record in (None, "") or _norm_number(record) == "0":
            continue
        if verification is not None and field_name in verification:
            differs = verification[field_name] == "disagree"
        else:
            differs = not _same(printed, str(record).strip())
        if differs and not _same_number(printed, record):
            notes.append(_note(field_name, printed, record, "minor"))
    return notes


def _status_of_notes(notes: list[dict]) -> str:
    """Any major difference, or two minor ones, and the reference disagrees."""
    counted = [n for n in notes if n["field"] not in _UNCOUNTED_FIELDS]
    differences = {_DIFFERENCE_OF.get(n["field"], n["field"]) for n in counted}
    if any(n["severity"] == "major" for n in counted) or len(differences) >= 2:
        return "fields_disagree"
    return "matched"


def _edition_dated(bib: dict, record_type: object) -> bool:
    """Is either side a book, whose record carries its registered edition's year?"""
    return (
        record_type in _EDITION_DATED_BIB_TYPES or bib.get("bib_type") in _EDITION_DATED_BIB_TYPES
    )


def _rejected_doi_notes(bib: dict, lookup: dict, reference: str) -> list[dict]:
    """The differences between a reference and the record its printed DOI
    resolved to, which enrichment did not accept because it is dated more than
    a year from the printed year (lookup outcome ``doi_rejected``).

    A year note always, with both years: minor when the record is a book (its
    year is the registered edition's) or dated two years before the printed
    year (an online-first date), major otherwise. When the record's title is
    another work's, the DOI note comes first: the printed DOI names that work.
    When the titles agree, the DOI is the cited work's and the year is what
    differs, so there is no DOI note."""
    title, record_title = bib.get("title"), lookup.get("record_title")
    title_note = None
    if title and record_title:
        score = title_score(title, record_title, reference, identified=False)
        if score is not None and score < TITLE_MINOR_BELOW:
            severity = "major" if score < TITLE_MAJOR_BELOW else "minor"
            title_note = _note("title", title, record_title, severity)
    notes = []
    if title_note is not None and title_note["severity"] == "major":
        notes.append(_note("doi", bib.get("doi"), None, "major"))
    if title_note is not None:
        notes.append(title_note)
    year, record_year = bib.get("year"), lookup.get("record_year")
    online_first = (
        isinstance(year, int)
        and isinstance(record_year, int)
        and not year_disagrees(year, record_year)
    )
    tolerated = online_first or _edition_dated(bib, lookup.get("record_bib_type"))
    notes.append(_note("year", year, record_year, "minor" if tolerated else "major"))
    return notes


def _unmatched(bib: dict, lookup: dict | None, reference: str = "") -> tuple[str, list[dict]]:
    lookup = lookup or {}
    outcome = lookup.get("outcome")
    if outcome == "doi_404":
        if lookup.get("registration_agency") == "none":
            return "doi_unregistered", [_note("doi", bib.get("doi"), None, "major")]
        # Registered elsewhere (DataCite, mEDRA), registered with Crossref but
        # not served by its API (indexing lag), or doi.org not asked or not
        # answering. In every case Crossref was not searched for the work by
        # its title, so nothing says Crossref lacks it.
        return "unchecked", []
    if outcome == "doi_rejected":
        # The printed DOI resolves, but to a record of another year.
        return "fields_disagree", _rejected_doi_notes(bib, lookup, reference)
    if outcome == "clean_miss":
        return "not_found", []
    return "unchecked", []


def check_bib(
    bib: dict,
    matches: list[dict],
    lookup: dict | None,
    reference: str = "",
    verification: dict | None = None,
    previous_authors: str = "",
) -> dict:
    """One ``bib_check`` row for *bib*. *previous_authors* is the author list
    of the reference before it, which a rule ("——") stands in for."""
    match = _record_for(matches)
    if match is None:
        status, notes = _unmatched(bib, lookup, reference)
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
    notes = _field_notes(bib, match, reference, verification, previous_authors)
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
    previous_authors = ""
    for bib in data.get("bib") or []:
        if not isinstance(bib, dict) or bib.get("bib_id") is None:
            continue
        reference = text_by_id.get(bib.get("text_id"), "")
        row = check_bib(
            bib,
            matches_by_bib.get(bib["bib_id"], []),
            lookups.get(bib["bib_id"]),
            reference,
            verification.get(bib["bib_id"]),
            previous_authors,
        )
        if not repeats_previous_author(bib, reference):
            previous_authors = bib.get("authors") or reference
        report.statuses[row["status"]] += 1
        report.retracted += row["retraction"] is not None
        rows.append(row)
    data["bib_check"] = rows
    return report
