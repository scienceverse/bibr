"""Compare the paper's title and authors with its own registry record.

With the paper's DOI looked up (full enrichment or the identity-only lookup),
``metadata_match`` holds the registry's record of the paper itself. Where
bibr's title and authors agree with that record they are very likely right;
where they disagree, either may be wrong. On a 120-paper development set
with this comparison, 91 papers had a record: all 78 agreeing titles and all
81 agreeing author lists were right, against 5 of 13 and 2 of 8 of the
disagreeing ones, so the 17 papers with a disagreement held every title or
author error. That makes agreement a per-field signal worth exporting.

The record never replaces a printed value: this module only reports. For
the title and for the author list, ``extraction.fields.<field>.registry``
becomes ``{service, outcome}``, with outcome ``agree`` or ``disagree``, or
``corrected`` when the value agrees and is a printed one the extraction
chose over its first reading because it agrees with the record
(``extraction.fields.<field>.source`` ``registry_printed``: the
FRONT_MATTER_REGISTRY_TITLE row, or ``llm_review``: the front-matter review's
answer), unless the field carries ``VAL_REGISTRY_DOI_CHANGED`` (the record
that chose it was another DOI's). It is absent when there is no confident
record or the record lacks the field.

The comparison is the one the validation scorer uses for title and authors,
ported here because the scorer is not part of the package:

- Titles are compared after case folding and replacing punctuation with
  spaces. A title that is a word-aligned prefix of the other and keeps at
  least 70% of its length also agrees (for scripts without spaces between
  words, the prefix must be at least six characters). The record's title is
  tried alone and followed by its subtitle, since Crossref keeps the
  subtitle separately.
- Authors are compared as whole names ("given family"), case- and
  accent-insensitive, so a different given/family split of the same printed
  name ("Leila Ben Ammar" + "Elgaaied" vs "Leila" + "Ben Ammar Elgaaied")
  still agrees. Names pair up greedily by Jaro-Winkler similarity (>= 0.85,
  in printed order or with each name's words sorted) with a guard against a
  dropped surname, and the lists agree when the F1 over pairs is >= 0.9.
  Group authors are left out when the record lists none, because bibr keeps
  only the record's people.

Operates on the export-dict shape, like :mod:`bibr.enrich.verify`, so it
runs at export and on a saved result alike.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from rapidfuzz.distance import JaroWinkler

from bibr.enrich.schemas import plain_text
from bibr.enrich.verify import _best_match

# Fields that get a registry outcome, keyed by their extraction.fields name.
REGISTRY_FIELDS = ("title", "author")

# ``extraction.fields.<field>.source`` of a printed value chosen because the
# record agrees with it: an agreeing one is reported ``corrected``.
RECORD_CHOSEN_SOURCES = frozenset({"registry_printed", "llm_review"})
# The field's issue when the record that chose its value was another DOI's
# and was dropped: the value then counts as extracted, not corrected.
_RECORD_DROPPED = "VAL_REGISTRY_DOI_CHANGED"

# A containment match must keep at least this share of the longer title.
_MIN_CONTAINMENT_LENGTH_RATIO = 0.7
# Scripts written without spaces between words (Han, kana, Thai): a title in
# them is one "word", so the two-word floor below becomes a character floor.
_SCRIPTIO_CONTINUA_RE = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u0e00-\u0e7f]"
)
_MIN_CONTINUA_PREFIX_CHARS = 6

# Two names pair up at this Jaro-Winkler similarity; the lists agree at this F1.
_NAME_THRESHOLD = 0.85
_AUTHORS_AGREE_F1 = 0.9


# --- titles -------------------------------------------------------------------


def _title_key(text: str) -> str:
    """Case-folded, punctuation replaced by spaces, whitespace collapsed."""
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _containment_eligible(short: str) -> bool:
    """Whether *short* is long enough for a prefix match to mean anything."""
    if len(short.split()) >= 2:
        return True
    return bool(_SCRIPTIO_CONTINUA_RE.search(short)) and len(short) >= _MIN_CONTINUA_PREFIX_CHARS


def _contained(a: str, b: str) -> bool:
    """The shorter key is a word-aligned prefix of the longer, and long enough."""
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not _containment_eligible(short):
        return False
    if len(short) < _MIN_CONTAINMENT_LENGTH_RATIO * len(long):
        return False
    return long.startswith(short) and (len(long) == len(short) or long[len(short)] == " ")


def titles_agree(printed: str, record: str) -> bool:
    """Whether a printed title and a record title say the same (see module doc)."""
    a, b = _title_key(printed), _title_key(record)
    if not a or not b:
        return False
    return a == b or _contained(a, b)


def _record_titles(match: dict) -> list[str]:
    """The record's title, alone and followed by its subtitle, markup removed."""
    title = plain_text(match.get("title")) or ""
    if not title.strip():
        return []
    titles = [title]
    subtitle = plain_text(match.get("subtitle")) or ""
    if subtitle.strip():
        titles.append(f"{title}: {subtitle}")
    return titles


# --- authors ------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Lowercase, accents removed, whitespace collapsed."""
    name = unicodedata.normalize("NFKD", name.lower())
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(name.split())


def _is_group(row: dict) -> bool:
    """A group or organization author: a ``literal`` name and no family name."""
    return bool(row.get("literal")) and not row.get("family")


def _full_name(row: dict) -> str:
    """A person's whole name ("given family"), or a group's literal name."""
    if _is_group(row):
        return _normalize_name(str(row["literal"]))
    parts = [str(row.get(key) or "") for key in ("given", "family")]
    return _normalize_name(" ".join(part for part in parts if part.strip()))


def _name_similarity(a: str, b: str) -> float:
    """Jaro-Winkler similarity, as printed or with each name's words sorted."""
    sim = JaroWinkler.normalized_similarity(a, b)
    a_sorted, b_sorted = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
    if a_sorted == a and b_sorted == b:
        return sim
    return max(sim, JaroWinkler.normalized_similarity(a_sorted, b_sorted))


def _token_covered(token: str, others: list[str]) -> bool:
    for other in others:
        if JaroWinkler.normalized_similarity(token, other) >= _NAME_THRESHOLD:
            return True
        if len(other) < len(token) and token.startswith(other):
            return True
        if len(token) < len(other) and other.startswith(token):
            return True
    return False


def _tokens_covered(printed: str, record: str) -> bool:
    """A printed name with fewer words than the record's must account for each
    of the record's words longer than an initial: Jaro-Winkler's prefix boost
    would otherwise pair "kaisin" with "kaisin yee"."""
    printed_tokens = re.sub(r"[^\w\s]", " ", printed).split()
    record_tokens = re.sub(r"[^\w\s]", " ", record).split()
    if len(printed_tokens) >= len(record_tokens):
        return True
    return all(len(tok) < 2 or _token_covered(tok, printed_tokens) for tok in record_tokens)


def authors_f1(printed: Iterable[str], record: Iterable[str]) -> float:
    """F1 of greedy one-to-one pairs between two lists of normalized names."""
    printed, record = [n for n in printed if n], [n for n in record if n]
    if not printed and not record:
        return 1.0
    if not printed or not record:
        return 0.0
    pairs = []
    for ri, record_name in enumerate(record):
        for pi, printed_name in enumerate(printed):
            sim = _name_similarity(printed_name, record_name)
            if sim >= _NAME_THRESHOLD and _tokens_covered(printed_name, record_name):
                pairs.append((sim, ri, pi))
    pairs.sort(key=lambda pair: -pair[0])
    used_record: set[int] = set()
    used_printed: set[int] = set()
    for _sim, ri, pi in pairs:
        if ri not in used_record and pi not in used_printed:
            used_record.add(ri)
            used_printed.add(pi)
    matched = len(used_record)
    if not matched:
        return 0.0
    precision, recall = matched / len(printed), matched / len(record)
    return 2 * precision * recall / (precision + recall)


def authors_agree(printed_rows: list[dict], record_rows: list[dict]) -> bool:
    """Whether the printed author list and the record's agree (see module doc).

    Both are lists of ``{given, family, literal}`` rows (``author[]`` and
    ``metadata_match[].author``).
    """
    record_has_groups = any(_is_group(row) for row in record_rows)
    printed = [_full_name(row) for row in printed_rows if record_has_groups or not _is_group(row)]
    record = [_full_name(row) for row in record_rows]
    return authors_f1(printed, record) >= _AUTHORS_AGREE_F1


# --- payload ------------------------------------------------------------------


def _same_doi(paper_doi, record_doi) -> bool:
    if not paper_doi or not record_doi:
        return True
    return str(paper_doi).strip().lower() == str(record_doi).strip().lower()


def _chosen_by_record(data: dict, name: str) -> bool:
    """Whether the field's exported value was chosen because the record agrees."""
    extraction = data.get("extraction")
    fields = extraction.get("fields") if isinstance(extraction, dict) else None
    record = fields.get(name) if isinstance(fields, dict) else None
    return (
        isinstance(record, dict)
        and record.get("source") in RECORD_CHOSEN_SOURCES
        and _RECORD_DROPPED not in (record.get("issues") or ())
    )


def _agreement(data: dict, name: str, agree: bool) -> str:
    if not agree:
        return "disagree"
    return "corrected" if _chosen_by_record(data, name) else "agree"


def verify_identity(data: dict, *, apply: bool = True) -> dict[str, str]:
    """Compare ``metadata.title`` and ``author[]`` with the paper's confident record.

    Returns the outcome per field (``{"title": "agree", "author": "disagree"}``,
    ``corrected`` for an agreeing value chosen because of the record);
    a field the record does not carry, or every field without a confident
    record of this paper's DOI, is left out. With ``apply`` each outcome is
    written to ``extraction.fields.<field>.registry`` as ``{service,
    outcome}``, replacing an earlier one, and a field without an outcome
    loses its stale ``registry``. Nothing printed is changed.
    """
    rows = [row for row in data.get("metadata_match") or [] if isinstance(row, dict)]
    match = _best_match(rows)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    if match is not None and not _same_doi(metadata.get("doi"), match.get("doi")):
        match = None

    outcomes: dict[str, str] = {}
    if match is not None:
        titles = _record_titles(match)
        if titles:
            printed_title = str(metadata.get("title") or "")
            agree = any(titles_agree(printed_title, title) for title in titles)
            outcomes["title"] = _agreement(data, "title", agree)
        record_authors = [row for row in match.get("author") or [] if isinstance(row, dict)]
        if record_authors:
            printed_authors = [row for row in data.get("author") or [] if isinstance(row, dict)]
            agree = authors_agree(printed_authors, record_authors)
            outcomes["author"] = _agreement(data, "author", agree)
    if apply:
        _record_registry(data, outcomes, None if match is None else match.get("service"))
    return outcomes


def _record_registry(data: dict, outcomes: dict[str, str], service) -> None:
    extraction = data.get("extraction")
    fields = extraction.get("fields") if isinstance(extraction, dict) else None
    if not isinstance(fields, dict):
        return
    for name in REGISTRY_FIELDS:
        record = fields.get(name)
        if not isinstance(record, dict):
            continue
        record.pop("registry", None)
        if name in outcomes:
            record["registry"] = {"service": service, "outcome": outcomes[name]}
