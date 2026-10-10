"""Check parsed reference locators against the matched record (verification).

A reference parser mislabels or drops the volume and page tokens far more
often than it misreads the title: on val120 (bibr 0.3.0) about a quarter of
the printed volumes were missing from ``bib`` while the matched record had the
right one. The record cannot simply be copied in, though: ``bib`` is what the
paper printed, and a record's pagination can legitimately differ from the
printed one (article numbers, e-locators, a reprint's page range).

So the record is used only as a key for *which printed token* is the volume,
issue or page. For each of those fields the parsed value is compared with a
confident match's value, and the outcome is one of:

- ``agree``: they are the same.
- ``filled``: the parser found nothing, and the record's value is printed in
  the reference string. It is written into ``bib``.
- ``corrected``: they differ, and the record's value is printed in the
  reference string. It replaces the parsed value.
- ``disagree``: they differ and the record's value is not printed. ``bib``
  keeps the parsed value; the disagreement is only reported.

A record value that another parsed field of the same reference already holds
(the issue, a page, the year) is never moved into this one: the parser saw
that token and gave it a role, and a record that disagrees about the role is
weaker evidence than the token's position. Operates on the export-dict shape,
like :mod:`bibr.enrich.consolidate`, so the same function runs at export and
on a saved result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bibr.enrich.consolidate import SERVICE_PRECEDENCE, record_consolidation

# Locator fields checked against the matched record, in the order a reference
# prints them.
VERIFIED_FIELDS = ("volume", "issue", "first_page", "last_page")

# Below this match score (0–1) the record is not trusted to say which token is
# which. A DOI lookup scores 1; a title search scores its title similarity.
MIN_MATCH_SCORE = 0.9

_DASHES = "-‐‑‒–—―"


def _norm(value) -> str:
    """Whitespace-free, lowercase, leading zeros dropped: ``"04"`` == ``"4"``."""
    return re.sub(r"\s+", "", str(value)).lower().lstrip("0") or "0"


def _same(parsed, record: str) -> bool:
    """Whether the parsed value already says what the record says. Besides
    equality, a record value that only adds to the parsed one ("50 Suppl",
    "10(1)" for a parsed "50", "10") and a parsed range whose either end is the
    record's value ("64-73" in ``last_page``) leave the parsed value standing:
    the record is not the better reading of the page there."""
    if _norm(parsed) == _norm(record):
        return True
    lead = re.match(r"\s*([0-9A-Za-z]+)", record)
    if lead and _norm(lead.group(1)) == _norm(parsed):
        return True
    ends = re.split(rf"\s*[{_DASHES}]\s*", str(parsed).strip())
    return len(ends) == 2 and any(_norm(end) == _norm(record) for end in ends if end)


def _printed(raw: str, value: str) -> bool:
    """Whether *value* is printed in *raw* as a whole token, not inside a longer
    number or word ("12" is not printed in "2012" or "e124")."""
    pattern = r"(?<![0-9A-Za-z])" + re.escape(value) + r"(?![0-9A-Za-z])"
    return re.search(pattern, raw, flags=re.IGNORECASE) is not None


def _printed_last_page(raw: str, first: str, last: str) -> str | None:
    """The last page as printed: in full, or abbreviated after the first page
    ("1267–89" for 1267–1289). ``None`` when neither form is printed."""
    if _printed(raw, last):
        return last
    for cut in range(1, len(last)):
        suffix = last[cut:]
        pattern = (
            r"(?<![0-9A-Za-z])"
            + re.escape(first)
            + rf"\s*[{_DASHES}]\s*"
            + re.escape(suffix)
            + r"(?![0-9A-Za-z])"
        )
        if last.startswith(first[: len(last) - len(suffix)]) and re.search(pattern, raw):
            return suffix
    return None


def _score(match: dict) -> float:
    """Match score on the 0–1 export scale (older exports scored 0–100)."""
    score = match.get("score")
    if not isinstance(score, (int, float)):
        return 0.0
    return score / 100 if score > 1 else float(score)


def _best_match(matches: list[dict]) -> dict | None:
    """The confident match to verify against: highest-precedence service first."""
    rank = {s: i for i, s in enumerate(SERVICE_PRECEDENCE)}
    confident = [m for m in matches if _score(m) >= MIN_MATCH_SCORE]
    confident.sort(key=lambda m: (rank.get(m.get("service"), len(rank)), -_score(m)))
    return confident[0] if confident else None


@dataclass
class VerificationReport:
    """Tally of field outcomes across one payload's references."""

    checked: int = 0
    outcomes: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, field_name: str, outcome: str) -> None:
        per_field = self.outcomes.setdefault(field_name, {})
        per_field[outcome] = per_field.get(outcome, 0) + 1


def _claimed(bib: dict, field_name: str, value: str) -> bool:
    """Whether another parsed field of this reference already holds *value*."""
    others = [bib.get(f) for f in (*VERIFIED_FIELDS, "year") if f != field_name]
    return any(o not in (None, "") and _norm(o) == _norm(value) for o in others)


def _verify_one(bib: dict, match: dict, raw: str, apply: bool) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for field_name in VERIFIED_FIELDS:
        record = match.get(field_name)
        if record in (None, ""):
            continue
        record = str(record).strip()
        parsed = bib.get(field_name)
        if parsed not in (None, "") and _same(parsed, record):
            outcomes[field_name] = "agree"
            continue
        printed: str | None = None
        if raw:
            if field_name == "last_page":
                first = str(match.get("first_page") or bib.get("first_page") or "").strip()
                printed = _printed_last_page(raw, first, record) if first else None
                if printed is None and _printed(raw, record):
                    printed = record
            elif _printed(raw, record):
                printed = record
        if printed is not None and parsed not in (None, "") and _norm(parsed) == _norm(printed):
            # An abbreviated last page the parser already read ("89" for 1289).
            outcomes[field_name] = "agree"
            continue
        if printed is None or _claimed(bib, field_name, printed):
            if parsed not in (None, ""):
                outcomes[field_name] = "disagree"
            continue
        outcomes[field_name] = "filled" if parsed in (None, "") else "corrected"
        if apply:
            bib[field_name] = printed
    return outcomes


def verify_bibs(data: dict, *, apply: bool = True) -> VerificationReport:
    """Check every ``bib`` row's locators against its confident ``bib_match``.

    With ``apply`` the ``filled`` and ``corrected`` values are written into
    ``bib``, and their field names are added to
    ``extraction.diagnostics.consolidation`` like any other consolidated field.
    Every outcome is recorded in ``extraction.diagnostics.verification``.
    The reference string is the ``text[]`` row the bib row's ``text_id``
    names; a row without one can still ``agree`` or ``disagree``, but never
    takes a value, since nothing shows the value was printed.
    """
    report = VerificationReport()
    raw_by_id = {
        t.get("text_id"): t.get("text") or "" for t in data.get("text") or [] if isinstance(t, dict)
    }
    matches_by_bib: dict[int, list[dict]] = {}
    for m in data.get("bib_match") or []:
        matches_by_bib.setdefault(m.get("bib_id"), []).append(m)

    rows: list[dict] = []
    taken_by_bib: dict[int, list[str]] = {}
    for bib in data.get("bib") or []:
        match = _best_match(matches_by_bib.get(bib.get("bib_id"), []))
        if match is None:
            continue
        raw = raw_by_id.get(bib.get("text_id"), "")
        outcomes = _verify_one(bib, match, raw, apply)
        if not outcomes:
            continue
        report.checked += 1
        for field_name, outcome in outcomes.items():
            report.add(field_name, outcome)
        rows.append({"bib_id": bib["bib_id"], "service": match.get("service"), **outcomes})
        taken = [f for f, o in outcomes.items() if o in ("filled", "corrected")]
        if apply and taken:
            taken_by_bib[bib["bib_id"]] = taken
    _record_verification(data, rows)
    if taken_by_bib:
        record_consolidation(data, taken_by_bib)
    return report


def _record_verification(data: dict, rows: list[dict]) -> None:
    extraction = data.get("extraction")
    if not isinstance(extraction, dict):
        return
    diagnostics = extraction.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = extraction["diagnostics"] = {}
    diagnostics["verification"] = rows
