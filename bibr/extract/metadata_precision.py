"""Conservative repairs using the selected paper's printed front matter.

These guards refine existing values. They neither discover new authors/dates nor
infer surname boundaries from token position alone.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import TYPE_CHECKING

from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.models import PaperAuthor

_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTHS = {name: i for i, name in enumerate(_MONTH_NAMES, 1)}
_MONTHS.update({name[:3]: i for i, name in enumerate(_MONTH_NAMES, 1)})
_MONTHS["sept"] = 9
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DATE_RE = re.compile(
    rf"(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})"
    rf"|(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH})\s*,?\s*"
    rf"(?P<year>\d{{4}})"
    rf"|(?P<month_first>{_MONTH})\s+(?P<day_second>\d{{1,2}})(?:st|nd|rd|th)?"
    rf",?\s+(?P<year_last>\d{{4}})",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(
    r"\b(?:(?:first\s+|online\s+)?published(?:\s+online)?|publication\s+date)"
    r"\s*:?\s*(?:on\s+)?",
    re.IGNORECASE,
)
_HISTORY_RE = re.compile(r"\b(?:received|accepted|revised)\s*:?\s*", re.IGNORECASE)
# Start of a date-like tail after a publication label: a digit (ISO or
# day-first date) or a month name. Anything else ('by Elsevier', 'under a
# CC BY licence') is publisher/licence boilerplate, not a date the label
# failed to parse, so the label is skipped rather than treated as ambiguity.
_DATE_LIKE_RE = re.compile(rf"(?:\d|{_MONTH})", re.IGNORECASE)
_ORGANIZATION_RE = re.compile(
    r"\b(?:consortium|collaboration|group|team|committee|society|association|university|"
    r"institute|department|centre|center|network|project|study|research)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"(?<![\w.+-])([\w.+-]+@[\w.-]+\.[a-z]{2,})(?![\w.-])", re.I)


def _date_after_label(text: str) -> str | None:
    match = _DATE_RE.match(text)
    if match is None:
        return None
    following = text[match.end() : match.end() + 1]
    if following and (following.isalnum() or following in "/-"):
        return None
    try:
        if match["iso"]:
            return date.fromisoformat(match["iso"]).isoformat()
        month = match["month"] or match["month_first"]
        return date(
            int(match["year"] or match["year_last"]),
            _MONTHS[month.casefold().rstrip(".")],
            int(match["day"] or match["day_second"]),
        ).isoformat()
    except ValueError:
        return None


def _metadata_prefix(prefix: str) -> bool:
    """A standalone label or a continuation of a printed article-history row."""
    prefix = prefix.strip()
    while prefix:
        for history in _HISTORY_RE.finditer(prefix):
            tail = prefix[history.end() :].rstrip(" /;|")
            if _DATE_RE.fullmatch(tail) and _date_after_label(tail):
                return True
        previous_labels = list(_LABEL_RE.finditer(prefix))
        if not previous_labels:
            return False
        previous = previous_labels[-1]
        tail = prefix[previous.end() :].rstrip(" /;|")
        if not (_DATE_RE.fullmatch(tail) and _date_after_label(tail)):
            return False
        prefix = prefix[: previous.start()].strip()
    return True


def refine_publication_date(published: str | None, source_text: str) -> str | None:
    """Refine a year/month only from one compatible, labelled full date.

    The caller supplies only the selected record, never appended tables,
    references, page furniture or another record. Ambiguous/invalid publication
    dates, prose mentions, absent values and already full dates remain unchanged.
    """
    if not published or not re.fullmatch(r"\d{4}(?:-\d{2})?", published):
        return published
    found: set[str] = set()
    for label in _LABEL_RE.finditer(source_text):
        line_start = source_text.rfind("\n", 0, label.start()) + 1
        if not _metadata_prefix(source_text[line_start : label.start()]):
            continue
        tail = source_text[label.end() :]
        value = _date_after_label(tail)
        if value is None:
            # Only a date-like tail that failed to parse marks the record
            # ambiguous. Publisher/licence boilerplate ('Published by ...',
            # 'Published under ...') carries no date, so skip the label and
            # keep scanning for a labelled date elsewhere in the block.
            if _DATE_LIKE_RE.match(tail.lstrip()):
                return published
            continue
        found.add(value)
    if len(found) == 1:
        value = next(iter(found))
        if value.startswith(published + "-"):
            return value
    return published


def _fold(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )


def _letters(value: str) -> str:
    return "".join(char for char in _fold(value) if char.isalpha())


def _email_supports(
    given: str, family: str, email: str, *, require_all_initials: bool = False
) -> bool:
    local = _fold(email.split("@", 1)[0])
    if not re.fullmatch(r"[a-z]+(?:[._-][a-z]+)+", local):
        return False
    parts = re.split(r"[._-]", local)
    given_tokens = [_letters(token) for token in given.split()]
    if not given_tokens or not all(given_tokens):
        return False
    initials = {"".join(token[0] for token in given_tokens)}
    if not require_all_initials:
        initials.add(given_tokens[0][0])
    surname = _letters(family)
    # Initial-only family fields and full-name email usernames cannot establish
    # which side denotes the surname. Require a distinct abbreviated given side.
    if len(surname) < 3:
        return False
    return (parts[0] in initials and "".join(parts[1:]) == surname) or (
        parts[-1] in initials and "".join(parts[:-1]) == surname
    )


def _partitions(name: str, email: str) -> set[tuple[str, str]]:
    tokens = name.split()
    if not 2 <= len(tokens) <= 8 or any(
        not _letters(token) or any(not (c.isalpha() or c in "-'’.") for c in token)
        for token in tokens
    ):
        return set()
    matches: set[tuple[str, str]] = set()
    for boundary in range(1, len(tokens)):
        left, right = " ".join(tokens[:boundary]), " ".join(tokens[boundary:])
        for given, family in ((left, right), (right, left)):
            # A username can omit a compound surname or particle. Matching
            # every given initial prevents moving those omitted words into
            # given merely to make the remaining surname fit the email.
            if _email_supports(given, family, email, require_all_initials=True):
                matches.add((given, family))
    return matches


def repair_author_partitions(
    authors: list[PaperAuthor],
    source_text: str,
) -> tuple[ValidationIssue, ...]:
    """Repair whole-given personal names only with unique printed email evidence.

    Preserve spelling and every name token; consider both printed name orders.
    Do not change a complete partition or infer a surname from the final token.
    All author fields except given/family remain untouched.
    """
    printed_emails = {match[1].casefold() for match in _EMAIL_RE.finditer(source_text)}
    folded_source = _fold(source_text)
    issues = []
    for author in authors:
        if (
            author.family
            or not author.given
            or not author.email
            or "organization" in author.role
            or _ORGANIZATION_RE.search(author.given)
            or author.email.casefold() not in printed_emails
        ):
            continue
        name_pattern = (
            r"(?<!\w)"
            + r"\s+".join(re.escape(_fold(token)) for token in author.given.split())
            + r"(?!\w)"
        )
        if not re.search(name_pattern, folded_source):
            continue
        choices = _partitions(author.given, author.email)
        if len(choices) != 1:
            continue
        if any(
            other is not author
            and (
                (other.email or "").casefold() == author.email.casefold()
                or _email_supports(other.given, other.family, author.email)
                or (not other.family and _partitions(other.given, author.email))
            )
            for other in authors
        ):
            continue
        author.given, author.family = next(iter(choices))
        issues.append(
            ValidationIssue(
                code="VAL_AUTHOR_PARTITION_REPAIRED",
                severity=IssueSeverity.WARNING,
                message="Partitioned a printed full author name using its printed initial/surname email",
                origin_stage="extract",
                evidence_ids=(f"author:{author.author_id}", "reason:printed_name_and_email"),
            )
        )
    return tuple(issues)
