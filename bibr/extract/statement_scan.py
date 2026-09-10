"""Lexical-anchor fallback for research-integrity statements.

:func:`copy_integrity_statements` (research_integrity.py) is a pure copy that
only fires when the section classifier tagged a section as FUNDING / COI /
ETHICS / OPEN_DATA. Real papers routinely bury these statements inside a
mis-typed section — run-in bold labels ("Declaration of conflicting interests:")
inside a Conclusions/Discussion block, an ethics sentence in the Methods, a
funding sentence in the Acknowledgments. This module scans every sentence for
word-boundary anchor phrases and recovers the verbatim statement for any field
the copy pass left ``None``.

Pure lexical (no LLM), so it is safe to run alongside the copy pass, including
under ``--no-llm``. Each recovered field appends a
``STATEMENT_LEXICAL_FALLBACK: <field>`` warning to ``contents.processing_warnings``
(surfaced onto ``Paper.processing_warnings`` in post_parse).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import cache
from typing import TYPE_CHECKING

from bibr.paper_contents import CanonicalSection

if TYPE_CHECKING:
    from bibr.models import PaperMetadata
    from bibr.paper_contents import PaperContents


def _compile(*phrases: str) -> tuple[re.Pattern[str], ...]:
    """Compile each anchor phrase with a leading word boundary (case-insensitive).

    The leading ``\\b`` keeps substrings from matching mid-word (e.g.
    "unsupported by" must not trip the "supported by" funding anchor)."""
    return tuple(re.compile(r"\b" + p, re.IGNORECASE) for p in phrases)


_DATA_DECLARATION_PATTERNS = (
    r"(?:data|datasets?|code|materials?)\s+availability\s*:\s*.{0,120}"
    r"\b(?:available|accessible|deposited|archived|repository|request)\b",
    r"(?:data|datasets?|code|materials?)\b.{0,80}"
    r"\b(?:generated|analy[sz]ed|collected|produced|used|underlying|supporting)\b"
    r".{0,160}\b(?:available|accessible|deposited|archived)\b",
    r"(?:data|datasets?|code|materials?)\s+"
    r"(?:are|is|were|was|will\s+be|can\s+be)\s+"
    r"(?:(?:openly|publicly|freely)\s+)?(?:made\s+)?"
    r"(?:available|accessible|deposited|archived)\b.{0,120}"
    r"\b(?:osf|zenodo|figshare|dryad|github|repository|corresponding author|"
    r"request|https?://)",
)
_DATA_SUBJECT = (
    r"(?:(?:the|these|our|all)\s+)?"
    r"(?:(?:raw|anonymi[sz]ed|de-identified|underlying|source|study|research|"
    r"survey|experimental|processed|supporting)\s+){0,2}"
    r"(?:data|datasets?|code|materials?)"
)
_DATA_TARGET = (
    r"(?:osf|zenodo|figshare|dryad|github|repository|corresponding author|request|https?://)"
)
_DATA_ASSERTIVE_DECLARATION = re.compile(
    rf"^\s*(?:(?:data|code|materials?)\s+availability\s*:\s*"
    rf"(?:.{{0,120}}\b(?:available|accessible|deposited|archived)\b.{{0,120}}\b{_DATA_TARGET}|"
    rf"available\b.{{0,120}}\b{_DATA_TARGET})|"
    rf"{_DATA_SUBJECT}\s+"
    rf"(?:generated|analy[sz]ed|collected|produced|used|underlying|supporting)\b"
    rf".{{0,160}}\b(?:available|accessible|deposited|archived)\b"
    rf".{{0,120}}\b{_DATA_TARGET}|"
    rf"{_DATA_SUBJECT}\s+(?:are|is|were|was|will\s+be|can\s+be)\s+"
    rf"(?:(?:openly|publicly|freely)\s+)?(?:made\s+)?"
    rf"(?:available|accessible|deposited|archived)\b.{{0,120}}\b{_DATA_TARGET})",
    re.IGNORECASE,
)
# PaperMetadata statement field → its ordered anchor phrases. Order mirrors
# research_integrity._STATEMENT_SECTION_TYPES.
_ANCHORS: dict[str, tuple[re.Pattern[str], ...]] = {
    "funding_statement": _compile(
        r"funding\s*:",
        r"(?:funding|financial support)\s+"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)(?:\s*[.;:]|\s*$)",
        r"received funding from",
        r"this work was supported",
        r"supported by",
        r"funded by",
        r"funding for this",
        r"financial support",
        r"grant no\.",
    ),
    "coi_statement": _compile(
        r"(?:conflicts? of interest|competing interests?)\s*:",
        r"(?:conflicts? of interest|competing interests?)\s+"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)(?:\s*[.;:]|\s*$)",
        r"conflicts? of interest",
        r"competing interests?",
        r"no potential conflict",
    ),
    "ethics_statement": _compile(
        r"ethics\s*:",
        r"(?:ethics|ethical approval|ethics approval)\s+"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)(?:\s*[.;:]|\s*$)",
        r"ethical approval",
        r"ethics committee",
        r"ethics approval",
        r"institutional review board",
        r"irb approval",
        r"informed consent",
        r"complies with ethical",
    ),
    "data_availability": _compile(
        r"data availability\s*:",
        r"(?:data|code|materials?)\s+availability\s+"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)(?:\s*[.;:]|\s*$)",
        r"data availability",
        r"code availability",
        r"materials availability",
        *_DATA_DECLARATION_PATTERNS,
    ),
}

_STATEMENT_FIELDS = tuple(_ANCHORS)

# Funding anchors split by reliability. The STRONG anchors are unambiguous —
# a funder-ish token anywhere in the captured window is enough (existing
# behavior). "supported by" alone is ambiguous — generic prose like
# "supported by empirical data" also matches it — so it additionally
# requires the funder-ish token to appear in the anchor sentence AFTER the
# anchor match itself (see `_scan_field`).
_FUNDING_STRONG_ANCHORS = _compile(
    r"funding\s*:",
    r"financial support\s*:",
    r"received funding from",
    r"this work was supported",
    r"funded by",
    r"funding for this",
    r"financial support",
    r"grant no\.",
)

_BARE_CATEGORY_LABELS: dict[str, frozenset[str]] = {
    "funding_statement": frozenset({"funding", "financial support"}),
    "coi_statement": frozenset(
        {"conflict of interest", "conflicts of interest", "competing interests"}
    ),
    "ethics_statement": frozenset({"ethics", "ethical approval", "ethics approval"}),
    "data_availability": frozenset(
        {"data availability", "code availability", "materials availability"}
    ),
}
_BARE_NEGATIVE_DECLARATION = re.compile(
    r"^(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
    re.IGNORECASE,
)
_FUNDING_AMBIGUOUS_ANCHOR = re.compile(r"\bsupported by", re.IGNORECASE)

# Max number of following sentences appended to an anchor sentence's capture.
_MAX_FOLLOW_SENTENCES = 2

# Funder-ish tokens that must appear near a funding anchor for it to count —
# guards against generic "supported by strong evidence" prose. Acronyms are an
# explicit allowlist: arbitrary upper-case scientific terms (HRV, IPCP) are not
# evidence that the sentence names a funder.
_FUNDER_KEYWORD = re.compile(
    r"\b(?:grant|grants|foundation|council|ministry|fellowship|"
    r"scholarship|endowment|institute|agency|fund|nsf|erc|nih|nserc|dfg|"
    r"wellcome trust|award)\b",
    re.IGNORECASE,
)
_FUNDER_STRONG = re.compile(
    r"#\d"  # grant number like "#647910"
    r"|\b(?:NSF|ERC|NIH|NSERC|DFG)\b"
    r"|\b[A-Z]{1,6}\d{2,}(?:-[A-Z0-9]+)+\b"  # e.g. R01-MH123
    r"|\b(?:[A-Z][A-Za-z&.-]+\s+){0,5}"
    r"(?:Foundation|Council|Trust|Institute|Fund|Agency|Ministry)\b"
    r"|\b(?:[A-Z][A-Za-z&.-]+\s+){1,5}University\b"
)
_FUNDING_ACTION = (
    r"(?:(?:was|were|is|are|has|have)\s+(?:been\s+)?(?:funded|supported)\b|"
    r"(?:had|has|have)\s+(?:received\s+)?no\s+"
    r"(?:external\s+)?(?:funding|financial support)\b|"
    r"(?:received|receives?|receive)\s+"
    r"(?:no\s+(?:external\s+)?(?:funding|financial support)|funding|financial support)\b|"
    r"acknowledg(?:e|es|ed)\s+(?:the\s+)?(?:funding|financial support)\b)"
)
_AUTHOR_FUNDING_ACTION = rf"(?:{_FUNDING_ACTION}|(?:funded|supported)\s+by\b)"
_AUTHOR_FUNDING_DECLARATION = re.compile(
    rf"^\s*(?P<subjects>.+?)(?:\s+|\s*[—–]\s*){_AUTHOR_FUNDING_ACTION}",
    re.IGNORECASE,
)
_FUNDING_LABEL = re.compile(
    r"^\s*(?:funding|financial support)\s*:\s*",
    re.IGNORECASE,
)
_FUNDING_SOURCE_CLAUSE_SEPARATOR = re.compile(r"\s*;\s*")
_FUNDING_ASSERTIVE_DECLARATION = re.compile(
    r"^\s*(?:(?:funding|financial support)\s+"
    r"(?:was\s+)?(?:provided|received|from|by)\b|"
    r"(?:supported|funded)\s+by\b|funding\s+for\s+(?:this|the)\b|"
    r"(?:grant|award)\s+(?:no\.?|number|#)\b|additional support came from\b|"
    rf"(?:this|the|our)\s+(?:present\s+)?"
    rf"(?:study|work|research|project|trial|article|manuscript)\s+{_FUNDING_ACTION}|"
    rf"work\s+on\s+(?:this|the)\b.{{0,80}}\s+{_FUNDING_ACTION}|"
    rf"research\s+reported\s+in\s+(?:this|the)\s+"
    rf"(?:publication|article|work)\s+{_FUNDING_ACTION}|"
    rf"(?:we|(?:the\s+)?(?:authors?|author\(s\))|investigators?)\s+{_FUNDING_ACTION})",
    re.IGNORECASE,
)
_FUNDING_SUBJECTLESS_FRAGMENT = re.compile(
    r"^\s*(?:supported|funded)\s+by\b",
    re.IGNORECASE,
)
_FUNDING_INDEPENDENT_NEGATIVE_DECLARATION = re.compile(
    r"^\s*(?:not applicable|none(?: declared)?|no(?:ne)?|"
    r"no\s+(?:external\s+)?(?:funding|financial support)\s+(?:was\s+)?received)"
    r"\s*[.;]?\s*$",
    re.IGNORECASE,
)
_FUNDING_NEGATIVE_DECLARATION = re.compile(
    r"\b(?:(?:had|has|have)\s+(?:received\s+)?no\s+"
    r"(?:external\s+)?(?:funding|financial support)|"
    r"received no\s+(?:external\s+)?(?:funding|financial support)|"
    r"no\s+(?:external\s+)?(?:funding|financial support)\s+(?:was\s+)?received)\b|"
    r"^\s*(?:not applicable|none(?: declared)?|no(?:ne)?)\s*[.;]?\s*$",
    re.IGNORECASE,
)
_FUNDING_FUNDER_ONLY_DECLARATION = re.compile(
    r"^\s*(?:(?:grant|award)\s+(?:(?:no\.?|number|#)\s*)?[A-Z0-9-]+|"
    r"(?:(?:NSF|ERC|NIH|NSERC|DFG)|(?:the\s+)?Wellcome Trust|"
    r"(?:[A-Z][A-Za-z&.'-]*\s+){0,5}"
    r"(?:Foundation|Council|Trust|Institute|Fund|Agency|Ministry))"
    r"(?:\s+(?:grant|award)(?:\s+(?:no\.?|number|#))?\s*[A-Z0-9-]+)?"
    r")\s*[.;]?\s*$",
    re.IGNORECASE,
)

_ETHICS_ASSERTIVE_DECLARATION = re.compile(
    r"^\s*(?:(?:(?:ethics|ethical approval|ethics approval)\s*:\s*)?"
    r"(?:(?:this|the|our|all)\s+"
    r"(?:study|research|protocol|procedures?|trial)\s+(?:was|were|is|are)\s+"
    r"(?:approved|reviewed|conducted|performed)\b|"
    r"(?:study|research|protocol|procedures?)\s+(?:was|were)\s+approved\b|"
    r"approved\s+by\s+(?:the\s+)?(?:ethics committee|institutional review board|irb)\b|"
    r"(?:the\s+)?(?:ethics committee|institutional review board|irb)\s+"
    r"(?:approved|reviewed)\b|"
    r"(?:ethical|ethics)\s+approval\s+(?:was\s+)?(?:obtained|granted|provided)\b|"
    r"(?:written\s+)?informed consent\s+(?:was\s+)?"
    r"(?:obtained|provided|given|secured|documented|waived)\b|"
    r"(?:all\s+)?(?:participants?|patients?|subjects?)\s+"
    r"(?:provided|gave|signed|granted)\s+(?:written\s+)?informed consent\b|"
    r"(?:an?\s+)?(?:waiver|exemption)\s+(?:was\s+)?(?:obtained|granted|approved)\b|"
    r"protocol\s+(?:no\.?|number|id)\s*[A-Z0-9-]+\b|"
    r"(?:this|the|our)\s+(?:study|research|protocol)\s+"
    r"compli(?:ed|es)\s+with\s+(?:the\s+)?(?:declaration of helsinki|ethical)\b))",
    re.IGNORECASE,
)

_COI_ASSERTIVE_DECLARATION = re.compile(
    r"(?:^\s*|\band\s+)(?:(?:(?:declaration of (?:conflicting|competing) interests?|"
    r"conflicts? of interest|competing interests?)\s*:\s*)?"
    r"(?:(?:the\s+)?(?:authors?|author\(s\))|we)\s+"
    r"(?:declar(?:e[sd]?|ed)|report(?:s|ed)?)\s+(?:that\s+)?(?:they\s+)?"
    r"(?:have\s+)?(?:(?:no\s+(?:potential\s+)?|potential\s+)?"
    r"(?:conflicts?|competing interests?)|none)\b|"
    r"(?:the\s+)?author\s+has\s+no\s+(?:conflicts?|competing interests?)\b|"
    r"no\s+(?:potential\s+)?(?:conflicts?|competing interests?)\s+"
    r"(?:were\s+)?(?:declared|reported)\b|none declared\b)",
    re.IGNORECASE,
)
_COI_TOPICAL_PROSE = re.compile(
    r"\b(?:as (?:a|an) .{0,30}(?:topic|concept)|in the literature|"
    r"literature on|debate about|role of|during interviews?|among firms|"
    r"between (?:the|these|two))\b",
    re.IGNORECASE,
)

_BOILERPLATE_BOUNDARY = re.compile(
    r"(?:\b(?:received|accepted)\b\s*(?:on|:)?\s*"
    r"(?:\d{1,2}\b|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b|20\d{2}\b)|"
    r"\bpublished\s+(?:online|by|under)\b|\bcopyright\b|"
    r"\bcreative commons\b|\blicen[cs]e\b|\bcorrespond(?:ence|ing author)\b|"
    r"\bpublisher\b)",
    re.IGNORECASE,
)


def _has_funder_hint(text: str) -> bool:
    return bool(_FUNDER_KEYWORD.search(text) or _FUNDER_STRONG.search(text))


def _has_funding_negative_declaration(text: str) -> bool:
    """Whether ``text`` contains a recognized funding-negative action."""
    _label, body = _funding_label_and_body(text)
    return bool(_FUNDING_NEGATIVE_DECLARATION.search(body))


def _normalize_author_alias(value: str) -> str:
    """Normalize a name to its ordered NFKC/casefolded alphanumeric string."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _name_initials(value: str) -> str:
    """Mirror the author model's whitespace-token initial convention."""
    return "".join(token[0] for token in value.split() if token).upper()


def _author_aliases(author_names: Iterable[tuple[str, str]]) -> frozenset[str]:
    """Build immutable aliases for already-grounded ``(given, family)`` names.

    A complete given-plus-family spelling is authoritative. Initial forms use
    every given initial or only the first given initial, followed by every
    whitespace-token family initial; this matches contribution attribution and
    preserves particles/suffixes represented inside ``PaperAuthor.family``.
    """
    aliases: set[str] = set()
    for given_value, family_value in author_names:
        given = given_value.strip()
        family = family_value.strip()
        if not given or not family:
            continue
        given_initials = _name_initials(given)
        family_initials = _name_initials(family)
        candidates = {f"{given} {family}"}
        if given_initials and family_initials:
            candidates.update(
                {
                    f"{given_initials}{family_initials}",
                    f"{given_initials[0]}{family_initials}",
                }
            )
        aliases.update(
            normalized
            for candidate in candidates
            if (normalized := _normalize_author_alias(candidate))
        )
    return frozenset(aliases)


def _author_subjects_are_grounded(
    subjects: str,
    author_aliases: frozenset[str],
) -> bool:
    if not author_aliases:
        return False

    raw_tokens: list[tuple[int, int, str]] = []
    token_start: int | None = None
    for offset, character in enumerate(subjects):
        is_mark = unicodedata.category(character).startswith("M")
        if token_start is None:
            if character.isalnum():
                token_start = offset
        elif not character.isalnum() and not is_mark:
            normalized = _normalize_author_alias(subjects[token_start:offset])
            if normalized:
                raw_tokens.append((token_start, offset, normalized))
            token_start = None
    if token_start is not None:
        normalized = _normalize_author_alias(subjects[token_start:])
        if normalized:
            raw_tokens.append((token_start, len(subjects), normalized))

    conjunction_spans = tuple(
        (start, end) for start, end, normalized in raw_tokens if normalized == "and"
    )
    name_tokens = tuple(token for token in raw_tokens if token[2] != "and")
    if not name_tokens:
        return False

    def has_conjunction_delimiter(left: int, right: int) -> bool:
        gap = subjects[left:right]
        return "&" in gap or any(left <= start and end <= right for start, end in conjunction_spans)

    def has_list_delimiter(left: int, right: int) -> bool:
        return "," in subjects[left:right] or has_conjunction_delimiter(left, right)

    max_alias_length = max(map(len, author_aliases))

    @cache
    def can_cover_from(start: int) -> bool:
        if start == len(name_tokens):
            return True
        if start and not has_list_delimiter(name_tokens[start - 1][1], name_tokens[start][0]):
            return False

        candidate_ends: list[int] = []
        normalized_span = ""
        for end in range(start, len(name_tokens)):
            if end > start and has_conjunction_delimiter(
                name_tokens[end - 1][1], name_tokens[end][0]
            ):
                break
            normalized_span += name_tokens[end][2]
            if len(normalized_span) > max_alias_length:
                break
            if normalized_span in author_aliases:
                candidate_ends.append(end + 1)

        return any(can_cover_from(end) for end in reversed(candidate_ends))

    return can_cover_from(0)


def _has_grounded_author_funding_declaration(
    text: str,
    author_aliases: frozenset[str],
) -> bool:
    match = _AUTHOR_FUNDING_DECLARATION.search(text)
    return bool(
        match
        and _author_subjects_are_grounded(
            match.group("subjects"),
            author_aliases,
        )
    )


def _funding_source_clauses(text: str) -> tuple[str, ...]:
    """Return raw funding clauses without detaching subjects from actions."""
    return tuple(
        clause for part in _FUNDING_SOURCE_CLAUSE_SEPARATOR.split(text) if (clause := part.strip())
    )


def _funding_label_and_body(text: str) -> tuple[re.Match[str] | None, str]:
    label = _FUNDING_LABEL.match(text)
    return label, text[label.end() :].strip() if label is not None else text.strip()


def _funding_candidate_ownership(
    text: str,
    author_aliases: frozenset[str],
    *,
    allow_subjectless_fragment: bool,
    preceding_subjects: tuple[str, ...] = (),
) -> tuple[bool, bool]:
    """Return ``(assertive, unresolved_named_subject)`` for one raw clause."""
    label, body = _funding_label_and_body(text)
    if _FUNDING_ASSERTIVE_DECLARATION.search(
        body
    ) or _FUNDING_INDEPENDENT_NEGATIVE_DECLARATION.fullmatch(body):
        assertive = not (
            label is None
            and not allow_subjectless_fragment
            and _FUNDING_SUBJECTLESS_FRAGMENT.match(body)
        )
        return assertive, False
    named_declaration = _AUTHOR_FUNDING_DECLARATION.search(body)
    if named_declaration is not None:
        grounded = bool(
            _has_grounded_author_funding_declaration(body, author_aliases)
            and all(
                _author_subjects_are_grounded(subjects, author_aliases)
                for subjects in preceding_subjects
            )
        )
        return grounded, not grounded
    funder_only = bool(label is not None and _FUNDING_FUNDER_ONLY_DECLARATION.fullmatch(body))
    return funder_only, False


def _funding_source_ownership(
    text: str,
    author_aliases: frozenset[str],
) -> tuple[bool, bool]:
    """Return source assertion and unresolved-name status using raw clauses."""
    preceding_subjects: list[str] = []
    has_assertive_declaration = False
    unresolved_named_subject = False
    for index, clause in enumerate(_funding_source_clauses(text)):
        other_categories = _categories_in(clause) - {"funding_statement"}
        if other_categories:
            funding_starts = [
                match.start()
                for pattern in _ANCHORS["funding_statement"]
                if (match := pattern.search(clause))
            ]
            other_starts = [
                match.start()
                for field in other_categories
                for pattern in _ANCHORS[field]
                if (match := pattern.search(clause))
            ]
            if not funding_starts or (other_starts and min(other_starts) < min(funding_starts)):
                preceding_subjects.clear()
                continue
        assertive, unresolved = _funding_candidate_ownership(
            clause,
            author_aliases,
            allow_subjectless_fragment=index == 0,
            preceding_subjects=tuple(preceding_subjects),
        )
        has_assertive_declaration = has_assertive_declaration or assertive
        unresolved_named_subject = unresolved_named_subject or unresolved
        if other_categories:
            preceding_subjects.clear()
            continue
        _label, body = _funding_label_and_body(clause)
        if not assertive and body and not _FUNDING_SUBJECTLESS_FRAGMENT.match(body):
            preceding_subjects.append(body)
    return (
        has_assertive_declaration and not unresolved_named_subject,
        unresolved_named_subject,
    )


def _has_unresolved_author_funding_declaration(
    text: str,
    author_aliases: frozenset[str],
) -> bool:
    """Whether named funding ownership failed with no independent assertion."""
    _assertive, unresolved = _funding_source_ownership(text, author_aliases)
    return unresolved


def _has_labeled_negative(field: str, text: str) -> bool:
    labels = {
        **_BARE_CATEGORY_LABELS,
        "coi_statement": frozenset(
            {
                *_BARE_CATEGORY_LABELS["coi_statement"],
                "declaration of conflicting interests",
                "declaration of competing interests",
            }
        ),
    }[field]
    negative = r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?"
    return any(
        re.fullmatch(
            rf"\s*{re.escape(label)}(?:\s*:\s*|\s+){negative}\s*",
            text,
            re.IGNORECASE,
        )
        for label in labels
    )


def _has_assertive_declaration(
    field: str,
    text: str,
    *,
    author_aliases: frozenset[str] = frozenset(),
    source_text: str | None = None,
) -> bool:
    """Whether ``text`` makes an owned declaration rather than mentioning one."""
    stripped = text.strip()
    if _has_labeled_negative(field, stripped):
        return True
    if field == "funding_statement":
        funding_source = (source_text or stripped).strip()
        assertive, _unresolved = _funding_source_ownership(funding_source, author_aliases)
        return bool(
            assertive
            and (
                _has_funder_hint(funding_source)
                or _has_funding_negative_declaration(funding_source)
            )
        )
    if field == "coi_statement":
        return bool(
            _COI_ASSERTIVE_DECLARATION.search(stripped) and not _COI_TOPICAL_PROSE.search(stripped)
        )
    if field == "ethics_statement":
        return bool(_ETHICS_ASSERTIVE_DECLARATION.search(stripped))
    return bool(_DATA_ASSERTIVE_DECLARATION.search(stripped))


def _categories_in(text: str) -> set[str]:
    """Statement fields whose anchor phrase appears in ``text``."""
    categories = {
        field
        for field, patterns in _ANCHORS.items()
        if any(pattern.search(text) for pattern in patterns)
    }
    normalized = text.strip().casefold()
    categories.update(
        field for field, labels in _BARE_CATEGORY_LABELS.items() if normalized in labels
    )
    return categories


def _has_field_anchor(field: str, text: str) -> bool:
    return any(pattern.search(text) for pattern in _ANCHORS[field]) or (
        text.strip().casefold() in _BARE_CATEGORY_LABELS[field]
    )


def _boilerplate_boundary_for_field(field: str, text: str, start: int = 0) -> re.Match[str] | None:
    for match in _BOILERPLATE_BOUNDARY.finditer(text, start):
        legitimate_data_contact = bool(
            field == "data_availability"
            and match.group().casefold().startswith("correspond")
            and re.search(
                r"\b(?:available|accessible)\b[^.!?]{0,80}"
                r"(?:\bfrom\s+|\bby\s+contacting\s+|\bthrough\s+)(?:the\s+)?$",
                text[: match.start()],
                re.IGNORECASE,
            )
        )
        if legitimate_data_contact:
            continue
        return match
    return None


def _bounded_sentence_for_field(field: str, text: str) -> str:
    """Return only ``field``'s declaration when categories share a sentence."""
    own_matches = [match for pat in _ANCHORS[field] if (match := pat.search(text))]
    if not own_matches:
        return text.strip()
    own_start = min(match.start() for match in own_matches)
    other_starts = sorted(
        match.start()
        for other_field, patterns in _ANCHORS.items()
        if other_field != field
        for pattern in patterns
        if (match := pattern.search(text))
    )
    start = own_start if any(position < own_start for position in other_starts) else 0
    ends = [position for position in other_starts if position > own_start]
    boilerplate = _boilerplate_boundary_for_field(field, text, own_start)
    if boilerplate is not None:
        ends.append(boilerplate.start())
    end = min(ends) if ends else len(text)
    return text[start:end].strip()


def scan_statements_fallback(contents: PaperContents, metadata: PaperMetadata) -> None:
    """Fill still-``None`` statement fields from lexical anchors in the body.

    Runs after :func:`copy_integrity_statements`; only touches fields the copy
    pass left unset. Records a ``STATEMENT_LEXICAL_FALLBACK: <field>`` warning
    for each field it fills."""
    sentences = [s for s in contents.sentences if not s.is_display_formula]
    if not sentences:
        return
    section_type_by_id = {sec.section_id: sec.section_type for sec in contents.sections}
    author_aliases = _author_aliases((author.given, author.family) for author in metadata.authors)

    for field in _STATEMENT_FIELDS:
        if getattr(metadata, field) is not None:
            continue
        captured = _scan_field(
            field,
            sentences,
            section_type_by_id,
            author_aliases=author_aliases,
        )
        if captured:
            setattr(metadata, field, captured)
            contents.processing_warnings.append(f"STATEMENT_LEXICAL_FALLBACK: {field}")


def _scan_field(
    field: str,
    sentences: list,
    section_type_by_id: dict[int, CanonicalSection],
    *,
    author_aliases: frozenset[str] = frozenset(),
) -> str | None:
    """First qualifying anchor for ``field``, captured as anchor sentence plus up
    to two same-section following sentences (stopping early at a different-type
    anchor). ``None`` when no anchor qualifies."""
    for i, sent in enumerate(sentences):
        # References-section anchors are citation prose, never statements.
        if section_type_by_id.get(sent.section_id) == CanonicalSection.REFERENCES:
            continue
        if not _has_field_anchor(field, sent.text):
            continue
        bare_label = sent.text.strip().casefold() in _BARE_CATEGORY_LABELS[field]

        strong_anchor_match = False
        if field == "funding_statement" and not bare_label:
            strong_anchor_match = any(p.search(sent.text) for p in _FUNDING_STRONG_ANCHORS)
            if not strong_anchor_match:
                # Only the ambiguous "supported by" anchor matched — it only
                # qualifies if a funder-ish token follows the anchor itself
                # (rejects generic prose like "... if supported by empirical
                # data" while still catching "supported by the Wellcome Trust").
                ambiguous_match = _FUNDING_AMBIGUOUS_ANCHOR.search(sent.text)
                if not ambiguous_match or not _has_funder_hint(sent.text[ambiguous_match.end() :]):
                    continue

        window = [_bounded_sentence_for_field(field, sent.text)]
        source_window = [sent.text]
        sec_id = sent.section_id
        paragraph_id = sent.paragraph_id
        for j in range(i + 1, min(i + 1 + _MAX_FOLLOW_SENTENCES, len(sentences))):
            nxt = sentences[j]
            if nxt.section_id != sec_id or nxt.paragraph_id != paragraph_id:
                break
            if _categories_in(nxt.text) - {field}:
                break  # a different statement type starts — stop the capture
            if _boilerplate_boundary_for_field(field, nxt.text):
                break
            window.append(_bounded_sentence_for_field(field, nxt.text))
            source_window.append(nxt.text)

        joined = " ".join(w for w in window if w).strip()
        source_joined = " ".join(source_window).strip()
        followup = ""
        if bare_label:
            followup = " ".join(w for w in window[1:] if w).strip()
            if not (
                _BARE_NEGATIVE_DECLARATION.fullmatch(followup) or _has_field_anchor(field, followup)
            ):
                continue
        if not _has_assertive_declaration(
            field,
            joined,
            author_aliases=author_aliases,
            source_text=followup if bare_label else source_joined,
        ):
            continue
        if joined:
            return joined
    return None
