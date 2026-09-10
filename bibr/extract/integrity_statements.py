"""Unified candidate resolution for research-integrity statements.

Section classifications are retrieval evidence, not permission to copy.  The
resolver builds section and anchored-paragraph candidates together, records
their source IDs, and delays rendering until callers have finished late text
cleaning.  ``shadow`` keeps compatibility values while surfacing typed
comparison evidence; ``active`` materializes the bounded selection.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from bibr.extract.statement_scan import (
    _ANCHORS,
    _BARE_CATEGORY_LABELS,
    _FUNDING_AMBIGUOUS_ANCHOR,
    _FUNDING_STRONG_ANCHORS,
    _author_aliases,
    _boilerplate_boundary_for_field,
    _bounded_sentence_for_field,
    _categories_in,
    _has_assertive_declaration,
    _has_field_anchor,
    _has_funder_hint,
    _has_funding_negative_declaration,
    _has_unresolved_author_funding_declaration,
)
from bibr.paper_contents import CanonicalSection
from bibr.utils.text import normalize_text
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.models import PaperMetadata
    from bibr.paper_contents import PaperContents, PaperSection, PaperSentence

IntegrityStatementMode = Literal["legacy", "shadow", "active"]

_FIELD_SECTION_TYPES: dict[str, CanonicalSection] = {
    "funding_statement": CanonicalSection.FUNDING,
    "coi_statement": CanonicalSection.COI,
    "ethics_statement": CanonicalSection.ETHICS,
    "data_availability": CanonicalSection.OPEN_DATA,
}
_FIELDS = tuple(_FIELD_SECTION_TYPES)
_SECTION_TYPE_FIELDS = {section_type: field for field, section_type in _FIELD_SECTION_TYPES.items()}

_STRONG_HEADINGS: dict[str, frozenset[str]] = {
    "funding_statement": frozenset(
        {
            "funding statement",
            "sources of funding",
            "source of funding",
            "financial support",
            "funding information",
        }
    ),
    "coi_statement": frozenset(
        {
            "competing interests",
            "conflict of interest statement",
            "conflicts of interest statement",
            "declaration of competing interest",
            "declaration of competing interests",
            "declaration of conflicting interest",
            "declaration of conflicting interests",
        }
    ),
    "ethics_statement": frozenset(
        {
            "ethical approval",
            "ethics approval",
            "ethics statement",
            "ethics approval and consent to participate",
            "institutional review board statement",
        }
    ),
    "data_availability": frozenset(
        {
            "data availability statement",
            "availability of data and materials",
            "data and code availability",
            "code availability statement",
        }
    ),
}
_GENERIC_HEADINGS: dict[str, frozenset[str]] = {
    "funding_statement": frozenset({"funding", "funding sources"}),
    "coi_statement": frozenset({"conflict of interest", "conflicts of interest"}),
    "ethics_statement": frozenset({"ethics", "ethical considerations"}),
    "data_availability": frozenset(
        {"data availability", "code availability", "materials availability"}
    ),
}

_UNLABELED_NEGATIVE_DECLARATION = re.compile(
    r"^(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
    re.IGNORECASE,
)
_FIELD_LABELED_NEGATIVE_DECLARATION: dict[str, re.Pattern[str]] = {
    "funding_statement": re.compile(
        r"^(?:funding|financial support)(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
    "coi_statement": re.compile(
        r"^(?:conflicts? of interest|competing interests?)(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
    "ethics_statement": re.compile(
        r"^(?:ethics|ethical approval|ethics approval)(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
    "data_availability": re.compile(
        r"^(?:data|code|materials?)\s+availability(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
}
_FIELD_NEGATIVE_DECLARATION: dict[str, re.Pattern[str]] = {
    "funding_statement": re.compile(
        r"\b(?:no\s+(?:external\s+)?(?:funding|financial support)|"
        r"received no (?:funding|financial support))\b",
        re.IGNORECASE,
    ),
    "coi_statement": re.compile(
        r"^no\s+(?:potential\s+)?"
        r"(?:conflicts?(?:\s+of\s+interest)?|competing interests?)"
        r"(?:\s+(?:were\s+)?(?:declared|reported))?\.?$",
        re.IGNORECASE,
    ),
    "ethics_statement": re.compile(
        r"\b(?:ethical approval|ethics approval|consent)\s+(?:was\s+)?not\s+(?:required|applicable)\b",
        re.IGNORECASE,
    ),
    "data_availability": re.compile(
        r"\b(?:no data|data (?:are|were) not (?:generated|available))\b",
        re.IGNORECASE,
    ),
}
_TOPICAL_ETHICS_HEADING = re.compile(
    r"\b(?:ethical problem|ethics of|ethical implications?|ethics? .{0,30} debate|"
    r"ethical and scientific judgements?)\b",
    re.IGNORECASE,
)
_PUBLICATION_CONSENT = re.compile(r"^consent for publication$", re.IGNORECASE)
_AMBIGUOUS_CLASSIFICATION_SOURCES = frozenset(
    {"model", "llm", "alias_prior", "substring_alias", "parent_context"}
)
_MAX_COMPACT_CHARS = 1200
_MAX_COMPACT_PARAGRAPHS = 3


def _legacy_compile(*phrases: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(r"\b" + phrase, re.IGNORECASE) for phrase in phrases)


# Frozen pre-resolver lexical behavior.  Shadow compares against this exact
# snapshot; it must not inherit bounded anchors, predicates, or rendering.
_LEGACY_ANCHORS: dict[str, tuple[re.Pattern[str], ...]] = {
    "funding_statement": _legacy_compile(
        r"this work was supported",
        r"supported by",
        r"funded by",
        r"funding for this",
        r"financial support",
        r"grant no\.",
    ),
    "coi_statement": _legacy_compile(
        r"conflicts? of interest",
        r"competing interests?",
        r"no potential conflict",
    ),
    "ethics_statement": _legacy_compile(
        r"ethical approval",
        r"ethics committee",
        r"ethics approval",
        r"institutional review board",
        r"irb approval",
        r"informed consent",
        r"complies with ethical",
    ),
    "data_availability": _legacy_compile(
        r"data availability",
        r"data are available",
        r"data are openly available",
        r"code availability",
        r"materials are available",
    ),
}
_LEGACY_FUNDING_STRONG_ANCHORS = _legacy_compile(
    r"this work was supported",
    r"funded by",
    r"funding for this",
    r"financial support",
    r"grant no\.",
)
_LEGACY_FUNDING_AMBIGUOUS_ANCHOR = re.compile(r"\bsupported by", re.IGNORECASE)
_LEGACY_FUNDER_KEYWORD = re.compile(
    r"\b(?:grant|grants|foundation|council|university|ministry|fellowship|"
    r"scholarship|endowment|nsf|erc|nih|nserc|dfg|funded|funding|award)\b",
    re.IGNORECASE,
)
_LEGACY_FUNDER_STRONG = re.compile(
    r"#\d|\b[A-Z]{2,}\b|\b[A-Z][A-Za-z]+\s+"
    r"(?:Foundation|Council|Trust|Institute|Fund|Agency)\b"
)


@dataclass(frozen=True)
class IntegrityStatementCandidate:
    field: str
    method: str
    heading: str | None
    section_ids: tuple[int, ...]
    text_ids: tuple[int, ...]
    paragraph_ids: tuple[int, ...]
    pages: tuple[int, ...]
    classification_source: str | None
    classification_score: float | None
    reason_flags: tuple[str, ...]
    accepted: bool


@dataclass(frozen=True)
class IntegrityStatementResolution:
    mode: IntegrityStatementMode
    candidates: tuple[IntegrityStatementCandidate, ...]
    legacy_candidate_indices: tuple[tuple[str, tuple[int, ...]], ...]
    selected_candidate_indices: tuple[tuple[str, tuple[int, ...]], ...]
    legacy_statement_snapshots: tuple[tuple[str, str | None], ...]
    issues: tuple[ValidationIssue, ...]

    def selected_indices(self, field: str) -> tuple[int, ...]:
        return dict(self.selected_candidate_indices).get(field, ())

    def legacy_indices(self, field: str) -> tuple[int, ...]:
        return dict(self.legacy_candidate_indices).get(field, ())

    def candidate_indices(self, field: str, *, effective: bool = True) -> tuple[int, ...]:
        if effective and self.mode == "active":
            return self.selected_indices(field)
        return self.legacy_indices(field)


def _ordered_unique(values) -> tuple:
    return tuple(dict.fromkeys(values))


def _section_rows(contents: PaperContents) -> dict[int, list[PaperSentence]]:
    rows: dict[int, list[PaperSentence]] = {}
    for sentence in contents.sentences:
        if not sentence.is_display_formula:
            rows.setdefault(sentence.section_id, []).append(sentence)
    return rows


def _legacy_categories_in(text: str) -> set[str]:
    return {
        field
        for field, patterns in _LEGACY_ANCHORS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def _legacy_has_funder_hint(text: str) -> bool:
    return bool(_LEGACY_FUNDER_KEYWORD.search(text) or _LEGACY_FUNDER_STRONG.search(text))


def _legacy_section_text(rows: list[PaperSentence]) -> str:
    """Reproduce the pre-resolver canonical-section join byte for byte."""
    return " ".join(row.text for row in rows if not row.is_display_formula).strip()


def _legacy_lexical_rows(
    field: str,
    rows: list[PaperSentence],
    section_by_id: dict[int, PaperSection],
) -> list[PaperSentence]:
    """Return the first pre-delta lexical capture, including cross-paragraph rows."""
    patterns = _LEGACY_ANCHORS[field]
    for index, sentence in enumerate(rows):
        section = section_by_id.get(sentence.section_id)
        if section is not None and section.section_type == CanonicalSection.REFERENCES:
            continue
        if not any(pattern.search(sentence.text) for pattern in patterns):
            continue

        strong_funding_anchor = False
        if field == "funding_statement":
            strong_funding_anchor = any(
                pattern.search(sentence.text) for pattern in _LEGACY_FUNDING_STRONG_ANCHORS
            )
            if not strong_funding_anchor:
                ambiguous = _LEGACY_FUNDING_AMBIGUOUS_ANCHOR.search(sentence.text)
                if ambiguous is None or not _legacy_has_funder_hint(
                    sentence.text[ambiguous.end() :]
                ):
                    continue

        captured = [sentence]
        for following in rows[index + 1 : index + 3]:
            if following.section_id != sentence.section_id:
                break
            if _legacy_categories_in(following.text) - {field}:
                break
            captured.append(following)

        joined = " ".join(row.text.strip() for row in captured if row.text.strip())
        if (
            field == "funding_statement"
            and strong_funding_anchor
            and not _legacy_has_funder_hint(joined)
        ):
            continue
        if joined:
            return captured
    return []


def _build_legacy_snapshot_candidates(
    contents: PaperContents,
    rows_by_section: dict[int, list[PaperSentence]],
    section_by_id: dict[int, PaperSection],
) -> list[IntegrityStatementCandidate]:
    """Freeze exact pre-delta section-copy then lexical-fallback ownership."""
    candidates: list[IntegrityStatementCandidate] = []
    linear_rows = [sentence for sentence in contents.sentences if not sentence.is_display_formula]
    for field, section_type in _FIELD_SECTION_TYPES.items():
        canonical_sections = [
            section for section in contents.sections if section.section_type == section_type
        ]
        canonical_start = len(candidates)
        if canonical_sections:
            for section in canonical_sections:
                rows = rows_by_section.get(section.section_id, [])
                text_ids, paragraph_ids, pages = _candidate_location(rows)
                if not _legacy_section_text(rows):
                    continue
                candidates.append(
                    IntegrityStatementCandidate(
                        field=field,
                        method="legacy_section_copy",
                        heading=section.header,
                        section_ids=(section.section_id,),
                        text_ids=text_ids,
                        paragraph_ids=paragraph_ids,
                        pages=pages,
                        classification_source=section.classification_source,
                        classification_score=section.classification_score,
                        reason_flags=("legacy_snapshot", "canonical_type"),
                        accepted=True,
                    )
                )
            if len(candidates) > canonical_start:
                continue

        rows = _legacy_lexical_rows(field, linear_rows, section_by_id)
        if not rows:
            continue
        section = section_by_id.get(rows[0].section_id)
        text_ids, paragraph_ids, pages = _candidate_location(rows)
        candidates.append(
            IntegrityStatementCandidate(
                field=field,
                method="legacy_lexical_capture",
                heading=section.header if section is not None else None,
                section_ids=(rows[0].section_id,),
                text_ids=text_ids,
                paragraph_ids=paragraph_ids,
                pages=pages,
                classification_source=(
                    section.classification_source if section is not None else None
                ),
                classification_score=(
                    section.classification_score if section is not None else None
                ),
                reason_flags=("legacy_snapshot", "lexical_anchor"),
                accepted=True,
            )
        )
    return candidates


def _candidate_location(
    rows: list[PaperSentence],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(sentence.text_id for sentence in rows),
        _ordered_unique(sentence.paragraph_id for sentence in rows),
        _ordered_unique(
            sentence.page_number for sentence in rows if sentence.page_number is not None
        ),
    )


def _strip_outer_punctuation(text: str) -> str:
    start = 0
    end = len(text)
    while start < end and (
        text[start].isspace() or unicodedata.category(text[start]).startswith("P")
    ):
        start += 1
    while end > start and (
        text[end - 1].isspace() or unicodedata.category(text[end - 1]).startswith("P")
    ):
        end -= 1
    return text[start:end]


def _heading_field(section: PaperSection) -> tuple[str | None, str, bool, bool]:
    normalized = _strip_outer_punctuation(normalize_text(section.header))
    publication_consent = bool(_PUBLICATION_CONSENT.fullmatch(normalized))
    for field in _FIELDS:
        if normalized in _STRONG_HEADINGS[field]:
            return field, normalized, True, publication_consent
        if normalized in _GENERIC_HEADINGS[field]:
            return field, normalized, False, publication_consent
    if "funding" in normalized or "financial support" in normalized:
        return "funding_statement", normalized, False, publication_consent
    if "conflict" in normalized or "competing interest" in normalized:
        return "coi_statement", normalized, False, publication_consent
    if "ethic" in normalized or publication_consent:
        return "ethics_statement", normalized, False, publication_consent
    if any(
        token in normalized
        for token in ("data availability", "code availability", "materials availability")
    ):
        return "data_availability", normalized, False, publication_consent
    return None, normalized, False, publication_consent


def _declaration_predicate(
    field: str,
    text: str,
    *,
    author_aliases: frozenset[str],
    source_text: str | None = None,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if field != "funding_statement" and _is_negative_declaration(field, stripped):
        return True
    return _has_assertive_declaration(
        field,
        stripped,
        author_aliases=author_aliases,
        source_text=source_text,
    )


def _is_negative_declaration(field: str, text: str) -> bool:
    stripped = text.strip()
    return bool(
        _UNLABELED_NEGATIVE_DECLARATION.fullmatch(stripped)
        or _FIELD_LABELED_NEGATIVE_DECLARATION[field].fullmatch(stripped)
        or _FIELD_NEGATIVE_DECLARATION[field].search(stripped)
    )


def _bounded_section_rows(field: str, rows: list[PaperSentence]) -> list[PaperSentence]:
    bounded: list[PaperSentence] = []
    for sentence in rows:
        other_category = _categories_in(sentence.text) - {field}
        if bounded and (_boilerplate_boundary_for_field(field, sentence.text) or other_category):
            break
        bounded.append(sentence)
        if other_category or _boilerplate_boundary_for_field(field, sentence.text):
            break
    return bounded


def _build_section_candidate_for_field(
    section: PaperSection,
    rows: list[PaperSentence],
    *,
    field: str,
    heading_field: str | None,
    strong_heading: bool,
    publication_consent: bool,
    author_aliases: frozenset[str],
) -> IntegrityStatementCandidate:
    canonical_field = _SECTION_TYPE_FIELDS.get(section.section_type)

    bounded_rows = _bounded_section_rows(field, rows)
    text = " ".join(sentence.text.strip() for sentence in bounded_rows if sentence.text.strip())
    text_ids, paragraph_ids, pages = _candidate_location(bounded_rows)
    compact = len(text) <= _MAX_COMPACT_CHARS and len(paragraph_ids) <= _MAX_COMPACT_PARAGRAPHS
    predicate = _declaration_predicate(field, text, author_aliases=author_aliases)
    author_grounding_failed = bool(
        field == "funding_statement"
        and _has_unresolved_author_funding_declaration(text, author_aliases)
    )
    explicit_negative = _is_negative_declaration(field, text)
    category_boundary_clipped = any(_categories_in(row.text) - {field} for row in bounded_rows)
    boilerplate_boundary_clipped = any(
        _boilerplate_boundary_for_field(field, row.text) for row in bounded_rows
    )
    topical_heading = field == heading_field == "ethics_statement" and bool(
        _TOPICAL_ETHICS_HEADING.search(section.header)
    )
    ambiguous_source = section.classification_source in _AMBIGUOUS_CLASSIFICATION_SOURCES

    accepted = bool(
        text_ids
        and not (publication_consent and field == heading_field)
        and not topical_heading
        and not author_grounding_failed
        and not (explicit_negative and not predicate)
        and not (boilerplate_boundary_clipped and not predicate)
        and not (category_boundary_clipped and not predicate)
    )
    if accepted:
        if ambiguous_source:
            accepted = predicate
        elif strong_heading:
            accepted = compact or predicate
        else:
            accepted = predicate and compact

    flags = []
    if canonical_field == field:
        flags.append("canonical_type")
    if heading_field == field:
        flags.append("heading_match")
    flags.append("strong_heading" if strong_heading else "ambiguous_heading")
    if compact:
        flags.append("compact")
    if predicate:
        flags.append("declaration_predicate")
    if explicit_negative:
        flags.append("explicit_negative")
    if author_grounding_failed:
        flags.append("author_grounding_failed")
    if category_boundary_clipped:
        flags.append("category_boundary_clipped")
    if boilerplate_boundary_clipped:
        flags.append("boilerplate_boundary_clipped")
    if topical_heading:
        flags.append("topical_heading")
    if publication_consent and field == heading_field:
        flags.append("publication_consent")
    if ambiguous_source:
        flags.append("ambiguous_classification_source")
    if not accepted:
        flags.append("rejected")
    return IntegrityStatementCandidate(
        field=field,
        method="trusted_section" if accepted else "classified_section",
        heading=section.header,
        section_ids=(section.section_id,),
        text_ids=text_ids,
        paragraph_ids=paragraph_ids,
        pages=pages,
        classification_source=section.classification_source,
        classification_score=section.classification_score,
        reason_flags=tuple(flags),
        accepted=accepted,
    )


def _build_section_candidates(
    section: PaperSection,
    rows: list[PaperSentence],
    *,
    author_aliases: frozenset[str],
) -> list[IntegrityStatementCandidate]:
    heading_field, _normalized_heading, strong_heading, publication_consent = _heading_field(
        section
    )
    canonical_field = _SECTION_TYPE_FIELDS.get(section.section_type)
    fields = _ordered_unique(
        field for field in (heading_field, canonical_field) if field is not None
    )
    return [
        _build_section_candidate_for_field(
            section,
            rows,
            field=field,
            heading_field=heading_field,
            strong_heading=strong_heading and field == heading_field,
            publication_consent=publication_consent,
            author_aliases=author_aliases,
        )
        for field in fields
    ]


def _funding_anchor_qualifies(text: str) -> bool:
    if any(pattern.search(text) for pattern in _FUNDING_STRONG_ANCHORS):
        return _has_funder_hint(text) or _has_funding_negative_declaration(text)
    ambiguous = _FUNDING_AMBIGUOUS_ANCHOR.search(text)
    return bool(ambiguous and _has_funder_hint(text[ambiguous.end() :]))


def _build_lexical_candidates(
    rows: list[PaperSentence],
    section_by_id: dict[int, PaperSection],
    *,
    author_aliases: frozenset[str],
) -> list[IntegrityStatementCandidate]:
    candidates: list[IntegrityStatementCandidate] = []
    for index, sentence in enumerate(rows):
        section = section_by_id.get(sentence.section_id)
        if section is not None and section.section_type == CanonicalSection.REFERENCES:
            continue
        for field in _ANCHORS:
            if not _has_field_anchor(field, sentence.text):
                continue
            bounded_rows = [sentence]
            for following in rows[index + 1 : index + 3]:
                if (
                    following.section_id != sentence.section_id
                    or following.paragraph_id != sentence.paragraph_id
                    or _boilerplate_boundary_for_field(field, following.text)
                    or _categories_in(following.text) - {field}
                ):
                    break
                bounded_rows.append(following)
            fragments = [_bounded_sentence_for_field(field, row.text) for row in bounded_rows]
            text = " ".join(fragment for fragment in fragments if fragment).strip()
            bare_label = sentence.text.strip().casefold() in _BARE_CATEGORY_LABELS[field]
            source_rows = bounded_rows[1:] if bare_label else bounded_rows
            source_text = " ".join(row.text.strip() for row in source_rows if row.text.strip())
            predicate = _declaration_predicate(
                field,
                text,
                author_aliases=author_aliases,
                source_text=source_text,
            )
            if field == "funding_statement" and not bare_label:
                predicate = predicate and _funding_anchor_qualifies(sentence.text)
            text_ids, paragraph_ids, pages = _candidate_location(bounded_rows)
            flags = ["lexical_anchor", "same_paragraph"]
            if any(_categories_in(row.text) - {field} for row in bounded_rows):
                flags.append("category_boundary_clipped")
            if any(_boilerplate_boundary_for_field(field, row.text) for row in bounded_rows):
                flags.append("boilerplate_boundary_clipped")
            if predicate:
                flags.append("declaration_predicate")
            else:
                flags.append("rejected")
            candidates.append(
                IntegrityStatementCandidate(
                    field=field,
                    method="anchored_paragraph",
                    heading=section.header if section is not None else None,
                    section_ids=(sentence.section_id,),
                    text_ids=text_ids,
                    paragraph_ids=paragraph_ids,
                    pages=pages,
                    classification_source=(
                        section.classification_source if section is not None else None
                    ),
                    classification_score=(
                        section.classification_score if section is not None else None
                    ),
                    reason_flags=tuple(flags),
                    accepted=bool(text_ids and predicate),
                )
            )
    return candidates


def _render_candidate(contents: PaperContents, candidate: IntegrityStatementCandidate) -> str:
    by_id = {sentence.text_id: sentence for sentence in contents.sentences}
    parts = []
    for text_id in candidate.text_ids:
        sentence = by_id.get(text_id)
        if sentence is None or sentence.is_display_formula:
            continue
        text = _bounded_sentence_for_field(candidate.field, sentence.text.strip())
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _render_legacy_candidate(
    contents: PaperContents, candidate: IntegrityStatementCandidate
) -> str:
    """Render a frozen compatibility snapshot without active-mode clipping."""
    by_id = {sentence.text_id: sentence for sentence in contents.sentences}
    rows = [
        sentence
        for text_id in candidate.text_ids
        if (sentence := by_id.get(text_id)) is not None and not sentence.is_display_formula
    ]
    if candidate.method == "legacy_section_copy":
        return _legacy_section_text(rows)
    return " ".join(sentence.text.strip() for sentence in rows if sentence.text.strip()).strip()


def _render_indices(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    indices: tuple[int, ...],
) -> str | None:
    parts = [
        rendered
        for index in indices
        if (
            rendered := (
                _render_legacy_candidate(contents, resolution.candidates[index])
                if resolution.candidates[index].method.startswith("legacy_")
                else _render_candidate(contents, resolution.candidates[index])
            )
        )
    ]
    return "\n\n".join(parts) if parts else None


def render_integrity_statement(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    field: str,
    *,
    effective: bool = True,
) -> str | None:
    """Render a resolved field from current sentence text (after late cleaning)."""
    if resolution.mode != "active" or not effective:
        return dict(resolution.legacy_statement_snapshots).get(field)
    return _render_indices(
        contents, resolution, resolution.candidate_indices(field, effective=effective)
    )


def render_selected_integrity_statement(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    field: str,
) -> str | None:
    """Render the bounded selected candidate independent of rollout mode."""
    return _render_indices(contents, resolution, resolution.selected_indices(field))


def _legacy_indices(
    candidates: list[IntegrityStatementCandidate],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            field,
            tuple(
                index
                for index, candidate in enumerate(candidates)
                if candidate.field == field and candidate.method.startswith("legacy_")
            ),
        )
        for field in _FIELDS
    )


def _active_indices(
    contents: PaperContents,
    candidates: list[IntegrityStatementCandidate],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    section_position = {
        section.section_id: index for index, section in enumerate(contents.sections)
    }
    selected: list[tuple[str, tuple[int, ...]]] = []
    for field in _FIELDS:
        section_candidates = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.field == field
            and candidate.accepted
            and candidate.method == "trusted_section"
        ]
        if section_candidates:
            first = min(
                section_candidates,
                key=lambda index: (
                    0 if "strong_heading" in candidates[index].reason_flags else 1,
                    section_position.get(candidates[index].section_ids[0], 10**9),
                ),
            )
            chosen = [first]
            position = section_position.get(candidates[first].section_ids[0], -1)
            by_position = {
                section_position.get(candidates[index].section_ids[0], -1): index
                for index in section_candidates
            }
            previous_position = position - 1
            while previous_position in by_position:
                chosen.insert(0, by_position[previous_position])
                previous_position -= 1
            next_position = position + 1
            while next_position in by_position:
                chosen.append(by_position[next_position])
                next_position += 1
            selected.append((field, tuple(chosen)))
            continue
        lexical = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if candidate.field == field
                and candidate.method == "anchored_paragraph"
                and candidate.accepted
            ),
            None,
        )
        selected.append((field, () if lexical is None else (lexical,)))
    return tuple(selected)


def _comparison_issues(
    contents: PaperContents,
    candidates: tuple[IntegrityStatementCandidate, ...],
    legacy: tuple[tuple[str, tuple[int, ...]], ...],
    selected: tuple[tuple[str, tuple[int, ...]], ...],
    legacy_snapshots: tuple[tuple[str, str | None], ...],
) -> tuple[ValidationIssue, ...]:
    temporary = IntegrityStatementResolution(
        mode="shadow",
        candidates=candidates,
        legacy_candidate_indices=legacy,
        selected_candidate_indices=selected,
        legacy_statement_snapshots=legacy_snapshots,
        issues=(),
    )
    issues = []
    for field in _FIELDS:
        legacy_indices = dict(legacy).get(field, ())
        selected_indices = dict(selected).get(field, ())
        legacy_text = dict(legacy_snapshots).get(field)
        selected_text = _render_indices(contents, temporary, dict(selected).get(field, ()))
        if _comparison_key(candidates, legacy_indices, legacy_text) == _comparison_key(
            candidates, selected_indices, selected_text
        ):
            continue
        implicated = _ordered_unique((*legacy_indices, *selected_indices))
        section_ids = _ordered_unique(
            section_id for index in implicated for section_id in candidates[index].section_ids
        )
        text_ids = _ordered_unique(
            text_id for index in implicated for text_id in candidates[index].text_ids
        )
        evidence_ids = (
            field,
            *(f"section:{section_id}" for section_id in section_ids),
            *(f"text:{text_id}" for text_id in text_ids),
        )[:20]
        issues.append(
            ValidationIssue(
                code="VAL_STATEMENT_SUSPECT",
                severity=IssueSeverity.WARNING,
                message=f"Legacy and bounded integrity-statement ownership differ for {field}",
                origin_stage="post_parse",
                evidence_ids=evidence_ids,
            )
        )
    return tuple(issues)


def _comparison_key(
    candidates: tuple[IntegrityStatementCandidate, ...],
    indices: tuple[int, ...],
    text: str | None,
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    from bibr.input.consolidate_text import clean_text_content_late

    normalized = " ".join(clean_text_content_late(text or "").casefold().split())
    return (
        normalized,
        _ordered_unique(
            section_id for index in indices for section_id in candidates[index].section_ids
        ),
        _ordered_unique(text_id for index in indices for text_id in candidates[index].text_ids),
        _ordered_unique(
            paragraph_id for index in indices for paragraph_id in candidates[index].paragraph_ids
        ),
        _ordered_unique(page for index in indices for page in candidates[index].pages),
    )


def resolve_integrity_statements(
    contents: PaperContents,
    *,
    mode: IntegrityStatementMode,
    author_names: tuple[tuple[str, str], ...] = (),
) -> IntegrityStatementResolution:
    """Build section and lexical candidates, then select compatibility/safe IDs."""
    if mode not in {"legacy", "shadow", "active"}:
        raise ValueError(f"unsupported integrity statement mode: {mode}")
    author_aliases = _author_aliases(author_names)
    rows_by_section = _section_rows(contents)
    section_by_id = {section.section_id: section for section in contents.sections}
    candidates = _build_legacy_snapshot_candidates(contents, rows_by_section, section_by_id)
    for section in contents.sections:
        candidates.extend(
            _build_section_candidates(
                section,
                rows_by_section.get(section.section_id, []),
                author_aliases=author_aliases,
            )
        )
    linear_rows = [sentence for sentence in contents.sentences if not sentence.is_display_formula]
    candidates.extend(
        _build_lexical_candidates(
            linear_rows,
            section_by_id,
            author_aliases=author_aliases,
        )
    )
    legacy = _legacy_indices(candidates)
    selected = _active_indices(contents, candidates)
    frozen_candidates = tuple(candidates)
    snapshot_resolution = IntegrityStatementResolution(
        mode="legacy",
        candidates=frozen_candidates,
        legacy_candidate_indices=legacy,
        selected_candidate_indices=selected,
        legacy_statement_snapshots=(),
        issues=(),
    )
    legacy_snapshots = tuple(
        (
            field,
            _render_indices(contents, snapshot_resolution, dict(legacy).get(field, ())),
        )
        for field in _FIELDS
    )
    issues = (
        _comparison_issues(contents, frozen_candidates, legacy, selected, legacy_snapshots)
        if mode == "shadow"
        else ()
    )
    return IntegrityStatementResolution(
        mode=mode,
        candidates=frozen_candidates,
        legacy_candidate_indices=legacy,
        selected_candidate_indices=selected,
        legacy_statement_snapshots=legacy_snapshots,
        issues=issues,
    )


def apply_integrity_resolution(
    contents: PaperContents,
    metadata: PaperMetadata,
    resolution: IntegrityStatementResolution,
) -> None:
    """Materialize mode-effective scalars without replacing native values."""
    native_metadata = contents.preparsed_metadata is not None
    for field in _FIELDS:
        current = getattr(metadata, field)
        selected = resolution.candidate_indices(field)
        if current is not None and (native_metadata or resolution.mode != "active" or not selected):
            continue
        rendered = render_integrity_statement(contents, resolution, field)
        setattr(metadata, field, rendered)
        if rendered is None:
            continue
        lexical_methods = {resolution.candidates[index].method for index in selected}
        if lexical_methods & {"legacy_lexical_capture", "anchored_paragraph"}:
            warning = f"STATEMENT_LEXICAL_FALLBACK: {field}"
            if warning not in contents.processing_warnings:
                contents.processing_warnings.append(warning)
