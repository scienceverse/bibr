"""Spatial front-matter candidates, record blocks, and deterministic selection.

Sentence and section text are the authoritative source.  ``RegionSummary`` is
used only to recover layout metadata and the parser's original region order;
its deliberately short content preview is never promoted into candidate text.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from bibr.paper_contents import (
    CANONICAL_SECTION_ALIASES,
    FRONT_MATTER_FURNITURE_LABELS,
    FRONT_MATTER_MASTHEAD_RE,
    CanonicalSection,
    is_exact_front_matter_furniture,
)
from bibr.utils.text import normalize_doi
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.extract.front_role import FrontRolePredictions, RoleScores
    from bibr.paper_contents import PaperContents, PaperSentence, RegionSummary
    from bibr.pipeline.identity import ExpectedIdentity

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True)
class FrontRolePolicy:
    """How much the front-role classifier's scores count as evidence.

    Defaults mirror ``ML_FRONT_ROLE_MIN_CONFIDENCE`` /
    ``ML_FRONT_ROLE_MASTHEAD_CONFIDENCE``; ``from_settings`` reads the live values.
    """

    min_confidence: float = 0.5
    masthead_confidence: float = 0.8
    record_root_confidence: float = 0.9

    @classmethod
    def from_settings(cls, settings: GlobalSettings | None) -> FrontRolePolicy:
        if settings is None:
            return cls()
        ml = settings.ml
        return cls(
            min_confidence=float(getattr(ml, "front_role_min_confidence", 0.5)),
            masthead_confidence=float(getattr(ml, "front_role_masthead_confidence", 0.8)),
            record_root_confidence=float(getattr(ml, "front_role_record_root_confidence", 0.9)),
        )


def _scores_for(
    predictions: FrontRolePredictions | None,
    summary: RegionSummary | None,
) -> RoleScores | None:
    if predictions is None or summary is None:
        return None
    return predictions.get(summary.page, summary.index)


def _model_role(scores: RoleScores | None, role: str, threshold: float) -> bool:
    return scores is not None and scores.get(role) >= threshold


def _model_denies_title_seed(scores: RoleScores | None, policy: FrontRolePolicy) -> bool:
    """The classifier scored this row and is confident it is not a title.

    Used only to deny a title seed the right to *root a record*; the title role
    itself is untouched, so a page whose only title seed is model-denied still
    reports that title.
    """

    if scores is None or scores.top == "title":
        return False
    if scores.get("title") >= policy.min_confidence:
        return False
    return scores.confidence >= policy.record_root_confidence


_BODY_SECTION_TYPES = frozenset(
    {
        CanonicalSection.INTRODUCTION,
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
    }
)
_FRONT_MATTER_SECTION_TYPES = frozenset(
    {
        CanonicalSection.TITLE,
        CanonicalSection.ABSTRACT,
        CanonicalSection.KEYWORDS,
        CanonicalSection.UNKNOWN,
    }
)
# Bare section headings may be misclassified as titles. The shared alias table recognizes
# translated headings without treating them as new front-matter records.
_MULTILINGUAL_SECTION_HEADINGS = frozenset(
    {
        "abstrak",
        "bibliografi",
        "bibliografia",
        "bibliografía",
        "hasil dan pembahasan",
        "kata kunci",
        "palabras clave",
        "referencias",
        "referências",
        "resumen",
        "resumo",
        "résumé",
        "zusammenfassung",
        "аннотация",
        "ключевые слова",
    }
)
_ORDINARY_HEADING_TEXT = (
    frozenset(
        alias
        for section_type in _BODY_SECTION_TYPES
        | {CanonicalSection.ABSTRACT, CanonicalSection.KEYWORDS, CanonicalSection.REFERENCES}
        for alias in CANONICAL_SECTION_ALIASES.get(section_type, ())
    )
    | _MULTILINGUAL_SECTION_HEADINGS
)
_STRUCTURAL_LABELS = frozenset({"header", "footer"})
_HEADING_LABELS = frozenset({"doc_title", "paragraph_title"})
AFFILIATION_MARKERS = (
    "department",
    "division",
    "faculty",
    "school",
    "college",
    "university",
    "universite",
    "université",
    "institute",
    "institution",
    "hospital",
    "centre",
    "center",
    "laboratory",
    "academy",
)
AFFILIATION_MARKER_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(marker) for marker in AFFILIATION_MARKERS)})\b",
    re.IGNORECASE,
)
_ABSTRACT_HEADING_RE = re.compile(
    r"^(?:abstract|background(?: and objectives)?|objectives?|methods?|results?|conclusions?)$",
    re.IGNORECASE,
)
# Defined in ``bibr.paper_contents`` so running-header detection can consult the
# same pattern one stage earlier; kept aliased here for readability.
_MASTHEAD_RE = FRONT_MATTER_MASTHEAD_RE
# Article-type labels, publisher badges, and information-box headings are page furniture rather
# than independent article titles. Match complete labels only.
_FRONT_MATTER_FURNITURE_LABELS = FRONT_MATTER_FURNITURE_LABELS
_NAME_PARTICLES = frozenset(
    {
        "al",
        "bin",
        "da",
        "de",
        "del",
        "der",
        "di",
        "dos",
        "du",
        "la",
        "le",
        "van",
        "von",
        "y",
    }
)
# Marks a row the section classifier called TITLE with no corroborating layout
# evidence, on text that is byline-shaped. See ``_candidate_roles``.
CLASSIFIED_BYLINE_TITLE_ROLE = "classified_byline_title"
BYLINE_PROBATION_ROLE = "byline_probation"
MODEL_NON_TITLE_SEED_ROLE = "model_non_title_seed"
BODY_HEADING_ROLE = "body_heading"
_NUMBERED_BODY_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[IVX]+[.)])\s+", re.I)
_NAME_LIST_SEPARATOR_RE = re.compile(r"\s*[;·•‣⁃∙⋅]\s*")
_CONTRIBUTION_ROLE_RE = re.compile(
    r"\b(?:conceptuali[sz]ation|data\s+curation|formal\s+analysis|funding\s+acquisition|"
    r"investigation|methodology|project\s+administration|resources?|software|supervision|"
    r"validation|visuali[sz]ation|writing|original\s+draft|review|editing)\b",
    re.IGNORECASE,
)
_REFERENCE_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
_REFERENCE_LOCATOR_RE = re.compile(
    r"\b(?:\d+(?:\s*\(\d+\))?\s*:\s*[A-Z]?\d+|"
    r"p{1,2}\.?\s*\d+\s*[-–—]\s*\d+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FrontMatterCandidate:
    candidate_id: str
    source_kind: str
    reading_order: int
    page: int | None
    bbox: tuple[float, float, float, float] | None
    region_label: str | None
    font_size: float | None
    font_bold: bool | None
    section_id: int | None
    text_ids: tuple[int, ...]
    paragraph_id: int | None
    raw_text: str
    normalized_text: str
    roles: frozenset[str]
    # Roles the front-role classifier contributed (subset of ``roles``) and its
    # top scores, for audit trails; empty when the model was absent or silent.
    model_roles: frozenset[str] = frozenset()
    model_scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class FrontMatterBlock:
    """A contiguous candidate group rooted at one plausible article title."""

    block_id: str
    candidate_ids: tuple[str, ...]
    title_candidate_ids: tuple[str, ...]
    pages: tuple[int, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    normalized_text: str = ""
    source_block_ids: tuple[str, ...] = ()
    merge_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrontMatterResolution:
    candidates: tuple[FrontMatterCandidate, ...]
    blocks: tuple[FrontMatterBlock, ...]
    selected_block_id: str | None
    selection_method: str
    reason_flags: tuple[str, ...]
    allowed_text_ids: frozenset[int]
    allowed_section_ids: frozenset[int]


@dataclass(frozen=True)
class _CandidateDraft:
    source_kind: str
    source_order: int
    region_order: tuple[int, int] | None
    page: int | None
    bbox: tuple[float, float, float, float] | None
    region_label: str | None
    font_size: float | None
    font_bold: bool | None
    section_id: int | None
    text_ids: tuple[int, ...]
    paragraph_id: int | None
    raw_text: str
    section_type: CanonicalSection
    byline_probation: bool = False


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.casefold().split())


def _bbox_tuple(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _bbox_union(
    values: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not values:
        return None
    return (
        min(value[0] for value in values),
        min(value[1] for value in values),
        max(value[2] for value in values),
        max(value[3] for value in values),
    )


def _same_bbox(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> bool:
    if left is None or right is None:
        return False
    return all(abs(a - b) <= 1e-6 for a, b in zip(left, right, strict=True))


def _matching_region_summary(
    summaries: list[RegionSummary],
    *,
    page: int | None,
    bbox: tuple[float, float, float, float] | None,
    section_id: int | None,
) -> RegionSummary | None:
    """Match source geometry to a region without consuming its text preview."""

    if page is None:
        return None
    page_rows = [summary for summary in summaries if summary.page == page]
    bbox_rows = [summary for summary in page_rows if _same_bbox(summary.bbox, bbox)]
    if section_id is not None:
        section_rows = [summary for summary in bbox_rows if summary.section_id == section_id]
        if len(section_rows) == 1:
            return section_rows[0]
    if len(bbox_rows) == 1:
        return bbox_rows[0]
    return None


def _sentence_pages(sentence: PaperSentence) -> set[int]:
    provenance_pages = {item.page_no for item in sentence.provenance}
    if provenance_pages:
        return provenance_pages
    return {sentence.page_number} if sentence.page_number is not None else set()


def _sentence_bbox(sentence: PaperSentence) -> tuple[float, float, float, float] | None:
    boxes = [provenance.bbox for provenance in sentence.provenance if provenance.bbox is not None]
    return _bbox_union(boxes)


def _first_page(contents: PaperContents) -> int | None:
    pages = {page for sentence in contents.sentences for page in _sentence_pages(sentence)}
    return min(pages) if pages else None


def _paragraph_drafts(
    contents: PaperContents,
    *,
    first_page: int | None,
    allow_byline_probation: bool,
    policy: FrontRolePolicy | None = None,
) -> list[_CandidateDraft]:
    policy = policy or FrontRolePolicy()
    predictions = contents.front_role_predictions
    section_map = {section.section_id: section for section in contents.sections}
    groups: OrderedDict[tuple[int, int], list[tuple[int, PaperSentence]]] = OrderedDict()
    # Groups admitted only on the promise of being a byline: the section
    # classifier routinely mistypes a first-page byline block ("Presenters" →
    # acknowledgment, an author-and-affiliation stack → author_contributions),
    # and dropping them here starves every downstream byline heuristic.
    byline_probation: set[tuple[int, int]] = set()
    for source_order, sentence in enumerate(contents.sentences):
        section = section_map.get(sentence.section_id)
        section_type = section.section_type if section is not None else CanonicalSection.UNKNOWN
        key = (sentence.section_id, sentence.paragraph_id)
        if section_type not in _FRONT_MATTER_SECTION_TYPES and sentence.section_id != 0:
            if not allow_byline_probation:
                continue
            if first_page is None or first_page not in _sentence_pages(sentence):
                continue
            byline_probation.add(key)
        groups.setdefault(key, []).append((source_order, sentence))

    drafts: list[_CandidateDraft] = []
    summaries = list(contents.region_summaries or [])
    for (section_id, paragraph_id), members in groups.items():
        raw_text = " ".join(
            sentence.text.strip() for _, sentence in members if sentence.text.strip()
        )
        if not raw_text:
            continue
        first_order, first = members[0]
        unique_pages = {page for _, sentence in members for page in _sentence_pages(sentence)}
        page = next(iter(unique_pages)) if len(unique_pages) == 1 else None
        boxes = [bbox for _, sentence in members if (bbox := _sentence_bbox(sentence)) is not None]
        # A union across pages has no meaningful coordinate system.  Preserve
        # geometry only when the entire paragraph belongs to one known page.
        bbox = _bbox_union(boxes) if len(unique_pages) == 1 else None
        metas = [sentence.region_meta for _, sentence in members if sentence.region_meta]
        first_meta = metas[0] if metas else {}
        matched_summaries = {
            (summary.page, summary.index): summary
            for _, sentence in members
            for provenance in sentence.provenance
            if (
                summary := _matching_region_summary(
                    summaries,
                    page=provenance.page_no,
                    bbox=provenance.bbox,
                    section_id=section_id,
                )
            )
            is not None
        }
        summary = matched_summaries[min(matched_summaries)] if matched_summaries else None
        if (section_id, paragraph_id) in byline_probation and not (
            _looks_like_byline(raw_text, _normalize_text(raw_text), source_kind="paragraph")
            or _model_role(_scores_for(predictions, summary), "byline", policy.min_confidence)
        ):
            continue
        region_label = first_meta.get("region_type") or (summary.label if summary else None)
        font_size = first_meta.get("font_size")
        if font_size is None and summary is not None:
            font_size = summary.font_size
        font_bold = first_meta.get("font_bold")
        if font_bold is None and summary is not None:
            font_bold = summary.font_bold
        section = section_map.get(section_id)
        section_type = section.section_type if section is not None else CanonicalSection.UNKNOWN
        drafts.append(
            _CandidateDraft(
                source_kind="paragraph",
                # Odd slots leave an insertion point for a source heading
                # immediately before the first paragraph in its section.
                source_order=first_order * 2 + 1,
                region_order=(summary.page, summary.index) if summary is not None else None,
                page=page,
                bbox=bbox,
                region_label=region_label,
                font_size=float(font_size) if isinstance(font_size, (int, float)) else None,
                font_bold=font_bold if isinstance(font_bold, bool) else None,
                section_id=section_id,
                text_ids=tuple(sentence.text_id for _, sentence in members),
                paragraph_id=paragraph_id,
                raw_text=raw_text,
                section_type=section_type,
                byline_probation=(section_id, paragraph_id) in byline_probation,
            )
        )
    return drafts


def _heading_drafts(
    contents: PaperContents,
    *,
    paragraph_drafts: list[_CandidateDraft],
    first_page: int | None,
    allow_byline_probation: bool,
    policy: FrontRolePolicy | None = None,
) -> list[_CandidateDraft]:
    policy = policy or FrontRolePolicy()
    predictions = contents.front_role_predictions
    summaries = list(contents.region_summaries or [])
    drafts: list[_CandidateDraft] = []
    for section_order, section in enumerate(contents.sections):
        if section.header_is_synthetic:
            continue
        if section.level <= 0 or not section.header.strip():
            continue
        page = section.provenance[0].page_no if section.provenance else None
        boxes = [item.bbox for item in section.provenance if item.bbox is not None]
        bbox = _bbox_union(boxes)
        summary = _matching_region_summary(
            summaries,
            page=page,
            bbox=bbox,
            section_id=section.section_id,
        )
        probation = section.section_type not in _FRONT_MATTER_SECTION_TYPES
        if probation:
            # A byline promoted to a section header (common for Cyrillic and
            # Latin-American layouts where the byline is set above the
            # affiliation) is mistyped, not body text.
            if not allow_byline_probation:
                continue
            header = section.header.strip()
            if page is None or page != first_page:
                continue
            if not (
                _looks_like_byline(header, _normalize_text(header), source_kind="heading")
                or _model_role(_scores_for(predictions, summary), "byline", policy.min_confidence)
            ):
                continue
        following_orders = [
            draft.source_order
            for draft in paragraph_drafts
            if draft.section_id is not None and draft.section_id >= section.section_id
        ]
        if following_orders:
            source_order = min(following_orders) - 1
        else:
            source_order = (
                max((draft.source_order for draft in paragraph_drafts), default=-1)
                + section_order
                + 1
            )
        drafts.append(
            _CandidateDraft(
                source_kind="heading",
                source_order=source_order,
                region_order=(summary.page, summary.index) if summary is not None else None,
                page=page,
                bbox=bbox,
                region_label=summary.label if summary is not None else "paragraph_title",
                font_size=summary.font_size if summary is not None else None,
                font_bold=summary.font_bold if summary is not None else None,
                section_id=section.section_id,
                text_ids=(),
                paragraph_id=None,
                raw_text=section.header.strip(),
                section_type=section.section_type,
                byline_probation=probation,
            )
        )
    return drafts


def _source_sort_key(draft: _CandidateDraft) -> tuple[int, int]:
    return (draft.source_order, 0 if draft.source_kind == "heading" else 1)


def _is_ordinary_heading(normalized: str, section_type: CanonicalSection) -> bool:
    return section_type in _BODY_SECTION_TYPES or normalized in _ORDINARY_HEADING_TEXT


def _starts_with_uppercase_title(text: str) -> bool:
    words = _WORD_RE.findall(text)
    # Proceedings parsers can place title, byline, affiliations, and the full
    # abstract in one paragraph.  Only inspect the leading title-sized window;
    # a total-length cap would miss exactly those multi-record pages.
    if len(words) < 4:
        return False
    leading = words[: min(10, len(words))]
    uppercase = sum(word.isupper() and len(word) > 1 for word in leading)
    return uppercase >= max(4, round(len(leading) * 0.6))


def _has_prohibited_separator_evidence(
    text: str,
    normalized: str,
    *,
    include_affiliation: bool = True,
) -> bool:
    return bool(
        _DOI_RE.search(text)
        or normalized in _ORDINARY_HEADING_TEXT
        or (include_affiliation and AFFILIATION_MARKER_RE.search(text))
        or _CONTRIBUTION_ROLE_RE.search(text)
        or _REFERENCE_YEAR_RE.search(text)
        or _REFERENCE_LOCATOR_RE.search(text)
    )


# Function words never appear inside a printed person-name chunk (lowercase
# surname particles live in _NAME_PARTICLES instead), but they are routine in
# topic, banner, and journal-name chunks ("Journal of ...", "History of ...").
_NON_NAME_CHUNK_WORDS = frozenset(
    {"among", "and", "for", "from", "in", "of", "on", "the", "to", "with"}
)


def _looks_like_separator_name_list(text: str, normalized: str) -> bool:
    """Return whether text has a bounded person-list *shape*, not person identity."""

    if len(text) > 1_500 or _has_prohibited_separator_evidence(text, normalized):
        return False
    words = _WORD_RE.findall(text)
    if len(words) > 160:
        return False
    chunks = _NAME_LIST_SEPARATOR_RE.split(text)
    if len(chunks) < 2:
        return False
    for chunk in chunks:
        chunk_words = _WORD_RE.findall(chunk)
        if not 2 <= len(chunk_words) <= 6:
            return False
        if any(word.casefold() in _NON_NAME_CHUNK_WORDS for word in chunk_words):
            return False
        if _MASTHEAD_RE.search(chunk.strip()) or is_exact_front_matter_furniture(chunk):
            return False
        name_like = sum(
            bool(word[:1].isupper()) or word.casefold() in _NAME_PARTICLES for word in chunk_words
        )
        if name_like * 2 < len(chunk_words):
            return False
    return True


# Superscript affiliation/correspondence markers, which sit *inside* a byline and
# would otherwise shatter it into numeric chunks ("Lane 1,2 , Edwards3 *").
_BYLINE_MARKER_RE = re.compile(r"[\d*†‡§¶#]+")
_BYLINE_CHUNK_SPLIT_RE = re.compile(r"[,;]|\band\b|&", re.IGNORECASE)


def _looks_like_long_name_list(text: str) -> bool:
    """Whether *text* is a consortium-scale list of printed person names.

    Long bylines can exceed ordinary size caps. Wider caps require independent
    name-list evidence so ordinary prose does not become eligible.
    """
    chunks = [
        chunk.strip()
        for chunk in _BYLINE_CHUNK_SPLIT_RE.split(_BYLINE_MARKER_RE.sub(" ", text))
        if chunk.strip()
    ]
    if len(chunks) < 4:
        return False
    name_shaped = 0
    for chunk in chunks:
        words = _WORD_RE.findall(chunk)
        if not 2 <= len(words) <= 5:
            continue
        if any(word.casefold() in _NON_NAME_CHUNK_WORDS for word in words):
            continue
        name_like = sum(
            bool(word[:1].isupper()) or word.casefold() in _NAME_PARTICLES for word in words
        )
        if name_like == len(words):
            name_shaped += 1
    return name_shaped >= 4 and name_shaped >= 0.75 * len(chunks)


def _looks_like_legacy_byline(text: str, normalized: str, *, source_kind: str) -> bool:
    if _DOI_RE.search(text) or normalized in _ORDINARY_HEADING_TEXT:
        return False
    # Ordinary caps, widened to the separator-shape limits for a byline that has
    # already proven itself a long name list.
    max_chars, max_words = (1_500, 160) if _looks_like_long_name_list(text) else (400, 45)
    if len(text) > max_chars:
        return False
    words = _WORD_RE.findall(text)
    if len(words) < 2 or len(words) > max_words:
        return False
    name_like = sum(
        bool(word[:1].isupper()) or word.casefold() in _NAME_PARTICLES for word in words
    )
    ratio = name_like / len(words)
    if source_kind == "heading":
        return ratio >= 0.65
    if "," in text:
        return ratio >= 0.5
    lowered = [word.casefold() for word in words]
    joiner_positions = [index for index, word in enumerate(lowered) if word in {"and", "&"}]
    balanced_joiner = any(index >= 2 and len(words) - index - 1 >= 2 for index in joiner_positions)
    title_prepositions = {"among", "for", "from", "in", "of", "on", "the", "to", "with"}
    return balanced_joiner and not title_prepositions.intersection(lowered) and ratio >= 0.5


def _leading_byline_before_affiliation(
    text: str,
    *,
    source_kind: str,
    allow_separator_shape: bool = True,
) -> bool:
    chunks = _NAME_LIST_SEPARATOR_RE.split(text)
    affiliation_index = next(
        (index for index, chunk in enumerate(chunks) if AFFILIATION_MARKER_RE.search(chunk)),
        None,
    )
    if affiliation_index is None or affiliation_index == 0:
        return False
    leading = " · ".join(chunks[:affiliation_index])
    normalized = _normalize_text(leading)
    if _has_prohibited_separator_evidence(
        leading,
        normalized,
        include_affiliation=False,
    ):
        return False
    return (
        allow_separator_shape
        and _looks_like_separator_name_list(
            leading,
            normalized,
        )
    ) or _looks_like_legacy_byline(
        leading,
        normalized,
        source_kind=source_kind,
    )


def _looks_like_byline(text: str, normalized: str, *, source_kind: str) -> bool:
    # Separator-rich rows are lexically indistinguishable from topic, location,
    # acronym, and numeric-label lists. They can become contextual bylines only
    # after ownership selection proves a trusted title/list/affiliation sequence.
    if _NAME_LIST_SEPARATOR_RE.search(text):
        if AFFILIATION_MARKER_RE.search(text):
            # Preserve the established high-confidence composite form where a
            # legacy comma/and byline precedes an affiliation in the same row.
            return _leading_byline_before_affiliation(
                text,
                source_kind=source_kind,
                allow_separator_shape=False,
            )
        return False
    if AFFILIATION_MARKER_RE.search(text) or _CONTRIBUTION_ROLE_RE.search(text):
        return False
    return _looks_like_legacy_byline(text, normalized, source_kind=source_kind)


# Middle initials and trailing affiliation superscripts help distinguish a byline from an
# article-type label.
_NAME_INITIAL_RE = re.compile(r"(?<![^\W\d_])([^\W\d_])\.", re.UNICODE)
# A name-length word carrying trailing affiliation markers ("DeKay1", "Kim1,5").
# The three-letter minimum keeps chemical and viral names ("D3", "CO2") out.
_AFFILIATION_SUPERSCRIPT_RE = re.compile(
    r"[^\W\d_]{3,}\d{1,2}(?:\s*,\s*\d{1,2})*",
    re.UNICODE,
)


def _has_person_name_evidence(text: str) -> bool:
    """Whether *text* carries a middle initial or an affiliation superscript."""

    if any(match.group(1).isupper() for match in _NAME_INITIAL_RE.finditer(text)):
        return True
    return bool(_AFFILIATION_SUPERSCRIPT_RE.search(text))


def _looks_like_affiliation(text: str) -> bool:
    return bool(AFFILIATION_MARKER_RE.search(text))


def _looks_like_masthead(text: str, label: str) -> bool:
    return label in _STRUCTURAL_LABELS or bool(_MASTHEAD_RE.match(text.strip()))


def _paragraph_title_evidence(
    draft: _CandidateDraft,
    *,
    label: str,
    words: list[str],
    abstract_owned: bool,
) -> bool:
    """Return strong parser evidence for a paragraph-title article title."""

    if (
        draft.source_kind != "paragraph"
        or label != "paragraph_title"
        or abstract_owned
        or draft.raw_text.rstrip().endswith((".", ":", ";"))
        or not 4 <= len(words) <= 30
        or _DOI_RE.search(draft.raw_text)
        or _looks_like_affiliation(draft.raw_text)
        or _looks_like_masthead(draft.raw_text, label)
    ):
        return False
    titlecase_ratio = sum(word[:1].isupper() for word in words) / len(words)
    return titlecase_ratio >= 0.5 or _starts_with_uppercase_title(draft.raw_text)


def _overloaded_abstract_section_ids(drafts: list[_CandidateDraft]) -> frozenset[int]:
    """Detect proceedings pages whose single Abstract section owns many records.

    Normal structured abstracts retain authoritative ABSTRACT ownership.  The
    exception requires at least two title-shaped rows that each own strong
    record anatomy, which preserves the known proceedings parser shape without
    letting arbitrary structured subheadings become record seeds.
    """

    by_section: dict[int, list[_CandidateDraft]] = {}
    for draft in drafts:
        if draft.section_id is not None and draft.section_type == CanonicalSection.ABSTRACT:
            by_section.setdefault(draft.section_id, []).append(draft)

    overloaded: set[int] = set()
    for section_id, rows in by_section.items():
        potential: list[int] = []
        for index, draft in enumerate(rows):
            label = (draft.region_label or "").casefold()
            normalized = _normalize_text(draft.raw_text)
            words = _WORD_RE.findall(draft.raw_text)
            embedded = bool(
                _starts_with_uppercase_title(draft.raw_text)
                and len(draft.raw_text) > 100
                and "," not in draft.raw_text[:80]
            )
            title_shape = _paragraph_title_evidence(
                draft,
                label=label,
                words=words,
                abstract_owned=False,
            ) or (_starts_with_uppercase_title(draft.raw_text) and embedded)
            if (
                title_shape
                and label != "abstract"
                and not _is_ordinary_heading(normalized, draft.section_type)
            ):
                potential.append(index)

        developed = 0
        for position, index in enumerate(potential):
            next_index = potential[position + 1] if position + 1 < len(potential) else len(rows)
            window = rows[index:next_index]
            has_doi = any(_DOI_RE.search(draft.raw_text) for draft in window)
            has_explicit_abstract = any(
                (draft.region_label or "").casefold() == "abstract" for draft in window
            )
            has_affiliation = any(_looks_like_affiliation(draft.raw_text) for draft in window)
            has_byline = any(
                _looks_like_byline(
                    draft.raw_text,
                    _normalize_text(draft.raw_text),
                    source_kind=draft.source_kind,
                )
                for draft in window[1:]
            )
            has_embedded_byline = any(
                len(draft.raw_text) > 100
                and "," not in draft.raw_text[:80]
                and bool(re.search(r"(?:,|\band\b|\b&\b)", draft.raw_text, re.IGNORECASE))
                for draft in window
            )
            if has_doi or (
                (has_byline or has_embedded_byline) and (has_affiliation or has_explicit_abstract)
            ):
                developed += 1
        if developed >= 2:
            overloaded.add(section_id)
    return frozenset(overloaded)


def _is_false_title_seed(draft: _CandidateDraft) -> bool:
    """Whether the row is printed journal furniture that must not root a record.

    Article-type kickers, badges, and information-box headings can be misclassified as TITLE sections and split the real record. Use anchored full-string matching so substrings of genuine titles remain eligible.
    """

    return is_exact_front_matter_furniture(draft.raw_text)


def _candidate_roles(
    draft: _CandidateDraft,
    normalized: str,
    *,
    detected_title: str | None,
    allow_abstract_title: bool,
    scores: RoleScores | None = None,
    policy: FrontRolePolicy | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(roles, model_roles)``; ``model_roles`` is what the classifier added."""
    policy = policy or FrontRolePolicy()
    roles: set[str] = set()
    model_roles: set[str] = set()
    label = (draft.region_label or "").casefold()
    # Front-role classifier evidence (bibr/extract/front_role.py). Additive
    # for byline/affiliation/abstract/title; a confident masthead only vetoes
    # the title seed. A doc_title layout label is trusted over the veto.
    threshold = policy.min_confidence
    model_title = _model_role(scores, "title", threshold)
    model_byline = _model_role(scores, "byline", threshold)
    model_affiliation = _model_role(scores, "affiliation", threshold)
    model_abstract = _model_role(scores, "abstract", threshold)
    model_masthead = (
        _model_role(scores, "masthead", policy.masthead_confidence) and label != "doc_title"
    )
    if draft.source_kind == "heading" or label in _HEADING_LABELS:
        roles.add("heading")
    abstract_owned = draft.section_type == CanonicalSection.ABSTRACT and not allow_abstract_title
    if (
        abstract_owned
        or label == "abstract"
        or bool(_ABSTRACT_HEADING_RE.fullmatch(draft.raw_text.strip()))
    ):
        roles.add("abstract")
    elif model_abstract and not model_title:
        roles.add("abstract")
        model_roles.add("abstract")
    if _DOI_RE.search(draft.raw_text):
        roles.add("doi")
    if _looks_like_affiliation(draft.raw_text):
        roles.add("affiliation")
    elif model_affiliation and not model_title:
        roles.add("affiliation")
        model_roles.add("affiliation")

    detected = _normalize_text(detected_title or "")
    detected_match = bool(
        detected
        and (
            detected == normalized
            or (
                normalized.startswith(detected)
                and len(normalized) <= max(len(detected) * 3, len(detected) + 120)
            )
        )
    )
    if _is_false_title_seed(draft):
        return frozenset(roles), frozenset(model_roles)
    explicit_title = bool(
        not abstract_owned
        and (
            label == "doc_title"
            or (draft.source_kind == "heading" and draft.section_type == CanonicalSection.TITLE)
            or detected_match
            or model_title
        )
    )
    textual_seed = _starts_with_uppercase_title(draft.raw_text)
    lexical_byline = _looks_like_byline(draft.raw_text, normalized, source_kind=draft.source_kind)
    # The model sees geometry and script-independent shape, so it admits the
    # 18-author consortium byline the 45-word cap rejects and the Cyrillic or
    # CJK byline the Latin name shape cannot read.
    byline = lexical_byline or (model_byline and not model_title)
    words = _WORD_RE.findall(draft.raw_text)
    paragraph_title = _paragraph_title_evidence(
        draft,
        label=label,
        words=words,
        abstract_owned=abstract_owned,
    )
    # Some proceedings OCR regions contain title + author + affiliation in one
    # authoritative paragraph.  Permit that composite shape only when the
    # uppercase title-sized prefix precedes author-list punctuation; this does
    # not promote an all-uppercase author list whose commas start immediately.
    embedded_title = bool(
        textual_seed and len(draft.raw_text) > 100 and "," not in draft.raw_text[:80]
    )
    inferred_title = bool(
        not abstract_owned
        and (textual_seed or paragraph_title)
        and (
            paragraph_title
            or not byline
            or embedded_title
            or (draft.source_kind == "heading" and textual_seed)
        )
        and ("affiliation" not in roles or embedded_title)
        and "abstract" not in roles
        and "doi" not in roles
        and not _looks_like_masthead(draft.raw_text, label)
    )
    is_title = bool(
        (explicit_title or inferred_title)
        and not _looks_like_masthead(draft.raw_text, label)
        and not _is_ordinary_heading(normalized, draft.section_type)
        and not model_masthead
    )
    if is_title:
        roles.add("title")
        # "Correspondence", "A R T I C L E I N F O", "CITATION", "Key Features":
        # section headers the heading+TITLE seed admits and the classifier types
        # as headings at 1.00. Harmless as titles, ruinous as record roots — the
        # abstract that follows them is anatomy enough to develop a second
        # record, which cuts the real title away from it. Layout's own doc_title
        # and a match against the parser's detected title outrank the veto.
        if label != "doc_title" and not detected_match and _model_denies_title_seed(scores, policy):
            roles.add(MODEL_NON_TITLE_SEED_ROLE)
        if model_title and not (
            label == "doc_title"
            or (draft.source_kind == "heading" and draft.section_type == CanonicalSection.TITLE)
            or detected_match
            or inferred_title
        ):
            model_roles.add("title")
    # Parser paragraph_title evidence and classified/detected titles outrank a
    # broad punctuation-based byline shape.  Embedded proceedings candidates
    # intentionally retain both roles because they contain title + authors.
    embedded_byline = bool(
        embedded_title and re.search(r"(?:,|\band\b|\b&\b)", draft.raw_text, re.IGNORECASE)
    )
    if (byline and (not is_title or embedded_title)) or embedded_byline:
        roles.add("byline")
        if not lexical_byline and not embedded_byline:
            model_roles.add("byline")
    # The section classifier has no byline class and can assign TITLE to an author line. Avoid
    # splitting that line into a separate record when byline evidence is present.
    if (
        is_title
        and byline
        and _has_person_name_evidence(draft.raw_text)
        and not paragraph_title
        and not detected_match
        and label != "doc_title"
        and draft.source_kind == "heading"
        and draft.section_type == CanonicalSection.TITLE
    ):
        roles.add(CLASSIFIED_BYLINE_TITLE_ROLE)
    if draft.byline_probation:
        roles.add(BYLINE_PROBATION_ROLE)
    return frozenset(roles), frozenset(model_roles)


def collect_front_matter_candidates(
    contents: PaperContents,
    *,
    policy: FrontRolePolicy | None = None,
) -> tuple[FrontMatterCandidate, ...]:
    """Aggregate authoritative paragraph/heading text into immutable candidates.

    Byline probation is a *rescue*, not a widening.  Admitting page-1 rows from
    body-typed sections recovers papers whose byline the section classifier
    mistyped, but on papers that already print a byline it only adds noise —
    related-works citations and author-contribution lines are byline-shaped too,
    and they displace the real record.  So probation runs only on the second
    pass, when the ordinary front matter yields no byline at all.
    """

    policy = policy or FrontRolePolicy()
    candidates = _with_byline_probation(contents, policy, use_model=True)
    # The classifier is evidence, never a veto of last resort. Its negative
    # paths -- a confident masthead vetoing the title seed, an abstract or
    # affiliation score claiming the block -- can erase the only title-bearing
    # candidate on the page, and front matter then abstains on a paper the
    # heuristics resolved. That was every title regression in the 2026-09-02
    # validation replay (7 of 192; 6 abstained outright), against 73 bylines
    # the same evidence recovered. So the model may add a role, but it may not
    # be the reason a page ends up with no title at all.
    if contents.front_role_predictions is not None and not any(
        "title" in candidate.roles for candidate in candidates
    ):
        heuristic_only = _with_byline_probation(contents, policy, use_model=False)
        if any("title" in candidate.roles for candidate in heuristic_only):
            return heuristic_only
    return candidates


def _with_byline_probation(
    contents: PaperContents, policy: FrontRolePolicy, *, use_model: bool
) -> tuple[FrontMatterCandidate, ...]:
    candidates = _collect_candidates(
        contents, allow_byline_probation=False, policy=policy, use_model=use_model
    )
    if any("byline" in candidate.roles for candidate in candidates):
        return candidates
    return _collect_candidates(
        contents, allow_byline_probation=True, policy=policy, use_model=use_model
    )


def _collect_candidates(
    contents: PaperContents,
    *,
    allow_byline_probation: bool,
    policy: FrontRolePolicy | None = None,
    use_model: bool = True,
) -> tuple[FrontMatterCandidate, ...]:
    policy = policy or FrontRolePolicy()
    predictions = contents.front_role_predictions if use_model else None
    first_page = _first_page(contents)
    paragraph_drafts = _paragraph_drafts(
        contents,
        first_page=first_page,
        allow_byline_probation=allow_byline_probation,
        policy=policy,
    )
    drafts = paragraph_drafts + _heading_drafts(
        contents,
        paragraph_drafts=paragraph_drafts,
        first_page=first_page,
        allow_byline_probation=allow_byline_probation,
        policy=policy,
    )
    overloaded_abstract_sections = _overloaded_abstract_section_ids(drafts)
    # Region order is authoritative only when it covers the whole candidate
    # sequence.  A partial match must not bucket matched rows ahead of unmatched
    # source rows; in that case preserve parser/source order for every draft.
    if drafts and all(draft.region_order is not None for draft in drafts):
        drafts.sort(key=lambda draft: (*draft.region_order, draft.source_order))  # type: ignore[misc]
    else:
        drafts.sort(key=_source_sort_key)
    candidates: list[FrontMatterCandidate] = []
    for reading_order, draft in enumerate(drafts):
        normalized = _normalize_text(draft.raw_text)
        scores = (
            predictions.get(*draft.region_order)
            if predictions is not None and draft.region_order is not None
            else None
        )
        roles, model_roles = _candidate_roles(
            draft,
            normalized,
            detected_title=contents.detected_title,
            allow_abstract_title=draft.section_id in overloaded_abstract_sections,
            scores=scores,
            policy=policy,
        )
        candidates.append(
            FrontMatterCandidate(
                candidate_id=f"front-matter-candidate-{reading_order + 1}",
                source_kind=draft.source_kind,
                reading_order=reading_order,
                page=draft.page,
                bbox=draft.bbox,
                region_label=draft.region_label,
                font_size=draft.font_size,
                font_bold=draft.font_bold,
                section_id=draft.section_id,
                text_ids=draft.text_ids,
                paragraph_id=draft.paragraph_id,
                raw_text=draft.raw_text,
                normalized_text=normalized,
                roles=roles,
                model_roles=model_roles,
                model_scores=(
                    tuple(sorted(scores.probs.items(), key=lambda item: -item[1])[:3])
                    if scores is not None
                    else ()
                ),
            )
        )
    # Numbering alone cannot distinguish a body heading from a proceedings
    # title. Require actual descendant hierarchy and no independently owned
    # byline/abstract before demoting a classifier-UNKNOWN heading. Root (0)
    # is a sentinel, not evidence of an article/body ancestor.
    sections = {section.section_id: section for section in contents.sections}
    title_indices = [index for index, row in enumerate(candidates) if "title" in row.roles]
    for position, index in enumerate(title_indices):
        row = candidates[index]
        section = sections.get(row.section_id)
        parent = sections.get(section.parent_section_id) if section is not None else None
        next_index = (
            title_indices[position + 1] if position + 1 < len(title_indices) else len(candidates)
        )
        if (
            row.source_kind == "heading"
            and section is not None
            and section.section_type == CanonicalSection.UNKNOWN
            and section.parent_section_id not in (None, 0)
            and parent is not None
            and section.level > parent.level
            and _NUMBERED_BODY_HEADING_RE.match(row.raw_text)
            and row.region_label != "doc_title"
            and row.raw_text.strip() != (contents.detected_title or "").strip()
            and not any(
                "byline" in following.roles or _is_abstract_content(following)
                for following in candidates[index + 1 : next_index]
            )
        ):
            candidates[index] = replace(
                row, roles=(row.roles - {"title"}) | frozenset({BODY_HEADING_ROLE})
            )
    from bibr.extract.title_source import SOURCE_QUALIFIED_TITLE_ROLE, native_title_evidence

    qualified = {row.candidate_id for row in native_title_evidence(contents, tuple(candidates))}
    return tuple(
        replace(
            row,
            roles=(row.roles - {"affiliation"}) | {SOURCE_QUALIFIED_TITLE_ROLE},
        )
        if row.candidate_id in qualified
        else row
        for row in candidates
    )


def _make_block(
    index: int,
    candidates: list[FrontMatterCandidate],
) -> FrontMatterBlock:
    pages = tuple(
        sorted({candidate.page for candidate in candidates if candidate.page is not None})
    )
    boxes = [candidate.bbox for candidate in candidates if candidate.bbox is not None]
    return FrontMatterBlock(
        block_id=f"front-matter-block-{index}",
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        title_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates if "title" in candidate.roles
        ),
        pages=pages,
        bbox=_bbox_union(boxes),
        normalized_text="\n".join(candidate.normalized_text for candidate in candidates),
    )


def _is_toc_listing(candidates: tuple[FrontMatterCandidate, ...]) -> bool:
    """Return whether candidates are a masthead-led title/author listing."""

    title_indices = [
        index for index, candidate in enumerate(candidates) if "title" in candidate.roles
    ]
    if not title_indices:
        return False
    first_title = title_indices[0]
    has_leading_masthead = any(
        _looks_like_masthead(candidate.raw_text, (candidate.region_label or "").casefold())
        for candidate in candidates[: first_title + 1]
    )
    has_local_article_anatomy = False
    for position, index in enumerate(title_indices):
        next_index = (
            title_indices[position + 1] if position + 1 < len(title_indices) else len(candidates)
        )
        local_roles = frozenset(
            role for candidate in candidates[index:next_index] for role in candidate.roles
        )
        if local_roles & {"abstract", "doi"} or {
            "byline",
            "affiliation",
        }.issubset(local_roles):
            has_local_article_anatomy = True
            break
    return has_leading_masthead and not has_local_article_anatomy


def _record_title_indices(
    candidates: tuple[FrontMatterCandidate, ...],
    *,
    allow_byline_only: bool,
) -> frozenset[int]:
    """Return title seeds that own nearby record anatomy.

    The lookahead stops at the next title-shaped row.  This is the critical
    distinction between a second record and a subtitle/parallel title: a
    subtitle does not borrow the byline or abstract that belongs to the title
    immediately after it.
    """

    title_indices = [
        index for index, candidate in enumerate(candidates) if "title" in candidate.roles
    ]
    developed: set[int] = set()
    for position, index in enumerate(title_indices):
        next_index = (
            title_indices[position + 1] if position + 1 < len(title_indices) else len(candidates)
        )
        window = candidates[index:next_index]
        # A probation row was admitted purely because it is byline-shaped and
        # sits on page 1 — it carries no evidence that a *record* starts here.
        # Letting its roles count as anatomy promotes any body heading above it
        # into a record root, which splits the block and makes selection
        # fail closed (VAL_METADATA_MULTI_ITEM) or pick the wrong record.
        roles = frozenset(
            role
            for candidate in window
            if BYLINE_PROBATION_ROLE not in candidate.roles
            for role in candidate.roles
        )
        strong_anatomy = bool(
            roles & {"abstract", "doi"}
            or {"byline", "affiliation"}.issubset(roles)
            or (allow_byline_only and "byline" in roles)
        )
        if strong_anatomy and not candidates[index].roles & {
            CLASSIFIED_BYLINE_TITLE_ROLE,
            BYLINE_PROBATION_ROLE,
            MODEL_NON_TITLE_SEED_ROLE,
            BODY_HEADING_ROLE,
        }:
            developed.add(index)
    return frozenset(developed)


def group_front_matter_blocks(
    candidates: tuple[FrontMatterCandidate, ...],
) -> tuple[FrontMatterBlock, ...]:
    """Split only between independently developed article records."""

    if not candidates:
        return ()
    record_titles = _record_title_indices(
        candidates,
        allow_byline_only=not _is_toc_listing(candidates),
    )
    grouped: list[list[FrontMatterCandidate]] = []
    current: list[FrontMatterCandidate] = []
    current_has_record = False
    for index, candidate in enumerate(candidates):
        begins_record = index in record_titles
        if begins_record and current and current_has_record:
            grouped.append(current)
            current = []
            current_has_record = False
        current.append(candidate)
        current_has_record = current_has_record or begins_record
    if current:
        grouped.append(current)
    blocks = tuple(_make_block(index, rows) for index, rows in enumerate(grouped, start=1))
    return _coalesce_repeated_records(blocks, candidates)


def _record_identity_evidence(
    rows: tuple[FrontMatterCandidate, ...],
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Extract conservative printed identifiers, separate from abstract prose."""
    titles = frozenset(
        row.normalized_text
        for row in rows
        if "title" in row.roles
        and row.roles.isdisjoint({"byline", "abstract", "affiliation", "doi", BODY_HEADING_ROLE})
    )
    bylines = frozenset(
        " ".join(_WORD_RE.findall(row.normalized_text))
        for row in rows
        if "byline" in row.roles
        and row.roles.isdisjoint({"title", "abstract", "affiliation", BYLINE_PROBATION_ROLE})
        and row.raw_text.strip()
    )
    title_sections = {row.section_id for row in rows if "title" in row.roles}
    dois = frozenset(
        doi.casefold()
        for row in rows
        if "doi" in row.roles
        and row.roles.isdisjoint({"title", "abstract", "affiliation", "byline"})
        and row.section_id in title_sections
        and (row.region_label or "").casefold() not in {"reference", "reference_content"}
        and len(row.raw_text) <= 240
        for match in _DOI_RE.finditer(row.raw_text)
        if (doi := normalize_doi(match.group(0))) is not None
    )
    return titles, bylines, dois


def _coalesce_repeated_records(
    blocks: tuple[FrontMatterBlock, ...],
    candidates: tuple[FrontMatterCandidate, ...],
) -> tuple[FrontMatterBlock, ...]:
    """Join adjacent presentations only with corroborated shared identity.

    Language or proximity alone never establishes that two records are one.
    Distinct DOI evidence vetoes a merge, even for identical titles/authors.
    """
    by_id = {row.candidate_id: row for row in candidates}
    merged: list[FrontMatterBlock] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        prior = merged[-1]
        left = _record_identity_evidence(_block_candidates(prior, by_id))
        right = _record_identity_evidence(_block_candidates(block, by_id))
        same_byline = bool(left[1] and left[1] == right[1])
        conflict = len(left[2] | right[2]) > 1
        same_doi = bool(left[2] and left[2] == right[2])
        same_title = bool(left[0] & right[0])
        # Do not join far-apart articles by the same authors in a collection.
        adjacent_pages = (
            not prior.pages or not block.pages or 0 <= min(block.pages) - max(prior.pages) <= 1
        )
        if not (same_byline and not conflict and adjacent_pages and (same_doi or same_title)):
            merged.append(block)
            continue
        reason = "shared_doi_and_byline" if same_doi else "repeated_title_and_byline"
        rows = [*_block_candidates(prior, by_id), *_block_candidates(block, by_id)]
        combined = _make_block(len(merged), rows)
        merged[-1] = replace(
            combined,
            block_id=prior.block_id,
            source_block_ids=(prior.source_block_ids or (prior.block_id,)) + (block.block_id,),
            merge_reasons=tuple(dict.fromkeys((*prior.merge_reasons, reason))),
        )
    return tuple(merged)


def _block_candidates(
    block: FrontMatterBlock,
    by_id: dict[str, FrontMatterCandidate],
) -> tuple[FrontMatterCandidate, ...]:
    return tuple(by_id[candidate_id] for candidate_id in block.candidate_ids)


def _is_trusted_context_title(
    candidate: FrontMatterCandidate,
    *,
    detected_title: str | None,
) -> bool:
    if "title" not in candidate.roles:
        return False
    label = (candidate.region_label or "").casefold()
    if label == "doc_title":
        return True
    detected = _normalize_text(detected_title or "")
    if not detected:
        return False
    actual = candidate.normalized_text
    return detected == actual or (
        actual.startswith(detected) and len(actual) <= max(len(detected) * 3, len(detected) + 120)
    )


def _context_geometry_is_compatible(*candidates: FrontMatterCandidate) -> bool:
    known_pages = {candidate.page for candidate in candidates if candidate.page is not None}
    if len(known_pages) > 1:
        return False
    boxes = [candidate.bbox for candidate in candidates if candidate.bbox is not None]
    if len(boxes) < 2:
        return True
    # A title can be wider than the rows below it, so require one common
    # horizontal interval rather than similar widths or exact coordinates.
    return max(box[0] for box in boxes) < min(box[2] for box in boxes)


_PROMOTION_BOUNDARY_ROLES = frozenset({"abstract", "doi", "heading", "title"})


def _separator_run(
    selected_candidates: tuple[FrontMatterCandidate, ...],
    start: int,
) -> tuple[FrontMatterCandidate, ...]:
    """Collect consecutive separator-shaped rows starting at ``start``.

    A wrapped author list can span several OCR rows; a composite row whose
    trailing chunks are the affiliation closes the run and anchors it itself.
    """

    run: list[FrontMatterCandidate] = []
    for row in selected_candidates[start:]:
        if (
            "byline" in row.roles
            or not row.roles.isdisjoint(_PROMOTION_BOUNDARY_ROLES)
            or _NAME_LIST_SEPARATOR_RE.search(row.raw_text) is None
        ):
            break
        if "affiliation" in row.roles:
            if _leading_byline_before_affiliation(row.raw_text, source_kind=row.source_kind):
                run.append(row)
            break
        run.append(row)
    return tuple(run)


def _run_affiliation_anchor(
    selected_candidates: tuple[FrontMatterCandidate, ...],
    start: int,
    run: tuple[FrontMatterCandidate, ...],
) -> tuple[bool, FrontMatterCandidate | None]:
    if "affiliation" in run[-1].roles:
        return True, None
    next_index = start + len(run)
    if next_index >= len(selected_candidates):
        return False, None
    anchor = selected_candidates[next_index]
    if "affiliation" in anchor.roles and anchor.roles.isdisjoint(_PROMOTION_BOUNDARY_ROLES):
        return True, anchor
    return False, None


def _run_has_name_list_shape(run: tuple[FrontMatterCandidate, ...]) -> bool:
    return all(
        _leading_byline_before_affiliation(row.raw_text, source_kind=row.source_kind)
        if "affiliation" in row.roles
        else _looks_like_separator_name_list(row.raw_text, row.normalized_text)
        for row in run
    )


def _promote_selected_contextual_bylines(
    candidates: tuple[FrontMatterCandidate, ...],
    *,
    selected: FrontMatterBlock | None,
    detected_title: str | None,
) -> tuple[FrontMatterCandidate, ...]:
    """Promote selected-record separator shapes without affecting ownership."""

    if selected is None:
        return candidates
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected_candidates = _block_candidates(selected, by_id)
    promoted: dict[str, FrontMatterCandidate] = {}
    index = 1
    while index < len(selected_candidates):
        run = _separator_run(selected_candidates, index)
        if not run:
            index += 1
            continue
        title = selected_candidates[index - 1]
        if not _is_trusted_context_title(title, detected_title=detected_title):
            index += len(run)
            continue
        anchored, anchor = _run_affiliation_anchor(selected_candidates, index, run)
        context_rows = (title, *run) + ((anchor,) if anchor is not None else ())
        if (
            anchored
            and _run_has_name_list_shape(run)
            and _context_geometry_is_compatible(*context_rows)
        ):
            for row in run:
                promoted[row.candidate_id] = replace(
                    row,
                    roles=row.roles | frozenset({"byline"}),
                )
        index += len(run)

    if not promoted:
        return candidates
    return tuple(promoted.get(candidate.candidate_id, candidate) for candidate in candidates)


def _matching_doi_blocks(
    blocks: tuple[FrontMatterBlock, ...],
    by_id: dict[str, FrontMatterCandidate],
    expected_doi: str | None,
    expected_doi_sha256: str | None,
) -> list[FrontMatterBlock]:
    normalized_expected = normalize_doi(expected_doi)
    expected_hash = expected_doi_sha256.casefold() if expected_doi_sha256 else None
    if normalized_expected is None and expected_hash is None:
        return []
    matches = []
    for block in blocks:
        visible = {
            normalized
            for candidate in _block_candidates(block, by_id)
            for match in _DOI_RE.finditer(candidate.raw_text)
            if (normalized := normalize_doi(match.group(0))) is not None
        }
        doi_matches = bool(
            normalized_expected is not None
            and normalized_expected.casefold() in {value.casefold() for value in visible}
        )
        hash_matches = bool(
            expected_hash is not None
            and any(
                hashlib.sha256(value.casefold().encode()).hexdigest() == expected_hash
                for value in visible
            )
        )
        if doi_matches or hash_matches:
            matches.append(block)
    return matches


def _matching_title_blocks(
    blocks: tuple[FrontMatterBlock, ...],
    by_id: dict[str, FrontMatterCandidate],
    expected_title: str,
) -> list[FrontMatterBlock]:
    expected = _normalize_text(expected_title)
    if not expected:
        return []
    matches = []
    for block in blocks:
        title_candidates = [
            by_id[candidate_id]
            for candidate_id in block.title_candidate_ids
            if candidate_id in by_id
        ]
        scores = []
        for candidate in title_candidates:
            actual = candidate.normalized_text
            if expected in actual or actual in expected:
                scores.append(1.0)
            else:
                scores.append(difflib.SequenceMatcher(None, expected, actual).ratio())
        if scores and max(scores) >= 0.9:
            matches.append(block)
    return matches


def _bbox_intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1]
    )


def _matching_hint_blocks(
    blocks: tuple[FrontMatterBlock, ...],
    by_id: dict[str, FrontMatterCandidate],
    hint: dict[str, object],
) -> list[FrontMatterBlock]:
    occurrence = hint.get("occurrence")
    if isinstance(occurrence, int) and not isinstance(occurrence, bool):
        index = occurrence - 1 if occurrence > 0 else occurrence
        return [blocks[index]] if 0 <= index < len(blocks) else []

    page = hint.get("page")
    page_value = page if isinstance(page, int) and not isinstance(page, bool) else None
    x = hint.get("x")
    y = hint.get("y")
    point = (
        (float(x), float(y))
        if isinstance(x, (int, float))
        and not isinstance(x, bool)
        and isinstance(y, (int, float))
        and not isinstance(y, bool)
        else None
    )
    hinted_bbox = _bbox_tuple(hint.get("bbox"))
    matches = []
    for block in blocks:
        candidates = _block_candidates(block, by_id)
        if page_value is not None and not any(
            candidate.page == page_value for candidate in candidates
        ):
            continue
        spatial = [
            candidate
            for candidate in candidates
            if candidate.bbox is not None and (page_value is None or candidate.page == page_value)
        ]
        if point is not None and not any(
            candidate.bbox[0] <= point[0] <= candidate.bbox[2]
            and candidate.bbox[1] <= point[1] <= candidate.bbox[3]
            for candidate in spatial
            if candidate.bbox is not None
        ):
            continue
        if hinted_bbox is not None and not any(
            _bbox_intersects(candidate.bbox, hinted_bbox)
            for candidate in spatial
            if candidate.bbox is not None
        ):
            continue
        matches.append(block)
    return matches


_HINT_KEYS = frozenset({"occurrence", "page", "x", "y", "bbox"})


def _finite_coordinate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    coordinate = float(value)
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1000.0:
        return None
    return coordinate


def _validated_target_hint(hint: object) -> dict[str, object] | None:
    """Validate the closed, fail-closed target-hint schema."""

    if not isinstance(hint, dict) or not hint or not set(hint).issubset(_HINT_KEYS):
        return None
    if "occurrence" in hint:
        occurrence = hint["occurrence"]
        if (
            len(hint) != 1
            or isinstance(occurrence, bool)
            or not isinstance(occurrence, int)
            or occurrence <= 0
        ):
            return None
        return {"occurrence": occurrence}

    normalized: dict[str, object] = {}
    if "page" in hint:
        page = hint["page"]
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            return None
        normalized["page"] = page

    has_x = "x" in hint
    has_y = "y" in hint
    if has_x != has_y or ("bbox" in hint and has_x):
        return None
    if has_x:
        x = _finite_coordinate(hint["x"])
        y = _finite_coordinate(hint["y"])
        if x is None or y is None:
            return None
        normalized.update(x=x, y=y)

    if "bbox" in hint:
        value = hint["bbox"]
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        coordinates = tuple(_finite_coordinate(part) for part in value)
        if any(part is None for part in coordinates):
            return None
        x1, y1, x2, y2 = coordinates
        if x1 is None or y1 is None or x2 is None or y2 is None or x1 >= x2 or y1 >= y2:
            return None
        normalized["bbox"] = (x1, y1, x2, y2)

    return normalized or None


def _has_expected_selectors(expected_identity: ExpectedIdentity | None) -> bool:
    return bool(
        expected_identity is not None
        and (
            expected_identity.expected_doi is not None
            or expected_identity.expected_doi_sha256 is not None
            or expected_identity.expected_title is not None
            or expected_identity.target_block_hint is not None
        )
    )


def _select_with_expected_identity(
    blocks: tuple[FrontMatterBlock, ...],
    by_id: dict[str, FrontMatterCandidate],
    expected_identity: ExpectedIdentity | None,
) -> tuple[FrontMatterBlock | None, str | None, tuple[str, ...]]:
    if expected_identity is None:
        return None, None, ()
    selectors: list[tuple[str, str, list[FrontMatterBlock]]] = []
    if expected_identity.expected_doi is not None:
        selectors.append(
            (
                "expected_doi",
                "expected_doi",
                _matching_doi_blocks(
                    blocks,
                    by_id,
                    expected_identity.expected_doi,
                    None,
                ),
            )
        )
    if expected_identity.expected_doi_sha256 is not None:
        selectors.append(
            (
                "expected_doi_sha256",
                "expected_doi",
                _matching_doi_blocks(
                    blocks,
                    by_id,
                    None,
                    expected_identity.expected_doi_sha256,
                ),
            )
        )
    if expected_identity.expected_title is not None:
        selectors.append(
            (
                "expected_title",
                "expected_title",
                _matching_title_blocks(blocks, by_id, expected_identity.expected_title),
            )
        )
    if expected_identity.target_block_hint is not None:
        validated_hint = _validated_target_hint(expected_identity.target_block_hint)
        if validated_hint is None:
            selectors.append(("target_block_hint", "target_block_hint", []))
        else:
            selectors.append(
                (
                    "target_block_hint",
                    "target_block_hint",
                    _matching_hint_blocks(blocks, by_id, validated_hint),
                )
            )

    flags: list[str] = []
    unique_matches: list[tuple[str, FrontMatterBlock]] = []
    selector_failed = False
    for flag_name, method, matches in selectors:
        if (
            flag_name == "target_block_hint"
            and expected_identity.target_block_hint is not None
            and _validated_target_hint(expected_identity.target_block_hint) is None
        ):
            flags.append("target_block_hint_invalid")
            selector_failed = True
            continue
        if len(matches) == 1:
            unique_matches.append((method, matches[0]))
        else:
            flags.append(f"{flag_name}_{'ambiguous' if matches else 'not_found'}")
            selector_failed = True

    unique_block_ids = {block.block_id for _, block in unique_matches}
    if len(unique_block_ids) > 1:
        flags.append("expected_identity_conflict")
        return None, None, tuple(flags)
    if selector_failed:
        return None, None, tuple(flags)
    if unique_matches:
        method, block = unique_matches[0]
        return block, method, tuple(flags)
    return None, None, tuple(flags)


_UNSAFE_TITLE_COMPANION_ROLES = frozenset({"abstract", "affiliation", "byline", "doi"})


def _is_abstract_content(candidate: FrontMatterCandidate) -> bool:
    """Abstract-role rows that carry content, not a bare printed heading."""

    return bool(
        "abstract" in candidate.roles
        and not _ABSTRACT_HEADING_RE.fullmatch(candidate.raw_text.strip())
        and candidate.normalized_text not in _ORDINARY_HEADING_TEXT
    )


def _select_dominant_coherent_block(
    blocks: tuple[FrontMatterBlock, ...],
    by_id: dict[str, FrontMatterCandidate],
) -> FrontMatterBlock | None:
    """Select the single coherent record when every competitor is anatomy-free.

    A block is coherent when it owns one safe non-composite title, byline
    evidence, and abstract content or DOI evidence. Dominance requires exactly
    one coherent block while every competing block lacks byline, abstract
    content, and DOI alike — two independently developed records always stay
    fail-closed, and raw score is never consulted.

    A competitor's veto weighs *heuristic* evidence only. The front-role
    classifier is additive evidence, so a page-1 row it alone calls a byline is
    not an independently developed record; letting it veto turned six papers in
    the 2026-09-02 validation replay from ``unique_block`` into
    ``multiple_plausible_blocks``, losing a title the heuristics had. The
    dominant block may still qualify on model evidence — that is the byline
    rescue the move exists for.
    """

    dominant: FrontMatterBlock | None = None
    for block in blocks:
        candidates = _block_candidates(block, by_id)
        has_byline = any("byline" in candidate.roles for candidate in candidates)
        has_abstract_content = any(_is_abstract_content(candidate) for candidate in candidates)
        has_doi = any("doi" in candidate.roles for candidate in candidates)
        has_safe_title = any(
            "title" in candidate.roles and candidate.roles.isdisjoint(_UNSAFE_TITLE_COMPANION_ROLES)
            for candidate in candidates
        )
        if has_safe_title and has_byline and (has_abstract_content or has_doi):
            if dominant is not None:
                return None
            dominant = block
        elif (
            _heuristic_role(candidates, "byline")
            or has_abstract_content
            or _heuristic_role(candidates, "doi")
        ):
            return None
    return dominant


def _heuristic_role(candidates: tuple[FrontMatterCandidate, ...], role: str) -> bool:
    """True when *role* is held on evidence the classifier did not supply."""
    return any(
        role in candidate.roles and role not in candidate.model_roles for candidate in candidates
    )


def _multi_item_issue(
    blocks: tuple[FrontMatterBlock, ...],
    expected_identity: ExpectedIdentity | None,
) -> ValidationIssue:
    evidence = [block.block_id for block in blocks]
    if expected_identity is not None:
        evidence.insert(0, expected_identity.queue_record_id)
    return ValidationIssue(
        code="VAL_METADATA_MULTI_ITEM",
        severity=IssueSeverity.ERROR,
        message="The required metadata record could not be selected uniquely and safely",
        origin_stage="extract",
        evidence_ids=tuple(evidence),
        blocking=True,
    )


def resolve_front_matter(
    contents: PaperContents,
    *,
    expected_identity: ExpectedIdentity | None = None,
    target_required: bool | None = None,
    settings: GlobalSettings | None = None,
) -> tuple[FrontMatterResolution, tuple[ValidationIssue, ...]]:
    """Build candidate blocks, select one deterministically, or abstain."""

    if target_required is None:
        target_required = bool(
            expected_identity is not None
            and (expected_identity.doi_required or _has_expected_selectors(expected_identity))
        )
    candidates = collect_front_matter_candidates(
        contents, policy=FrontRolePolicy.from_settings(settings)
    )
    blocks = group_front_matter_blocks(candidates)
    toc_listing = _is_toc_listing(candidates)
    selectable_blocks = () if toc_listing else blocks
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected: FrontMatterBlock | None = None
    method = "no_candidates" if not blocks else "abstained"
    reason_flags: list[str] = []
    if any(block.merge_reasons for block in blocks):
        reason_flags.append("shared_identity_presentations_merged")
    if any(candidate.model_roles for candidate in candidates):
        reason_flags.append("front_role_model")

    expected_selection, expected_method, expected_flags = _select_with_expected_identity(
        selectable_blocks,
        by_id,
        expected_identity,
    )
    reason_flags.extend(expected_flags)
    if expected_selection is not None:
        selected = expected_selection
        method = expected_method or "expected_identity"
    elif toc_listing:
        reason_flags.append("toc_listing")
    elif len(blocks) == 1 and not _has_expected_selectors(expected_identity):
        selected = blocks[0]
        method = "unique_block"
    elif len(blocks) > 1:
        # A supplied but unresolved expected DOI/title/hint stays fail-closed;
        # heuristic dominance applies to untargeted inputs only.
        dominant = (
            _select_dominant_coherent_block(blocks, by_id)
            if not _has_expected_selectors(expected_identity)
            else None
        )
        if dominant is not None:
            selected = dominant
            method = "coherent_dominance"
        else:
            reason_flags.append("multiple_plausible_blocks")
    elif not blocks:
        reason_flags.append("no_candidates")

    candidates = _promote_selected_contextual_bylines(
        candidates,
        selected=selected,
        detected_title=contents.detected_title,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected_candidates = _block_candidates(selected, by_id) if selected is not None else ()
    resolution = FrontMatterResolution(
        candidates=candidates,
        blocks=blocks,
        selected_block_id=selected.block_id if selected is not None else None,
        selection_method=method,
        reason_flags=tuple(dict.fromkeys(reason_flags)),
        allowed_text_ids=frozenset(
            text_id for candidate in selected_candidates for text_id in candidate.text_ids
        ),
        allowed_section_ids=frozenset(
            candidate.section_id
            for candidate in selected_candidates
            if candidate.section_id is not None
        ),
    )
    issues = (
        (_multi_item_issue(blocks, expected_identity),)
        if target_required and selected is None
        else ()
    )
    return resolution, issues


__all__ = [
    "FrontMatterBlock",
    "FrontMatterCandidate",
    "FrontMatterResolution",
    "FrontRolePolicy",
    "collect_front_matter_candidates",
    "group_front_matter_blocks",
    "is_exact_front_matter_furniture",
    "resolve_front_matter",
]
