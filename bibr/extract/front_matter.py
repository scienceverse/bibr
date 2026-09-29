"""Spatial front-matter candidates, record blocks, and deterministic selection.

Sentence and section text are the authoritative source.  ``RegionSummary`` is
used only to recover layout metadata and the parser's original region order;
its deliberately short content preview is never promoted into candidate text.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
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
from bibr.utils.text import DOI_CANDIDATE_RE, normalize_doi
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.extract.front_role import FrontRolePredictions, RoleScores
    from bibr.paper_contents import PaperContents, PaperSentence, RegionSummary
    from bibr.pipeline.identity import ExpectedIdentity

logger = logging.getLogger(__name__)

_DOI_RE = DOI_CANDIDATE_RE
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
    # The region (page, index) of the title a reprinted title block's byline
    # sits under; the draft is placed right after that title's draft.
    follows_region: tuple[int, int] | None = None


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


# Section types that close the title-to-abstract gap on the first page.
_FRONT_GAP_END_TYPES = _BODY_SECTION_TYPES | {CanonicalSection.ABSTRACT, CanonicalSection.KEYWORDS}


def _front_gap_section_ids(contents: PaperContents, first_page: int | None) -> frozenset[int]:
    """Sections printed between the first-page title and the abstract or body.

    The header classifier has no byline class. When the parser promotes the
    cells of a grid or column byline to headings ("Enyuan Tian", "Sung Whan
    Yoon shyoon8@kaist.ac.kr"), the classifier types some of them as endnotes
    or acknowledgments, and those sections never reach the front matter. Only a
    paper with no byline at all gets the page-1 probation rescue, so a byline
    that loses some cells this way keeps the rest and drops those authors.
    Between the title and the abstract nothing is body text, so a row there
    with name evidence is admitted whatever its type: a byline-shaped heading
    that is no field label, a name over an e-mail address, or a merged
    paragraph whose first region reads as a byline. The gap opens only at the
    parser's detected title: a body heading the classifier calls TITLE
    ("1.1.1. An Efficient ...") would otherwise open one over related-work
    citations. Without a closing abstract or body section the gap has no end,
    and nothing is admitted.
    """

    detected_title = (contents.detected_title or "").strip()
    if first_page is None or not detected_title:
        return frozenset()
    gap: list[int] = []
    title_seen = False
    for section in contents.sections:
        if section.level <= 0:
            continue
        if not title_seen:
            title_seen = section.header.strip() == detected_title and any(
                item.page_no == first_page for item in section.provenance
            )
            continue
        if section.section_type in _FRONT_GAP_END_TYPES:
            return frozenset(gap)
        gap.append(section.section_id)
    return frozenset()


_EMAIL_ADDRESS_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)


def _looks_like_name_email_cell(text: str) -> bool:
    """One byline cell: an author's name followed by that author's e-mail address.

    Column bylines print each author over their address ("Jun Seo" /
    "tjwns0630@kaist.ac.kr"); the address makes the legacy name-ratio test
    fail, although the cell is the plainest byline evidence there is.
    """

    emails = _EMAIL_ADDRESS_RE.findall(text)
    if len(emails) != 1:
        return False
    name, _, tail = text.partition(emails[0])
    if tail.strip(" \t\r\n.,;)>"):
        return False
    name = name.strip(" \t\r\n.,;:(<")
    return bool(name) and _looks_like_legacy_byline(
        name, _normalize_text(name), source_kind="heading"
    )


_URL_TEXT_RE = re.compile(r"https?://|\bwww\.", re.IGNORECASE)


def _front_gap_byline_evidence(text: str, summary: RegionSummary | None, *, regions: int) -> bool:
    """Byline evidence accepted only for a mistyped paragraph inside the front gap.

    A name over its author's e-mail address is a byline cell. Beyond that, the
    parser can merge a byline row with the rows that follow it (a
    correspondence block) into one paragraph spanning several layout regions,
    and the merged text no longer reads as a byline. The paragraph's first
    region still does; its preview is read here as evidence only, never as
    candidate text. A one-region paragraph gets no such pass (its preview is
    its own text), nor does one carrying a DOI or URL, like a how-to-cite box.
    An editorial line ("Handling Editor: Jane Smith" over her address, "Edited
    by Jane Smith and John Doe" leading a merged block) gets neither pass.
    """

    if _looks_like_name_email_cell(text):
        return not _is_front_gap_editorial_line(text)
    if regions < 2 or _DOI_RE.search(text) or _URL_TEXT_RE.search(text):
        return False
    preview = (summary.content or "").strip() if summary is not None else ""
    return (
        bool(preview)
        and not _is_front_gap_editorial_line(preview)
        and _looks_like_byline(preview, _normalize_text(preview), source_kind="paragraph")
    )


# Words that make a row a front-matter label ("Author Note", "Corresponding
# Author:", "Credit Author Statement", "Supporting Information"), never a name.
_FRONT_GAP_LABEL_WORDS = frozenset(
    {
        "abstract",
        "acknowledgement",
        "acknowledgements",
        "acknowledgment",
        "acknowledgments",
        "address",
        "affiliation",
        "affiliations",
        "author",
        "authors",
        "contact",
        "contribution",
        "contributions",
        "correspondence",
        "corresponding",
        "credit",
        "e-mail",
        "email",
        "information",
        "keywords",
        "note",
        "notes",
        "statement",
        "supporting",
    }
)


# Words of the editorial and article-metadata lines that share the gap with the
# byline ("Handling Editor: Jane Smith", "Edited by Jane Smith", "Received 12
# March 2020", "Data Availability", "Competing Interests", "Specialty Section").
# A name printed there is an editor's or a reviewer's, not an author's.
_FRONT_GAP_EDITORIAL_WORDS = frozenset(
    {
        "accepted",
        "availability",
        "count",
        "declaration",
        "declarations",
        "edited",
        "editor",
        "editors",
        "ethics",
        "funding",
        "history",
        "interests",
        "practices",
        "published",
        "received",
        "reviewed",
        "reviewer",
        "reviewers",
        "revised",
        "section",
    }
)


def _is_front_gap_editorial_line(text: str) -> bool:
    # An author's address can carry one of the words in its domain
    # (jane.smith@history.ox.ac.uk), so only the words outside it count.
    words = _WORD_RE.findall(_EMAIL_ADDRESS_RE.sub(" ", text))
    return any(word.casefold() in _FRONT_GAP_EDITORIAL_WORDS for word in words)


def _front_gap_heading_is_byline(
    header: str,
    summary: RegionSummary | None,
    predictions: FrontRolePredictions | None,
    policy: FrontRolePolicy,
) -> bool:
    """A mistyped heading inside the front gap, on a page that has a byline.

    Label headings there ("Author Note", "Corresponding Author:") read as two
    capitalised words, like a name, and a topic line set in capitals ("THE
    DECISION TO REFINANCE") passes the capital-ratio test; neither is admitted.
    Nor is an editorial or metadata line ("Edited by Jane Smith", "Received 12
    March 2020"), which the capital-ratio test and the classifier's byline vote
    both pass, even over an e-mail address; nor any other labelled line (one
    with a colon) that is not a name over its e-mail address.
    """

    words = [word.casefold() for word in _WORD_RE.findall(header)]
    if _is_front_gap_editorial_line(header):
        return False
    if _looks_like_name_email_cell(header):
        return True
    if ":" in header or any(
        word in _FRONT_GAP_LABEL_WORDS or word in _NON_NAME_CHUNK_WORDS for word in words
    ):
        return False
    return _looks_like_byline(
        header, _normalize_text(header), source_kind="heading"
    ) or _model_role(_scores_for(predictions, summary), "byline", policy.min_confidence)


# How far above the title's top edge a page-head byline may end, in the 0..1000
# layout space of region boxes.
_PAGE_HEAD_BYLINE_MAX_GAP = 60.0
_PAGE_HEAD_BYLINE_MAX_WORDS = 5
# Words of journal names, publisher badges, article-type and section labels,
# which are as capitalised as a name ("Educational Review", "BioMed Central",
# "Original Manuscript", "Case Report", "Scientific Reports", "Revista de
# Psicología", "Special Issue"), and of an editor's credit ("John Smith,
# Editor"). None of them is part of a person's name. Compared accent-folded.
_PAGE_HEAD_NON_NAME_WORDS = frozenset(
    {
        "access",
        "anales",
        "annales",
        "annals",
        "article",
        "articles",
        "articulo",
        "association",
        "boletin",
        "bulletin",
        "cahiers",
        "central",
        "chapter",
        "communication",
        "communications",
        "cuadernos",
        "editor",
        "editorial",
        "editors",
        "estudios",
        "etudes",
        "issue",
        "journal",
        "journals",
        "letters",
        "magazine",
        "manuscript",
        "paper",
        "papers",
        "proceedings",
        "publishing",
        "quarterly",
        "report",
        "reports",
        "research",
        "review",
        "reviews",
        "revista",
        "revue",
        "rivista",
        "science",
        "sciences",
        "section",
        "series",
        "societe",
        "society",
        "studies",
        "transactions",
    }
)


def _looks_like_person_name_row(text: str) -> bool:
    """A row that is nothing but one person's name ("Eva-Maria Biermann-Ratjen")."""

    # "&" joins two names or a journal's two subjects ("Memory & Cognition").
    if any(char.isdigit() or char in "@:/()[]&" for char in text):
        return False
    words = _WORD_RE.findall(text)
    if not 2 <= len(words) <= _PAGE_HEAD_BYLINE_MAX_WORDS:
        return False
    if any(
        word.casefold() in _NON_NAME_CHUNK_WORDS or _fold_accents(word) in _PAGE_HEAD_NON_NAME_WORDS
        for word in words
    ):
        return False
    if not all(word[:1].isupper() or word.casefold() in _NAME_PARTICLES for word in words):
        return False
    # Capitals mark a journal, publisher or section banner ("BMC Public Health",
    # "CASE REPORT"), not a name row.
    if any(len(word) > 1 and word.isupper() for word in words):
        return False
    normalized = _normalize_text(text)
    return not (
        is_exact_front_matter_furniture(text)
        or _MASTHEAD_RE.match(text.strip())
        or normalized in _ORDINARY_HEADING_TEXT
    )


# What separates the names of a byline row: commas, semicolons, "&", and the
# word that joins the last two names ("and", French "et", Dutch "en", German
# "und", Danish/Norwegian "og", Swedish "och"), in lower case.
_NAME_ROW_SEPARATOR_RE = re.compile(r"\s*(?:[,;&]|\b(?:and|et|en|und|og|och)\b)\s*")
# Words that name an institution in the languages of European bylines
# ("Universiteit", "Universidad", "Institut", "Departamento", "Faculteit",
# "Laboratoire", "Hôpital"), beyond the English markers of an affiliation.
_INSTITUTION_WORD_RE = re.compile(
    r"\b(?:univers|uniwers|institu|istitu|instytu|departam|departem|départem|facult|fakult|"
    r"laborat|hospit|hôpit)",
    re.IGNORECASE,
)
# A token of an e-mail address or a link ("jan.peeters[at]example.org",
# "https://…", "www.…", "example.org"): its lower-case words are names and
# hosts, not the paper's prose.
_ADDRESS_TOKEN_RE = re.compile(
    r"\S*(?:@|\[at\]|\(at\)|://|www\.|[^\W\d_]\.[^\W\d_])\S*", re.IGNORECASE
)


def _name_list_chunks(text: str) -> list[str]:
    return [chunk for chunk in _NAME_ROW_SEPARATOR_RE.split(text) if chunk.strip()]


def _looks_like_person_name_list(text: str) -> bool:
    """A row that is nothing but two or more people's names.

    "Marie Dubois, Pieter Janssens, Anne-Sophie Martin et Luc Van Damme":
    every chunk between the separators reads as one person's name, and no
    chunk names an institution ("Vrije Universiteit Zuid").
    """

    if AFFILIATION_MARKER_RE.search(text) or _INSTITUTION_WORD_RE.search(text):
        return False
    chunks = _name_list_chunks(text)
    return len(chunks) >= 2 and all(_looks_like_person_name_row(chunk) for chunk in chunks)


def _lowercase_vocabulary(contents: PaperContents) -> frozenset[str]:
    """The words the paper prints in lower case, outside addresses and links."""

    texts = [sentence.text or "" for sentence in contents.sentences]
    texts += [summary.content or "" for summary in contents.region_summaries or []]
    return frozenset(
        word.casefold()
        for text in texts
        for word in _WORD_RE.findall(_ADDRESS_TOKEN_RE.sub(" ", text))
        if word.islower()
    )


def _reads_as_common_words(chunk: str, vocabulary: frozenset[str]) -> bool:
    """Whether every word of *chunk* is one the paper prints in lower case.

    A title set in title case ("Urban Parks and Public Life") splits into
    capitalised chunks as a byline does, but a paper that writes of urban
    parks and public life prints those words in lower case too, and people's
    names seldom. Initials and surname particles count neither way.
    """

    words = [
        word.casefold()
        for word in _WORD_RE.findall(chunk)
        if len(word) > 1 and word.casefold() not in _NAME_PARTICLES
    ]
    return bool(words) and all(word in vocabulary for word in words)


def _reprinted_title_block_bylines(
    contents: PaperContents, rows: list[RegionSummary], running_heads: set[str]
) -> list[tuple[RegionSummary, RegionSummary]]:
    """The byline rows of a title block that a cover page reprints, each with its title.

    A repository cover page prints the article's title, its translated titles
    and its byline, and the article's first page prints them again. Repeated
    on both pages as a block of body rows, the rows under the title read as
    float furniture reprinted with each figure, and the parser files every
    copy with the running heads, so neither page keeps a byline. A row of that
    block is a byline when layout labelled it body text, it sits under the
    first page's title, the text of every row between the title and it is a
    running head too, and it is nothing but a list of people's names. The
    block also holds the translated titles, so a name list is not made of the
    title's own words, and none of its names is a run of words the paper
    prints in lower case: a translated title set in title case has a byline's
    shape. A row whose text is a running head can still have been kept on
    this page; the candidate collection leaves such a row to its own draft.
    """

    found: list[tuple[RegionSummary, RegionSummary]] = []
    title: RegionSummary | None = None
    title_words = {word.casefold() for word in _WORD_RE.findall(contents.detected_title or "")}
    vocabulary: frozenset[str] | None = None
    for summary in sorted(rows, key=lambda row: row.index):
        label = (summary.label or "").casefold()
        if label == "doc_title":
            title = summary
            continue
        text = (summary.content or "").strip()
        if title is None or not text or _normalize_text(text) not in running_heads:
            # The block ends at the first row whose text is not a running head.
            title = None
            continue
        if (
            label != "text"
            or summary.bbox is None
            or title.bbox is None
            or summary.bbox[1] < title.bbox[3]
            or not _looks_like_person_name_list(text)
        ):
            continue
        chunks = _name_list_chunks(text)
        words = {word.casefold() for chunk in chunks for word in _WORD_RE.findall(chunk)}
        if words <= title_words | {
            word.casefold() for word in _WORD_RE.findall(title.content or "")
        }:
            continue
        if vocabulary is None:
            vocabulary = _lowercase_vocabulary(contents)
        if any(_reads_as_common_words(chunk, vocabulary) for chunk in chunks):
            continue
        found.append((title, summary))
    return found


def _page_head_draft(
    summary: RegionSummary, text: str, *, follows_region: tuple[int, int] | None = None
) -> _CandidateDraft:
    return _CandidateDraft(
        source_kind="paragraph",
        source_order=-1,
        region_order=(summary.page, summary.index),
        page=summary.page,
        bbox=summary.bbox,
        # Not "header": a structural label reads as a leading masthead, and a
        # masthead-led page without an abstract is taken for a table of
        # contents.
        region_label=None,
        font_size=summary.font_size,
        font_bold=summary.font_bold,
        section_id=None,
        text_ids=(),
        paragraph_id=None,
        raw_text=text,
        section_type=CanonicalSection.UNKNOWN,
        byline_probation=True,
        follows_region=follows_region,
    )


def _place_after_their_titles(
    drafts: list[_CandidateDraft], followers: list[_CandidateDraft]
) -> list[_CandidateDraft]:
    """Put each reprinted title block's byline right after its title's draft.

    Source order has no slot for a row the parser filed with the running
    heads, and region order places it only when it covers every draft. Either
    way the byline belongs under its title, ahead of the heading the page sets
    next ("Édition électronique" on a repository cover, which would otherwise
    read as the title's subtitle). A byline whose title has no draft, or whose
    row the parser kept after all and so already has a draft, is left out.
    """

    placed = list(drafts)
    for follower in sorted(followers, key=lambda draft: draft.region_order or (0, 0)):
        if any(draft.region_order == follower.region_order for draft in placed):
            continue
        slots = [
            position
            for position, draft in enumerate(placed)
            if follower.follows_region in (draft.region_order, draft.follows_region)
        ]
        if slots:
            placed.insert(slots[-1] + 1, follower)
    return placed


def _page_head_byline_drafts(
    contents: PaperContents, first_page: int | None
) -> list[_CandidateDraft]:
    """A byline printed above the title that layout labelled a page header.

    Some journals and edited volumes set the author's name over the title
    ("Hubert Heinen" / "German-Texan Attitudes toward the Civil War"), often
    repeated later as the running head. Layout labels that row a header, the
    parser files it with the running heads, and the sentence stream never sees
    it, so the front matter has no author text at all. The row is admitted as
    plain text, on probation: it widens what the author call reads, and its
    shape alone never makes it a title or a record root (a front-role
    classifier vote still counts, as for any row). The row is tested and
    emitted as this page prints it; the parser's running heads only confirm it
    was filed with them, since a later page can set the same head in other
    casing ("HUBERT HEINEN", or "Psychological Science" under a first-page
    "PSYCHOLOGICAL SCIENCE"). The byline of a title block that a cover page
    reprints is filed with the running heads too, and is admitted the same
    way (see ``_reprinted_title_block_bylines``).
    """

    if first_page is None:
        return []
    rows = [
        summary
        for summary in contents.region_summaries or []
        if summary.page == first_page and summary.bbox is not None
    ]
    title_tops = [
        summary.bbox[1] for summary in rows if (summary.label or "").casefold() == "doc_title"
    ]
    if not title_tops:
        return []
    title_top = min(title_tops)
    running_heads = {_normalize_text(header) for header in contents.detected_headers or []}
    title_words = {word.casefold() for word in _WORD_RE.findall(contents.detected_title or "")}
    drafts: list[_CandidateDraft] = []
    for summary in rows:
        if (summary.label or "").casefold() != "header":
            continue
        text = (summary.content or "").strip()
        if (
            not text
            or _normalize_text(text) not in running_heads
            or not _looks_like_person_name_row(text)
        ):
            continue
        # A page head made of the title's own words is the short title set as
        # the running head ("Remythologising Satan." over "Remythologising
        # Satan: A New Version of The Fall of Lucifer."), not a name.
        if {word.casefold() for word in _WORD_RE.findall(text)} <= title_words:
            continue
        gap = title_top - summary.bbox[3]
        if not 0.0 <= gap <= _PAGE_HEAD_BYLINE_MAX_GAP:
            continue
        drafts.append(_page_head_draft(summary, text))
    drafts.extend(
        _page_head_draft(
            byline, (byline.content or "").strip(), follows_region=(title.page, title.index)
        )
        for title, byline in _reprinted_title_block_bylines(contents, rows, running_heads)
    )
    return drafts


def _paragraph_drafts(
    contents: PaperContents,
    *,
    first_page: int | None,
    allow_byline_probation: bool,
    policy: FrontRolePolicy | None = None,
    front_gap: frozenset[int] = frozenset(),
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
            if not (allow_byline_probation or sentence.section_id in front_gap):
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
        if (section_id, paragraph_id) in byline_probation:
            gap_evidence = section_id in front_gap and _front_gap_byline_evidence(
                raw_text, summary, regions=len(matched_summaries)
            )
            if allow_byline_probation:
                admitted = (
                    _looks_like_byline(raw_text, _normalize_text(raw_text), source_kind="paragraph")
                    or _model_role(
                        _scores_for(predictions, summary), "byline", policy.min_confidence
                    )
                    or gap_evidence
                )
            else:
                # Outside the rescue only the front gap let this paragraph in,
                # and on a page with a byline the looser shape tests admit
                # publisher lines and affiliations.
                admitted = gap_evidence
            if not admitted:
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
    front_gap: frozenset[int] = frozenset(),
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
            in_gap = section.section_id in front_gap
            if not (allow_byline_probation or in_gap):
                continue
            header = section.header.strip()
            if page is None or page != first_page:
                continue
            if allow_byline_probation:
                admitted = (
                    _looks_like_byline(header, _normalize_text(header), source_kind="heading")
                    or _model_role(
                        _scores_for(predictions, summary), "byline", policy.min_confidence
                    )
                    or (
                        in_gap
                        and _looks_like_name_email_cell(header)
                        and not _is_front_gap_editorial_line(header)
                    )
                )
            else:
                admitted = _front_gap_heading_is_byline(header, summary, predictions, policy)
            if not admitted:
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
    if not candidates:
        opening = _opening_section_as_front_matter(contents)
        if opening is not None:
            return collect_front_matter_candidates(opening, policy=policy)
    if not any("title" in candidate.roles for candidate in candidates):
        return _with_manuscript_title_seed(candidates, _first_page(contents))
    return candidates


# The longest first row of an opening section still read as its byline.
_OPENING_BYLINE_MAX_CHARS = 300


def _opening_section_as_front_matter(contents: PaperContents) -> PaperContents | None:
    """Read a body-typed opening section as front matter, when nothing else is.

    A short paper can print its title as the header of its only section, with
    the byline as that section's first row and the whole text under it. When
    the section classifier types that section as body text (an introduction),
    no row reaches front matter and the paper abstains with no candidate at
    all. Only then, and only when the first page opens with that section, its
    header is title-shaped (two to thirty words, not numbered, not an ordinary
    body heading) and its first row, on the first page and at most
    :data:`_OPENING_BYLINE_MAX_CHARS` long, prints a capitalized name and an
    initial, an affiliation superscript or an affiliation anywhere in the row,
    is the section read as untyped front matter. The row is not parsed as a
    byline: a short sentence naming a person with an initial also passes.
    Anything else leaves the page without candidates.
    """

    first_page = _first_page(contents)
    opening = next(
        (
            section
            for section in contents.sections
            if section.level > 0 and not section.header_is_synthetic
        ),
        None,
    )
    if (
        first_page is None
        or opening is None
        or opening.section_type in _FRONT_MATTER_SECTION_TYPES
        or not any(item.page_no == first_page for item in opening.provenance)
    ):
        return None
    header = opening.header.strip()
    if (
        not 2 <= len(_WORD_RE.findall(header)) <= 30
        or _NUMBERED_HEADING_RE.match(header)
        or _normalize_text(header) in _ORDINARY_HEADING_TEXT
    ):
        return None
    first_row = next(
        (sentence for sentence in contents.sentences if sentence.text.strip()),
        None,
    )
    if first_row is None or first_row.section_id != opening.section_id:
        return None
    text = first_row.text.strip()
    if (
        first_page not in _sentence_pages(first_row)
        or len(text) > _OPENING_BYLINE_MAX_CHARS
        or not _person_surnames(text)
        or not (_has_person_name_evidence(text) or _looks_like_affiliation(text))
    ):
        return None
    logger.info("Reading the opening section %s as front matter", opening.section_id)
    return replace(
        contents,
        sections=[
            replace(section, section_type=CanonicalSection.UNKNOWN)
            if section is opening
            else section
            for section in contents.sections
        ],
    )


# A manuscript title row: a few words on at most three printed rows, not
# closed like a sentence of prose or a field label.
_MANUSCRIPT_TITLE_MAX_ROWS = 3
_MANUSCRIPT_TITLE_WORDS = (4, 30)


def _with_manuscript_title_seed(
    candidates: tuple[FrontMatterCandidate, ...], first_page: int | None
) -> tuple[FrontMatterCandidate, ...]:
    """Seed the title of a manuscript set entirely in body font, as a last resort.

    An anonymised submission prints its title as a plain text row at the top of
    the first page, in the body font and size, with the abstract under its own
    heading further on. No layout label, capitals or heading marks it, and the
    front-role classifier can score the row as abstract text, so neither pass
    above finds a title and the record has none. When no candidate has the
    title role, the first row of the first page becomes the title seed if it is
    title-shaped, carries no role but the classifier's abstract guess, and an
    ``Abstract`` heading follows it: the printed heading places the abstract
    elsewhere. Anything else leaves the candidates as they are.
    """

    first = next((candidate for candidate in candidates if candidate.page == first_page), None)
    if first is None or first_page is None:
        return candidates
    text = first.raw_text.strip()
    words = _WORD_RE.findall(text)
    low, high = _MANUSCRIPT_TITLE_WORDS
    if (
        first.source_kind != "paragraph"
        or (first.region_label or "").casefold() != "text"
        or not first.roles <= (first.model_roles & {"abstract"})
        or not low <= len(words) <= high
        or text.count("\n") >= _MANUSCRIPT_TITLE_MAX_ROWS
        or text.endswith((".", ":", ";", ","))
        or _DOI_RE.search(text)
        or _looks_like_affiliation(text)
        or _looks_like_masthead(text, "text")
        or is_exact_front_matter_furniture(text)
        or first.normalized_text in _ORDINARY_HEADING_TEXT
    ):
        return candidates
    if not any(
        candidate.reading_order > first.reading_order
        and candidate.raw_text.strip().rstrip(":.").casefold() == "abstract"
        for candidate in candidates
    ):
        return candidates
    logger.info("Seeding the manuscript title from the first-page row %s", first.candidate_id)
    seeded = replace(first, roles=frozenset({"title"}), model_roles=frozenset())
    return tuple(seeded if candidate is first else candidate for candidate in candidates)


def _with_byline_probation(
    contents: PaperContents, policy: FrontRolePolicy, *, use_model: bool
) -> tuple[FrontMatterCandidate, ...]:
    candidates = _collect_candidates(
        contents, allow_byline_probation=False, policy=policy, use_model=use_model
    )
    allow_byline_probation = not any("byline" in candidate.roles for candidate in candidates)
    if allow_byline_probation:
        candidates = _collect_candidates(
            contents, allow_byline_probation=True, policy=policy, use_model=use_model
        )
    first_page = _first_page(contents)
    # A name printed above the title joins only a first page that prints no
    # byline ahead of its abstract, whichever pass found the page's bylines:
    # bylines found only past the abstract (a keyword line, a body heading that
    # names a theorist, a closing biography) leave it the only byline there.
    if _first_page_prints_byline(candidates, first_page) or not _page_head_byline_drafts(
        contents, first_page
    ):
        return candidates
    return _collect_candidates(
        contents,
        allow_byline_probation=allow_byline_probation,
        page_heads=True,
        policy=policy,
        use_model=use_model,
    )


def _first_page_prints_byline(
    candidates: tuple[FrontMatterCandidate, ...], first_page: int | None
) -> bool:
    """Whether a byline row precedes the abstract on the first page."""

    for candidate in candidates:
        if candidate.page != first_page:
            continue
        if candidate.roles & {"abstract", "keywords"}:
            return False
        if "byline" in candidate.roles:
            return True
    return False


def _collect_candidates(
    contents: PaperContents,
    *,
    allow_byline_probation: bool,
    page_heads: bool = False,
    policy: FrontRolePolicy | None = None,
    use_model: bool = True,
) -> tuple[FrontMatterCandidate, ...]:
    policy = policy or FrontRolePolicy()
    predictions = contents.front_role_predictions if use_model else None
    first_page = _first_page(contents)
    front_gap = _front_gap_section_ids(contents, first_page)
    paragraph_drafts = _paragraph_drafts(
        contents,
        first_page=first_page,
        allow_byline_probation=allow_byline_probation,
        policy=policy,
        front_gap=front_gap,
    )
    drafts = paragraph_drafts + _heading_drafts(
        contents,
        paragraph_drafts=paragraph_drafts,
        first_page=first_page,
        allow_byline_probation=allow_byline_probation,
        policy=policy,
        front_gap=front_gap,
    )
    if page_heads:
        # A page head counts as author text only on a first page that prints
        # no byline ahead of its abstract (see ``_with_byline_probation``).
        drafts += _page_head_byline_drafts(contents, first_page)
    overloaded_abstract_sections = _overloaded_abstract_section_ids(drafts)
    followers = [draft for draft in drafts if draft.follows_region is not None]
    drafts = [draft for draft in drafts if draft.follows_region is None]
    # Region order is authoritative only when it covers the whole candidate
    # sequence.  A partial match must not bucket matched rows ahead of unmatched
    # source rows; in that case preserve parser/source order for every draft.
    if drafts and all(draft.region_order is not None for draft in drafts):
        drafts.sort(key=lambda draft: (*draft.region_order, draft.source_order))  # type: ignore[misc]
    else:
        drafts.sort(key=_source_sort_key)
    if followers:
        drafts = _place_after_their_titles(drafts, followers)
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
    return tuple(candidates)


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


def _is_parallel_title_above(title: FrontMatterCandidate, following: FrontMatterCandidate) -> bool:
    """A name-less uppercase title printed right above its translation.

    Both rows are article titles on one page, in two languages, and the upper
    one holds a byline role with no names after its title: the punctuation
    rule for proceedings rows ("TITLE Name, Name") gave it that role.
    """

    return bool(
        title.page is not None
        and following.page == title.page
        and "byline" in title.roles
        and not title.roles & {"abstract", "affiliation", "doi"}
        and _composite_name_tail(title.raw_text) is None
        and _is_identity_title(title, None)
        and _is_identity_title(following, None)
        and _languages_differ(_title_language(title.raw_text), _title_language(following.raw_text))
    )


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
        # An uppercase title set directly above its parallel title in another
        # language, on the same page, owns no row but itself. When it prints no
        # names after the title, its byline role came from its capitals and
        # punctuation alone, and it develops no record of its own.
        if (
            len(window) == 1
            and next_index < len(candidates)
            and _is_parallel_title_above(candidates[index], candidates[next_index])
        ):
            continue
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
    return tuple(_make_block(index, rows) for index, rows in enumerate(grouped, start=1))


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


# Record agreement: the fallback for pages coherent dominance abstains on.
#
# Dominance vetoes on any competing anatomy, so a page abstains whenever the
# paper's own record is printed twice (a publisher cover page, a repository
# landing page, a citation box, a translated title and abstract) or a
# furniture row (an email list, a date line, "a r t i c l e i n f o", a
# sidebar heading) roots a second block that then owns the abstract. This
# fallback reads what each block prints about its paper — title, author
# surnames, DOIs — and selects only when every record on the page agrees.
# Two records that disagree still abstain: compiled abstract books and
# proceedings pages print several complete records with different titles.

_AGREE = "agree"
_CONFLICT = "conflict"
_UNVERIFIED = "unverified"

# Funder-registry DOIs name a funder, not a paper.
_FUNDER_DOI_PREFIX = "10.13039/"
# Typeset hyphens (U+2010 and kin) stop the DOI pattern early, which cut two
# different DOIs down to one shared prefix; read them as ASCII hyphens.
_DOI_HYPHENS = str.maketrans(
    {"\u00ad": None, **dict.fromkeys("\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d", "-")}
)
# Anchored at the start of a token and possessive, so an unbroken CJK row is
# scanned once instead of once per character.
_URL_OR_EMAIL_RE = re.compile(r"(?<!\S)[^\s@]*+@\S*|https?://\S*|www\.\S*", re.IGNORECASE)
# "1. Introduction", "2. MATERIALS AND METHODS", "IV. Results": numbered
# body headings are never the article title.
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[IVX]+[.)])\s+\S")
# Rows that open a cover page, a repository landing page or a citation box.
# Such a block repeats the article's identity; it ranks below the article's
# own title page.
_COVER_CUE_RE = re.compile(
    r"^\W*(?:to cite this|cite this article|how to cite|please cite|"
    r"(?:recommended|suggested|scholar commons) citation|citation\s*(?::|$)|"
    r"citation for the published version|this article was downloaded|downloaded from|"
    r"follow this and additional works|published in\s*:|document version|"
    r"link to publication|terms and conditions of use|take-down policy|"
    r"please scroll down for article|para citar|c[oó]mo citar|pour citer|zitierweise|"
    r"цитування|для цитирования)",
    re.IGNORECASE,
)
# Scripts written without spaces between words.
_UNSPACED_SCRIPTS = frozenset({"CJK", "HIRAGANA", "KATAKANA", "THAI"})
# Common function words per language. A title's language is the unique best
# match; it only has to tell a translated title from a different paper's title.
_TITLE_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    language: frozenset(words.split())
    for language, words in {
        "en": "the of and in for on with to from by an among between through using its their "
        "how what why does is are",
        "pt": "de da do das dos e em no na nos nas para com um uma os ao aos pela pelo pelas "
        "pelos sobre entre sua seu não como",
        "es": "de la el los las y en del para con un una por al sobre entre su sus como",
        "fr": "de la le les des du et en un une pour dans sur par au aux entre d l leur leurs",
        "de": "der die das und im von zur zum für mit bei ein eine einer eines des den dem auf "
        "über als zwischen aus nach",
        "it": "di del della delle dei degli e il lo gli per con nel nella nelle tra fra sul "
        "sulla una",
        "nl": "het een van en voor met op bij naar over door tussen uit",
        "pl": "i w z na do dla o oraz od po przez we ze jako nie się",
        "id": "dan yang di dari untuk pada dengan dalam terhadap sebagai ke oleh atau",
    }.items()
}
# Surname comparison folds scripts to Latin so a transliterated byline
# ("Ivanov I.I.") can match the original ("Иванов И.И.").
_TRANSLITERATION = (
    "а=a б=b в=v г=g ґ=g д=d е=e ё=e є=ie ж=zh з=z и=i і=i ї=i й=i к=k л=l м=m н=n о=o "
    "п=p р=r с=s т=t у=u ў=u ф=f х=kh ц=ts ч=ch ш=sh щ=shch ъ= ы=y ь= э=e ю=yu я=ya ђ=dj "
    "ј=j љ=lj њ=nj ћ=c џ=dz α=a β=b γ=g δ=d ε=e ζ=z η=i θ=th ι=i κ=k λ=l μ=m ν=n ξ=x ο=o "
    "π=p ρ=r σ=s ς=s τ=t υ=y φ=f χ=ch ψ=ps ω=o ł=l ø=o đ=d ß=ss æ=ae œ=oe þ=th ı=i"
)
_SURNAME_TRANSLITERATION = str.maketrans({pair[0]: pair[2:] for pair in _TRANSLITERATION.split()})
# Romanizations disagree on й, ю and я (y, i or j): compare those letters as one.
_SURNAME_VARIANTS = str.maketrans("yj", "ii")
_PERSON_CHUNK_SPLIT_RE = re.compile(
    r"\s*(?:[,;·•&]|\band\b|\bund\b|\bet\b|\bи\b)\s*", re.IGNORECASE
)
_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*(\.)?")
_NON_SURNAME_TOKENS = frozenset(
    {"by", "prof", "dr", "phd", "md", "msc", "author", "authors", "corresponding", "mail"}
)
# Longest title (in words) and citation row (in characters) worth comparing;
# longer rows are prose, and bounding them keeps every comparison linear.
_MAX_TITLE_WORDS = 60
_MAX_CITATION_CHARS = 2_000
# A figure an abstract prints ("2016", "20.5%", "0,05"), not a digit glued to a
# word ("Example1", "e2099") or hyphenated to one ("COVID-19").
_FIGURE_RE = re.compile(r"(?<![\w.,])(?<![^\W\d_]-)\d+(?:[.,]\d+)*(?!\w)")
# Whole numbers up to ten number a list ("(1)", "2)") or count what another
# language may spell out; they are no figures.
_MAX_LIST_NUMBER = 10
# A year: two years alone ("2019 and 2020") are shared by many papers.
_YEAR_RE = re.compile(r"(?:1[5-9]|20)\d\d")
# Abstracts in this many languages printing the same two years are one paper's.
_ECHO_LANGUAGES = 3
# "© O. Example, 2024": the holder a copyright line names, up to its comma.
_COPYRIGHT_HOLDER_RE = re.compile(r"©\s*(?:\d{4}\s+)?([^,\r\n]+)")


@dataclass(frozen=True)
class _RecordIdentity:
    """What one block prints about the paper it describes.

    Evidence comes from the block's opening rows only — up to the page after
    the one its first title sits on — so body paragraphs a block swallowed,
    and the references they cite, never speak for its identity.
    """

    block: FrontMatterBlock
    order: int
    titles: tuple[str, ...]
    citations: tuple[str, ...]
    language: str | None
    surnames: frozenset[str]
    dois: frozenset[str]
    byline: bool
    abstract: bool
    anatomy: bool
    doc_title: bool
    # "exact" when a title row is the parser's detected title, "prefix" when
    # one runs on past it, else None.
    detected: str | None
    cover: bool
    # The figures its abstract rows print, with decimal commas read as points.
    figures: frozenset[str] = frozenset()

    @property
    def is_record(self) -> bool:
        """A title plus byline or layout-title evidence: something that can be a paper."""

        return bool(self.titles) and (self.byline or self.doc_title or self.detected is not None)

    @property
    def is_bare(self) -> bool:
        """No record, and nothing a record prints: no byline, abstract, DOI or cover cue.

        Such a block is a heading the grouping split on because the heading's
        own capitals or punctuation read as a byline (the heading of a
        committee's member list, an outline, a body heading), not a second item.
        """

        return not (self.is_record or self.byline or self.anatomy or self.cover)


def _identity_key(text: str) -> str:
    """Letters and digits only, casefolded and accent-free, in any script.

    Dropping every space, hyphen and punctuation mark makes the key immune to
    line wraps, hyphenation ("self-" / "reports"), letter spacing and a lost
    space ("rightsand"), so a title compares equal however it was laid out.
    """

    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in unicodedata.normalize("NFKD", folded) if char.isalnum())


def _detected_match(candidate: FrontMatterCandidate, detected_title: str | None) -> str | None:
    """Whether a title row is the parser's detected title, or runs on past it."""

    if "title" not in candidate.roles or not detected_title:
        return None
    key = _identity_key(detected_title)
    if key and _identity_key(candidate.raw_text) == key:
        return "exact"
    detected = _normalize_text(detected_title)
    actual = candidate.normalized_text
    if actual.startswith(detected) and len(actual) <= max(len(detected) * 3, len(detected) + 120):
        return "prefix"
    return None


def _title_word_count(text: str) -> int:
    count = sum(len(word) > 1 for word in _WORD_RE.findall(text))
    if _dominant_script(text) in _UNSPACED_SCRIPTS:
        count = max(count, sum(char.isalpha() for char in text) // 2)
    return count


def _is_identity_title(candidate: FrontMatterCandidate, detected_title: str | None) -> bool:
    """A title-role row that could be an article title rather than furniture.

    Layout's own title and the parser's detected title always count, even
    when they name a university, open with a number or run to two words.
    """

    roles = candidate.roles
    if "title" not in roles or roles & {
        "abstract",
        "doi",
        MODEL_NON_TITLE_SEED_ROLE,
        CLASSIFIED_BYLINE_TITLE_ROLE,
        BYLINE_PROBATION_ROLE,
    }:
        return False
    text = candidate.raw_text.strip()
    if _URL_OR_EMAIL_RE.search(text):
        return False
    if (candidate.region_label or "").casefold() == "doc_title" or (
        _detected_match(candidate, detected_title) == "exact"
    ):
        return True
    # An affiliation-bearing title is an institution line, unless it is the
    # proceedings composite that also carries the byline.
    if "affiliation" in roles and "byline" not in roles:
        return False
    return not _NUMBERED_HEADING_RE.match(text) and _title_word_count(text) >= 3


def _dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for char in text:
        if char.isalpha():
            script = unicodedata.name(char, "").split(" ", 1)[0]
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    script, count = max(counts.items(), key=lambda item: item[1])
    return script if count * 5 >= sum(counts.values()) * 3 else None


def _title_language(text: str) -> str | None:
    """The title's language where it can be told, else its script, else None.

    Latin script is resolved by function words; Cyrillic by the letters only
    Ukrainian (і ї є ґ) or only Russian (ы э ъ ё) prints.
    """

    script = _dominant_script(text)
    if script == "CYRILLIC":
        letters = set(text.casefold())
        if letters & set("іїєґ"):
            return "uk"
        return "ru" if letters & set("ыэъё") else "cyrillic"
    if script != "LATIN":
        return script.casefold() if script else None
    words = [part.casefold() for word in _WORD_RE.findall(text) for part in re.split(r"['’]", word)]
    scores = sorted(
        (
            (sum(word in vocabulary for word in words), language)
            for language, vocabulary in _TITLE_FUNCTION_WORDS.items()
        ),
        reverse=True,
    )
    (best, language), (runner_up, _) = scores[0], scores[1]
    return language if best > runner_up else None


def _person_surnames(text: str) -> frozenset[str]:
    """Surnames of the name-shaped chunks printed before any affiliation.

    A chunk is one to five capitalized words once initials ("Yu.", "F.",
    "FH") and particles are skipped ("Sample, J. K." yields "Sample"); its
    last word is the surname, transliterated and accent-folded.
    """

    bounded = _URL_OR_EMAIL_RE.sub(" ", text)
    marker = AFFILIATION_MARKER_RE.search(bounded)
    if marker is not None:
        bounded = bounded[: marker.start()]
    surnames: set[str] = set()
    for chunk in _PERSON_CHUNK_SPLIT_RE.split(_BYLINE_MARKER_RE.sub(" ", bounded)):
        words = []
        for match in _NAME_TOKEN_RE.finditer(chunk):
            word = match.group(0).rstrip(".")
            initial = len(word) == 1 or (
                len(word) <= 2 and (match.group(1) is not None or word.isupper())
            )
            if not initial and word.casefold() not in _NAME_PARTICLES:
                words.append(word)
        if not 1 <= len(words) <= 5 or not all(word[:1].isupper() for word in words):
            continue
        folded = unicodedata.normalize("NFKD", words[-1].casefold())
        surname = "".join(
            char for char in folded.translate(_SURNAME_TRANSLITERATION) if char.isalpha()
        )
        if 2 <= len(surname) <= 40 and surname not in _NON_SURNAME_TOKENS:
            surnames.add(surname)
    return frozenset(surnames)


def _composite_name_tail(text: str) -> str | None:
    """The mixed-case name list after an uppercase title in one proceedings row.

    "COMMUNITY GARDENS AND SHARED SPACES Morgan Example, Mei Lin" prints title
    and byline together; an all-uppercase title whose byline role came only
    from its capitalization has no such tail.
    """

    words = list(re.finditer(r"\S+", text))
    index = 0
    while index < len(words) and not any(char.islower() for char in words[index].group(0)):
        index += 1
    if index == 0 or index == len(words):
        return None
    # The tail opens on a capitalized name ("Morgan", "Mei"), not OCR debris.
    first = words[index].group(0)
    if not (first[:1].isupper() and first[1:2].islower()):
        return None
    tail = text[words[index].start() :]
    return tail if _person_surnames(tail) else None


def _identity_byline_text(candidate: FrontMatterCandidate) -> str | None:
    """Printed author text a row contributes to its block's identity."""

    roles = candidate.roles
    if "abstract" in roles:
        return None
    if "title" in roles and CLASSIFIED_BYLINE_TITLE_ROLE not in roles:
        return _composite_name_tail(candidate.raw_text) if "byline" in roles else None
    if "byline" in roles:
        return candidate.raw_text
    if "affiliation" in roles:
        # "Ada K. Example, Ben L. Sample, Example University, Utrecht":
        # an author list typed as an affiliation still names the authors.
        names = _person_surnames(candidate.raw_text)
        return candidate.raw_text if len(names) >= 2 else None
    return None


def _is_citation_row(candidate: FrontMatterCandidate) -> bool:
    """A non-title row that could cite the paper: a year, a DOI or a cover cue."""

    text = candidate.raw_text
    return bool(
        "title" not in candidate.roles
        and len(text) <= _MAX_CITATION_CHARS
        and (
            _REFERENCE_YEAR_RE.search(text)
            or _DOI_RE.search(text)
            or _COVER_CUE_RE.match(text.strip())
        )
    )


def _copyright_named_row(rows: tuple[FrontMatterCandidate, ...]) -> str | None:
    """A bare name row whose surnames the page's copyright line prints.

    "Oleh Example" on a row of its own, with no role, above a title whose
    copyright line reads "© O. Example, 2024": the name is the byline, even
    though a lone given name and surname carry no byline shape of their own.
    """

    # The holder must be a person, printed with an initial: a publisher
    # ("Taylor & Francis Group", "Springer Nature Switzerland AG") is not.
    holders = frozenset(
        surname
        for row in rows
        for match in _COPYRIGHT_HOLDER_RE.finditer(row.raw_text)
        if _has_person_name_evidence(match.group(1))
        for surname in _person_surnames(match.group(1))
    )
    if not holders:
        return None
    for row in rows:
        text = row.raw_text.strip()
        words = text.split()
        if (
            row.roles
            or "©" in text
            or any(char.isdigit() for char in text)
            or not 2 <= len(words) <= 4
            or not all(word[:1].isupper() for word in words)
        ):
            continue
        surnames = _person_surnames(text)
        if surnames and surnames <= holders:
            return text
    return None


def _record_identity(
    block: FrontMatterBlock,
    order: int,
    by_id: dict[str, FrontMatterCandidate],
    *,
    detected_title: str | None,
) -> _RecordIdentity:
    rows = _block_candidates(block, by_id)
    anchor = next(
        (
            row.page
            for row in rows
            if row.page is not None and _is_identity_title(row, detected_title)
        ),
        next((row.page for row in rows if row.page is not None), None),
    )
    window = rows
    if anchor is not None:
        end = next(
            (
                index
                for index, row in enumerate(rows)
                if row.page is not None and row.page > anchor + 1
            ),
            len(rows),
        )
        window = rows[:end]
    titles = tuple(row for row in window if _is_identity_title(row, detected_title))
    byline_texts = [text for row in window if (text := _identity_byline_text(row)) is not None]
    if not byline_texts and anchor is not None:
        name_row = _copyright_named_row(tuple(row for row in window if row.page == anchor))
        if name_row is not None:
            byline_texts = [name_row]
    dois = frozenset(
        doi.casefold()
        for row in window
        for match in _DOI_RE.finditer(row.raw_text.translate(_DOI_HYPHENS))
        if (doi := normalize_doi(match.group(0))) is not None
        and not doi.startswith(_FUNDER_DOI_PREFIX)
    )
    abstract = any(_is_abstract_content(row) for row in window)
    matches = {_detected_match(row, detected_title) for row in window}
    return _RecordIdentity(
        block=block,
        order=order,
        titles=tuple(row.raw_text for row in titles),
        citations=tuple(row.raw_text for row in window if _is_citation_row(row)),
        language=_title_language(titles[0].raw_text) if titles else None,
        surnames=frozenset(name for text in byline_texts for name in _person_surnames(text)),
        dois=dois,
        byline=bool(byline_texts),
        abstract=abstract,
        anatomy=abstract or bool(dois) or any("doi" in row.roles for row in window),
        doc_title=any((row.region_label or "").casefold() == "doc_title" for row in titles),
        detected="exact" if "exact" in matches else "prefix" if "prefix" in matches else None,
        cover=any(_COVER_CUE_RE.match(row.raw_text.strip()) for row in window),
        figures=frozenset(
            figure
            for row in window
            if _is_abstract_content(row) and "doi" not in row.roles
            for match in _FIGURE_RE.finditer(row.raw_text)
            if not _is_list_number(figure := match.group(0).replace(",", "."))
        ),
    )


def _title_words(title: str) -> list[str]:
    return [key for word in re.findall(r"\w+", title) if (key := _identity_key(word))]


def _one_letter_apart(left: str, right: str) -> bool:
    """One misread, dropped or added letter in a word of five letters or more."""

    if min(len(left), len(right)) < 5 or left.isdigit() or right.isdigit():
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    shorter, longer = sorted((left, right), key=len)
    return len(longer) - len(shorter) == 1 and any(
        longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer))
    )


def _same_title(left: str, right: str) -> bool:
    """One printed title, however it was laid out or lightly misread.

    The identity keys must be equal, or the words must be the same but for at
    most one word in eight misread by a single letter. Printed numbers never
    differ ("Study 1" is not "Study 2"), and a title plus words ("... with
    autism") is a different title.
    """

    left_key, right_key = _identity_key(left), _identity_key(right)
    if len(left_key) < 12 or len(right_key) < 12:
        return False
    if left_key == right_key:
        return True
    left_words, right_words = _title_words(left), _title_words(right)
    if len(left_words) != len(right_words) or not 4 <= len(left_words) <= _MAX_TITLE_WORDS:
        return False
    misread = [(a, b) for a, b in zip(left_words, right_words, strict=True) if a != b]
    return len(misread) <= max(1, len(left_words) // 8) and all(
        _one_letter_apart(a, b) for a, b in misread
    )


def _main_title(title: str) -> str | None:
    """The main title before a subtitle break, when it is long enough to stand alone.

    A cover page prints "Title: Subtitle" on one line where the title page
    sets the subtitle on its own row. Subtitles alone ("A Randomized
    Controlled Trial") never stand for the title.
    """

    parts = re.split(r"\s*(?::|\s[-–—]\s)\s*", title, maxsplit=1)
    return parts[0] if len(parts) == 2 and len(_WORD_RE.findall(parts[0])) >= 5 else None


def _cites_title(title: str, text: str) -> bool:
    """A citation line that prints *title* whole: all its words in order, closed
    by punctuation rather than running on into a longer title."""

    words = [re.escape(word) for word in re.findall(r"\w+", _fold_accents(title))]
    if not 6 <= len(words) <= _MAX_TITLE_WORDS:
        return False
    pattern = r"(?<!\w)" + r"\W*".join(words) + r"(?=\s*(?:[^\w\s]|$))"
    return re.search(pattern, _fold_accents(text)) is not None


def _fold_accents(text: str) -> str:
    folded = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text).casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


def _titles_agree(left: _RecordIdentity, right: _RecordIdentity) -> bool:
    """Same printed title, allowing a subtitle split, light OCR noise, or a
    citation box that prints the other's title."""

    for left_title in left.titles:
        left_main = _main_title(left_title)
        for right_title in right.titles:
            right_main = _main_title(right_title)
            if (
                _same_title(left_title, right_title)
                or (left_main is not None and _same_title(left_main, right_title))
                or (right_main is not None and _same_title(left_title, right_main))
            ):
                return True
    return any(
        _cites_title(title, citation)
        for titles, citations in ((left.titles, right.citations), (right.titles, left.citations))
        for title in titles
        for citation in citations
    )


def _same_surname(left: str, right: str) -> bool:
    """One surname across romanizations: exact below six letters (Zhang is not
    Zhong), a close spelling from six letters on."""

    left, right = left.translate(_SURNAME_VARIANTS), right.translate(_SURNAME_VARIANTS)
    if left == right:
        return True
    return min(len(left), len(right)) >= 6 and (
        difflib.SequenceMatcher(None, left, right).ratio() >= 0.8
    )


def _surnames_agree(left: frozenset[str], right: frozenset[str]) -> bool:
    """Every surname of the shorter byline, or at least two, found in the other."""

    smaller, larger = sorted((left, right), key=len)
    matched = sum(any(_same_surname(a, b) for b in larger) for a in smaller)
    return bool(matched) and (matched == len(smaller) or matched >= 2)


def _languages_differ(left: str | None, right: str | None) -> bool:
    """Two titles known to be in different languages, so one may translate the other."""

    if left is None or right is None or left == right:
        return False
    # A Cyrillic title without Ukrainian- or Russian-only letters may be either.
    return not ({left, right} <= {"cyrillic", "uk", "ru"} and "cyrillic" in {left, right})


def _is_list_number(figure: str) -> bool:
    return figure.isdigit() and int(figure) <= _MAX_LIST_NUMBER


def _distinctive_figures(figures: frozenset[str], *, echoed: bool) -> bool:
    """Whether the figures two abstracts share tie them to one paper.

    Three figures do, and so do two with a count or a measure among them. Two
    years alone ("2019 and 2020") are common to the papers of one period: they
    count only when *echoed*, printed by abstracts in three languages.
    """

    if len(figures) >= 3:
        return True
    if len(figures) < 2:
        return False
    return echoed or not all(_YEAR_RE.fullmatch(figure) for figure in figures)


def _echoed_figures(identities: list[_RecordIdentity]) -> frozenset[frozenset[str]]:
    """The figure sets that abstracts in three or more languages print alike."""

    echoed = set()
    for figures in {identity.figures for identity in identities if identity.figures}:
        languages: list[str] = []
        for identity in identities:
            if (
                identity.figures == figures
                and identity.language is not None
                and all(_languages_differ(identity.language, seen) for seen in languages)
            ):
                languages.append(identity.language)
        if len(languages) >= _ECHO_LANGUAGES:
            echoed.add(figures)
    return frozenset(echoed)


def _translates_abstract(
    presentation: _RecordIdentity,
    record: _RecordIdentity,
    *,
    echoed_figures: frozenset[frozenset[str]],
) -> bool:
    """A byline-less title and abstract whose abstract prints the record's figures.

    A multilingual journal sets the translated title and abstract under a
    layout title of their own, with no byline or DOI; the figures of a
    translated abstract (years, counts, percentages) are the original's. They
    must be the same figures and distinctive (see :func:`_distinctive_figures`).
    """

    return (
        not presentation.byline
        and not presentation.dois
        and record.byline
        and presentation.figures == record.figures
        and _distinctive_figures(
            presentation.figures, echoed=presentation.figures in echoed_figures
        )
    )


def _record_relation(
    left: _RecordIdentity,
    right: _RecordIdentity,
    *,
    echoed_figures: frozenset[frozenset[str]] = frozenset(),
) -> tuple[str, str]:
    """Whether two records print the same paper, a different one, or cannot tell.

    Different DOIs are different papers. A shared DOI or title joins them,
    but a DOI never joins two different titles in one language: a proceedings
    volume or a supplement lends its DOI to many papers. Different titles in
    one language are different papers. A title in another language or script
    is a translation only when the bylines name the same authors, or when it
    has no byline and its abstract prints exactly the other's figures, and
    distinctive ones; *echoed_figures* are the page's figure sets that
    abstracts in three languages print.
    """

    titles_agree = _titles_agree(left, right)
    translated = _languages_differ(left.language, right.language)
    if left.dois and right.dois:
        if not left.dois & right.dois:
            return _CONFLICT, "doi"
        return (_AGREE, "doi") if titles_agree or translated else (_CONFLICT, "doi_title")
    if titles_agree:
        return _AGREE, "title"
    if not translated:
        return _CONFLICT, "title"
    if not left.surnames or not right.surnames:
        if any(
            _translates_abstract(presentation, record, echoed_figures=echoed_figures)
            for presentation, record in ((left, right), (right, left))
        ):
            return _AGREE, "abstract_figures"
        return _UNVERIFIED, "translation"
    if _surnames_agree(left.surnames, right.surnames):
        return _AGREE, "authors"
    if any(_same_surname(a, b) for a in left.surnames for b in right.surnames):
        return _UNVERIFIED, "authors"
    return _CONFLICT, "authors"


def _primary_rank(identity: _RecordIdentity) -> tuple[bool, bool, bool, bool, bool, bool, int]:
    """The article's own title page first: the parser's detected title (exact
    before a prefix), not a cover or citation box, a byline, abstract content,
    a layout title, then the first printed."""

    return (
        identity.detected != "exact",
        identity.detected is None,
        identity.cover,
        not identity.byline,
        not identity.abstract,
        not identity.doc_title,
        identity.order,
    )


def _without_cover_pages(
    block: FrontMatterBlock,
    by_id: dict[str, FrontMatterCandidate],
    *,
    detected_title: str | None,
) -> tuple[FrontMatterBlock, tuple[int, ...]]:
    """The block without the cover pages printed ahead of its title page.

    A download cover ("This article was downloaded by", "PLEASE SCROLL DOWN
    FOR ARTICLE") can head the block of the scanned article behind it; its
    download stamp, disclaimer and the publisher's registered office would
    read as the article's date, abstract and affiliation. Pages before the
    record's first title that print a cover cue are dropped, and returned; a
    page with a title of the record's own never is. A cue set as a heading
    ("PLEASE SCROLL DOWN FOR ARTICLE") is no title of the record's.
    """

    rows = _block_candidates(block, by_id)
    title_page = next(
        (
            row.page
            for row in rows
            if row.page is not None
            and _is_identity_title(row, detected_title)
            and not _COVER_CUE_RE.match(row.raw_text.strip())
        ),
        None,
    )
    if title_page is None:
        return block, ()
    cover_pages = tuple(
        sorted(
            {
                row.page
                for row in rows
                if row.page is not None
                and row.page < title_page
                and _COVER_CUE_RE.match(row.raw_text.strip())
            }
        )
    )
    if not cover_pages:
        return block, ()
    kept = [row for row in rows if row.page not in cover_pages]
    return replace(_make_block(0, kept), block_id=block.block_id), cover_pages


def _select_agreeing_record(
    blocks: tuple[FrontMatterBlock, ...],
    by_id: dict[str, FrontMatterCandidate],
    *,
    detected_title: str | None,
) -> tuple[FrontMatterBlock | None, tuple[str, ...]]:
    """Select the paper's own record when every record on the page agrees.

    Records (a title with byline or layout-title evidence) are grouped by
    agreement: the same DOI, the same title, or — for a title in another
    language — the same author surnames. Any conflicting pair, or records
    that cannot be linked, abstains. A block that is not a record attaches to
    the group only if it prints no other DOI, a byline of its own names the
    group's authors, title or DOI, and a title and abstract of its own match
    a record or are in another language. The group must add up to a complete
    record, the selected block must hold byline evidence, and the parser's
    detected title must belong to the group or to a translation of it. The one
    exception is a single record whose every other block is bare (see
    :attr:`_RecordIdentity.is_bare`): it is selected as a unique block would be,
    without a cover page ahead of its title page (see :func:`_without_cover_pages`).
    """

    identities = [
        _record_identity(block, order, by_id, detected_title=detected_title)
        for order, block in enumerate(blocks)
    ]
    records = [identity for identity in identities if identity.is_record]
    if not records:
        return None, ("no_record_identity",)

    parent = {identity.block.block_id: identity.block.block_id for identity in records}

    def root(block_id: str) -> str:
        while parent[block_id] != block_id:
            block_id = parent[block_id]
        return block_id

    reasons: dict[str, str] = {}
    echoed_figures = _echoed_figures(identities)
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            relation, reason = _record_relation(left, right, echoed_figures=echoed_figures)
            if relation == _CONFLICT:
                return None, (
                    f"conflicting_records:{left.block.block_id}:{right.block.block_id}:{reason}",
                )
            if relation == _AGREE:
                parent[root(right.block.block_id)] = root(left.block.block_id)
                reasons.setdefault(right.block.block_id, reason)
                reasons.setdefault(left.block.block_id, reason)
    if len({root(identity.block.block_id) for identity in records}) > 1:
        return None, ("unlinked_records",)

    group_dois = frozenset(doi for identity in records for doi in identity.dois)
    group_surnames = frozenset(name for identity in records for name in identity.surnames)
    attached = [identity for identity in identities if not identity.is_record]
    for identity in attached:
        block_id = identity.block.block_id
        if identity.dois and group_dois and not identity.dois & group_dois:
            return None, (f"doi_conflict:{block_id}",)
        # A byline with an abstract or DOI of its own is a record the title
        # filter missed (a title naming a university, opening with a number):
        # it must name the group's authors, title or DOI.
        if (
            identity.byline
            and (identity.abstract or identity.dois)
            and not (
                any(_same_surname(a, b) for a in identity.surnames for b in group_surnames)
                or any(_titles_agree(identity, record) for record in records)
                or identity.dois & group_dois
            )
        ):
            return None, (f"unlinked_block:{block_id}",)
        # A title and abstract without a byline is the record translated, or
        # another paper whose byline went unseen (an abstract book): only the
        # same title or another language tells the two apart.
        if (
            identity.titles
            and identity.abstract
            and not any(_titles_agree(identity, record) for record in records)
            and not all(_languages_differ(identity.language, record.language) for record in records)
        ):
            return None, (f"unmatched_presentation:{block_id}",)
    if not any(identity.byline for identity in identities) or not any(
        identity.anatomy for identity in identities
    ):
        # One record beside bare headings only is the page a unique block
        # would be had the grouping not split on those headings, and it is
        # selected as that block would be, with or without a byline or DOI.
        if len(records) == 1 and all(identity.is_bare for identity in attached):
            selected, cover_pages = _without_cover_pages(
                records[0].block, by_id, detected_title=detected_title
            )
            return selected, (
                f"lone_record:{records[0].block.block_id}",
                *(f"attached_block:{identity.block.block_id}" for identity in attached),
                *(f"cover_page_dropped:{page}" for page in cover_pages),
            )
        return None, ("incomplete_record",)

    # The metadata call reads only the selected block, so it must print a
    # byline. A translation with a byline never stands in for an original
    # title page without one.
    best = min(records, key=_primary_rank)
    eligible = [
        identity
        for identity in records
        if identity.byline and not _languages_differ(identity.language, best.language)
    ]
    if not eligible:
        return None, (f"record_without_byline:{best.block.block_id}",)
    primary = min(eligible, key=_primary_rank)

    # The parser's detected title must be the group's, or a translated
    # presentation of it; otherwise a null metadata title would later be
    # filled from another paper's printed title.
    if detected_title:
        matched = [identity for identity in identities if identity.detected == "exact"] or [
            identity for identity in identities if identity.detected == "prefix"
        ]
        if matched and not any(identity.is_record for identity in matched):
            detected_language = _title_language(detected_title)
            for identity in matched:
                if not _languages_differ(detected_language, primary.language) or (
                    identity.byline and not _surnames_agree(identity.surnames, group_surnames)
                ):
                    return None, (f"detected_title_outside_record:{identity.block.block_id}",)

    flags = [
        f"agreeing_record:{identity.block.block_id}:{reasons[identity.block.block_id]}"
        for identity in records
        if identity is not primary
    ]
    flags.extend(f"attached_block:{identity.block.block_id}" for identity in attached)
    return primary.block, tuple(flags)


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
            # Only where dominance abstained, and never against a supplied
            # expected identity: pick the paper's record if all records agree.
            agreed: FrontMatterBlock | None = None
            agreement_flags: tuple[str, ...] = ()
            if not _has_expected_selectors(expected_identity):
                try:
                    agreed, agreement_flags = _select_agreeing_record(
                        blocks, by_id, detected_title=contents.detected_title
                    )
                except Exception:  # noqa: BLE001 — a fallback must never fail an abstaining paper
                    logger.warning(
                        "Front-matter record agreement failed; abstaining", exc_info=True
                    )
                    agreed, agreement_flags = None, ("record_agreement_error",)
            if agreed is not None:
                selected = agreed
                method = "record_agreement"
                # The selected record may have been trimmed of a cover page.
                blocks = tuple(
                    agreed if block.block_id == agreed.block_id else block for block in blocks
                )
            else:
                reason_flags.append("multiple_plausible_blocks")
            reason_flags.extend(agreement_flags)
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
