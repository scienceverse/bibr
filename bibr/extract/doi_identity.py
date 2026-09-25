"""Source-provenance DOI candidate collection and deterministic selection."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from functools import partial
from typing import TYPE_CHECKING

from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.paper_contents import CanonicalSection
from bibr.pipeline.identity import DoiCandidate, DoiSelection, ExpectedIdentity
from bibr.utils.text import DOI_CANDIDATE_RE, normalize_doi
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.extract.pdf_doi_evidence import PdfDoiEvidence, TextLayerLine

EXPECTED_VISIBLE = 4
EXPLICIT_SELF_ID = 3
FRONT_MATTER_OR_REPEATED_FURNITURE = 2
UNCONTESTED_UNTYPED = 1

# Candidate sources read from the PDF itself (``pdf_doi_evidence``) rather than
# from the parsed text. A text-layer DOI is printed on the page; a link target
# or a metadata DOI is not, so it can only agree with a printed candidate.
TEXT_LAYER = "text_layer"
LINK_ANNOTATION = "link_annotation"
PDF_INFO = "pdf_info"
PDF_XMP = "pdf_xmp"
AGREEMENT_ONLY = "agreement_only"
LINE_JOIN_OVERRUN = "line_join_overrun"

_DOI_RE = DOI_CANDIDATE_RE
_REFERENCE_PREFIX_RE = re.compile(r"^\s*(?:\[\d+[A-Za-z]?\]|\d+[.)]\s)")
# Component and supplement DOIs extend the article DOI they belong to. PLOS
# numbers a figure, table or supporting file with a zero-padded suffix
# (``.g001``, ``.t002``, ``.s003``); the padding is what separates it from the
# BMJ's 2013-14 article DOIs (``bmj.f1049``, ``bmj.g2276``), whose e-locators use
# the same letters. Supplements append a token instead: APA ``.supp``,
# Copernicus ``-supplement``, MDPI ``/s1`` and PeerJ ``/supp-1`` (PeerJ also
# prints ``/fig-1`` and ``/table-1``).
_COMPONENT_SUFFIX_RE = re.compile(
    r"(?:\.(?:[fgst]0\d{2}|fig\d+|table\d+)[A-Za-z]*"
    r"|\.supp|-supplement|/[^/]+/s\d+|/(?:fig|table|supp)-\d+)$",
    re.IGNORECASE,
)
# A sentence that labels a figure, table or supplement is a caption, and a DOI in
# it is the component's (eLife PDFs print "DOI: 10.7554/eLife.00013.003" under
# each figure, a number the suffix rule cannot tell from an article's).
_COMPONENT_LABEL_RE = re.compile(r"\b(?:fig(?:ure)?|table|component|supplement)\s*\d+")
# A supplement issue in a citation line ("30 (Supplement 5)", "Volume 30,
# Supplement 5") is the journal's issue, not such a label.
_SUPPLEMENT_ISSUE_RE = re.compile(r"(?:\(\s*|\bvol(?:ume)?\.?\s*\d+\s*,?\s*)supplement\s*\d+")
_NON_SELF_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"reference\s+doi\s*[:.]?\s*$", re.IGNORECASE), "reference_doi"),
    (re.compile(r"parent(?:\s+article)?\s+doi\s*[:.]?\s*$", re.IGNORECASE), "parent_doi"),
    (
        re.compile(
            r"(?:component|supplement(?:ary)?|figure|table)\s+doi\s*[:.]?\s*$",
            re.IGNORECASE,
        ),
        "component_doi",
    ),
    # APA author notes label the supplement's own DOI: "Supplemental
    # materials: https://doi.org/10.1037/xge0001234.supp".
    (
        re.compile(
            r"supplement(?:al|ary)?\s+(?:materials?|information|data|files?)\s*[:.]\s*$",
            re.IGNORECASE,
        ),
        "component_doi",
    ),
    (
        re.compile(
            r"(?:data(?:set)?|code|software|repository|archive|materials?)\s+doi\s*[:.]?\s*$",
            re.IGNORECASE,
        ),
        "data_doi",
    ),
)
_SELF_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"article\s+doi\s*[:.]?\s*$", re.IGNORECASE), "article_doi"),
    (re.compile(r"please\s+cite\s+as\s*[:.]?\s*$", re.IGNORECASE), "please_cite_as"),
    (
        re.compile(r"abstract\s+citation\s+(?:id|doi)\s*[:.]?\s*$", re.IGNORECASE),
        "abstract_citation_id",
    ),
    (re.compile(r"\bcitation\s*[:.]?\s*$", re.IGNORECASE), "citation"),
    (
        re.compile(r"(?<!journal\s)\b(?:paper\s+)?doi\s*[:.]?\s*$", re.IGNORECASE),
        "explicit_doi",
    ),
)
_REPOSITORY_NAME_RE = re.compile(
    r"\b(?:osf|zenodo|dryad|mendeley(?:\s+data)?|figshare)\b",
    re.IGNORECASE,
)
_FUNDER_REGISTRY_PREFIX = "10.13039/"
# Registrants that mint DOIs only for deposited data, code and materials:
# Zenodo, OSF projects and registrations, Figshare and Dryad. The OSF-hosted
# preprint servers (PsyArXiv ``10.31234/osf.io/...``, SocArXiv ``10.31235``, OSF
# Preprints ``10.31219``, ...) are articles and use registrants of their own.
_DATA_REGISTRANT_RE = re.compile(
    r"10\.(?:5281/zenodo\.|17605/osf\.io/|6084/m9\.figshare\.|5061/dryad\.)",
    re.IGNORECASE,
)
_REPOSITORY_CONTEXT_RE = re.compile(
    r"\b(?:data|datasets?|code|software|materials?)\s+(?:availability|repository)\b"
    r"|\b(?:data|datasets?|code|software|materials?)\s+(?:are|is)\s+"
    r"(?:available|deposited|archived|hosted)\b"
    r"|\brepository\s+(?:doi|record|link|url)\s*[:.]?",
    re.IGNORECASE,
)
_FRONT_MATTER_SECTIONS = frozenset(
    {
        CanonicalSection.TITLE.value,
        CanonicalSection.ABSTRACT.value,
        CanonicalSection.KEYWORDS.value,
    }
)


def _canonical_doi(value: str | None) -> str | None:
    normalized = normalize_doi(value)
    return normalized.casefold() if normalized else None


def doi_sha256(value: str) -> str:
    return hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()


# Publishers print the resolver host with or without ``www.``; without it the
# label in front ("Journal DOI: www.doi.org/…") is left stranded on the prefix
# and every marker pattern below misses.
_DOI_HOST = r"(?:https?://)?(?:www\.)?(?:dx\.)?doi\.org/"


def _marker_kind(text: str, start: int) -> str:
    prefix = text[max(0, start - 100) : start]
    if re.search(rf"journal\s+doi\s*[:.]?\s*{_DOI_HOST}\s*$", prefix, re.IGNORECASE):
        return "journal_doi"
    marker_prefix = re.sub(rf"{_DOI_HOST}\s*$", "", prefix)
    for pattern, kind in _NON_SELF_MARKERS:
        if pattern.search(marker_prefix):
            return kind
    for pattern, kind in _SELF_MARKERS:
        if pattern.search(marker_prefix):
            return kind
    if re.search(rf"{_DOI_HOST}\s*$", prefix, re.IGNORECASE):
        return "doi_url"
    return "bare"


# A citation line that opens with the publication year ("2017. Proc Soc 2,
# 20:1-15. https://doi.org/…", printed as a running header above the title) is
# the article citing itself, not item 2017 of a numbered list.
_YEAR_LED_CITATION_RE = re.compile(r"^\s*(?:19|20)\d{2}[.)]\s")


def _candidate_from_match(
    text: str,
    match: re.Match[str],
    *,
    raw: str,
    source_kind: str,
    page: int | None,
    section_id: int | None,
    section_type: str | None,
    region_index: int | None,
    region_type: str | None,
    text_id: int | None,
    repeated_count: int = 0,
    front_block: bool = False,
) -> DoiCandidate | None:
    normalized = _canonical_doi(match.group(0))
    if normalized is None:
        return None
    marker_kind = _marker_kind(text, match.start())
    lowered = text.casefold()
    # A repository name inside this DOI is not context: PsyArXiv's
    # 10.31234/osf.io/... names OSF in the identifier itself. Another DOI in the
    # sentence still is (a Zenodo DOI beside it marks a software citation).
    context = text.replace(match.group(0), " ")
    section_value = str(section_type or "").casefold()
    # Where a paper names itself: the page furniture, the title, abstract and
    # keywords sections, pages 1-2, and an unpaged input's front block. A DOI
    # label anywhere else is how a cited work's DOI is printed.
    in_front = (
        source_kind in {"header", "footer"}
        or section_value in _FRONT_MATTER_SECTIONS
        or (page is not None and page <= 2)
        or front_block
    )

    rejection_reason = None
    if (
        section_value == CanonicalSection.REFERENCES.value
        or (_REFERENCE_PREFIX_RE.match(text) and not _YEAR_LED_CITATION_RE.match(text))
        or marker_kind == "reference_doi"
    ):
        semantic_context = "reference"
        rejection_reason = "reference_candidate"
        tier = 0
    elif marker_kind in {"parent_doi", "component_doi"}:
        semantic_context = "parent_or_component"
        rejection_reason = "component_candidate"
        tier = 0
    elif marker_kind == "data_doi":
        semantic_context = "data_or_code"
        rejection_reason = "data_or_code_candidate"
        tier = 0
    elif normalized.startswith(_FUNDER_REGISTRY_PREFIX):
        # 10.13039 is Crossref's Funder Registry registrant: these identify a
        # funding body, never an article, and are printed bare inside funding
        # statements where they otherwise land in the front-matter tier.
        semantic_context = "funder_registry"
        rejection_reason = "funder_registry_candidate"
        tier = 0
    elif marker_kind in {"article_doi", "please_cite_as", "abstract_citation_id"}:
        semantic_context = "article_self"
        tier = EXPLICIT_SELF_ID
    elif (
        section_value == CanonicalSection.OPEN_DATA.value
        or _REPOSITORY_NAME_RE.search(context)
        or _REPOSITORY_CONTEXT_RE.search(context)
        or _DATA_REGISTRANT_RE.match(normalized)
    ):
        semantic_context = "data_or_code"
        rejection_reason = "data_or_code_candidate"
        tier = 0
    elif (
        section_value in {CanonicalSection.FIGURE.value, CanonicalSection.TABLE.value}
        or _COMPONENT_SUFFIX_RE.search(normalized)
        or _COMPONENT_LABEL_RE.search(_SUPPLEMENT_ISSUE_RE.sub(" ", lowered))
    ):
        semantic_context = "parent_or_component"
        rejection_reason = "component_candidate"
        tier = 0
    elif marker_kind == "explicit_doi" and in_front:
        semantic_context = "article_self"
        tier = EXPLICIT_SELF_ID
    elif marker_kind == "citation":
        semantic_context = "article_self"
        tier = FRONT_MATTER_OR_REPEATED_FURNITURE
    elif source_kind in {"header", "footer"}:
        semantic_context = "repeated_furniture" if repeated_count > 1 else "structural_furniture"
        tier = FRONT_MATTER_OR_REPEATED_FURNITURE
    elif in_front:
        semantic_context = "front_matter"
        tier = FRONT_MATTER_OR_REPEATED_FURNITURE
    else:
        semantic_context = "untyped"
        tier = UNCONTESTED_UNTYPED

    if marker_kind == "journal_doi" and rejection_reason is None:
        semantic_context = "journal_identity"
        tier = UNCONTESTED_UNTYPED
    # No DOI ends in a slash. One that does ran on into the next field, as when
    # a line join glues the ISSN line under a DOI onto it ("…04.006" +
    # "1234-5678/© 2026 The Authors").
    if rejection_reason is None and normalized.endswith("/"):
        semantic_context = "line_join_overrun"
        rejection_reason = LINE_JOIN_OVERRUN
        tier = 0

    return DoiCandidate(
        raw=raw,
        normalized=normalized,
        source_kind=source_kind,
        page=page,
        section_id=section_id,
        section_type=section_type,
        region_index=region_index,
        region_type=region_type,
        text_id=text_id,
        marker_kind=marker_kind,
        repeated_header_footer_count=repeated_count,
        semantic_context=semantic_context,
        selection_tier=tier,
        rejection_reason=rejection_reason,
    )


# OCR sometimes wraps a DOI between the ``10.`` and its registrant digits
# ("https://doi.org/10. 1016/j.lanwpc.2023. 100933"), which every bridge in
# ``consolidate_text`` misses because they all require an intact ``10.\d{4,9}/``.
# Close that one gap only when an explicit DOI marker sits immediately in front
# of it and real registrant digits plus the path slash follow, so ordinary prose
# containing "10." can never be joined into a fabricated identifier.
_DOI_REGISTRANT_WRAP_RE = re.compile(
    rf"((?:{_DOI_HOST}|\bdoi\s*[:.]?[ \t]*)10\.)[ \t]*\r?\n?[ \t]*(?=\d{{4,9}}/)",
    re.IGNORECASE,
)


def _repair_doi_text(text: str) -> str:
    # Bridge the registrant wrap first: the downstream ``consolidate_text``
    # bridges (dot wrap, slash-space) only engage on an intact registrant.
    bridged = _DOI_REGISTRANT_WRAP_RE.sub(r"\1", text) if "10." in text else text
    cleaned = fix_ocr_artifacts(bridged)
    return re.sub(
        r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]*-)\s+"
        r"([-._;()/:A-Za-z0-9]*[\d.][-._;()/:A-Za-z0-9]*)",
        r"\1\2",
        cleaned,
    )


def _cleaned_char_source_ranges(source: str, cleaned: str) -> list[tuple[int, int]]:
    """Map each cleaned character to its source interval for receipt provenance."""

    ranges: list[tuple[int, int]] = [(0, 0)] * len(cleaned)
    for tag, source_start, source_end, clean_start, clean_end in SequenceMatcher(
        None, source, cleaned, autojunk=False
    ).get_opcodes():
        clean_length = clean_end - clean_start
        source_length = source_end - source_start
        if tag == "equal":
            for offset in range(clean_length):
                source_index = source_start + offset
                ranges[clean_start + offset] = (source_index, source_index + 1)
        elif clean_length:
            for offset in range(clean_length):
                interval_start = source_start + (offset * source_length) // clean_length
                interval_end = (
                    source_start + ((offset + 1) * source_length + clean_length - 1) // clean_length
                )
                ranges[clean_start + offset] = (interval_start, interval_end)
    return ranges


def _raw_source_match(
    source: str,
    match: re.Match[str],
    source_ranges: list[tuple[int, int]],
) -> str:
    if match.start() >= len(source_ranges) or match.end() <= match.start():
        return match.group(0)
    start = source_ranges[match.start()][0]
    end = source_ranges[match.end() - 1][1]
    if end <= start:
        return match.group(0)
    return source[start:end]


def _candidates_from_text(text: str, **provenance) -> list[DoiCandidate]:
    source = text or ""
    cleaned = _repair_doi_text(source)
    matches = tuple(_DOI_RE.finditer(cleaned))
    if not matches:
        return []
    source_ranges = _cleaned_char_source_ranges(source, cleaned)
    return [
        candidate
        for match in matches
        if (
            candidate := _candidate_from_match(
                cleaned,
                match,
                raw=_raw_source_match(source, match, source_ranges),
                **provenance,
            )
        )
        is not None
    ]


def _publication_table_doi_candidates(contents) -> list[DoiCandidate]:
    """Recover a publisher's identity box above one selected article title.

    Layout models sometimes call the journal's masthead a table. Require spatial
    ownership, complete OCR, one explicit DOI, an ISSN and publication furniture; ordinary data,
    literature-review tables and ambiguous multi-record pages remain excluded.
    """
    resolution = getattr(contents, "front_matter_resolution", None)
    if resolution is None or resolution.selected_block_id is None:
        return []
    by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    selected = next(
        (block for block in resolution.blocks if block.block_id == resolution.selected_block_id),
        None,
    )
    if selected is None:
        return []
    titles = [
        by_id[key]
        for key in selected.title_candidate_ids
        if key in by_id and by_id[key].page is not None and by_id[key].bbox is not None
    ]
    if not titles:
        return []
    page = min(candidate.page for candidate in titles)
    top = min(candidate.bbox[1] for candidate in titles if candidate.page == page)
    for block in resolution.blocks:
        if block.block_id == selected.block_id:
            continue
        pages = {by_id[key].page for key in block.candidate_ids if key in by_id}
        if not pages or None in pages or page in pages:
            return []

    candidates = []
    for region in getattr(contents, "region_summaries", ()):
        if region.label != "table" or region.page != page or region.bbox is None:
            continue
        if region.bbox[3] > top:
            continue
        # A captioned table is a document object, even when its cells contain
        # publication metadata. It is not the article's masthead.
        if any(
            table.caption
            and any(p.page_no == page and p.bbox == region.bbox for p in table.provenance)
            for table in getattr(contents, "tables", ())
        ):
            continue
        # The compact summary may be truncated at 200 characters. Only complete
        # retained OCR can establish that a publisher box has one DOI. Using it
        # also recovers evidence from a one-row box dropped by the table parser,
        # without inventing an additional scientific table in the public export.
        raw = region.canonical_ocr_content or region.raw_ocr_content
        if not raw or len(raw) > 2000:
            continue
        text = re.sub(r"<[^>]*>", "\n", raw)
        # Journal identifiers belong to the serial, even in an otherwise owned
        # publisher box. Keep the label intact before restoring field breaks.
        if re.search(r"\bjournal\s+doi\s*[:.]", text, re.IGNORECASE):
            continue
        # OCR can concatenate masthead lines, including a year immediately
        # followed by DOI. Restore boundaries only at explicit field labels.
        text = re.sub(
            r"(?<!\n)(?=DOI\s*:\s*10\.\d{4,9}/|Article\s+Number\s*:"
            r"|(?:e[- ]?)?ISSN\s*:?\s*\d{4}[- ]\d{3}[\dX]"
            r"|Copyright\s*(?:©|\(c\)|\d{4}))",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        if len(re.findall(r"\bDOI\s*:\s*10\.\d{4,9}/", text, re.IGNORECASE)) != 1:
            continue
        if not re.search(r"\b(?:e[- ]?)?ISSN\s*:?\s*\d{4}[- ]\d{3}[\dX]\b", text, re.IGNORECASE):
            continue
        if not re.search(
            r"\b(?:Vol(?:ume)?\.?\s*\d|Copyright\b|Article\s+Number\s*:)", text, re.IGNORECASE
        ):
            continue
        found = _candidates_from_text(
            text,
            source_kind="publication_region",
            page=page,
            section_id=region.section_id,
            section_type=None,
            region_index=region.index,
            region_type="publication_metadata",
            text_id=None,
        )
        # A second bare/linked DOI still makes this box ambiguous. Do not select
        # a favourite just because only one candidate has the explicit marker.
        if len({candidate.normalized.casefold() for candidate in found}) == 1:
            candidates.extend(found)
    return candidates


def _sentence_region_index(sentence) -> int | None:
    """Index of the layout region that began the sentence's paragraph.

    Together with the sentence's page it must name an ``extraction.regions``
    row. A paragraph joined across a page break assigns its later sentences to
    the page they were printed on, where that region does not exist, so those
    sentences get None.
    """
    region_meta = sentence.region_meta or {}
    region_page = region_meta.get("region_page")
    if region_page is None or region_page != sentence.page_number:
        return None
    return region_meta.get("region_index")


_UNCLASSIFIED_FRONT_SECTIONS = frozenset({None, CanonicalSection.UNKNOWN, CanonicalSection.TITLE})


def _pageless_front_block_end(contents, section_map) -> int | None:
    """First text id after the front block of an input without pages.

    Native parses (DOCX, ePub, HTML, JATS) give no sentence a page, so "page 1
    or 2" cannot mark their title page. The front block is the unclassified
    text (the Root section, a title heading) before the first sentence of a
    classified section, usually the Abstract. Without a classified section
    there is no bound, and nothing counts.
    """

    if not contents.sentences or any(s.page_number is not None for s in contents.sentences):
        return None
    classified = [
        sentence.text_id
        for sentence in contents.sentences
        if (section := section_map.get(sentence.section_id)) is not None
        and section.section_type not in _UNCLASSIFIED_FRONT_SECTIONS
    ]
    return min(classified, default=None)


def collect_doi_candidates(
    contents, pdf_evidence: PdfDoiEvidence | None = None
) -> tuple[DoiCandidate, ...]:
    """Collect every source-visible DOI with sentence or furniture provenance.

    With *pdf_evidence* (a PDF input) the pool also takes the DOIs the front
    pages' text layer prints outside the parsed text, and the link targets and
    document metadata as agreement-only rows; see ``_with_pdf_evidence``.
    """

    section_map = {section.section_id: section for section in contents.sections}
    front_block_end = _pageless_front_block_end(contents, section_map)
    candidates: list[DoiCandidate] = []
    for sentence in contents.sentences:
        section = section_map.get(sentence.section_id)
        region_meta = sentence.region_meta or {}
        candidates.extend(
            _candidates_from_text(
                sentence.text,
                source_kind="sentence",
                page=sentence.page_number,
                section_id=sentence.section_id,
                section_type=section.section_type.value
                if section and section.section_type
                else None,
                region_index=_sentence_region_index(sentence),
                region_type=region_meta.get("region_type"),
                text_id=sentence.text_id,
                front_block=front_block_end is not None and sentence.text_id < front_block_end,
            )
        )

    furniture = [("header", line) for line in contents.detected_headers] + [
        ("footer", line) for line in contents.detected_footers
    ]
    counts = Counter(line.strip().casefold() for _, line in furniture)
    for source_kind, line in furniture:
        candidates.extend(
            _candidates_from_text(
                line,
                source_kind=source_kind,
                page=None,
                section_id=None,
                section_type=None,
                region_index=None,
                region_type=source_kind,
                text_id=None,
                repeated_count=counts[line.strip().casefold()],
            )
        )

    preparsed = getattr(contents, "preparsed_metadata", None)
    structured_doi = _canonical_doi(getattr(preparsed, "doi", None))
    if structured_doi:
        candidates.append(
            DoiCandidate(
                raw=str(preparsed.doi),
                normalized=structured_doi,
                source_kind="structured_metadata",
                page=None,
                section_id=None,
                section_type=None,
                region_index=None,
                region_type=None,
                text_id=None,
                marker_kind="structured_doi",
                repeated_header_footer_count=0,
                semantic_context="article_self",
                selection_tier=EXPLICIT_SELF_ID,
            )
        )
    candidates.extend(_publication_table_doi_candidates(contents))
    if pdf_evidence is not None:
        candidates = _with_pdf_evidence(contents, candidates, pdf_evidence, section_map)
    return tuple(candidates)


# Layout labels whose text a DOI inside them belongs to: a reference entry, a
# table, or a figure and its caption. The parse keeps these out of the body
# text, so their sentences carry these section types.
_REFERENCE_LABELS = frozenset({"reference", "reference_content"})
_FIGURE_LABELS = frozenset({"image", "chart", "figure_title", "header_image", "footer_image"})
# Characters that may follow a DOI without being part of the token after it.
_DOI_CLOSERS = frozenset(".,;:)]}>\"'\u2019\u201d")
_BANNER_RE = re.compile(r"\bfirst\s+published\s+as\b", re.IGNORECASE)


def _region_at(regions: list, point: tuple[float, float]):
    """The smallest layout region on the page that contains *point*."""
    x, y = point
    best = None
    for region in regions:
        if region.bbox is None:
            continue
        x1, y1, x2, y2 = region.bbox
        if x1 - 2 <= x <= x2 + 2 and y1 - 2 <= y <= y2 + 2:
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best[0]:
                best = (area, region)
    return best[1] if best is not None else None


def _region_context(region, section_map) -> tuple[int | None, str | None]:
    """Section id and type a DOI printed inside *region* gets, as its sentences would."""
    if region is None:
        return None, None
    if region.label in _REFERENCE_LABELS:
        return region.section_id, CanonicalSection.REFERENCES.value
    if region.label == "table":
        return region.section_id, CanonicalSection.TABLE.value
    if region.label in _FIGURE_LABELS:
        return region.section_id, CanonicalSection.FIGURE.value
    section = section_map.get(region.section_id)
    if section is None or section.section_type is None:
        return region.section_id, None
    return region.section_id, section.section_type.value


def _text_layer_candidates(
    line: TextLayerLine, regions: list, section_map
) -> list[tuple[DoiCandidate, str]]:
    """Candidates of one text-layer line, each with the text after it on the line.

    Each takes the context of the layout region it is printed in, so a DOI in a
    reference entry, a table or a figure is rejected as its sentence would be.
    Text outside every region (a margin banner, a masthead the layout missed)
    has only its page and its own line for context. A repository banner states
    the article's DOI ("… first published as 10.1234/… on 1 May 1999.
    Downloaded from …"); the text layer may give its phrases in either order.
    """

    source = line.text
    cleaned = _repair_doi_text(source)
    matches = tuple(_DOI_RE.finditer(cleaned))
    if not matches:
        return []
    ranges = _cleaned_char_source_ranges(source, cleaned)
    banner = bool(_BANNER_RE.search(cleaned))
    found = []
    for match in matches:
        start = ranges[match.start()][0]
        end = max(ranges[match.end() - 1][1], start + 1)
        points = [point for point in line.centers[start:end] if point is not None]
        region = None
        if points:
            center = (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
            region = _region_at(regions, center)
        section_id, section_type = _region_context(region, section_map)
        candidate = _candidate_from_match(
            cleaned,
            match,
            raw=source[start:end],
            source_kind=TEXT_LAYER,
            page=line.page,
            section_id=section_id,
            section_type=section_type,
            region_index=region.index if region is not None else None,
            region_type=region.label if region is not None else None,
            text_id=None,
        )
        if candidate is None:
            continue
        if banner and candidate.rejection_reason is None:
            candidate = replace(
                candidate,
                marker_kind="first_published_as",
                semantic_context="article_self",
                selection_tier=EXPLICIT_SELF_ID,
            )
        # The line after the DOI, including punctuation normalization dropped.
        doi_end = cleaned.casefold().find(candidate.normalized, match.start())
        doi_end = doi_end + len(candidate.normalized) if doi_end >= 0 else match.end()
        found.append((candidate, cleaned[doi_end:]))
    return found


def _agreement_rows(evidence: PdfDoiEvidence) -> list[DoiCandidate]:
    """One row per DOI a link target or the document metadata names.

    A link target counts once per page. ``semantic_context`` says whether the
    page prints that DOI under or next to the link (``printed_link``); the
    printed DOI itself is a candidate of the text it is printed in.
    """

    rows: list[DoiCandidate] = []
    seen: set[tuple[str, int | None, str]] = set()

    def row(raw, normalized, source_kind, page, marker_kind, semantic_context) -> None:
        if (source_kind, page, normalized) in seen:
            return
        seen.add((source_kind, page, normalized))
        rows.append(
            DoiCandidate(
                raw=raw[:300],
                normalized=normalized,
                source_kind=source_kind,
                page=page,
                section_id=None,
                section_type=None,
                region_index=None,
                region_type=None,
                text_id=None,
                marker_kind=marker_kind,
                repeated_header_footer_count=0,
                semantic_context=semantic_context,
                selection_tier=0,
                rejection_reason=AGREEMENT_ONLY,
            )
        )

    for link in evidence.links:
        normalized = _canonical_doi(link.doi)
        if normalized is None:
            continue
        printed = normalized in "".join(link.printed_text.split()).casefold()
        context = "printed_link" if printed else "link_target"
        row(link.uri, normalized, LINK_ANNOTATION, link.page, "link_uri", context)
    for item in evidence.metadata:
        for match in _DOI_RE.finditer(item.value):
            normalized = _canonical_doi(match.group(0))
            if normalized is not None:
                row(match.group(0), normalized, item.source, None, item.key, "pdf_metadata")
    return rows


def _overruns_line(
    parsed: DoiCandidate, reading: DoiCandidate, tail: str, line: TextLayerLine, agreeing
) -> bool | None:
    """Whether *parsed* ran past the line end where the text layer ends *reading*.

    *tail* is what the line prints after *reading*. None when *parsed* does
    not continue *reading* onto the next line. Otherwise the parse joined the
    two lines: True when the join ran into the next field (the continuation is
    glued to more text, or only *reading* agrees with the link and metadata
    DOIs), False for a DOI wrapped onto the next line.
    """

    rest = parsed.normalized[len(reading.normalized) :]
    joined = ("".join(tail.split()) + line.next_text.lstrip()).casefold()
    if not rest or not joined.startswith(rest):
        return None
    after = joined[len(rest) : len(rest) + 1]
    glued = bool(after) and not after.isspace() and after not in _DOI_CLOSERS
    return glued or (reading.normalized in agreeing and parsed.normalized not in agreeing)


def _with_pdf_evidence(
    contents, parsed: list[DoiCandidate], evidence: PdfDoiEvidence, section_map
) -> list[DoiCandidate]:
    """Add the PDF's own DOI evidence to the parsed-text candidates.

    A text-layer DOI joins the pool unless the parsed text already holds it on
    that page (or in the pageless page furniture) or reads more of it (the
    text layer broke a wrapped DOI at the line end), or the parse read the same
    layout region's DOI differently: the parse chose that region's text. The
    line geometry also sets a DOI's end: a parsed DOI that runs from a line's
    end into the next field, such as an ISSN printed on the next line
    ("…04.006" + "1234-5678/© 2026"), is rejected. Link targets and metadata
    DOIs are added as agreement-only rows.
    """

    regions_by_page: dict[int, list] = defaultdict(list)
    for region in getattr(contents, "region_summaries", ()) or ():
        if region.page is not None:
            regions_by_page[region.page].append(region)
    rows = _agreement_rows(evidence)
    agreeing = _agreeing_dois(rows)

    readings = [
        (candidate, tail, line)
        for line in evidence.lines
        for candidate, tail in _text_layer_candidates(
            line, regions_by_page.get(line.page, []), section_map
        )
    ]
    checked: list[DoiCandidate] = []
    for candidate in parsed:
        for reading, tail, line in readings:
            if (
                candidate.rejection_reason is None
                and candidate.page in (None, reading.page)
                and len(candidate.normalized) > len(reading.normalized)
                and candidate.normalized.startswith(reading.normalized)
                and all(ch.isspace() or ch in _DOI_CLOSERS for ch in tail)
                and _overruns_line(candidate, reading, tail, line, agreeing)
            ):
                candidate = replace(
                    candidate,
                    semantic_context="line_join_overrun",
                    selection_tier=0,
                    rejection_reason=LINE_JOIN_OVERRUN,
                )
                break
        checked.append(candidate)

    def read_by_parse(reading: DoiCandidate) -> bool:
        for candidate in checked:
            if candidate.page not in (None, reading.page):
                continue
            if candidate.normalized == reading.normalized:
                return True
            if candidate.rejection_reason != LINE_JOIN_OVERRUN and candidate.normalized.startswith(
                reading.normalized
            ):
                return True
            if (
                reading.region_index is not None
                and candidate.page == reading.page
                and candidate.region_index == reading.region_index
                and not reading.normalized.startswith(candidate.normalized)
            ):
                return True
        return False

    added: list[DoiCandidate] = []
    for reading, _tail, _line in readings:
        if read_by_parse(reading) or any(
            (c.page, c.normalized) == (reading.page, reading.normalized) for c in added
        ):
            continue
        added.append(reading)
    return [*checked, *added, *rows]


_FURNITURE_SOURCES = frozenset({"header", "footer"})
_NUMERIC_EXTENSION_RE = re.compile(r"^[/.]\d")
# The end a parse can lose off a printed DOI: a few characters, none a letter.
_LOST_END_RE = re.compile(r"^[-._;()/:0-9]{1,4}$")


def _distinct_dois(candidates: list[DoiCandidate]) -> set[str]:
    return {candidate.normalized.casefold() for candidate in candidates}


def _drop_truncated_prefixes(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Drop a DOI another candidate strictly extends at a ``/`` or ``.`` boundary.

    Only a numeric extension counts: mastheads print journal-level stems that
    the article DOI extends with an issue or article number
    (``10.30574/wjarr`` → ``10.30574/wjarr.2022.14.3.0574``), while supplement
    and component spellings extend with a letter (``/s1``, ``.s001``, ``.g001``)
    and must never displace the article they belong to.

    A text-layer DOI is read from one printed line, so a parsed DOI it extends
    by a few characters without letters lost its end in the parse
    (``…/25.202.3`` for the printed ``…/25.202.33``) and is dropped too.
    """

    values = _distinct_dois(candidates)
    complete = {c.normalized.casefold() for c in candidates if c.source_kind == TEXT_LAYER}
    truncated = {
        value
        for value in values
        for other in values
        if other != value
        and other.startswith(value)
        and (
            _NUMERIC_EXTENSION_RE.match(other[len(value) :])
            or (other in complete and _LOST_END_RE.match(other[len(value) :]))
        )
    }
    kept = [c for c in candidates if c.normalized.casefold() not in truncated]
    return kept or candidates


def _prefer_agreeing(
    candidates: list[DoiCandidate], agreeing: frozenset[str]
) -> list[DoiCandidate]:
    """Prefer a DOI that a link target or the document metadata also names.

    The PDF's own metadata and its link targets name the paper's DOI far more
    often than any other, so of two printed rivals the one they name is kept.
    """

    agreed = [c for c in candidates if c.normalized.casefold() in agreeing]
    return agreed or candidates


def _prefer_structured(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Prefer the publisher's structured article DOI (JATS, HTML meta) over body text.

    Body text of the same tier is figure, box or supplement furniture
    (eLife's ``DOI: 10.7554/eLife.00013.005``) or a cited work.
    """

    structured = [c for c in candidates if c.source_kind == "structured_metadata"]
    return structured or candidates


def _prefer_marked(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Prefer a labelled or resolver-URL DOI over one printed with no marker."""

    marked = [c for c in candidates if c.marker_kind != "bare"]
    return marked or candidates


def _prefer_body_sources(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Prefer a DOI read from the page body over running header/footer furniture.

    A header or footer printed once may be an OCR twin or a stray reference line.
    One the running furniture repeats on several pages is the paper's own and is
    kept, so a DOI cited in an early footnote cannot displace it: the tie then
    abstains.
    """

    body = [
        c
        for c in candidates
        if c.source_kind not in _FURNITURE_SOURCES or c.repeated_header_footer_count > 1
    ]
    return body or candidates


def _prefer_lowest_page(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Prefer the earliest page; the paper's own DOI is printed in front matter."""

    if any(c.page is None for c in candidates):
        return candidates
    lowest = min(c.page for c in candidates)  # type: ignore[type-var]
    return [c for c in candidates if c.page == lowest]


# Lowest tier that can name the paper without an expected DOI.
_MIN_IDENTITY_TIER = FRONT_MATTER_OR_REPEATED_FURNITURE


def _tie_break_ladder(agreeing: frozenset[str]):
    return (
        _prefer_structured,
        partial(_prefer_agreeing, agreeing=agreeing),
        _drop_truncated_prefixes,
        _prefer_marked,
        _prefer_body_sources,
        _prefer_lowest_page,
    )


def _agreeing_dois(candidates) -> frozenset[str]:
    """DOIs the PDF names outside its printed text.

    The document metadata counts, and a link whose text is not the DOI itself
    (a journal citation line, a "cite this" button). A link over a printed DOI
    only repeats that print, and front pages link a cited work's DOI as
    readily as the paper's own.
    """
    return frozenset(
        c.normalized.casefold()
        for c in candidates
        if c.source_kind in (PDF_INFO, PDF_XMP)
        or (c.source_kind == LINK_ANNOTATION and c.semantic_context == "link_target")
    )


def _confirmed_by_agreement(
    candidates: tuple[DoiCandidate, ...], agreeing: frozenset[str]
) -> DoiCandidate | None:
    """The one printed tier-1 DOI that a link target or the metadata also names.

    A tier-1 DOI alone does not name the paper: printed unmarked in the body it
    is usually a cited work. When the PDF's metadata or a link target names the
    same DOI, it is the paper's own, printed where the tiers do not look.
    """

    confirmed = [
        c
        for c in candidates
        if c.rejection_reason is None
        and c.selection_tier == UNCONTESTED_UNTYPED
        and c.normalized.casefold() in agreeing
    ]
    return confirmed[0] if len(_distinct_dois(confirmed)) == 1 else None


def _select_without_expected(
    candidates: tuple[DoiCandidate, ...], *, min_tier: int = UNCONTESTED_UNTYPED
) -> DoiSelection:
    agreeing = _agreeing_dois(candidates)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.rejection_reason is None and candidate.selection_tier >= min_tier
    ]
    if not eligible:
        confirmed = _confirmed_by_agreement(candidates, agreeing) if agreeing else None
        return DoiSelection(confirmed, candidates, ())
    highest_tier = max(candidate.selection_tier for candidate in eligible)
    highest = [candidate for candidate in eligible if candidate.selection_tier == highest_tier]
    if len(_distinct_dois(highest)) > 1:
        evidence = tuple(
            f"text:{candidate.text_id}" if candidate.text_id is not None else candidate.source_kind
            for candidate in highest
        )
        # A tier tie is real ambiguity and is always reported, but one spurious
        # co-tier candidate must not suppress a correctly read DOI: walk a
        # deterministic provenance ladder and take the winner only if the tie
        # collapses to a single DOI. Otherwise abstain, as before.
        resolved = highest
        for rule in _tie_break_ladder(agreeing):
            if len(_distinct_dois(resolved)) == 1:
                break
            resolved = rule(resolved)
        selected = resolved[0] if len(_distinct_dois(resolved)) == 1 else None
        message = f"Conflicting source-visible DOI candidates at tier {highest_tier}"
        if selected is not None:
            message += f", resolved by source provenance to {selected.normalized}"
        issue = ValidationIssue(
            code="VAL_DOI_AMBIGUOUS",
            severity=IssueSeverity.WARNING,
            message=message,
            origin_stage="identity",
            evidence_ids=evidence,
        )
        return DoiSelection(selected, candidates, (issue,))
    return DoiSelection(highest[0], candidates, ())


def select_doi_candidates(
    candidates, expected_identity: ExpectedIdentity | None = None
) -> DoiSelection:
    """Select by explicit evidence tiers, then by source provenance within a tier.

    An equal-tier conflict always raises ``VAL_DOI_AMBIGUOUS``; it resolves to a
    candidate only when the provenance ladder in ``_select_without_expected``
    collapses it to a single DOI, and abstains otherwise.

    Without a matching expected DOI only tiers 2 and 3 can name the paper. A
    tier-1 candidate is a DOI that no label names the article's (a bare DOI or
    a doi.org link) printed outside the front matter and the running
    furniture, or a journal-level DOI. In a manuscript with no DOI of its own
    such a DOI is a cited work. It becomes the paper's DOI only when it matches
    the expected identity; when only tier-1 candidates remain, a required or
    mismatched expected DOI reports ``VAL_EXPECTED_ID_MISSING``. The front
    matter is the title, abstract and keywords sections, pages 1-2, and in an
    input without pages the unclassified block before its first classified
    section.
    """

    candidate_tuple = tuple(candidates)
    if expected_identity is None or not (
        expected_identity.expected_doi or expected_identity.expected_doi_sha256
    ):
        selection = _select_without_expected(candidate_tuple, min_tier=_MIN_IDENTITY_TIER)
        if expected_identity and expected_identity.doi_required and selection.selected is None:
            issue = ValidationIssue(
                code="VAL_EXPECTED_ID_MISSING",
                severity=IssueSeverity.ERROR,
                message="A DOI is required but no eligible source-visible DOI candidate was found",
                origin_stage="identity",
                blocking=True,
                evidence_ids=(expected_identity.queue_record_id,),
            )
            return replace(selection, issues=(*selection.issues, issue))
        return selection

    expected_doi = _canonical_doi(expected_identity.expected_doi)
    expected_hash = (
        expected_identity.expected_doi_sha256.casefold()
        if expected_identity.expected_doi_sha256
        else None
    )
    matching_indexes = [
        index
        for index, candidate in enumerate(candidate_tuple)
        if candidate.rejection_reason is None
        and (
            (
                expected_doi is not None
                and candidate.normalized.casefold() == expected_doi.casefold()
            )
            or (expected_hash is not None and doi_sha256(candidate.normalized) == expected_hash)
        )
    ]
    if matching_indexes:
        boosted = tuple(
            replace(candidate, selection_tier=EXPECTED_VISIBLE)
            if index in matching_indexes
            else candidate
            for index, candidate in enumerate(candidate_tuple)
        )
        return DoiSelection(boosted[matching_indexes[0]], boosted, ())

    fallback = _select_without_expected(candidate_tuple, min_tier=_MIN_IDENTITY_TIER)
    if fallback.selected is None:
        issue = ValidationIssue(
            code="VAL_EXPECTED_ID_MISSING",
            severity=IssueSeverity.ERROR,
            message="Expected DOI is not present in eligible source-visible evidence",
            origin_stage="identity",
            evidence_ids=(expected_identity.queue_record_id,),
            blocking=True,
        )
    else:
        issue = ValidationIssue(
            code="VAL_EXPECTED_ID_MISMATCH",
            severity=IssueSeverity.ERROR,
            message=(
                f"Source-selected DOI {fallback.selected.normalized!r} does not match expected identity"
            ),
            origin_stage="identity",
            evidence_ids=(expected_identity.queue_record_id,),
            blocking=True,
        )
    return replace(fallback, issues=(*fallback.issues, issue))
