"""Preserve separately printed metadata variants with conservative ownership.

Variant capture never translates or mutates model metadata. A title must own
independent local article anatomy, so adjacent title fragments and subtitles
are not mistaken for alternate titles. An abstract needs both a printed
abstract heading and original abstract-labelled paragraph regions; a section
classifier label alone cannot establish that all its text belongs together.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from bibr.extract.front_matter import _record_title_indices
from bibr.utils.metadata import PRINTED_ABSTRACT_LABELS, PRINTED_ABSTRACT_PREFIX

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterCandidate, FrontMatterResolution
    from bibr.paper_contents import PaperContents


@dataclass(frozen=True)
class PrintedMetadataVariant:
    variant_id: str
    record_id: str
    field: Literal["title", "abstract"]
    text: str
    language: str | None
    source_text_ids: tuple[int, ...]
    source_section_ids: tuple[int, ...]
    pages: tuple[int, ...]
    is_primary: bool
    presentation_ids: tuple[str, ...] = ()
    byline_source_text_ids: tuple[int, ...] = ()
    byline_source_section_ids: tuple[int, ...] = ()


# Printed labels are boundary evidence, not proof of the prose's language.
# Language remains unknown until the source supplies explicit language metadata.
_ABSTRACT_LABELS = PRINTED_ABSTRACT_LABELS
_UNSAFE_ROLES = frozenset(
    {
        "title",
        "byline",
        "affiliation",
        "doi",
        "correspondence",
        "metadata",
        "structural",
        "keywords",
    }
)


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def has_explicit_original_label(resolution, field: Literal["title", "abstract"]) -> bool:
    """Defer source-order preference when selected evidence identifies an original."""
    if resolution is None or resolution.selected_block_id is None:
        return False
    members = {
        candidate_id
        for block in resolution.blocks
        if block.block_id == resolution.selected_block_id
        for candidate_id in block.candidate_ids
    }
    pattern = re.compile(
        rf"^\s*(?:original(?:-language)?\s+{field}|"
        rf"{field}\s+in\s+(?:the\s+)?original\s+language)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    return any(
        pattern.search(candidate.raw_text)
        for candidate in resolution.candidates
        if candidate.candidate_id in members
    )


def _ordered_union(*groups: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _variant(
    record_id: str,
    field: Literal["title", "abstract"],
    text: str,
    sources: list[FrontMatterCandidate],
) -> PrintedMetadataVariant:
    return PrintedMetadataVariant(
        variant_id="",  # Stable field-local numbering is assigned after deduplication.
        record_id=record_id,
        field=field,
        text=text.strip(),
        language=None,
        source_text_ids=tuple(
            dict.fromkeys(text_id for source in sources for text_id in source.text_ids)
        ),
        source_section_ids=tuple(
            dict.fromkeys(source.section_id for source in sources if source.section_id is not None)
        ),
        pages=tuple(sorted({source.page for source in sources if source.page is not None})),
        is_primary=False,
    )


def _abstract_variants(contents, selected, resolution):
    sections = {section.section_id: section for section in contents.sections}
    explicit_roots = {
        candidate.section_id: candidate
        for candidate in selected
        if candidate.source_kind == "heading"
        and candidate.section_id in sections
        and not sections[candidate.section_id].header_is_synthetic
        and _normalized(sections[candidate.section_id].header).casefold().rstrip(":.")
        in _ABSTRACT_LABELS
    }

    def root_for(section_id):
        visited = set()
        while section_id in sections and section_id not in visited:
            if section_id in explicit_roots:
                return section_id
            visited.add(section_id)
            section_id = sections[section_id].parent_section_id
        return None

    if any(
        ("abstract" in candidate.roles or (candidate.region_label or "").casefold() == "abstract")
        and root_for(candidate.section_id) is None
        for candidate in selected
    ):
        # An unsupported or inferred abstract may precede the recognized
        # printed variants. Omitting it would make a later language appear
        # first, so the primary choice is not established for this field.
        return []

    variants = []
    for root_id, heading in explicit_roots.items():
        sources = [candidate for candidate in selected if root_for(candidate.section_id) == root_id]
        paragraph_sources = [
            candidate for candidate in sources if candidate.source_kind == "paragraph"
        ]
        if not paragraph_sources:
            return []
        # Shared sections can contain neighbouring articles even when a coarse
        # resolver selected this block. Every source row needs local ownership.
        source_rows = {
            sentence.text_id
            for sentence in contents.sentences
            if root_for(sentence.section_id) == root_id and sentence.text.strip()
        }
        candidate_rows = {text_id for source in paragraph_sources for text_id in source.text_ids}
        if (
            not source_rows
            or source_rows != candidate_rows
            or not source_rows.issubset(resolution.allowed_text_ids)
        ):
            return []
        if any(
            (candidate.region_label or "").casefold() != "abstract"
            or not candidate.roles.isdisjoint(_UNSAFE_ROLES)
            for candidate in paragraph_sources
        ):
            return []
        # A hidden article heading or affiliation inside the abstract tree is
        # not a structured abstract subheading. Decline the whole capture.
        if any(not candidate.roles.isdisjoint(_UNSAFE_ROLES) for candidate in sources):
            return []
        chunks = []
        for candidate in sources:
            if candidate.candidate_id == heading.candidate_id:
                continue
            if candidate.source_kind == "heading":
                section = sections.get(candidate.section_id)
                if section is None or section.header_is_synthetic:
                    continue
            if candidate.raw_text.strip():
                chunks.append(candidate.raw_text.strip())
        if chunks:
            variants.append(
                (
                    heading.reading_order,
                    _variant(resolution.selected_block_id, "abstract", "\n".join(chunks), sources),
                )
            )
        else:
            return []
    # A partial inventory cannot establish which supported language was the
    # first printed variant. Never promote a later variant after silently
    # skipping an unsupported earlier explicit abstract.
    return variants


_KEYWORDS_PREFIX = re.compile(
    r"^\s*(?:keywords|key words|palavras[- ]chave|palabras clave|kata kunci)\s*[:：]",
    re.IGNORECASE,
)
_CITATION_LABELS = frozenset({"citation", "how to cite", "how to cite this article"})
_CITATION_PREFIX = re.compile(
    r"^\s*(?:citation|how to cite(?: this article)?)\s*[:：]", re.IGNORECASE
)


def _region_abstract_variants(contents, selected, resolution):
    """Capture printed abstract boundaries independently of semantic sections.

    Section normalization may place several abstracts and their keywords in one
    section. Physical headings, inline labels and keyword rows delimit the source
    instead. Each prose region must have complete, exclusive sentence ownership;
    only an abstract-labelled region may continue an inline-labelled text region.
    """
    regions = sorted(contents.region_summaries, key=lambda row: (row.page, row.index))
    keys = [(row.page, row.index) for row in regions]
    boxes = [(row.page, row.bbox) for row in regions if row.bbox is not None]
    if len(keys) != len(set(keys)) or len(boxes) != len(set(boxes)):
        return []

    def paragraph_sources(region, printed_text):
        if region.bbox is None:
            return []
        sentences = [
            row
            for row in contents.sentences
            if any(
                point.page_no == region.page and point.bbox == region.bbox
                for point in row.provenance
            )
        ]
        ids = {row.text_id for row in sentences}
        sources = [row for row in selected if ids & set(row.text_ids)]
        owned_ids = [key for row in sources for key in row.text_ids]
        if (
            not ids
            or not ids.issubset(resolution.allowed_text_ids)
            or ids != set(owned_ids)
            or len(owned_ids) != len(set(owned_ids))
            or any(row.source_kind != "paragraph" for row in sources)
            or _normalized(" ".join(row.text for row in sentences)) != _normalized(printed_text)
            or _normalized(" ".join(row.raw_text for row in sources)) != _normalized(printed_text)
            or any(
                any(
                    point.page_no != region.page or point.bbox != region.bbox
                    for point in row.provenance
                )
                for row in sentences
            )
        ):
            return []
        return sources

    variants = []
    sources = []
    chunks = []
    start_page = None
    citation_box = False

    def finish():
        if chunks:
            variants.append(
                (
                    min(row.reading_order for row in sources),
                    _variant(resolution.selected_block_id, "abstract", "\n".join(chunks), sources),
                )
            )
        sources.clear()
        chunks.clear()

    for region in regions:
        if region.label in {"header", "footer", "footnote", "vision_footnote"}:
            continue
        printed_text = region.canonical_ocr_content or region.content or ""
        if region.label in {"doc_title", "paragraph_title"}:
            # A heading is source evidence only if the selected record owns it.
            headings = [
                row
                for row in selected
                if row.source_kind == "heading"
                and row.page == region.page
                and row.bbox is not None
                and row.bbox == region.bbox
                and _normalized(row.raw_text) == _normalized(printed_text)
            ]
            if sources and not chunks:
                return []  # Do not skip an empty earlier printed abstract.
            finish()
            label = _normalized(printed_text).casefold().rstrip(":.")
            citation_box = label in _CITATION_LABELS
            if label in _ABSTRACT_LABELS:
                if len(headings) == 1:
                    sources.extend(headings)
                    start_page = region.page
                elif headings:
                    return []
            continue
        if _KEYWORDS_PREFIX.match(printed_text) or _CITATION_PREFIX.match(printed_text):
            if sources and not chunks:
                return []
            finish()
            continue
        inline = PRINTED_ABSTRACT_PREFIX.match(printed_text)
        if inline and region.label in {"abstract", "text"}:
            if sources and not chunks:
                return []
            finish()
            owned = paragraph_sources(region, printed_text)
            if not owned:
                # Unselected records must not contribute even a partial field.
                if any(
                    row.page == region.page and row.bbox == region.bbox for row in selected
                ) or any(
                    row.text_id in resolution.allowed_text_ids
                    and any(
                        point.page_no == region.page and point.bbox == region.bbox
                        for point in row.provenance
                    )
                    for row in contents.sentences
                ):
                    return []
                continue
            sources.extend(owned)
            chunks.append(inline.group(1).strip())
            start_page = region.page
            citation_box = False
        elif sources:
            if region.label != "abstract" or region.page > start_page + 1:
                return []  # An unexplained gap cannot certify a complete abstract.
            owned = paragraph_sources(region, printed_text)
            if not owned:
                return []
            sources.extend(owned)
            chunks.append(printed_text.strip())
        elif (
            region.label == "abstract"
            and not citation_box
            and paragraph_sources(region, printed_text)
        ):
            return []  # Never silently promote a later label over an unlabelled abstract.
        if chunks and re.search(
            r"(?im)^\s*(?:(?:introduction|references|bibliography)\s*[:：]?\s*$|"
            r"(?:keywords|key words|palavras[- ]chave|palabras clave|kata kunci)\s*[:：])",
            chunks[-1],
        ):
            return []
    if sources and not chunks:
        return []
    finish()
    return variants


def collect_metadata_variants(
    contents: PaperContents,
    resolution: FrontMatterResolution | None,
) -> list[PrintedMetadataVariant]:
    """Capture safe variants of the selected record, in printed source order.

    ``is_primary`` initially marks the first printed variant of each field; the
    exporter reconciles it with the final scalar. Selecting scalar fields requires
    a complete linked presentation, not two independent first-field choices.
    Unknown or incomplete ownership returns no variant rather than filling gaps
    from a model or concatenating unsupported section text.
    """
    if resolution is None or resolution.selected_block_id is None:
        return []
    block = next(
        (block for block in resolution.blocks if block.block_id == resolution.selected_block_id),
        None,
    )
    if block is None:
        return []
    members = frozenset(block.candidate_ids)
    selected = tuple(
        sorted(
            (candidate for candidate in resolution.candidates if candidate.candidate_id in members),
            key=lambda candidate: candidate.reading_order,
        )
    )
    candidates = []
    title_indices = _record_title_indices(selected, allow_byline_only=True)
    all_titles = [
        (index, candidate) for index, candidate in enumerate(selected) if "title" in candidate.roles
    ]
    safe_titles = [
        (index, candidate)
        for index, candidate in all_titles
        if (
            candidate.roles.isdisjoint(_UNSAFE_ROLES - {"title"} | {"abstract"})
            and candidate.raw_text.strip()
        )
    ]
    # A title without its own anatomy may be a line fragment or subtitle.
    # Capturing only the following fragment would let a caller replace a
    # complete scalar with that incomplete fragment as the first variant.
    if len(safe_titles) == len(all_titles) and all(
        index in title_indices for index, _ in safe_titles
    ):
        for _, candidate in safe_titles:
            candidates.append(
                (
                    candidate.reading_order,
                    _variant(block.block_id, "title", candidate.raw_text, [candidate]),
                )
            )
    candidates.extend(
        _abstract_variants(contents, selected, resolution)
        or _region_abstract_variants(contents, selected, resolution)
    )
    from bibr.extract.printed_presentations import link_printed_presentations

    candidates = link_printed_presentations(candidates, selected, block.block_id)
    deduplicated: list[PrintedMetadataVariant] = []
    positions: dict[tuple[str, str], int] = {}
    for _, variant in sorted(candidates, key=lambda item: item[0]):
        normalized = _normalized(variant.text)
        key = variant.field, normalized.casefold() if variant.field == "title" else normalized
        if key in positions:
            position = positions[key]
            previous = deduplicated[position]
            deduplicated[position] = replace(
                previous,
                source_text_ids=_ordered_union(previous.source_text_ids, variant.source_text_ids),
                source_section_ids=_ordered_union(
                    previous.source_section_ids, variant.source_section_ids
                ),
                pages=tuple(sorted(set(previous.pages + variant.pages))),
                presentation_ids=tuple(
                    dict.fromkeys((*previous.presentation_ids, *variant.presentation_ids))
                ),
                byline_source_text_ids=_ordered_union(
                    previous.byline_source_text_ids, variant.byline_source_text_ids
                ),
                byline_source_section_ids=_ordered_union(
                    previous.byline_source_section_ids, variant.byline_source_section_ids
                ),
            )
        else:
            positions[key] = len(deduplicated)
            deduplicated.append(variant)
    counts = {"title": 0, "abstract": 0}
    result = []
    for variant in deduplicated:
        counts[variant.field] += 1
        result.append(
            replace(
                variant,
                variant_id=f"{block.block_id}-{variant.field}-{counts[variant.field]}",
                is_primary=counts[variant.field] == 1,
            )
        )
    return result


__all__ = ["PrintedMetadataVariant", "collect_metadata_variants"]
