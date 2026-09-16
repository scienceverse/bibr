"""Select one complete printed presentation using frozen source ownership."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bibr.extract.metadata_variants import has_explicit_original_label

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterCandidate, FrontMatterResolution
    from bibr.extract.metadata_variants import PrintedMetadataVariant


@dataclass(frozen=True)
class PrintedPresentation:
    presentation_id: str
    record_id: str
    title_variant_id: str
    abstract_variant_id: str
    byline_candidate_ids: tuple[str, ...]
    byline_source_text_ids: tuple[int, ...]
    byline_source_section_ids: tuple[int, ...]
    original_marker_ids: tuple[str, ...]


@dataclass(frozen=True)
class PresentationSelection:
    presentations: tuple[PrintedPresentation, ...] = ()
    selected_presentation_id: str | None = None
    reason: str = "no_complete_printed_presentation"


_ORIGINAL = re.compile(
    r"^\s*(?:original(?:-language)?\s+(?:title|abstract)|"
    r"(?:title|abstract)\s+in\s+(?:the\s+)?original\s+language)\s*:",
    re.IGNORECASE,
)


def select_printed_presentation(
    variants: list[PrintedMetadataVariant], resolution: FrontMatterResolution | None
) -> PresentationSelection:
    """An original marker selects the whole presentation, otherwise source order.

    A link is rechecked against its local title/byline/abstract interval. This
    preserves the physical byline when equal title/abstract text was deduplicated
    across several presentations. A model's scalar choice never establishes a link.
    """
    if resolution is None or resolution.selected_block_id is None:
        return PresentationSelection(reason="article_identity_unresolved")
    block = next(
        (row for row in resolution.blocks if row.block_id == resolution.selected_block_id), None
    )
    if block is None:
        return PresentationSelection(reason="article_identity_unresolved")
    rows = sorted(
        (row for row in resolution.candidates if row.candidate_id in block.candidate_ids),
        key=lambda row: row.reading_order,
    )
    if len({row.reading_order for row in rows}) != len(rows):
        return PresentationSelection(reason="source_order_ambiguous")
    titles = [row for row in rows if "title" in row.roles]
    owned = [row for row in variants if row.record_id == block.block_id]
    if len({row.variant_id for row in owned}) != len(owned):
        return PresentationSelection(reason="variant_identity_ambiguous")
    presentations = []
    for number, title in enumerate(titles, 1):
        key = f"{block.block_id}-presentation-{number}"
        members = [row for row in owned if key in row.presentation_ids]
        if len(members) != 2 or {row.field for row in members} != {"title", "abstract"}:
            continue
        title_variant = next(row for row in members if row.field == "title")
        abstract_variant = next(row for row in members if row.field == "abstract")
        end = titles[number].reading_order if number < len(titles) else float("inf")
        local = [row for row in rows if title.reading_order < row.reading_order < end]
        bylines = [row for row in local if "byline" in row.roles]
        if len(bylines) != 1:
            continue
        byline = bylines[0]
        if (
            not byline.roles.isdisjoint(
                {"title", "abstract", "affiliation", "doi", "byline_probation"}
            )
            or not byline.text_ids
            or not set(byline.text_ids).issubset(resolution.allowed_text_ids)
            or not all(set(byline.text_ids).issubset(row.byline_source_text_ids) for row in members)
            or title.section_id not in title_variant.source_section_ids
        ):
            continue
        abstract_rows = [
            row for row in local if set(row.text_ids) & set(abstract_variant.source_text_ids)
        ]
        if (
            not abstract_rows
            or min(row.reading_order for row in abstract_rows) <= byline.reading_order
        ):
            continue
        markers = tuple(
            row.candidate_id
            for row in rows
            if "metadata" in row.roles
            and _ORIGINAL.match(row.raw_text)
            and row.section_id in {title.section_id, *abstract_variant.source_section_ids}
            and row.page == title.page
            and row.reading_order < min(item.reading_order for item in abstract_rows)
        )
        presentations.append(
            PrintedPresentation(
                key,
                block.block_id,
                title_variant.variant_id,
                abstract_variant.variant_id,
                (byline.candidate_id,),
                byline.text_ids,
                (byline.section_id,) if byline.section_id is not None else (),
                markers,
            )
        )
    complete = tuple(presentations)
    originals = [row for row in complete if row.original_marker_ids]
    if len(originals) == 1:
        return PresentationSelection(complete, originals[0].presentation_id, "explicit_original")
    if originals or any(
        has_explicit_original_label(resolution, field) for field in ("title", "abstract")
    ):
        return PresentationSelection(complete, reason="original_presentation_ambiguous")
    if complete:
        return PresentationSelection(
            complete, complete[0].presentation_id, "first_complete_printed"
        )
    return PresentationSelection()


def presentation_author_context(
    presentation: PrintedPresentation, resolution: FrontMatterResolution
) -> str:
    """Use this presentation's byline and immediately following affiliations."""
    block = next(row for row in resolution.blocks if row.block_id == presentation.record_id)
    rows = sorted(
        (row for row in resolution.candidates if row.candidate_id in block.candidate_ids),
        key=lambda row: row.reading_order,
    )
    members = set(presentation.byline_candidate_ids)
    selected: list[FrontMatterCandidate] = []
    active = False
    for row in rows:
        if row.candidate_id in members:
            selected.append(row)
            active = True
        elif active:
            if row.roles & {"title", "abstract", "byline", "body_heading", "structural"}:
                break
            if row.text_ids and not set(row.text_ids).issubset(resolution.allowed_text_ids):
                break
            if row.roles & {"affiliation", "correspondence", "metadata", "doi"}:
                selected.append(row)
            else:
                break
    return "\n".join(row.raw_text for row in selected)
