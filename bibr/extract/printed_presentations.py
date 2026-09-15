"""Link printed field versions only when their local source anatomy agrees.

The links describe presentations within an already selected article. They never
establish article identity, infer a translation, or merge separate records.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterCandidate
    from bibr.extract.metadata_variants import PrintedMetadataVariant


def link_printed_presentations(
    variants: list[tuple[int, PrintedMetadataVariant]],
    selected: tuple[FrontMatterCandidate, ...],
    record_id: str,
) -> list[tuple[int, PrintedMetadataVariant]]:
    """Pair a title and abstract with one intervening, independently owned byline.

    A title-only cover, several abstracts under one title, out-of-order blocks,
    and incomplete evidence remain unlinked. Repeated field text can acquire
    several presentation IDs when the caller subsequently deduplicates it.
    """
    orders = [row.reading_order for row in selected]
    if len(orders) != len(set(orders)) or orders != sorted(orders):
        return variants
    title_rows = [row for row in selected if "title" in row.roles]
    linked = list(variants)
    for number, title in enumerate(title_rows, 1):
        end = title_rows[number].reading_order if number < len(title_rows) else float("inf")
        local = [row for row in selected if title.reading_order < row.reading_order < end]
        # Count mixed-role rows before deciding whether the inventory is safe;
        # dropping an unsafe byline could make the remaining one look unique.
        bylines = [row for row in local if "byline" in row.roles]
        titles = [
            index
            for index, (order, variant) in enumerate(variants)
            if variant.field == "title" and order == title.reading_order
        ]
        abstracts = [
            index
            for index, (order, variant) in enumerate(variants)
            if variant.field == "abstract" and title.reading_order < order < end
        ]
        if len(titles) != 1 or len(abstracts) != 1 or len(bylines) != 1:
            continue
        byline = bylines[0]
        if not byline.roles.isdisjoint(
            {"title", "abstract", "affiliation", "doi", "byline_probation"}
        ):
            continue
        abstract_order, abstract = variants[abstracts[0]]
        if (
            byline.reading_order >= abstract_order
            or title.page is None
            or byline.page != title.page
            or not abstract.pages
            or min(abstract.pages) < title.page
            or max(abstract.pages) > title.page + 1
            or not byline.raw_text.strip()
            or (not byline.text_ids and byline.section_id is None)
        ):
            continue
        # Every paragraph of the abstract must be inside this local interval.
        # A shared section spanning the following title cannot supply a link.
        local_ids = {key for row in local for key in row.text_ids}
        if not abstract.source_text_ids or not set(abstract.source_text_ids).issubset(local_ids):
            continue
        # Any unsupported abstract/title row within the interval makes pairing
        # ambiguous even if a later supported field could otherwise be linked.
        abstract_sections = set(abstract.source_section_ids)
        if any(
            "abstract" in row.roles and row.section_id not in abstract_sections for row in local
        ):
            continue
        for index in (titles[0], abstracts[0]):
            order, variant = variants[index]
            linked[index] = (
                order,
                replace(
                    variant,
                    presentation_ids=(f"{record_id}-presentation-{number}",),
                    byline_source_text_ids=byline.text_ids,
                    byline_source_section_ids=(byline.section_id,)
                    if byline.section_id is not None
                    else (),
                ),
            )
    return linked
