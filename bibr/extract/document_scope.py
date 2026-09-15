"""Conservative source intervals for independently detected article records.

This is a bounded partitioner, not a general PDF article segmentation model.
Ambiguous source order or object ownership produces an unresolved record; it
never falls back to handing the complete document to a multi-record extractor.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace

from bibr.extract.front_matter import (
    FrontMatterBlock,
    FrontMatterCandidate,
    FrontMatterResolution,
    _is_toc_listing,
    _record_title_indices,
)
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperFigure,
    PaperSection,
    PaperSentence,
    PaperTable,
)


@dataclass(frozen=True)
class DocumentRecordScope:
    record_id: str
    contents: PaperContents | None
    source_text_ids: tuple[int, ...]
    source_section_ids: tuple[int, ...]
    pages: tuple[int, ...]
    reason_flags: tuple[str, ...]


_AUXILIARY = {CanonicalSection.TABLE, CanonicalSection.FIGURE, CanonicalSection.FOOTNOTE}
_SourceKey = tuple[int, int, int]


def _pages(
    sentences: Iterable[PaperSentence],
    sections: Iterable[PaperSection] = (),
    objects: Iterable[PaperTable | PaperFigure] = (),
) -> tuple[int, ...]:
    pages: set[int] = set()
    for row in sentences:
        if row.page_number is not None:
            pages.add(row.page_number)
        pages.update(p.page_no for p in row.provenance)
    pages.update(p.page_no for section in sections for p in section.provenance)
    for obj in objects:
        if obj.page_number is not None:
            pages.add(obj.page_number)
        pages.update(p.page_no for p in obj.provenance)
        pages.update(part.page_number for part in obj.parts if part.page_number is not None)
        pages.update(p.page_no for part in obj.parts for p in part.provenance)
    return tuple(sorted(pages))


def _resolution_for(
    resolution: FrontMatterResolution,
    block: FrontMatterBlock,
    text_ids: set[int],
    section_ids: set[int],
) -> FrontMatterResolution:
    members = set(block.candidate_ids)
    candidates = tuple(
        row
        for row in resolution.candidates
        if row.candidate_id in members
        and row.section_id in section_ids
        and set(row.text_ids).issubset(text_ids)
    )
    present = {row.candidate_id for row in candidates}
    block = replace(
        block,
        candidate_ids=tuple(key for key in block.candidate_ids if key in present),
        title_candidate_ids=tuple(key for key in block.title_candidate_ids if key in present),
    )
    return FrontMatterResolution(
        candidates=candidates,
        blocks=(block,),
        selected_block_id=block.block_id,
        selection_method="document_record_scope",
        reason_flags=("document_record_scope",),
        allowed_text_ids=frozenset(value for row in candidates for value in row.text_ids),
        allowed_section_ids=frozenset(
            row.section_id for row in candidates if row.section_id is not None
        ),
    )


def _reset_derived(contents: PaperContents) -> None:
    contents.front_role_predictions = None
    contents.metadata_variants = []
    contents.citation_receipt = None
    contents.caption_assignment_receipt = None
    contents.reference_yield_receipt = None
    contents.reference_boundary_reason_flags = []
    contents.structure_validation_issues = []
    contents.processing_warnings = []
    contents.xrefs = [xref for xref in contents.xrefs if xref.xref_type != "bib"]
    for section in contents.sections:
        section.children = []
    contents.invalidate_text_caches()


def scope_document_records(
    contents: PaperContents, resolution: FrontMatterResolution
) -> tuple[DocumentRecordScope, ...]:
    """Return independent article contents or an explicit unresolved scope.

    Exact OCR region order establishes source intervals when available; section
    and paragraph order provide bounded fallbacks. Exact paragraph anchors
    allow multiple records within one section/on one page.
    Appended media/footnote sections are assigned through their source links,
    never by their appended position. All IDs and source pages are preserved.
    """
    blocks = resolution.blocks

    def unresolved(reason: str) -> tuple[DocumentRecordScope, ...]:
        return tuple(
            DocumentRecordScope(block.block_id, None, (), (), block.pages, (reason,))
            for block in blocks
        )

    if not blocks:
        return ()
    if "toc_listing" in resolution.reason_flags or _is_toc_listing(resolution.candidates):
        return unresolved("toc_listing")
    if len({row.text_id for row in contents.sentences}) != len(contents.sentences):
        return unresolved("duplicate_source_text_ids")
    if len({section.section_id for section in contents.sections}) != len(contents.sections):
        return unresolved("duplicate_source_section_ids")
    for kind, objects in (("table", contents.tables), ("figure", contents.figures)):
        if len({getattr(obj, f"{kind}_id") for obj in objects}) != len(objects):
            return unresolved("duplicate_source_object_ids")

    by_id = {row.candidate_id: row for row in resolution.candidates}
    if len(by_id) != len(resolution.candidates) or any(
        key not in by_id for block in blocks for key in block.candidate_ids
    ):
        return unresolved("invalid_candidate_membership")
    all_members = [key for block in blocks for key in block.candidate_ids]
    if len(set(all_members)) != len(all_members):
        return unresolved("shared_record_candidates")

    if len(blocks) == 1:
        block = blocks[0]
        if resolution.selected_block_id != block.block_id:
            return unresolved("unresolved_record_identity")
        if not any(
            key in by_id and "title" in by_id[key].roles for key in block.title_candidate_ids
        ):
            return unresolved("missing_record_title")
        scoped = deepcopy(contents)
        _reset_derived(scoped)
        single_text_ids = tuple(row.text_id for row in scoped.sentences)
        single_section_ids = tuple(section.section_id for section in scoped.sections)
        scoped.front_matter_resolution = _resolution_for(
            resolution, block, set(single_text_ids), set(single_section_ids)
        )
        return (
            DocumentRecordScope(
                block.block_id,
                scoped,
                single_text_ids,
                single_section_ids,
                _pages(scoped.sentences, scoped.sections, [*scoped.tables, *scoped.figures]),
                ("single_record_scope",),
            ),
        )

    sections = {section.section_id: section for section in contents.sections}
    section_order = {section.section_id: index for index, section in enumerate(contents.sections)}
    auxiliary_ids = {
        section.section_id for section in contents.sections if section.section_type in _AUXILIARY
    }
    if any(row.section_id not in sections for row in contents.sentences):
        return unresolved("missing_source_section")
    coordinates: dict[int, _SourceKey] = {
        sentence.text_id: (section_order[sentence.section_id], position, 0)
        for position, sentence in enumerate(contents.sentences)
        if sentence.section_id not in auxiliary_ids
    }
    ends = dict(coordinates)
    heading_coordinates: dict[int, _SourceKey] = {
        section_id: (position, -1, -1)
        for section_id, position in section_order.items()
        if section_id not in auxiliary_ids
    }
    region_indices: defaultdict[tuple[int, tuple[float, float, float, float]], set[int]] = (
        defaultdict(set)
    )
    for region in contents.region_summaries:
        if region.bbox is not None:
            region_indices[region.page, region.bbox].add(region.index)

    def source_keys(provenance) -> list[tuple[int, int]] | None:
        if not provenance:
            return None
        result = []
        for item in provenance:
            matches = region_indices.get((item.page_no, item.bbox), set())
            if len(matches) != 1:
                return None
            result.append((item.page_no, next(iter(matches))))
        return result

    physical_coordinates: dict[int, _SourceKey] = {}
    physical_ends: dict[int, _SourceKey] = {}
    for position, sentence in enumerate(contents.sentences):
        if sentence.section_id in auxiliary_ids:
            continue
        keys = source_keys(sentence.provenance)
        if keys is None:
            continue
        if keys != sorted(keys) or (
            sentence.page_number is not None
            and sentence.page_number not in {page for page, _index in keys}
        ):
            return unresolved("conflicting_source_provenance")
        physical_coordinates[sentence.text_id] = (*keys[0], position)
        physical_ends[sentence.text_id] = (*keys[-1], position)
    use_regions = bool(coordinates) and len(physical_coordinates) == len(coordinates)
    if use_regions:
        coordinates, ends = physical_coordinates, physical_ends
        heading_coordinates = {}
        for section in contents.sections:
            keys = source_keys(section.provenance)
            if section.section_id not in auxiliary_ids and keys:
                if keys != sorted(keys):
                    return unresolved("conflicting_heading_provenance")
                heading_coordinates[section.section_id] = (*keys[0], -1)
    source_positions = list(coordinates.values())
    use_paragraphs = False
    if not use_regions and source_positions != sorted(source_positions):
        body_sentences = [
            sentence for sentence in contents.sentences if sentence.section_id not in auxiliary_ids
        ]
        backward = [
            right
            for left, right in zip(body_sentences, body_sentences[1:], strict=False)
            if section_order[right.section_id] < section_order[left.section_id]
        ]
        # Some parsers reuse one abstract container after each later title.
        # Exact paragraph order plus each title's own section-local byline can
        # still locate the boundary. Arbitrary body-section returns do not.
        if backward and all(
            sections[row.section_id].section_type == CanonicalSection.ABSTRACT for row in backward
        ):
            use_paragraphs = True
            coordinates = {
                sentence.text_id: (position, 0, 0)
                for position, sentence in enumerate(contents.sentences)
                if sentence.section_id not in auxiliary_ids
            }
            ends = dict(coordinates)
            heading_coordinates = {}
            for position, sentence in enumerate(contents.sentences):
                if sentence.section_id not in auxiliary_ids:
                    heading_coordinates.setdefault(sentence.section_id, (position, -1, 0))
            source_positions = list(coordinates.values())
    if source_positions != sorted(source_positions):
        return unresolved("nonmonotonic_source_order")
    source_pages = [
        row.page_number
        for row in contents.sentences
        if row.section_id not in auxiliary_ids and row.page_number is not None
    ]
    if source_pages != sorted(source_pages):
        return unresolved("nonmonotonic_source_pages")
    paragraph_rows: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    sentences_by_id = {row.text_id: row for row in contents.sentences}
    for row in contents.sentences:
        paragraph_rows[row.section_id, row.paragraph_id].add(row.text_id)

    def candidate_position(row: FrontMatterCandidate, *, anchor: bool = False) -> _SourceKey | None:
        if row.section_id not in section_order or row.section_id in auxiliary_ids:
            return None
        if row.source_kind == "heading":
            if _normalize(row.raw_text) != _normalize(sections[row.section_id].header):
                return None
            position = heading_coordinates.get(row.section_id)
            if (
                use_regions
                and row.page is not None
                and position is not None
                and row.page != position[0]
            ):
                return None
            if (
                row.page is not None
                and sections[row.section_id].provenance
                and (row.page != sections[row.section_id].provenance[0].page_no)
            ):
                return None
            return position
        if row.source_kind != "paragraph" or not row.text_ids:
            return None
        if any(text_id not in coordinates for text_id in row.text_ids):
            return None
        if _normalize(row.raw_text) != _normalize(
            " ".join(
                sentence.text for sentence in contents.sentences if sentence.text_id in row.text_ids
            )
        ) or any(sentences_by_id[value].section_id != row.section_id for value in row.text_ids):
            return None
        if anchor and (
            row.paragraph_id is None
            or set(row.text_ids) != paragraph_rows[row.section_id, row.paragraph_id]
        ):
            return None
        if (
            use_regions
            and row.page is not None
            and any(
                coordinates[value][0] != row.page or ends[value][0] != row.page
                for value in row.text_ids
            )
        ):
            return None
        return min(coordinates[text_id] for text_id in row.text_ids)

    developed = _record_title_indices(resolution.candidates, allow_byline_only=True)
    developed_ids = {resolution.candidates[index].candidate_id for index in developed}
    anchors: list[_SourceKey] = []
    anchor_titles: list[str] = []
    for block in blocks:
        titles = [
            by_id[key]
            for key in block.title_candidate_ids
            if key in by_id and key in developed_ids and "body_heading" not in by_id[key].roles
        ]
        if not titles:
            return unresolved("undeveloped_record_anchor")
        title = min(titles, key=lambda row: row.reading_order)
        anchor_position = candidate_position(title, anchor=True)
        if anchor_position is None:
            return unresolved("ambiguous_record_anchor")
        anchors.append(anchor_position)
        anchor_titles.append(title.raw_text)
    if anchors != sorted(set(anchors)):
        return unresolved("interleaved_record_anchors")

    def owner(position: _SourceKey | None) -> int | None:
        if position is None:
            return None
        index = bisect_right(anchors, position) - 1
        return index if index >= 0 else None

    flags: list[set[str]] = [set() for _ in blocks]
    candidate_positions = [candidate_position(row) for row in resolution.candidates]
    ordered = [position for position in candidate_positions if position is not None]
    if ordered != sorted(ordered):
        return unresolved("nonmonotonic_candidate_order")
    # Incomplete provenance cannot erase contradictory physical evidence. Use
    # every exact source match as a veto even when paragraph/section order had
    # to establish the full coordinate system. Both paragraph endpoints matter:
    # one merged paragraph may begin before, and finish after, another title.
    paragraph_footprints: defaultdict[
        tuple[int, int], set[tuple[tuple[int, tuple[float, float, float, float] | None], ...]]
    ] = defaultdict(set)
    for sentence in contents.sentences:
        paragraph_footprints[sentence.section_id, sentence.paragraph_id].add(
            tuple((item.page_no, item.bbox) for item in sentence.provenance)
        )
    known_physical: list[tuple[_SourceKey, tuple[int, int]]] = []
    previous_paragraph: tuple[int, int] | None = None
    for sentence in contents.sentences:
        if sentence.text_id not in coordinates:
            previous_paragraph = None
            continue
        paragraph = (sentence.section_id, sentence.paragraph_id)
        # Source provenance is paragraph-wide, whereas page_number is the
        # sentence's own page. Inspect an identical shared interval once per
        # contiguous paragraph run, never once globally per region. A later
        # revisit by another paragraph must still trigger the physical veto.
        repeated_footprint = (
            paragraph == previous_paragraph
            and len(paragraph_footprints[paragraph]) == 1
            and bool(sentence.provenance)
        )
        previous_paragraph = paragraph
        if repeated_footprint:
            continue
        for provenance in sentence.provenance:
            if provenance.bbox is None:
                continue
            exact_indices = region_indices.get((provenance.page_no, provenance.bbox), set())
            if len(exact_indices) == 1:
                known_physical.append(
                    (coordinates[sentence.text_id], (provenance.page_no, next(iter(exact_indices))))
                )
    for section in contents.sections:
        if section.section_id not in heading_coordinates:
            continue
        for provenance in section.provenance:
            if provenance.bbox is None:
                continue
            exact_indices = region_indices.get((provenance.page_no, provenance.bbox), set())
            if len(exact_indices) == 1:
                known_physical.append(
                    (
                        heading_coordinates[section.section_id],
                        (provenance.page_no, next(iter(exact_indices))),
                    )
                )
    physical_order = [point for _, point in sorted(known_physical, key=lambda pair: pair[0])]
    if physical_order != sorted(physical_order):
        return unresolved("conflicting_source_provenance")
    for index, block in enumerate(blocks):
        for key in block.candidate_ids:
            candidate = by_id[key]
            evidence_position = candidate_position(candidate)
            if evidence_position is None:
                if (
                    candidate.source_kind == "heading"
                    and candidate.roles.isdisjoint({"title", "byline", "doi", "affiliation"})
                    and candidate.section_id not in heading_coordinates
                ):
                    flags[index].add("unlocated_container_heading_omitted")
                    continue
                return unresolved("unlocated_record_evidence")
            if owner(evidence_position) != index:
                # An ordinary container heading may precede the first article
                # paragraph. A byline/title/DOI before an anchor is ambiguous.
                if owner(evidence_position) is None and candidate.roles.isdisjoint(
                    {"title", "byline", "doi", "affiliation"}
                ):
                    flags[index].add("leading_document_content_omitted")
                    continue
                else:
                    return unresolved("record_evidence_crosses_boundary")
            if any(owner(coordinates[value]) != index for value in candidate.text_ids):
                return unresolved("record_paragraph_crosses_boundary")

    if any(owner(coordinates[text_id]) != owner(ends[text_id]) for text_id in coordinates):
        return unresolved("source_paragraph_crosses_record_boundary")

    text_owners = {text_id: owner(position) for text_id, position in coordinates.items()}
    section_owners: defaultdict[int | None, set[int]] = defaultdict(set)
    for section_id, heading_position in heading_coordinates.items():
        heading_owner = owner(heading_position)
        if heading_owner is not None:
            section_owners[section_id].add(heading_owner)
    for row in contents.sentences:
        sentence_owner = text_owners.get(row.text_id)
        if sentence_owner is not None:
            section_owners[row.section_id].add(sentence_owner)

    page_owners: defaultdict[int, set[int]] = defaultdict(set)
    for row in contents.sentences:
        sentence_owner = text_owners.get(row.text_id)
        if sentence_owner is not None:
            if row.page_number is not None:
                page_owners[row.page_number].add(sentence_owner)
            for provenance in row.provenance:
                page_owners[provenance.page_no].add(sentence_owner)
    for section in contents.sections:
        if section.section_id not in auxiliary_ids:
            for provenance in section.provenance:
                page_owners[provenance.page_no].update(section_owners[section.section_id])

    object_owners: dict[tuple[str, int], int] = {}
    invalid: list[set[str]] = [set() for _ in blocks]
    for index, block in enumerate(blocks):
        if not block.merge_reasons and any(
            key not in developed_ids for key in block.title_candidate_ids
        ):
            invalid[index].add("unresolved_internal_title_boundaries")

    def reject(possible: set[int], reason: str) -> None:
        for index in possible or range(len(blocks)):
            invalid[index].add(reason)

    for kind, objects in (("table", contents.tables), ("figure", contents.figures)):
        for obj in objects:
            source_section = (
                obj._body_section_id if obj._body_section_id is not None else obj.section_id
            )
            possible = section_owners[source_section]
            if len(possible) != 1:
                reject(possible, "ambiguous_object_ownership")
                continue
            index = next(iter(possible))
            declared_owners = section_owners[obj.section_id]
            if obj.section_id not in auxiliary_ids and declared_owners != {index}:
                reject(possible | declared_owners, "conflicting_object_section_ownership")
                continue
            physical = [
                (part.page_number, part.bbox) for part in obj.parts if part.page_number is not None
            ] + [(p.page_no, p.bbox) for p in obj.provenance]
            physical.extend((p.page_no, p.bbox) for part in obj.parts for p in part.provenance)
            if obj.page_number is not None:
                physical.append((obj.page_number, None))
            crossed = False
            for page, bbox in physical:
                owners = page_owners[page]
                exact_owners = set()
                if bbox is not None:
                    for region in contents.region_summaries:
                        if region.page == page and region.bbox == bbox:
                            exact_owners.update(section_owners[region.section_id])
                    for sentence in contents.sentences:
                        if (
                            any(p.page_no == page and p.bbox == bbox for p in sentence.provenance)
                            and (source_owner := text_owners.get(sentence.text_id)) is not None
                        ):
                            exact_owners.add(source_owner)
                if exact_owners and exact_owners != {index}:
                    crossed = True
                if owners and index not in owners:
                    crossed = True
                elif len(owners) > 1 and len(obj.parts) > 1:
                    # Multipart stitching predates record scopes. A shared page
                    # cannot prove every part belongs to the first part's body.
                    matches = [
                        region
                        for region in contents.region_summaries
                        if bbox is not None and region.page == page and region.bbox == bbox
                    ]
                    if not matches or any(
                        section_owners[row.section_id] != {index} for row in matches
                    ):
                        crossed = True
            if crossed:
                reject(possible, "object_parts_cross_record_boundary")
                continue
            object_owners[kind, getattr(obj, f"{kind}_id")] = index
            section_owners[obj.section_id].add(index)
            for row in contents.sentences:
                if row.section_id == obj.section_id and row.section_id in auxiliary_ids:
                    text_owners[row.text_id] = index

    foot_owners = {}
    for section in contents.sections:
        if section.section_type != CanonicalSection.FOOTNOTE:
            continue
        match = re.fullmatch(r"Footnote (\d+)", section.header)
        if match is None:
            reject(set(), "ambiguous_footnote_ownership")
            continue
        possible = {
            foot_owner
            for xref in contents.xrefs
            if match and xref.xref_type == "foot" and xref.xref_id == int(match[1])
            for foot_owner in [text_owners.get(xref.text_id)]
            if foot_owner is not None
        }
        if len(possible) != 1:
            reject(possible, "ambiguous_footnote_ownership")
            continue
        index = next(iter(possible))
        foot_owners[int(match[1])] = index
        section_owners[section.section_id].add(index)
        for row in contents.sentences:
            if row.section_id == section.section_id:
                text_owners[row.text_id] = index

    for section_id in auxiliary_ids:
        if section_owners[section_id]:
            continue
        rows = [row for row in contents.sentences if row.section_id == section_id]
        if rows:
            # Appended caption/content rows without their linked source object
            # cannot be silently discarded from apparently complete records.
            reject(set(), "unowned_auxiliary_content")

    results = []
    for index, block in enumerate(blocks):
        sentences = [row for row in contents.sentences if text_owners.get(row.text_id) == index]
        text_ids = {row.text_id for row in sentences}
        section_ids = {
            section_id
            for section_id, owners in section_owners.items()
            if section_id is not None and index in owners
        }
        tables = [
            obj for obj in contents.tables if object_owners.get(("table", obj.table_id)) == index
        ]
        figures = [
            obj for obj in contents.figures if object_owners.get(("figure", obj.figure_id)) == index
        ]
        scoped_sections = deepcopy(
            [section for section in contents.sections if section.section_id in section_ids]
        )
        for section in scoped_sections:
            if section.parent_section_id not in section_ids:
                section.parent_section_id = None
            if len(section_owners[section.section_id]) > 1 or (
                section.section_id not in auxiliary_ids
                and owner(heading_coordinates.get(section.section_id)) != index
            ):
                section.header = ""
                section.header_is_synthetic = True
                section.classification_source = "document_scope_shared_container"
                section.provenance = []
        xrefs = []
        equation_groups = {eq.grp_id for eq in contents.equations if eq.text_id in text_ids}
        for xref in contents.xrefs:
            if xref.text_id not in text_ids or xref.xref_type == "bib":
                continue
            target_owner = (
                object_owners.get((xref.xref_type, xref.xref_id))
                if xref.xref_type in {"table", "figure"}
                else foot_owners.get(xref.xref_id)
                if xref.xref_type == "foot"
                else None
            )
            valid = (
                xref.xref_id == 0
                or target_owner == index
                or (xref.xref_type == "section" and xref.xref_id in section_ids)
                or (xref.xref_type == "equation" and xref.xref_id in equation_groups)
            )
            if valid:
                xrefs.append(xref)
            else:
                invalid[index].add("unresolved_or_cross_record_link")
        scoped_pages = _pages(sentences, scoped_sections, [*tables, *figures])
        if invalid[index]:
            results.append(
                DocumentRecordScope(
                    block.block_id,
                    None,
                    tuple(row.text_id for row in sentences),
                    tuple(section.section_id for section in scoped_sections),
                    scoped_pages,
                    tuple(sorted(invalid[index] | flags[index])),
                )
            )
            continue
        region_keys = {
            (p.page_no, p.bbox) for row in sentences for p in row.provenance if p.bbox is not None
        }
        regions = [
            region
            for region in contents.region_summaries
            if (not page_owners[region.page] or index in page_owners[region.page])
            and (
                section_owners[region.section_id] == {index}
                or (region.page, region.bbox) in region_keys
            )
        ]
        scoped = PaperContents(
            sentences=deepcopy(sentences),
            sections=scoped_sections,
            tables=deepcopy(tables),
            figures=deepcopy(figures),
            links=deepcopy([link for link in contents.links if link.text_id in text_ids]),
            xrefs=deepcopy(xrefs),
            equations=deepcopy([eq for eq in contents.equations if eq.text_id in text_ids]),
            sections_text={
                section_id: " ".join(row.text for row in sentences if row.section_id == section_id)
                for section_id in section_ids
            },
            detected_title=anchor_titles[index],
            region_summaries=deepcopy(regions),
        )
        scoped.front_matter_resolution = _resolution_for(resolution, block, text_ids, section_ids)
        _reset_derived(scoped)
        results.append(
            DocumentRecordScope(
                block.block_id,
                scoped,
                tuple(row.text_id for row in sentences),
                tuple(section.section_id for section in scoped_sections),
                scoped_pages,
                (
                    "source_region_interval_scope"
                    if use_regions
                    else "source_paragraph_interval_scope"
                    if use_paragraphs
                    else "source_interval_scope",
                    *sorted(flags[index]),
                ),
            )
        )
    return tuple(results)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())
