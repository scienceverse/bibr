"""Document-only recovery of article anatomy hidden in composite OCR rows."""

from __future__ import annotations

import re
from dataclasses import replace

from bibr.extract.front_matter import (
    _NAME_PARTICLES,
    _NON_NAME_CHUNK_WORDS,
    _WORD_RE,
    AFFILIATION_MARKER_RE,
    BODY_HEADING_ROLE,
    BYLINE_PROBATION_ROLE,
    MODEL_NON_TITLE_SEED_ROLE,
    FrontMatterBlock,
    FrontMatterCandidate,
    FrontMatterResolution,
    _context_geometry_is_compatible,
    _has_prohibited_separator_evidence,
    _is_toc_listing,
    _make_block,
    _normalize_text,
    _record_identity_evidence,
    group_front_matter_blocks,
)
from bibr.paper_contents import PaperContents, RegionSummary

_SEPARATORS = re.compile(r"[,;·•]|\band\b|&", re.IGNORECASE)
_MARKERS = re.compile(r"[\d*†‡§¶#]+")
_CONTEXT_ROLE = "document_contextual_byline"


def _person_list_identity(text: str) -> str:
    return "|".join(
        _normalize_text(" ".join(words))
        for chunk in _SEPARATORS.split(text)
        if (words := _WORD_RE.findall(_MARKERS.sub(" ", chunk)))
    )


def _name_prefix(text: str, title: str) -> str | None:
    """Require bounded person-shaped chunks before the first affiliation chunk.

    This establishes local boundary anatomy, not parsed author identities. The
    original complete row is retained, including any glued abstract text.
    """
    chunks = [" ".join(chunk.split()) for chunk in _SEPARATORS.split(text) if chunk.strip()]
    affiliation = next(
        (index for index, chunk in enumerate(chunks) if AFFILIATION_MARKER_RE.search(chunk)),
        None,
    )
    if affiliation is None or affiliation == 0:
        return None
    names = [chunk for chunk in chunks[:affiliation] if _WORD_RE.search(chunk)]
    if not names or len(names) > 16 or sum(map(len, names)) > 600:
        return None
    prefix = " ".join(names)
    if _has_prohibited_separator_evidence(prefix, prefix.casefold(), include_affiliation=False):
        return None
    # OCR may repeat the final title word(s) at the start of a composite byline.
    # Remove only an exact uppercase suffix of this same immediately preceding
    # title, and only for the name-shape check; never alter the source text.
    title_words = title.split()
    for count in range(min(12, len(title_words)), 0, -1):
        suffix = " ".join(title_words[-count:])
        if suffix.isupper() and names[0].startswith(suffix + " "):
            names[0] = names[0][len(suffix) :].strip()
            break
    for name in names:
        words = _WORD_RE.findall(_MARKERS.sub(" ", name))
        if not 2 <= len(words) <= 6:
            return None
        if any(word.casefold() in _NON_NAME_CHUNK_WORDS for word in words):
            return None
        if not all(word[0].isupper() or word.casefold() in _NAME_PARTICLES for word in words):
            return None
        if not any(not word.isupper() and word[0].isupper() for word in words):
            return None
    return _person_list_identity(", ".join(names))


def _running_prose(text: str) -> bool:
    words = _WORD_RE.findall(text)
    return (
        len(words) >= 45
        and sum(word[0].islower() for word in words) >= 0.55 * len(words)
        and bool(re.search(r"[.!?。]", text))
    )


def _local_byline(title: FrontMatterCandidate, row: FrontMatterCandidate) -> bool:
    return (
        "affiliation" in row.roles
        and row.roles.isdisjoint(
            {"title", "abstract", "doi", BODY_HEADING_ROLE, BYLINE_PROBATION_ROLE}
        )
        and title.page is not None
        and title.page == row.page
        and title.bbox is not None
        and row.bbox is not None
        and row.bbox[1] >= title.bbox[1]
        and _context_geometry_is_compatible(title, row)
        and bool(_name_prefix(row.raw_text, title.raw_text))
    )


def _coalesce_contextual_presentations(
    candidates: tuple[FrontMatterCandidate, ...],
    blocks: tuple[FrontMatterBlock, ...],
    geometry: dict[str, FrontMatterCandidate],
) -> tuple[FrontMatterBlock, ...]:
    """Keep recovered copies/translations under one corroborated article identity."""
    by_id = {row.candidate_id: row for row in candidates}

    def identity(block):
        rows = tuple(by_id[key] for key in block.candidate_ids)
        titles, _bylines, dois = _record_identity_evidence(rows)
        bylines = frozenset(
            _person_list_identity(row.raw_text)
            for row in rows
            if "byline" in row.roles
            and row.roles.isdisjoint({"title", "abstract", "affiliation", BYLINE_PROBATION_ROLE})
            and row.raw_text.strip()
        )
        for index, row in enumerate(rows[:-1]):
            if "title" not in row.roles or not _local_byline(
                geometry[row.candidate_id], geometry[rows[index + 1].candidate_id]
            ):
                continue
            prefix = _name_prefix(rows[index + 1].raw_text, row.raw_text)
            if prefix is not None:
                bylines = bylines | {prefix}
                titles = titles | {row.normalized_text}
        return titles, bylines, dois

    merged: list[FrontMatterBlock] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        prior = merged[-1]
        left_titles, left_names, left_dois = identity(prior)
        right_titles, right_names, right_dois = identity(block)
        adjacent = (
            bool(prior.pages and block.pages) and 0 <= min(block.pages) - max(prior.pages) <= 1
        )
        if not (
            adjacent
            and left_names
            and left_names == right_names
            and len(left_dois | right_dois) <= 1
            and (left_titles & right_titles or (left_dois and left_dois == right_dois))
        ):
            merged.append(block)
            continue
        combined = _make_block(
            1, [by_id[key] for key in (*prior.candidate_ids, *block.candidate_ids)]
        )
        merged[-1] = replace(
            combined,
            block_id=prior.block_id,
            source_block_ids=(prior.source_block_ids or (prior.block_id,))
            + (block.source_block_ids or (block.block_id,)),
            merge_reasons=tuple(
                dict.fromkeys(
                    [
                        *prior.merge_reasons,
                        *block.merge_reasons,
                        "shared_document_contextual_identity",
                    ]
                )
            ),
        )
    return tuple(merged)


def _source_anchors(
    resolution: FrontMatterResolution, contents: PaperContents | None
) -> dict[str, FrontMatterCandidate]:
    """Use a proven first OCR region, not the union of a column-spanning row.

    The returned copies are only for geometric checks. Original source spans,
    candidate text and bounding boxes remain unchanged in the resolution.
    """
    geometry = {row.candidate_id: row for row in resolution.candidates}
    if contents is None:
        return geometry
    regions: dict[tuple[int, tuple[float, float, float, float] | None], list[RegionSummary]] = {}
    for region in contents.region_summaries:
        regions.setdefault((region.page, region.bbox), []).append(region)
    sentences = {row.text_id: row for row in contents.sentences}
    sections = {row.section_id: row for row in contents.sections}
    for row in resolution.candidates:
        provenance = []
        if row.source_kind == "heading" and row.section_id in sections:
            provenance = sections[row.section_id].provenance
        elif row.source_kind == "paragraph" and row.text_ids:
            if any(key not in sentences or not sentences[key].provenance for key in row.text_ids):
                continue
            members = [sentences[key] for key in row.text_ids]
            # Segmentation copies the complete paragraph footprint to every
            # sibling sentence. Repeating that same interval is not a return
            # to an earlier source region. Only coalesce proven siblings;
            # distinct or contradictory footprints retain the order checks.
            if len({(item.section_id, item.paragraph_id) for item in members}) == 1 and all(
                item.provenance == members[0].provenance for item in members
            ):
                provenance = members[0].provenance
            else:
                provenance = [item for member in members for item in member.provenance]
        if not provenance:
            continue
        matches = [regions.get((item.page_no, item.bbox), []) for item in provenance]
        if any(len(match) != 1 for match in matches):
            continue
        order = [(match[0].page, match[0].index) for match in matches]
        first = matches[0][0]
        if order == sorted(order) and row.page == first.page and first.bbox is not None:
            geometry[row.candidate_id] = replace(row, bbox=first.bbox)
    return geometry


def refine_document_records(
    resolution: FrontMatterResolution, *, contents: PaperContents | None = None
) -> FrontMatterResolution:
    """Recover local composite bylines before document-wide record partitioning.

    Only a title heading followed immediately by a same-column name/affiliation
    row and substantive local prose can supply missing anatomy. This opt-in
    refinement does not change single-paper targeting or classify any prose as
    an abstract. Full source ownership still has to pass the separate partitioner.
    """
    candidates = list(resolution.candidates)
    geometry = _source_anchors(resolution, contents)
    if _is_toc_listing(resolution.candidates) or "toc_listing" in resolution.reason_flags:
        return resolution
    title_indices = [index for index, row in enumerate(candidates) if "title" in row.roles]
    if len(title_indices) < 2:
        return resolution
    promoted = False
    demoted_title = False
    overrode_root_veto = False
    for offset, index in enumerate(title_indices):
        end = title_indices[offset + 1] if offset + 1 < len(title_indices) else len(candidates)
        title = candidates[index]
        if (
            "title" not in title.roles
            or title.source_kind != "heading"
            or title.roles & {"affiliation", "abstract", BODY_HEADING_ROLE, BYLINE_PROBATION_ROLE}
            or index + 1 >= len(candidates)
        ):
            continue
        byline = candidates[index + 1]
        title_geometry = geometry[title.candidate_id]
        byline_geometry = geometry[byline.candidate_id]
        # A section classifier can label a name/affiliation heading as TITLE.
        # Require the same local byline/prose proof before removing that role;
        # otherwise it remains an independent possible title boundary.
        classified_byline_title = (
            end == index + 1
            and byline.source_kind == "heading"
            and {"title", "affiliation"}.issubset(byline.roles)
        )
        if classified_byline_title:
            byline = replace(
                byline, roles=byline.roles - {"title"}, model_roles=byline.model_roles - {"title"}
            )
            byline_geometry = replace(byline_geometry, roles=byline.roles)
            end = title_indices[offset + 2] if offset + 2 < len(title_indices) else len(candidates)
        elif index + 1 >= end:
            continue
        if "byline" in byline.roles or not _local_byline(title_geometry, byline_geometry):
            continue
        assert byline_geometry.bbox is not None  # Established by the local anatomy guard.
        local_prose = candidates[index + 1 : end]
        if not any(
            row.source_kind == "paragraph"
            and row.roles.isdisjoint({"title", "doi", BODY_HEADING_ROLE})
            and row.page == title.page
            and (row_geometry := geometry[row.candidate_id]).bbox is not None
            and row_geometry.bbox[1] >= byline_geometry.bbox[1]
            and _context_geometry_is_compatible(title_geometry, byline_geometry, row_geometry)
            and _running_prose(row.raw_text)
            for row in local_prose
        ):
            continue
        candidates[index + 1] = replace(byline, roles=byline.roles | {"byline", _CONTEXT_ROLE})
        geometry[byline.candidate_id] = replace(byline_geometry, roles=candidates[index + 1].roles)
        # A learned heading prior cannot veto this document-local record
        # after its own title/byline/affiliation/prose anatomy is proven.
        # Keep the original model scores and every other role veto; this
        # exception never applies to a title borrowing another row's anatomy.
        if MODEL_NON_TITLE_SEED_ROLE in title.roles:
            candidates[index] = replace(title, roles=title.roles - {MODEL_NON_TITLE_SEED_ROLE})
            geometry[title.candidate_id] = replace(title_geometry, roles=candidates[index].roles)
            overrode_root_veto = True
        promoted = True
        demoted_title = demoted_title or classified_byline_title
    if not promoted:
        return resolution
    blocks = _coalesce_contextual_presentations(
        tuple(candidates), group_front_matter_blocks(tuple(candidates)), geometry
    )
    selected = blocks[0] if len(blocks) == 1 else None
    members = set(selected.candidate_ids) if selected else set()
    return replace(
        resolution,
        candidates=tuple(candidates),
        blocks=blocks,
        selected_block_id=selected.block_id if selected else None,
        selection_method="document_contextual_anatomy",
        reason_flags=tuple(
            dict.fromkeys(
                [
                    *resolution.reason_flags,
                    "document_contextual_anatomy",
                    *(["document_byline_title_role_repaired"] if demoted_title else []),
                    *(
                        ["document_local_anatomy_overrode_model_root_veto"]
                        if overrode_root_veto
                        else []
                    ),
                ]
            )
        ),
        allowed_text_ids=frozenset(
            text_id for row in candidates if row.candidate_id in members for text_id in row.text_ids
        ),
        allowed_section_ids=frozenset(
            row.section_id
            for row in candidates
            if row.candidate_id in members and row.section_id is not None
        ),
    )
