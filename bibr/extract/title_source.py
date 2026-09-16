"""Bounded source proof for document titles containing institution vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterCandidate
    from bibr.paper_contents import PaperContents, RegionSummary

SOURCE_QUALIFIED_TITLE_ROLE = "source_qualified_title"


@dataclass(frozen=True)
class NativeTitleEvidence:
    candidate_id: str
    title: str
    pre_byline_rows: tuple[str, ...]
    byline_candidate_ids: tuple[str, ...]


def native_title_evidence(
    contents: PaperContents, candidates: tuple[FrontMatterCandidate, ...]
) -> tuple[NativeTitleEvidence, ...]:
    """Prove an institution word occurs inside a complete native title.

    A native title, one optional separately printed row, a person byline and
    a printed abstract must occur consecutively in one column. The source
    rows must also belong to the supplied candidate inventory. This does not
    classify the intervening row as a translation, title variant or subtitle.
    """
    from bibr.extract.front_matter import (
        _BYLINE_MARKER_RE,
        _WORD_RE,
        AFFILIATION_MARKER_RE,
        _looks_like_legacy_byline,
        _looks_like_separator_name_list,
        _normalize_text,
    )

    sections = {section.section_id: section for section in contents.sections}
    regions = sorted(contents.region_summaries, key=lambda row: (row.page, row.index))
    if len({(row.page, row.index) for row in regions}) != len(regions):
        return ()
    sentences = {row.text_id: row for row in contents.sentences}
    evidence: list[NativeTitleEvidence] = []

    def owners(region: RegionSummary) -> tuple[str, ...]:
        normalized = _normalize_text(region.content or "")
        if not normalized:
            return ()
        matched = tuple(
            row.candidate_id
            for row in candidates
            if row.source_kind == "paragraph"
            and row.text_ids
            and row.section_id == region.section_id
            and normalized in _normalize_text(row.raw_text)
            and any(
                key in sentences
                # Implicit sections can reassign the sentence's semantic
                # container. The frozen candidate/region identity above and
                # exact source provenance below still establish ownership.
                and any(
                    point.page_no == region.page and point.bbox == region.bbox
                    for point in sentences[key].provenance
                )
                for key in row.text_ids
            )
        )
        return matched if len(matched) == 1 else ()

    for title in candidates:
        section = sections.get(title.section_id) if title.section_id is not None else None
        marker = AFFILIATION_MARKER_RE.search(title.raw_text)
        if (
            "title" not in title.roles
            or title.roles & {"abstract", "byline", "doi", "body_heading", "byline_probation"}
            or title.source_kind != "heading"
            or title.region_label != "doc_title"
            or section is None
            or section.header_is_synthetic
            or _normalize_text(section.header) != _normalize_text(title.raw_text)
            or not any(
                point.page_no == title.page and point.bbox == title.bbox
                for point in section.provenance
            )
            or marker is None
            or len(_WORD_RE.findall(title.raw_text[: marker.start()])) < 6
            or not 10 <= len(_WORD_RE.findall(title.raw_text)) <= 60
            or re.search(r"[,;]|[:\-–—]\s*$", title.raw_text)
            or title.raw_text.split()[-1].casefold()
            in {"and", "or", "of", "in", "for", "with", "the"}
        ):
            continue
        positions = [
            index
            for index, region in enumerate(regions)
            if region.section_id == title.section_id
            and region.page == title.page
            and region.label == "doc_title"
            and region.bbox == title.bbox
            and _normalize_text(region.content or "") == _normalize_text(title.raw_text)
        ]
        if len(positions) != 1 or title.bbox is None:
            continue
        following = regions[positions[0] + 1 : positions[0] + 4]
        pre_byline: list[str] = []
        byline_owners: tuple[str, ...] = ()
        previous_bottom = title.bbox[3]
        for region in following:
            if (
                region.page != title.page
                or region.bbox is None
                or region.bbox[1] < previous_bottom
                or max(region.bbox[0], title.bbox[0]) >= min(region.bbox[2], title.bbox[2])
                or not owners(region)
            ):
                break
            previous_bottom = region.bbox[3]
            text = (region.content or "").strip()
            if byline_owners:
                if region.label == "abstract":
                    evidence.append(
                        NativeTitleEvidence(
                            title.candidate_id, title.raw_text, tuple(pre_byline), byline_owners
                        )
                    )
                break
            if region.label not in {"text", "content", "paragraph_title"}:
                break
            clean = _BYLINE_MARKER_RE.sub(" ", text)
            normalized = _normalize_text(clean)
            if not AFFILIATION_MARKER_RE.search(clean) and (
                _looks_like_separator_name_list(clean, normalized)
                or _looks_like_legacy_byline(clean, normalized, source_kind="paragraph")
            ):
                byline_owners = owners(region)
            elif not pre_byline and text[:1].isupper() and 6 <= len(_WORD_RE.findall(text)) <= 60:
                pre_byline.append(text)
            else:
                break
    return tuple(evidence)
