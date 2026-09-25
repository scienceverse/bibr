"""Structured research-integrity metadata extraction and compatibility helpers.

The pipeline resolves scalar statement ownership in
``bibr.extract.integrity_statements``.  This module retains
``copy_integrity_statements`` for direct legacy callers and provides
``extract_structured_integrity``: one LLM call that parses the selected funding
statement into structured ``FundingEntry`` records, maps the
author-contributions statement onto ``PaperAuthor.role``, and parses the
deduped author-byline affiliation strings into structured ``Affiliation``
records. Skipped entirely when the funding statement, contributions statement,
and affiliation list are all empty; the caller gates it on LLM availability.
The affiliation ``text`` fields are our own verbatim strings (keyed by index
into the list we send); the LLM only supplies the parsed sub-fields.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from bibr.exceptions import ProcessingError
from bibr.models import Affiliation, FundingEntry
from bibr.paper_contents import CanonicalSection

if TYPE_CHECKING:
    from bibr.clients.llm_protocol import LlmClient
    from bibr.extract.integrity_statements import IntegrityStatementResolution
    from bibr.models import PaperAuthor, PaperMetadata
    from bibr.paper_contents import PaperContents
    from bibr.schemas import AffiliationLLM, AuthorContributionLLM

logger = logging.getLogger(__name__)

# PaperMetadata statement field → the canonical section type it copies from.
_STATEMENT_SECTION_TYPES: dict[str, CanonicalSection] = {
    "funding_statement": CanonicalSection.FUNDING,
    "coi_statement": CanonicalSection.COI,
    "ethics_statement": CanonicalSection.ETHICS,
    "data_availability": CanonicalSection.OPEN_DATA,
}


def _build_section_text_map(contents: PaperContents) -> dict[int, str]:
    """Join sentence text per ``section_id`` (display formulas excluded).

    Rebuilt from ``sentences`` rather than reusing ``contents.sections_text``
    so it reflects the final section assignment after post-parse mutations,
    mirroring the abstract fallback in ``post_parse._finalize_abstract_and_keywords``.
    """
    grouped: dict[int, list[str]] = {}
    for sent in contents.sentences:
        if sent.is_display_formula:
            continue
        grouped.setdefault(sent.section_id, []).append(sent.text)
    return {sid: " ".join(parts).strip() for sid, parts in grouped.items()}


def _text_for_type(
    contents: PaperContents, text_map: dict[int, str], section_type: CanonicalSection
) -> str | None:
    """Text of every section of ``section_type``, in document order, joined by
    a blank line. ``None`` when no such section carries text."""
    parts: list[str] = []
    for section in contents.sections:
        if section.section_type == section_type:
            text = text_map.get(section.section_id, "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts) if parts else None


def copy_integrity_statements(contents: PaperContents, metadata: PaperMetadata) -> None:
    """Copy FUNDING / COI / ETHICS / OPEN_DATA section bodies verbatim onto the
    corresponding ``PaperMetadata`` statement fields. Pure copy — no LLM."""
    text_map = _build_section_text_map(contents)
    for field, section_type in _STATEMENT_SECTION_TYPES.items():
        setattr(metadata, field, _text_for_type(contents, text_map, section_type))


def _initials(name: str) -> str:
    """Leading letter of each whitespace-separated token, uppercased."""
    return "".join(tok[0] for tok in name.split() if tok).upper()


def _author_initial_forms(author: PaperAuthor) -> set[str]:
    """Accepted initials spellings for an author: all-given-initials + family
    initial, and first-given-initial + family initial (so 'JW' and 'JMW' both
    match 'Jakub M Werner')."""
    given_inits = _initials(author.given)
    fam_init = _initials(author.family)
    if not given_inits or not fam_init:
        return set()
    return {given_inits + fam_init, given_inits[0] + fam_init}


def _is_initials_token(token: str) -> bool:
    """True for tokens that are purely initials (e.g. 'J.W.', 'JW', 'JMW')."""
    core = re.sub(r"[.\s]", "", token)
    return 2 <= len(core) <= 4 and core.isalpha() and core.isupper()


def _family_matches(token: str, author: PaperAuthor) -> bool:
    """The author's family name appears as a whole word in the printed token."""
    family = author.family.strip()
    if not family:
        return False
    return re.search(rf"\b{re.escape(family)}\b", token, re.IGNORECASE) is not None


def _match_author(token: str, authors: list[PaperAuthor]) -> PaperAuthor | None:
    """Resolve a printed contribution token to a single author, or ``None``.

    Conservative: an exact family-name match wins; failing that, an initials
    match is tried only when the token is purely initials. A token that matches
    more than one author (ambiguous) resolves to ``None`` and is dropped.
    """
    token = token.strip()
    if not token:
        return None

    fam_matches = [a for a in authors if _family_matches(token, a)]
    if len(fam_matches) == 1:
        return fam_matches[0]
    if len(fam_matches) > 1:
        return None  # ambiguous — drop

    if _is_initials_token(token):
        target = re.sub(r"[^A-Za-z]", "", token).upper()
        init_matches = [a for a in authors if target in _author_initial_forms(a)]
        if len(init_matches) == 1:
            return init_matches[0]
    return None


def _apply_contributions(
    authors: list[PaperAuthor], contributions: list[AuthorContributionLLM]
) -> None:
    """Map contribution roles onto ``authors[*].role`` (verbatim). Unmatched or
    ambiguous entries are dropped; authors with no match keep ``role=[]``.
    Roles merge (order-preserving, deduplicated) — role-keyed statements
    ("Conceptualization: J.W.; Writing: J.W.") yield one entry per role for
    the same author, and a later entry must not overwrite an earlier one."""
    for entry in contributions:
        roles = [r.strip() for r in entry.roles if r and r.strip()]
        if not roles:
            continue
        author = _match_author(entry.author, authors)
        if author is None:
            logger.debug("Dropping unmatched contribution entry for %r", entry.author)
            continue
        author.role.extend(r for r in roles if r not in author.role)


def collect_affiliations(authors: list[PaperAuthor]) -> tuple[list[str], list[list[int]]]:
    """Split each author's ``affiliation`` on "; " into verbatim components and
    dedupe across authors, preserving first-seen order.

    Returns ``(unique, author_ids)``: ``unique[k]`` is the k-th distinct
    affiliation string, and ``author_ids[k]`` is the ``author_id``s (in author
    order, deduped) whose affiliation contains that string as a component.
    """
    unique: list[str] = []
    author_ids: list[list[int]] = []
    index_of: dict[str, int] = {}
    for author in authors:
        seen_for_author: set[str] = set()
        for raw in author.affiliation.split("; "):
            comp = raw.strip()
            if not comp or comp in seen_for_author:
                continue
            seen_for_author.add(comp)
            if comp not in index_of:
                index_of[comp] = len(unique)
                unique.append(comp)
                author_ids.append([])
            ids = author_ids[index_of[comp]]
            if author.author_id not in ids:
                ids.append(author.author_id)
    return unique, author_ids


async def extract_structured_integrity(
    contents: PaperContents,
    metadata: PaperMetadata,
    llm_client: LlmClient,
    file_hash: str,
    *,
    integrity_resolution: IntegrityStatementResolution | None = None,
) -> None:
    """Parse structured funding, author contributions, and affiliations via a
    single LLM call.

    Skips the call when the funding statement, author-contributions statement,
    and (deduped) affiliation list are all empty. Populates
    ``metadata.funding``, ``metadata.authors[*].role``, and
    ``metadata.affiliations`` (verbatim ``text`` from our own byline strings,
    parsed sub-fields from the LLM).
    """
    text_map = _build_section_text_map(contents)
    funding_text = _text_for_type(contents, text_map, CanonicalSection.FUNDING)
    if contents.preparsed_metadata is not None:
        funding_text = metadata.funding_statement
    elif integrity_resolution is not None and integrity_resolution.mode in {"shadow", "active"}:
        from bibr.extract.integrity_statements import render_selected_integrity_statement

        funding_text = render_selected_integrity_statement(
            contents,
            integrity_resolution,
            "funding_statement",
        )
    elif integrity_resolution is not None and integrity_resolution.mode == "legacy":
        legacy_indices = integrity_resolution.legacy_indices("funding_statement")
        canonical_legacy = any(
            integrity_resolution.candidates[index].method == "legacy_section_copy"
            for index in legacy_indices
        )
        funding_text = (
            dict(integrity_resolution.legacy_statement_snapshots).get("funding_statement")
            if canonical_legacy
            else None
        )
    contributions_text = _text_for_type(contents, text_map, CanonicalSection.AUTHOR_CONTRIBUTIONS)
    unique_affils, affil_author_ids = collect_affiliations(metadata.authors)
    if not funding_text and not contributions_text and not unique_affils:
        return

    author_names = [(a.given, a.family) for a in metadata.authors]
    try:
        result = await llm_client.extract_research_integrity(
            funding_text=funding_text or "",
            contributions_text=contributions_text or "",
            author_names=author_names,
            affiliation_list=unique_affils,
            file_hash=file_hash,
        )
    except ProcessingError:
        raise
    except Exception as e:  # noqa: BLE001 — degrade gracefully, keep the paper
        from bibr.clients.llm import llm_failure_code
        from bibr.processing_warnings import ProcessingWarning, WarningCode

        logger.warning("Research-integrity extraction failed (hash=%s): %s", file_hash, e)
        contents.processing_warnings.append(
            ProcessingWarning(
                WarningCode.RESEARCH_INTEGRITY_LLM_FAILED,
                f"{llm_failure_code(e)}: structured funding, author roles and affiliation "
                "parts were not parsed",
            )
        )
        return

    # Gate funding on the funding statement actually existing: when funding_text
    # is empty (only affiliations/contributions drove the call), NuExtract3 can
    # hallucinate placeholder funding entries — discard them wholesale.
    metadata.funding = (
        [
            FundingEntry(
                funder=f.funder.strip(), award_ids=[a.strip() for a in f.award_ids if a.strip()]
            )
            for f in result.funding
            if f.funder and f.funder.strip()
        ]
        if funding_text
        else []
    )
    _apply_contributions(metadata.authors, result.contributions)
    metadata.affiliations = _build_affiliations(
        unique_affils, affil_author_ids, result.affiliations
    )


def _clean_component(value: str | None) -> str | None:
    """Strip a parsed affiliation component; empty-after-strip → ``None``."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _build_affiliations(
    unique: list[str], author_ids: list[list[int]], parsed: list[AffiliationLLM]
) -> list[Affiliation]:
    """Merge our verbatim affiliation strings with the LLM's structured parse.

    Every unique string appears in output order (regardless of LLM emission
    order); an entry with an out-of-range index is dropped and, on a duplicate
    index, the first entry wins. Strings the LLM omitted are still emitted with
    all parse fields ``None`` — the verbatim list is ours, the parse is
    best-effort.
    """
    by_index: dict[int, AffiliationLLM] = {}
    for entry in parsed:
        if 1 <= entry.index <= len(unique) and entry.index not in by_index:
            by_index[entry.index] = entry

    affiliations: list[Affiliation] = []
    for position, text in enumerate(unique, start=1):
        entry = by_index.get(position)
        affiliations.append(
            Affiliation(
                text=text,
                institution=_clean_component(entry.institution) if entry else None,
                department=_clean_component(entry.department) if entry else None,
                city=_clean_component(entry.city) if entry else None,
                country=_clean_component(entry.country) if entry else None,
                author_ids=author_ids[position - 1],
            )
        )
    return affiliations
