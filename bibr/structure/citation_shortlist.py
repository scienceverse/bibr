"""Conservative reference retrieval for the citation LLM fallback.

This only selects evidence to send. It never assigns a citation to a reference.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bibr.structure.citation_matcher import _find_candidates, extract_families
from bibr.utils.text import collapse_ws

if TYPE_CHECKING:
    from bibr.schemas import CitationMatch

_YEAR_RE = re.compile(r"\b([12]\d{3})[a-z]?\b", re.IGNORECASE)
_NAME_STOPWORDS = frozenset({"and", "et", "al", "van", "von", "de", "der", "la", "le", "del"})


def _words(value: str) -> frozenset[str]:
    normalized = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(ch)
    )
    return frozenset(
        word
        for word in re.findall(r"[^\W\d_]+", normalized)
        if len(word) > 1 and word not in _NAME_STOPWORDS
    )


def reference_table(references: list[dict]) -> str:
    """Keep the existing wire format, source order and original bib IDs."""
    return "\n".join(
        f'  bib_id={r["bib_id"]}: {r["author"]} ({r["year"]}) "{r["title"]}"' for r in references
    )


@dataclass(frozen=True)
class CitationShortlist:
    references: list[dict]
    candidate_ids: dict[str, frozenset[int]]
    narrowed: bool
    reason: str


def shortlist_references(
    citations: list[tuple[int, str]], references: list[dict]
) -> CitationShortlist:
    """Select a recall-oriented union, or retain the complete bibliography.

    A citation needs parsed author/year evidence and a matching reference for
    every year before any pruning is attempted. Keep all references sharing
    either its year or a name, plus records with incomplete retrieval fields.
    Numeric IDs are never interpreted as bibliography row numbers.
    """
    full = CitationShortlist(references, {}, False, "uncertain_citation")
    if not citations or not references:
        return full
    ids = [r.get("bib_id") for r in references]
    if any(type(bib_id) is not int for bib_id in ids) or len(set(ids)) != len(ids):
        return full

    indexed = []
    for ref in references:
        families = extract_families(ref.get("author"))
        names = frozenset().union(*(_words(family) for family in families))
        match = _YEAR_RE.fullmatch(str(ref.get("year", "")))
        year = int(match.group(1)) if match else None
        indexed.append((ref["bib_id"], names, year))

    candidates_by_citation = {}
    union: set[int] = set()
    for text_id, citation in citations:
        # The matcher handles parenthetical/narrative forms. Square brackets
        # have the same author-year grammar; changing only the wrapper keeps
        # raw citation_text intact in the actual request.
        parsed_text = citation.replace("[", "(").replace("]", ")")
        # Multi-citation splitting can supply a bare inner "Smith, 2020".
        if not any(ch in parsed_text for ch in "()"):
            parsed_text = f"({parsed_text})"
        candidates = _find_candidates(parsed_text, text_id)
        years = {int(m.group(1)) for m in _YEAR_RE.finditer(parsed_text)}
        if (
            not candidates
            or not years
            or any(c.is_in_press or c.year is None for c in candidates)
            or years != {c.year for c in candidates}
        ):
            return full
        # Unparsed pieces of a multi-citation could refer to another paper.
        # Only punctuation/whitespace may fall outside the parsed spans.
        covered = [False] * len(parsed_text)
        for candidate in candidates:
            covered[candidate.start : candidate.end] = [True] * (candidate.end - candidate.start)
        if any(ch.isalnum() and not covered[i] for i, ch in enumerate(parsed_text)):
            return full

        chosen: set[int] = set()
        for candidate in candidates:
            names = frozenset().union(*(_words(family) for family in candidate.families))
            if not names or not any(
                year == candidate.year and names & ref_names for _, ref_names, year in indexed
            ):
                return full
            chosen.update(
                bib_id
                for bib_id, ref_names, year in indexed
                if year == candidate.year or names & ref_names or year is None or not ref_names
            )
        key = collapse_ws(citation)
        candidates_by_citation[key] = frozenset(chosen)
        union.update(chosen)

    selected = [ref for ref in references if ref["bib_id"] in union]
    before, after = len(reference_table(references)), len(reference_table(selected))
    # Avoid changing the inference context for negligible savings. These are
    # character counts, not tokenizer claims; full retries must be worthwhile.
    if before - after < 512 or after > before * 0.75:
        return CitationShortlist(references, {}, False, "small_saving")
    return CitationShortlist(selected, candidates_by_citation, True, "author_year_union")


def partition_shortlist_matches(
    citations: list[tuple[int, str]],
    matches: list[CitationMatch],
    candidate_ids: dict[str, frozenset[int]],
) -> tuple[list[CitationMatch], list[tuple[int, str]]]:
    """Accept one consistent, in-candidate target per citation; retry the rest.

    Match on normalized citation text, like the linker. The linker restores
    original sentence IDs and repeated occurrences from its source spans.
    """
    by_text = {}
    for match in matches:
        by_text.setdefault(collapse_ws(match.citation_text), []).append(match)
    accepted, pending = [], []
    seen = set()
    for text_id, text in citations:
        key = collapse_ws(text)
        if key in seen:
            continue
        seen.add(key)
        answers = by_text.get(key, [])
        targets = {match.bib_id for match in answers}
        allowed = candidate_ids.get(key, frozenset())
        if len(targets) == 1 and None not in targets and targets <= allowed:
            accepted.append(answers[0])
        else:
            pending.append((text_id, text))
    return accepted, pending
