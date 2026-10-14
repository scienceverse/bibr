"""Title candidates other than the model's: printed rows that can stand in for it.

The model reads the selected front-matter record and returns a title; the rows
here are the alternatives the title decision
(:func:`bibr.extract.field_decisions.decide_title`) weighs against it or falls
back to: a selected-record title row, the layout-detected title, the row
printed directly above the byline, the row that prints the paper's registry
title, the first unclassified heading, and the filters that keep journal
furniture, generic article labels, bare section headings and mastheads out.
Nothing here writes the title.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from bibr.extract.field_decisions import FieldCandidate
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bibr.extract.front_matter import FrontMatterCandidate

logger = logging.getLogger(__name__)

# Layout doc_title vs LLM title reconciliation (masthead guard). A journal /
# publisher name shorter than this is too generic to treat a match as a
# masthead signal.
_MASTHEAD_MIN_NAME_CHARS = 6
# detected_title fuzzy ratio at/above which it "is" the journal/publisher name.
_MASTHEAD_NAME_RATIO = 0.90
# detected vs LLM title ratio at/above which they agree (keep the verbatim
# layout title even on a masthead match — avoids overriding when the LLM merely
# paraphrased a genuine title).
_TITLE_AGREE_RATIO = 0.85
# Use a narrow masthead grammar: a URL or ISSN is stronger page-furniture evidence than a volume
# label or bare DOI.
_MASTHEAD_MARKER_RE = re.compile(r"https?://|www\.|ISSN\s*\d{4}", re.IGNORECASE)


def _normalize_for_match(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .:;,-")


def _title_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_for_match(a), _normalize_for_match(b)).ratio()


def _name_matches(detected_norm: str, name: str | None) -> bool:
    if not name:
        return False
    n = _normalize_for_match(name)
    if len(n) < _MASTHEAD_MIN_NAME_CHARS:
        return False
    return n in detected_norm or difflib.SequenceMatcher(None, detected_norm, n).ratio() >= (
        _MASTHEAD_NAME_RATIO
    )


def _detected_title_is_masthead(detected: str, journal: str | None, publisher: str | None) -> bool:
    """True when the layout-detected title is really a masthead/banner.

    Layout detection can label the journal or publisher banner as a document title. Matching an independently extracted journal/publisher name or finding a banner-only URL/ISSN marker provides specific evidence that the text is page furniture.
    """
    d = _normalize_for_match(detected)
    if _name_matches(d, journal) or _name_matches(d, publisher):
        return True
    return bool(_MASTHEAD_MARKER_RE.search(detected))


# Reject a composite heading only when every content token belongs to the heading vocabulary,
# using accent-insensitive matching.
_BODY_HEADING_WORDS = frozenset(
    {
        # presentation / introduction
        "apresentacao",
        "presentacion",
        "presentazione",
        "presentation",
        "introducao",
        "introduccion",
        "introduzione",
        "introduction",
        # method / materials
        "metodologia",
        "metodologias",
        "metodologie",
        "methodologie",
        "metodo",
        "metodos",
        "metodi",
        "methode",
        "methodes",
        "materiais",
        "materiales",
        "materiali",
        "materiel",
        "materiels",
        "procedimentos",
        "procedimientos",
        # results / analysis
        "resultado",
        "resultados",
        "resultat",
        "resultats",
        "risultati",
        "risultato",
        "analise",
        "analises",
        "analisis",
        "analisi",
        "analyse",
        "analyses",
        "dados",
        "datos",
        "dati",
        "donnees",
        # discussion / conclusion
        "discussao",
        "discussoes",
        "discusion",
        "discusiones",
        "discussione",
        "discussioni",
        "discussion",
        "conclusao",
        "conclusoes",
        "conclusion",
        "conclusiones",
        "conclusione",
        "conclusioni",
        "conclusions",
        "consideracoes",
        "consideraciones",
        "considerazioni",
        "considerations",
        "sintese",
        "sintesi",
        "synthese",
        "limitacoes",
        "limitaciones",
        "limitazioni",
        "recomendacoes",
        "recomendaciones",
        "raccomandazioni",
        "recommandations",
        "final",
        "finais",
        "finales",
        "finali",
        # framing / literature
        "revisao",
        "revision",
        "revisione",
        "revue",
        "literatura",
        "letteratura",
        "litterature",
        "fundamentacao",
        "fundamentacion",
        "fundamentos",
        "teorica",
        "teorico",
        "teoricos",
        "theorique",
        "objetivo",
        "objetivos",
        "obiettivi",
        "obiettivo",
        "objectif",
        "objectifs",
        "hipotese",
        "hipoteses",
        "hipotesis",
        "ipotesi",
        "hypothese",
        "hypotheses",
        # front/back matter
        "resumo",
        "resumen",
        "riassunto",
        "resume",
        "palavras",
        "palabras",
        "parole",
        "chave",
        "clave",
        "chiave",
        "cle",
        "cles",
        "agradecimentos",
        "agradecimientos",
        "ringraziamenti",
        "remerciements",
        "referencias",
        "riferimenti",
        "bibliografia",
        "bibliografias",
    }
)
# Function words and articles that carry no heading/title signal on their own.
_HEADING_FILLER_WORDS = frozenset(
    {
        "a",
        "ai",
        "al",
        "alla",
        "and",
        "as",
        "com",
        "con",
        "da",
        "das",
        "de",
        "degli",
        "dei",
        "del",
        "della",
        "delle",
        "dello",
        "des",
        "di",
        "do",
        "dos",
        "du",
        "e",
        "ed",
        "el",
        "em",
        "en",
        "et",
        "gli",
        "i",
        "il",
        "in",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "of",
        "os",
        "para",
        "per",
        "pour",
        "the",
        "u",
        "um",
        "uma",
        "un",
        "una",
        "unas",
        "uno",
        "unos",
        "y",
    }
)
_HEADING_TOKEN_RE = re.compile(r"[^\W\d_]+")
# A printed heading is short; beyond this the row is prose, not a heading.
_MAX_HEADING_TOKENS = 8


def _strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", value) if not unicodedata.combining(char)
    )


def _is_ordinary_body_heading(normalized: str) -> bool:
    """Whether a normalized row is a bare body heading rather than a title.

    Complements ``_ORDINARY_HEADING_TEXT``'s exact-membership test: numbering
    and function words are dropped, and the row is a heading only when every
    remaining token is heading vocabulary.
    """

    tokens = _HEADING_TOKEN_RE.findall(_strip_accents(normalized))
    if not tokens or len(tokens) > _MAX_HEADING_TOKENS:
        return False
    content = [token for token in tokens if token not in _HEADING_FILLER_WORDS]
    return bool(content) and all(token in _BODY_HEADING_WORDS for token in content)


def _title_text_is_unsafe(
    text: str, normalized: str, region_label: str, *, journal: str | None, publisher: str | None
) -> bool:
    """Shared text-level filters for any printed row proposed as the title.

    Journal furniture, generic article labels, bare section headings and
    mastheads all fail closed: asserting a wrong title is worse than asserting
    none.
    """

    from bibr.extract.front_matter import (
        _ORDINARY_HEADING_TEXT,
        _looks_like_masthead,
        is_exact_front_matter_furniture,
    )
    from bibr.utils.metadata import is_exact_generic_article_label

    return bool(
        not text
        or is_exact_generic_article_label(text)
        or is_exact_front_matter_furniture(text)
        or normalized in _ORDINARY_HEADING_TEXT
        or _is_ordinary_body_heading(normalized)
        or _looks_like_masthead(text, region_label)
        or _detected_title_is_masthead(text, journal, publisher)
    )


def _selected_front_matter_block(resolution):
    """The resolved front-matter block, or None when there is no ownership."""

    if resolution is None or resolution.selected_block_id is None:
        return None
    return next(
        (block for block in resolution.blocks if block.block_id == resolution.selected_block_id),
        None,
    )


def _safe_title_candidate(candidate, *, journal: str | None, publisher: str | None) -> bool:
    """Whether one selected-record candidate may stand in as the article title."""

    text = candidate.raw_text.strip()
    return not (
        "title" not in candidate.roles
        or not candidate.roles.isdisjoint({"abstract", "affiliation", "byline", "doi"})
        or _title_text_is_unsafe(
            text,
            candidate.normalized_text,
            (candidate.region_label or "").casefold(),
            journal=journal,
            publisher=publisher,
        )
    )


def selected_title_candidate(
    resolution, *, journal: str | None, publisher: str | None
) -> tuple[FieldCandidate | None, str]:
    """Exactly one safe title row of the selected front-matter record.

    Reads only the selected block's title candidates — never the global
    detected title or unknown-section headers. Ambiguity (two distinct
    candidates after normalization), mastheads, journal furniture, generic
    article labels, bare section headings, and composite title+byline/abstract/
    affiliation/DOI rows all fail closed. Returns the candidate, or None, and
    the reason.
    """

    selected_block = _selected_front_matter_block(resolution)
    if selected_block is None:
        return None, "no selected record"

    by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    safe = [
        candidate
        for candidate_id in selected_block.title_candidate_ids
        if (candidate := by_id.get(candidate_id)) is not None
        and _safe_title_candidate(candidate, journal=journal, publisher=publisher)
    ]

    distinct: dict[str, FrontMatterCandidate] = {}
    for candidate in safe:
        distinct.setdefault(_normalize_for_match(candidate.raw_text), candidate)
    if len(distinct) != 1:
        return None, (
            f"{len(distinct)} distinct safe title rows" if distinct else "no safe title row"
        )
    candidate = next(iter(distinct.values()))
    issue = ValidationIssue(
        code="VAL_TITLE_RECOVERED",
        severity=IssueSeverity.WARNING,
        message="Recovered null model title from the selected front-matter record",
        origin_stage="extract",
        evidence_ids=(candidate.candidate_id,),
        blocking=False,
    )
    return (
        FieldCandidate(
            "title",
            "front_matter_candidate",
            candidate.raw_text,
            evidence_ids=(candidate.candidate_id,),
            issues=(issue,),
        ),
        "one safe title row",
    )


def detected_title_candidate(
    detected_title: str | None, *, journal: str | None, publisher: str | None
) -> tuple[FieldCandidate | None, str]:
    """The layout ``detected_title`` under the selected-title filters.

    Under ownership scope, a null model title and no safe selected candidate
    can leave the title null even when layout has the printed title. Reuses the
    selected-title filters without enabling the broader unknown-header scan.
    """

    from bibr.extract.front_matter import _normalize_text

    detected = (detected_title or "").strip()
    if _title_text_is_unsafe(
        detected, _normalize_text(detected), "doc_title", journal=journal, publisher=publisher
    ):
        return None, "no safe layout title"
    issue = ValidationIssue(
        code="VAL_TITLE_RECOVERED",
        severity=IssueSeverity.WARNING,
        message="Recovered null model title from the layout-detected title",
        origin_stage="extract",
        blocking=False,
    )
    return FieldCandidate("title", "layout_title", detected, issues=(issue,)), "safe layout title"


def byline_adjacent_candidate(
    resolution, title: str, *, journal: str | None, publisher: str | None
) -> tuple[FieldCandidate | None, str]:
    """The printed title row directly above the byline, when it disagrees with *title*.

    Multilingual front matter may print the original title above the byline
    and a translation above another abstract. Using this row replaces an
    asserted title, so the decision gates it on
    PIPELINE_TITLE_PREFER_BYLINE_ADJACENT (off by default pending broader
    validation).
    """

    llm_title = (title or "").strip()
    if not llm_title:
        return None, "no title to compare"
    selected_block = _selected_front_matter_block(resolution)
    if selected_block is None:
        return None, "no selected record"

    owned = set(selected_block.candidate_ids) | set(selected_block.title_candidate_ids)
    in_block = [candidate for candidate in resolution.candidates if candidate.candidate_id in owned]
    byline_order = min(
        (candidate.reading_order for candidate in in_block if "byline" in candidate.roles),
        default=None,
    )
    if byline_order is None:
        return None, "no byline"
    above = [
        candidate
        for candidate in in_block
        if candidate.reading_order < byline_order
        and _safe_title_candidate(candidate, journal=journal, publisher=publisher)
    ]
    if not above:
        return None, "no safe row above the byline"
    candidate = max(above, key=lambda item: item.reading_order)

    wanted = _normalize_for_match(llm_title)
    if (
        wanted in _normalize_for_match(candidate.raw_text)
        or _title_ratio(candidate.raw_text, llm_title) >= _TITLE_AGREE_RATIO
    ):
        return None, "agrees with the title"
    issue = ValidationIssue(
        code="VAL_TITLE_BYLINE_ADJACENT",
        severity=IssueSeverity.WARNING,
        message="Preferred the printed title row directly above the byline over the model title",
        origin_stage="extract",
        evidence_ids=(candidate.candidate_id,),
        blocking=False,
    )
    return (
        FieldCandidate(
            "title",
            "byline_adjacent",
            candidate.raw_text,
            evidence_ids=(candidate.candidate_id,),
            issues=(issue,),
        ),
        "row above the byline disagrees with the title",
    )


def _contains_words(haystack: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {haystack} "


def _selected_rows(resolution) -> list[FrontMatterCandidate]:
    """The selected record's rows in reading order; empty without one."""

    block = _selected_front_matter_block(resolution)
    if block is None:
        return []
    owned = set(block.candidate_ids) | set(block.title_candidate_ids)
    return sorted(
        (candidate for candidate in resolution.candidates if candidate.candidate_id in owned),
        key=lambda candidate: candidate.reading_order,
    )


def _is_title_row(candidate: FrontMatterCandidate) -> bool:
    return "title" in candidate.roles and candidate.roles.isdisjoint(
        {"abstract", "affiliation", "byline", "doi"}
    )


def printed_title_rows(title: str | None, resolution) -> tuple[FrontMatterCandidate, ...] | None:
    """The selected record's title rows that print *title*, in reading order.

    One title row whose text holds it, or up to three consecutive title rows
    whose joined text is it (a title printed over two lines, or a title and
    its subtitle row), case, punctuation and spacing aside. None when no
    title row prints it.
    """
    from bibr.extract.title_subtitle import _MAX_TITLE_ROWS, _normalize
    from bibr.utils.metadata import is_exact_generic_article_label

    wanted = _normalize(title or "")
    if not wanted or is_exact_generic_article_label(title or ""):
        return None
    rows = [candidate for candidate in _selected_rows(resolution) if _is_title_row(candidate)]
    loose = _normalize_for_match(title or "")
    for row in rows:
        if _contains_words(_normalize(row.raw_text), wanted) or (
            loose and loose in _normalize_for_match(row.raw_text)
        ):
            return (row,)
    for start in range(len(rows)):
        joined = ""
        for end in range(start, min(start + _MAX_TITLE_ROWS, len(rows))):
            joined = f"{joined} {_normalize(rows[end].raw_text)}".strip()
            if joined == wanted:
                return tuple(rows[start : end + 1])
            if not wanted.startswith(f"{joined} "):
                break
    return None


def printed_after_language_version(
    title: str | None,
    resolution,
    *,
    journal: str | None = None,
    publisher: str | None = None,
) -> bool:
    """Whether the selected record prints *title* only after a version of it in another
    language or script.

    Multilingual front matter prints the original title first, and the title
    to use is that one (the printed-first rule). The versions compared are
    the safe title rows printed before *title*'s own row, alone or up to
    three consecutive ones joined (an original title printed with its
    subtitle), and the text its own row prints before it ("<original> /
    <translation>"). False when no title row prints *title*.
    """
    from bibr.extract.title_subtitle import _MAX_TITLE_ROWS, _are_language_versions, _normalize

    located = printed_title_rows(title, resolution)
    if located is None:
        return False
    first = located[0]
    earlier_texts: list[str] = []
    if len(located) == 1:
        row_text, wanted = f" {_normalize(first.raw_text)} ", f" {_normalize(title or '')} "
        at = row_text.find(wanted)
        if at > 0:
            earlier_texts.append(row_text[:at].strip())
    earlier = [
        candidate
        for candidate in _selected_rows(resolution)
        if candidate.reading_order < first.reading_order
        and _safe_title_candidate(candidate, journal=journal, publisher=publisher)
    ]
    for start in range(len(earlier)):
        joined = ""
        for end in range(start, min(start + _MAX_TITLE_ROWS, len(earlier))):
            joined = f"{joined} {earlier[end].raw_text.strip()}".strip()
            earlier_texts.append(joined)
    return any(_are_language_versions(text, title or "") for text in earlier_texts)


def _record_title_rows(
    in_block: list[FrontMatterCandidate],
    record_titles: list[str],
    *,
    journal: str | None,
    publisher: str | None,
) -> tuple[FrontMatterCandidate, str, tuple[str, ...]] | None:
    """The safe row printing the record's title, its printed text and evidence ids.

    With a subtitle in the record, the row must print the title and the
    subtitle, or the title alone directly above a row that prints the
    subtitle (the two rows' printed text joined, as a folded subtitle is).
    """
    from bibr.enrich.verify_identity import _title_key
    from bibr.extract.title_subtitle import _separator

    safe = [
        candidate
        for candidate in in_block
        if _safe_title_candidate(candidate, journal=journal, publisher=publisher)
    ]
    # With a subtitle the record's title alone would drop a printed part.
    wanted = _title_key(record_titles[-1])
    row = next((candidate for candidate in safe if _title_key(candidate.raw_text) == wanted), None)
    if row is not None:
        return row, row.raw_text, (row.candidate_id,)
    if len(record_titles) < 2:
        return None
    title_key = _title_key(record_titles[0])
    subtitle_key = wanted[len(title_key) :].strip()
    for candidate in safe:
        if _title_key(candidate.raw_text) != title_key:
            continue
        following = [item for item in in_block if item.reading_order > candidate.reading_order]
        if not following:
            continue
        subtitle_row = min(following, key=lambda item: item.reading_order)
        if (
            subtitle_row.roles.isdisjoint({"abstract", "affiliation", "byline", "doi"})
            and _title_key(subtitle_row.raw_text) == subtitle_key
        ):
            stem, subtitle = candidate.raw_text.rstrip(), subtitle_row.raw_text.strip()
            return (
                candidate,
                f"{stem}{_separator(stem, subtitle)}{subtitle}",
                (candidate.candidate_id, subtitle_row.candidate_id),
            )
    return None


def registry_title_candidate(
    resolution,
    title: str,
    record: Mapping[str, Any],
    *,
    journal: str | None,
    publisher: str | None,
) -> tuple[FieldCandidate | None, str]:
    """The printed title row that is the paper's registry title, when *title* disagrees.

    *record* is the paper's own confident ``metadata_match`` row. When the
    extracted title disagrees with it, the selected record's safe title row
    whose text (case and punctuation aside) is the record's title, followed
    by its subtitle when the record keeps one, stands in; a title row
    directly above a row printing the subtitle counts too. The rows' printed
    text is used, never the record's. It is refused when:

    - the extracted title extends the row (the record lacks a printed part);
    - the extracted title is printed in the selected record before the row,
      whatever its language (the printed-first rule, as measured);
    - the row is a version in another language or script of a title row
      printed before it, or of the extracted title, and not the version
      printed first (multilingual front matter prints the original first,
      and registries often hold the English one);
    - no row prints the record's title, as when the registry has a typo.

    The decision gates it on FRONT_MATTER_REGISTRY_TITLE (off by default).
    """
    from bibr.enrich.verify_identity import _record_titles, _title_key, titles_agree
    from bibr.extract.title_subtitle import _are_language_versions, drop_parallel_title

    extracted = (title or "").strip()
    if not extracted:
        return None, "no title to compare"
    record_titles = _record_titles(dict(record))
    if not record_titles:
        return None, "the record has no title"
    if any(titles_agree(extracted, record_title) for record_title in record_titles):
        return None, "agrees with the record"
    if _selected_front_matter_block(resolution) is None:
        return None, "no selected record"

    in_block = _selected_rows(resolution)
    found = _record_title_rows(in_block, record_titles, journal=journal, publisher=publisher)
    if found is None:
        return None, "no printed row is the record's title"
    row, printed, row_ids = found
    wanted = _title_key(printed)

    extracted_key = _title_key(extracted)
    if extracted_key.startswith(wanted + " "):
        return None, "the title extends the row"
    if any(
        candidate.reading_order < row.reading_order
        and _contains_words(_title_key(candidate.raw_text), extracted_key)
        for candidate in in_block
    ):
        return None, "the title is printed before the row"
    if printed_after_language_version(
        row.raw_text, resolution, journal=journal, publisher=publisher
    ):
        return None, "the row is a translation not printed first"
    if _are_language_versions(extracted, printed):
        printed_text = "\n".join(candidate.raw_text for candidate in in_block)
        kept, _ = drop_parallel_title(f"{extracted} / {printed}", printed_text)
        if _title_key(kept) != wanted:
            return None, "the row is a translation not printed first"

    service = str(record.get("service") or "registry")
    evidence = (*row_ids, f"metadata_match:{service}")
    issue = ValidationIssue(
        code="VAL_TITLE_REGISTRY_PRINTED",
        severity=IssueSeverity.WARNING,
        message=(
            "Preferred the printed title row that matches the paper's registry record over the "
            "extracted title"
        ),
        origin_stage="extract",
        evidence_ids=evidence,
        blocking=False,
    )
    return (
        FieldCandidate(
            "title", "registry_printed", printed, evidence_ids=evidence, issues=(issue,)
        ),
        "the printed row matches the record and the title does not",
    )


def grounded_in_selected_title(title: str | None, resolution) -> bool:
    """Whether *title* is printed inside one of the selected record's title rows."""

    from bibr.utils.metadata import is_exact_generic_article_label

    if (
        not title
        or is_exact_generic_article_label(title)
        or resolution is None
        or resolution.selected_block_id is None
    ):
        return False
    selected = {
        candidate_id
        for block in resolution.blocks
        if block.block_id == resolution.selected_block_id
        for candidate_id in block.title_candidate_ids
    }
    wanted = _normalize_for_match(title)
    return any(
        candidate.candidate_id in selected
        and "title" in candidate.roles
        and candidate.roles.isdisjoint({"abstract", "byline", "affiliation", "doi"})
        and wanted
        and wanted in _normalize_for_match(candidate.raw_text)
        for candidate in resolution.candidates
    )


def layout_title_yields(
    detected: str, title: str | None, *, journal, publisher, resolution
) -> str | None:
    """Why the layout-detected title must not replace *title*, or None when it may.

    An exact generic label ("Research Article") yields to a title printed in
    the selected record. A layout title with specific masthead evidence
    (journal/publisher agreement, a URL or ISSN) yields to a title it
    disagrees with; disagreement alone is not enough.
    """

    from bibr.utils.metadata import is_exact_generic_article_label

    if is_exact_generic_article_label(detected) and grounded_in_selected_title(title, resolution):
        return "generic label; the title is printed in the selected record"
    if (
        title
        and _detected_title_is_masthead(detected, journal, publisher)
        and _title_ratio(detected, title) < _TITLE_AGREE_RATIO
    ):
        logger.info(
            "Masthead-title guard: layout doc_title %r matches journal/publisher; "
            "using LLM title %r instead",
            detected,
            title,
        )
        return "masthead"
    return None


def section_header_candidate(sections) -> FieldCandidate | None:
    """The first unclassified heading that is not a canonical section name."""

    from bibr.paper_contents import CanonicalSection

    canonical_headers = {s.value for s in CanonicalSection if s != CanonicalSection.UNKNOWN}
    for section in sections:
        if section.level > 0 and section.section_type == CanonicalSection.UNKNOWN:
            header_lower = section.header.lower().strip()
            if header_lower and header_lower not in canonical_headers:
                return FieldCandidate("title", "section_header", section.header)
    return None
