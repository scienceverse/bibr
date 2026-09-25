"""Core paper metadata extraction (title, authors, DOI, abstract, keywords).

``CoreMetadataExtractor`` builds the metadata LLM context from the rows a
:class:`bibr.extract.ref_locator.RefLocator` selects, finds the paper's own
DOI deterministically, calls the LLM, and applies the post-LLM guards
(correction-notice suppression, commentary-abstract fabrication guard,
author sanitisation, OECD/paper-type validation).
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd

from bibr.config import snapshot_settings
from bibr.exceptions import ProcessingError, UpstreamServiceError
from bibr.extract.author_email_harvester import _CORRESPONDING_MARKER_RE, AuthorEmailHarvester
from bibr.extract.front_matter import AFFILIATION_MARKER_RE
from bibr.extract.metadata_precision import refine_publication_date, repair_author_partitions
from bibr.extract.ref_locator import _ORCID_BARE_INLINE_RE, _REF_HEADER_RE, RefLocator
from bibr.extract.title_subtitle import fold_printed_subtitle
from bibr.field_states import set_field_source
from bibr.input.consolidate_text import strip_affiliation_markers
from bibr.models import ErrorCode
from bibr.paper import PaperAuthor, PaperMetadata
from bibr.paper_contents import FRONT_MATTER_FURNITURE_LABELS, CanonicalSection, PaperContents
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.schemas import PaperClassificationLLM
from bibr.utils.metadata import EXACT_GENERIC_ARTICLE_LABELS
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bibr.config import GlobalSettings
    from bibr.extract.front_matter import FrontMatterCandidate, FrontMatterResolution
    from bibr.pipeline.classifier_resources import ClassifierResources

logger = logging.getLogger(__name__)

# Above this incoming author count the list is treated as a model degeneration
# and trimmed to the leading distinct run. Real bylines in bibr's domain
# (psych/econ/MDPI) sit well under it; the 40-author consortium gold cases pass.
_MAX_LLM_AUTHORS = 64

# Drop/collapse count above which the sanitizer records a summary warning. A
# lone stray blank/duplicate is trivial noise; two or more signals an anomaly.
_AUTHOR_ANOMALY_MIN_DROP = 2

# ``CoreMetadataLLM`` field -> the export's name for it, where they differ.
_EXPORT_FIELD_NAMES = {
    "authors": "author",
    "oecd_domain": "oecd_l1",
    "oecd_subdomain": "oecd_l2",
}

# Correction-notice title prefix — matches Psychological Science (and most
# journals) corrigendum / erratum / correction / retraction front-matter.
# The trailing-character class disambiguates real notices ("Corrigendum: ...",
# 'Erratum to "..."') from research-paper titles that coincidentally start
# with one of the prefix words. See
# docs/superpowers/specs/2026-05-07-corrigendum-erratum-guard-design.md
# for the failure mode this guards against.
# The trailing requirement must carry information: with ``\s`` inside the
# character class it added nothing beyond the preceding ``\b``, so any title
# opening with the bare word matched — "Correction for attenuation in
# meta-analysis: a simulation study" classified as a corrigendum and had its
# authors, abstract and keywords wiped. Require notice punctuation, the
# preposition, or the "Note"/"Notice" form journals print.
_CORRECTION_NOTICE_TITLE_RE = re.compile(
    r"^\s*(?P<kind>corrigendum|erratum|correction|retraction)"
    r"\b(?:\s*[:.\-—–]|\s+to\b|\s+(?:note|notice)\b|\s*[\"\u201c\u2018'])",
    re.IGNORECASE,
)

_NUMBERED_AFFILIATION_RE = re.compile(
    r"(?<![\w,])(?P<number>\d{1,2})\s+(?=[A-Z\u00c0-\u00d6\u00d8-\u00de])"
)
_CORRESPONDENCE_SUFFIX_RE = re.compile(
    r"\s+(?:corresponding\s+author|correspondence)\s*:", re.IGNORECASE
)
_AFFILIATION_MATCH_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
_AFFILIATION_ORG_RE = re.compile(
    r"\b(?:department|faculty|institute|university|college|school|hospital|"
    r"centre|center|clinic|laborator(?:y|ies))\b",
    re.IGNORECASE,
)

# Positive evidence that a captured ``<number> <Capital…>`` run really is an
# institution. Without it the numbered-affiliation reconciler accepted any
# page-1 line of that shape, so a figure or table caption ("Figure 1 Study
# design and participant flow") or a publication-history line overwrote a
# correctly-extracted affiliation. Wider than ``_AFFILIATION_ORG_RE`` above,
# which gates a different, already name-anchored path.
_AFFILIATION_ORG_EVIDENCE_RE = re.compile(
    r"\b(?:"
    r"depart[ae]ment\w*|d[ée]partement\w*|dept"
    r"|facult\w*"
    r"|institut\w*"
    r"|universit\w*|universida\w*|univ"
    r"|college\w*|school\w*|academ(?:y|ia|ie)"
    r"|hospital\w*|h[ôo]pital|clinic\w*|klinik\w*|infirmary"
    r"|cent(?:re|er)\w*|centro|zentrum"
    r"|laborator\w*|laboratoire"
    r"|ministry|minist[èe]re|foundation|fondation|fundac[ií][óo]n"
    r"|corporation|inc|ltd|llc|gmbh|plc"
    r"|nhs|cnrs|inserm|cdc|nih"
    r")\b",
    re.IGNORECASE,
)

# Postal-address shape, for institutions the keyword list cannot name — Dutch
# "Stichting Mindfit, Thubble, Deventer, The Netherlands" carries no English
# organisational word but is unmistakably an address. Requires a place-like
# final segment, which is what separates it from a comma-bearing caption
# ("Baseline characteristics, by group, for all participants").
_AFFILIATION_PLACE_TAIL_RE = re.compile(
    r"^(?:[A-Z\u00c0-\u00de][\w.'\u2019-]*)(?:\s+[A-Za-z\u00c0-\u00ff][\w.'\u2019-]*){0,3}$"
)


def _looks_like_affiliation(value: str) -> bool:
    """Positive evidence that *value* is an institution, not caption prose."""
    if _AFFILIATION_ORG_EVIDENCE_RE.search(value):
        return True
    segments = [segment.strip() for segment in value.split(",")]
    return len(segments) >= 3 and bool(_AFFILIATION_PLACE_TAIL_RE.match(segments[-1]))


def _front_page(df) -> int:
    """Lowest page number present in ``df`` — the paper's front page here.

    Page numbers are absolute so export provenance stays honest, so under page
    slicing (``--pages 5-12``, ``chew(pages=…)``, serve ``start_page``) nothing
    carries page 1. A literal ``== 1`` then fails closed (byline slices come
    out empty, affiliation reconcilers no-op) while ``> 1`` fails *open* —
    front-matter rows become affiliation candidates. Anchoring on the minimum
    keeps both honest; unsliced it is 1, so the default is unchanged.
    """
    if "page_number" not in getattr(df, "columns", ()):
        return 1
    pages = df["page_number"].dropna()
    return int(pages.min()) if not pages.empty else 1


def _selected_block_candidates(
    resolution: FrontMatterResolution,
) -> tuple[FrontMatterCandidate, ...]:
    if resolution.selected_block_id is None:
        return ()
    block = next(
        (block for block in resolution.blocks if block.block_id == resolution.selected_block_id),
        None,
    )
    if block is None:
        return ()
    candidate_by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    return tuple(
        candidate_by_id[candidate_id]
        for candidate_id in block.candidate_ids
        if candidate_id in candidate_by_id
    )


def render_block_context(
    resolution: FrontMatterResolution,
    *,
    roles=None,
) -> str:
    """Render authoritative text from the selected front-matter block only."""

    wanted = frozenset(roles) if roles is not None else None
    return "\n".join(
        candidate.raw_text.strip()
        for candidate in _selected_block_candidates(resolution)
        if candidate.raw_text.strip() and (wanted is None or not candidate.roles.isdisjoint(wanted))
    )


def render_author_context(
    resolution: FrontMatterResolution,
    *,
    full_text: str,
) -> str:
    """Render a contiguous selected-record author zone without losing evidence."""

    selected = _selected_block_candidates(resolution)
    byline_indices = [
        index for index, candidate in enumerate(selected) if "byline" in candidate.roles
    ]
    if not byline_indices:
        return full_text
    title_indices = [
        index for index, candidate in enumerate(selected) if "title" in candidate.roles
    ]
    if len(title_indices) != 1:
        return full_text

    title_index = title_indices[0]
    # A title can resemble a byline lexically. When it is the only byline
    # candidate, narrowing to it hides actual authors in the surrounding rows.
    if byline_indices == [title_index]:
        return full_text
    first_byline = byline_indices[0]
    last_byline = byline_indices[-1]
    if title_index > first_byline:
        return full_text
    if any(
        not candidate.roles.isdisjoint({"abstract", "doi"})
        for candidate in selected[title_index + 1 : last_byline + 1]
    ):
        return full_text

    start = title_index if "byline" in selected[title_index].roles else title_index + 1
    end = next(
        (
            index
            for index in range(last_byline + 1, len(selected))
            if not selected[index].roles.isdisjoint({"abstract", "title"})
        ),
        len(selected),
    )
    author_zone = "\n".join(
        candidate.raw_text.strip()
        for candidate in selected[start:end]
        if candidate.raw_text.strip()
    )
    return author_zone or full_text


def author_table_context(contents: PaperContents, resolution: FrontMatterResolution | None) -> str:
    """Read explicitly labelled author-information tables on a single-record front page.

    These boxes are excluded from the sentence stream. Page membership alone
    cannot assign one on a multi-record page, so that case stays out of scope.
    """
    if resolution is None or resolution.selected_block_id is None:
        return ""
    candidates = _selected_block_candidates(resolution)
    pages = [candidate.page for candidate in candidates if candidate.page is not None]
    if not pages:
        return ""
    front_page = min(pages)
    by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    for block in resolution.blocks:
        if block.block_id == resolution.selected_block_id:
            continue
        other_pages = {
            by_id[candidate_id].page
            for candidate_id in block.candidate_ids
            if candidate_id in by_id
        }
        if not other_pages or None in other_pages or front_page in other_pages:
            return ""

    labels = {
        "author information",
        "author details",
        "info penulis",
        "informasi penulis",
    }
    texts = []
    for table in getattr(contents, "tables", None) or ():
        if table.page_number != front_page or table.caption:
            continue
        rows = table.contents
        if not any(" ".join(cell.casefold().split()) in labels for row in rows[:2] for cell in row):
            continue
        text = "\n".join(" | ".join(cell.strip() for cell in row if cell.strip()) for row in rows)
        if text and len(text) <= 12_000:
            texts.append(text)
    return "\n".join(texts)


@dataclass(frozen=True)
class BylineGroup:
    """Selected-block byline evidence used to assess extracted authors."""

    candidate_ids: tuple[str, ...]
    text_ids: tuple[int, ...]
    raw_texts: tuple[str, ...]
    missing_inference_raw_texts: tuple[str, ...]
    title_candidate_ids: tuple[str, ...]
    title_candidate_indices: tuple[int, ...]
    title_bounded_raw_texts: tuple[str | None, ...]
    # Composite title+byline records, verbatim — recovery evidence ONLY. They
    # are deliberately absent from ``missing_inference_raw_texts``: see
    # ``_composite_recovery_text``.
    composite_recovery_raw_texts: tuple[str, ...] = ()


def build_byline_group(resolution: FrontMatterResolution) -> BylineGroup:
    """Collect byline candidates from the selected block, including headings."""

    candidates = tuple(
        candidate
        for candidate in _selected_block_candidates(resolution)
        if "byline" in candidate.roles
    )
    return BylineGroup(
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        text_ids=tuple(text_id for candidate in candidates for text_id in candidate.text_ids),
        raw_texts=tuple(candidate.raw_text for candidate in candidates),
        missing_inference_raw_texts=tuple(
            bounded
            for candidate in candidates
            if (bounded := _missing_inference_text(candidate.raw_text, candidate.roles))
        ),
        title_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates if "title" in candidate.roles
        ),
        title_candidate_indices=tuple(
            index for index, candidate in enumerate(candidates) if "title" in candidate.roles
        ),
        title_bounded_raw_texts=tuple(
            _bounded_byline_text(candidate.raw_text, candidate.roles)
            if "title" in candidate.roles
            else None
            for candidate in candidates
        ),
        composite_recovery_raw_texts=tuple(
            composite
            for candidate in candidates
            if (composite := _composite_recovery_text(candidate.raw_text, candidate.roles))
        ),
    )


_NAME_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
# A translator credit printed under the byline ("Traducido del inglés por …",
# "Translated by …", "Übersetzt von …") names someone who is not an author.
# Folded tokens, as ``_name_tokens`` emits them.
_TRANSLATOR_CREDIT_WORDS = frozenset(
    {
        "translated",
        "translation",
        "translator",
        "traduccion",
        "traducido",
        "traducida",
        "traductor",
        "traductora",
        "traduit",
        "traduite",
        "traduction",
        "traducteur",
        "traductrice",
        "ubersetzt",
        "ubersetzung",
        "ubersetzer",
        "ubersetzerin",
        "traducao",
        "traduzido",
        "traduzida",
        "tradutor",
        "tradutora",
        "vertaald",
        "vertaling",
        "vertaler",
        "tradotto",
        "tradotta",
        "traduzione",
        "traduttore",
        "traduttrice",
    }
)
# The word that hands the credit to the translator's name: "… by", "… de",
# "… por", "… par", "… von", "… door", "… da", "… di".
_TRANSLATOR_CREDIT_PREPOSITIONS = frozenset(
    {"by", "de", "del", "por", "par", "von", "door", "da", "di", "dal"}
)
# How far back from a name the credit word may sit: "translated from the
# original French by" is six tokens.
_TRANSLATOR_CREDIT_WINDOW = 8
_BYLINE_SUFFIX_STOP_WORDS = _TRANSLATOR_CREDIT_WORDS | frozenset(
    {
        "abstract",
        "affiliation",
        "college",
        "correspondence",
        "corresponding",
        "department",
        "doi",
        "email",
        "faculty",
        "hospital",
        "institute",
        "laboratory",
        "health",
        "national",
        "nhs",
        "orcid",
        "school",
        "service",
        "university",
    }
)
_NAME_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "`": "'",
        "ʼ": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)
# A spacing diacritic, optionally followed by the single space the PDF inserts
# to give it width. Both go: the accent belongs to the letter before it.
# U+0060 is absent on purpose — `_NAME_PUNCTUATION_TRANSLATION` has already
# turned it into an apostrophe by the time this runs.
_SPACING_ACCENT_RE = re.compile("[¨¯´ˆ-˛˝] ?")
# Letters NFKD leaves alone because their accent is not a separable mark. Only
# the ones a broken text layer actually emits for an accented Latin letter.
_LATIN_BASE_TRANSLATION = str.maketrans(
    {
        "ı": "i",
        "ȷ": "j",
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ħ": "h",
        "ŧ": "t",
    }
)
_CONSORTIUM_NAME_RE = re.compile(
    r"\b(?:association|collaboration|collective|committee|consortium|group|"
    r"investigators?|network|society|study\s+group|team)\b",
    re.IGNORECASE,
)
_NUMBERED_AFFILIATION_BOUNDARY_RE = re.compile(r"\s+\d+[.)]\s+")
_ABSTRACT_BYLINE_BOUNDARY_RE = re.compile(r"\babstract\b", re.IGNORECASE)
_EMAIL_FRAGMENT_RE = re.compile(r"\S*@\S*")


def _strip_email_fragments(value: str) -> str:
    """Remove an email address from a name field.

    A correspondence block can cause email addresses to be emitted as author names. Addresses belong in the email field: an address-only name becomes empty, while a real name joined to an address keeps its name portion.
    """

    if "@" not in value:
        return value
    return " ".join(_EMAIL_FRAGMENT_RE.sub(" ", value).split()).strip(" ,;")


# Printed affiliation markers ride the name as superscripts; NFKC folds them
# into trailing ASCII digits ("Kian Jafari¹" -> "Jafari1"). Strip only digits
# directly attached to letters so standalone numbers stay untouched.
_ATTACHED_MARKER_DIGITS_RE = re.compile(r"(?<=[^\W\d_])\d+", re.UNICODE)
_ATTACHED_MARKER_LIST_RE = re.compile(
    r"(?<=[^\W\d_])[a-z](?:\s*,\s*[a-z])+(?![^\W\d_])", re.UNICODE
)
# Letter superscripts fold the same way ("Htin Aung^A" -> "AungA") and leave no
# punctuation to tokenize on. Strip a word-final capital riding a lowercase stem
# only: requiring the preceding letter to be lowercase keeps an all-caps printed
# surname ("NIKOLAI DINEV") whole, and the word-final lookahead keeps an interior
# capital ("McDonald") whole. A word-final capital after a lowercase letter
# matches no given/family value across the 193 gold files.
_ATTACHED_MARKER_LETTER_RE = re.compile(r"(?<=[^\W\d_])([^\W\d_])(?![^\W\d_])", re.UNICODE)


def _drop_letter_marker(match: re.Match[str]) -> str:
    letter = match.group(1)
    if not letter.isupper():
        return letter
    return "" if match.string[match.start(1) - 1].islower() else letter


def _strip_attached_markers(value: str) -> str:
    """Remove printed affiliation markers glued to a name by OCR/NFKC folding."""

    return _ATTACHED_MARKER_LETTER_RE.sub(
        _drop_letter_marker,
        _ATTACHED_MARKER_DIGITS_RE.sub("", _ATTACHED_MARKER_LIST_RE.sub("", value)),
    )


def _fold_broken_diacritics(value: str) -> str:
    """Repair PDF text layers that render an accent as a separate glyph.

    Type-1 fonts routinely encode ``í`` as dotless-i plus a *spacing* acute, and
    ``Á`` as ``A`` plus a spacing acute followed by a space — so a byline that
    prints ``María Ángeles`` extracts as ``Marı´a A´ ngeles``. NFKC cannot undo
    it: U+00B4 is a compatibility character whose decomposition is *space* plus
    a combining acute, so the accent binds to the space rather than to the
    letter before it.

    Broken font encodings can prevent correct accented names from matching source text. Normalize that encoding damage before grounding so a correctly extracted author list is not discarded in favor of mangled recovery text.

    Folding is comparison-only. Emitted values keep their original spelling, and
    both sides of every comparison run through here, so it stays symmetric.

    The spacing accents have to go **before** NFKC, not after: NFKC is precisely
    what turns U+00B4 into space-plus-combining-acute, so running it first
    destroys the evidence this needs.
    """
    stripped = _SPACING_ACCENT_RE.sub("", value)
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", stripped))
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) not in {"Mn", "Sk"}
    )
    return without_marks.translate(_LATIN_BASE_TRANSLATION)


# Surname prefixes printed with an inner capital ("McDonald", "DeKay",
# "VanderWeele", "AlQahtani"). Any other lower-to-upper step inside a word is
# two words whose space the PDF text layer never emitted: it positions the
# next word instead of printing a space glyph, so "Selin Deniz Aksoy"
# extracts as "Selin DenizAksoy".
_CASE_JOIN_PREFIXES = frozenset(
    {
        "abd",
        "abdel",
        "abdul",
        "abu",
        "al",
        "ben",
        "bin",
        "da",
        "dal",
        "das",
        "de",
        "del",
        "della",
        "des",
        "di",
        "do",
        "dos",
        "du",
        "el",
        "fitz",
        "ibn",
        "la",
        "le",
        "lo",
        "mac",
        "mc",
        "ni",
        "st",
        "te",
        "ten",
        "ter",
        "van",
        "vande",
        "vanden",
        "vander",
        "von",
    }
)
_LETTER_RUN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _case_join_words(word: str) -> list[str]:
    """Split *word* at every lower-to-upper step not owned by a surname prefix."""

    words: list[str] = []
    start = 0
    for index in range(1, len(word) - 1):
        if not (word[index - 1].islower() and word[index].isupper() and word[index + 1].islower()):
            continue
        if word[start:index].casefold() in _CASE_JOIN_PREFIXES:
            continue
        words.append(word[start:index])
        start = index
    words.append(word[start:])
    return words


def _split_case_joins(value: str) -> str:
    return _LETTER_RUN_RE.sub(lambda match: " ".join(_case_join_words(match.group(0))), value)


def _name_tokens(value: str) -> tuple[str, ...]:
    # Marker stripping happens here, before casefolding, so that every name
    # comparison in this module is symmetric: printed byline text and extracted
    # author values are folded the same way, whichever side carries the marker.
    # Case joins split after it, once "Aksoy1" has lost its marker.
    normalized = _split_case_joins(
        _strip_attached_markers(
            _fold_broken_diacritics(value).translate(_NAME_PUNCTUATION_TRANSLATION)
        )
    )
    return tuple(token.casefold() for token in _NAME_TOKEN_RE.findall(normalized))


def _bounded_byline_text(value: str, roles: frozenset[str]) -> str | None:
    if roles.issubset({"byline", "heading"}):
        return value

    boundary_matches = []
    if "abstract" in roles:
        boundary_matches.extend(_ABSTRACT_BYLINE_BOUNDARY_RE.finditer(value))
    if "affiliation" in roles:
        boundary_matches.extend(_NUMBERED_AFFILIATION_BOUNDARY_RE.finditer(value))
        boundary_matches.extend(AFFILIATION_MARKER_RE.finditer(value))
    if not boundary_matches:
        return None

    boundary = min(boundary_matches, key=lambda match: match.start())
    return value[: boundary.start()].strip(" ,;")


def _missing_inference_text(value: str, roles: frozenset[str]) -> str | None:
    if "title" in roles:
        return None
    return _bounded_byline_text(value, roles)


# A composite record must split into at least this many name-like entries before
# it counts as byline evidence. One entry is indistinguishable from the title
# chunk itself; two or more is a printed author list.
_COMPOSITE_MIN_NAME_ENTRIES = 2


def _composite_recovery_text(value: str, roles: frozenset[str]) -> str | None:
    """Recovery evidence from a record carrying BOTH the title and byline roles.

    _missing_inference_text must refuse composite title/byline records because the title would be mistaken for a missing author. Recovery still needs their printed byline evidence: reusing the same wide context as the failed primary call provides no focused correction.

    Returned verbatim rather than title-stripped, because the strip cannot be
    positioned safely — composites print the byline before the title as often as
    after it, and a mis-positioned cut deletes the first or last author. Handing
    the model the whole record instead makes this "pick the authors out of this
    line", and the recovered list is grounded against this same text, so title
    prose can neither be invented into a name nor survive as one.
    """

    if "title" not in roles:
        return None
    bounded = _bounded_byline_text(value, roles) or value
    entries = [entry for entry in _split_byline_entries(bounded) if _has_name_like_residual(entry)]
    if len(entries) < _COMPOSITE_MIN_NAME_ENTRIES:
        return None
    return bounded.strip()


def _author_token_variants(author: PaperAuthor) -> tuple[tuple[str, ...], ...]:
    given = (author.given or "").strip()
    family = (author.family or "").strip()
    values = []
    if given and family:
        values.extend((f"{given} {family}", f"{family} {given}"))
    elif given:
        values.append(given)
    elif family and "organization" in (author.role or []):
        values.append(given or family)
    return tuple(dict.fromkeys(tokens for value in values if (tokens := _name_tokens(value))))


def _find_token_sequences(
    haystack: tuple[str, ...],
    needle: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    if not needle or len(needle) > len(haystack):
        return ()
    matches = []
    for start in range(len(haystack) - len(needle) + 1):
        end = start + len(needle)
        if haystack[start:end] == needle:
            matches.append((start, end))
    return tuple(matches)


def _locate_author(
    author: PaperAuthor,
    byline_tokens: tuple[tuple[str, ...], ...],
    byline_texts: tuple[str, ...],
    *,
    after: tuple[int, int, int] | None,
    used: set[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    matches: list[tuple[int, int, int]] = []
    for candidate_index, tokens in enumerate(byline_tokens):
        for variant in _author_token_variants(author):
            if "organization" in (author.role or []):
                entries = tuple(
                    _name_tokens(entry)
                    for entry in re.split(
                        r"\s*(?:[;,&]|\band\b)\s*",
                        byline_texts[candidate_index],
                        flags=re.IGNORECASE,
                    )
                    if entry.strip()
                )
                if variant not in entries:
                    continue
            for match in _find_token_sequences(tokens, variant):
                location = (candidate_index, *match)
                overlaps_previous = bool(
                    after is not None
                    and (
                        candidate_index < after[0]
                        or (candidate_index == after[0] and match[0] < after[2])
                    )
                )
                if location in used or overlaps_previous:
                    continue
                matches.append(location)
    return min(matches) if matches else None


def grounded_authors_in_context(
    authors: list[PaperAuthor],
    context: str,
) -> tuple[list[PaperAuthor], list[PaperAuthor]]:
    """Split recovered authors into context-grounded and ungrounded lists.

    One token stream, one-to-one span reservation: a printed name grounds at
    most one recovered author, and input order is preserved on both sides.
    """

    context_tokens = (_name_tokens(context),)
    context_texts = (context,)
    used: set[tuple[int, int, int]] = set()
    grounded: list[PaperAuthor] = []
    rejected: list[PaperAuthor] = []
    for author in authors:
        location = _locate_author(
            author,
            context_tokens,
            context_texts,
            after=None,
            used=used,
        )
        if location is None:
            rejected.append(author)
        else:
            used.add(location)
            grounded.append(author)
    return grounded, rejected


def _follows_translator_credit(preceding: tuple[str, ...]) -> bool:
    window = preceding[-_TRANSLATOR_CREDIT_WINDOW:]
    return bool(
        window
        and window[-1] in _TRANSLATOR_CREDIT_PREPOSITIONS
        and not _TRANSLATOR_CREDIT_WORDS.isdisjoint(window[:-1])
    )


def _is_translator_credit(author: PaperAuthor, context_tokens: tuple[str, ...]) -> bool:
    """Whether *author* is printed only as the name a translator credit hands off to.

    A translator who is also an author is printed in the byline as well, and
    that occurrence keeps them.
    """

    starts = [
        start
        for variant in _author_token_variants(author)
        for start, _ in _find_token_sequences(context_tokens, variant)
    ]
    return bool(starts) and all(
        _follows_translator_credit(context_tokens[:start]) for start in starts
    )


# Initials the text layer glued to the surname after them: "Kerem B.Yalcin"
# comes back as given "Kerem", family "B.Yalcin".
_GLUED_INITIALS_RE = re.compile(r"((?:[^\W\d_]\.)+)(?=[^\W\d_]{2})", re.UNICODE)


def _unglue_leading_initials(given: str, family: str) -> tuple[str, str] | None:
    match = _GLUED_INITIALS_RE.match(family)
    if match is None or not family[match.end()].isupper():
        return None
    initials = match.group(1)
    if not all(char.isupper() for char in initials if char != "."):
        return None
    return f"{given} {initials}".strip(), family[match.end() :]


# Email addresses, URLs and their labels in a byline row are Latin whatever
# script the names are printed in.
_ADDRESS_FRAGMENT_RE = re.compile(r"\S*(?:@|://|www\.)\S*|\b(?:e-?mail|orcid)\b", re.IGNORECASE)
# A script with less of the byline's letters than this is not the script the
# byline prints its names in.
_BYLINE_SCRIPT_MIN_SHARE = 0.2


def _letter_scripts(value: str) -> dict[str, int]:
    """Count the letters of *value* by Unicode script ("LATIN", "CYRILLIC", "CJK")."""

    counts: dict[str, int] = {}
    for char in value:
        if not char.isalpha():
            continue
        script = unicodedata.name(char, "").split(" ", 1)[0]
        if script and script != "MODIFIER":
            counts[script] = counts.get(script, 0) + 1
    return counts


def _has_name_like_residual(tokens: list[str] | tuple[str, ...]) -> bool:
    residual = list(tokens)
    while residual and (residual[0] in _BYLINE_CONJUNCTION_TOKENS or residual[0].isdigit()):
        residual.pop(0)
    bounded: list[str] = []
    for token in residual:
        if token in _BYLINE_SUFFIX_STOP_WORDS:
            break
        if not token.isdigit():
            bounded.append(token)
    return len(bounded) >= 2


def _has_unmatched_author_evidence(
    byline_tokens: tuple[tuple[str, ...], ...],
    locations: list[tuple[int, int, int]],
) -> bool:
    for left, right in zip(locations, locations[1:], strict=False):
        left_candidate, _, left_end = left
        right_candidate, right_start, _ = right
        if left_candidate == right_candidate:
            gap = list(byline_tokens[left_candidate][left_end:right_start])
        else:
            gap = list(byline_tokens[left_candidate][left_end:])
            for middle in byline_tokens[left_candidate + 1 : right_candidate]:
                gap.extend(middle)
            gap.extend(byline_tokens[right_candidate][:right_start])
        if _has_name_like_residual(gap):
            return True

    candidate_index, _, end = locations[-1]
    suffix = list(byline_tokens[candidate_index][end:])
    for later in byline_tokens[candidate_index + 1 :]:
        suffix.extend(later)
    return _has_name_like_residual(suffix)


# The word a byline joins its last two names with: "and", French "et", Dutch
# "en", German "und", Danish/Norwegian "og", Swedish "och", Indonesian "dan".
# The new words match in lower or upper case only, so a capitalised given name
# ("Dan", "En") is never taken for one.
_BYLINE_CONJUNCTION_RE = re.compile(
    r"\s*(?:;|&|\b(?i:and)\b|\b(?:et|en|und|og|och|dan|ET|EN|UND|OG|OCH)\b)\s*"
)
# Spanish "y", Portuguese/Italian "e" and Catalan/Polish "i" also join one
# person's two surnames ("Ramón y Cajal", "Puig i Cadafalch"). They split only
# in lower case (an upper-case letter is an initial) and only between two
# names of at least two words each.
_BYLINE_SURNAME_JOINER_RE = re.compile(r"(\s+[yei]\s+)")
_BYLINE_CONJUNCTION_TOKENS = frozenset(
    {"and", "et", "en", "und", "og", "och", "dan", "y", "e", "i"}
)


def _split_surname_joiners(chunk: str) -> list[str]:
    pieces = _BYLINE_SURNAME_JOINER_RE.split(chunk)
    merged = [pieces[0]]
    for joiner, piece in zip(pieces[1::2], pieces[2::2], strict=True):
        left_name = merged[-1].rsplit(",", 1)[-1]
        right_name = piece.split(",", 1)[0]
        if len(_name_tokens(left_name)) >= 2 and len(_name_tokens(right_name)) >= 2:
            merged.append(piece)
        else:
            merged[-1] += joiner + piece
    return merged


def _split_byline_entries(value: str) -> tuple[tuple[str, ...], ...]:
    chunks = [
        piece.strip()
        for chunk in _BYLINE_CONJUNCTION_RE.split(value)
        for piece in _split_surname_joiners(chunk)
        if piece.strip()
    ]
    entries: list[tuple[str, ...]] = []
    for chunk in chunks:
        comma_parts = [part.strip() for part in chunk.split(",") if part.strip()]
        comma_tokens = [_name_tokens(part) for part in comma_parts]
        if len(comma_parts) >= 2 and all(len(tokens) >= 2 for tokens in comma_tokens):
            entries.extend(comma_tokens)
        else:
            entries.append(_name_tokens(chunk))
    return tuple(entry for entry in entries if entry)


def _expected_byline_entries(byline: BylineGroup) -> tuple[tuple[str, ...], ...]:
    entries: list[tuple[str, ...]] = []
    for text in byline.missing_inference_raw_texts:
        for entry in _split_byline_entries(text):
            if _has_name_like_residual(entry):
                entries.append(entry)
    return tuple(entries)


def byline_recovery_context(resolution: FrontMatterResolution | None) -> str | None:
    """The selected byline verbatim, to reground an empty-author recovery.

    The retry otherwise re-asks with the same wide evidence the failing call
    already saw. But when the selected block's byline parses into at least one
    name-like entry, bibr already knows the authors are there — that entry count
    is exactly what ``VAL_AUTHOR_MISSING`` reports. Handing the model just the
    byline turns "find the authors" into "split these names", and narrows
    grounding to the byline itself so a name from elsewhere on the page cannot
    slip in.

    A composite title+byline row contributes its whole record instead (see
    ``_composite_recovery_text``). Returns None when neither source parses into
    name-like entries, leaving the caller's existing context untouched.
    """
    if resolution is None:
        return None
    byline = build_byline_group(resolution)
    # Bounded raw text, not _name_tokens: tokens are casefolded, and the model
    # should see the printed form.
    texts: list[str] = []
    if _expected_byline_entries(byline):
        texts.extend(text.strip() for text in byline.missing_inference_raw_texts if text.strip())
    texts.extend(text.strip() for text in byline.composite_recovery_raw_texts if text.strip())
    if not texts:
        return None
    return "\n".join(dict.fromkeys(texts))


def _entry_matching_author(entry: tuple[str, ...], author: PaperAuthor) -> bool:
    variants = _author_token_variants(author)
    if "organization" in (author.role or []):
        return entry in variants
    return any(_find_token_sequences(entry, variant) for variant in variants)


def _anchored_title_missing_count(
    byline: BylineGroup,
    byline_tokens: tuple[tuple[str, ...], ...],
    grounded_locations: list[tuple[int, int, int]],
) -> int:
    missing_count = 0
    for candidate_index in byline.title_candidate_indices:
        candidate_locations = sorted(
            location for location in grounded_locations if location[0] == candidate_index
        )
        if not candidate_locations:
            continue

        for left, right in zip(candidate_locations, candidate_locations[1:], strict=False):
            if _has_name_like_residual(byline_tokens[candidate_index][left[2] : right[1]]):
                missing_count += 1

        bounded_text = byline.title_bounded_raw_texts[candidate_index]
        if bounded_text is None:
            continue
        bounded_tokens = _name_tokens(bounded_text)
        last_end = candidate_locations[-1][2]
        if last_end <= len(bounded_tokens) and _has_name_like_residual(bounded_tokens[last_end:]):
            missing_count += 1
    return missing_count


# A rewrite worth repairing is a near-copy: same title, a word or two altered.
# Below this the two strings are different titles and the printed one must not
# be substituted.
_TITLE_GROUNDING_MIN_RATIO = 0.90
# The printed form must not be materially shorter than what was extracted. A
# correct title assembled from a title + subtitle region pair can score well
# against the title region alone; requiring near-equal length keeps the repair
# from truncating it.
_TITLE_GROUNDING_LENGTH_BAND = (0.95, 1.10)
# Short strings hit high similarity ratios by accident.
_TITLE_GROUNDING_MIN_LENGTH = 12
_GROUNDING_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_for_grounding(value: str) -> str:
    """Casefold, strip punctuation and collapse whitespace for verbatim tests."""

    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_GROUNDING_PUNCT_RE.sub(" ", folded).split())


def _grounding_sources(
    candidates: tuple[FrontMatterCandidate, ...],
    printed_rows: Mapping[int, str] | None,
) -> list[str]:
    """Printed strings the repair may substitute, longest granularity first.

    Candidates are paragraph-granular, so a title may be joined to a preceding affiliation line and fall outside the substitution length band. The rows behind each candidate restore line granularity.

    Only rows a candidate already owns are eligible: the same paper prints its
    own title inside a "how to cite" line and inside reference entries, and
    those must never become a substitution source.
    """

    sources = [candidate.raw_text for candidate in candidates]
    if printed_rows:
        for candidate in candidates:
            sources.extend(
                printed_rows[text_id] for text_id in candidate.text_ids if text_id in printed_rows
            )
    return sources


# A parenthetical opening a printed title row: "(Rural) Clinics as layered civic
# organizations", or fused as in "(Re)thinking ...". Models read it as an
# annotation and return only the rest, which is still printed verbatim.
_LEADING_PARENTHETICAL_RE = re.compile(r"\(([^()]{1,40})\)\s*")
# List or section numbering: "(1)", "(2.1)", "(b)", "(iv)". A bare year is not
# title text either: a citation line wrapped after its author list starts with
# "(2020)" and then the title.
_ENUMERATOR_RE = re.compile(r"\d+(?:\.\d+)*|[a-z]|x{0,3}(?:ix|iv|v?i{0,3})", re.IGNORECASE)


def _labels_title_row(parenthetical: str) -> bool:
    """Whether a leading parenthetical labels the title row instead of belonging to it.

    Numbering and article-type labels ("(Review)", "(Original Article)") are
    furniture the model is right to drop. Anything else is printed title text.
    """

    from bibr.structure.paper_classifier import PAPER_TYPE_LABELS

    if _ENUMERATOR_RE.fullmatch(parenthetical.strip()):
        return True
    labels = (*PAPER_TYPE_LABELS, *EXACT_GENERIC_ARTICLE_LABELS, *FRONT_MATTER_FURNITURE_LABELS)
    return _normalize_for_grounding(parenthetical) in {
        _normalize_for_grounding(label) for label in labels
    }


def _restore_leading_parenthetical(
    title: str,
    resolution: FrontMatterResolution,
    printed_rows: Mapping[int, str] | None,
) -> tuple[str, str] | None:
    """Put back a leading parenthetical the model dropped from the printed title.

    The truncated title still occurs verbatim on the page, so the verbatim test
    alone accepts it. Only the selected record's title rows are eligible, and
    the model title must start exactly where a row continues after its
    parenthetical; a subtitle printed on another row may follow. Returns the
    restored title and the candidate it came from, or None when no row fits or
    rows disagree on the parenthetical.
    """

    normalized = _normalize_for_grounding(title)
    restorations: dict[str, tuple[str, str]] = {}
    for candidate in _selected_block_candidates(resolution):
        if "title" not in candidate.roles:
            continue
        for source in _grounding_sources((candidate,), printed_rows):
            printed = source.strip()
            match = _LEADING_PARENTHETICAL_RE.match(printed)
            if match is None or _labels_title_row(match.group(1)):
                continue
            rest = _normalize_for_grounding(printed[match.end() :])
            if len(rest) < _TITLE_GROUNDING_MIN_LENGTH or not (
                normalized == rest or normalized.startswith(f"{rest} ")
            ):
                continue
            parenthetical = match.group(0).strip()
            # "(Rural) Clinics" keeps its space; "(Re)thinking" stays fused.
            separator = " " if match.group(0) != parenthetical else ""
            restorations.setdefault(
                parenthetical.casefold(),
                (f"{parenthetical}{separator}{title.strip()}", candidate.candidate_id),
            )
    return next(iter(restorations.values())) if len(restorations) == 1 else None


# A short label set off from the rest of a printed title row by a colon, full
# stop or dash: "Research Report: ...", "Case report. ...", "FIELD NOTES – ...".
# A full stop counts only before a space, so "e.g." and decimals do not split a
# row.
_TITLE_LABEL_SEPARATOR_RE = re.compile(r"\s*[:.]\s+|\s*[–—]\s*|\s+-\s+")
# The same label printed as a row of its own above the title: "Review:". Only a
# colon marks such a row as continuing into the next one; an article-type kicker
# ("ORIGINAL PAPER", "Case report") prints no colon.
_TITLE_LABEL_ROW_RE = re.compile(r"(?P<label>[^:]+?)\s*:")
_TITLE_LABEL_MAX_CHARS = 40
_TITLE_LABEL_MAX_WORDS = 5
# A comma, bracket, quote, slash, year or "et al." makes a prefix a citation or
# a running head ("Smith, J. (2020).", "Zhao et al.:"), and so does a word that
# introduces one ("To cite this version:", "Cómo citar:", "Для цитирования:").
_NOT_A_TITLE_LABEL_RE = re.compile(
    r"[,;()\[\]{}\"“”„‘’«»|/]|\d{4}|\bet\.?\s+al\b"
    r"|\b(?:cite|citation|citer|citar|citare|zitier\w*|cytow\w*|цитир\w*|цитат\w*)\b",
    re.IGNORECASE,
)
# Field labels name the row instead of belonging to it: "Title: ...".
_TITLE_FIELD_LABELS = frozenset(
    {
        "title",
        "article title",
        "paper title",
        "full title",
        "short title",
        "running title",
        "running head",
        "titel",
        "titre",
        "titolo",
        "título",
        "tytuł",
        "název",
        "название",
        "abstract",
        "summary",
        "highlights",
        "key points",
        "keywords",
        "key words",
        "doi",
        "source",
        "article type",
        "special issue",
    }
)


def _is_title_label(label: str) -> bool:
    """Whether a short prefix of the printed title row is title text.

    The title is what the heading prints, so a label printed inside the title
    heading ("Research Report:", "Case report.", "Opinion:") belongs to it even
    when it names the article type. Numbering, citation and running-head
    prefixes, field labels and page furniture ("Open access") do not.
    """

    stripped = label.strip()
    if (
        not stripped
        or len(stripped) > _TITLE_LABEL_MAX_CHARS
        or len(stripped.split()) > _TITLE_LABEL_MAX_WORDS
        or not any(character.isalpha() for character in stripped)
        or _NOT_A_TITLE_LABEL_RE.search(stripped)
        or _ENUMERATOR_RE.fullmatch(stripped)
    ):
        return False
    normalized = _normalize_for_grounding(stripped)
    return normalized not in _TITLE_FIELD_LABELS and normalized not in FRONT_MATTER_FURNITURE_LABELS


def _continues_into_title(normalized_title: str, printed: str) -> bool:
    """Whether the model title is exactly *printed*, or *printed* then a subtitle row."""

    rest = _normalize_for_grounding(printed)
    return len(rest) >= _TITLE_GROUNDING_MIN_LENGTH and (
        normalized_title == rest or normalized_title.startswith(f"{rest} ")
    )


def _restore_leading_label(
    title: str,
    resolution: FrontMatterResolution,
    printed_rows: Mapping[int, str] | None,
) -> tuple[str, str] | None:
    """Put back a leading label the model dropped from the printed title.

    Two printed shapes: the label opens the title row ("Research Report: ...")
    or is a row of its own ending in a colon, directly above the row where the
    model title starts ("Review:" over the rest). Only the selected record's
    title rows can supply a label. Abstains when rows
    disagree on the label, or when another title row of the record prints the
    title without it: then the label may be a kicker the model was right to
    drop. Returns the restored title and the candidate the label came from.
    """

    normalized = _normalize_for_grounding(title)
    selected = _selected_block_candidates(resolution)
    restorations: dict[str, tuple[str, str]] = {}
    label_sources: set[str] = set()
    for index, candidate in enumerate(selected):
        if "title" not in candidate.roles:
            continue
        for source in _grounding_sources((candidate,), printed_rows):
            printed = source.strip()
            for separator in _TITLE_LABEL_SEPARATOR_RE.finditer(printed):
                if separator.start() > _TITLE_LABEL_MAX_CHARS:
                    break
                if not _continues_into_title(normalized, printed[separator.end() :]):
                    continue
                if _is_title_label(printed[: separator.start()]):
                    label = printed[: separator.end()].strip()
                    # A dash fused to the next word stays fused.
                    space = " " if printed[separator.end() - 1].isspace() else ""
                    restorations.setdefault(
                        _normalize_for_grounding(label),
                        (f"{label}{space}{title.strip()}", candidate.candidate_id),
                    )
                    label_sources.add(candidate.candidate_id)
                break
        row = _TITLE_LABEL_ROW_RE.fullmatch(candidate.raw_text.strip())
        if (
            row is not None
            and index + 1 < len(selected)
            and _is_title_label(row.group("label"))
            and _continues_into_title(normalized, selected[index + 1].raw_text)
        ):
            label = candidate.raw_text.strip()
            restorations.setdefault(
                _normalize_for_grounding(label),
                (f"{label} {title.strip()}", candidate.candidate_id),
            )
            label_sources.update((candidate.candidate_id, selected[index + 1].candidate_id))
    if len(restorations) != 1:
        return None
    for candidate in selected:
        if "title" not in candidate.roles or candidate.candidate_id in label_sources:
            continue
        if any(
            _normalize_for_grounding(source) == normalized
            for source in _grounding_sources((candidate,), printed_rows)
        ):
            return None
    return next(iter(restorations.values()))


def ground_title_to_printed_text(
    title: str,
    resolution: FrontMatterResolution | None,
    printed_text: str,
    *,
    printed_rows: Mapping[int, str] | None = None,
) -> tuple[str, ValidationIssue | None]:
    """Prefer the printed title over a model's silent rewrite of it.

    The source of truth is the printed title, including grammatical errors. A model may silently correct a word to one absent from the page; grounding checks the extracted title against the actual front-matter text.

    Returns the title to use and an optional issue. The substitution is
    deliberately narrow — it fires only on a near-copy of a single printed row —
    because the model legitimately joins a title split across regions, and a
    loose rule would truncate those. Everything ungrounded that is not a
    near-copy is reported and left alone. The exceptions are a leading
    parenthetical such as "(Rural)" and a leading label such as "Research
    Report:", which a model drops as if they were annotations: the rest still
    occurs verbatim, so they are restored from the selected title rows unless
    they are numbering, a field label or page furniture (a parenthetical
    article type such as "(Review)" stays dropped too).
    """

    normalized = _normalize_for_grounding(title)
    if len(normalized) < _TITLE_GROUNDING_MIN_LENGTH:
        return title, None

    for restore, what in (
        (_restore_leading_parenthetical, "parenthetical"),
        (_restore_leading_label, "label"),
    ):
        if resolution is None:
            break
        restored = restore(title, resolution, printed_rows)
        if restored is None:
            continue
        restored_title, candidate_id = restored
        logger.info(
            "Title regrounded to its printed leading %s: %r -> %r",
            what,
            title[:80],
            restored_title[:80],
        )
        return restored_title, ValidationIssue(
            code="VAL_TITLE_REGROUNDED",
            severity=IssueSeverity.WARNING,
            message=(
                f"Extracted title dropped the leading {what} printed in the "
                "selected title row; restored it"
            ),
            origin_stage="extract",
            evidence_ids=(candidate_id, f"reason:title_leading_{what}_dropped"),
            count=1,
        )

    candidates: tuple[FrontMatterCandidate, ...] = (
        resolution.candidates if resolution is not None else ()
    )
    haystack = _normalize_for_grounding(
        " ".join((printed_text, *(candidate.raw_text for candidate in candidates)))
    )
    if normalized in haystack:
        return title, None

    low, high = _TITLE_GROUNDING_LENGTH_BAND
    best: tuple[float, str] | None = None
    for source in _grounding_sources(candidates, printed_rows):
        printed = source.strip()
        candidate_normalized = _normalize_for_grounding(printed)
        if not candidate_normalized:
            continue
        if not low <= len(candidate_normalized) / len(normalized) <= high:
            continue
        ratio = difflib.SequenceMatcher(None, normalized, candidate_normalized).ratio()
        if ratio >= _TITLE_GROUNDING_MIN_RATIO and (best is None or ratio > best[0]):
            best = (ratio, printed)

    if best is not None:
        logger.info(
            "Title regrounded to printed front matter (similarity %.3f): %r -> %r",
            best[0],
            title[:80],
            best[1][:80],
        )
        return best[1], ValidationIssue(
            code="VAL_TITLE_REGROUNDED",
            severity=IssueSeverity.WARNING,
            message=(
                "Extracted title did not occur in the printed page text; replaced "
                "with the near-identical printed row"
            ),
            origin_stage="extract",
            evidence_ids=("reason:title_not_printed_verbatim",),
            count=1,
        )

    return title, ValidationIssue(
        code="VAL_TITLE_UNGROUNDED",
        severity=IssueSeverity.WARNING,
        message="Extracted title does not occur verbatim in the printed page text",
        origin_stage="extract",
        evidence_ids=("reason:title_not_printed_verbatim",),
        count=1,
    )


def assess_author_grounding(
    authors: list[PaperAuthor],
    byline: BylineGroup,
) -> tuple[ValidationIssue, ...]:
    """Report selected-byline omissions and hallucinations without mutating authors."""

    if not byline.raw_texts:
        # No byline reached the selected block. Returning silently here made the
        # detector no-op in exactly the case where fabrication is most likely:
        # an author list asserted against a page that prints no byline at all.
        if not authors:
            # Nothing printed and nothing extracted: expose a validation issue as well as the
            # existing processing warning so validation-only consumers see the missing metadata.
            return (
                ValidationIssue(
                    code="VAL_AUTHOR_MISSING",
                    severity=IssueSeverity.WARNING,
                    message=(
                        "Metadata contains no authors and the selected front matter "
                        "prints no byline to check against"
                    ),
                    origin_stage="extract",
                    evidence_ids=("reason:no_front_matter_byline",),
                    count=1,
                ),
            )
        # No byline reached the selected block, so this is missing evidence rather than a
        # contradiction of the extracted author list. Report the scope limitation separately.
        return (
            ValidationIssue(
                code="VAL_AUTHOR_UNCHECKED",
                severity=IssueSeverity.WARNING,
                message=(
                    f"{len(authors)} extracted author(s) could not be checked: the selected "
                    "front matter carries no byline row"
                ),
                origin_stage="extract",
                evidence_ids=(
                    "reason:no_printed_byline",
                    *(f"author:{author.author_id}" for author in authors),
                ),
                count=len(authors),
            ),
        )
    expected_entries = _expected_byline_entries(byline)
    if not authors:
        if expected_entries:
            missing_count = len(expected_entries)
            evidence_ids = byline.candidate_ids
        elif byline.title_candidate_ids:
            missing_count = 1
            evidence_ids = (
                "reason:byline_present_authors_empty",
                *byline.title_candidate_ids,
            )
        elif byline.missing_inference_raw_texts:
            missing_count = 1
            evidence_ids = byline.candidate_ids
        else:
            return ()
        return (
            ValidationIssue(
                code="VAL_AUTHOR_MISSING",
                severity=IssueSeverity.WARNING,
                message="Selected front-matter byline exists but metadata contains no authors",
                origin_stage="extract",
                evidence_ids=evidence_ids,
                count=missing_count,
            ),
        )

    title_candidate_indices = frozenset(byline.title_candidate_indices)
    grounding_texts = tuple(
        (
            byline.title_bounded_raw_texts[index]
            if index in title_candidate_indices
            and byline.title_bounded_raw_texts[index] is not None
            else text
        )
        for index, text in enumerate(byline.raw_texts)
    )
    byline_tokens = tuple(_name_tokens(text) for text in grounding_texts)
    locations: list[tuple[int, int, int] | None] = []
    used_locations: set[tuple[int, int, int]] = set()
    previous: tuple[int, int, int] | None = None
    for author in authors:
        location = _locate_author(
            author,
            byline_tokens,
            grounding_texts,
            after=previous,
            used=used_locations,
        )
        locations.append(location)
        if location is not None:
            used_locations.add(location)
            previous = location
    ungrounded = [
        author for author, location in zip(authors, locations, strict=True) if not location
    ]
    issues: list[ValidationIssue] = []
    if ungrounded:
        issues.append(
            ValidationIssue(
                code="VAL_AUTHOR_UNGROUNDED",
                severity=IssueSeverity.WARNING,
                message=(
                    f"{len(ungrounded)} extracted author(s) are not grounded in the selected byline"
                ),
                origin_stage="extract",
                evidence_ids=(
                    *(f"author:{author.author_id}" for author in ungrounded),
                    *byline.candidate_ids,
                ),
                count=len(ungrounded),
            )
        )

    grounded_locations = [location for location in locations if location is not None]
    grounded_authors = [
        author for author, location in zip(authors, locations, strict=True) if location is not None
    ]
    matched_entry_indices: list[int] = []
    remaining_authors = list(grounded_authors)
    for entry_index, entry in enumerate(expected_entries):
        match_index = next(
            (
                index
                for index, author in enumerate(remaining_authors)
                if _entry_matching_author(entry, author)
            ),
            None,
        )
        if match_index is not None:
            matched_entry_indices.append(entry_index)
            remaining_authors.pop(match_index)
    missing_entry_indices = [
        index for index in range(len(expected_entries)) if index not in matched_entry_indices
    ]
    missing_count = len(missing_entry_indices)
    anchored_title_missing_count = _anchored_title_missing_count(
        byline,
        byline_tokens,
        grounded_locations,
    )
    missing_count += anchored_title_missing_count
    if (
        missing_count == 0
        and byline.missing_inference_raw_texts == byline.raw_texts
        and grounded_locations
        and _has_unmatched_author_evidence(byline_tokens, grounded_locations)
    ):
        missing_count = 1

    if missing_count:
        strict_prefix = bool(
            anchored_title_missing_count == 0
            and not ungrounded
            and matched_entry_indices == list(range(len(matched_entry_indices)))
            and missing_entry_indices
            and min(missing_entry_indices) >= len(matched_entry_indices)
        )
        if strict_prefix:
            reason = "reason:strict_prefix_or_truncation"
        elif anchored_title_missing_count:
            reason = "reason:anchored_byline_gap"
        else:
            reason = "reason:byline_omission"
        issues.append(
            ValidationIssue(
                code="VAL_AUTHOR_MISSING",
                severity=IssueSeverity.WARNING,
                message=(
                    "Selected byline contains author-like evidence missing from the extracted "
                    "list"
                    + ("; this may be a strict prefix or truncation" if strict_prefix else "")
                ),
                origin_stage="extract",
                evidence_ids=(
                    reason,
                    *(f"author:{author.author_id}" for author in grounded_authors),
                    *byline.candidate_ids,
                ),
                count=missing_count,
            )
        )
    return tuple(issues)


# CRediT role vocabulary plus the section's own boilerplate. A contribution line
# printed role-first ("Conceptualization: D.Z.") must never be read as a byline.
_CREDIT_ROLE_TERMS = frozenset(
    {
        "acquisition",
        "administration",
        "all",
        "analysis",
        "and",
        "author",
        "authors",
        "conceptualization",
        "contributed",
        "contribution",
        "contributions",
        "curation",
        "data",
        "draft",
        "editing",
        "equally",
        "formal",
        "funding",
        "investigation",
        "methodology",
        "original",
        "project",
        "resources",
        "review",
        "software",
        "statement",
        "supervision",
        "validation",
        "visualization",
        "writing",
    }
)
_NAME_PARTICLES = frozenset(
    {
        "al",
        "bin",
        "binti",
        "da",
        "das",
        "de",
        "del",
        "della",
        "den",
        "der",
        "di",
        "dos",
        "du",
        "el",
        "ibn",
        "la",
        "le",
        "ten",
        "ter",
        "van",
        "von",
    }
)
_CREDIT_NAME_MAX_TOKENS = 5


def _is_credit_name(value: str) -> bool:
    if any(character.isdigit() or character == "@" for character in value):
        return False
    tokens = value.split()
    if not 2 <= len(tokens) <= _CREDIT_NAME_MAX_TOKENS:
        return False
    folded = [token.casefold().strip(".,") for token in tokens]
    if any(token in _CREDIT_ROLE_TERMS for token in folded):
        return False
    return all(
        token[:1].isupper() or folded_token in _NAME_PARTICLES
        for token, folded_token in zip(tokens, folded, strict=True)
    )


def _split_person_name(value: str) -> tuple[str, str] | None:
    """Split a printed full name into (given, family), carrying particles."""

    tokens = value.split()
    if len(tokens) < 2:
        return None
    start = len(tokens) - 1
    while start > 1 and tokens[start - 1].casefold().strip(".") in _NAME_PARTICLES:
        start -= 1
    given = " ".join(tokens[:start])
    family = " ".join(tokens[start:])
    if not given or not family:
        return None
    return given, family


def credit_statement_authors(contents) -> list[tuple[str, str]]:
    """Names printed at the head of CRediT contribution lines, in printed order.

    A byline may fall outside page-one front matter while its names remain in a contribution statement. Each accepted line contributes the span before its first colon only when that span reads as a person name. Role-first lines and general prose are rejected by the contribution vocabulary and token-shape test.
    """

    section_ids = {
        section.section_id
        for section in getattr(contents, "sections", None) or ()
        if getattr(section, "section_type", None) == CanonicalSection.AUTHOR_CONTRIBUTIONS
    }
    if not section_ids:
        return []
    names: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in getattr(contents, "sentences", None) or ():
        if sentence.section_id not in section_ids or ":" not in sentence.text:
            continue
        prefix = sentence.text.split(":", 1)[0].strip()
        if not _is_credit_name(prefix):
            continue
        split = _split_person_name(prefix)
        if split is None or split in seen:
            continue
        seen.add(split)
        names.append(split)
    return names


def _normalize_affiliation_match_text(value: str) -> str:
    translated = value.translate(_AFFILIATION_MATCH_TRANSLATION)
    return re.sub(r"\s+", " ", translated).strip()


class CoreMetadataExtractor:
    """Extracts the paper's own metadata (not references) via the LLM."""

    def __init__(
        self,
        contents: PaperContents,
        file_hash: str = "unknown",
        llm_client=None,
        locator: RefLocator | None = None,
        email_harvester: AuthorEmailHarvester | None = None,
        settings: GlobalSettings | None = None,
        classifier_resources: ClassifierResources | None = None,
        front_matter_resolution: FrontMatterResolution | None = None,
    ):
        self.contents = contents
        self.sentences_df = contents.sentences_df
        self.file_hash = file_hash
        self.llm_client = llm_client
        self._settings = settings if settings is not None else snapshot_settings()
        self._classifier_resources = classifier_resources
        self._explicit_classifier_runtime = settings is not None or classifier_resources is not None
        self._front_matter_resolution = front_matter_resolution
        self.validation_issues: list[ValidationIssue] = []
        # Which step decided paper_type, for ``extraction.fields``.
        self._paper_type_source = "llm"
        self.locator = locator or RefLocator(contents, settings=self._settings)
        if email_harvester is not None:
            self._email_harvester = email_harvester
        elif front_matter_resolution is None:
            self._email_harvester = AuthorEmailHarvester(contents)
        else:
            selected = _selected_block_candidates(front_matter_resolution)
            allowed_text_ids = front_matter_resolution.allowed_text_ids
            selected_section_ids = {
                candidate.section_id for candidate in selected if candidate.section_id is not None
            }
            scoped_contents = SimpleNamespace(
                sentences=[
                    sentence
                    for sentence in contents.sentences
                    if sentence.text_id in allowed_text_ids
                ],
                sections=[
                    section
                    for section in contents.sections
                    if section.section_id in selected_section_ids
                ],
            )
            self._email_harvester = AuthorEmailHarvester(scoped_contents)

    async def extract(self) -> PaperMetadata:
        """Extract metadata from text before the cutoff section and ORCID lines."""
        try:
            self.validation_issues.clear()
            resolution = self._front_matter_resolution
            if resolution is not None and resolution.selected_block_id is None:
                logger.warning(
                    "Front-matter selection abstained; skipping core metadata extraction."
                )
                return PaperMetadata(doi="", title="", keywords=[], authors=[])

            cutoff_iloc = self.locator.get_cutoff_index()
            allowed_text_ids = resolution.allowed_text_ids if resolution is not None else None
            meta_df = self.locator.collect_core_metadata_rows(
                cutoff_iloc,
                allowed_text_ids=allowed_text_ids,
            )

            if resolution is not None:
                full_text = render_block_context(resolution)
            else:
                full_text = self._format_meta_text(meta_df) if not meta_df.empty else ""

            if not full_text:
                logger.warning("No metadata text available to extract.")
                return PaperMetadata(doi="", title="", keywords=[], authors=[])

            table_text = author_table_context(self.contents, resolution)
            if table_text:
                full_text += "\n" + table_text

            hf_lines = (
                self.contents.detected_headers + self.contents.detected_footers
                if resolution is None
                else []
            )
            if hf_lines:
                full_text += "\n\n[Page headers/footers]\n" + "\n".join(hf_lines)
            logger.info(f"Extracted {len(meta_df)} sentences for metadata.")
            logger.debug(f"Constructed input text of length {len(full_text)}")

            # DOI search must not see the ORCID rows pulled from beyond the
            # cutoff — a reference-list row carrying a doi.org URL would
            # outrank the paper's own DOI when it prints only bare (M2).
            if resolution is not None:
                doi = self._find_doi(full_text, include_furniture=False)
            else:
                doi_text = self._format_meta_text(self.sentences_df.iloc[:cutoff_iloc])
                if hf_lines:
                    doi_text += "\n\n[Page headers/footers]\n" + "\n".join(hf_lines)
                doi = self._find_doi_with_fallback(doi_text)

            # Per-task context slices (LLM_PER_TASK_CONTEXT): the authors and
            # classification calls get narrower, task-specific text; title/
            # keywords keeps the full blob. Off => None => client uses full text.
            authors_text = None
            classification_text = None
            if self._settings.llm.per_task_context:
                if resolution is not None:
                    authors_text = render_author_context(
                        resolution,
                        full_text=full_text,
                    )
                    classification_text = (
                        render_block_context(
                            resolution,
                            roles={"title", "abstract"},
                        )
                        or full_text
                    )
                else:
                    authors_text = self._build_authors_text(meta_df, full_text)
                    classification_text = self._build_classification_text(meta_df)

            if table_text and authors_text is not None and table_text not in authors_text:
                authors_text += "\n" + table_text

            llm_metadata = await self._call_core_llm(
                full_text,
                authors_text=authors_text,
                classification_text=classification_text,
            )
            if llm_metadata is None:
                # ``_call_core_llm`` degraded (see its broad-except branch). The
                # empty record below is the correct fail-closed value, but on its
                # own it is byte-identical to a model that read the front matter
                # and found nothing — mark the difference.
                self._record_degraded(
                    "VAL_CORE_METADATA_DEGRADED",
                    "core_metadata_call_failed",
                    "The core-metadata call did not complete; the empty record is not a refusal",
                )
                return PaperMetadata(doi=doi if doi else "", title="", keywords=[], authors=[])
            recorded = getattr(llm_metadata, "_field_failures", None)
            field_failures = dict(recorded) if isinstance(recorded, dict) else {}
            title_call_failed = "title" in field_failures
            # The trained classifier reads the title and abstract; after a failed
            # title/keywords call it would classify empty input, so its fields
            # fail with that call.
            skip_classifier = bool(
                title_call_failed and self._settings.ml.paper_classifier_model_id
            )
            if skip_classifier:
                for field in ("paper_type", "oecd_domain", "oecd_subdomain"):
                    field_failures.setdefault(field, field_failures["title"])
            if title_call_failed:
                self._record_field_failures(field_failures)
            else:
                # The fan-out degraded these calls silently; say so.
                if "authors" in field_failures:
                    self._record_metadata_warning(
                        WarningCode.AUTHORS_LLM_FAILED,
                        f"{field_failures['authors']}: the author LLM call failed",
                    )
                if "paper_type" in field_failures:
                    self._record_metadata_warning(
                        WarningCode.PAPER_CLASSIFICATION_FAILED,
                        f"{field_failures['paper_type']}: the paper classification LLM call failed",
                    )
            self._record_author_salvage(
                getattr(llm_metadata, "_authors_salvaged_after", None), len(llm_metadata.authors)
            )
            repair_note = getattr(llm_metadata, "_repair_note", None)
            if isinstance(repair_note, str):
                self._record_metadata_warning(WarningCode.LLM_RESPONSE_REPAIRED, repair_note)

            authors = self._convert_llm_authors(llm_metadata.authors)
            self._record_author_anomaly(llm_metadata.authors, authors)
            authors = self._drop_translator_credits(authors, full_text)
            self._repair_glued_names(authors)
            if resolution is not None:
                authors = self._drop_fabricated_authors(authors, full_text, resolution)
                authors = self._drop_transliterated_authors(authors, full_text, resolution)

            author_source = "llm"
            # Gate recovery on the sanitized author list: a nonempty raw response may contain no
            # usable names. Prefer the selected byline over repeating the same wide context.
            if not authors:
                recovered = await self._recover_empty_authors(
                    context=byline_recovery_context(resolution) or authors_text or full_text,
                )
                if recovered:
                    llm_metadata = llm_metadata.model_copy(update={"authors": recovered})
                    authors = self._convert_llm_authors(llm_metadata.authors)
                    authors = self._drop_translator_credits(authors, full_text)
                    self._repair_glued_names(authors)
                    author_source = "llm_recovery"
            # Last resort, strictly inside the empty-author branch so it cannot
            # replace or reorder anything the extraction already found.
            if not authors:
                authors = self._harvest_credit_authors()
                if authors:
                    author_source = "credit_statement"
            # Keep the complete frame so affiliation footnotes and repeated-name reconciliation
            # can inspect evidence on later pages.
            self._reconcile_numbered_affiliations(authors, self.sentences_df)
            self._reconcile_repeated_name_affiliations(authors, self.sentences_df)
            self._email_harvester.demote_implausible_flags(authors)

            # Title/abstract must resolve BEFORE classification: the trained
            # classifier consumes them as input. Order is behavior-neutral for
            # the LLM-validation path (which ignores title/abstract).
            model_title = strip_affiliation_markers(llm_metadata.title or "")
            title, title_issue = ground_title_to_printed_text(
                model_title, resolution, full_text, printed_rows=self._printed_rows()
            )
            if title_issue is not None:
                self.validation_issues.append(title_issue)
            title, subtitle_issue = fold_printed_subtitle(title, resolution, authors=authors)
            if subtitle_issue is not None:
                self.validation_issues.append(subtitle_issue)
            abstract = (llm_metadata.abstract or "").strip()
            keywords = llm_metadata.keywords
            classification_context = classification_text or full_text

            paper_type, oecd_l1, oecd_l2 = "", "", ""
            paper_type_confidence: float | None = None
            oecd_confidence: float | None = None
            if not skip_classifier:
                (
                    paper_type,
                    oecd_l1,
                    oecd_l2,
                    paper_type_confidence,
                    oecd_confidence,
                ) = await self._classify_paper(
                    title, abstract, llm_metadata, classification_context
                )

            is_notice, notice_type = self._apply_correction_notice_guard(title)
            if is_notice:
                authors = []
                abstract = ""
                keywords = []
                oecd_l1 = ""
                oecd_l2 = ""
                oecd_confidence = None
                paper_type = notice_type
                paper_type_confidence = None
                logger.info(
                    "Correction-notice guard fired (kind=%s); "
                    "suppressing fabricated authors/abstract/keywords/oecd",
                    notice_type,
                )

            # Refinements see only owned front matter. Supplemental tables and
            # unowned headers can contain another publication's dates/names.
            precision_context = render_block_context(resolution) if resolution is not None else ""
            published = refine_publication_date(llm_metadata.published, precision_context)
            if published != llm_metadata.published:
                self.validation_issues.append(
                    ValidationIssue(
                        code="VAL_PUBLICATION_DATE_REFINED",
                        severity=IssueSeverity.WARNING,
                        message="Restored publication-date precision from labelled front matter",
                        origin_stage="extract",
                        evidence_ids=("reason:printed_publication_date", f"published:{published}"),
                    )
                )

            metadata = PaperMetadata(
                doi=doi if doi else "",
                title=title,
                abstract=abstract,
                keywords=keywords,
                authors=authors,
                oecd_l1=oecd_l1,
                oecd_l2=oecd_l2,
                oecd_confidence=oecd_confidence,
                paper_type=paper_type,
                paper_type_confidence=paper_type_confidence,
                journal=llm_metadata.journal,
                volume=llm_metadata.volume,
                issue=llm_metadata.issue,
                first_page=llm_metadata.first_page,
                last_page=llm_metadata.last_page,
                issn=llm_metadata.issn,
                publisher=llm_metadata.publisher,
                published=published,
                license=llm_metadata.license,
            )
            metadata._abstract_explicitly_absent = llm_metadata._abstract_explicitly_absent
            for field, source in (
                ("title", "llm" if title == model_title else "title_grounding"),
                ("abstract", "llm"),
                ("keywords", "llm"),
                ("published", "llm"),
                ("journal", "llm"),
                ("author", author_source),
                ("paper_type", "correction_notice" if is_notice else self._paper_type_source),
            ):
                set_field_source(metadata, field, source)

            self._email_harvester.harvest(metadata.authors)
            self.validation_issues.extend(
                repair_author_partitions(metadata.authors, precision_context)
            )

            if resolution is not None and not is_notice:
                self.validation_issues.extend(
                    assess_author_grounding(
                        metadata.authors,
                        build_byline_group(resolution),
                    )
                )

            # Observability: a 0-author result on a non-notice paper is a likely
            # silent author-drop (the byline is normally in the input and the LLM
            # client already re-rolled once, and the grounded free-JSON recovery
            # found nothing either). Surface it so eval/monitoring can see it
            # instead of shipping an empty author list quietly. A notice
            # legitimately has no authors — the guard handles that, don't warn.
            if not metadata.authors and not is_notice:
                self._record_metadata_warning(
                    WarningCode.AUTHORS_EMPTY, f"0 authors extracted (title={title[:60]!r})"
                )

            return metadata

        except ValueError as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise

    def _record_metadata_warning(self, code: WarningCode, message: str) -> None:
        """Log and persist a metadata-extraction warning onto
        ``PaperContents.processing_warnings`` so it reaches the export instead
        of failing silently."""
        logger.warning("Metadata extraction warning %s: %s", code, message)
        if not hasattr(self.contents, "processing_warnings"):
            self.contents.processing_warnings = []
        self.contents.processing_warnings.append(ProcessingWarning(code, message))

    def _harvest_credit_authors(self) -> list[PaperAuthor]:
        """Build an author list from the CRediT contribution statement.

        Only ever called with an already-empty author list, so it cannot delete
        or reorder an extracted author. Every name is printed in the document,
        so the result is grounded by construction.
        """

        names = credit_statement_authors(self.contents)[:_MAX_LLM_AUTHORS]
        if not names:
            return []
        self.validation_issues.append(
            ValidationIssue(
                code="VAL_AUTHOR_CREDIT_RECOVERED",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Recovered {len(names)} author(s) from the CRediT contribution "
                    "statement; no byline reached the extraction context"
                ),
                origin_stage="extract",
                evidence_ids=("reason:credit_statement_fallback",),
                count=len(names),
            )
        )
        return [
            PaperAuthor(
                author_id=index,
                given=given,
                family=family,
                affiliation="",
            )
            for index, (given, family) in enumerate(names, start=1)
        ]

    def _drop_fabricated_authors(
        self,
        authors: list[PaperAuthor],
        context: str,
        resolution: FrontMatterResolution,
    ) -> list[PaperAuthor]:
        """Delete an author list that was invented rather than read off the page.

        All three conditions are required: a failed text match alone does not prove fabrication. Single-token names in non-Latin scripts and names joined to affiliation letters can be correct despite failing a given/family comparison. Deleting on that signal alone would discard valid metadata.

        What separates the real case is that **no byline record reached the
        selected block at all** — bibr found nothing byline-shaped to read names
        off, and the model still returned four. The two false positives both
        carry byline candidates (and already report ``VAL_AUTHOR_UNGROUNDED``
        against them), so this gate excludes both and fires on exactly the one
        paper that shipped the prompt's illustrative names.

        Also skipped unless every author is comparable: a family-only,
        non-organization author yields no token variant and is permanently
        "ungrounded", which must not be read as fabrication. Dropping only
        clears the field — the empty-author recovery downstream still gets its
        grounded attempt, and the CRediT harvest after it.
        """

        if not authors or not all(_author_token_variants(author) for author in authors):
            return authors
        if build_byline_group(resolution).raw_texts:
            # A byline was printed: a name that fails to match it is a grounding
            # failure (script, OCR, glued marker), not evidence of invention.
            return authors
        grounded, _ = grounded_authors_in_context(authors, context)
        if grounded:
            return authors
        self.validation_issues.append(
            ValidationIssue(
                code="VAL_AUTHOR_FABRICATED",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Discarded {len(authors)} extracted author(s): none appear in the "
                    "front-matter text the extraction was given"
                ),
                origin_stage="extract",
                evidence_ids=(
                    "reason:authors_absent_from_context",
                    *(f"author:{author.author_id}" for author in authors),
                ),
                count=len(authors),
            )
        )
        # None of the extracted names appears in the supplied text.
        self._record_metadata_warning(
            WarningCode.AUTHORS_FABRICATED,
            f"discarded {len(authors)} author(s) absent from the extraction context",
        )
        return []

    def _drop_transliterated_authors(
        self,
        authors: list[PaperAuthor],
        context: str,
        resolution: FrontMatterResolution,
    ) -> list[PaperAuthor]:
        """Delete an author list the model romanised instead of copying.

        A byline printed in Cyrillic came back as "N. O. Petrova". The
        fabrication guard stands aside once a byline is printed, because a name
        that fails to match it is usually a script, OCR or marker difference.
        This case has positive evidence instead: each name is written in a
        script the byline hardly uses, and none of them is printed in the
        extraction context, where a paper that prints both forms carries the
        romanised one. All-or-nothing, like the fabrication guard; the
        empty-author recovery then retries against the byline alone.
        """

        if not authors:
            return authors
        byline = build_byline_group(resolution)
        printed = _letter_scripts(_ADDRESS_FRAGMENT_RE.sub(" ", " ".join(byline.raw_texts)))
        total = sum(printed.values())
        if not total:
            return authors
        for author in authors:
            scripts = _letter_scripts(f"{author.given} {author.family}")
            if len(scripts) != 1:
                return authors
            (script,) = scripts
            if printed.get(script, 0) / total >= _BYLINE_SCRIPT_MIN_SHARE:
                return authors
        grounded, _ = grounded_authors_in_context(authors, context)
        if grounded:
            return authors
        self.validation_issues.append(
            ValidationIssue(
                code="VAL_AUTHOR_FABRICATED",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Discarded {len(authors)} extracted author(s): written in a script the "
                    "selected byline does not print them in"
                ),
                origin_stage="extract",
                evidence_ids=(
                    "reason:authors_script_mismatch",
                    *(f"author:{author.author_id}" for author in authors),
                    *byline.candidate_ids,
                ),
                count=len(authors),
            )
        )
        self._record_metadata_warning(
            WarningCode.AUTHORS_FABRICATED,
            f"discarded {len(authors)} author(s) romanised from the printed byline",
        )
        return []

    def _drop_translator_credits(
        self,
        authors: list[PaperAuthor],
        context: str,
    ) -> list[PaperAuthor]:
        """Remove people the front matter credits only as the translator.

        A "Traducido del inglés por …" line under the byline made the
        translator a second author.
        """

        context_tokens = _name_tokens(context)
        kept: list[PaperAuthor] = []
        dropped: list[PaperAuthor] = []
        for author in authors:
            (dropped if _is_translator_credit(author, context_tokens) else kept).append(author)
        if not dropped:
            return authors
        self.validation_issues.append(
            ValidationIssue(
                code="VAL_AUTHOR_TRANSLATOR_DROPPED",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Discarded {len(dropped)} extracted author(s) printed only as the translator"
                ),
                origin_stage="extract",
                evidence_ids=(
                    "reason:translator_credit",
                    *(f"author:{author.author_id}" for author in dropped),
                ),
                count=len(dropped),
            )
        )
        for author_id, author in enumerate(kept, start=1):
            author.author_id = author_id
        return kept

    def _repair_glued_names(self, authors: list[PaperAuthor]) -> None:
        """Split a family name glued to the word printed before it.

        PDFs often position the next word instead of printing a space glyph,
        and the text layer then reads "Kerem B.Yalcin1, Selin DenizAksoy1".
        The model copies that verbatim, so the given name loses its initial or
        middle name to the family name. Glued initials move back to the given
        name unconditionally: an initial is never part of a surname. A
        lower-to-upper join splits only when the paper prints the spaced form
        somewhere, such as a contribution statement; otherwise the join may be
        the printed spelling, as in "DeLaCruz".
        """

        printed: str | None = None
        for author in authors:
            if not author.family or "organization" in (author.role or []):
                continue
            repaired = _unglue_leading_initials(author.given, author.family)
            if repaired is None and _LETTER_RUN_RE.fullmatch(author.family):
                words = _case_join_words(author.family)
                if len(words) > 1:
                    if printed is None:
                        printed = " ".join(self.sentences_df["text"].dropna().astype(str))
                    spaced = r"\s+".join(re.escape(word) for word in words)
                    if re.search(rf"(?<![^\W\d_]){spaced}(?![^\W\d_])", printed):
                        repaired = (f"{author.given} {' '.join(words[:-1])}".strip(), words[-1])
            if repaired is None:
                continue
            author.given, author.family = repaired
            self.validation_issues.append(
                ValidationIssue(
                    code="VAL_AUTHOR_PARTITION_REPAIRED",
                    severity=IssueSeverity.WARNING,
                    message="Split a family name glued to the given name printed before it",
                    origin_stage="extract",
                    evidence_ids=(f"author:{author.author_id}", "reason:glued_family_name"),
                )
            )

    def _record_author_anomaly(self, llm_authors, cleaned: list[PaperAuthor]) -> None:
        """Record ONE ``AUTHORS_ANOMALY`` warning when the author sanitizer had
        to intervene beyond trivially — a runaway repetition loop (incoming over
        the cap, e.g. NuExtract3-FP8 emitting the same fragment to the token
        cap) or a non-trivial number of dropped/collapsed entries (blanks,
        affiliation fragments mislabeled as organization authors, duplicates).
        See :meth:`_convert_llm_authors`."""
        incoming = len(llm_authors)
        kept = len(cleaned)
        if incoming > _MAX_LLM_AUTHORS:
            self._record_metadata_warning(
                WarningCode.AUTHORS_ANOMALY,
                f"{incoming} LLM authors exceeds cap {_MAX_LLM_AUTHORS}; kept leading {kept}",
            )
        elif incoming and not kept:
            # Total wipeout is always worth a warning even below the drop
            # threshold: a single dropped entry used to ship zero authors in
            # complete silence. Name the clause that fired — production exports
            # carry no other trace of the raw model output.
            blank = sum(
                1
                for a in llm_authors
                if not (a.given or "").strip() and not (a.family or "").strip()
            )
            organization = sum(
                1
                for a in llm_authors
                if list(getattr(a, "role", None) or []) == ["organization"]
                and not (a.given or "").strip()
            )
            email = sum(1 for a in llm_authors if "@" in (a.given or "") or "@" in (a.family or ""))
            self._record_metadata_warning(
                WarningCode.AUTHORS_ANOMALY,
                f"sanitizer dropped all {incoming} LLM author entries "
                f"(blank_name={blank}, organization_fragment={organization}, email={email})",
            )
        elif incoming - kept >= _AUTHOR_ANOMALY_MIN_DROP:
            self._record_metadata_warning(
                WarningCode.AUTHORS_ANOMALY,
                f"dropped {incoming - kept} of {incoming} LLM author entries "
                "(blank/organization-fragment/duplicate)",
            )

    @staticmethod
    def _format_meta_text(meta_df) -> str:
        """Concatenate metadata sentences with section-name headers."""
        selected_sections = meta_df["section_name"].values
        selected_text = meta_df["text"].values
        parts: list[str] = []
        current_section = None
        for section, text in zip(selected_sections, selected_text, strict=True):
            if pd.notna(section) and section != current_section:
                parts.append(f"\n\n{section}")
                current_section = section
            parts.append(str(text))
        return "\n".join(parts)

    def _find_doi(self, text: str, *, include_furniture: bool = True) -> str | None:
        """Find the paper's own DOI from explicit markers or headers/footers.

        The paper's own DOI is typically formatted with an explicit marker like
        ``DOI: 10.xxxx/...`` or ``doi.org/10.xxxx/...``.  Bare DOIs appearing
        inline in the metadata text are usually reference citations and should
        not be used.

        Selection is delegated to ``doi_identity.select_doi_from_text``, which
        classifies candidates and breaks ties like ``select_doi_candidates``
        (see its docstring and ``_TIE_BREAK_LADDER``): reference, component,
        data/code and funder-registry candidates are rejected outright, the
        highest ``selection_tier`` wins, and a tie that the provenance ladder
        cannot reduce to one DOI **abstains** with ``VAL_DOI_AMBIGUOUS``.  The
        text here has no page or section provenance, so unlike
        ``select_doi_candidates`` it still selects an uncontested tier-1 DOI.
        In the pipeline ``IdentityValidationStage`` then overwrites
        ``metadata.doi`` with the provenance-aware selection.
        Wrap-truncated DOIs are repaired both upstream in ``fix_ocr_artifacts``
        (``bibr/input/consolidate_text.py``) and, for breaks falling between
        ``10.`` and the registrant digits, by a marker-gated bridge in
        ``doi_identity``.
        """
        from bibr.extract.doi_identity import normalize_candidate_doi, select_doi_from_text

        selection = select_doi_from_text(text)
        if selection.selected is not None:
            return normalize_candidate_doi(selection.selected.raw)

        if not include_furniture:
            return None

        # Fallback: headers/footers — publisher-printed, almost always the paper's own.
        for line in self.contents.detected_footers + self.contents.detected_headers:
            furniture_selection = select_doi_from_text(line)
            if furniture_selection.selected is not None:
                logger.debug("DOI found in header/footer: %s", furniture_selection.selected.raw)
                return normalize_candidate_doi(furniture_selection.selected.raw)

        return None

    def _find_doi_with_fallback(self, full_text: str) -> str | None:
        """Find DOI in *full_text*; fall back to pages 1-2 sentences if absent.

        The fallback excludes rows under a references heading so a reference DOI cannot become the paper DOI. Footnote rows remain eligible, including a paper DOI printed only in an early-page footnote.
        """
        doi = self._find_doi(full_text)
        if doi is None:
            df = self.sentences_df
            first_page = _front_page(df)
            early_mask = df["page_number"].isin([first_page, first_page + 1])
            if "section_name" in df.columns:
                under_ref_heading = df["section_name"].map(
                    lambda n: bool(_REF_HEADER_RE.match(str(n))) if pd.notna(n) else False
                )
                early_mask &= ~under_ref_heading
            early_texts = df.loc[early_mask, "text"]
            if not early_texts.empty:
                early_blob = "\n".join(early_texts.astype(str))
                doi = self._find_doi(early_blob)
                if doi:
                    logger.info("DOI found via early-page fallback: %s", doi)
        return doi

    async def _recover_empty_authors(self, *, context: str) -> list[PaperAuthor] | None:
        """One extractor-owned free-JSON recovery for schema-valid empty authors.

        Returns the grounded recovered authors, or ``None`` when recovery
        degraded (upstream failure or invalid structured output — both optional
        branches). Any other typed processing failure propagates.
        """
        if not self.llm_client.json_mode_reroll_is_distinct():
            # The free-JSON client is only actually free-JSON for the openai
            # provider; elsewhere this would re-send an identical request at
            # temperature 0 and get the identical empty result back, for the
            # price of a second call on every affected paper.
            logger.debug("Skipping empty-author recovery: provider has no distinct JSON mode")
            self._record_recovery_degraded("json_mode_unavailable")
            return None

        try:
            recovered = await self.llm_client.extract_authors(
                context,
                file_hash=self.file_hash,
                json_mode=True,
            )
        except UpstreamServiceError as exc:
            logger.warning("Empty-author recovery failed (hash=%s): %s", self.file_hash, exc)
            self._record_recovery_degraded("upstream_failure")
            return None
        except ProcessingError as exc:
            if exc.error_code != ErrorCode.LLM_INVALID_OUTPUT.value:
                raise
            logger.warning(
                "Empty-author recovery returned invalid output (hash=%s)", self.file_hash
            )
            self._record_recovery_degraded("invalid_output")
            return None
        self._record_author_salvage(
            getattr(recovered, "_salvaged_after", None), len(recovered.authors)
        )
        grounded, rejected = grounded_authors_in_context(recovered.authors, context)
        if rejected:
            self.validation_issues.append(
                ValidationIssue(
                    code="VAL_AUTHOR_RECOVERY_UNGROUNDED",
                    severity=IssueSeverity.WARNING,
                    message="Discarded ungrounded authors from empty-author recovery",
                    origin_stage="extract",
                    evidence_ids=("reason:authors_empty_recovery",),
                    count=len(rejected),
                )
            )
        return grounded

    def _record_recovery_degraded(self, reason: str) -> None:
        """Mark an empty-author recovery that never got a usable answer."""

        self._record_degraded(
            "VAL_AUTHOR_RECOVERY_DEGRADED",
            reason,
            "Empty-author recovery did not complete; the empty author list is not a refusal",
        )

    def _record_author_salvage(self, salvaged_after: object, count: int) -> None:
        """Say that an author list is the salvaged head of a failed response."""

        if not isinstance(salvaged_after, str) or not count:
            return
        self._record_metadata_warning(
            WarningCode.AUTHORS_TRUNCATED
            if salvaged_after == ErrorCode.LLM_TRUNCATED.value
            else WarningCode.AUTHORS_PARTIAL,
            f"{salvaged_after}: kept the {count} leading author(s) of an unfinished response",
        )

    def _record_field_failures(self, field_failures: dict[str, str]) -> None:
        """Mark metadata fields whose LLM call failed while the rest was kept.

        Raised for a failed title/keywords call (or the merged core call), which
        used to fail the whole paper. Blocking like ``VAL_REFERENCES_INCOMPLETE``:
        the record is written but not promotable, so a checkpointed run routes it
        to quarantine and keeps it retryable.
        """

        codes = sorted(set(field_failures.values()))
        self.validation_issues.append(
            ValidationIssue(
                code="VAL_METADATA_FIELD_FAILED",
                severity=IssueSeverity.ERROR,
                message=(
                    f"A metadata LLM call failed ({', '.join(codes)}); its fields are empty "
                    "and the independently extracted metadata and references were kept"
                ),
                origin_stage="extract",
                evidence_ids=tuple(f"reason:{code}" for code in codes)
                + tuple(
                    f"field:{_EXPORT_FIELD_NAMES.get(name, name)}"
                    for name in sorted(field_failures)
                ),
                blocking=True,
            )
        )

    def _record_degraded(self, code: str, reason: str, message: str) -> None:
        """Mark an extraction path that returned empty because a call failed.

        An extraction failure and a valid empty result can otherwise produce identical exports. Record degradation so consumers can distinguish a completed search with no byline from a provider call that never completed.
        """

        self.validation_issues.append(
            ValidationIssue(
                code=code,
                severity=IssueSeverity.WARNING,
                message=f"{message} ({reason})",
                origin_stage="extract",
                evidence_ids=(f"reason:{reason}",),
            )
        )

    async def _call_core_llm(
        self,
        full_text: str,
        *,
        authors_text: str | None = None,
        classification_text: str | None = None,
    ):
        """Call the core-metadata LLM.

        ``UpstreamServiceError`` propagates; non-systemic compatibility failures return ``None``.
        """
        try:
            return await self.llm_client.extract_core_metadata(
                full_text,
                file_hash=self.file_hash,
                authors_text=authors_text,
                classification_text=classification_text,
                include_classification=not bool(self._settings.ml.paper_classifier_model_id),
            )
        except (ProcessingError, UpstreamServiceError):
            raise
        except Exception as e:  # noqa: BLE001 — LLM exceptions are heterogeneous
            logger.warning(f"LLM core metadata extraction failed: {e}")
            return None

    @staticmethod
    def _build_authors_text(meta_df: pd.DataFrame, full_text: str) -> str:
        """Authors-call context slice: page-1 rows plus the ORCID/correspondence
        rescue rows (orcid.org URL, corresponding-author marker, or a bare inline
        ORCID). No headers/footers appended. Falls back to ``full_text`` when
        ``meta_df`` has no ``page_number`` column or the slice comes out empty."""
        if "page_number" not in meta_df.columns:
            return full_text
        text = meta_df["text"]
        mask = (
            (meta_df["page_number"] == _front_page(meta_df))
            | text.str.contains("orcid.org", na=False)
            | text.str.contains(_CORRESPONDING_MARKER_RE, na=False)
            | text.str.contains(_ORCID_BARE_INLINE_RE, na=False)
        )
        sliced = meta_df[mask]
        if sliced.empty:
            return full_text
        return CoreMetadataExtractor._format_meta_text(sliced)

    def _build_classification_text(self, meta_df: pd.DataFrame) -> str:
        """Classification-call context slice: front matter through the last row of
        the first abstract-typed section. Abstract-typed sections are the
        ``self.contents.sections`` entries with ``section_type == ABSTRACT``,
        matched to ``meta_df`` via their header. When no abstract section has
        rows in ``meta_df`` (commentaries/editorials print no abstract), fall
        back to the first 30 rows — the opening rows carry the topical signal.
        No headers/footers appended."""
        abstract_headers = [
            s.header
            for s in self.contents.sections
            if getattr(s, "section_type", None) == CanonicalSection.ABSTRACT and s.header
        ]
        if abstract_headers and "section_name" in meta_df.columns:
            section_names = list(meta_df["section_name"].values)
            for header in abstract_headers:
                positions = [i for i, name in enumerate(section_names) if name == header]
                if positions:
                    return self._format_meta_text(meta_df.iloc[: positions[-1] + 1])
        return self._format_meta_text(meta_df.iloc[:30])

    @staticmethod
    def _convert_llm_authors(llm_authors) -> list[PaperAuthor]:
        """Convert LLM author models to PaperAuthor; strip affiliation markers,
        collapse same-name given/family pairs (consortium echoes), and sanitise
        degenerate output provider-independently.

        AuthorLLM IS A PaperAuthor (subclass), so we promote via model_validate
        on the LLM dump with the sanitisation transforms applied. Post-transform
        the list is cleaned so a repetition-loop / affiliation-fragment storm
        (e.g. NuExtract3-FP8 emitting hundreds of ``role=["organization"]``
        fragments) can't ship as a byline:

        (a) drop entries with no name at all (both given and family blank);
        (b) drop affiliation fragments the model mislabeled as organization
            authors (``role == ["organization"]`` with an empty given);
        (c) collapse exact-duplicate ``(given, family)`` pairs, order-preserving;
        (d) trim a runaway list to the leading distinct run at ``_MAX_LLM_AUTHORS``.

        ``author_id`` is assigned after cleaning so the surviving authors stay
        contiguously numbered. Anomaly *warnings* are recorded by the caller
        (:meth:`extract`), which has the ``PaperContents`` handle.
        """
        authors: list[PaperAuthor] = []
        seen: set[tuple[str, str]] = set()
        for a in llm_authors:
            given = _strip_email_fragments(strip_affiliation_markers(a.given))
            family = _strip_email_fragments(strip_affiliation_markers(a.family))
            # The LLM sometimes echoes the full "First M. Surname Suffix" into
            # given while family already holds "Surname Suffix" — trim the
            # duplicated trailing token(s). The leading-space boundary check
            # keeps substrings safe ("Jonson"/"son" is not trimmed).
            if given and family:
                g, f = given.rstrip(" .,"), family.rstrip(" .,")
                if len(g) > len(f) and g.lower().endswith(" " + f.lower()):
                    # Keep a trailing initial period ("John J.") — only space/
                    # comma are noise here.
                    trimmed = g[: len(g) - len(f)].rstrip(" ,")
                    if trimmed:
                        given = trimmed
            if given and given.strip() == family.strip():
                given = ""

            given_s = (given or "").strip()
            family_s = (family or "").strip()
            # (a) no name at all — drop. An entry that was nothing but an email
            # address lands here, since the address was stripped above.
            if not given_s and not family_s:
                continue
            # (b) affiliation fragment mislabeled as an organization author.
            role = list(getattr(a, "role", None) or [])
            if (
                role == ["organization"]
                and not given_s
                and not _CONSORTIUM_NAME_RE.search(family_s)
            ):
                continue
            # (c) exact-duplicate (given, family) pair — collapse, keep first.
            key = (given_s.casefold(), family_s.casefold())
            if key in seen:
                continue
            seen.add(key)

            authors.append(
                PaperAuthor.model_validate(
                    {
                        **a.model_dump(),
                        "author_id": len(authors) + 1,
                        "given": given,
                        "family": family,
                        "affiliation": a.affiliation or "",
                    }
                )
            )

        # (d) runaway repetition loop — keep the leading distinct run.
        if len(authors) > _MAX_LLM_AUTHORS:
            authors = authors[:_MAX_LLM_AUTHORS]
        return authors

    def _printed_rows(self) -> dict[int, str]:
        """``text_id`` -> printed row text, for row-granular grounding."""
        df = self.sentences_df
        if not {"text_id", "text"}.issubset(df.columns):
            return {}
        return {
            int(text_id): str(text)
            for text_id, text in zip(df["text_id"].values, df["text"].values, strict=True)
            if pd.notna(text_id) and pd.notna(text)
        }

    @staticmethod
    def _reconcile_numbered_affiliations(authors: list[PaperAuthor], meta_df: pd.DataFrame) -> None:
        """Resolve page-one numeric bylines against eligible affiliation blocks."""
        required = {"text", "page_number"}
        if not authors or not required.issubset(meta_df.columns):
            return

        page1_rows = meta_df.loc[meta_df["page_number"] == _front_page(meta_df)]
        marker_lines = [str(value) for value in page1_rows["text"].values if pd.notna(value)]
        if not marker_lines:
            return

        author_numbers: dict[int, list[int]] = {}
        for author in authors:
            full_name = " ".join(part.strip() for part in (author.given, author.family) if part)
            normalized_name = _normalize_affiliation_match_text(full_name)
            if not normalized_name:
                continue
            marker_re = re.compile(
                rf"(?<!\w){re.escape(normalized_name)}\s*"
                r"(?P<numbers>\d{1,2}(?:\s*[,;]\s*\d{1,2})*)(?!\d)",
                re.IGNORECASE,
            )
            for line in marker_lines:
                match = marker_re.search(_normalize_affiliation_match_text(line))
                if match:
                    author_numbers[author.author_id] = [
                        int(number) for number in re.findall(r"\d{1,2}", match.group("numbers"))
                    ]
                    break

        wanted = {number for numbers in author_numbers.values() for number in numbers}
        if not wanted:
            return

        # Two tiers, deliberately kept apart. The page-1 tier is where the
        # affiliation is usually printed, and the back-matter "Author
        # information" tier restates it in a cleaner, article-scoped form.
        # Back-matter wins on disagreement — that is a *refinement* of the same
        # institution, not a conflict — so the tiers must not be flattened into
        # one list before the ambiguity check below.
        backmatter_lines: list[str] = []
        if "section_type" in meta_df.columns:
            backmatter_rows = meta_df.loc[
                meta_df["section_type"] == CanonicalSection.AUTHOR_CONTRIBUTIONS
            ]
            backmatter_lines = [
                str(value) for value in backmatter_rows["text"].values if pd.notna(value)
            ]

        def _definitions(lines: list[str]) -> dict[int, list[str]]:
            found: dict[int, list[str]] = {}
            for line in lines:
                matches = list(_NUMBERED_AFFILIATION_RE.finditer(line))
                for index, match in enumerate(matches):
                    number = int(match.group("number"))
                    if number not in wanted:
                        continue
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                    value = line[match.end() : end].strip(" ,;")
                    value = _CORRESPONDENCE_SUFFIX_RE.split(value, maxsplit=1)[0].strip(" ,;")
                    # "<digit> <Capital>" is also the shape of a figure or
                    # table caption and of a publication-history line, so the
                    # captured run has to look like an institution before it
                    # can define a marker. Abstaining makes the
                    # ``len(resolved) == len(numbers)`` guard below drop the
                    # author, which is the fail-closed answer.
                    if value and _looks_like_affiliation(value):
                        found.setdefault(number, []).append(value)
            return found

        def _unambiguous(values: list[str]) -> str | None:
            """The single institution these definitions agree on, else nothing.

            Within one tier a marker number is only a valid key while it means
            one thing. Page 1 of an abstract book or a combined-issue PDF
            carries several articles, each restarting its affiliation list at 1,
            so ``1`` can be defined more than once with genuinely different
            text. Taking the last definition would assign a *neighbouring
            article's* institution to this article's author — the ownership leak
            the removed ``meta_df`` scoping was reaching for, except that
            scoping also deleted the page-1 footnote this reconciler exists to
            read. Abstaining here instead makes the ``len(resolved) ==
            len(numbers)`` guard below drop the author, which is the fail-closed
            answer. Repeated identical rows (duplicated OCR lines, a header
            echoed per column) collapse and are not an ambiguity.
            """
            distinct = list(dict.fromkeys(_normalize_affiliation_match_text(v) for v in values))
            return values[0] if len(distinct) == 1 else None

        page1_definitions = _definitions(marker_lines)
        backmatter_definitions = _definitions(backmatter_lines)

        affiliations: dict[int, str] = {}
        for number in wanted:
            # Back-matter first: it is the article's own restatement, so it both
            # outranks the page-1 form and can rescue a number the page-1 tier
            # defines ambiguously.
            for tier in (backmatter_definitions, page1_definitions):
                values = tier.get(number)
                if not values:
                    continue
                resolved_value = _unambiguous(values)
                if resolved_value is not None:
                    affiliations[number] = resolved_value
                    break

        for author in authors:
            numbers = author_numbers.get(author.author_id, [])
            resolved = [affiliations[number] for number in numbers if number in affiliations]
            if resolved and len(resolved) == len(numbers):
                author.affiliation = "; ".join(dict.fromkeys(resolved))

    @staticmethod
    def _reconcile_repeated_name_affiliations(
        authors: list[PaperAuthor], meta_df: pd.DataFrame
    ) -> None:
        """Resolve semicolon-delimited full-name affiliation blocks."""
        required = {"text", "page_number"}
        if len(authors) < 2 or not required.issubset(meta_df.columns):
            return

        authors_by_name = {}
        for author in authors:
            full_name = " ".join(part.strip() for part in (author.given, author.family) if part)
            normalized = _normalize_affiliation_match_text(full_name).casefold()
            if normalized:
                authors_by_name[normalized] = author

        candidate_rows = meta_df.loc[meta_df["page_number"] > _front_page(meta_df), "text"]
        for raw_line in candidate_rows.values:
            if pd.isna(raw_line):
                continue
            entries = [entry.strip() for entry in str(raw_line).split(";") if entry.strip()]
            resolved: dict[int, tuple[PaperAuthor, str]] = {}
            for entry in entries:
                name_text, separator, affiliation = entry.partition(",")
                affiliation = affiliation.strip(" ,;")
                if not separator or not affiliation or not _AFFILIATION_ORG_RE.search(affiliation):
                    continue
                normalized_name = _normalize_affiliation_match_text(name_text).casefold()
                author = authors_by_name.get(normalized_name)
                if author is not None:
                    resolved[author.author_id] = (author, affiliation)

            if len(resolved) < 2:
                continue
            for author, affiliation in resolved.values():
                author.affiliation = affiliation
            return

    async def _classify_paper(
        self, title: str, abstract: str, llm_metadata, classification_text: str
    ) -> tuple[str, str, str, float | None, float | None]:
        """Resolve ``(paper_type, oecd_l1, oecd_l2, paper_type_conf, oecd_conf)``.

        When the trained multitask classifier is configured
        (``ML_PAPER_CLASSIFIER_MODEL_ID``, set by default), it predicts OECD
        L1/L2 + paper_type from title+abstract, with a confidence-gated LLM
        escalation for paper_type ONLY; OECD predictions are never escalated.
        Per its published model card, the default checkpoint's training
        supervision is DeepSeek-v4-Flash teacher labels, which changed 16,220
        OECD L1 labels relative to a matched OpenAlex baseline. With the model
        id set to null, the LLM validation path runs verbatim and confidences
        stay ``None``.
        """
        self._paper_type_source = "llm"
        if not self._settings.ml.paper_classifier_model_id:
            pt, l1, l2 = self._validate_classification(
                oecd_domain=llm_metadata.oecd_domain,
                oecd_subdomain=llm_metadata.oecd_subdomain,
                paper_type=llm_metadata.paper_type,
            )
            return pt, l1, l2, None, None

        from bibr.structure import paper_classifier

        degraded_reason: str | None = None
        try:
            if self._explicit_classifier_runtime:
                result = await paper_classifier.classify_paper_async(
                    title,
                    abstract,
                    classifier_resources=self._classifier_resources,
                    settings=self._settings,
                )
            else:
                result = await paper_classifier.classify_paper_async(title, abstract)
        except ProcessingError:
            raise
        except Exception as exc:
            logger.warning("Trained paper classifier unavailable; using LLM fallback: %s", exc)
            result = None
            degraded_reason = type(exc).__name__
        if result is None:
            fallback_failure: str | None = None
            if self._settings.llm.merged_core_metadata:
                fallback = llm_metadata
            else:
                try:
                    fallback = await self.llm_client.extract_paper_classification(
                        classification_text,
                        file_hash=self.file_hash,
                    )
                except ProcessingError:
                    raise
                except Exception as exc:
                    from bibr.clients.llm import llm_failure_code

                    logger.warning("Broad paper-classification LLM fallback failed: %s", exc)
                    fallback = PaperClassificationLLM()
                    fallback_failure = llm_failure_code(exc)
            # The exception type only — never its message, which can quote
            # document text. Recorded once the fallback's outcome is known.
            self._record_metadata_warning(
                WarningCode.PAPER_CLASSIFIER_DEGRADED,
                (
                    f"trained classifier raised {degraded_reason}"
                    if degraded_reason
                    else "trained classifier unavailable"
                )
                + (
                    f"; the LLM fallback failed too ({fallback_failure})"
                    if fallback_failure
                    else "; the LLM classified the paper"
                ),
            )
            if fallback_failure:
                self._record_metadata_warning(
                    WarningCode.PAPER_CLASSIFICATION_FAILED,
                    f"{fallback_failure}: neither the trained classifier nor the LLM fallback "
                    "answered",
                )
            pt, l1, l2 = self._validate_classification(
                oecd_domain=fallback.oecd_domain,
                oecd_subdomain=fallback.oecd_subdomain,
                paper_type=fallback.paper_type,
            )
            return pt, l1, l2, None, None

        oecd_l1, l1_score, oecd_l2, l2_score, paper_type, pt_score = result
        self._paper_type_source = "classifier"
        paper_type_confidence: float | None = pt_score
        oecd_confidence: float | None = l1_score

        # OECD L2 is the most ambiguous head; below the gate, emit null rather
        # than a low-confidence subdomain (no LLM fallback for OECD labels).
        l2_gate = float(self._settings.ml.paper_classifier_l2_min_confidence)
        if oecd_l2 and l2_score < l2_gate:
            oecd_l2 = ""

        # The two heads are independent softmaxes over disjoint label spaces
        # sharing one encoding — nothing ties the L2 argmax to the L1 argmax,
        # so the pair can violate the taxonomy. Each L2 belongs to exactly one
        # L1: backfill an empty L1 from it, and otherwise apply the same
        # preference the gate above states — null beats a wrong subdomain.
        if oecd_l2:
            from bibr.structure.paper_classifier import OECD_L2_TO_L1

            parent = OECD_L2_TO_L1.get(oecd_l2)
            if parent is None:
                oecd_l2 = ""
            elif not oecd_l1:
                oecd_l1 = parent
            elif parent != oecd_l1:
                logger.info(
                    "classifier heads disagree: OECD L2 '%s' belongs to '%s', not '%s'; "
                    "dropping the subdomain",
                    oecd_l2,
                    parent,
                    oecd_l1,
                )
                oecd_l2 = ""

        threshold = float(self._settings.ml.paper_classifier_min_confidence)
        if pt_score < threshold and self._settings.ml.paper_classifier_llm_escalation:
            try:
                label = await self.llm_client.label_paper_type(
                    title, abstract, file_hash=self.file_hash
                )
            except ProcessingError:
                raise
            except Exception as e:  # noqa: BLE001 — escalation is best-effort
                logger.warning("paper_type LLM escalation failed: %s", e)
                label = None
            if label is not None and getattr(label, "paper_type", None):
                paper_type = label.paper_type
                paper_type_confidence = label.confidence
                self._paper_type_source = "llm_label"

        return paper_type, oecd_l1, oecd_l2, paper_type_confidence, oecd_confidence

    @staticmethod
    def _validate_classification(
        *,
        oecd_domain: str | None,
        oecd_subdomain: str | None,
        paper_type: str | None,
    ) -> tuple[str, str, str]:
        """Validate LLM classification fields. Returns ``(paper_type, oecd_l1, oecd_l2)``.

        When the subdomain doesn't validate under the model's L1 (or L1 is
        empty), it is canonicalized against ALL L2 labels; a hit yields a
        parent L1 (each L2 belongs to exactly one L1), which flips/backfills
        the L1. This rescues real cross-L1 mislabels — e.g. OECD files
        Computer Science under Natural Sciences, not Engineering.
        """
        from bibr.structure.paper_classifier import (
            OECD_L2_TO_L1,
            PAPER_TYPE_LABELS,
            canonicalize_oecd_l2_any,
            validate_oecd_l1,
            validate_oecd_l2,
        )

        _VALID_PAPER_TYPES = set(PAPER_TYPE_LABELS)
        pt = paper_type or ""
        if pt and pt not in _VALID_PAPER_TYPES:
            logger.warning("LLM returned unknown paper_type '%s', discarding", pt)
            pt = ""

        l1 = validate_oecd_l1(oecd_domain)
        l2 = ""
        if oecd_subdomain:
            if l1:
                l2 = validate_oecd_l2(l1, oecd_subdomain)
            if not l2:
                # Within-L1 miss (or empty L1): canonicalize L1-agnostically and
                # adopt the canonical label's parent L1.
                canon = canonicalize_oecd_l2_any(oecd_subdomain)
                if canon:
                    parent = OECD_L2_TO_L1[canon]
                    if parent != l1:
                        logger.info(
                            "reassigned OECD L1 '%s' → '%s' based on subdomain '%s'",
                            l1,
                            parent,
                            canon,
                        )
                    l1, l2 = parent, canon
                else:
                    logger.warning(
                        "LLM returned oecd_subdomain '%s' not valid for L1 '%s', discarding",
                        oecd_subdomain,
                        l1,
                    )
        return pt, l1, l2

    @staticmethod
    def _apply_correction_notice_guard(title: str | None) -> tuple[bool, str]:
        """Detect correction-notice titles and return the suppression decision.

        Returns ``(is_notice, notice_paper_type)``. When ``is_notice`` is
        True, the caller MUST clear fabricated content fields (authors,
        abstract, keywords, OECD classification) and overwrite ``paper_type``
        with ``notice_paper_type``. When False, ``notice_paper_type`` is
        the empty string.

        Mapping:
          - corrigendum, correction → "corrigendum"
          - erratum                 → "erratum"
          - retraction              → "retraction"

        ``correction`` collapses into ``corrigendum`` because they're the
        same publisher concept under two names. ``retraction`` keeps its
        own value because retractions are semantically distinct (the
        article is being withdrawn, not amended).

        Title sanitization (``strip_affiliation_markers``) is expected to
        have run before this guard — the regex's ``^\\s*`` anchor would be
        defeated by a leading superscript affiliation marker.
        """
        if not title:
            return False, ""
        m = _CORRECTION_NOTICE_TITLE_RE.match(title)
        if not m:
            return False, ""
        kind = m.group("kind").lower()
        notice_type = {
            "corrigendum": "corrigendum",
            "correction": "corrigendum",
            "erratum": "erratum",
            "retraction": "retraction",
        }[kind]
        return True, notice_type

    @staticmethod
    def _should_suppress_commentary_abstract(
        paper_type: str | None,
        abstract: str | None,
        keywords: list[str] | None,
    ) -> bool:
        """Compatibility shim: length/share suspicion never authorizes deletion."""
        del paper_type, abstract, keywords
        return False
