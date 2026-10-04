"""Section tree helpers: numbering-aware level inference and tree building.

All helpers operate on the existing flat ``PaperSection`` list — no
dataclass migration needed.  ``infer_level_from_numbering`` returns a
1-6 depth from leading numeric markers like ``1.2.3``.  ``build_section_tree``
populates each section's ``children`` list in place using ``parent_section_id``.

Scoped-Hierarchy v5-lite: multi-study papers ("Study 1" / "Study 2" headers)
get per-study scopes so Study 2's Methods/Results don't fold under Study 1's
anchors.  ``detect_study_markers`` + ``assign_provisional_scopes`` run before
classification (regex on raw headers); ``close_scopes`` runs after
classification (needs section types).  Scope ids are transient — passed as a
plain dict, never stored on sections, never serialized.
"""

import re
from dataclasses import dataclass

from bibr.paper_contents import (
    CanonicalSection,
    PaperSection,
    is_exact_front_matter_furniture,
    is_section_container_heading,
    is_section_furniture_heading,
)
from bibr.utils.text import normalize_text

_LEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.|\s|:|$)")
# Unambiguous lettered-appendix numbering. "Appendix"/"Appendices" (optionally
# followed by a letter) reads as a top-level (depth-1) heading. Dotted-letter
# forms ("A.1", "B.2.1") read as sub-headings — depth = 1 + number of dots,
# mirroring the digit semantics. A bare single letter ("A") is deliberately
# NOT matched here: it is ambiguous with the article "A" ("A Framework for X").
_APPENDIX_HEADER_RE = re.compile(r"^\s*appendi(?:x|ces)\b", re.IGNORECASE)
_LEADING_LETTER_DOTTED_RE = re.compile(r"^\s*[A-Z]((?:\.\d+)+)")
_MAX_LEVEL = 6

# IMRaD container types — these reset the positional anchor.
IMRAD_ANCHORS: frozenset[CanonicalSection] = frozenset(
    {
        CanonicalSection.INTRODUCTION,
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
    }
)

# Exact alias headings that name a whole IMRaD part, as opposed to the
# subsection names the alias table also holds ("Study design", "Statistical
# analysis", "Limitations"). In ``assign_hierarchy_from_top_level`` an
# unnumbered heading with one of these names always opens a level-1 section.
_PART_HEADINGS: frozenset[str] = frozenset(
    {
        "introduction",
        "background",
        "method",
        "methods",
        "methodology",
        "materials and methods",
        "materials & methods",
        "experimental",
        "experimental section",
        "results",
        "findings",
        "results and discussion",
        "results & discussion",
        "experiments",
        "experimental results",
        "empirical results",
        "experiments and results",
        "discussion",
        "general discussion",
        "conclusion",
        "conclusions",
        "concluding remarks",
        "summary and conclusions",
        # Non-English part names (Dutch, French, German, Italian, Portuguese,
        # Spanish, Indonesian, Turkish).
        "inleiding",
        "einleitung",
        "introducción",
        "introduzione",
        "introdução",
        "pendahuluan",
        "giriş",
        "metodi",
        "metodología",
        "metodologia",
        "materiali e metodi",
        "metode penelitian",
        "yöntem",
        "resultaten",
        "résultats",
        "ergebnisse",
        "resultados",
        "risultati",
        "bulgular",
        "discussie",
        "diskussion",
        "discusión",
        "discussione",
        "discussão",
        "pembahasan",
        "tartışma",
        "conclusiones",
        "conclusioni",
        "conclusões",
        "kesimpulan",
    }
)

# "Interlude" types — top-level sections that interrupt the IMRaD flow but
# don't represent a new IMRaD anchor (so a following UNKNOWN should still
# attach to the previous IMRaD anchor, not the interlude).
INTERLUDE_TYPES: frozenset[CanonicalSection] = frozenset(
    {
        CanonicalSection.ACKNOWLEDGMENT,
        CanonicalSection.FUNDING,
        CanonicalSection.KEYWORDS,
        CanonicalSection.OPEN_DATA,
        CanonicalSection.AUTHOR_CONTRIBUTIONS,
        CanonicalSection.COI,
        CanonicalSection.ETHICS,
        CanonicalSection.ENDNOTE,
        CanonicalSection.APPENDIX,
        CanonicalSection.REFERENCES,
    }
)


def infer_level_from_numbering(header: str) -> int | None:
    """Return depth from a leading dotted-number prefix, or ``None``.

    Examples
    --------
    "1 Introduction"      -> 1
    "1.1 Methods"         -> 2
    "2.3.1 Subsection"    -> 3
    "Appendix B"          -> 1
    "A.1 Proof"           -> 2
    "B.2.1 Lemma"         -> 3
    "A Framework for X"   -> None  (bare single letter — ambiguous with "A")
    "Methods"             -> None
    """
    if not header:
        return None
    m = _LEADING_NUMBER_RE.match(header)
    if m:
        level = m.group(1).count(".") + 1
        return min(level, _MAX_LEVEL)
    if _APPENDIX_HEADER_RE.match(header):
        return 1
    m = _LEADING_LETTER_DOTTED_RE.match(header)
    if m:
        level = m.group(1).count(".") + 1
        return min(level, _MAX_LEVEL)
    return None


# Section numbers for the hierarchy pass. Stricter than
# ``infer_level_from_numbering``: every component has one or two digits, and an
# undotted leading integer above 50 is text, not a section number ("75 years
# of ...", a "668 References" line-number prefix).
_SECTION_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)(?:\.|\s|:|$)")
_MAX_SECTION_NUMBER = 50
# Roman section numbers ("IV PROPOSED METHOD", "II. Related Work"): I-XXX before
# ".", ")" or a space and an uppercase letter. They count only when the document
# numbers at least two headings this way, two of them in sequence.
_ROMAN_HEADING_RE = re.compile(r"^\s*([IVX]{1,6})(?:[.)]\s*|\s+)(?=[A-Z])")
_ROMAN_CANONICAL_RE = re.compile(r"^X{0,3}(?:IX|IV|V?I{0,3})$")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}
_MAX_ROMAN = 30
# "Chapter 3: Research Method", "Part II" open a depth-1 section.
_CHAPTER_HEADING_RE = re.compile(r"^\s*(?:chapter|part)\s+(\d{1,2}|[IVX]{1,6})\b", re.IGNORECASE)


def _roman_value(token: str) -> int | None:
    token = token.upper()
    if not token or not _ROMAN_CANONICAL_RE.match(token):
        return None
    total = 0
    previous = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total if 0 < total <= _MAX_ROMAN else None


@dataclass(frozen=True)
class _SectionNumber:
    """A heading's section number: ``path`` (2.3 -> (2, 3)), its kind, and the
    heading text after the number."""

    path: tuple[int | str, ...]
    kind: str  # "arabic", "roman", "chapter" or "appendix"
    rest: str


def _section_number(header: str) -> _SectionNumber | None:
    if not header:
        return None
    m = _CHAPTER_HEADING_RE.match(header)
    if m:
        token = m.group(1)
        value = int(token) if token.isdigit() else _roman_value(token)
        if value is not None:
            return _SectionNumber((value,), "chapter", header[m.end() :])
    m = _SECTION_NUMBER_RE.match(header)
    if m:
        path = tuple(int(part) for part in m.group(1).split("."))
        if len(path) > 1 or path[0] <= _MAX_SECTION_NUMBER:
            return _SectionNumber(path, "arabic", header[m.end() :])
        return None
    m = _ROMAN_HEADING_RE.match(header)
    if m:
        value = _roman_value(m.group(1))
        if value is not None:
            return _SectionNumber((value,), "roman", header[m.end() :])
        return None
    m = _LEADING_LETTER_DOTTED_RE.match(header)
    if m:
        letter = header.strip()[0].upper()
        path = (letter, *(int(part) for part in m.group(1).strip(".").split(".")))
        return _SectionNumber(path, "appendix", header[m.end() :])
    m = _APPENDIX_MARKER_RE.match(header)
    if m:
        token = m.group(1)
        letter = token.upper() if token and len(token) == 1 and token.isalpha() else "appendix"
        return _SectionNumber((letter,), "appendix", header[m.end() :])
    return None


def _document_section_numbers(sections: list[PaperSection]) -> dict[int, _SectionNumber]:
    """Section numbers of a document's headings, by section id.

    Roman numbers count only as a document-wide scheme: at least two Roman
    headings, two of them consecutive (I then II), so a lone "I Want ..." or
    "V Model" heading is text.
    """
    numbers: dict[int, _SectionNumber] = {}
    romans: list[tuple[int, _SectionNumber]] = []
    for sec in sections:
        if sec.level == 0:
            continue
        number = _section_number(sec.header)
        if number is None:
            continue
        if number.kind == "roman":
            romans.append((sec.section_id, number))
        else:
            numbers[sec.section_id] = number
    values = [number.path[0] for _sid, number in romans]
    if len(values) >= 2 and any(b == a + 1 for a, b in zip(values, values[1:], strict=False)):
        numbers.update(romans)
    return numbers


# Study/experiment marker headers, with an optional leading numbering prefix
# ("2 Study 1", "3. Experiment 2"). Runs on the RAW header — normalize_text
# strips digits, which would destroy the marker token. "Study 2: Methods"
# matches with token "2" and remainder "Methods".
STUDY_MARKER_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:study|experiment|exp\.?)\s+(\d+[a-z]?|[ivxl]+|[a-z])\b",
    re.IGNORECASE,
)

# Scope separators ("Pilot Study", "Replication", "Follow-up Experiment").
# Full-header match only (on the normalized header) — avoids body-phrase
# false positives like "pilot study results showed".
SEPARATOR_RE = re.compile(
    r"^\s*(?:pilot(?:\s+study)?|replication|follow-?up)\s*(?:study|experiment)?\s*$",
    re.IGNORECASE,
)

# Leading punctuation between the marker token and a compound remainder
# ("Study 2: Methods" / "Study 2 — Methods").
_MARKER_REMAINDER_STRIP_RE = re.compile(r"^[\s:.\-–—]+")


@dataclass(frozen=True)
class MarkerInfo:
    """A detected study/experiment marker header ("Study 2", "Exp. 1a")."""

    base: str  # scope key — numeric markers coalesce "1a"/"1b" -> "1"
    header: str  # raw header text (used for the LLM scope-context line)
    remainder_type: CanonicalSection | None = None  # alias hit on the post-marker remainder
    is_separator: bool = False


def detect_study_markers(sections: list[PaperSection]) -> dict[int, MarkerInfo]:
    """Detect study/experiment marker headers (pure regex, pre-classification).

    Returns ``section_id -> MarkerInfo`` for marker and separator headers.
    Compound markers ("Study 2: Methods") also carry the canonical type of the
    post-marker remainder when it hits the alias table.
    """
    from bibr.structure.section_classifier import _classify_lookup

    markers: dict[int, MarkerInfo] = {}
    separator_count = 0
    for sec in sections:
        if sec.level == 0:
            continue
        header = sec.header or ""
        m = STUDY_MARKER_RE.match(header)
        if m:
            token = m.group(1).lower()
            digits = re.match(r"\d+", token)
            base = digits.group(0) if digits else token
            remainder = _MARKER_REMAINDER_STRIP_RE.sub("", header[m.end() :]).strip()
            remainder_type: CanonicalSection | None = None
            if remainder:
                canon, _score = _classify_lookup(normalize_text(remainder))
                if canon != CanonicalSection.UNKNOWN:
                    remainder_type = canon
            markers[sec.section_id] = MarkerInfo(
                base=base, header=header.strip(), remainder_type=remainder_type
            )
        elif SEPARATOR_RE.match(normalize_text(header)):
            separator_count += 1
            markers[sec.section_id] = MarkerInfo(
                base=f"separator:{separator_count}",
                header=header.strip(),
                is_separator=True,
            )
    # Separators are only meaningful alongside real numbered/lettered markers:
    # a lone "Pilot Study" Method subsection in a single-study paper must not
    # open a scope (it would promote itself and following subsections to
    # top-level — a regression invisible to the zero-marker identity gate).
    if markers and all(info.is_separator for info in markers.values()):
        return {}
    return markers


def assign_provisional_scopes(
    sections: list[PaperSection], markers: dict[int, MarkerInfo]
) -> dict[int, int]:
    """Provisional ``section_id -> scope_id``: 0 before the first marker; a
    new scope opens only when the marker base changes or a separator fires
    (so "Study 1a"/"1b"/"1c" share one scope)."""
    scope_ids: dict[int, int] = {}
    current_scope = 0
    next_scope = 1
    current_base: str | None = None
    for sec in sections:
        info = markers.get(sec.section_id)
        if info is not None and (info.is_separator or info.base != current_base):
            current_scope = next_scope
            next_scope += 1
            current_base = None if info.is_separator else info.base
        scope_ids[sec.section_id] = current_scope
    return scope_ids


def close_scopes(
    sections: list[PaperSection],
    scope_ids: dict[int, int],
    markers: dict[int, MarkerInfo],
) -> dict[int, int]:
    """Post-classification scope closing — returns an updated copy.

    A section reverts itself and all followers to scope 0 (until a new marker
    reopens a scope) iff its type is in ``INTERLUDE_TYPES`` (back matter) or
    it is a DISCUSSION whose normalized header starts with "general".  A plain
    "Discussion" inside the last study stays in-scope.
    """
    closed_ids = dict(scope_ids)
    closed = False
    for sec in sections:
        sid = sec.section_id
        if sid in markers:
            closed = False
            continue
        if closed:
            closed_ids[sid] = 0
            continue
        if sec.section_type in INTERLUDE_TYPES or (
            sec.section_type == CanonicalSection.DISCUSSION
            and normalize_text(sec.header).startswith("general")
        ):
            closed_ids[sid] = 0
            closed = True
    return closed_ids


# A table-of-contents heading must never become a positional anchor — a thesis
# "CONTENTS" listing otherwise captures every following heading as its child.
_TOC_HEADERS: frozenset[str] = frozenset({"contents", "table of contents"})

# Heading-shape classifiers for the lettered-appendix repair pass. Order
# matters at the call site: appendix-marker, then dotted child, then bare root.
_APPENDIX_MARKER_RE = re.compile(r"^\s*appendi(?:x|ces)\b\s*([A-Za-z0-9]{1,3})?", re.IGNORECASE)
_LETTER_DOTTED_HEAD_RE = re.compile(r"^\s*([A-Z])(?:\.\d+)+")
_LETTER_ROOT_HEAD_RE = re.compile(r"^\s*([A-Z])(?:[\s.:]|$)")


def _appendix_head_info(header: str) -> tuple[str | None, str | None]:
    """Classify a heading's appendix shape as ``(kind, letter)``.

    ``kind`` is one of ``"appendix_marker"`` (an "Appendix"/"Appendices"
    heading, letter optional), ``"dotted"`` (a lettered sub-heading like
    "A.1"/"B.2.1"), ``"root_letter"`` (a bare lettered heading like "A" /
    "A Proofs" / "A. Proofs" / "A: Proofs"), or ``None``. ``letter`` is the
    uppercase root letter when derivable, else ``None``.
    """
    if not header:
        return (None, None)
    m = _APPENDIX_MARKER_RE.match(header)
    if m:
        tok = m.group(1)
        letter = tok.upper() if (tok and len(tok) == 1 and tok.isalpha()) else None
        return ("appendix_marker", letter)
    m = _LETTER_DOTTED_HEAD_RE.match(header)
    if m:
        return ("dotted", m.group(1).upper())
    m = _LETTER_ROOT_HEAD_RE.match(header)
    if m:
        return ("root_letter", m.group(1).upper())
    return (None, None)


def _mark_appendix_block(
    sections: list[PaperSection],
    infos: list[tuple[str | None, str | None]],
    block: list[int],
    references_idx: int | None,
    handled: set[int],
) -> None:
    """Validate one contiguous appendix-shaped block and, if it qualifies,
    pin its roots to top level (siblings, parent=0) and nest dotted children.

    A block qualifies only with a real appendix anchor: an "Appendix" marker
    heading, or the reference list that ends the body before it. A lettered
    run with no anchor is left alone however late it sits: IEEE/ACM papers and
    many regional journals letter the subsections of their last body section
    ("IV. EXPERIMENTS" / "A. Datasets" / "B. Results", "Results and Discussion"
    / "A. ..." / "B. ..."), and a Roman "V. CONCLUSION" reads as root letter V.

    Also re-types qualifying roots APPENDIX (see below). Adds every mutated
    section id to ``handled``.
    """
    root_positions = [k for k in block if infos[k][0] in ("root_letter", "appendix_marker")]
    if not root_positions:
        return

    letters = [infos[k][1] for k in root_positions if infos[k][1] is not None]
    non_decreasing = all(letters[i] <= letters[i + 1] for i in range(len(letters) - 1))
    has_marker = any(infos[k][0] == "appendix_marker" for k in block)
    after_refs = references_idx is not None and block[0] > references_idx

    qualifies = (
        # An explicit "Appendix" marker heading anchors even a single letter.
        (has_marker and len(root_positions) >= 1)
        # Lettered headings directly following the References section.
        or (after_refs and bool(letters) and non_decreasing)
    )
    if not qualifies:
        return

    root_id_by_letter: dict[str, int] = {}
    for k in block:
        kind, letter = infos[k]
        if kind not in ("root_letter", "appendix_marker"):
            continue
        sec = sections[k]
        sec.level = 1
        sec.parent_section_id = 0
        # An appendix root is re-typed APPENDIX unless it already carries a
        # type from an *exact* alias hit ("Appendix" tables to APPENDIX itself,
        # a rare "References"-in-back-matter, etc.). A weak substring/prior type
        # (e.g. RESULTS leaked from the word "Results" inside "A Additional
        # Results") is overridden — it is back matter, not a body section.
        if sec.classification_source != "exact_alias":
            sec.section_type = CanonicalSection.APPENDIX
            sec.classification_source = "appendix_repair"
        handled.add(sec.section_id)
        if letter is not None:
            root_id_by_letter.setdefault(letter, sec.section_id)

    for k in block:
        kind, letter = infos[k]
        if kind != "dotted":
            continue
        parent_id = root_id_by_letter.get(letter)
        if parent_id is None:
            continue
        sec = sections[k]
        sec.level = 2
        sec.parent_section_id = parent_id
        handled.add(sec.section_id)


def repair_appendix_hierarchy(sections: list[PaperSection]) -> set[int]:
    """Repair mis-nested lettered appendices, returning the handled section ids.

    Lettered appendix headings ("A Additional Results", "Appendix B", "A.1")
    carry no digit numbering, so the generic reparent rules fold B and C under
    A, or the whole run under References. This pass finds a run of
    appendix-shaped headings anchored by an "Appendix" marker or by the
    References section before it, and re-pins the roots as top-level siblings
    (parent=0), nesting each dotted "X.n" child under its root "X".

    Conservative by construction: a lettered run with no Appendix/References
    anchor never qualifies, whether it is a lone early "A Framework for X" or
    the lettered subsections of a paper's last body section. Mutates
    ``sections`` in place (level/parent, and the roots' type).
    """
    handled: set[int] = set()
    n = len(sections)
    if n < 2:
        return handled

    # The anchor is the reference list that ends the body: the first
    # REFERENCES section after a core body section. A pre-body panel typed
    # REFERENCES (a Frontiers "Citation" box above the title) anchors nothing.
    # With no body section typed before any REFERENCES section (an essay whose
    # headings name no IMRaD part, or Methods printed after the references),
    # the first REFERENCES section anchors.
    references_idx: int | None = None
    first_references_idx: int | None = None
    body_seen = False
    for i, s in enumerate(sections):
        if s.section_type in IMRAD_ANCHORS:
            body_seen = True
        elif s.section_type == CanonicalSection.REFERENCES:
            if first_references_idx is None:
                first_references_idx = i
            if body_seen:
                references_idx = i
                break
    if references_idx is None:
        references_idx = first_references_idx

    # Level-0 sections (title/root) never participate; treat them as gaps that
    # break an appendix block.
    infos = [_appendix_head_info(s.header) if s.level > 0 else (None, None) for s in sections]

    i = 0
    while i < n:
        if infos[i][0] not in ("root_letter", "dotted", "appendix_marker"):
            i += 1
            continue
        # An "Appendix" heading anchors the lettered headings after it, not the
        # ones before it, so it always opens a block of its own.
        j = i + 1
        while j < n and infos[j][0] in ("root_letter", "dotted"):
            j += 1
        _mark_appendix_block(sections, infos, list(range(i, j)), references_idx, handled)
        i = j

    return handled


# Back matter and front matter that sit at level 1 but never contain the body
# headings after them ("Data sharing" printed inside Methods, a "Funding"
# block between Results and Discussion).
_INTERLUDE_HEADING_TYPES: frozenset[CanonicalSection] = INTERLUDE_TYPES | frozenset(
    {CanonicalSection.FIGURE, CanonicalSection.TABLE, CanonicalSection.FOOTNOTE}
)
# Front and back matter types that need no alias hit; a model or LLM guess of
# them for a body heading ("HISTORY" typed abstract) does not count.
_ALWAYS_INTERLUDE_TYPES: frozenset[CanonicalSection] = frozenset(
    {
        CanonicalSection.TITLE,
        CanonicalSection.ABSTRACT,
        CanonicalSection.KEYWORDS,
        CanonicalSection.REFERENCES,
    }
)
_TRUSTED_ALIAS_SOURCES = frozenset({"exact_alias", "substring_alias"})
# Numbering schemes of body parts; lettered appendices ("A.1") are not body.
_BODY_NUMBER_KINDS = frozenset({"arabic", "roman", "chapter"})
# BMC and Springer group their statements under one printed heading; the
# statement headings printed after it are its subsections.
_DECLARATIONS_HEADINGS = frozenset({"declarations", "statements and declarations"})
_STATEMENT_TYPES = frozenset(
    {
        CanonicalSection.ACKNOWLEDGMENT,
        CanonicalSection.FUNDING,
        CanonicalSection.OPEN_DATA,
        CanonicalSection.AUTHOR_CONTRIBUTIONS,
        CanonicalSection.COI,
        CanonicalSection.ETHICS,
    }
)
_GUESSED_TYPE_SOURCES = frozenset({"model", "llm", "alias_prior"})
_SUBDIVIDED_PART_TYPES = frozenset(
    {CanonicalSection.METHODS, CanonicalSection.RESULTS, CanonicalSection.DISCUSSION}
)


def _part_name_key(text: str) -> str:
    return normalize_text(text).strip(" :;-–—")


_PART_NAME_JOIN_RE = re.compile(r"\s*(?:,|&|\band\b)\s*")


# Singular part names that are not section aliases of their own ("RESULT AND
# DISCUSSIONS").
_SINGULAR_PART_NAMES = frozenset({"result"})


def _names_part(key: str) -> bool:
    return (
        key in _PART_HEADINGS
        or key.removesuffix("s") in _PART_HEADINGS
        or key in _SINGULAR_PART_NAMES
    )


# Words that name a results part on their own or beside another part
# ("Findings and discussion", "Experiments and discussion").
_RESULTS_PART_NAMES = frozenset(
    {
        "result",
        "results",
        "finding",
        "findings",
        "experiment",
        "experiments",
        "experimental results",
        "empirical results",
        "resultaten",
        "résultats",
        "ergebnisse",
        "resultados",
        "risultati",
        "bulgular",
    }
)


def _names_results_part(text: str) -> bool:
    """Whether a heading is, or joins, a results part name ("Results and
    discussion", "Findings and discussion"; not "Discussion of the results")."""
    pieces = [piece for piece in _PART_NAME_JOIN_RE.split(_part_name_key(text)) if piece]
    return any(piece in _RESULTS_PART_NAMES for piece in pieces)


def _opens_discussion(sec: PaperSection) -> bool:
    """Whether a heading opens a discussion part. A part that also reports
    results ("Results and discussion") is where the results are, not the
    discussion that ends the body."""
    header = sec.header or ""
    if _names_results_part(header):
        return False
    return sec.section_type == CanonicalSection.DISCUSSION or "discussion" in _part_name_key(header)


def _is_part_name(text: str) -> bool:
    """Whether a heading names a whole part: "Methods", or a short compound
    with one ("Patients and methods", "Result and discussions")."""
    key = _part_name_key(text)
    if _names_part(key):
        return True
    pieces = [piece for piece in _PART_NAME_JOIN_RE.split(key) if piece]
    return (
        2 <= len(pieces) <= 3
        and all(len(piece.split()) <= 2 for piece in pieces)
        and any(_names_part(piece) for piece in pieces)
    )


# Part names that close a document rather than name its discussion: a paper
# that ends with "Conclusion" can still open a section with its first
# discussion-typed heading ("Implications for Theory and Research").
_CLOSING_PART_RE = re.compile(r"^(?:conclu|summary|final remarks)")
# Headings that close a "Part 1" / "Part 2" body: the general discussion and
# the conclusion belong to the whole paper, not to the last part.
_CLOSES_PARTS_RE = re.compile(r"^(?:general discussion|conclu|concluding|overall discussion)")
_CHAPTER_WORD_RE = re.compile(r"^\s*chapter\b", re.IGNORECASE)


def _is_panel_heading(text: str) -> bool:
    """A boxed panel ("Research in context", "Key messages"): printed beside
    the body, so it holds no body headings."""
    return is_section_container_heading(text) and _part_name_key(text) not in _DECLARATIONS_HEADINGS


def _is_all_caps(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return len(letters) >= 3 and all(char.isupper() for char in letters)


def _is_interlude(sec: PaperSection) -> bool:
    if (
        sec.section_type in _ALWAYS_INTERLUDE_TYPES
        and sec.classification_source not in _GUESSED_TYPE_SOURCES
    ):
        return True
    header = sec.header or ""
    if is_section_furniture_heading(header) or is_exact_front_matter_furniture(header):
        return True
    return (
        sec.classification_source in _TRUSTED_ALIAS_SOURCES
        and sec.section_type in _INTERLUDE_HEADING_TYPES
    )


@dataclass(frozen=True)
class _DocumentHeadings:
    """Document-wide facts the unnumbered-heading rules read."""

    numbers: dict[int, _SectionNumber]
    # IMRaD types the document names with a part heading ("Methods").
    part_types: frozenset[CanonicalSection]
    # Part names are ALL CAPS while other body headings are not: the casing
    # is the document's level-1 typography.
    caps_mode: bool
    # "Chapter N" headings number the document (a thesis): the part names
    # inside a chapter ("Introduction", "Methodology") are its subsections.
    chapter_mode: bool
    # The chapters are "Part N" only (a two-survey paper): a general
    # discussion or a conclusion after them is not inside the last part.
    part_mode: bool
    # Unnumbered headings between the last numbered body heading of a
    # numbered paper and its reference list. A back-matter type guessed for
    # one of them ("Ethics and consent" after "5. Conclusions") is back
    # matter, not a subsection of the last part, and back matter there holds
    # the unnumbered headings printed under it ("Data availability" >
    # "Underlying data").
    trailing_ids: frozenset[int]

    @classmethod
    def of(cls, sections: list[PaperSection]) -> "_DocumentHeadings":
        numbers = _document_section_numbers(sections)
        body = [s for s in sections if s.level > 0]
        parts = [s for s in body if s.section_id not in numbers and _is_part_name(s.header or "")]
        part_types = frozenset(
            s.section_type
            for s in parts
            if s.section_type in IMRAD_ANCHORS
            and not _CLOSING_PART_RE.match(_part_name_key(s.header or ""))
        )
        caps_parts = sum(_is_all_caps(s.header or "") for s in parts)
        other_body = [
            s
            for s in body
            if s.section_id not in numbers
            and not _is_part_name(s.header or "")
            and not _is_interlude(s)
            and s.section_type not in _INTERLUDE_HEADING_TYPES
        ]
        caps_mode = (
            bool(parts)
            and caps_parts > len(parts) // 2
            and any(not _is_all_caps(s.header or "") for s in other_body)
        )
        chapter_ids = [sid for sid, number in numbers.items() if number.kind == "chapter"]
        chapter_mode = len(chapter_ids) >= 2
        by_id = {s.section_id: s for s in sections}
        part_mode = chapter_mode and not any(
            _CHAPTER_WORD_RE.match(by_id[sid].header or "") for sid in chapter_ids if sid in by_id
        )
        # The body ends at the reference list: numbered headings after it
        # (a peer-review report's points, numbered appendices) are not body.
        references_at = next(
            (
                index
                for index, sec in enumerate(sections)
                if sec.section_type == CanonicalSection.REFERENCES
                and sec.classification_source not in _GUESSED_TYPE_SOURCES
            ),
            len(sections),
        )
        body_numbered = [
            index
            for index, sec in enumerate(sections[:references_at])
            if sec.section_id in numbers and numbers[sec.section_id].kind in _BODY_NUMBER_KINDS
        ]
        trailing_ids: frozenset[int] = frozenset()
        if len(body_numbered) >= 3:
            trailing_ids = frozenset(
                sec.section_id
                for sec in sections[body_numbered[-1] + 1 : references_at]
                if sec.level > 0 and sec.section_id not in numbers
            )
        return cls(numbers, part_types, caps_mode, chapter_mode, part_mode, trailing_ids)


def _numbered_parent(
    path: tuple[int | str, ...], earlier: list[tuple[PaperSection, tuple[int | str, ...]]]
) -> PaperSection | None:
    """Nearest earlier numbered heading whose number is a prefix of ``path``
    (2.3 -> 2), else the nearest earlier one with a shorter number in the same
    top-level part (2.3.1 -> 2.2 when "2" and "2.3" are not printed). A
    sub-number never crosses into another part: "3.1" after "2 Methods" with
    no "3" printed has no numbered parent."""
    for prev, prev_path in reversed(earlier):
        if len(prev_path) < len(path) and path[: len(prev_path)] == prev_path:
            return prev
    for prev, prev_path in reversed(earlier):
        if len(prev_path) < len(path) and prev_path[0] == path[0]:
            return prev
    return None


def is_part_heading(text: str) -> bool:
    """Whether a heading names a whole IMRaD part ("Methods", "Results and
    discussion", "Patients and methods")."""
    return _is_part_name(text)


def numbering_parent_ids(sections: list[PaperSection]) -> dict[int, int]:
    """Section id -> id of the nearest earlier heading numbered as its parent
    (2.3 -> 2, 4.2 -> IV). Only children whose parent number is printed."""
    numbers = _document_section_numbers(sections)
    last_by_path: dict[tuple[int | str, ...], int] = {}
    parents: dict[int, int] = {}
    for sec in sections:
        number = numbers.get(sec.section_id)
        if number is None:
            continue
        if len(number.path) > 1:
            parent_id = last_by_path.get(number.path[:-1])
            if parent_id is not None:
                parents[sec.section_id] = parent_id
        last_by_path[number.path] = sec.section_id
    return parents


def assign_hierarchy_from_top_level(
    sections: list[PaperSection],
    scope_ids: dict[int, int] | None = None,
    marker_ids: set[int] | None = None,
) -> None:
    """Reassign level + parent_section_id from numbering and document order.

    Numbered headings ("2.3", "IV", "Chapter 3") take their depth as level and
    the nearest earlier heading whose number is a prefix of theirs as parent
    (2.3 -> 2, 4.2 -> IV), never the title. A sub-number with no numbered
    parent ("1.1" under an unnumbered "Introduction") goes under the most
    recent body section.

    Unnumbered headings follow document order:
    - a part name ("Methods", "RESULTS AND DISCUSSION"), a section-container
      heading ("Declarations"), an ALL-CAPS heading when the part names are
      ALL CAPS and other body headings are not, the first heading of an IMRaD
      type when the paper prints no part name of that type (and is not a
      numbered paper), and a heading the trained classifier predicts top-level
      open a level-1 section;
    - back and front matter (title, abstract, keywords, references, alias-typed
      acknowledgments, funding, statements, appendices; cover-sheet labels)
      sits at level 1 but never contains the headings after it; so does an
      unnumbered heading guessed as back matter after the last numbered body
      heading of a numbered paper. Statements printed under a "Declarations"
      heading are its subsections;
    - every other heading is a child of the most recent level-1 or numbered
      body section; before any, or after the reference list, it opens a
      level-1 section itself.

    In a thesis numbered by "Chapter N", the part names inside a chapter are
    its subsections. Study markers ("Study 2", from ``detect_study_markers``) open level-1
    sections. ``scope_ids`` keeps the first-of-type rule per study scope, so
    Study 2's first results-typed heading opens its own section. The paper
    title, table-of-contents headings, lettered appendices handled by
    ``repair_appendix_hierarchy`` and PDF-outline levels keep their place.

    Level-0 sections (root) are never reparented.
    """
    markers = marker_ids or frozenset()
    appendix_ids = repair_appendix_hierarchy(sections)
    doc = _DocumentHeadings.of(sections)
    seen_types: dict[int, set[CanonicalSection]] = {}
    numbered: list[tuple[PaperSection, tuple[int | str, ...]]] = []
    last_body: PaperSection | None = None
    # The most recent level-1 methods, results or discussion part heading
    # ("METHOD"); an introduction part heading clears it.
    anchor: PaperSection | None = None
    # A discussion part has opened: guessed back matter after it is back
    # matter, not a subsection of the discussion. A study marker ends it:
    # Study 2's own parts follow (unless the marker is itself a discussion,
    # "Study 2: Discussion").
    after_discussion = False

    for sec in sections:
        if sec.level == 0:
            continue
        scope = scope_ids.get(sec.section_id, 0) if scope_ids else 0
        seen = seen_types.setdefault(scope, set())
        number = doc.numbers.get(sec.section_id)

        if sec.section_id in appendix_ids:
            # Placed by the appendix repair; an appendix root contains the
            # unnumbered headings after it.
            if sec.level == 1:
                last_body = sec
            continue

        if normalize_text(sec.header) in _TOC_HEADERS:
            # A table-of-contents heading stays a bare top-level orphan.
            sec.level = 1
            sec.parent_section_id = 0
            continue

        if getattr(sec, "outline_level_authoritative", False):
            # Level/parent came from a matched PDF-outline (bookmark) entry,
            # the document's own declared hierarchy: keep it.
            if number is not None:
                numbered.append((sec, number.path))
            if not _is_interlude(sec):
                last_body = sec
            continue

        if sec.section_id in markers:
            # Study marker header ("Study 2"): a level-1 scope opener. A
            # numbered one ("2 Study 1") is the numbered parent of 2.1.
            sec.level = 1
            sec.parent_section_id = 0
            last_body = sec
            anchor = None
            after_discussion = _opens_discussion(sec)
            if number is not None:
                numbered.append((sec, number.path))
            continue

        if sec.section_type == CanonicalSection.TITLE and sec.classification_source == "title":
            # The paper title keeps its place.
            continue

        if number is not None:
            level = min(len(number.path), _MAX_LEVEL)
            parent = _numbered_parent(number.path, numbered)
            last_number = doc.numbers.get(last_body.section_id) if last_body is not None else None
            if (
                parent is None
                and level > 1
                and last_body is not None
                and last_body.level < level
                and (last_number is None or last_number.path[0] == number.path[0])
            ):
                # "1.1" under an unnumbered "Introduction"; never "3.1" under
                # "2 Methods".
                parent = last_body
            if parent is not None and parent.level >= level:
                level = min(parent.level + 1, _MAX_LEVEL)
            sec.level = level
            sec.parent_section_id = parent.section_id if parent is not None else 0
            numbered.append((sec, number.path))
            last_body = sec
            if sec.level == 1 and sec.section_type in IMRAD_ANCHORS:
                seen.add(sec.section_type)
            continue

        header = sec.header or ""
        first_of_type = (
            sec.section_type in IMRAD_ANCHORS
            and sec.section_type not in seen
            and sec.section_type not in doc.part_types
            and len(doc.numbers) < 3
        )
        if (
            _is_interlude(sec)
            or (sec.section_id in doc.trailing_ids and sec.section_type in _INTERLUDE_HEADING_TYPES)
            or (
                after_discussion
                and not doc.numbers
                and sec.section_type in _INTERLUDE_HEADING_TYPES
                and sec.classification_source in _GUESSED_TYPE_SOURCES
            )
        ):
            if (
                last_body is not None
                and sec.section_type in _STATEMENT_TYPES
                and _part_name_key(last_body.header or "") in _DECLARATIONS_HEADINGS
            ):
                sec.level = min(last_body.level + 1, _MAX_LEVEL)
                sec.parent_section_id = last_body.section_id
                continue
            sec.level = 1
            sec.parent_section_id = 0
            if sec.section_type == CanonicalSection.REFERENCES:
                # The reference list ends the body: what follows (appendices,
                # supplementary material, notes) is not inside its last part.
                last_body = None
                anchor = None
            elif sec.section_id in doc.trailing_ids:
                # Past the numbered body only back matter follows.
                last_body = sec
            continue
        if doc.caps_mode:
            # The casing is the document's own level-1 typography: it
            # outranks type guesses. Some papers set two levels in capitals
            # ("METHOD" > "PARTICIPANTS"): a capitals heading that names no
            # part goes under the methods, results or discussion part heading
            # before it. After "INTRODUCTION" a capitals heading is a part
            # of its own (a review's "HISTORY").
            opens_section = (
                _is_part_name(header)
                or is_section_container_heading(header)
                or (_is_all_caps(header) and anchor is None)
            )
        else:
            opens_section = (
                _is_part_name(header)
                or is_section_container_heading(header)
                or first_of_type
                or getattr(sec, "is_top_level_predicted", None) is True
            )
        closes_parts = doc.part_mode and bool(_CLOSES_PARTS_RE.match(_part_name_key(header)))
        if last_body is None or (opens_section and (not doc.chapter_mode or closes_parts)):
            sec.level = 1
            sec.parent_section_id = 0
            if not _is_panel_heading(header):
                last_body = sec
            if _is_part_name(header):
                anchor = sec if sec.section_type in _SUBDIVIDED_PART_TYPES else None
        else:
            sec.level = min(last_body.level + 1, _MAX_LEVEL)
            sec.parent_section_id = last_body.section_id
        if sec.level == 1 and sec.section_type in IMRAD_ANCHORS:
            seen.add(sec.section_type)
        if sec.level == 1 and _opens_discussion(sec):
            after_discussion = True


def build_section_tree(sections: list[PaperSection]) -> None:
    """Populate ``children`` on each section from ``parent_section_id``.

    Idempotent: clears existing ``children`` before rebuilding so this can
    be called repeatedly after section list mutations (e.g. after
    implicit-section reorganisation).
    """
    by_id = {s.section_id: s for s in sections}
    for s in sections:
        s.children.clear()
    for s in sections:
        if s.parent_section_id is None:
            continue
        parent = by_id.get(s.parent_section_id)
        if parent is None or parent is s:
            continue
        parent.children.append(s)
