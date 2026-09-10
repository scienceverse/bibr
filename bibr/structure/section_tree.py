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
from dataclasses import dataclass, field

from bibr.paper_contents import CanonicalSection, PaperSection
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


# Study/experiment marker headers, anchored at header start. Runs on the RAW
# header — normalize_text strips digits, which would destroy the marker token.
# "Study 2: Methods" matches with token "2" and remainder "Methods".
STUDY_MARKER_RE = re.compile(
    r"^\s*(?:study|experiment|exp\.?)\s+(\d+[a-z]?|[ivxl]+|[a-z])\b",
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


@dataclass
class _ScopeState:
    """Per-scope hierarchy state for ``assign_hierarchy_from_top_level``."""

    most_recent_top_level: int = 0
    most_recent_imrad_anchor: int = 0
    first_of_type: dict[CanonicalSection, int] = field(default_factory=dict)


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
    in_zone,
    handled: set[int],
) -> None:
    """Validate one contiguous appendix-shaped block and, if it qualifies,
    pin its roots to top level (siblings, parent=0) and nest dotted children.

    Level/parent only — section_type stays untouched here (the APPENDIX type is
    assigned by chunk 4a). Adds every mutated section id to ``handled``.
    """
    root_positions = [k for k in block if infos[k][0] in ("root_letter", "appendix_marker")]
    if not root_positions:
        return

    letters = [infos[k][1] for k in root_positions if infos[k][1] is not None]
    distinct = list(dict.fromkeys(letters))
    non_decreasing = all(letters[i] <= letters[i + 1] for i in range(len(letters) - 1))
    has_marker = any(infos[k][0] == "appendix_marker" for k in block)
    in_back = any(in_zone(k) for k in root_positions)
    after_refs = references_idx is not None and block[0] > references_idx

    qualifies = (
        # A coherent A, B, C, ... run in the back matter.
        (non_decreasing and len(distinct) >= 2 and distinct[0] == "A" and in_back)
        # An explicit "Appendix" marker heading anchors even a single letter.
        or (has_marker and len(root_positions) >= 1)
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
    A, or the whole run under References. This pass finds a coherent run of
    appendix-shaped headings in the back matter (last ~40% of the section list,
    or after the References section) and re-pins the roots as top-level siblings
    (parent=0), nesting each dotted "X.n" child under its root "X".

    Conservative by construction: a lone early "A Framework for X" with no
    sibling run and no Appendix/References anchor never qualifies. Mutates
    ``sections`` in place (level/parent only).
    """
    handled: set[int] = set()
    n = len(sections)
    if n < 2:
        return handled

    zone_start = int(n * 0.6)
    references_idx: int | None = None
    for i, s in enumerate(sections):
        if s.section_type == CanonicalSection.REFERENCES:
            references_idx = i
            break

    # Level-0 sections (title/root) never participate; treat them as gaps that
    # break an appendix block.
    infos = [_appendix_head_info(s.header) if s.level > 0 else (None, None) for s in sections]

    def in_zone(i: int) -> bool:
        return i >= zone_start or (references_idx is not None and i > references_idx)

    i = 0
    while i < n:
        if infos[i][0] not in ("root_letter", "dotted", "appendix_marker"):
            i += 1
            continue
        j = i
        while j < n and infos[j][0] in ("root_letter", "dotted", "appendix_marker"):
            j += 1
        _mark_appendix_block(sections, infos, list(range(i, j)), references_idx, in_zone, handled)
        i = j

    return handled


def assign_hierarchy_from_top_level(
    sections: list[PaperSection],
    scope_ids: dict[int, int] | None = None,
    marker_ids: set[int] | None = None,
) -> None:
    """Reassign level + parent_section_id using two complementary signals.

    Source 1 — `is_top_level_predicted` (set by alias-driven classification or
    by the trained classifier when available):
    - True  -> level=1, parent_section_id=0
    - False -> level=2, parent_section_id=most-recent level-1 section

    Source 2 — type-based positional rule (when `is_top_level_predicted` is
    unset, i.e. None):
    - section_type in IMRAD_ANCHORS  -> level=1, parent=0; updates anchor
    - section_type in INTERLUDE_TYPES -> level=1, parent=0; does NOT update
      the IMRaD anchor (so a following UNKNOWN still folds into the previous
      true IMRaD anchor, not into the interlude)
    - section_type == UNKNOWN with a prior IMRaD anchor -> level=2,
      parent=anchor (the supervisor's positional fold)
    - Otherwise -> left untouched

    Numbered headings (leading dotted-number prefix) are skipped — their
    level/parent already came from numbering inference in
    `pdf_parser._handle_heading`.

    Level-0 sections (title) are never reparented.

    Scoped mode (``scope_ids``/``marker_ids`` from study-marker detection):
    all state (first_of_type, most_recent_top_level, most_recent_imrad_anchor)
    is kept per scope, so Study 2's Methods becomes a fresh level-1 anchor
    instead of folding under Study 1's.  Marker sections themselves pin to
    level=1/parent=0, keep their classified type, update only the positional
    parent (most_recent_top_level), and are exempt from the UNKNOWN fold.
    With ``scope_ids`` None (or all zero) and no ``marker_ids``, behavior is
    identical to the unscoped implementation.
    """
    states: dict[int, _ScopeState] = {}
    markers = marker_ids or frozenset()
    # Coherence-based lettered-appendix repair runs first: it re-pins mis-nested
    # appendix roots as top-level siblings and returns the ids it fully handled
    # so the main loop leaves them alone (like numbered headings).
    appendix_ids = repair_appendix_hierarchy(sections)
    for sec in sections:
        if sec.level == 0:
            continue
        scope = scope_ids.get(sec.section_id, 0) if scope_ids else 0
        state = states.setdefault(scope, _ScopeState())

        if sec.section_id in appendix_ids:
            # Fully handled by the appendix repair pass — do not let the
            # positional rules reparent it or update anchors from it.
            continue

        if normalize_text(sec.header) in _TOC_HEADERS:
            # A table-of-contents heading stays a bare top-level orphan and
            # never becomes most_recent_top_level / IMRaD anchor.
            sec.level = 1
            sec.parent_section_id = 0
            continue

        if getattr(sec, "outline_level_authoritative", False):
            # Level/parent came from a confidently matched PDF-outline
            # (bookmark) entry — the document's own declared hierarchy, the
            # most authoritative signal. Preserve it exactly, only updating the
            # positional anchor state (mirroring the numbered-heading branch)
            # so following unnumbered sections still fold under the right anchor.
            if sec.level == 1:
                state.most_recent_top_level = sec.section_id
                if sec.section_type in IMRAD_ANCHORS:
                    state.most_recent_imrad_anchor = sec.section_id
                    state.first_of_type.setdefault(sec.section_type, sec.section_id)
            continue

        if sec.section_id in markers:
            # Study marker header ("Study 2"): top-level scope opener. It
            # updates the positional parent (an immediately following
            # subsection folds under the study header) but NOT the IMRaD
            # anchor, never enters first_of_type, and skips the UNKNOWN fold
            # (else "Study 2" folds under Study 1's last anchor).
            sec.level = 1
            sec.parent_section_id = 0
            state.most_recent_top_level = sec.section_id
            continue

        if infer_level_from_numbering(sec.header) is not None:
            if sec.level == 1:
                state.most_recent_top_level = sec.section_id
                if sec.section_type in IMRAD_ANCHORS:
                    state.most_recent_imrad_anchor = sec.section_id
                    state.first_of_type.setdefault(sec.section_type, sec.section_id)
            continue

        is_top = getattr(sec, "is_top_level_predicted", None)

        if is_top is True:
            sec.level = 1
            sec.parent_section_id = 0
            state.most_recent_top_level = sec.section_id
            if sec.section_type in IMRAD_ANCHORS:
                state.most_recent_imrad_anchor = sec.section_id
                state.first_of_type.setdefault(sec.section_type, sec.section_id)
        elif is_top is False:
            sec.level = 2
            sec.parent_section_id = state.most_recent_top_level
        else:
            # is_top unknown — fall back to type-based positional rule.
            if sec.section_type in IMRAD_ANCHORS:
                if sec.section_type in state.first_of_type:
                    # Repeat of an already-seen IMRaD type (e.g. second METHODS-
                    # typed heading like "Statistical Analysis") — fold under
                    # the first occurrence rather than starting a new anchor.
                    sec.level = 2
                    sec.parent_section_id = state.first_of_type[sec.section_type]
                else:
                    sec.level = 1
                    sec.parent_section_id = 0
                    state.most_recent_top_level = sec.section_id
                    state.most_recent_imrad_anchor = sec.section_id
                    state.first_of_type[sec.section_type] = sec.section_id
            elif sec.section_type in INTERLUDE_TYPES:
                sec.level = 1
                sec.parent_section_id = 0
                state.most_recent_top_level = sec.section_id
                # IMRaD anchor not reset — interludes are parallel concepts,
                # not new flow anchors.
            elif sec.section_type == CanonicalSection.UNKNOWN and state.most_recent_imrad_anchor:
                sec.level = 2
                sec.parent_section_id = state.most_recent_imrad_anchor
            # else: leave as-is (no anchor yet, or non-foldable type)


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
