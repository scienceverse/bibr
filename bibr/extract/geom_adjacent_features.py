"""Adjacent-line features for the geometry GBM (``adjacent_boundary_v2``) — production inputs.

These are GBM *inputs*, not general-purpose text heuristics: ``GeomSegmenter``
loads a frozen bundle trained against exactly this feature computation, so
``ADJACENT_FEATURE_KEYS``, the multilingual header/onset patterns below, and
``augment_adjacent_features`` may only change in lockstep with a retrain of
``scienceverse/bibr-geom-segmenter-v1`` and a re-run of the promotion gate
(``boundary_f1_gate``). This is the same production/free-heuristic boundary
``bibr/extract/region_seg.py``'s ``_looks_like_ref_onset`` docstring draws for
its own trained-feature regexes ("must stay byte-for-byte stable") versus the
looser onset patterns that live beside them "on purpose" — this module is
the feature side of that same boundary for the geom segmenter.

Until this module existed, this computation had two independent copies — one
here, one in ``bibr_training.seg_geom.adjacent_features`` — which let the
training copy drift (widened multilingual patterns) out of step with what
production actually served. This module is now the single source; the
training side imports from here (mirroring ``bibr/extract/geom_features.py``,
which ``bibr_training.seg_geom.train_artifact`` already imports for the base
feature set). Do not reintroduce a second copy.

The header/onset pattern vocabulary was ported verbatim from
``bibr_training.seg_geom.multilingual_patterns`` (training branch head as of
this move, commit 217fbc3cd, "port bibr multilingual header/onset patterns
into geom features"). That module in turn ported these from bibr's own
inference-side recognisers — ``bibr/ocr/ref_patterns.py`` and
``bibr/extract/region_seg.py`` as of commit c98dc9b9 — which already
recognised non-Anglo reference headers and entry onsets before the training
features did, so the model had been learning from an English-only view of
signals production already supplied multilingually. The header vocabulary is
deliberately enumerated rather than stemmed: "Literature Review" and
"Tinjauan Pustaka" are body sections that must NOT match, and only an exact
phrase list separates them from "Literature" and "Daftar Pustaka".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_REF_HEADER_WORDS = (
    # English
    r"references?(?:\s+and\s+notes)?",
    r"notes\s+and\s+references",
    r"reference\s+list",
    r"list\s+of\s+references",
    r"literature\s+cited",
    r"cited\s+literature",
    r"works\s+cited",
    # Romance
    r"referencias(?:\s+bibliogr[áa]ficas)?",
    r"refer[êe]ncias(?:\s+bibliogr[áa]ficas)?",
    r"r[ée]f[ée]rences(?:\s+bibliographiques)?",
    r"riferimenti\s+bibliografici",
    r"obras\s+citadas",
    r"literatura\s+citada",
    r"bibliogra(?:f|ph)(?:ia|ía|ie|y|i)?",
    # Germanic / Nordic
    r"literatur(?:verzeichnis)?",
    r"quellenverzeichnis",
    r"literatuur(?:lijst)?",
    r"referenties",
    r"litteratur",
    r"referanser|referencer|referenser",
    r"k[äa]llor",
    r"l[äa]hteet",
    r"kirjallisuus",
    # Slavic
    r"список\s+литературы",
    r"список\s+использованной\s+литературы",
    r"использованная\s+литература",
    r"список\s+источников",
    r"литература",
    r"библиография",
    r"список\s+використаних\s+джерел",
    r"використані\s+джерела",
    r"перелік\s+посилань",
    r"література",
    r"pi[śs]miennictwo",
    r"literat[uú]ra",
    r"(?:zoznam|seznam)\s+literat[uú]ry",
    r"popis\s+literature",
    # Turkic / Uralic
    r"kaynak(?:lar|ça|ca)",
    r"irodalomjegyz[ée]k",
    r"hivatkoz[áa]sok",
    # Malay / Indonesian
    r"referensi",
    r"daftar\s+(?:pustaka|rujukan|referensi)",
    r"kepustakaan",
    r"rujukan",
    # CJK / other scripts
    r"引用文献",
    r"参考文献",
    r"參考文獻",
    r"참고문헌",
    r"tài\s+liệu\s+tham\s+khảo",
    r"βιβλιογραφία",
    r"αναφορές",
    r"المراجع",
    r"قائمة\s+المراجع",
    r"منابع",
)

# Optional section numbering ("6 REFERENCES") and decoration ("■ REFERENCES").
REF_HEADER_RE = re.compile(
    r"^[\s\W\d]{0,8}(?:" + "|".join(_REF_HEADER_WORDS) + r")[\s\W]{0,4}$",
    re.IGNORECASE | re.UNICODE,
)

_UPPER = r"A-ZÀ-ÖØ-ÞЀ-ЯЀ-ҁҊ-ҾΆ-Ϋ"

# Bracketed/dotted/parenthesised, plus the loose forms bibr added: Vancouver
# whitespace-only numbering ("1 European Commission."), and the legal/Arabic
# paren and dash forms ("(1) Viñuales", "1- Birnie").
NUMBERED_START_RE = re.compile(
    rf"^\s*(?:\[\d{{1,4}}\]\s*|\d{{1,4}}[.)]\s+|\(?\d{{1,3}}\)?\s*[-–—]?\s+(?=[\"'“(\[]?[{_UPPER}]))",
    re.UNICODE,
)

# Surname: incumbent's `[A-Z][A-Za-z'`-]+` (see bibr-training commit
# d52e5699c, adjacent_features.py) allowed internal apostrophes and hyphens
# anywhere in the run ("O'Brien", "Smith-Jones"). `[^\W\d_]` alone is
# letters-only and drops them, which narrows the incumbent and violates the
# "only what they match widens" contract. The leading run is `*` (not `+`)
# because a real surname can carry the apostrophe/hyphen immediately after
# the initial capital ("O'Brien" has zero letters between "O" and "'") —
# `+` there would still reject it.
_SURNAME = rf"[{_UPPER}][^\W\d_]*(?:[-'’`][^\W\d_]+)*"

_AUTHOR_YEAR = rf"{_SURNAME}(?:,\s+|\s+)[{_UPPER}].{{0,120}}\b(?:19|20)\d{{2}}\b"
_VANCOUVER = (
    rf"[\"'(]?{_SURNAME}(?:\s+[{_UPPER}][^\W\d_]+)?"
    rf"\s+[{_UPPER}]{{1,3}}\.?(?:\s?[{_UPPER}]\.?){{0,2}}[,.]"
)
_QUOTED_TITLE = rf"[\"'(]?[{_UPPER}][^\W\d_]+(?:\s+[{_UPPER}][^\W\d_.]*){{0,5}},\s*[“\"«‘„]"
_CJK = r"[々぀-ヿ㐀-䶿一-鿿].{0,60}?(?:19|20)\d\d"

REF_ONSET_RE = re.compile(
    rf"^\s*(?:{_AUTHOR_YEAR}|{_VANCOUVER}|{_QUOTED_TITLE}|{_CJK})",
    re.UNICODE,
)

_DOI_RE = re.compile(r"\bdoi\s*:|https?://doi\.org/|10\.\d{4,9}/", re.IGNORECASE)


@dataclass(frozen=True)
class AdjacentPatterns:
    """The three regexes ``augment_adjacent_features`` scores a line's text
    shape against.

    Production always uses the module-level ``CURRENT_PATTERNS`` (today's
    live patterns, built from this module's own ``NUMBERED_START_RE`` /
    ``REF_ONSET_RE`` / ``REF_HEADER_RE``) — ``GeomSegmenter`` never passes a
    different triple. Passing one is a training/eval-only affordance: it lets
    ``bibr_training``'s eval harness reproduce a frozen historical pattern
    set (what an artifact was actually trained against, before these
    patterns were last widened) instead of always scoring through today's
    live patterns. See ``bibr_training.seg_geom.adjacent_features.LEGACY_PATTERNS``.
    """

    numbered_start: re.Pattern[str]
    author_year: re.Pattern[str]
    ref_header: re.Pattern[str]


#: The live patterns this module (and therefore ``GeomSegmenter``) actually
#: serves today. Default for ``augment_adjacent_features``.
CURRENT_PATTERNS = AdjacentPatterns(
    numbered_start=NUMBERED_START_RE,
    author_year=REF_ONSET_RE,
    ref_header=REF_HEADER_RE,
)

ADJACENT_FEATURE_KEYS = (
    "prev_len_chars",
    "next_len_chars",
    "dy_from_prev",
    "dy_to_next",
    "indent_delta_prev",
    "indent_delta_next",
    "same_page_prev",
    "same_page_next",
    "looks_numbered_start",
    "looks_author_year_start",
    "looks_doi_continuation",
    "prev_looks_reference_header",
    "next_looks_reference_header",
)


class _AdjacentFeatureLine(Protocol):
    """The four attributes this module's computation touches.

    Both callers have their own ``LineRecord`` dataclass
    (``bibr.ocr.ref_geometry.LineRecord`` and
    ``bibr_training.seg_geom.select_dataset.LineRecord``); neither imports the
    other's. This is a structural (not nominal) contract so either satisfies
    it without either repo depending on the other's dataclass.

    Declared as read-only properties, not plain attributes: bibr's
    ``LineRecord`` is a frozen dataclass, and a plain-attribute ``Protocol``
    requires a *settable* attribute to match, which a frozen dataclass never
    satisfies.
    """

    @property
    def text(self) -> str: ...
    @property
    def page(self) -> int: ...
    @property
    def y_top(self) -> float: ...
    @property
    def x0(self) -> float: ...


def _line_len(line: _AdjacentFeatureLine | None) -> int:
    return len(line.text) if line is not None else 0


def _dy(other: _AdjacentFeatureLine | None, line: _AdjacentFeatureLine, *, previous: bool) -> float:
    if other is None or other.page != line.page:
        return 0.0
    return (other.y_top - line.y_top) if previous else (line.y_top - other.y_top)


def _indent_delta(other: _AdjacentFeatureLine | None, line: _AdjacentFeatureLine) -> float:
    return (line.x0 - other.x0) if other is not None and other.page == line.page else 0.0


def _flag(pattern: re.Pattern[str], text: str) -> int:
    return 1 if pattern.search(text or "") else 0


def augment_adjacent_features(
    feature_rows: list[dict],
    lines: Sequence[_AdjacentFeatureLine],
    *,
    patterns: AdjacentPatterns = CURRENT_PATTERNS,
) -> list[dict]:
    """Add the 13 ``ADJACENT_FEATURE_KEYS`` to each row from its line's neighbors.

    ``feature_rows[i]`` and ``lines[i]`` must describe the same line in the
    same order; a length mismatch raises ``ValueError`` rather than silently
    truncating or zip-erroring further from the call site.

    ``GeomSegmenter`` always calls this with the default ``patterns=
    CURRENT_PATTERNS`` (today's live patterns) — production never passes
    anything else. The ``patterns`` argument exists solely so
    ``bibr_training``'s eval harness can reproduce a frozen historical
    pattern set (``AdjacentPatterns``/``LEGACY_PATTERNS`` in
    ``bibr_training.seg_geom.adjacent_features``) instead of always scoring
    through today's live patterns; this is the *only* caller of that
    affordance, so if you're reading this from production code, you almost
    certainly want the default.
    """
    if len(feature_rows) != len(lines):
        raise ValueError(
            f"feature/line length mismatch: {len(feature_rows)} feature rows, {len(lines)} lines"
        )

    augmented: list[dict] = []
    # strict=True is redundant with the length check above (ruff B905 requires
    # an explicit value regardless) -- kept as a harmless defense-in-depth
    # belt-and-suspenders, not because it's expected to ever fire here.
    for idx, (row, line) in enumerate(zip(feature_rows, lines, strict=True)):
        prev_line = lines[idx - 1] if idx > 0 else None
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None
        augmented.append(
            {
                **row,
                "prev_len_chars": _line_len(prev_line),
                "next_len_chars": _line_len(next_line),
                "dy_from_prev": _dy(prev_line, line, previous=True),
                "dy_to_next": _dy(next_line, line, previous=False),
                "indent_delta_prev": _indent_delta(prev_line, line),
                "indent_delta_next": _indent_delta(next_line, line),
                "same_page_prev": int(prev_line is not None and prev_line.page == line.page),
                "same_page_next": int(next_line is not None and next_line.page == line.page),
                "looks_numbered_start": _flag(patterns.numbered_start, line.text),
                "looks_author_year_start": _flag(patterns.author_year, line.text),
                "looks_doi_continuation": _flag(_DOI_RE, line.text),
                "prev_looks_reference_header": _flag(
                    patterns.ref_header, prev_line.text if prev_line else ""
                ),
                "next_looks_reference_header": _flag(
                    patterns.ref_header, next_line.text if next_line else ""
                ),
            }
        )
    return augmented
