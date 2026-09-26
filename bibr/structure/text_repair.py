"""Display-text repair helpers shared across the PDF-parser region mixins.

Pure string / bbox utilities extracted verbatim from
:mod:`bibr.structure.pdf_parser` so the region-handler mixins
(:mod:`~bibr.structure.parse_headings`, :mod:`~bibr.structure.parse_media`,
:mod:`~bibr.structure.parse_text`) can share them without importing the parser.

The cross-module helpers (:func:`bbox_to_tuple`,
:func:`collapse_numbered_prefix_spaces`, :func:`repair_heading_artifacts`,
:func:`strip_markdown_emphasis`) are public; ``_display_alias`` and
``_repair_study_marker_spacing`` remain private module-internal helpers.
"""

import re

from bibr.paper_contents import CANONICAL_SECTION_ALIASES


def bbox_to_tuple(bbox: list | None) -> tuple[float, float, float, float] | None:
    """Coerce glmocr ``bbox_2d`` (list[float]) to the 4-tuple Provenance expects."""
    if bbox is None or len(bbox) < 4:
        return None
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


# Matches a leading dotted-number prefix that has at least one space *inside*
# the dotted run, e.g. "3. 1", "3. 2.1", "3. 2. 1". Used to repair a spaced
# "3. 1 Encoder" back to "3.1 Encoder" before level inference runs. Most of
# these came from bibr's own OCR list-marker normalisation
# (``bibr.ocr.postprocess.clean_ocr_content``), which no longer splits a
# number; OCR output and older cached regions can still carry them.
_NUMBERED_PREFIX_WITH_SPACES_RE = re.compile(r"^(\s*\d+(?:\.\s*\d+)+)")
# Same repair for lettered-appendix sub-headings: "A. 1" -> "A.1", "B. 2. 1"
# -> "B.2.1". A single leading uppercase letter followed by a dotted-number
# run (a digit is required after each dot, so "A. Title" is left untouched).
_LETTER_PREFIX_WITH_SPACES_RE = re.compile(r"^(\s*[A-Z](?:\.\s*\d+)+)")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_HEADING_COMPACT_RE = re.compile(r"[^a-z0-9]+")
_COMPACT_ALIAS_DISPLAY: dict[str, str] = {
    _HEADING_COMPACT_RE.sub("", alias.lower()): alias
    for aliases in CANONICAL_SECTION_ALIASES.values()
    for alias in aliases
}
_ALIAS_LOWER_SET: set[str] = {
    alias.lower() for aliases in CANONICAL_SECTION_ALIASES.values() for alias in aliases
}
_COMPACT_STUDY_MARKER_RE = re.compile(
    r"^\s*(?P<label>study|experiment|exp\.?)\s*"
    r"(?P<token>\d+[a-z]?|[ivxl]+|[a-z])"
    r"(?P<rest>$|[\s:.\-–—].*)",
    re.IGNORECASE,
)

_MD_EMPHASIS_RE = re.compile(r"^(\*{1,3}|_{1,3})(.+?)\1$", re.DOTALL)


def collapse_numbered_prefix_spaces(text: str) -> str:
    """Remove stray whitespace inside a leading dotted-number/-letter prefix.

    "3. 1 Encoder"  -> "3.1 Encoder"
    "3. 2. 1 Foo"   -> "3.2.1 Foo"
    "A. 1 Proof"    -> "A.1 Proof"
    "B. 2. 1 Lemma" -> "B.2.1 Lemma"
    "3.1 Encoder"   -> "3.1 Encoder"   (unchanged)
    "3. Background" -> "3. Background" (single number — not a multi-level prefix)
    "A. Appendix"   -> "A. Appendix"   (letter + word — no dotted-number run)
    """
    m = _NUMBERED_PREFIX_WITH_SPACES_RE.match(text) or _LETTER_PREFIX_WITH_SPACES_RE.match(text)
    if m is None:
        return text
    return re.sub(r"\.\s+", ".", m.group(1)) + text[m.end() :]


def _display_alias(alias: str) -> str:
    """Human-readable display form for a canonical lowercase alias."""
    return " ".join(part.capitalize() if part.islower() else part for part in alias.split())


def _repair_study_marker_spacing(text: str) -> str:
    """Repair compact study markers such as ``Study1`` → ``Study 1``."""
    m = _COMPACT_STUDY_MARKER_RE.match(text)
    if m is None:
        return text
    label = m.group("label")
    if label.lower().startswith("exp."):
        label_display = "Exp."
    else:
        label_display = label.capitalize()
    return f"{label_display} {m.group('token')}{m.group('rest')}".strip()


def repair_heading_artifacts(text: str) -> str:
    """Clean display-only heading artifacts left after raw text cleanup.

    Raw C0 controls are stripped before parser dispatch. When those controls
    were acting as layout separators, the surviving heading can be compacted
    (``Study1`` / ``GeneralDiscussion``). This repairs only trusted, short
    section-label shapes.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text

    text = _repair_study_marker_spacing(text)

    # A header that already matches a canonical alias is not an artifact —
    # leave it alone. Without this, compact-key collisions in the alias map
    # (e.g. "keywords" vs "key words") rewrite legitimate headers. A camel
    # boundary means the text is a fused artifact, not a plain alias match.
    if _CAMEL_BOUNDARY_RE.search(text) is None and text.lower() in _ALIAS_LOWER_SET:
        return text

    key = _HEADING_COMPACT_RE.sub("", text.lower())
    alias = _COMPACT_ALIAS_DISPLAY.get(key)
    if alias and " " in alias and " " not in text:
        return _display_alias(alias)

    split = _CAMEL_BOUNDARY_RE.sub(" ", text)
    if split != text:
        key = _HEADING_COMPACT_RE.sub("", split.lower())
        alias = _COMPACT_ALIAS_DISPLAY.get(key)
        if alias and " " in alias:
            return _display_alias(alias)

    return text


def strip_markdown_emphasis(text: str) -> str:
    """Unwrap leading+trailing markdown emphasis from a heading string.

    Handles "**References**", "__Methods__", "*Discussion*", "_Title_" by
    stripping matched-pair ``*``/``_`` markers (1-3 chars) at both ends.
    Iterates so that nested wrappers like ``**_X_**`` collapse fully.
    Strips at most a few iterations — heading text shouldn't contain
    pathologically nested emphasis.
    """
    text = text.strip()
    for _ in range(3):
        m = _MD_EMPHASIS_RE.match(text)
        if m is None:
            break
        text = m.group(2).strip()
    return text
