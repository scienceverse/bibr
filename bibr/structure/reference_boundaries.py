"""High-precision structural boundaries at the end of reference lists."""

import re

# Exact publisher boilerplate, not a bare keyword match. A bibliography can
# legitimately contain titles beginning with "Publisher's note on ..."; the
# Springer neutrality clause is the evidence that a plain content row is back
# matter. The shorter whole-heading form remains available to RefLocator,
# which additionally requires two credible reference starts before cutting.
_PUBLISHER_NOTE_STRONG_PATTERN = (
    r"publisher[’']s note\s+springer nature remains neutral with regard to\b"
)
_PUBLISHER_NOTE_SHORT_HEADING_PATTERN = r"publisher[’']s note(?:\s*:\s*$|\s*$)"

PUBLISHER_NOTE_BOUNDARY_RE = re.compile(
    rf"^\s*(?:{_PUBLISHER_NOTE_STRONG_PATTERN})",
    re.IGNORECASE,
)

# Other explicit terminal headings consumed by RefLocator. These remain
# row-initial and whole-heading shaped so reference titles containing the same
# words do not become cuts.
TERMINAL_REFERENCE_BOUNDARY_RE = re.compile(
    rf"^\s*(?:"
    rf"{_PUBLISHER_NOTE_STRONG_PATTERN}"
    rf"|{_PUBLISHER_NOTE_SHORT_HEADING_PATTERN}"
    r"|authors?\s+and\s+affiliations?\s*:?\s*$"
    r"|(?:about\s+the\s+)?authors?\s+(?:information|biograph(?:y|ies)|details)\s*:?\s*$"
    r"|affiliations?\s*:?\s*$"
    r"|(?:article|additional)\s+information\s*:?\s*$"
    r"|correspondence\s+and\s+requests\s*:?\s*$"
    r")",
    re.IGNORECASE,
)


def is_publisher_note_boundary(text: str) -> bool:
    """Return whether *text* is unmistakable terminal publisher boilerplate."""
    return bool(PUBLISHER_NOTE_BOUNDARY_RE.match(text))
