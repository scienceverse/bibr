"""Model-specific OCR output normalization."""

import re
from dataclasses import dataclass

from bibr.ocr.otsl import decode_otsl
from bibr.ocr.profiles import OcrProfile, OcrTask
from bibr.processing_warnings import ProcessingWarning


@dataclass(frozen=True)
class NormalizedOcrOutput:
    """Canonical OCR content with optional source preservation."""

    content: str
    raw_content: str | None
    warnings: tuple[ProcessingWarning, ...] = ()


def normalize_ocr_output(
    profile: OcrProfile, task: OcrTask, raw_content: str
) -> NormalizedOcrOutput:
    """Normalize raw model output according to its model profile."""
    if profile.name == "glm":
        return NormalizedOcrOutput(content=raw_content, raw_content=None)

    content = raw_content
    if profile.name == "paddle" and task == "table":
        decoded = decode_otsl(content)
        content = decoded.html
        return NormalizedOcrOutput(
            content=content,
            raw_content=raw_content,
            warnings=decoded.warnings,
        )
    elif task == "formula":
        content = _strip_markdown_fence(content)
        content = strip_one_balanced_formula_wrapper(content)

    return NormalizedOcrOutput(content=content, raw_content=raw_content)


def _strip_markdown_fence(value: str) -> str:
    """Remove a single outer latex/tex fence when it encloses the whole value."""
    trimmed = value.strip()
    opening_end = trimmed.find("\n")
    if opening_end == -1:
        return trimmed

    opening = trimmed[:opening_end].rstrip("\r")
    closing_start = trimmed.rfind("\n") + 1
    if opening not in {"```latex", "```tex"} or trimmed[closing_start:] != "```":
        return trimmed

    body = trimmed[opening_end + 1 : closing_start]
    if any(line.strip() == "```" for line in body.splitlines()):
        return trimmed
    return body.strip()


_FORMULA_DELIMITERS = ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"))

# A dollar sign behind an odd number of backslashes is a literal ``\$``
# ("\text{cost} = \$5"), not a math delimiter.
_ESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)((?:\\\\)*)\\\$")


def strip_one_balanced_formula_wrapper(value: str, *, single_dollar: bool = False) -> str:
    """Remove one matching outer formula delimiter without touching LaTeX bytes.

    The wrapper is removed only when its opening delimiter is closed by the
    final one: ``\\(a\\) + \\(b\\)`` is two formulas, not one wrapped formula.
    ``single_dollar`` also accepts one inline ``$…$`` pair, which the OCR stage
    would otherwise nest inside the ``$$`` it wraps formula regions in. An
    escaped ``\\$`` is a dollar sign in the formula and never counts as a
    delimiter.
    """
    trimmed = value.strip()
    # Match delimiters on a copy where each escaped dollar is blanked out; it
    # keeps every index, so the wrapper is sliced off the original.
    probe = _ESCAPED_DOLLAR_RE.sub(lambda m: m.group(1) + "\x00\x00", trimmed)
    delimiters = (*_FORMULA_DELIMITERS, ("$", "$")) if single_dollar else _FORMULA_DELIMITERS
    if not all(
        _has_balanced_delimiters(probe, opening, closing) for opening, closing in delimiters
    ):
        return trimmed

    for opening, closing in delimiters:
        if _has_balanced_outer_pair(probe, opening, closing):
            return trimmed[len(opening) : -len(closing)].strip()
    return trimmed


def _has_balanced_outer_pair(value: str, opening: str, closing: str) -> bool:
    """Return whether the first delimiter is matched only by the final one."""
    if not value.startswith(opening) or not value.endswith(closing):
        return False

    depth = 0
    index = 0
    while index < len(value):
        if value.startswith(opening, index):
            depth = 1 - depth if opening == closing else depth + 1
            index += len(opening)
        elif opening != closing and value.startswith(closing, index):
            depth -= 1
            index += len(closing)
        else:
            index += 1

        if depth < 0 or (depth == 0 and index != len(value)):
            return False

    return depth == 0


def _has_balanced_delimiters(value: str, opening: str, closing: str) -> bool:
    """Return whether one formula delimiter family is balanced throughout a value."""
    if opening == closing:
        return value.count(opening) % 2 == 0

    depth = 0
    index = 0
    while index < len(value):
        if value.startswith(opening, index):
            depth += 1
            index += len(opening)
        elif value.startswith(closing, index):
            depth -= 1
            index += len(closing)
        else:
            index += 1

        if depth < 0:
            return False

    return depth == 0
