"""Character spans of references, links and expressions within ``text[].text``.

The exported ``xref``, ``url`` and ``eq`` rows point at a whole sentence by
``text_id``; ``start``/``end`` narrow that to the characters that printed the
item (0-based, end exclusive, in Unicode code points of the exported ``text``
string). Detectors that matched the item in the sentence record the span, and
the exporter checks it here; otherwise the item's printed form is located in
the sentence. A span that cannot be verified or located unambiguously is left
``None`` rather than guessed.

All patterns are escaped literals joined by ``\\s*``/``\\s+``, so matching is
linear in the sentence length.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Span = tuple[int, int]

_WS = re.compile(r"\s+")


def _squash(value: str) -> str:
    return _WS.sub("", value)


def verified_span(text: str, start: int | None, end: int | None, printed: str) -> Span | None:
    """(start, end) when it lies in *text* and covers *printed* (ignoring whitespace)."""
    if start is None or end is None or not 0 <= start < end <= len(text):
        return None
    return (start, end) if _squash(text[start:end]) == _squash(printed) else None


def _whitespace_flexible(printed: str) -> re.Pattern[str] | None:
    """*printed* as a pattern where every run of whitespace matches any run."""
    tokens = printed.split()
    if not tokens:
        return None
    return re.compile(r"\s+".join(re.escape(token) for token in tokens))


def _gap_flexible(printed: str) -> re.Pattern[str] | None:
    """*printed* as a pattern allowing whitespace between any two characters.

    For URLs, whose export form has line-wrap whitespace removed.
    """
    chars = [c for c in printed if not c.isspace()]
    if not chars:
        return None
    return re.compile(r"\s*".join(re.escape(c) for c in chars))


# Printed spellings of each normalized comparator (see
# ``bibr.extract.equation_extractor._normalize_comp``).
_COMPARATORS = {
    "≤": r"(?:≤|<=|⩽)",
    "≥": r"(?:≥|>=|⩾)",
    "≪": r"(?:≪|<<)",
    "≫": r"(?:≫|>>)",
}


def equation_pattern(lhs: str, df: str | None, comp: str, rhs: str) -> re.Pattern[str] | None:
    """Pattern for one parsed expression as it may be printed, e.g. ``t(28) = 2.10``."""
    parts = [_whitespace_flexible(v) for v in (lhs, comp, rhs)]
    if any(p is None for p in parts):
        return None
    lhs_p, _, rhs_p = parts
    comp_p = _COMPARATORS.get(comp.strip(), re.escape(comp.strip()))
    df_p = ""
    if df:
        df_inner = _whitespace_flexible(df)
        if df_inner is None:
            return None
        df_p = rf"\s*[(\[]\s*{df_inner.pattern}\s*[)\]]"
    return re.compile(rf"{lhs_p.pattern}{df_p}\s*{comp_p}\s*{rhs_p.pattern}")  # type: ignore[union-attr]


class SpanLocator:
    """Assigns spans to items of one kind across the whole paper.

    ``shared`` items that print identically in one sentence share the first
    occurrence (several bibliography targets of one ``[1, 2]`` citation);
    otherwise successive identical items take successive occurrences.
    """

    def __init__(self, texts: dict[int, str], *, shared: bool) -> None:
        self._texts = texts
        self._shared = shared
        self._taken: dict[tuple[int, str], int] = {}

    def locate(
        self,
        text_id: int,
        key: str,
        pattern: Callable[[], re.Pattern[str] | None],
    ) -> Span | None:
        text = self._texts.get(text_id)
        if text is None:
            return None
        compiled = pattern()
        if compiled is None:
            return None
        matches = [m.span() for m in compiled.finditer(text)]
        index = 0 if self._shared else self._taken.get((text_id, key), 0)
        if index >= len(matches):
            return None
        self._taken[(text_id, key)] = index + 1
        return matches[index]


def xref_span(locator: SpanLocator, texts: dict[int, str], xref) -> Span | None:
    """Span of one ``PaperXref``: its recorded span if it checks out, else located.

    A footnote reference is never located: its mark is not in the sentence
    text (a DOCX note mark is a field, and bibr does not find PDF marks), so a
    search for "1" would land on any 1 in the sentence.
    """
    contents = xref.contents or ""
    text = texts.get(xref.text_id)
    if text is not None:
        recorded = verified_span(
            text, getattr(xref, "start", None), getattr(xref, "end", None), contents
        )
        if recorded is not None:
            return recorded
    if getattr(xref, "xref_type", None) == "foot":
        return None
    return locator.locate(xref.text_id, contents, lambda: _whitespace_flexible(contents))


def url_span(locator: SpanLocator, link, href: str) -> Span | None:
    """Span of one link: its visible text when that differs from the URL, else the URL."""
    shown = (link.link_text or "").strip()
    if shown and shown != href:
        return locator.locate(link.text_id, "text:" + shown, lambda: _whitespace_flexible(shown))
    return locator.locate(link.text_id, "url:" + href, lambda: _gap_flexible(href))


def equation_span(locator: SpanLocator, eq) -> Span | None:
    """Span of one parsed expression located by its printed components."""
    key = "\x1f".join((eq.lhs, eq.df or "", eq.comp, eq.rhs))
    return locator.locate(eq.text_id, key, lambda: equation_pattern(eq.lhs, eq.df, eq.comp, eq.rhs))
