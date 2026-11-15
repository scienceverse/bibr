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

from bibr.extract.equation_extractor import repair_statistic_glitches

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


# A value printed on past a parsed rhs: its digits or decimals continue, or a
# range does ("n = 5" is not the start of "n = 5–6")
_VALUE_CONTINUES = r"(?!\d|\.\d|[−–-]\d)"


def equation_pattern(
    lhs: str, df: str | None, comp: str, rhs: str, *, fused_lhs: bool = False
) -> re.Pattern[str] | None:
    """Pattern for one parsed expression as it may be printed, e.g. ``t(28) = 2.10``.

    The rhs's tokens may be printed closer together than parsed: late
    clean-up prints the "10 − 6" of "9.62 × 10 − 6" as "10 −6". With
    *fused_lhs*, so may the lhs's, but not all as far apart as parsed: late
    clean-up prints the "η p 2" that the extractor read from MathML as "ηp2".
    """
    lhs_tokens = lhs.split()
    rhs_tokens = rhs.split()
    if not lhs_tokens or not comp.strip() or not rhs_tokens:
        return None
    lhs_p = r"\s+".join(re.escape(token) for token in lhs_tokens)
    if fused_lhs:
        lhs_p = rf"(?!{lhs_p})" + r"\s*".join(re.escape(token) for token in lhs_tokens)
    rhs_p = r"\s*".join(re.escape(token) for token in rhs_tokens)
    comp_p = _COMPARATORS.get(comp.strip(), re.escape(comp.strip()))
    df_p = ""
    if df:
        df_inner = _whitespace_flexible(df)
        if df_inner is None:
            return None
        df_p = rf"\s*[(\[]\s*{df_inner.pattern}\s*[)\]]"
    return re.compile(rf"{lhs_p}{df_p}\s*{comp_p}\s*{rhs_p}{_VALUE_CONTINUES}")


def _read_text(text: str) -> tuple[str, list[int] | None]:
    """*text* as the equation passes read it, and where each character of it lies in *text*.

    The passes read each line break, ``"\\r\\n"`` or ``"\\r"``, as one ``"\\n"``
    (``_single_char_line_breaks``). ``offsets[i]`` is the position in *text* of
    the character ``i`` of the result, and ``offsets[len(result)]`` the length
    of *text*; ``None`` when the two are the same string.
    """
    if "\r" not in text:
        return text, None
    out: list[str] = []
    offsets: list[int] = []
    i = 0
    while i < len(text):
        offsets.append(i)
        if text[i] == "\r":
            out.append("\n")
            i += 2 if text.startswith("\r\n", i) else 1
        else:
            out.append(text[i])
            i += 1
    offsets.append(len(text))
    return "".join(out), offsets


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
        self._repaired: dict[int, tuple[str, str, list[int] | None]] = {}

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

    def locate_repaired(
        self,
        text_id: int,
        key: str,
        pattern: Callable[[], re.Pattern[str] | None],
    ) -> Span | None:
        """Like ``locate``, among the matches that are there once glitches are repaired.

        An expression read through ``repair_statistic_glitches`` ("t(59) = 0.66"
        of the printed "t(59) 5 0.66") is not in the sentence as read. The
        repaired text has as many characters, so a span found in it is a span of
        the printed one (once a line break the passes read as one character is
        mapped back to the two it was printed with). Only matches the printed text
        does not hold are taken, since ``locate`` has placed those.
        """
        text = self._texts.get(text_id)
        compiled = pattern()
        if text is None or compiled is None:
            return None
        if text_id not in self._repaired:
            read, offsets = _read_text(text)
            self._repaired[text_id] = (read, repair_statistic_glitches(read), offsets)
        read, repaired, offsets = self._repaired[text_id]
        if repaired == read:
            return None
        printed = {m.span() for m in compiled.finditer(text)}
        matches = []
        for found in compiled.finditer(repaired):
            span = (
                found.span() if offsets is None else (offsets[found.start()], offsets[found.end()])
            )
            if span not in printed:
                matches.append(span)
        taken = (text_id, key + "\x1frepaired")
        index = self._taken.get(taken, 0)
        if index >= len(matches):
            return None
        self._taken[taken] = index + 1
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
    """Span of one parsed expression located by its printed components.

    A name with spaces that is not printed with them is looked for printed
    closer together (see ``equation_pattern``), among the places where it is
    not printed with them, so a row placed as parsed keeps its place.

    A row read through a repaired glitch ("t(59) = 0.66" for "t(59) 5 0.66";
    ``eq.repaired``) is last looked for in the repaired sentence, which has the
    span of the printed one. A row that was not read that way, an LLM's among
    them, is looked for as printed only.
    """
    key = "\x1f".join((eq.lhs, eq.df or "", eq.comp, eq.rhs))

    def as_parsed():
        return equation_pattern(eq.lhs, eq.df, eq.comp, eq.rhs)

    def fused():
        return equation_pattern(eq.lhs, eq.df, eq.comp, eq.rhs, fused_lhs=True)

    spaced = len(eq.lhs.split()) > 1
    span = locator.locate(eq.text_id, key, as_parsed)
    if span is None and spaced:
        span = locator.locate(eq.text_id, key + "\x1ffused", fused)
    if span is None and getattr(eq, "repaired", False):
        span = locator.locate_repaired(eq.text_id, key, as_parsed)
        if span is None and spaced:
            span = locator.locate_repaired(eq.text_id, key + "\x1ffused", fused)
    return span
