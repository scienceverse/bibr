"""Flattened element text, with the whitespace between MathML elements
resolved the way a renderer treats it, but without fusing words.

Publishers pretty-print MathML: PLOS and eLife put a space or a newline
between elements (``<mi>t</mi> <mo>-</mo> <mn>1</mn>``). A renderer ignores
that whitespace, since only token elements (``mi``, ``mn``, ``mo``,
``mtext``, ``ms``) hold content. Kept as text it splits a decimal into
"2 . 5" and an index into "t - 1", while the same formula from a publisher
that writes no whitespace reads "2.5" and "t-1".

Dropped outright it fuses words, which a renderer spaces by other means or
which the source spells one letter per element: "ln dbh" would read "lndbh",
"0.93 GeV" "0.93GeV" and "direct effect" (Word's export, ``<mi>d</mi><mi>i</mi>…``)
"directeffect". So whitespace-only text inside ``<math>`` is held as a gap and
resolved against its neighbours:

* next to text outside the formula, the space is kept;
* between two letters or digits, it is kept when either side is a word: an
  ``mtext``/``ms`` with a letter, or two letters or more in ``mi``/``mn``/
  ``mo`` elements, in one ("ln", "GeV") or spelled across siblings ("direct");
* between two digits, it is always kept: the parts of a fraction read "1 2",
  not "12", and a coefficient and its root "3 2", while an index pair
  pretty-printed as ``<msubsup><mi>g</mi> <mn>1</mn> <mn>1</mn></msubsup>``
  reads "g1 1";
* after the invisible function application, it is kept before a letter or a
  digit ("sin x", where a renderer spaces the name from its argument) but not
  before a bracket ("sin(x)");
* after a comma, a semicolon, a colon or a closing bracket, it is kept before
  an ``mtext``/``ms`` word (", and", ") for");
* never before closing punctuation or after an opening bracket, and nowhere
  else: operators and punctuation close up ("SD=1.2", "a_ij = 5" reads
  "aij=5"), as the formula reads from a publisher that writes no whitespace.

An ``<mspace>`` shows as space unless its width is negative (``\\quad`` is
``<mspace width="1em"/>``), so the parsers treat it as a word boundary, like
a matrix cell (:func:`mspace_separates`).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

# XML whitespace. A no-break space is content, not formatting.
_XML_WHITESPACE = " \t\r\n"
_TOKENS = frozenset({"mi", "mn", "mo", "mtext", "ms"})
_TEXT_TOKENS = frozenset({"mtext", "ms"})
# Invisible operators (function application, times, separator, plus) attach
# to what comes before them, and all but function application to what comes
# after.
_APPLY_FUNCTION = "\u2061"
_INVISIBLE = _APPLY_FUNCTION + "\u2062\u2063\u2064"
_NO_SPACE_AFTER = "([{⟨〈\u2062\u2063\u2064"
_NO_SPACE_BEFORE = ")]}⟩〉,.;:!?" + _INVISIBLE
_SPACE_BEFORE_TEXT = ",;:)]}⟩〉"

_NONE, _LETTERS, _TEXT, _PROSE = 0, 1, 2, 3


class FlatText:
    """Text pieces in document order; see the module docstring for gaps.

    ``parts`` holds the pieces. A gap is stored as ``" "`` so that a caller's
    "does the text end in whitespace" check reads it as the whitespace it was,
    and :meth:`join` decides what it becomes.

    ``settled`` counts the leading pieces whose rendering no later piece can
    change: the last gap is still open, since one at the end waits for the
    piece after it and one between tokens reads the tokens after it, up to the
    next gap (:meth:`_side`). ``text_end`` is the index after the last piece
    that is not a gap.
    """

    __slots__ = ("_gaps", "_groups", "_lefts", "_words", "parts", "settled", "text_end")

    def __init__(self) -> None:
        self.parts: list[str] = []
        self._gaps: set[int] = set()
        # Pieces held by MathML token elements, mapped to the parent element:
        # siblings may spell a word together. Every other piece is prose.
        self._groups: dict[int, Hashable] = {}
        # Token pieces that are words by themselves (mtext/ms with a letter).
        self._words: set[int] = set()
        # How the text left of a gap reads (:meth:`_side`), by the piece before
        # the gap: nothing added later changes it, and a caller joining part of
        # the text again would read a long run of digits again.
        self._lefts: dict[int, int] = {}
        self.settled = self.text_end = 0

    def add(self, text: str) -> None:
        """Add text read outside ``<math>``."""
        self.parts.append(text)
        self.settled = self.text_end = len(self.parts)

    def add_math(self, text: str, token: str | None, group: Hashable) -> None:
        """Add text read inside ``<math>``.

        *token* is the local name of the element holding the text (``None``
        for text between elements) and *group* identifies that element's
        parent, so sibling tokens can be read together.
        """
        if not text.strip(_XML_WHITESPACE):
            self._gaps.add(len(self.parts))
            self.parts.append(" ")
            self.settled = self.text_end  # the gaps before it read no further
            return
        if token in _TOKENS:
            self._groups[len(self.parts)] = group
            if token in _TEXT_TOKENS and any(ch.isalpha() for ch in text):
                self._words.add(len(self.parts))
            if self.settled < len(self.parts):  # a token after an open gap
                self.parts.append(text)
                self.text_end = len(self.parts)
                return
        self.parts.append(text)
        self.settled = self.text_end = len(self.parts)

    def separate(self) -> None:
        """Append a word separator unless the text already ends in whitespace.

        A trailing gap stands in for that whitespace, so it becomes a real
        space instead: the separator it replaced must not be lost.
        """
        if not self.parts:
            return
        last = len(self.parts) - 1
        if last in self._gaps:
            self._gaps.discard(last)
        elif not self.parts[last][-1:].isspace():
            self.parts.append(" ")
        else:
            return
        self.settled = self.text_end = len(self.parts)

    def _side(self, index: int, step: int) -> int:
        """How the text next to a gap reads, from piece *index* outward
        (*step* -1 for the left side, +1 for the right)."""
        group = self._groups.get(index, self)
        if group is self:
            return _PROSE
        if index in self._words:
            return _TEXT
        letters = 0
        while 0 <= index < len(self.parts) and index not in self._gaps:
            if self._groups.get(index, self) != group:
                break
            text = self.parts[index]
            for ch in text if step > 0 else reversed(text):
                if not ch.isalnum():
                    return _LETTERS if letters >= 2 else _NONE
                letters += ch.isalpha()
            if letters >= 2:
                return _LETTERS
            index += step
        return _NONE

    def _keeps_space(self, left: int, right: int) -> bool:
        before, after = self.parts[left][-1:], self.parts[right][:1]
        if not before or not after or before in _NO_SPACE_AFTER or after in _NO_SPACE_BEFORE:
            return False
        side = self._lefts.get(left)
        if side is None:
            side = self._lefts[left] = self._side(left, -1)
        sides = (side, self._side(right, 1))
        if _PROSE in sides:
            return True
        if before in _SPACE_BEFORE_TEXT:
            return sides[1] == _TEXT
        if before == _APPLY_FUNCTION:
            return after.isalnum()
        if before.isdigit() and after.isdigit():
            return True
        return max(sides) >= _LETTERS and before.isalnum() and after.isalnum()

    def join(self, start: int = 0, stop: int | None = None) -> str:
        """The text, or what the pieces from *start* to *stop* add to it when
        neither bound follows a gap (as ``settled`` and ``text_end`` do not)."""
        if not self._gaps:
            return "".join(self.parts[start:stop])
        out: list[str] = []
        left = start - 1  # the last piece that is not a gap
        pending = False  # a run of gaps waits for the piece after it
        for index in range(start, len(self.parts) if stop is None else stop):
            text = self.parts[index]
            if index in self._gaps:
                pending = True
                continue
            if pending and left >= 0 and self._keeps_space(left, index):
                out.append(" ")
            pending = False
            out.append(text)
            left = index
        return "".join(out)


def mspace_separates(attributes: Mapping[str, object]) -> bool:
    """Whether an ``<mspace>`` with these *attributes* separates the text
    around it: unless its width is negative (``negativethinmathspace``,
    ``-0.17em``), which pulls its neighbours together."""
    width = str(attributes.get("width", "")).strip().lower()
    return not width.startswith(("-", "negative"))
