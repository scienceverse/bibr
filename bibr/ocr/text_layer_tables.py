"""Rebuild a table region from text-layer glyph positions, with no OCR model.

The text-layer backend (:mod:`bibr.ocr.text_layer`) reads a table region by
geometry alone:

- glyphs join into words in the text layer's own order, split at its
  whitespace and wherever a glyph leaves the line or the gap to it is wider
  than a word space;
- words are grouped into rows by clustering their vertical centres;
- a row's words form phrases: runs of words with no gap of separator width
  between them;
- columns are the x-ranges between separators, where a separator is a run of
  empty horizontal space that almost every row leaves empty. Only rows of two
  or more phrases vote: a title, a note or a group header printed as one
  phrase ("Note. Values are means ...") spans columns and says nothing about
  where they are. A voting row may still cross a separator (a long label, a
  group header over two columns);
- each word goes to the column holding its centre, except in a row of one
  phrase that a separator cuts through a word of: that row spans columns,
  and its text goes whole into the column it starts in, as a merged cell
  would. In a row of several phrases a crossing phrase is split word by
  word, since it is more often two cells set close than one wide cell.

The result is plain ``<table><tr><td>`` HTML, which the table parser
(:mod:`bibr.structure.parse_media`) reads like an OCR model's table. There
are no merged cells, and a cell whose text wraps onto a second line becomes a
row of its own.

Coordinates are the 0..1000 layout space of the rendered page (y down), as
:func:`bibr.ocr.text_layer.read_text_layer` passes them.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

#: One char: its text and box ``(x0, y0, x1, y1)`` in layout space, y down.
Glyph = tuple[str, float, float, float, float]

# A word whose vertical centre is within this share of the median glyph
# height of the current row's centre joins the row (superscripts sit about a
# third of a height up).
_ROW_CENTRE_TOLERANCE = 0.5
# A gap between two glyphs wider than this share of the median glyph height
# ends a word even where the text layer has no space. Tight glyph boxes leave
# a third of a height between digits ("1" is narrow), so this is wider.
_WORD_GAP = 0.8
# A column separator is at least this share of the median glyph height wide.
_MIN_COLUMN_GAP = 0.8
# A separator may be crossed by words in at most this share of the voting
# rows (a long label, a group header over two columns) ...
_MAX_CROSSING_ROW_SHARE = 0.15
# ... and by one of them in any table of at least this many voting rows, so
# a small table keeps its columns under one such row.
_MIN_ROWS_FOR_ONE_CROSSING = 3


@dataclass
class _Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def x_centre(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def y_centre(self) -> float:
        return (self.y0 + self.y1) / 2.0


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[len(ordered) // 2]


def _words(chars: Sequence[Glyph], height: float) -> list[_Word]:
    """Words in text-layer order: split at whitespace, line changes and wide gaps."""
    words: list[_Word] = []
    current: _Word | None = None
    for ch, x0, y0, x1, y1 in chars:
        if not ch or ch.isspace() or x1 <= x0 or y1 <= y0:
            current = None
            continue
        if current is not None:
            same_line = abs((y0 + y1) / 2.0 - current.y_centre) <= _ROW_CENTRE_TOLERANCE * height
            gap = x0 - current.x1
            if not same_line or gap > _WORD_GAP * height or gap < -_WORD_GAP * height:
                current = None
        if current is None:
            current = _Word(ch, x0, y0, x1, y1)
            words.append(current)
        else:
            current.text += ch
            current.x1 = max(current.x1, x1)
            current.y0 = min(current.y0, y0)
            current.y1 = max(current.y1, y1)
    return words


def _rows(words: list[_Word], height: float) -> list[list[_Word]]:
    """Words grouped into rows, top to bottom, each sorted left to right."""
    rows: list[list[_Word]] = []
    centre = 0.0
    for word in sorted(words, key=lambda w: w.y_centre):
        if rows and abs(word.y_centre - centre) <= _ROW_CENTRE_TOLERANCE * height:
            rows[-1].append(word)
            centre = sum(w.y_centre for w in rows[-1]) / len(rows[-1])
        else:
            rows.append([word])
            centre = word.y_centre
    return [sorted(row, key=lambda w: w.x0) for row in rows]


def _phrases(row: list[_Word], height: float) -> list[list[_Word]]:
    """The row's words in runs split at gaps as wide as a column separator.

    Measured on the same integer grid as :func:`_column_bounds`, so a row of
    two or more phrases could show a separator on its own.
    """
    phrases: list[list[_Word]] = []
    reach = 0.0
    for word in row:
        if phrases and int(word.x0) - int(reach) - 1 < _MIN_COLUMN_GAP * height:
            phrases[-1].append(word)
            reach = max(reach, word.x1)
        else:
            phrases.append([word])
            reach = word.x1
    return phrases


def _column_bounds(rows: list[list[_Word]], height: float) -> list[float]:
    """The x positions that separate columns, left to right."""
    words = [word for row in rows for word in row]
    left = int(min(word.x0 for word in words))
    right = int(max(word.x1 for word in words)) + 1
    width = right - left
    if width <= 0:
        return []
    voters = [row for row in rows if len(_phrases(row, height)) > 1]
    if len(voters) < 2:
        # Too few rows show a column gap to place columns by vote alone.
        voters = rows
    # Per x unit, how many voting rows have a word over it.
    crossing = [0] * width
    for row in voters:
        covered = [False] * width
        for word in row:
            start = max(0, int(word.x0) - left)
            stop = min(width, int(word.x1) - left + 1)
            for x in range(start, stop):
                covered[x] = True
        for x, hit in enumerate(covered):
            if hit:
                crossing[x] += 1
    allowed = int(_MAX_CROSSING_ROW_SHARE * len(voters))
    if len(voters) >= _MIN_ROWS_FOR_ONE_CROSSING:
        allowed = max(1, allowed)
    bounds: list[float] = []
    run_start: int | None = None
    for x, count in enumerate(crossing):
        if count <= allowed:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            # Only a gap with words on both sides separates columns.
            if run_start > 0 and x - run_start >= _MIN_COLUMN_GAP * height:
                bounds.append(left + (run_start + x) / 2.0)
            run_start = None
    return bounds


def _column(x: float, bounds: list[float]) -> int:
    return sum(1 for bound in bounds if x > bound)


def _spans_columns(phrases: list[list[_Word]], bounds: list[float]) -> bool:
    """Whether the row is one phrase that a separator cuts through a word of."""
    if len(phrases) != 1:
        return False
    return any(word.x0 < bound < word.x1 for word in phrases[0] for bound in bounds)


def build_table_html(chars: Sequence[Glyph]) -> str | None:
    """Plain HTML for the table the chars form, or None when they hold no text.

    *chars* are the region's chars in text-layer order, whitespace included,
    in layout space. Every row becomes a ``<tr>`` of ``<td>`` cells; no row
    is promoted to a header, so the table parser decides that as it does for
    OCR output.
    """
    visible = [g for g in chars if g[0] and not g[0].isspace() and g[3] > g[1] and g[4] > g[2]]
    if not visible:
        return None
    height = _median([g[4] - g[2] for g in visible]) or 1.0
    rows = _rows(_words(chars, height), height)
    bounds = _column_bounds(rows, height)
    out = ["<table>"]
    for row in rows:
        cells: list[list[str]] = [[] for _ in range(len(bounds) + 1)]
        if _spans_columns(_phrases(row, height), bounds):
            # One cell, in the column the row starts in.
            cells[_column(row[0].x_centre, bounds)].append(" ".join(w.text for w in row))
        else:
            for word in row:
                cells[_column(word.x_centre, bounds)].append(word.text)
        out.append("<tr>")
        out.extend(f"<td>{html.escape(' '.join(cell))}</td>" for cell in cells)
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
