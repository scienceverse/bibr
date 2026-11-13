"""Rebuild a table region from text-layer glyph positions, with no OCR model.

The text-layer backend (:mod:`bibr.ocr.text_layer`) reads a table region by
geometry alone:

- a table printed turned a quarter (its text running up or down the page) is
  turned upright first;
- glyphs join into words in the text layer's own order, split at its
  whitespace and wherever a glyph leaves the line or the gap to it is wider
  than a word space;
- words are grouped into lines by clustering their vertical centres; a small
  mark (a comma, a full stop) joins the line whose band it sits in;
- a line's words form phrases: runs of words with no gap of separator width
  between them;
- columns are the x-ranges between separators, where a separator is a run of
  empty horizontal space that almost every line leaves empty. Only lines of
  two or more phrases vote: a title, a note or a group header printed as one
  phrase ("Note. Values are means ...") spans columns and says nothing about
  where they are. A voting line may still cross a separator (a long label, a
  group header over two columns);
- lines join into rows: a line that sits closer under the text above it in
  its column than the table's rows sit to each other continues that cell
  (wrapped text). A line of numbers always starts a row, and a number never
  continues a cell that holds one;
- each word goes to the column holding its centre, except in a line of one
  phrase that a separator cuts through a word of: that line spans columns,
  and its text goes whole into the column it starts in, as a merged cell
  would. In a line of several phrases a crossing phrase is split word by
  word, since it is more often two cells set close than one wide cell.

The result is plain ``<table><tr><td>`` HTML, which the table parser
(:mod:`bibr.structure.parse_media`) reads like an OCR model's table. There
are no merged cells: a label printed once beside several rows sits in the
first of them.

Coordinates are the 0..1000 layout space of the rendered page (y down), as
:func:`bibr.ocr.text_layer.read_text_layer` passes them. Gaps along a line are
measured in x units, so the page's shape (*aspect*) converts glyph heights to
them.
"""

from __future__ import annotations

import html
import re
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
# Height over width of an A4 page, the shape the gaps above were set on.
_A4_ASPECT = 2**0.5
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


def _words(chars: Sequence[Glyph], height: float, width: float) -> list[_Word]:
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
            if not same_line or gap > _WORD_GAP * width or gap < -_WORD_GAP * width:
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


def _lines(words: list[_Word], height: float) -> list[list[_Word]]:
    """Words grouped into printed lines, top to bottom, each sorted left to right.

    Lines are built from the words of full height; a word shorter than half
    the median glyph height (a full stop, a comma, a dash) then joins the line
    whose band it sits in, since its own centre sits low or high.
    """
    tall = [w for w in words if w.y1 - w.y0 >= 0.5 * height]
    small = [w for w in words if w.y1 - w.y0 < 0.5 * height]
    rows: list[list[_Word]] = []
    centre = 0.0
    for word in sorted(tall, key=lambda w: w.y_centre):
        if rows and abs(word.y_centre - centre) <= _ROW_CENTRE_TOLERANCE * height:
            rows[-1].append(word)
            centre = sum(w.y_centre for w in rows[-1]) / len(rows[-1])
        else:
            rows.append([word])
            centre = word.y_centre
    bands = [(min(w.y0 for w in row), max(w.y1 for w in row)) for row in rows]
    for word in small:
        best, best_d = None, None
        for i, (top, bottom) in enumerate(bands):
            if top - 0.25 * height <= word.y_centre <= bottom + 0.25 * height:
                d = abs(word.y_centre - (top + bottom) / 2.0)
                if best_d is None or d < best_d:
                    best, best_d = i, d
        if best is None:
            rows.append([word])
            bands.append((word.y0, word.y1))
        else:
            rows[best].append(word)
    order = sorted(range(len(rows)), key=lambda i: sum(w.y_centre for w in rows[i]) / len(rows[i]))
    return [sorted(rows[i], key=lambda w: w.x0) for i in order]


def _phrases(row: list[_Word], width: float) -> list[list[_Word]]:
    """The row's words in runs split at gaps as wide as a column separator.

    Measured on the same integer grid as :func:`_column_bounds`, so a row of
    two or more phrases could show a separator on its own.
    """
    phrases: list[list[_Word]] = []
    reach = 0.0
    for word in row:
        if phrases and int(word.x0) - int(reach) - 1 < _MIN_COLUMN_GAP * width:
            phrases[-1].append(word)
            reach = max(reach, word.x1)
        else:
            phrases.append([word])
            reach = word.x1
    return phrases


def _column_bounds(rows: list[list[_Word]], unit: float) -> list[float]:
    """The x positions that separate columns, left to right."""
    words = [word for row in rows for word in row]
    left = int(min(word.x0 for word in words))
    right = int(max(word.x1 for word in words)) + 1
    width = right - left
    if width <= 0:
        return []
    voters = [row for row in rows if len(_phrases(row, unit)) > 1]
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
            if run_start > 0 and x - run_start >= _MIN_COLUMN_GAP * unit:
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


def _rotation(visible: Sequence[Glyph]) -> int:
    """+1 when the table is printed turned a quarter anticlockwise (its text runs up the page),
    -1 when turned clockwise (text runs down), 0 when upright."""
    up = down = across = 0
    for a, b in zip(visible, visible[1:], strict=False):
        dx = (b[1] + b[3]) / 2.0 - (a[1] + a[3]) / 2.0
        dy = (b[2] + b[4]) / 2.0 - (a[2] + a[4]) / 2.0
        size = max(a[3] - a[1], a[4] - a[2], b[3] - b[1], b[4] - b[2])
        if abs(dx) > 2 * size or abs(dy) > 2 * size or (dx == 0 and dy == 0):
            continue
        if abs(dy) > abs(dx):
            up += dy < 0
            down += dy > 0
        else:
            across += 1
    pairs = up + down + across
    if pairs < 8 or across > 0.4 * pairs:
        return 0
    return 1 if up >= down else -1


def _turn(g: Glyph, turn: int) -> Glyph:
    """The glyph in the table's own upright frame (x along its text, y down its rows)."""
    ch, x0, y0, x1, y1 = g
    if turn > 0:
        return (ch, 1000.0 - y1, x0, 1000.0 - y0, x1)
    return (ch, y0, 1000.0 - x1, y1, 1000.0 - x0)


# A line closer than this many glyph heights under the text above it, in the
# same column, continues that cell: wrapped text, not a new row ...
_CELL_LEADING = 1.0
# ... and closer than this share of the gap the table's rows leave between
# them, where at least _MIN_ROW_GAPS rows show it: a table set solid wraps no
# cell, however close its lines sit. The rows that show it are the data rows
# in a table of numbers, and the lines that fill every column in a table of
# text, where only those lines are held to it.
_ROW_GAP_SHARE = 0.75
_MIN_ROW_GAPS = 2
# A cell that starts as a number: a sign, a bracket or a comparison may come first.
_NUMERIC = re.compile(r"^[-\u2212\u2013+<>\u2264\u2265\u00b1(\[]*\.?\d")

_Line = list[_Word]
_LineCells = dict[int, list[_Word]]


def _line_cells(line: _Line, bounds: list[float]) -> _LineCells:
    """The line's words by column."""
    cells: _LineCells = {}
    for word in line:
        cells.setdefault(_column(word.x_centre, bounds), []).append(word)
    return cells


def _is_data_line(cells: _LineCells) -> bool:
    """Whether at least half the line's cells after the stub column start as a number."""
    after_stub = [words for column, words in cells.items() if column != 0]
    numbers = sum(1 for words in after_stub if _NUMERIC.match(words[0].text))
    return numbers > 0 and 2 * numbers >= len(after_stub)


def _has_number(words: list[_Word]) -> bool:
    return any(_NUMERIC.match(word.text) for word in words)


def _table_rows(lines: list[_Line], bounds: list[float], height: float) -> list[list[_Line]]:
    """The lines grouped into table rows, top to bottom: a wrapped cell's lines share a row."""
    cells_of = [_line_cells(line, bounds) for line in lines]
    gaps = [0.0] + [
        (min(w.y0 for w in line) - max(w.y1 for w in prev)) / height
        for prev, line in zip(lines, lines[1:], strict=False)
    ]
    data = [_is_data_line(cells) for cells in cells_of]
    full = [len(cells) == len(bounds) + 1 for cells in cells_of]
    data_gaps = [gap for gap, is_data in zip(gaps[1:], data[1:], strict=True) if is_data]
    full_gaps = [gap for gap, is_full in zip(gaps[1:], full[1:], strict=True) if is_full]
    # The leading each line is held to: (for a full line, for any other).
    if len(data_gaps) >= _MIN_ROW_GAPS:
        solid = min(_CELL_LEADING, _ROW_GAP_SHARE * _median(data_gaps))
        leading = (solid, solid)
    elif len(full_gaps) >= _MIN_ROW_GAPS:
        leading = (min(_CELL_LEADING, _ROW_GAP_SHARE * _median(full_gaps)), _CELL_LEADING)
    else:
        leading = (_CELL_LEADING, _CELL_LEADING)
    rows: list[list[_Line]] = []
    bottoms: dict[int, float] = {}
    numbers: dict[int, bool] = {}
    for line, cells, is_data, is_full in zip(lines, cells_of, data, full, strict=True):
        continues = False
        if rows and not is_data:
            top = min(w.y0 for w in line)
            above = [top - bottoms[column] for column in cells if column in bottoms]
            gap = min(above) if above else top - max(bottoms.values())
            clash = any(
                numbers.get(column) and _has_number(words) for column, words in cells.items()
            )
            continues = not clash and gap < leading[0 if is_full else 1] * height
        if not continues:
            rows.append([])
            bottoms, numbers = {}, {}
        rows[-1].append(line)
        for column, words in cells.items():
            bottoms[column] = max(bottoms.get(column, words[0].y1), *(w.y1 for w in words))
            numbers[column] = numbers.get(column, False) or _has_number(words)
    return rows


def _cell_text(lines: list[str]) -> str:
    """One cell's text from its lines, with words broken at a line end joined again."""
    if len(lines) == 1:
        return lines[0]
    from bibr.input.consolidate_text import _dehyphenate_word_linewraps, _resolve_stx_marks

    return _dehyphenate_word_linewraps(_resolve_stx_marks("\n".join(lines))).replace("\n", " ")


def build_table_html(chars: Sequence[Glyph], aspect: float | None = None) -> str | None:
    """Plain HTML for the table the chars form, or None when they hold no text.

    *chars* are the region's chars in text-layer order, whitespace included,
    in layout space. *aspect* is the rendered page's height over its width
    (A4 when not given). Every row becomes a ``<tr>`` of ``<td>`` cells; no
    row is promoted to a header, so the table parser decides that as it does
    for OCR output.
    """
    visible = [g for g in chars if g[0] and not g[0].isspace() and g[3] > g[1] and g[4] > g[2]]
    if not visible:
        return None
    turn = _rotation(visible)
    if turn:
        chars = [_turn(g, turn) for g in chars]
        visible = [_turn(g, turn) for g in visible]
    height = _median([g[4] - g[2] for g in visible]) or 1.0
    # Layout space is 0..1000 on both axes, so on an upright page an x unit is
    # shorter than a y unit. Gaps along a line are measured against the glyph
    # height in x units; the gap constants were set on A4 pages.
    width = height * (aspect or _A4_ASPECT) / _A4_ASPECT
    if turn:
        # Turned, the table's x runs along the page's y and its y along the
        # page's x, so a glyph height is height / aspect in its x units.
        width = height / ((aspect or _A4_ASPECT) * _A4_ASPECT)
    lines = _lines(_words(chars, height, width), height)
    bounds = _column_bounds(lines, width)
    out = ["<table>"]
    for row_lines in _table_rows(lines, bounds, height):
        cells: list[list[str]] = [[] for _ in range(len(bounds) + 1)]
        for line in row_lines:
            parts: dict[int, list[str]] = {}
            if _spans_columns(_phrases(line, width), bounds):
                # One cell, in the column the line starts in.
                parts[_column(line[0].x_centre, bounds)] = [w.text for w in line]
            else:
                for word in line:
                    parts.setdefault(_column(word.x_centre, bounds), []).append(word.text)
            for column, words in parts.items():
                cells[column].append(" ".join(words))
        out.append("<tr>")
        out.extend(f"<td>{html.escape(_cell_text(cell)) if cell else ''}</td>" for cell in cells)
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
