"""HTML ``<table>`` markup -> DataFrame of the printed cell text.

Every HTML table reaches ``table[].contents`` through here: OCR table regions
on the PDF path (:class:`~bibr.structure.parse_media.MediaHandlersMixin`) and
``<table>`` elements in HTML and ePub input
(:class:`~bibr.input.html_native.HtmlParser`).

``pandas.read_html`` did this job before, but it infers column types, so the
contents stopped being what the paper printed: "2.50" came back as 2.5, "007"
as 7, "1,234" as 1234, and a decimal comma was read as a thousands separator
("1,5" became 15). An int column with one empty cell turned into floats
("12.0"), "TRUE" became True, and an empty cell, "NA", "n/a" or "None" became
NaN. Here every cell stays the text it was printed as, and an empty cell is
"".

Everything else follows ``read_html(flavor="html5lib")``, so frame shapes and
column labels are unchanged: the first table with any text wins; hidden
(``display:none``) tables, cells and ``<style>`` are skipped; ``<br>`` is a
space; header rows are the ``<thead>`` rows, or else the leading rows made
only of ``<th>``, and several of them give MultiIndex columns; a table with
none has integer column labels; ``colspan``/``rowspan`` copy the cell's text
into every position it covers. The rows then go through pandas' own row
parser (the one ``read_html`` uses) with every conversion turned off, so the
header naming ("Unnamed: 1", a repeated "Mean" becoming "Mean.1") is pandas'
too. Three inputs that made ``read_html`` fail or lose a cell are read
instead: a span that is not a plain integer ("2px") counts from its leading
digits, as a browser counts it; ``colspan="0"`` counts as 1 instead of
deleting the cell; and several header rows with no text at all are read as no
header instead of raising IndexError.

Spans are capped at the HTML limits (1000 columns, 65534 rows), and a
``rowspan`` ends at the table's last row, as a browser draws it, where
``read_html`` added a row of copies for each row it reached past the end.
The expanded table must also stay in proportion to its markup: at most twice
as wide as its widest row of cells (plus ``_FREE_COLUMNS``), and at most
``_SLOTS_PER_ITEM`` slots for each of its cells and rows (plus
``_FREE_SLOTS``). A ``colspan`` ends where its row would grow wider than
that, so the "span every column" ``colspan="100"`` over five columns still
reads; a table that is wider all the same (one wide row padding many short
ones, or rowspans piling up row after row) is not read. Spans and padding
otherwise made a few bytes allocate millions of slots, from a garbled OCR
span or crafted HTML alike; every column costs pandas about what html5lib
spends parsing a cell. More than ``_MAX_HEADER_ROWS`` header rows are read
as data rather than as that many MultiIndex levels.

The text is bounded too, since every slot a cell spans holds its text again:
one long cell spanning 20 columns down 5,000 bare ``<tr>`` made 40 KB of
HTML into 2 GB of contents. A table holding more than ``MAX_TABLE_CHARS`` of
slot text is not read, and a :class:`TableBudget` bounds the text and slots
of all the tables of one document, which the JATS reader charges as well.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag
from pandas.errors import EmptyDataError
from pandas.io.parsers import TextParser  # type: ignore[attr-defined]  # no stub

from bibr.processing_warnings import ProcessingWarning, WarningCode

logger = logging.getLogger(__name__)

# ``read_html``'s cell whitespace rule: newlines and runs of whitespace
# become one space, after stripping the ends.
_WHITESPACE_RE = re.compile(r"[\r\n]+|\s{2,}")
_HIDDEN_STYLE_RE = re.compile(r"display:\s*none")
# ``read_html``'s default ``match``: a table is a candidate when any of its
# strings has a character other than a newline.
_ANY_TEXT_RE = re.compile(".+")
_SPAN_RE = re.compile(r"\s*\+?(\d+)")
_MAX_COLSPAN = 1000
_MAX_ROWSPAN = 65534
# How far spans and padding may expand a table past its markup (see the
# module docstring). Generous for any printed table: its widest row holds
# nearly every column as a cell, and spans merge cells rather than add them.
_FREE_COLUMNS = 20
_FREE_SLOTS = 1000
_SLOTS_PER_ITEM = 20
# Header rows a table may name its columns with. Each one is a MultiIndex
# level, which costs pandas about a millisecond however narrow the table:
# 8,000 one-cell <th> rows (150 KB) took 9 s.
_MAX_HEADER_ROWS = 100
# Text a table may hold, a spanned cell's text counted in every slot it fills
# (2,000 rows of twenty 400-character cells fit), and the text and slots the
# tables of one document may hold together. The slot limit is the DOCX one:
# JATS tables are rendered with DataFrame.to_html at some 8 us a slot.
MAX_TABLE_CHARS = 16 * 1024 * 1024
MAX_DOCUMENT_TABLE_CHARS = 64 * 1024 * 1024
MAX_DOCUMENT_TABLE_SLOTS = 4_000_000

# One pending rowspan: (column index, cell text, rows still covered).
_Pending = tuple[int, str, int]


class TableBudget:
    """The text and slots the tables of one document hold together.

    A parser keeps one for its document and charges every table it reads to
    it; a table over the limits is not read, and ``refused`` counts it with
    the tables out of proportion to their markup.
    """

    def __init__(self) -> None:
        self.chars = 0
        self.slots = 0
        self.refused = 0

    def max_chars(self) -> int:
        """The text the next table may hold."""
        return min(MAX_TABLE_CHARS, MAX_DOCUMENT_TABLE_CHARS - self.chars)

    def charge(self, chars: int, slots: int) -> bool:
        """Count a table of *chars* text in *slots* slots, or return False,
        counting nothing, when it does not fit."""
        if chars > self.max_chars() or self.slots + slots > MAX_DOCUMENT_TABLE_SLOTS:
            return False
        self.chars += chars
        self.slots += slots
        return True

    def warnings(self) -> list[ProcessingWarning]:
        """The warning recording the tables refused, if there were any."""
        if not self.refused:
            return []
        return [
            ProcessingWarning(
                WarningCode.TABLE_CONTENTS_OMITTED,
                f"Left out the contents of {self.refused} table(s) out of proportion to their "
                f"markup or over the table limits ({MAX_TABLE_CHARS // 2**20} Mi characters "
                f"of cell text per table; {MAX_DOCUMENT_TABLE_CHARS // 2**20} Mi characters "
                f"and {MAX_DOCUMENT_TABLE_SLOTS:,} cells per document)",
            )
        ]


def max_table_width(rows: int, cells: int, widest: int) -> int:
    """How wide a table of *rows* rows and *cells* cells, *widest* of them in
    one row, may be read: twice its widest row plus ``_FREE_COLUMNS``, and
    ``_SLOTS_PER_ITEM`` slots for each row and cell plus ``_FREE_SLOTS``."""
    return min(
        2 * widest + _FREE_COLUMNS,
        (_SLOTS_PER_ITEM * (rows + cells) + _FREE_SLOTS) // max(rows, 1),
    )


def html_table_frame(source: str | Tag, budget: TableBudget | None = None) -> pd.DataFrame | None:
    """Parse the first table in *source* into a DataFrame of cell strings.

    *source* is HTML markup, or a ``<table>`` element already parsed with
    html5lib (which is read from a copy, never modified). Returns ``None``
    when there is no table with text and at least one row, or when that table
    is out of proportion to its markup even with its colspans ended, or over
    the text and slots *budget* has left (by default, a budget of its own).
    """
    if budget is None:
        budget = TableBudget()
    if isinstance(source, Tag):
        root: Tag = copy.copy(source)
        tables = ([root] if root.name == "table" else []) + root.find_all("table")
    else:
        root = BeautifulSoup(source, "html5lib")
        tables = root.find_all("table")
    for br in root.find_all("br"):
        br.replace_with("\n")
    candidates: list[Tag] = []
    for table in tables:
        if is_hidden_table(table):
            continue
        for element in table.find_all("style"):
            element.decompose()
        for element in table.find_all(style=_HIDDEN_STYLE_RE):
            element.decompose()
        if table.find(string=_ANY_TEXT_RE) is not None:
            candidates.append(table)
    for table in candidates:
        try:
            return _frame(*_sections(table, budget))
        except EmptyDataError:  # no rows: ``read_html`` moves on to the next table
            continue
        except _TooManyCells as exc:
            budget.refused += 1
            logger.warning("HTML table %s; its contents are not read", exc)
            return None
    return None


def is_hidden_table(table: Tag) -> bool:
    """True when *table*'s own style hides it (``display:none``), which is
    how ``read_html`` decides a table is not shown."""
    return "display:none" in str(table.get("style") or "").replace(" ", "")


class _TooManyCells(Exception):
    """The table expands out of proportion to its markup, or past its budget.

    The message says which, as the end of "HTML table ...".
    """

    def __init__(self, reason: str = "spans expand far past its markup") -> None:
        super().__init__(reason)


def _cells(row: Tag) -> list[Tag]:
    return row.find_all(("td", "th"), recursive=False)


def _sections(
    table: Tag, budget: TableBudget
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Header, body and footer rows of *table* as text, spans expanded."""
    head_rows: list[Tag] = table.select("thead tr")
    body_rows = table.select("tbody tr") + table.find_all("tr", recursive=False)
    foot_rows = table.select("tfoot tr")
    if not head_rows:
        # No <thead>: the leading rows made only of <th> are the header.
        lead = 0
        while lead < len(body_rows) and all(cell.name == "th" for cell in _cells(body_rows[lead])):
            lead += 1
        head_rows, body_rows = body_rows[:lead], body_rows[lead:]
    # A rowspan carries on from one section into the next, as in read_html.
    grid = _expand_spans(head_rows + body_rows + foot_rows, budget)
    body_end = len(head_rows) + len(body_rows)
    return grid[: len(head_rows)], grid[len(head_rows) : body_end], grid[body_end:]


def _span(value: Any, limit: int) -> int:
    match = _SPAN_RE.match(str(value or ""))
    if not match:
        return 1
    # Seven significant digits are past either limit already, and int()
    # raises on more than 4,300.
    digits = match.group(1).lstrip("0")[:7]
    return min(max(int(digits or "0"), 1), limit)


def _expand_spans(rows: list[Tag], budget: TableBudget) -> list[list[str]]:
    """Rows of cell text, each spanned cell copied into every slot it covers.

    A rowspan still open after the last row ends there, and a colspan where
    its row would grow wider than the table may be. Raises
    :class:`_TooManyCells` when a row's own cells and the rowspans it carries
    are wider than that already, or when the slots' text or the padded slots
    pass what *budget* has left, so the frame is never built past its bounds.
    The table is charged to *budget* otherwise.
    """
    row_cells = [_cells(tr) for tr in rows]
    max_width = max_table_width(
        len(rows), sum(map(len, row_cells)), max(map(len, row_cells), default=0)
    )
    max_chars = budget.max_chars()
    chars = 0
    grid: list[list[str]] = []
    # The rowspans still open from the rows above, in column order.
    pending: list[_Pending] = []
    for cells in row_cells:
        if len(cells) + len(pending) > max_width:
            raise _TooManyCells
        texts: list[str] = []
        still_open: list[_Pending] = []
        index = 0
        placed = 0
        for k, td in enumerate(cells):
            # A cell spanning down from an earlier row takes its slot first.
            while placed < len(pending) and pending[placed][0] <= index:
                prev_index, prev_text, prev_rows = pending[placed]
                placed += 1
                texts.append(prev_text)
                if prev_rows > 1:
                    still_open.append((prev_index, prev_text, prev_rows - 1))
                index += 1
            text = _WHITESPACE_RE.sub(" ", td.text.strip())
            rowspan = _span(td.get("rowspan"), _MAX_ROWSPAN)
            # A colspan past the table's width ("span every column" over a
            # narrow table) ends there, leaving a slot for each cell and
            # rowspan still to come in the row; the check above keeps that
            # room at least 1.
            room = max_width - len(texts) - (len(cells) - k - 1) - (len(pending) - placed)
            for _ in range(min(_span(td.get("colspan"), _MAX_COLSPAN), room)):
                texts.append(text)
                if rowspan > 1:
                    still_open.append((index, text, rowspan - 1))
                index += 1
        for prev_index, prev_text, prev_rows in pending[placed:]:
            texts.append(prev_text)
            if prev_rows > 1:
                still_open.append((prev_index, prev_text, prev_rows - 1))
        # The grid shares each cell's string, but the frame and the contents
        # copy it into every slot.
        chars += sum(map(len, texts))
        if chars > max_chars:
            raise _TooManyCells("text passes the table text limits")
        grid.append(texts)
        pending = still_open
    if not budget.charge(chars, len(grid) * max(map(len, grid), default=0)):
        raise _TooManyCells("cells pass the document's table cell limit")
    return grid


def _frame(head: list[list[str]], body: list[list[str]], foot: list[list[str]]) -> pd.DataFrame:
    if len(head) > 1 and not any(any(row) for row in head):
        # Several header rows and none with text name nothing: read the table
        # as one without a header (``read_html`` raised IndexError).
        head = []
    elif len(head) > _MAX_HEADER_ROWS:
        # More rows than a printed header has are not one: they are read as
        # data under integer column labels.
        head, body = [], head + body
    # One header row names the columns; several give MultiIndex columns,
    # leaving out the rows with no text.
    header: int | list[int] | None = None
    if len(head) == 1:
        header = 0
    elif head:
        header = [i for i, row in enumerate(head) if any(row)]
    rows = head + body + foot
    width = max((len(row) for row in rows), default=0)
    rows = [row + [""] * (width - len(row)) for row in rows]
    # dtype=str keeps every cell a string and na_filter=False keeps "NA" and
    # "" as text, so no cell is rewritten or becomes NaN. dtype=str already
    # exempts every column from thousands-separator stripping; thousands=None
    # (TextParser's default) is spelled out because ``read_html`` passes ","
    # and that stripping is what turned "1,5" into "15".
    with TextParser(rows, header=header, dtype=str, na_filter=False, thousands=None) as parser:
        frame: pd.DataFrame = parser.read()
    return frame
