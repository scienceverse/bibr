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
header instead of raising IndexError. Spans are capped at the HTML limits
(1000 columns, 65534 rows), so a garbled OCR span cannot allocate millions of
cells.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag
from pandas.errors import EmptyDataError
from pandas.io.parsers import TextParser  # type: ignore[attr-defined]  # no stub

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

# One pending rowspan: (column index, cell text, rows still covered).
_Pending = tuple[int, str, int]


def html_table_frame(source: str | Tag) -> pd.DataFrame | None:
    """Parse the first table in *source* into a DataFrame of cell strings.

    *source* is HTML markup, or a ``<table>`` element already parsed with
    html5lib (which is read from a copy, never modified). Returns ``None``
    when there is no table with text and at least one row.
    """
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
        if "display:none" in str(table.get("style") or "").replace(" ", ""):
            continue
        for element in table.find_all("style"):
            element.decompose()
        for element in table.find_all(style=_HIDDEN_STYLE_RE):
            element.decompose()
        if table.find(string=_ANY_TEXT_RE) is not None:
            candidates.append(table)
    for table in candidates:
        try:
            return _frame(*_sections(table))
        except EmptyDataError:  # no rows: ``read_html`` moves on to the next table
            continue
    return None


def _cells(row: Tag) -> list[Tag]:
    return row.find_all(("td", "th"), recursive=False)


def _sections(table: Tag) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Header, body and footer rows of *table* as text, spans expanded."""
    head_rows = table.select("thead tr")
    body_rows = table.select("tbody tr") + table.find_all("tr", recursive=False)
    foot_rows = table.select("tfoot tr")
    if not head_rows:
        # No <thead>: the leading rows made only of <th> are the header.
        while body_rows and all(cell.name == "th" for cell in _cells(body_rows[0])):
            head_rows.append(body_rows.pop(0))
    head, pending = _expand_spans(head_rows, [], flush=False)
    body, pending = _expand_spans(body_rows, pending, flush=not foot_rows)
    foot, _ = _expand_spans(foot_rows, pending, flush=True)
    return head, body, foot


def _span(value: Any, limit: int) -> int:
    match = _SPAN_RE.match(str(value or ""))
    return min(max(int(match.group(1)), 1), limit) if match else 1


def _expand_spans(
    rows: list[Tag], pending: list[_Pending], *, flush: bool
) -> tuple[list[list[str]], list[_Pending]]:
    """Rows of cell text, each spanned cell copied into every slot it covers.

    *pending* holds the rowspans still open from the rows above; they carry
    over into the next section unless *flush* adds rows for them here.
    """
    grid: list[list[str]] = []
    for tr in rows:
        texts: list[str] = []
        still_open: list[_Pending] = []
        index = 0
        for td in _cells(tr):
            # A cell spanning down from an earlier row takes its slot first.
            while pending and pending[0][0] <= index:
                prev_index, prev_text, prev_rows = pending.pop(0)
                texts.append(prev_text)
                if prev_rows > 1:
                    still_open.append((prev_index, prev_text, prev_rows - 1))
                index += 1
            text = _WHITESPACE_RE.sub(" ", td.text.strip())
            rowspan = _span(td.get("rowspan"), _MAX_ROWSPAN)
            for _ in range(_span(td.get("colspan"), _MAX_COLSPAN)):
                texts.append(text)
                if rowspan > 1:
                    still_open.append((index, text, rowspan - 1))
                index += 1
        for prev_index, prev_text, prev_rows in pending:
            texts.append(prev_text)
            if prev_rows > 1:
                still_open.append((prev_index, prev_text, prev_rows - 1))
        grid.append(texts)
        pending = still_open
    if flush:
        # Rows that exist only because a rowspan runs past the last <tr>.
        while pending:
            grid.append([text for _, text, _ in pending])
            pending = [(i, text, left - 1) for i, text, left in pending if left > 1]
    return grid, pending


def _frame(head: list[list[str]], body: list[list[str]], foot: list[list[str]]) -> pd.DataFrame:
    if len(head) > 1 and not any(any(row) for row in head):
        # Several header rows and none with text name nothing: read the table
        # as one without a header (``read_html`` raised IndexError).
        head = []
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
    # dtype=str with na_filter off and no thousands separator: no cell is
    # converted, and none becomes NaN.
    with TextParser(rows, header=header, dtype=str, na_filter=False, thousands=None) as parser:
        frame: pd.DataFrame = parser.read()
    return frame
