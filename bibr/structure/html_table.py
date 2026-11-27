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
too (a one-row header is named here as pandas names it, in linear time where
pandas took the square of the width). Three inputs that made ``read_html`` fail or lose a cell are read
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
HTML into 2 GB of contents. A table is measured as its HTML renders
(:func:`rendered_size`, as the DOCX reader measures its tables); one past
``MAX_TABLE_BYTES`` is not read, and a :class:`TableBudget` bounds the size
and slots of all the tables of one document, which the JATS reader charges
as well.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Literal

import pandas as pd
from bs4 import BeautifulSoup, CData, NavigableString, Tag
from pandas.errors import EmptyDataError
from pandas.io.parsers import TextParser  # type: ignore[attr-defined]  # no stub

from bibr.processing_warnings import ProcessingWarning, WarningCode

HtmlParserName = Literal["html5lib", "lxml"]
_VALID_HTML_PARSERS = frozenset({"html5lib", "lxml"})


def validate_html_parser(parser: str) -> HtmlParserName:
    """Validate an HTML parser backend name (``"html5lib"`` or ``"lxml"``)."""
    if parser not in _VALID_HTML_PARSERS:
        raise ValueError(
            f"Unsupported html_parser {parser!r}; expected one of {sorted(_VALID_HTML_PARSERS)}"
        )
    return parser  # type: ignore[return-value]


def resolved_html_parser(settings: Any) -> HtmlParserName:
    """Read ``settings.pipeline.html_parser``, defaulting safely to ``"html5lib"``."""
    pipeline = getattr(settings, "pipeline", None)
    value = getattr(pipeline, "html_parser", "html5lib")
    if isinstance(value, str) and value in _VALID_HTML_PARSERS:
        return value  # type: ignore[return-value]
    return "html5lib"


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
# The size a table may render to, and the size and slots the tables of one
# document may hold together (the DOCX limits). A table is measured as its
# HTML renders: every slot a spanned cell fills writes its text again,
# escaped (pandas turns "&" into "&amp;"), with some _CELL_MARKUP_CHARS of
# markup, and one character past U+00FF or U+FFFF makes the whole string 2 or
# 4 bytes a character (PEP 393). Counting characters let 274 KB of emoji
# header cells spanning 1,000 columns take 945 MiB, and an 815 KB JATS file
# 1.2 GB of table HTML. 2,000 rows of twenty 400-character ASCII cells fit.
# JATS tables are rendered with DataFrame.to_html at some 8 us a slot.
MAX_TABLE_BYTES = 16 * 1024 * 1024
MAX_DOCUMENT_TABLE_BYTES = 64 * 1024 * 1024
MAX_DOCUMENT_TABLE_SLOTS = 4_000_000
_CELL_MARKUP_CHARS = 16

# One pending rowspan: (column index, cell text, rows still covered).
_Pending = tuple[int, str, int]


def rendered_size(text: str) -> tuple[int, int]:
    """Characters *text* takes in table HTML (``&``, ``<`` and ``>`` escaped),
    and the bytes per character it forces on the whole HTML string."""
    size = len(text) + 4 * text.count("&") + 3 * (text.count("<") + text.count(">"))
    if text.isascii():
        return size, 1
    top = ord(max(text))
    return size, 4 if top > 0xFFFF else 2 if top > 0xFF else 1


class TableSize:
    """The rendered size of a table read slot by slot, within *limit* bytes."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.chars = 0
        self.char_bytes = 1
        # A spanned cell repeats one string: it is measured once.
        self._costs: dict[str, int] = {}

    @property
    def bytes(self) -> int:
        return self.chars * self.char_bytes

    def add(self, texts: list[str]) -> bool:
        """Count a slot for each of *texts*; False once past the limit."""
        for text in texts:
            cost = self._costs.get(text)
            if cost is None:
                cost, char_bytes = rendered_size(text)
                cost = self._costs[text] = cost + _CELL_MARKUP_CHARS
                self.char_bytes = max(self.char_bytes, char_bytes)
            self.chars += cost
        return self.bytes <= self.limit


class TableBudget:
    """The rendered size and slots the tables of one document hold together.

    A parser keeps one for its document and charges every table it reads to
    it; a table over the limits is not read, and ``refused`` counts it with
    the tables out of proportion to their markup.
    """

    def __init__(self) -> None:
        self.bytes = 0
        self.slots = 0
        self.refused = 0

    def size(self) -> TableSize:
        """A measure for the next table, limited to what is left."""
        return TableSize(min(MAX_TABLE_BYTES, MAX_DOCUMENT_TABLE_BYTES - self.bytes))

    def charge(self, size: TableSize, slots: int) -> bool:
        """Count a table of *size* in *slots* slots, or return False,
        counting nothing, when it does not fit."""
        if size.bytes > size.limit or self.slots + slots > MAX_DOCUMENT_TABLE_SLOTS:
            return False
        self.bytes += size.bytes
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
                f"markup or over the table limits ({MAX_TABLE_BYTES // 2**20} MiB of table "
                f"HTML as rendered per table; {MAX_DOCUMENT_TABLE_BYTES // 2**20} MiB and "
                f"{MAX_DOCUMENT_TABLE_SLOTS:,} cells per document)",
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


_CELL_STRING_TYPES = (NavigableString, CData)


def _is_hidden_el(tag: Tag) -> bool:
    """True when *tag* is a ``<style>`` or carries ``display:none`` inline style."""
    if tag.name == "style":
        return True
    style = tag.get("style")
    return style is not None and _HIDDEN_STYLE_RE.search(str(style)) is not None


def _table_has_text(table: Tag) -> bool:
    """True when *table* has any visible non-newline string (matches ``find(string=re.compile('.+'))``)."""
    stack: list[Iterator[Any]] = [iter(table.children)]
    while stack:
        child = next(stack[-1], None)
        if child is None:
            stack.pop()
            continue
        if isinstance(child, Tag):
            if child.name == "br" or _is_hidden_el(child):
                continue
            stack.append(iter(child.children))
        elif isinstance(child, NavigableString) and _ANY_TEXT_RE.search(str(child)) is not None:
            return True
    return False


def _cell_text(td: Tag) -> str:
    """Read-only cell text extraction: ``<br>`` -> ``\\n``, skip ``<style>`` and ``display:none``."""
    parts: list[str] = []
    stack: list[Iterator[Any]] = [iter(td.children)]
    while stack:
        child = next(stack[-1], None)
        if child is None:
            stack.pop()
            continue
        if isinstance(child, Tag):
            if child.name == "br":
                parts.append("\n")
                continue
            if _is_hidden_el(child):
                continue
            stack.append(iter(child.children))
        elif type(child) in _CELL_STRING_TYPES:
            parts.append(str(child))
    if not parts:
        return ""
    raw = "".join(parts).strip()
    if not raw:
        return ""
    return _WHITESPACE_RE.sub(" ", raw)


def html_table_frame(
    source: str | Tag,
    budget: TableBudget | None = None,
    *,
    parser: HtmlParserName = "html5lib",
) -> pd.DataFrame | None:
    """Parse the first table in *source* into a DataFrame of cell strings.

    *source* is HTML markup, or a ``<table>`` element already parsed into a
    BeautifulSoup ``Tag`` (read-only: never modified or copied). When *source*
    is raw markup, *parser* selects ``"html5lib"`` (default, spec-compliant
    HTML5) or ``"lxml"`` (direct C-level ``lxml.html`` walker without
    BeautifulSoup wrapper objects). Returns ``None`` when there is no table
    with text and at least one row, or when that table is out of proportion to
    its markup even with its colspans ended, or over the text and slots
    *budget* has left (by default, a budget of its own).
    """
    if budget is None:
        budget = TableBudget()
    if isinstance(source, Tag):
        root: Tag = source
        tables = ([root] if root.name == "table" else []) + root.find_all("table")
    else:
        chosen = validate_html_parser(parser)
        if chosen == "lxml":
            return _lxml_html_table_frame(source, budget)
        root = BeautifulSoup(source, "html5lib")
        tables = root.find_all("table")
    for table in tables:
        if is_hidden_table(table) or not _table_has_text(table):
            continue
        try:
            return _frame(*_sections(table, budget))
        except EmptyDataError:  # no rows: ``read_html`` moves on to the next table
            continue
        except _TooManyCells as exc:
            budget.refused += 1
            logger.warning("HTML table %s; its contents are not read", exc)
            return None
    return None


def _lxml_is_hidden_table(table: Any) -> bool:
    return "display:none" in str(table.attrib.get("style") or "").replace(" ", "")


def _lxml_is_hidden_el(el: Any) -> bool:
    if el.tag == "style":
        return True
    style = el.attrib.get("style")
    return style is not None and _HIDDEN_STYLE_RE.search(str(style)) is not None


def _lxml_table_has_text(table: Any) -> bool:
    """True when *table* has any visible non-newline string (matches ``_table_has_text``)."""
    if table.text and _ANY_TEXT_RE.search(table.text) is not None:
        return True
    stack: list[Iterator[Any]] = [iter(table)]
    while stack:
        child = next(stack[-1], None)
        if child is None:
            stack.pop()
            continue
        if child.tail and _ANY_TEXT_RE.search(child.tail) is not None:
            return True
        tag = child.tag
        if not isinstance(tag, str):
            if child.text and _ANY_TEXT_RE.search(child.text) is not None:
                return True
            continue
        if tag == "br" or _lxml_is_hidden_el(child):
            continue
        if child.text and _ANY_TEXT_RE.search(child.text) is not None:
            return True
        if len(child):
            stack.append(iter(child))
    return False


def _lxml_cell_text(td: Any) -> str:
    """Read-only cell text extraction for an lxml element (mirrors :func:`_cell_text`)."""
    parts: list[str] = []
    if td.text:
        parts.append(td.text)
    if len(td):
        stack: list[tuple[Iterator[Any], str | None]] = [(iter(td), None)]
        while stack:
            children, tail = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                if tail:
                    parts.append(tail)
                continue
            tag = child.tag
            if not isinstance(tag, str):
                if child.tail:
                    parts.append(child.tail)
                continue
            if tag == "br":
                parts.append("\n")
                if child.tail:
                    parts.append(child.tail)
                continue
            if _lxml_is_hidden_el(child):
                if child.tail:
                    parts.append(child.tail)
                continue
            if child.text:
                parts.append(child.text)
            if len(child):
                stack.append((iter(child), child.tail))
            elif child.tail:
                parts.append(child.tail)
    if not parts:
        return ""
    raw = "".join(parts).strip()
    if not raw:
        return ""
    return _WHITESPACE_RE.sub(" ", raw)


def _lxml_row_cells(row: Any) -> list[Any]:
    return [
        child
        for child in row
        if isinstance(child.tag, str)
        and child.tag in ("td", "th")
        and not _lxml_is_hidden_el(child)
    ]


def _lxml_sections(
    table: Any, budget: TableBudget
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Header, body and footer rows of an lxml ``<table>``, spans expanded."""
    head_rows: list[list[Any]] = []
    tbody_rows: list[list[Any]] = []
    direct_tr_rows: list[list[Any]] = []
    foot_rows: list[list[Any]] = []

    def _append_row(cells: list[Any], section: str | None, is_direct: bool) -> None:
        if section == "thead":
            head_rows.append(cells)
        elif section == "tbody":
            tbody_rows.append(cells)
        elif section == "tfoot":
            foot_rows.append(cells)
        elif is_direct:
            direct_tr_rows.append(cells)

    # Frame: (children_iter, section, is_direct, pending_bare_cells).
    # Grouping bare <th>/<td> children into an implicit row matches HTML5's
    # table tree builder when OCR/malformed markup omits <tr>.
    stack: list[tuple[Iterator[Any], str | None, bool, list[Any]]] = [(iter(table), None, True, [])]
    while stack:
        children, section, is_direct, bare_cells = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            if bare_cells:
                _append_row(bare_cells, section, is_direct)
            continue
        tag = child.tag
        if not isinstance(tag, str) or _lxml_is_hidden_el(child):
            continue
        if tag in ("td", "th") and (section in ("thead", "tbody", "tfoot") or is_direct):
            bare_cells.append(child)
            continue
        if bare_cells:
            _append_row(list(bare_cells), section, is_direct)
            bare_cells.clear()
        if tag == "table":
            continue
        if tag == "tr":
            _append_row(_lxml_row_cells(child), section, is_direct)
            continue
        if tag in ("thead", "tbody", "tfoot"):
            stack.append((iter(child), tag, False, []))
        elif len(child):
            stack.append((iter(child), section, False, []))

    body_rows = tbody_rows + direct_tr_rows
    if not head_rows:
        lead = 0
        while lead < len(body_rows) and all(cell.tag == "th" for cell in body_rows[lead]):
            lead += 1
        head_rows, body_rows = body_rows[:lead], body_rows[lead:]
    grid = _expand_spans_cells(
        head_rows + body_rows + foot_rows,
        _lxml_cell_text,
        lambda td, attr: td.attrib.get(attr),
        budget,
    )
    body_end = len(head_rows) + len(body_rows)
    return grid[: len(head_rows)], grid[len(head_rows) : body_end], grid[body_end:]


def _lxml_html_table_frame(source: str, budget: TableBudget) -> pd.DataFrame | None:
    if not source or not source.strip():
        return None
    from lxml import etree
    from lxml import html as lxml_html

    try:
        doc = lxml_html.document_fromstring(
            source, parser=lxml_html.HTMLParser(recover=True, no_network=True)
        )
    except etree.ParserError:
        return None
    for table in doc.iter("table"):
        if _lxml_is_hidden_table(table) or not _lxml_table_has_text(table):
            continue
        try:
            return _frame(*_lxml_sections(table, budget))
        except EmptyDataError:
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
    return [
        child
        for child in row.children
        if isinstance(child, Tag) and child.name in ("td", "th") and not _is_hidden_el(child)
    ]


def _sections(
    table: Tag, budget: TableBudget
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Header, body and footer rows of *table* as text, spans expanded.

    The rows of a table nested in a cell are that cell's text, not rows of
    *table* (``read_html`` read them as both).
    """
    head_rows: list[Tag] = []
    tbody_rows: list[Tag] = []
    direct_tr_rows: list[Tag] = []
    foot_rows: list[Tag] = []

    # Single read-only walk of *table*'s own structure without soupsieve CSS
    # selectors or DOM mutation. Never descends into nested <table>, hidden
    # elements, or inside <tr> cells.
    stack: list[tuple[Iterator[Any], str | None, bool]] = [(iter(table.children), None, True)]
    while stack:
        children, section, is_direct = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            continue
        if not isinstance(child, Tag) or _is_hidden_el(child):
            continue
        name = child.name
        if name == "table":
            continue
        if name == "tr":
            if section == "thead":
                head_rows.append(child)
            elif section == "tbody":
                tbody_rows.append(child)
            elif section == "tfoot":
                foot_rows.append(child)
            elif is_direct:
                direct_tr_rows.append(child)
            continue
        if name in ("thead", "tbody", "tfoot"):
            stack.append((iter(child.children), name, False))
        else:
            stack.append((iter(child.children), section, False))

    body_rows = tbody_rows + direct_tr_rows
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
    """Rows of cell text, each spanned cell copied into every slot it covers."""
    return _expand_spans_cells(
        [_cells(tr) for tr in rows],
        _cell_text,
        lambda td, attr: td.get(attr),
        budget,
    )


def _expand_spans_cells(
    row_cells: list[list[Any]],
    cell_text: Callable[[Any], str],
    cell_attr: Callable[[Any, str], Any],
    budget: TableBudget,
) -> list[list[str]]:
    """Expand spans over pre-extracted rows of cell elements.

    A rowspan still open after the last row ends there, and a colspan where
    its row would grow wider than the table may be. Raises
    :class:`_TooManyCells` when a row's own cells and the rowspans it carries
    are wider than that already, or when the slots' rendered size or the
    padded slots pass what *budget* has left, so the frame is never built past
    its bounds. The table is charged to *budget* otherwise.
    """
    max_width = max_table_width(
        len(row_cells), sum(map(len, row_cells)), max(map(len, row_cells), default=0)
    )
    size = budget.size()
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
            text = cell_text(td)
            rowspan = _span(cell_attr(td, "rowspan"), _MAX_ROWSPAN)
            # A colspan past the table's width ("span every column" over a
            # narrow table) ends there, leaving a slot for each cell and
            # rowspan still to come in the row; the check above keeps that
            # room at least 1.
            room = max_width - len(texts) - (len(cells) - k - 1) - (len(pending) - placed)
            for _ in range(min(_span(cell_attr(td, "colspan"), _MAX_COLSPAN), room)):
                texts.append(text)
                if rowspan > 1:
                    still_open.append((index, text, rowspan - 1))
                index += 1
        for prev_index, prev_text, prev_rows in pending[placed:]:
            texts.append(prev_text)
            if prev_rows > 1:
                still_open.append((prev_index, prev_text, prev_rows - 1))
        # The grid shares each cell's string, but the frame and the contents
        # copy it into every slot (pandas names repeated header text anew).
        if not size.add(texts):
            raise _TooManyCells("text passes the table size limits")
        grid.append(texts)
        pending = still_open
    if not budget.charge(size, len(grid) * max(map(len, grid), default=0)):
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
    # pandas names a header of one row, also one given as [k] (head rows
    # around it blank), with the same quadratic loop.
    named = header if isinstance(header, int) else header[0] if len(header or ()) == 1 else None
    if named is not None and width > 1:
        # (One blank cell is a blank line, which pandas skips.)
        rows[named] = _header_names(rows[named])
    # dtype=str keeps every cell a string and na_filter=False keeps "NA" and
    # "" as text, so no cell is rewritten or becomes NaN. dtype=str already
    # exempts every column from thousands-separator stripping; thousands=None
    # (TextParser's default) is spelled out because ``read_html`` passes ","
    # and that stripping is what turned "1,5" into "15".
    with TextParser(rows, header=header, dtype=str, na_filter=False, thousands=None) as parser:
        frame: pd.DataFrame = parser.read()
    return frame


def _header_names(row: list[str]) -> list[str]:
    """The column names pandas gives a one-row header, named here.

    pandas looks every repeated name up in the list of names, and every
    column up in the list of blank ones, so the time grew with the square of
    the width: naming 20,000 columns of one name took 3 s, and twice as many
    four times as long. These are its names: a blank cell is
    "Unnamed: <column>", named cells are named before blank ones, and a
    repeated name takes the first ".<n>" no other column holds. Unique,
    pandas keeps them.
    """
    names = [text or f"Unnamed: {i}" for i, text in enumerate(row)]
    held = Counter(names)
    counts: defaultdict[str, int] = defaultdict(int)
    order = [i for i, text in enumerate(row) if text] + [
        i for i, text in enumerate(row) if not text
    ]
    for i in order:
        name = old = names[i]
        count = counts[name]
        while count > 0:
            counts[old] = count + 1
            name = f"{old}.{count}"
            count = count + 1 if held[name] else counts[name]
        if name != old:
            held[old] -= 1
            held[name] += 1
            names[i] = name
        counts[name] = count + 1
    return names
