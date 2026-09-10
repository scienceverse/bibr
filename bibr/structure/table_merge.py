"""Conservative continuation assembly retaining source cells and physical parts."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from bibr.paper_contents import PaperTable, PaperTablePart


def _rows(html):
    soup = BeautifulSoup(html or "", "html.parser")
    tables = soup.find_all("table")
    if len(tables) != 1:
        return None
    table = tables[0]
    if table.find("tfoot"):
        return None  # Footer placement across pages needs an explicit relationship.
    rows = table.find_all("tr")
    headers = []
    occupied = set()
    width = 0
    for ri, row in enumerate(rows):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            return None
        if ri == len(headers) and (
            row.find_parent("thead") is not None or all(c.name == "th" for c in cells)
        ):
            headers.append(row)
        ci = 0
        for cell in cells:
            while (ri, ci) in occupied:
                ci += 1
            try:
                rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            except (TypeError, ValueError):
                return None
            if not (1 <= rs <= len(rows) - ri and 1 <= cs <= 1000):
                return None  # Unresolved spans crossing the boundary stay in their source part.
            positions = {(r, c) for r in range(ri, ri + rs) for c in range(ci, ci + cs)}
            if occupied & positions:
                return None
            occupied.update(positions)
            ci += cs
            width = max(width, ci)
    if not rows or any((r, c) not in occupied for r in range(len(rows)) for c in range(width)):
        return None
    # Removing a repeated header must not remove a cell owning a data-row position.
    for ri, row in enumerate(headers):
        if any(ri + int(c.get("rowspan", 1)) > len(headers) for c in row.find_all(["th", "td"])):
            return None
    return soup, table, rows, headers, width


def _signature(rows):
    return [
        [
            (
                c.name,
                " ".join(c.get_text().split()),
                int(c.get("rowspan", 1)),
                int(c.get("colspan", 1)),
            )
            for c in row.find_all(["th", "td"], recursive=False)
        ]
        for row in rows
    ]


def merge_table_contents(survivor: PaperTable, continuation: PaperTable) -> bool:
    """Mutate only after a complete, compatible cell-grid merge succeeds.

    Callers establish continuation ownership; this function checks structure.
    Different explicit headers are ambiguous, even if flattened widths match.
    """
    left, right = _rows(survivor.tbl_html), _rows(continuation.tbl_html)
    if left is None or right is None:
        return False
    soup, table, _, headers, width = left
    _, _, incoming, repeated, incoming_width = right
    if width != incoming_width or (repeated and _signature(headers) != _signature(repeated)):
        return False
    # No explicit headers: only remove an identical first row, matching bibr's
    # existing first-row-as-column-names convention for OCR <td>-only tables.
    skip = len(repeated)
    if not headers and not repeated and _signature(left[2][:1]) == _signature(incoming[:1]):
        skip = 1
    body = table.find("tbody", recursive=False)
    target = body if body is not None else table
    for row in incoming[skip:]:
        target.append(deepcopy(row))
    html = str(table)
    try:
        df = pd.read_html(
            StringIO(html), flavor="html5lib", converters=dict.fromkeys(range(width), str)
        )[0].fillna("")
    except (ValueError, IndexError):
        return False
    if not headers and len(df):
        df.columns = [str(v) for v in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)

    def parts(t):
        return t.parts or [
            PaperTablePart(t.page_number, None, t.tbl_html, t.df.copy(), list(t.provenance))
        ]

    physical_parts = [*parts(survivor), *parts(continuation)]
    survivor.df = df
    survivor.tbl_html = html
    survivor.parts = physical_parts
    survivor.provenance.extend(continuation.provenance)
    return True
