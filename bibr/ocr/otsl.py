"""Decode PaddleOCR-VL OTSL table output into safe canonical HTML."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

_MARKERS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>")
_MARKER_RE = re.compile("(" + "|".join(re.escape(marker) for marker in _MARKERS) + ")")
_ANCHOR_MARKERS = {"<fcel>", "<ecel>"}
_GRID_MARKERS = frozenset(_MARKERS) - {"<nl>"}


@dataclass(frozen=True)
class OtslDecodeResult:
    """Decoded table HTML and any recoverable structural warning."""

    html: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Cell:
    marker: str
    content: str


@dataclass(frozen=True)
class OtslCompleteness:
    """Whether raw Paddle OTSL is a closed rectangular grid."""

    complete: bool
    reasons: tuple[str, ...] = ()


def check_otsl_completeness(raw: str) -> OtslCompleteness:
    """Validate raw OTSL before tolerant decoding pads ragged rows."""
    stripped = raw.strip()
    if not stripped:
        return OtslCompleteness(False, ("empty",))

    reasons: list[str] = []
    rows = _tokenize(stripped)
    widths = [sum(cell.marker in _GRID_MARKERS for cell in row) for row in rows]
    if not widths or not any(widths):
        reasons.append("no_grid_cells")
    elif any(width <= 0 for width in widths) or len(set(widths)) != 1:
        reasons.append("ragged_rows")
    if not stripped.endswith("<nl>"):
        reasons.append("missing_terminal_nl")

    try:
        validation_rows = [list(row) for row in rows]
        _pad_rows(validation_rows)
        _render_spanned(validation_rows)
    except ValueError:
        reasons.append("malformed_structure")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return OtslCompleteness(not unique_reasons, unique_reasons)


def decode_otsl(raw: str) -> OtslDecodeResult:
    """Decode native Paddle OTSL, preserving text if its spans are malformed."""
    rows = _tokenize(raw)
    _pad_rows(rows)
    try:
        return OtslDecodeResult(_render_spanned(rows))
    except ValueError as exc:
        return OtslDecodeResult(
            _render_unmerged(rows),
            (f"Malformed Paddle OTSL: {exc}",),
        )


def _tokenize(raw: str) -> list[list[_Cell]]:
    """Split exactly the OTSL structural markers into rows and cell tokens."""
    parts = _MARKER_RE.split(raw)
    rows: list[list[_Cell]] = []
    row: list[_Cell] = []

    if parts[0]:
        row.append(_Cell("<text>", parts[0]))

    for index in range(1, len(parts), 2):
        marker = parts[index]
        content = parts[index + 1]
        if marker == "<nl>":
            if content:
                row.append(_Cell("<text>", content))
            rows.append(row)
            row = []
        else:
            row.append(_Cell(marker, content))

    if row:
        rows.append(row)
    return rows


def _pad_rows(rows: list[list[_Cell]]) -> None:
    """Pad ragged OTSL rows so continuation references share one grid."""
    width = max((len(row) for row in rows), default=0)
    for row in rows:
        row.extend(_Cell("<ecel>", "") for _ in range(width - len(row)))


def _render_spanned(rows: list[list[_Cell]]) -> str:
    """Resolve OTSL continuation markers and render their rectangular spans."""
    owners: list[list[tuple[int, int] | None]] = [[None for _ in row] for row in rows]
    anchors: dict[tuple[int, int], str] = {}
    owned: dict[tuple[int, int], set[tuple[int, int]]] = {}

    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            coordinate = (row_index, column_index)
            owner = _resolve_owner(cell.marker, owners, coordinate)
            if cell.marker in _ANCHOR_MARKERS:
                if cell.marker == "<ecel>" and cell.content:
                    raise ValueError("empty-cell marker carries text")
                anchors[owner] = cell.content
                owned[owner] = {coordinate}
            elif cell.content:
                raise ValueError("continuation marker carries text")

            owners[row_index][column_index] = owner
            if owner not in owned:
                raise ValueError("continuation has no anchor")
            owned[owner].add(coordinate)

    for coordinates in owned.values():
        row_numbers = [coordinate[0] for coordinate in coordinates]
        column_numbers = [coordinate[1] for coordinate in coordinates]
        top, bottom = min(row_numbers), max(row_numbers)
        left, right = min(column_numbers), max(column_numbers)
        rectangle = {
            (row_index, column_index)
            for row_index in range(top, bottom + 1)
            for column_index in range(left, right + 1)
        }
        if coordinates != rectangle:
            raise ValueError("continuation coverage is not rectangular")

    return _render_anchors(rows, owners, anchors, owned)


def _resolve_owner(
    marker: str,
    owners: list[list[tuple[int, int] | None]],
    coordinate: tuple[int, int],
) -> tuple[int, int]:
    """Find the anchor a structural cell marker belongs to."""
    row_index, column_index = coordinate
    if marker in _ANCHOR_MARKERS:
        return coordinate
    if marker == "<lcel>":
        if column_index == 0 or owners[row_index][column_index - 1] is None:
            raise ValueError("left continuation has no left anchor")
        return owners[row_index][column_index - 1]  # type: ignore[return-value]
    if marker == "<ucel>":
        if row_index == 0 or owners[row_index - 1][column_index] is None:
            raise ValueError("up continuation has no upper anchor")
        return owners[row_index - 1][column_index]  # type: ignore[return-value]
    if marker == "<xcel>":
        if row_index == 0 or column_index == 0:
            raise ValueError("two-dimensional continuation has no left or upper anchor")
        left = owners[row_index][column_index - 1]
        upper = owners[row_index - 1][column_index]
        if left is None or upper is None or left != upper:
            raise ValueError("two-dimensional continuation has disagreeing owners")
        return left
    raise ValueError("text appears outside a cell marker")


def _render_anchors(
    rows: list[list[_Cell]],
    owners: list[list[tuple[int, int] | None]],
    anchors: dict[tuple[int, int], str],
    owned: dict[tuple[int, int], set[tuple[int, int]]],
) -> str:
    rendered_rows: list[str] = []
    for row_index, row in enumerate(rows):
        cells: list[str] = []
        for column_index in range(len(row)):
            owner = owners[row_index][column_index]
            if owner != (row_index, column_index):
                continue
            coordinates = owned[owner]
            row_span = max(point[0] for point in coordinates) - row_index + 1
            column_span = max(point[1] for point in coordinates) - column_index + 1
            attributes = ""
            if row_span > 1:
                attributes += f' rowspan="{row_span}"'
            if column_span > 1:
                attributes += f' colspan="{column_span}"'
            cells.append(f"<td{attributes}>{_escape_cell(anchors[owner])}</td>")
        rendered_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table>{''.join(rendered_rows)}</table>"


def _render_unmerged(rows: list[list[_Cell]]) -> str:
    """Render every input cell separately when OTSL spans cannot be trusted."""
    rendered_rows = [
        "<tr>" + "".join(f"<td>{_escape_cell(cell.content)}</td>" for cell in row) + "</tr>"
        for row in rows
    ]
    return f"<table>{''.join(rendered_rows)}</table>"


def _escape_cell(content: str) -> str:
    """Escape source text while retaining OCR line breaks in HTML."""
    return (
        re.sub(r"\\n(?![A-Za-z])", "<br>", html.escape(content, quote=True))
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "<br>")
    )
