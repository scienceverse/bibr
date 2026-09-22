"""Corpus tables: bibr exports as one Parquet file per table.

The JSON export is one paper per file. For a corpus, :func:`write_tables`
turns any number of exports into one Parquet file per table, each row
prefixed with ``paper_id``, so a table covers the whole corpus and joins on
``(paper_id, <table>_id)``:

- ``paper.parquet``: one row per paper — ``paper_id``, ``schema_version``,
  the ``source`` fields, the ``metadata`` fields, and ``producer`` /
  ``producer_version`` / ``completed_at`` from ``extraction``;
- one file per record table (``author``, ``affiliation``, ``funding``,
  ``text``, ``section``, ``url``, ``bib``, ``xref``, ``figure``, ``table``,
  ``eq`` and the ``*_match`` tables), with the export's columns;
- ``extraction_*`` files for the tidy processing lists (page sizes, float
  parts, text regions, section classification, xref tiers, consolidation,
  validation issues).

Column types come from the export models, not from the data, so every file
has the same schema however many papers it holds, and a table with no rows
is still written with its columns. List and nested fields stay Arrow lists
and structs; free-form objects are JSON strings. Readers: ``pandas`` /
``polars`` / ``duckdb`` in Python, ``arrow::read_parquet()`` in R.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from bibr.export import models as m

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

PAPER_TABLE = "paper"

# name -> (path of the row list inside the export, row model)
TABLES: dict[str, tuple[tuple[str, ...], type[BaseModel]]] = {
    "author": (("author",), m.AuthorExport),
    "affiliation": (("affiliation",), m.AffiliationExport),
    "funding": (("funding",), m.FundingExport),
    "text": (("text",), m.TextExport),
    "section": (("section",), m.SectionExport),
    "url": (("url",), m.UrlExport),
    "bib": (("bib",), m.BibExport),
    "xref": (("xref",), m.XrefExport),
    "figure": (("figure",), m.FigureExport),
    "table": (("table",), m.TableExport),
    "eq": (("eq",), m.EqExport),
    "metadata_match": (("metadata_match",), m.MetadataMatchExport),
    "affiliation_match": (("affiliation_match",), m.AffiliationMatchExport),
    "funding_match": (("funding_match",), m.FundingMatchExport),
    "bib_match": (("bib_match",), m.BibMatchExport),
    "extraction_pages": (("extraction", "pages"), m.PageExport),
    "extraction_float_parts": (("extraction", "float_parts"), m.FloatPartExport),
    "extraction_text_regions": (("extraction", "text_regions"), m.TextRegionExport),
    "extraction_section_classification": (
        ("extraction", "diagnostics", "section_classification"),
        m.SectionClassificationExport,
    ),
    "extraction_xref_tier": (("extraction", "diagnostics", "xref_tier"), m.XrefTierExport),
    "extraction_consolidation": (
        ("extraction", "diagnostics", "consolidation"),
        m.ConsolidationExport,
    ),
    "extraction_validation_issues": (
        ("extraction", "validation", "issues"),
        m.ValidationIssueExport,
    ),
}

# The ``paper`` table: (column, path inside the export, annotation).
_PAPER_COLUMNS: list[tuple[str, tuple[str, ...], Any]] = [
    ("paper_id", ("paper_id",), str),
    ("schema_version", ("schema_version",), str),
    *(
        (name, ("source", name), info.annotation)
        for name, info in m.SourceExport.model_fields.items()
    ),
    *(
        (name, ("metadata", name), info.annotation)
        for name, info in m.MetadataExport.model_fields.items()
    ),
    ("producer", ("extraction", "producer", "name"), str),
    ("producer_version", ("extraction", "producer", "version"), str),
    ("completed_at", ("extraction", "completed_at"), str),
]

# Papers buffered per Parquet row group.
_PAPERS_PER_ROW_GROUP = 200

Converter = Callable[[Any], Any]


def _column(annotation: Any) -> tuple[pa.DataType, Converter]:
    """Arrow type of a model field and the converter of its JSON value."""
    import pyarrow as pa

    origin = get_origin(annotation)
    if origin is Annotated:  # a constrained type (1-based id, 4-item box)
        return _column(get_args(annotation)[0])
    if origin in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _column(args[0])
        return pa.string(), _json
    if origin is Literal:
        return pa.string(), _as_str
    if origin is list:
        (item,) = get_args(annotation) or (Any,)
        item_type, convert = _column(item)
        return pa.list_(item_type), lambda v: None if v is None else [convert(x) for x in v]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        fields = [
            (name, _column(info.annotation)) for name, info in annotation.model_fields.items()
        ]
        struct = pa.struct([pa.field(name, dtype) for name, (dtype, _) in fields])

        def convert_model(value: Any) -> Any:
            if not isinstance(value, Mapping):
                return None
            return {name: conv(value.get(name)) for name, (_, conv) in fields}

        return struct, convert_model
    scalar = {str: pa.string(), int: pa.int64(), float: pa.float64(), bool: pa.bool_()}
    if annotation in scalar:
        return scalar[annotation], _identity if annotation is not float else _as_float
    return pa.string(), _json


def _identity(value: Any) -> Any:
    return value


def _as_str(value: Any) -> Any:
    return None if value is None else str(value)


def _as_float(value: Any) -> Any:
    return None if value is None else float(value)


def _json(value: Any) -> Any:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dig(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


@dataclass
class _Table:
    name: str
    path: Path
    columns: list[tuple[str, Converter]]
    schema: pa.Schema
    rows: list[dict[str, Any]] = field(default_factory=list)
    writer: Any = None
    n_rows: int = 0

    def flush(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema)
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.n_rows += len(self.rows)
            self.rows = []

    def close(self) -> None:
        self.flush()
        self.writer.close()


def _row_table(name: str, model: type[BaseModel], out_dir: Path) -> _Table:
    import pyarrow as pa

    fields = [
        (field_name, _column(info.annotation)) for field_name, info in model.model_fields.items()
    ]
    schema = pa.schema(
        [pa.field("paper_id", pa.string(), nullable=False)]
        + [pa.field(field_name, dtype) for field_name, (dtype, _) in fields]
    )
    return _Table(
        name=name,
        path=out_dir / f"{name}.parquet",
        columns=[(field_name, conv) for field_name, (_, conv) in fields],
        schema=schema,
    )


def _paper_table(out_dir: Path) -> _Table:
    import pyarrow as pa

    columns = [(name, _column(annotation)) for name, _, annotation in _PAPER_COLUMNS]
    schema = pa.schema(
        [pa.field(name, dtype, nullable=name != "paper_id") for name, (dtype, _) in columns]
    )
    return _Table(
        name=PAPER_TABLE,
        path=out_dir / f"{PAPER_TABLE}.parquet",
        columns=[(name, conv) for name, (_, conv) in columns],
        schema=schema,
    )


@dataclass(frozen=True)
class TablesReport:
    """What :func:`write_tables` wrote."""

    out_dir: Path
    files: dict[str, Path]
    rows: dict[str, int]
    papers: int
    skipped: tuple[tuple[str, str], ...] = ()  # (source, reason)


def _payloads(sources: Iterable[Any]) -> Iterator[tuple[str, Mapping[str, Any] | None, str | None]]:
    """``(label, payload, skip_reason)`` for each source: a dict, a
    :class:`bibr.Result`, or a path to an export JSON file."""
    for source in sources:
        if isinstance(source, Mapping):
            yield "<dict>", source, None
            continue
        if getattr(source, "ok", True) is False:  # a ChewFailure slot of a batch
            yield str(getattr(source, "path", "<failure>")), None, "failed extraction"
            continue
        data = getattr(source, "data", None)
        if isinstance(data, Mapping):
            yield str(getattr(source, "paper_id", "<result>")), data, None
            continue
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            yield str(path), None, f"not readable as JSON: {exc}"
            continue
        if not isinstance(payload, Mapping) or "schema_version" not in payload:
            yield str(path), None, "not a bibr export (no root schema_version)"
            continue
        yield str(path), payload, None


def write_tables(sources: Iterable[Any], out_dir: str | Path) -> TablesReport:
    """Write the exports in *sources* as one Parquet file per table in *out_dir*.

    *sources* are export dicts, :class:`bibr.Result` objects, or paths to
    export JSON files; files that are not bibr exports, and the
    :class:`bibr.ChewFailure` slots of a batch, are skipped and listed in the
    report. Every export is validated with the lenient 12.x reader, so
    one from another major version raises :class:`pydantic.ValidationError`.
    A ``paper_id`` seen twice raises :class:`ValueError`: it is the key that
    joins the tables. The files are written to a temporary directory and
    moved into *out_dir* only when every source was read, replacing earlier
    table files; a failure leaves *out_dir* as it was.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - pyarrow is a core dependency
        raise ImportError("Parquet tables need pyarrow: pip install pyarrow") from exc
    from bibr.export.models import PaperExportReader

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bibr-tables-", dir=out))
    paper = _paper_table(staging)
    tables = [paper] + [_row_table(name, model, staging) for name, (_, model) in TABLES.items()]
    paths = {name: path for name, (path, _) in TABLES.items()}
    seen: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []
    papers = 0
    completed = False
    try:
        for label, payload, reason in _payloads(sources):
            if payload is None:
                skipped.append((label, reason or "skipped"))
                continue
            PaperExportReader.model_validate(payload)
            paper_id = str(payload["paper_id"])
            if paper_id in seen:
                raise ValueError(
                    f"paper_id {paper_id!r} appears in both {seen[paper_id]} and {label}; "
                    "paper_id joins the tables, so it must be unique across the corpus"
                )
            seen[paper_id] = label
            paper.rows.append(
                {
                    name: conv(_dig(payload, path))
                    for (name, conv), (_, path, _) in zip(
                        paper.columns, _PAPER_COLUMNS, strict=True
                    )
                }
            )
            for table in tables[1:]:
                for row in _dig(payload, paths[table.name]) or []:
                    if isinstance(row, Mapping):
                        table.rows.append(
                            {"paper_id": paper_id}
                            | {name: conv(row.get(name)) for name, conv in table.columns}
                        )
            papers += 1
            if papers % _PAPERS_PER_ROW_GROUP == 0:
                for table in tables:
                    table.flush()
        for table in tables:
            table.close()
        files = {}
        for table in tables:
            files[table.name] = out / table.path.name
            os.replace(table.path, files[table.name])
        completed = True
    finally:
        if not completed:
            for table in tables:
                if table.writer is not None:
                    try:
                        table.writer.close()
                    except Exception:  # noqa: BLE001 - the original error is the one to raise
                        logger.debug("closing %s failed", table.path, exc_info=True)
        shutil.rmtree(staging, ignore_errors=True)
    return TablesReport(
        out_dir=out,
        files=files,
        rows={table.name: table.n_rows for table in tables},
        papers=papers,
        skipped=tuple(skipped),
    )
