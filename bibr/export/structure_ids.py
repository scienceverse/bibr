"""Export ids for the paper's structure.

The pipeline gives each figure caption, table caption and footnote a synthetic
section of its own (``PaperSection.synthetic_kind``), and several stages rely
on that. The export has no such sections: their sentences stay in ``text``
with no section, ``figure`` and ``table`` rows point at their caption rows, and
each footnote gets a ``footnote`` row. Every id the export publishes is a
1-based position in document order, whatever id the pipeline used internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from bibr.export.models import FootnoteExport
from bibr.paper_contents import sections_in_document_order

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bibr.paper_contents import (
        PaperContents,
        PaperFigure,
        PaperSection,
        PaperTable,
        PaperXref,
    )

_Float = TypeVar("_Float", bound="PaperFigure | PaperTable")


@dataclass(frozen=True)
class ExportIds:
    """Maps the pipeline's ids to the export's, and holds the rows in export order."""

    sections: list[PaperSection]
    figures: list[PaperFigure]
    tables: list[PaperTable]
    footnotes: list[FootnoteExport]
    _section: dict[int, int]
    _figure: dict[int, int]
    _table: dict[int, int]
    _synthetic: frozenset[int]
    _first_text_id: dict[int, int]
    _footnote_by_text_id: dict[int, int]

    def section_id(self, internal: int | None) -> int | None:
        """Exported id of a section; None for the root, a synthetic or an unknown one."""
        return None if internal is None else self._section.get(internal)

    def figure_id(self, internal: int) -> int | None:
        return self._figure.get(internal)

    def table_id(self, internal: int) -> int | None:
        return self._table.get(internal)

    def float_section_id(self, item: PaperFigure | PaperTable) -> int | None:
        """The section a figure or table is printed in."""
        if item.section_id in self._synthetic:
            return self.section_id(item._body_section_id)
        return self.section_id(item.section_id)

    def caption_text_id(self, item: PaperFigure | PaperTable) -> int | None:
        """The text row holding a figure's or table's caption."""
        if item.section_id in self._synthetic:
            return self._first_text_id.get(item.section_id)
        return None

    def target_id(self, xref: PaperXref) -> int | None:
        """The exported row an in-text reference points at, if any."""
        if xref.xref_type == "bib":
            return xref.xref_id or None
        if xref.xref_type == "figure":
            return self._figure.get(xref.xref_id)
        if xref.xref_type == "table":
            return self._table.get(xref.xref_id)
        if xref.xref_type == "foot":
            return self._footnote_by_text_id.get(xref.xref_id)
        # Equation, section and supplementary references carry the number they
        # print, which is no row's key.
        return None


def export_ids(contents: PaperContents) -> ExportIds:
    """Work out every structural id the export publishes for ``contents``."""
    synthetic = {s.section_id: s for s in contents.sections if s.synthetic_kind}
    held: dict[int, list[int]] = {}
    for sentence in contents.sentences:
        held.setdefault(sentence.section_id, []).append(sentence.text_id)
    first_text_id = {section_id: min(text_ids) for section_id, text_ids in held.items()}

    body = [s for s in contents.sections if s.section_id != 0 and s.section_id not in synthetic]
    sections = sections_in_document_order(body, first_text_id)
    figures = _float_order(contents.figures)
    tables = _float_order(contents.tables)

    footnotes: list[FootnoteExport] = []
    footnote_by_text_id: dict[int, int] = {}
    for section in synthetic.values():
        if section.synthetic_kind != "footnote" or section.section_id not in held:
            continue
        footnote_id = len(footnotes) + 1
        footnotes.append(
            FootnoteExport(
                footnote_id=footnote_id,
                label=(section.footnote_label or "").strip() or None,
                text_id=first_text_id[section.section_id],
            )
        )
        for text_id in held[section.section_id]:
            footnote_by_text_id[text_id] = footnote_id

    return ExportIds(
        sections=sections,
        figures=figures,
        tables=tables,
        footnotes=footnotes,
        _section={s.section_id: position for position, s in enumerate(sections, start=1)},
        _figure={f.figure_id: position for position, f in enumerate(figures, start=1)},
        _table={t.table_id: position for position, t in enumerate(tables, start=1)},
        _synthetic=frozenset(synthetic),
        _first_text_id=first_text_id,
        _footnote_by_text_id=footnote_by_text_id,
    )


def _float_order(items: Sequence[_Float]) -> list[_Float]:
    """Figures or tables in document order: by page, then in the order listed.

    Label-less figure and table references fall back to this order
    (``xref_utils``), so a reference resolved by position lands on the row with
    that position.
    """
    return [
        item
        for _, item in sorted(
            enumerate(items), key=lambda pair: (pair[1].page_number or 0, pair[0])
        )
    ]
