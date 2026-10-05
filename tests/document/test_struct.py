"""The structure tree: one element for each place in it, and how they join to text and links."""

from __future__ import annotations

import re
from dataclasses import replace

import pypdfium2
import pypdfium2.raw as pdfium_raw
import pytest

from bibr.document import harvest, ids, structure, views
from bibr.document.harvest import build_document_layer
from bibr.document.model import StructElem
from bibr.ocr.utils import pdfium_lock
from tests.document import _linked
from tests.document._linked import ROLE_MAP, TREE, Tag
from tests.document.test_layer import _BUDGET, _inspect
from tests.document.test_links import _link


def _layer(pages=range(_linked.N_PAGES), **kwargs):
    return build_document_layer(_linked.linked_paper(**kwargs), pages, budget=_BUDGET)


def _page_nodes(tag: Tag, path: tuple[int, ...], page: int):
    """*tag* and its kids that hold content on *page*, as the page's tree holds them; None if none."""
    mine = tuple(kid for kid in tag.kids if isinstance(kid, tuple) and kid[0] == page)
    kids = [
        node
        for index, kid in enumerate(tag.kids)
        if isinstance(kid, Tag) and (node := _page_nodes(kid, (*path, index), page))
    ]
    return (tag, path, mine, kids) if mine or kids else None


def _expected(pages, tree: Tag = TREE, role_map=ROLE_MAP) -> list[tuple]:
    """What the elements of *pages* should read as, from the tags the fixture was written from.

    One row for each place in the tree, in the order the first page's tree reaches it, with
    the marked content it holds on *pages*.
    """
    rows: dict[tuple[int, ...], list] = {}

    def flatten(node) -> None:
        tag, path, mine, kids = node
        parent = ids.struct_element(path[:-1]) if len(path) > 1 else None
        role = role_map.get(tag.role, tag.role)
        row = rows.setdefault(
            path, [ids.struct_element(path), parent, role, [], path, tag.alt, tag.actual, tag.lang]
        )
        row[3].extend(mine)
        for kid in kids:
            flatten(kid)

    for page in pages:
        root = _page_nodes(tree, (0,), page)
        if root is not None:
            flatten(root)
    return [(*row[:3], tuple(row[3]), *row[4:]) for row in rows.values()]


def _rows(layer) -> list[tuple]:
    return [
        (e.elem_id, e.parent, e.role, e.mcrs, e.path, e.alt, e.actual, e.lang) for e in layer.struct
    ]


def _pages_of(layer) -> set[int]:
    return {page for elem in layer.struct for page, _mcid in elem.mcrs}


def test_the_tree_reads_as_one_element_for_each_place_in_it():
    layer = _layer()

    assert _rows(layer) == _expected(range(_linked.N_PAGES))
    assert layer.component_errors == {}
    assert _pages_of(layer) == {0, 1, 2, 3, 4}
    assert len({elem.elem_id for elem in layer.struct}) == len(layer.struct)
    for elem in layer.struct:
        assert elem.elem_id == ids.struct_element(elem.path)
        assert elem.parent == (ids.struct_element(elem.path[:-1]) if len(elem.path) > 1 else None)
        pages = [page for page, _mcid in elem.mcrs]
        assert pages == sorted(pages)


def test_a_page_without_tagged_content_has_a_tree_and_no_elements():
    layer = _layer()

    assert 5 not in _pages_of(layer)
    assert layer.presence.is_tagged is True


def test_a_type_is_read_after_the_role_map():
    layer = _layer()
    first = layer.struct[2]

    assert first.role == "H1"
    assert ROLE_MAP == {"Heading": "H1"} and "Heading" not in {e.role for e in layer.struct}


def test_a_chain_in_the_role_map_stops_at_its_first_step():
    tree = Tag(
        "Document",
        (
            Tag("Kopf", ((0, 0),)),
            Tag("Custom", ((0, 1),)),
            Tag("Loop", ((0, 2),)),
        ),
    )
    chain = {"Kopf": "Heading", "Heading": "H1", "Loop": "Loop2", "Loop2": "Loop"}
    layer = _layer(tree=tree, role_map=chain)

    assert [e.role for e in layer.struct] == ["Document", "Heading", "Custom", "Loop2"]


def test_alternate_and_actual_text_and_language_are_read():
    layer = _layer()
    # An element with marked content is told apart by the page its first is on.
    by_role = {(e.role, e.mcrs[0][0]): e for e in layer.struct if e.alt or e.actual or e.lang}

    assert by_role[("Figure", 3)].alt == "A diagram of the linked pipeline."
    assert by_role[("H1", 0)].lang == "en-US"
    assert by_role[("P", 1)].lang == "de-DE"
    assert by_role[("P", 2)].actual == "E = mc squared (3)"
    plain = next(e for e in layer.struct if e.role == "Document")
    assert (plain.alt, plain.actual, plain.lang) == (None, None, None)


def test_marked_content_is_listed_with_its_page_and_in_the_kid_order():
    layer = _layer()
    paragraph = views.StructIndex(layer).by_mcr[(1, 1)]

    # The paragraph's own marked content, between its Link kids.
    assert paragraph.role == "P" and paragraph.mcrs == ((1, 1), (1, 8), (1, 9), (1, 10))
    # A figure whose marked content draws nothing still holds it.
    figure = next(e for e in layer.struct if e.role == "Figure")
    assert figure.mcrs == ((3, 5),)


def test_a_container_holds_no_marked_content_of_its_own_and_the_root_is_the_only_top_element():
    layer = _layer()
    root = layer.struct[0]

    assert root.role == "Document" and root.parent is None
    assert root.mcrs == () and root.path == (0,) and root.elem_id == "st0"
    assert all(e.parent is not None for e in layer.struct[1:])


def test_an_element_across_a_page_break_is_one_element_holding_the_content_of_both_pages():
    layer = _layer()
    index = views.StructIndex(layer)
    paragraph = index.by_mcr[(1, 4)]

    assert index.by_mcr[(2, 0)] is paragraph
    assert paragraph.role == "P" and paragraph.path == (0, 1, 4) and paragraph.elem_id == "st0.1.4"
    assert paragraph.mcrs == ((1, 4), (2, 0))
    # An element on one page holds that page's content only.
    assert {page for page, _mcid in index.by_mcr[(3, 0)].mcrs} == {3}
    # Its ancestors are read again on each page, and are one element each.
    assert [e.elem_id for e in index.ancestors(paragraph)] == ["st0.1", "st0"]


def test_a_tree_can_be_read_for_a_page_range_only():
    layer = _layer(pages=[2])

    assert _rows(layer) == _expected([2])
    # The ids do not move with the range: an element of the range is the whole document's,
    # holding what it has on the range.
    whole = {row[0]: row for row in _rows(_layer())}
    for row in _rows(layer):
        full = whole[row[0]]
        assert row[:3] == full[:3] and row[4:] == full[4:]
        assert row[3] == tuple(mcr for mcr in full[3] if mcr[0] == 2)


@pytest.mark.parametrize(
    ("tagging", "is_tagged", "elements"),
    [("full", True, True), ("marked", True, False), ("none", False, False)],
)
def test_the_layer_says_whether_the_pdf_is_tagged(tagging, is_tagged, elements):
    layer = _layer(tagging=tagging)

    assert layer.presence.is_tagged is is_tagged
    assert bool(layer.struct) is elements
    assert layer.component_errors == {}


def test_a_structure_tree_root_without_mark_info_reads_as_no_tree():
    # pdfium gives no tree to a PDF whose /MarkInfo does not say it is tagged, whatever
    # the catalog holds: such a paper is untagged here.
    layer = _layer(tagging="tree")

    assert layer.struct == [] and layer.presence.is_tagged is False
    assert layer.component_errors == {}


def test_a_missing_tree_function_leaves_the_tree_unread(monkeypatch):
    monkeypatch.setattr(harvest, "missing_apis", lambda: ("FPDF_StructTree_GetForPage",))

    # /MarkInfo still says the paper is tagged.
    full = _layer(tagging="full")
    assert full.struct == [] and full.presence.is_tagged is True
    assert full.presence.missing_apis == ("FPDF_StructTree_GetForPage",)
    assert full.component_errors == {}


def test_a_missing_catalog_function_leaves_the_flag_to_the_tree(monkeypatch):
    monkeypatch.setattr(harvest, "missing_apis", lambda: ("FPDFCatalog_IsTagged",))

    assert _layer(tagging="full").presence.is_tagged is True
    assert _layer(tagging="marked").presence.is_tagged is True
    # No tree and no flag to read: nothing says either way.
    assert _layer(tagging="none").presence.is_tagged is None


def test_a_page_whose_tree_cannot_be_read_is_left_out_and_the_rest_kept(monkeypatch):
    real = structure.read_page_tree

    def fail_on_page_2(api, page, page_index, **kwargs):
        if page_index == 2:
            raise RuntimeError("bad tree")
        return real(api, page, page_index, **kwargs)

    monkeypatch.setattr(structure, "read_page_tree", fail_on_page_2)
    layer = _layer()

    assert layer.component_errors == {"struct:2": "RuntimeError: bad tree"}
    assert _rows(layer) == _expected([0, 1, 3, 4])
    assert layer.presence.is_tagged is True


def _copy(path, role="P", mcrs=(), kids=0) -> structure.Copy:
    elem = StructElem(
        elem_id=ids.struct_element(path),
        parent=ids.struct_element(path[:-1]) if len(path) > 1 else None,
        role=role,
        mcrs=mcrs,
        path=path,
        alt=None,
        actual=None,
        lang=None,
    )
    return structure.Copy(elem, kids)


def test_the_copies_of_an_element_are_joined_in_the_order_of_the_first():
    copies = [
        _copy((0,), "Document"),
        _copy((0, 1), "Sect"),
        _copy((0, 1, 0), "P", ((0, 3),)),
        _copy((0,), "Document"),
        _copy((0, 1), "Sect"),
        _copy((0, 1, 0), "P", ((1, 0), (1, 2))),
        _copy((0, 1, 1), "P", ((1, 5),)),
    ]

    merged, differing = structure.merge(copies)

    assert [(e.elem_id, e.mcrs) for e in merged] == [
        ("st0", ()),
        ("st0.1", ()),
        ("st0.1.0", ((0, 3), (1, 0), (1, 2))),
        ("st0.1.1", ((1, 5),)),
    ]
    assert differing == 0
    assert structure.merge([]) == ([], 0)


def test_a_copy_that_differs_from_the_first_is_counted_and_its_content_kept():
    merged, differing = structure.merge(
        [_copy((0,), "P", ((0, 0),)), _copy((0,), "H1", ((1, 0),)), _copy((0,), "P", ((2, 0),))]
    )

    assert differing == 1
    assert [(e.role, e.mcrs) for e in merged] == [("P", ((0, 0), (1, 0), (2, 0)))]


def test_a_copy_with_another_number_of_kids_is_counted_though_it_reads_the_same():
    merged, differing = structure.merge(
        [
            _copy((0, 1), "P", ((0, 0),), kids=2),
            _copy((0, 1), "P", ((1, 0),), kids=3),
            _copy((0, 1), "P", ((2, 0),), kids=2),
        ]
    )

    assert differing == 1
    assert [(e.role, e.mcrs) for e in merged] == [("P", ((0, 0), (1, 0), (2, 0)))]


def _tag_at(path: tuple[int, ...]) -> Tag:
    """The fixture's tag at *path*: (0,) is the root and each later index a place in ``kids``."""
    tag = TREE
    for index in path[1:]:
        tag = tag.kids[index]
    return tag


def test_every_copy_counts_the_kids_the_fixture_gave_the_element_on_every_page():
    pdf_bytes = _linked.linked_paper()
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            api = harvest._Api()
            trees = [
                structure.read_page_tree(api, doc[index], index) for index in range(_linked.N_PAGES)
            ]
        finally:
            doc.close()

    copies = [copy for tree in trees for copy in tree.copies]
    assert len(copies) > 20
    # Each page's copy has a slot for every kid of the element: the marked content of
    # other pages and the elements without content on this page included.
    assert [(c.elem.path, c.kids) for c in copies] == [
        (c.elem.path, len(_tag_at(c.elem.path).kids)) for c in copies
    ]
    # The paragraph across a page break is counted whole on both pages, each holding its own.
    across = [c for c in copies if c.elem.path == (0, 1, 4)]
    assert [(c.elem.mcrs, c.kids) for c in across] == [(((1, 4),), 2), (((2, 0),), 2)]


@pytest.mark.parametrize("change", ["type", "number of kids"])
def test_copies_that_differ_are_noted_in_the_layer(monkeypatch, change):
    real = structure.read_page_tree

    def change_on_page_2(api, page, page_index, **kwargs):
        tree = real(api, page, page_index, **kwargs)
        if page_index == 2:
            if change == "type":
                tree.copies = [c._replace(elem=replace(c.elem, role="Other")) for c in tree.copies]
            else:
                tree.copies = [c._replace(kids=c.kids + 1) for c in tree.copies]
        return tree

    monkeypatch.setattr(structure, "read_page_tree", change_on_page_2)
    layer = _layer()

    assert set(layer.component_errors) == {"struct"}
    assert re.fullmatch(
        r"[1-9][0-9]* copies of a structure element differ from its first",
        layer.component_errors["struct"],
    )
    # The first copy gives the element's type; the content of the other is kept.
    assert views.StructIndex(layer).by_mcr[(2, 0)].role == "P"


def test_a_note_for_the_document_is_said_once_and_a_different_one_follows_it():
    builder = harvest.LayerBuilder(b"%PDF-1.4\n", None)

    builder._note("struct", "a tree")
    builder._note("struct", "a tree")
    builder._note("struct", "an element")
    builder._note("struct", "a tree")
    builder._note("links", "a tree")

    assert builder.errors == {"struct": "a tree; an element", "links": "a tree"}


def test_a_failure_to_join_the_copies_leaves_the_elements_out_and_the_paper_alone(monkeypatch):
    def broken(copies):
        raise RuntimeError("bad copies")

    monkeypatch.setattr(structure, "merge", broken)
    layer = _layer()

    assert layer.struct == []
    assert layer.component_errors == {"struct": "RuntimeError: bad copies"}
    assert layer.presence.is_tagged is True


def _tree_calls(monkeypatch) -> list[int]:
    """One item for each page pdfium is asked a structure tree for (FPDF_StructTree_GetForPage)."""
    real = pdfium_raw.FPDF_StructTree_GetForPage
    calls: list[int] = []

    def counted(page):
        calls.append(len(calls))
        return real(page)

    monkeypatch.setattr(pdfium_raw, "FPDF_StructTree_GetForPage", counted)
    return calls


def test_a_document_over_the_limit_is_read_to_the_limit_and_says_so_once(monkeypatch):
    monkeypatch.setattr(structure, "MAX_ELEMENTS", 4)
    layer = _layer()

    # The allowance runs out on the first page, and the note is the document's.
    assert _rows(layer) == _expected([0])[:4]
    assert layer.component_errors == {"struct": "more than 4 structure elements, the rest unread"}
    assert layer.presence.is_tagged is True


def test_once_the_allowance_is_spent_no_later_page_is_given_to_pdfium(monkeypatch):
    monkeypatch.setattr(structure, "MAX_ELEMENTS", 4)
    calls = _tree_calls(monkeypatch)
    layer = _layer()

    # pdfium builds a page's whole tree in one call, so the pages after the first, which
    # would keep nothing, are not asked for.
    assert len(calls) == 1
    assert _rows(layer) == _expected([0])[:4]
    # The inline build hands its pages to the same builder.
    del calls[:]
    inline = _inspect(_linked.linked_paper(), layer=True).document
    assert len(calls) == 1
    assert len(inline.struct) == 4


def test_an_allowance_spent_exactly_at_the_end_of_a_page_is_stopped_by_the_next_one(monkeypatch):
    first = len(_expected([0]))
    monkeypatch.setattr(structure, "MAX_ELEMENTS", first)
    calls = _tree_calls(monkeypatch)
    layer = _layer()

    # Whether another element is there is known only from the next page's tree: one more
    # call, which finds an element and stops the rest.
    assert len(calls) == 2
    assert _rows(layer) == _expected([0])
    assert layer.component_errors == {
        "struct": f"more than {first} structure elements, the rest unread"
    }


def test_an_allowance_spent_exactly_with_no_element_after_it_says_nothing(monkeypatch):
    tree = Tag("Document", (Tag("P", ((0, 0),)),))
    monkeypatch.setattr(structure, "MAX_ELEMENTS", 2)
    calls = _tree_calls(monkeypatch)
    layer = _layer(tree=tree)

    # The later pages have a tree and no element: they are asked, and nothing is cut.
    assert len(calls) == _linked.N_PAGES
    assert [element.role for element in layer.struct] == ["Document", "P"]
    assert layer.component_errors == {}


def test_a_page_that_fails_does_not_stop_the_pages_after_it(monkeypatch):
    real = structure.read_page_tree

    def fail_on_page_0(api, page, page_index, **kwargs):
        if page_index == 0:
            raise RuntimeError("bad tree")
        return real(api, page, page_index, **kwargs)

    monkeypatch.setattr(structure, "read_page_tree", fail_on_page_0)
    calls = _tree_calls(monkeypatch)
    layer = _layer()

    # The failure is raised before pdfium is asked, so four pages are.
    assert len(calls) == _linked.N_PAGES - 1
    assert layer.component_errors == {"struct:0": "RuntimeError: bad tree"}
    assert _rows(layer) == _expected([1, 2, 3, 4])


def test_a_document_with_exactly_the_limit_is_read_whole_and_says_nothing(monkeypatch):
    # The limit counts each page's copy of an element.
    copies = sum(len(_expected([page])) for page in range(_linked.N_PAGES))
    assert copies > len(_layer().struct)
    monkeypatch.setattr(structure, "MAX_ELEMENTS", copies)
    layer = _layer()

    assert _rows(layer) == _expected(range(_linked.N_PAGES))
    assert layer.component_errors == {}


def test_an_element_with_too_many_kids_is_read_to_the_limit_and_says_so(monkeypatch):
    monkeypatch.setattr(structure, "MAX_KIDS", 2)
    tree = Tag(
        "Document",
        (Tag("P", ((0, 0), (0, 1), (0, 2), (0, 3))), Tag("P", ((0, 4),)), Tag("P", ((0, 5),))),
    )
    layer = _layer(tree=tree)

    # The first two kids of the root are read, and the first two marked-content
    # references of the paragraph that holds four.
    assert [(e.role, e.mcrs) for e in layer.struct] == [
        ("Document", ()),
        ("P", ((0, 0), (0, 1))),
        ("P", ((0, 4),)),
    ]
    assert layer.component_errors == {"struct": "an element with more than 2 kids, the rest unread"}


def test_a_tree_with_too_many_top_level_elements_is_read_to_the_limit_and_says_so(monkeypatch):
    monkeypatch.setattr(structure, "MAX_KIDS", 0)
    layer = _layer()

    assert layer.struct == []
    note = "a tree with more than 0 top-level elements, the rest unread"
    assert layer.component_errors == {"struct": note}


def test_the_inline_build_reads_the_same_tree():
    pdf_bytes = _linked.linked_paper()
    inline = _inspect(pdf_bytes, layer=True).document

    assert inline.struct == _layer().struct
    assert inline.presence.is_tagged is True


# --- Joining the tree to text ----------------------------------------------------------


def _roles(layer, page_index: int) -> dict[str, str | None]:
    """The role of the element each span of the page sits in, by the span's text."""
    index = views.StructIndex(layer)
    page = layer.page(page_index)
    found = {}
    for span in range(len(page.cols.span_rec)):
        elem = index.span_element(page, span)
        found[views.span_text(page, span).strip()] = elem.role if elem else None
    return found


def test_a_span_joins_to_the_element_holding_its_marked_content():
    layer = _layer()

    page0 = _roles(layer, 0)
    assert page0["Abstract"] == "H1"
    assert page0["1 This footnote is read after the page."] == "P"
    page1 = _roles(layer, 1)
    assert page1["Table 1"] == page1["Smith [1]"] == page1["late"] == "Link"
    assert page1["The introduction reads"] == "P"
    assert page1["1.1 Background"] == "H2"
    page3 = _roles(layer, 3)
    assert page3["Arm"] == page3["Score"] == "TD"
    assert page3[_linked.FIGURE_CAPTION] == "Caption"


def test_text_in_an_artifact_or_in_no_marked_content_joins_to_no_element():
    layer = _layer()

    assert _roles(layer, 0)["Linked Paper Fixture 2026"] is None
    assert _roles(layer, 0)["Supplement"] is None
    assert _roles(layer, 5) == {"Appendix: Supplementary material": None}
    cols = layer.page(0).cols
    artifact = [obj for obj in range(len(cols.obj_mcid)) if cols.obj_artifact[obj]]
    assert artifact and all(cols.obj_mcid[obj] == -1 for obj in artifact)


def test_every_tagged_span_finds_its_element():
    layer = _layer()
    index = views.StructIndex(layer)

    for page in layer.pages:
        for span in range(len(page.cols.span_rec)):
            obj = int(page.cols.span_obj[span])
            tagged = page.cols.obj_mcid[obj] >= 0 and not page.cols.obj_artifact[obj]
            assert (index.span_element(page, span) is not None) is bool(tagged), (page.index, span)


def test_a_page_without_a_text_layer_joins_nothing():
    layer = _layer()
    index = views.StructIndex(layer)
    blank = layer.page(5)
    blank.cols = None

    assert index.span_element(blank, 0) is None


def test_a_layer_whose_columns_were_freed_keeps_the_tree():
    layer = _layer()
    index = views.StructIndex(layer)
    cell = index.by_mcr[(3, 3)]
    tree = list(layer.struct)
    roles = [e.role for e in index.ancestors(cell)]

    layer.free_columns()
    freed = views.StructIndex(layer)

    assert layer.struct == tree
    assert (
        freed.by_mcr.keys() == index.by_mcr.keys()
        and freed.children.keys() == index.children.keys()
    )
    assert [e.role for e in freed.ancestors(freed.by_mcr[(3, 3)])] == roles


def _join_args(layer, index):
    """What the joins are asked about, found while the layer still holds its columns."""
    page = layer.page(1)
    span = next(n for n in range(len(page.cols.span_rec)) if index.span_element(page, n))
    return page, span, _link(layer, "table_ref"), index.by_mcr[(1, 1)]


def _joins(index, page, span, link, paragraph):
    """The three joins of *index* as calls."""
    return {
        "span_element": lambda: index.span_element(page, span),
        "spans_of": lambda: index.spans_of(paragraph),
        "link_element": lambda: index.link_element(link),
    }


def test_the_joins_answer_on_a_layer_that_holds_its_columns():
    layer = _layer()
    index = views.StructIndex(layer)

    assert not layer.columns_freed
    assert all(call() for call in _joins(index, *_join_args(layer, index)).values())


@pytest.mark.parametrize("join", ["span_element", "spans_of", "link_element"])
@pytest.mark.parametrize("order", ["answered before", "built before, not used", "built after"])
def test_a_join_on_a_layer_whose_columns_were_freed_raises_whatever_the_call_order(join, order):
    layer = _layer()
    index = views.StructIndex(layer)
    args = _join_args(layer, index)
    if order == "answered before":
        # Every join answers, and spans_of keeps what it found: that must not outlive the columns.
        assert all(call() for call in _joins(index, *args).values())
    layer.free_columns()
    if order == "built after":
        index = views.StructIndex(layer)

    with pytest.raises(views.ColumnsFreedError, match="glyph columns were freed"):
        _joins(index, *args)[join]()


def test_the_error_of_a_freed_layer_is_a_runtime_error():
    assert issubclass(views.ColumnsFreedError, RuntimeError)


def _span_texts(layer, span_ids: list[str]) -> list[str]:
    return [
        views.span_text(
            layer.page(int(span_id[1:].split(".")[0])), int(span_id.rsplit(".sp", 1)[1])
        )
        for span_id in span_ids
    ]


def test_an_element_finds_its_own_text_and_that_of_the_elements_below_it():
    layer = _layer()
    index = views.StructIndex(layer)

    assert _span_texts(layer, index.spans_of(index.by_mcr[(0, 0)])) == ["Abstract"]
    paragraph = index.by_mcr[(1, 1)]
    assert "".join(_span_texts(layer, index.spans_of(paragraph))) == (
        "The introduction reads Table 1 and Smith [1] and late."
    )
    # A Link element inside the paragraph holds only its own text.
    assert _span_texts(layer, index.spans_of(index.by_mcr[(1, 5)])) == ["Table 1 "]


def test_an_element_across_a_page_break_finds_the_text_on_both_pages():
    layer = _layer()
    index = views.StructIndex(layer)

    spans = index.spans_of(index.by_mcr[(1, 4)])
    assert [span_id.split(".")[0] for span_id in spans] == ["p1", "p2"]
    assert _span_texts(layer, spans)[1] == "finishes here, then Methods begin."
    assert index.spans_of(index.by_mcr[(2, 0)]) == spans


def test_an_elements_spans_are_the_spans_that_join_to_it_or_to_an_element_below_it():
    layer = _layer()
    index = views.StructIndex(layer)

    for elem in layer.struct:
        below = {elem.elem_id}
        pending = [elem.elem_id]
        while pending:
            for child in index.children.get(pending.pop(), ()):
                below.add(child.elem_id)
                pending.append(child.elem_id)
        expected = [
            f"p{page.index}.sp{span}"
            for page in layer.pages
            if page.cols is not None
            for span in range(len(page.cols.span_rec))
            if (joined := index.span_element(page, span)) is not None and joined.elem_id in below
        ]
        assert index.spans_of(elem) == expected, elem.elem_id


def test_the_root_holds_every_tagged_span():
    layer = _layer()
    index = views.StructIndex(layer)
    root = next(e for e in layer.struct if e.parent is None)

    tagged = sum(
        index.span_element(page, span) is not None
        for page in layer.pages
        for span in range(len(page.cols.span_rec))
    )
    assert len(index.spans_of(root)) == tagged


def test_an_elements_ancestors_run_from_its_parent_up_to_the_root():
    layer = _layer()
    index = views.StructIndex(layer)
    cell = index.by_mcr[(3, 3)]

    assert [e.role for e in index.ancestors(cell)] == ["TR", "Table", "Sect", "Document"]
    assert list(index.ancestors(index.by_mcr[(3, 0)]))[-1].parent is None


# --- Joining the tree to links ----------------------------------------------------------


def test_a_link_joins_to_the_link_element_that_wraps_its_text():
    layer = _layer()
    index = views.StructIndex(layer)

    for run, mcid in (("table_ref", 5), ("second_ref", 6), ("late_ref", 7)):
        element = index.link_element(_link(layer, run))
        assert element.role == "Link" and element.mcrs == ((1, mcid),), run
    assert [e.role for e in index.ancestors(index.link_element(_link(layer, "table_ref")))] == [
        "P",
        "Sect",
        "Document",
    ]


def test_a_link_whose_text_is_not_in_a_link_element_joins_to_none():
    layer = _layer()
    index = views.StructIndex(layer)

    # Over text in a paragraph, with no Link element around it.
    for run in ("fig_ref", "doi_ref", "bare_ref"):
        assert index.link_element(_link(layer, run)) is None, run


def test_no_link_joins_in_a_paper_without_a_tree():
    layer = _layer(tagging="none")
    index = views.StructIndex(layer)

    assert all(index.link_element(link) is None for link in layer.links)
