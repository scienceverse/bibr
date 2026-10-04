"""The structure tree: the elements each page's tree holds, and how they join to text and links."""

from __future__ import annotations

import pytest

from bibr.document import harvest, structure, views
from bibr.document.harvest import build_document_layer
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


def _expected(page: int, tree: Tag = TREE, role_map=ROLE_MAP) -> list[tuple]:
    """What page *page*'s tree should read as, from the tags the fixture was written from."""
    rows: list[tuple] = []

    def flatten(node, parent: str | None) -> None:
        tag, path, mine, kids = node
        elem_id = f"p{page}.st{len(rows)}"
        role = role_map.get(tag.role, tag.role)
        rows.append((elem_id, parent, role, mine, path, tag.alt, tag.actual, tag.lang))
        for kid in kids:
            flatten(kid, elem_id)

    root = _page_nodes(tree, (0,), page)
    if root is not None:
        flatten(root, None)
    return rows


def _rows(layer) -> list[tuple]:
    return [
        (e.elem_id, e.parent, e.role, e.mcrs, e.path, e.alt, e.actual, e.lang) for e in layer.struct
    ]


def test_each_page_reads_the_elements_its_tree_holds():
    layer = _layer()

    assert _rows(layer) == [row for page in range(_linked.N_PAGES) for row in _expected(page)]
    assert layer.component_errors == {}
    assert {e.page for e in layer.struct} == {0, 1, 2, 3, 4}
    for elem in layer.struct:
        assert elem.elem_id.startswith(f"p{elem.page}.st")
        assert all(page == elem.page for page, _mcid in elem.mcrs)


def test_a_page_without_tagged_content_has_a_tree_and_no_elements():
    layer = _layer()

    assert [e for e in layer.struct if e.page == 5] == []
    assert layer.presence.is_tagged is True


def test_a_type_is_read_after_the_role_map():
    layer = _layer()
    first = [e for e in layer.struct if e.page == 0][2]

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

    assert [e.role for e in layer.struct if e.page == 0] == [
        "Document",
        "Heading",
        "Custom",
        "Loop2",
    ]


def test_alternate_and_actual_text_and_language_are_read():
    layer = _layer()
    by_role = {(e.role, e.page): e for e in layer.struct if e.alt or e.actual or e.lang}

    assert by_role[("Figure", 3)].alt == "A diagram of the linked pipeline."
    assert by_role[("H1", 0)].lang == "en-US"
    assert by_role[("P", 1)].lang == "de-DE"
    assert by_role[("P", 2)].actual == "E = mc squared (3)"
    plain = next(e for e in layer.struct if e.role == "Document")
    assert (plain.alt, plain.actual, plain.lang) == (None, None, None)


def test_marked_content_is_listed_with_its_page_and_in_the_kid_order():
    layer = _layer()
    paragraph = next(e for e in layer.struct if e.page == 1 and e.role == "P" and len(e.mcrs) == 4)

    # The paragraph's own marked content, between its Link kids.
    assert paragraph.mcrs == ((1, 1), (1, 8), (1, 9), (1, 10))
    # A figure whose marked content draws nothing still holds it.
    figure = next(e for e in layer.struct if e.role == "Figure")
    assert figure.mcrs == ((3, 5),)


def test_a_container_holds_no_marked_content_of_its_own_and_every_tagged_page_has_the_root():
    layer = _layer()

    for page in range(5):
        elements = [e for e in layer.struct if e.page == page]
        assert elements[0].role == "Document" and elements[0].parent is None
        assert elements[0].mcrs == () and elements[0].path == (0,)
        assert all(e.parent is not None for e in elements[1:])


def test_an_element_across_a_page_break_is_read_once_per_page_and_keeps_its_path():
    layer = _layer()
    index = views.StructIndex(layer)
    first, second = index.by_mcr[(1, 4)], index.by_mcr[(2, 0)]

    assert (first.page, second.page) == (1, 2)
    assert first.role == second.role == "P"
    assert first.path == second.path == (0, 1, 4)
    assert (first.mcrs, second.mcrs) == (((1, 4),), ((2, 0),))
    assert index.copies(first) == index.copies(second) == [first, second]
    # An element on one page has itself only.
    heading = index.by_mcr[(3, 0)]
    assert index.copies(heading) == [heading]
    # Its ancestors are read again on each page, with the same paths.
    assert index.by_id[first.parent].path == index.by_id[second.parent].path == (0, 1)


def test_a_tree_can_be_read_for_a_page_range_only():
    layer = _layer(pages=[2])

    assert _rows(layer) == _expected(2)
    # The ids do not move with the range.
    assert _rows(layer) == [row for row in _rows(_layer()) if row[0].startswith("p2.")]


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

    def fail_on_page_2(api, page, page_index):
        if page_index == 2:
            raise RuntimeError("bad tree")
        return real(api, page, page_index)

    monkeypatch.setattr(structure, "read_page_tree", fail_on_page_2)
    layer = _layer()

    assert layer.component_errors == {"struct:2": "RuntimeError: bad tree"}
    assert {e.page for e in layer.struct} == {0, 1, 3, 4}
    assert layer.presence.is_tagged is True


def test_a_tree_over_the_limit_is_read_to_the_limit_and_says_so(monkeypatch):
    monkeypatch.setattr(structure, "MAX_ELEMENTS", 4)
    layer = _layer()

    assert [len([e for e in layer.struct if e.page == page]) for page in range(5)] == [4] * 5
    assert layer.component_errors["struct:0"] == "more than 4 structure elements, the rest unread"
    assert _rows(layer)[:4] == _expected(0)[:4]


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
        page = layer.page(elem.page)
        # The copy on this page alone: the spans of this page that join to the copy or below it.
        expected = [
            f"p{page.index}.sp{span}"
            for span in range(len(page.cols.span_rec))
            if (joined := index.span_element(page, span)) is not None and joined.elem_id in below
        ]
        here = [
            span_id for span_id in index.spans_of(elem) if span_id.startswith(f"p{page.index}.")
        ]
        assert here == expected, elem.elem_id


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
