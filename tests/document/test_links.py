"""Link annotations: where each goes, what it points at and which text it covers."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pypdfium2.raw as pdfium_raw
import pytest

from bibr.document import destinations, harvest, links, serialize, views
from bibr.document.harvest import build_document_layer
from bibr.document.model import Block
from bibr.document.rebuild import attach_blocks
from bibr.ocr.types import OcrRegionResult
from tests.document import _linked, _pdfs
from tests.document.test_layer import _BUDGET, _geometry, _inspect

# Each fixture link, by its first run: (action, uri, dest_name, name_source,
# target_page, target_xy, target_class, the component that decided the class).
EXPECTED = {
    "fig_ref": ("dest", None, "figure.1", "annot", 3, (72.0, 662.0), "float", "name"),
    "cite_ref": ("goto", None, "cite.smith2020", "table", 4, (72.0, 704.0), "bib", "name"),
    "sec_ref": ("dest", None, "section.2", "annot", 2, (72.0, 722.0), "section", "name"),
    "note_ref": ("dest", None, "Hfootnote.1", "annot", 0, (72.0, 110.0), "footnote", "name"),
    "eq_ref": ("dest", None, "equation.3", "annot", 2, (72.0, 252.0), "equation", "name"),
    "doi_ref": (
        "uri",
        "https://doi.org/10.1000/xyz123",
        None,
        None,
        None,
        None,
        "external",
        "action",
    ),
    "remote_ref": ("remote", "other.pdf", None, None, None, None, "external", "action"),
    "launch_ref": ("launch", "run.sh", None, None, None, None, "external", "action"),
    "named_ref": ("other", None, None, None, None, None, None, None),
    "broken_ref": ("dest", None, "nowhere", "annot", None, None, "unresolved", "default"),
    "bare_ref": ("none", None, None, None, None, None, None, None),
    "old_ref": ("dest", None, "old.dest", "annot", 1, (72.0, 700.0), "other", "default"),
    "array_bib_ref": ("dest", None, None, None, 4, (72.0, 704.0), "bib", "text"),
    "array_fig_ref": ("dest", None, None, None, 3, (72.0, 662.0), "float", "text"),
    "fit_ref": ("dest", None, None, None, 3, (None, None), "other", "default"),
    "table_ref": ("dest", None, "table.1", "annot", 3, (72.0, 582.0), "float", "name"),
    "second_ref": ("dest", None, "page.4", "annot", 3, (72.0, 582.0), "float", "text"),
    "late_ref": ("dest", None, "aaa.unsorted", "annot", 3, (72.0, 582.0), "float", "text"),
    "wrap_a": ("goto", None, "figure.1", "table", 3, (72.0, 662.0), "float", "name"),
}


def _layer(pdf_bytes: bytes | None = None, pages=range(_linked.N_PAGES)):
    return build_document_layer(pdf_bytes or _linked.linked_paper(), pages, budget=_BUDGET)


def _link(layer, run: str):
    """The link the fixture draws over the run named *run*."""
    (found,) = [
        link for link, spec in zip(layer.links, _linked.LINKS, strict=True) if spec.runs[0] == run
    ]
    return found


def test_the_fixture_has_the_links_the_expectations_list():
    assert [spec.runs[0] for spec in _linked.LINKS] == list(EXPECTED)


def test_every_link_is_read_with_its_action_target_and_class():
    layer = _layer()

    assert len(layer.links) == len(_linked.LINKS)
    on_page: dict[int, int] = {}
    for link, spec in zip(layer.links, _linked.LINKS, strict=True):
        action, uri, name, source, page, xy, target_class, component = EXPECTED[spec.runs[0]]
        label = spec.runs[0]
        number = on_page[spec.page] = on_page.get(spec.page, -1) + 1
        assert link.link_id == f"p{spec.page}.lk{number}", label
        assert link.page == spec.page, label
        assert (link.action, link.uri) == (action, uri), label
        assert (link.dest_name, link.name_source) == (name, source), label
        assert (link.target_page, link.target_xy) == (page, xy), label
        assert link.target_class == target_class, label
        if component is None:
            assert link.target is None, label
        else:
            assert link.target.component == f"link_target.{component}", label
            assert link.target.version == "link_target/1" and not link.target.calibrated, label
    assert layer.component_errors == {}


def test_a_link_covers_the_spans_of_the_text_under_it():
    layer = _layer()
    runs = _linked._pages()[1]

    for link, spec in zip(layer.links, _linked.LINKS, strict=True):
        covered = [
            views.span_text(layer.page(link.page), int(span_id.split(".sp")[1])).strip()
            for span_id in link.source_span_ids
        ]
        assert covered == [runs[key].string.strip() for key in spec.runs], spec.runs
        assert all(span_id.startswith(f"p{link.page}.sp") for span_id in link.source_span_ids)


def test_a_link_over_no_text_covers_no_span():
    link = links.RawLink(
        page=0,
        number=0,
        rect=(400.0, 100.0, 450.0, 120.0),
        quads=(),
        action="none",
        uri=None,
        dest_name=None,
        name_source=None,
        target_page=None,
        target_xy=None,
    )
    assert links.covered_spans(_layer().page(0), link) == ()


def test_the_rectangle_and_quadrilaterals_are_the_annotations():
    layer = _layer()
    runs = _linked._pages()[1]
    cite, wrap = _link(layer, "cite_ref"), _link(layer, "wrap_a")

    assert cite.quads == (pytest.approx(runs["cite_ref"].quad, abs=0.01),)
    assert cite.rect == pytest.approx(runs["cite_ref"].rect, abs=0.01)
    assert wrap.quads == (
        pytest.approx(runs["wrap_a"].quad, abs=0.01),
        pytest.approx(runs["wrap_b"].quad, abs=0.01),
    )
    left, bottom, right, top = wrap.rect
    assert left == pytest.approx(min(runs["wrap_a"].rect[0], runs["wrap_b"].rect[0]), abs=0.01)
    assert top == pytest.approx(max(runs["wrap_a"].rect[3], runs["wrap_b"].rect[3]), abs=0.01)
    assert layer.links[0].quads == ()


def test_a_note_annotation_is_not_a_link():
    layer = _layer()

    assert sum(link.page == 0 for link in layer.links) == 15


def test_a_remote_jump_is_never_resolved_against_this_document():
    remote = _link(_layer(), "remote_ref")

    assert (remote.action, remote.uri) == ("remote", "other.pdf")
    assert remote.target_page is None and remote.target_xy is None and remote.dest_name is None


def test_uri_links_are_the_inspections_uri_links():
    for name, pdf_bytes in _pdfs.fixture_pdfs().items():
        inspection = _inspect(pdf_bytes, layer=True)
        uris = [link.uri for link in inspection.document.links if link.action == "uri" and link.uri]

        assert uris == [link["uri"] for link in inspection.uri_links], name


def test_the_layer_says_whether_the_document_has_internal_links():
    assert _layer().presence.has_internal_links is True
    assert _layer(_pdfs.synthetic_paper(), range(7)).presence.has_internal_links is False


def test_links_are_read_for_the_pages_of_a_range_only_and_keep_their_ids():
    layer = _layer(pages=[1])

    assert [link.page for link in layer.links] == [1, 1, 1, 1]
    assert [link.link_id for link in layer.links] == ["p1.lk0", "p1.lk1", "p1.lk2", "p1.lk3"]
    # What the page range does change is the text rule: a target page it lacks has no text.
    whole = [link for link in _layer().links if link.page == 1]
    assert [replace(link, target_class=None, target=None) for link in layer.links] == [
        replace(link, target_class=None, target=None) for link in whole
    ]


def test_a_target_page_the_layer_lacks_gives_the_text_rule_nothing_to_read():
    layer = _layer(pages=[0])
    by_class = {spec.runs[0]: link for link, spec in zip(layer.links, _linked.LINKS, strict=False)}

    # The array destinations were classed by the text on pages 3 and 4.
    assert by_class["array_bib_ref"].target_class == "other"
    assert by_class["array_fig_ref"].target_class == "other"
    # Names still decide, and the destination is still resolved.
    assert by_class["fig_ref"].target_class == "float" and by_class["fig_ref"].target_page == 3


def test_without_the_annotation_functions_a_name_is_found_from_the_destination(monkeypatch):
    monkeypatch.setattr(harvest, "missing_apis", lambda: ("FPDFLink_GetAnnot",))
    layer = _layer()

    first = _link(layer, "fig_ref")
    assert (first.action, first.dest_name, first.name_source) == ("dest", "figure.1", "table")
    # A name that resolves to nothing cannot be told from a link without a destination.
    broken = _link(layer, "broken_ref")
    assert (broken.action, broken.dest_name) == ("none", None)
    assert layer.presence.missing_apis == ("FPDFLink_GetAnnot",)


def test_a_missing_core_function_leaves_the_links_unread(monkeypatch):
    monkeypatch.setattr(harvest, "missing_apis", lambda: ("FPDFLink_GetDest",))
    layer = _layer()

    assert layer.links == [] and layer.presence.has_internal_links is None
    assert layer.component_errors == {}


def test_a_link_that_cannot_be_read_is_left_out_and_the_rest_keep_their_ids(monkeypatch):
    whole = [link.link_id for link in _layer().links]
    real = links._read_link
    calls = []

    def fail_on_the_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("bad annotation")
        return real(*args, **kwargs)

    monkeypatch.setattr(links, "_read_link", fail_on_the_second)
    layer = _layer()

    assert [link.link_id for link in layer.links] == [i for i in whole if i != "p0.lk1"]
    assert layer.component_errors == {"links:0": "RuntimeError: bad annotation"}
    assert layer.presence.has_internal_links is True


def test_classing_that_fails_leaves_the_links_empty_and_says_so(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("cannot class")

    monkeypatch.setattr(links, "build_links", fail)
    layer = _layer()

    assert layer.links == [] and layer.presence.has_internal_links is None
    assert layer.component_errors == {"links": "RuntimeError: cannot class"}


def test_a_name_pdfiums_lookup_misses_is_found_in_the_table():
    late = _link(_layer(), "late_ref")
    assert late.dest_name == "aaa.unsorted" and late.target_page == 3 and late.action == "dest"

    # Without the table, the name is all the link has: it is "dest" and unresolved.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(destinations.NamedDests, "dest_of", lambda self, name: None)
        lost = _link(_layer(), "late_ref")
    assert (lost.action, lost.dest_name, lost.target_page, lost.target_xy) == (
        "dest",
        "aaa.unsorted",
        None,
        None,
    )
    assert lost.target_class == "unresolved"


def _enumerations(monkeypatch) -> list[int]:
    """One item for each call of FPDFLink_Enumerate, the call that walks a page's annotations."""
    real = pdfium_raw.FPDFLink_Enumerate
    calls: list[int] = []

    def counted(*args):
        calls.append(len(calls))
        return real(*args)

    monkeypatch.setattr(pdfium_raw, "FPDFLink_Enumerate", counted)
    return calls


def test_a_document_over_the_link_limit_is_read_to_the_limit_and_says_so_once(monkeypatch):
    monkeypatch.setattr(links, "MAX_LINKS", 5)
    layer = _layer()

    assert [link.link_id for link in layer.links] == [f"p0.lk{n}" for n in range(5)]
    # The allowance runs out on the first page, and the note is the document's: the
    # fixture has links on a second page too.
    assert {spec.page for spec in _linked.LINKS} == {0, 1}
    assert layer.component_errors == {
        "links": "more than 5 link annotations, the rest unread from page 0"
    }
    # The first of those five is internal, so the answer is yes.
    assert layer.presence.has_internal_links is True


def test_once_the_allowance_is_spent_no_later_page_is_enumerated(monkeypatch):
    monkeypatch.setattr(links, "MAX_LINKS", 5)
    calls = _enumerations(monkeypatch)
    layer = _layer()

    # Five links and the sixth, which is the stop; the other five pages are left alone.
    assert len(calls) == 6
    assert len(layer.links) == 5


def test_an_allowance_spent_exactly_at_the_end_of_a_page_is_stopped_by_the_next_one(monkeypatch):
    first = sum(1 for spec in _linked.LINKS if spec.page == 0)
    monkeypatch.setattr(links, "MAX_LINKS", first)
    calls = _enumerations(monkeypatch)
    layer = _layer()

    # The first page is read whole, with the call that finds no more; the second page has a
    # link, which is the stop, and the four pages after it are not enumerated.
    assert len(calls) == first + 1 + 1
    assert [link.page for link in layer.links] == [0] * first
    assert layer.component_errors == {
        "links": f"more than {first} link annotations, the rest unread from page 1"
    }
    assert layer.presence.has_internal_links is True


def test_the_limit_note_names_the_page_the_read_stopped_on_which_is_read_in_part(monkeypatch):
    first = sum(1 for spec in _linked.LINKS if spec.page == 0)
    monkeypatch.setattr(links, "MAX_LINKS", first + 1)
    layer = _layer()

    # Page 0 is read whole, page 1 to its first link: the pages from 1 on are the unread ones.
    assert [link.page for link in layer.links] == [0] * first + [1]
    assert layer.component_errors == {
        "links": f"more than {first + 1} link annotations, the rest unread from page 1"
    }


def test_links_cut_short_do_not_say_the_document_has_none(monkeypatch):
    monkeypatch.setattr(links, "MAX_LINKS", 0)
    layer = _layer()

    assert layer.links == []
    assert layer.presence.has_internal_links is None


def test_a_document_with_exactly_the_link_limit_is_read_whole_and_says_nothing(monkeypatch):
    whole = _layer().links
    monkeypatch.setattr(links, "MAX_LINKS", len(whole))
    layer = _layer()

    assert layer.links == whole
    assert layer.component_errors == {}


def test_a_link_with_too_many_quadrilaterals_is_kept_and_read_to_the_limit(monkeypatch):
    monkeypatch.setattr(links, "MAX_QUADS", 1)
    layer = _layer()
    wrap = _link(layer, "wrap_a")

    assert len(layer.links) == len(_linked.LINKS)
    assert len(wrap.quads) == 1 and (wrap.action, wrap.target_class) == ("goto", "float")
    note = "a link with more than 1 quadrilaterals, the rest unread"
    assert layer.component_errors == {"links": note}


def test_a_string_over_the_text_limit_is_left_unread_and_the_link_kept(monkeypatch):
    monkeypatch.setattr(destinations, "MAX_TEXT", 20)
    layer = _layer()

    doi, cite = _link(layer, "doi_ref"), _link(layer, "cite_ref")
    assert (doi.action, doi.uri, doi.target_class) == ("uri", None, "external")
    # A name over the limit is missing from the table: the link is still resolved.
    assert (cite.dest_name, cite.name_source, cite.target_page) == (None, None, 4)
    assert len(layer.links) == len(_linked.LINKS)


def test_a_string_the_getter_sizes_over_the_limit_is_never_read():
    asked = []

    def getter(*args):
        asked.append(args)
        return destinations.MAX_TEXT + 2

    assert destinations.utf16_text(getter) is None
    assert asked == [(None, 0)]


def test_the_inline_build_reads_the_same_links():
    pdf_bytes = _linked.linked_paper()

    assert _inspect(pdf_bytes, layer=True).document.links == _layer().links


def test_pages_the_inspection_does_not_read_give_the_layer_a_rebuild_gives():
    # With neither the native fill nor reference geometry the inspection opens no
    # text page and the builder reads each page itself, labels, links and tree
    # included.
    pdf_bytes = _linked.linked_paper()
    unread = _inspect(
        pdf_bytes, layer=True, fill_native_text=False, include_ref_geometry=False
    ).document

    assert unread.links and unread.struct and unread.outline
    assert [page.label for page in unread.pages] == _linked.PAGE_LABELS
    assert serialize.digest(unread) == serialize.digest(_layer())


def test_a_layer_that_never_started_leaves_what_it_could_not_read_unknown(monkeypatch):
    # The links need the open document: with no start, "no internal links" would
    # be a claim the layer cannot make, so the flags stay None, not False.
    def broken(_self, _doc):
        raise RuntimeError("start failed")

    monkeypatch.setattr(harvest.LayerBuilder, "start", broken)
    layer = _inspect(_linked.linked_paper(), layer=True).document

    assert layer.component_errors == {"start": "RuntimeError: start failed"}
    assert not layer.links and not layer.outline
    presence = layer.presence
    assert presence.has_outline is None
    assert presence.has_internal_links is None
    assert presence.has_named_dests is None
    assert presence.outline_guard_pass is None


# --- The rules ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cite.smith2020", "bib"),
        ("bib12", "bib"),
        ("ref-3", "bib"),
        ("R5", "bib"),
        ("cr7", "bib"),
        ("figure.3", "float"),
        ("Fig1", "float"),
        ("table.2", "float"),
        ("tbl3", "float"),
        ("scheme.1", "float"),
        ("section.1", "section"),
        ("sec2", "section"),
        ("appendix.A", "section"),
        ("Hfootnote.3", "footnote"),
        ("footnote.1", "footnote"),
        ("fn2", "footnote"),
        ("equation.12", "equation"),
        ("eq:energy", "equation"),
        ("eqn7", "equation"),
        # A page number, not what is on the page.
        ("page.12", None),
        ("Item.5", None),
        ("toc3", None),
        ("doc-start", None),
        ("page7", None),
        ("pg.3", None),
        ("whatever", None),
        ("", None),
        (None, None),
    ],
)
def test_class_by_name(name, expected):
    assert links.class_by_name(name) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Figure 3: Overview of the pipeline", "float"),
        ("  Fig. 2a Results", "float"),
        ("Table 1 Participants", "float"),
        ("Scheme 4", "float"),
        ("[12] Smith, J. A study", "bib"),
        ("12. Smith, J. A study", "bib"),
        ("Smith, J. and Doe, A.", "bib"),
        ("Introduction to the study (2020) and more", "bib"),
        ("A study that appeared in 2019, in Nature", "bib"),
        ("Introduction to the methods of the study", None),
        ("", None),
    ],
)
def test_class_by_text(text, expected):
    assert links.class_by_text(text) == expected


def test_the_band_reads_from_the_top_down_and_stops_at_fourteen_words():
    layer = _layer()
    words = links.PageWords(layer.page(4))
    text, starts = words.band(704.0)

    # The first reference sits just under y = 704; its words run in order.
    assert text.startswith("[1] Smith, J. (2020). A study of linked examples.")
    assert len(text.split()) <= 14
    assert len(starts) == min(14, len(text.split()))
    # A band over nothing is empty.
    assert words.band(20.0)[0] == ""


def test_a_page_without_a_text_layer_has_no_words():
    layer = _layer(_pdfs.synthetic_paper(), range(7))

    with pytest.raises(ValueError):
        links.PageWords(layer.page(5))


# --- Where a destination lands ---------------------------------------------------


def _with_blocks():
    """The fixture layer with blocks on page 3, whose destinations point at its captions."""
    layer = _layer()
    boxes = [
        ((60.0, 716.0, 540.0, 736.0), "Section-header"),
        ((60.0, 640.0, 540.0, 660.0), "Caption"),
        ((60.0, 560.0, 540.0, 580.0), "Caption"),
        ((60.0, 530.0, 400.0, 556.0), "Table"),
        ((100.0, 535.0, 200.0, 550.0), "Text"),
    ]
    layer.page(3).blocks.extend(_block(position, *spec) for position, spec in enumerate(boxes))
    return layer


def _block(position: int, box, label: str) -> Block:
    return Block(
        block_id=f"p3.r{position}",
        page=3,
        region_index=position,
        bbox_pdf=box,
        label=label,
        native_label=label,
    )


def test_a_destination_above_a_caption_lands_in_the_caption():
    layer = _with_blocks()
    figure, table = _link(layer, "fig_ref"), _link(layer, "table_ref")

    # The anchor sits a little above its target, as hyperref puts it.
    assert views.block_at(layer, figure.target_page, figure.target_xy).block_id == "p3.r1"
    assert views.block_at(layer, table.target_page, table.target_xy).block_id == "p3.r2"


def test_a_destination_inside_blocks_lands_in_the_smallest():
    layer = _with_blocks()

    assert views.block_at(layer, 3, (150.0, 540.0)).block_id == "p3.r4"
    assert views.block_at(layer, 3, (300.0, 540.0)).block_id == "p3.r3"


def test_a_destination_far_from_every_block_lands_in_none():
    layer = _with_blocks()

    assert views.block_at(layer, 3, (72.0, 400.0)) is None
    # Left of the column by more than the band's slack.
    assert views.block_at(layer, 3, (30.0, 662.0)) is None


def test_a_block_without_a_box_lands_nothing():
    layer = _with_blocks()
    page = layer.page(3)
    # A region the layout gave no box (D1 keeps it as a block with bbox_pdf None).
    page.blocks.insert(0, replace(_block(9, None, "Text"), block_id="p3.r9"))

    assert views.block_at(layer, 3, (72.0, 662.0)).block_id == "p3.r1"
    assert views.block_at(layer, 3, (72.0, 400.0)) is None


def test_a_block_whose_box_is_not_finite_lands_nothing():
    # A box with NaN in it compares false with any point, so such a block is never the one a
    # destination lands in, and nothing raises.
    layer = _with_blocks()
    nan = float("nan")
    layer.page(3).blocks.insert(0, _block(9, (nan, nan, nan, nan), "Text"))

    assert views.block_at(layer, 3, (72.0, 662.0)).block_id == "p3.r1"
    assert views.block_at(layer, 3, (72.0, 400.0)) is None


def test_a_destination_with_an_open_coordinate_is_placed_by_the_other_one_or_not_at_all():
    layer = _with_blocks()

    assert views.block_at(layer, 3, (None, 662.0)).block_id == "p3.r1"
    assert views.block_at(layer, 3, (72.0, None)) is None
    assert views.block_at(layer, 3, (None, None)) is None
    assert views.block_at(layer, 3, None) is None


def test_a_destination_on_a_page_the_layer_lacks_lands_in_none():
    layer = _with_blocks()

    assert views.block_at(layer, 9, (72.0, 662.0)) is None
    assert views.block_at(_layer(pages=[0]), 3, (72.0, 662.0)) is None


# --- Pages the harvest could not read ---------------------------------------------------


def _numbers(layer) -> list[float]:
    """Every number the layer's links and outline hold, to see that none is NaN."""
    found = []
    for link in layer.links:
        found.extend(link.rect)
        found.extend(value for quad in link.quads for value in quad)
        found.extend(value for value in link.target_xy or () if value is not None)
    for entry in layer.outline:
        found.extend(value for value in (entry.x, entry.y) if value is not None)
    return found


@pytest.mark.parametrize("failing", ["_read_objects", "_pack", "_spans_and_lines"])
def test_links_and_the_rest_survive_pages_whose_text_could_not_be_read(monkeypatch, failing):
    whole = _layer()

    def broken(*_args, **_kwargs):
        raise RuntimeError("no text")

    monkeypatch.setattr(harvest, failing, broken)
    layer = _layer()

    assert all(page.cols is None and page.error for page in layer.pages)
    # A page that failed after its size was read keeps its size.
    assert [_geometry(page) for page in layer.pages] == [_geometry(page) for page in whole.pages]
    assert all(page.crop_box is not None for page in layer.pages)
    # What the PDF declares is read from pdfium's page and not from its text: it stays.
    assert layer.struct == whole.struct and layer.outline == whole.outline
    assert [page.label for page in layer.pages] == _linked.PAGE_LABELS
    for link, read in zip(layer.links, whole.links, strict=True):
        assert (link.link_id, link.rect, link.quads, link.action, link.uri) == (
            read.link_id,
            read.rect,
            read.quads,
            read.action,
            read.uri,
        )
        assert (link.dest_name, link.target_page, link.target_xy) == (
            read.dest_name,
            read.target_page,
            read.target_xy,
        )
        # The text it covers and the text under its destination are unavailable.
        assert link.source_span_ids == ()
    # A class a name or an action gives stands; one the text under the destination gave does not.
    for link, spec in zip(layer.links, _linked.LINKS, strict=True):
        *_, target_class, component = EXPECTED[spec.runs[0]]
        assert link.target_class == ("other" if component == "text" else target_class), spec.runs[0]
    assert all(math.isfinite(value) for value in _numbers(layer))
    assert all(
        views.block_at(layer, link.target_page, link.target_xy) is None
        for link in layer.links
        if link.target_page is not None
    )
    presence = layer.presence
    assert presence.has_internal_links is True
    assert (presence.is_tagged, presence.has_outline, presence.has_named_dests) == (True,) * 3
    # The outline guard's R3 reads the text, so a pass cannot be judged.
    assert layer.outline_guard is None and presence.outline_guard_pass is None


def test_a_page_that_could_not_be_opened_leaves_internal_links_unknown():
    # Pages 2 and 3 hold no link; page 99 does not exist, so it might have held one.
    assert {spec.page for spec in _linked.LINKS}.isdisjoint({2, 3})
    layer = _layer(pages=[2, 3, 99])

    assert layer.links == [] and layer.page(99).error is not None
    assert layer.presence.has_internal_links is None
    assert _layer(pages=[2, 3]).presence.has_internal_links is False


def test_pages_that_failed_after_their_size_still_show_a_paper_has_no_internal_links(monkeypatch):
    # The links are read from pdfium's page, ahead of its text: a page whose text failed was
    # examined for them all the same.
    def broken(*_args, **_kwargs):
        raise RuntimeError("no text")

    monkeypatch.setattr(harvest, "_spans_and_lines", broken)
    layer = _layer(_pdfs.synthetic_paper(), range(7))

    assert any(page.error for page in layer.pages)
    assert layer.links == [] and layer.presence.has_internal_links is False


@pytest.mark.parametrize("unsized", [1, 3])
def test_a_page_that_could_not_be_sized_keeps_its_links_and_gives_no_text(monkeypatch, unsized):
    whole = _layer()
    real = harvest.page_header

    def fail_on_one_page(page, *, page_index, **kwargs):
        if page_index == unsized:
            raise RuntimeError("no size")
        return real(page, page_index=page_index, **kwargs)

    monkeypatch.setattr(harvest, "page_header", fail_on_one_page)
    layer = _layer()

    page = layer.page(unsized)
    assert (page.text_source, page.cols, _geometry(page)) == ("unread", None, (None,) * 4)
    assert page.error == "RuntimeError: no size"
    assert layer.component_errors == {f"harvest:{unsized}": "RuntimeError: no size"}
    assert len(layer.links) == len(whole.links)
    assert layer.presence.has_internal_links is True
    assert all(math.isfinite(value) for value in _numbers(layer))
    for link, read in zip(layer.links, whole.links, strict=True):
        assert (link.link_id, link.rect, link.target_page, link.target_xy) == (
            read.link_id,
            read.rect,
            read.target_page,
            read.target_xy,
        )
        # What the text rule or the covered spans needed from the unsized page is gone.
        if link.page == unsized:
            assert link.source_span_ids == ()
        if read.target_page == unsized and read.target.component == "link_target.text":
            assert link.target_class == "other"
        else:
            assert link.target_class == read.target_class
    assert whole.page(unsized).crop_box is not None


def test_a_page_without_a_size_has_no_words():
    page = replace(_layer().page(4), crop_box=None)

    with pytest.raises(ValueError):
        links.PageWords(page)


def test_a_target_page_without_a_size_is_not_read_for_text():
    layer = _layer()
    references = layer.page(4)
    raw = links.RawLink(
        page=0,
        number=0,
        rect=(0.0, 0.0, 1.0, 1.0),
        quads=(),
        action="dest",
        uri=None,
        dest_name=None,
        name_source=None,
        target_page=4,
        target_xy=(72.0, 704.0),
    )

    [sized] = links.build_links([raw], [layer.page(0), references])
    [unsized] = links.build_links([raw], [layer.page(0), replace(references, crop_box=None)])

    assert sized.target_class == "bib" and sized.target.evidence
    assert (unsized.target_class, unsized.target.component) == ("other", "link_target.default")


def test_blocks_on_a_page_without_a_size_have_no_box_and_nothing_lands_in_them():
    layer = _layer(pages=[3, 99])
    regions: list[list] = [[] for _page in range(100)]
    regions[99] = [
        OcrRegionResult.from_layout_region(
            {"label": "text", "bbox_2d": [0.0, 0.0, 1000.0, 500.0]}, slot_idx=0, content="text"
        )
    ]
    attach_blocks(layer, regions)

    [block] = layer.page(99).blocks
    assert layer.page(99).crop_box is None and block.bbox_pdf is None
    assert views.block_at(layer, 99, (72.0, 662.0)) is None
    assert views.block_at(layer, 99, (None, 662.0)) is None


# --- Numbers too large for a float ---------------------------------------------------------


def test_a_link_that_cannot_be_placed_is_left_out_and_nothing_infinite_is_kept():
    layer = _layer(_linked.huge_numbers_pdf(), [0])

    # Link 0 has an infinite rectangle: left out, with its number unused.
    assert [link.link_id for link in layer.links] == ["p0.lk1", "p0.lk2"]
    assert layer.component_errors == {
        "links": "a link annotation with a rectangle that is not finite, left out"
    }
    first, second = layer.links
    # The infinite quadrilateral is dropped and the finite one kept.
    assert first.quads == ((10.0, 40.0, 100.0, 40.0, 10.0, 20.0, 100.0, 20.0),)
    # An infinite coordinate is an open one: pdfium's FitR gave the left and top at infinity.
    assert first.target_xy == (None, None) and second.target_xy == (72.0, None)
    assert (first.target_page, second.target_page) == (0, 0)
    [entry] = layer.outline
    assert (entry.page, entry.x, entry.y) == (0, None, None)
    assert all(math.isfinite(value) for value in _numbers(layer))
    assert layer.presence.has_internal_links is True
    restored = serialize.from_dict(json.loads(serialize.canonical_bytes(layer)))
    assert serialize.digest(restored) == serialize.digest(layer)


def test_a_left_out_link_leaves_internal_links_unknown_when_nothing_else_shows_one():
    layer = _layer(_linked.huge_numbers_pdf(internal_rest=False), [0])

    # The link left out was the only internal one: "none" would be wrong.
    assert [link.action for link in layer.links] == ["uri", "uri"]
    assert layer.presence.has_internal_links is None


def test_a_destination_at_infinity_is_open_and_a_finite_one_is_not():
    assert destinations.finite(72.0) == 72.0 and destinations.finite(0.0) == 0.0
    for value in (float("inf"), float("-inf"), float("nan")):
        assert destinations.finite(value) is None
