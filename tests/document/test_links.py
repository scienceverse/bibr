"""Link annotations: where each goes, what it points at and which text it covers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bibr.document import destinations, harvest, links, views
from bibr.document.harvest import build_document_layer
from bibr.document.model import Block
from tests.document import _linked, _pdfs
from tests.document.test_layer import _BUDGET, _inspect

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


def test_the_inline_build_reads_the_same_links():
    pdf_bytes = _linked.linked_paper()

    assert _inspect(pdf_bytes, layer=True).document.links == _layer().links


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
