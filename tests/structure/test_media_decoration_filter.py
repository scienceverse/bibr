"""Publisher-badge suppression: geometry bounds and the unowned-only gate.

The predecessor filter deleted regions inline in ``_handle_figure`` with no
ownership information, so it was tuned to a single example — ``height <= 24``
and ``area <= 5_000`` in 0..1000 page-fraction space. The 2026-08-01 extraction
campaign measured "Check for updates" / Crossmark badges at heights 35, 36, 36
and 66: every real badge cleared both bounds, and ~31 of the campaign's 279
spurious media objects were badges. (The ~31 figure is the campaign's own and
is not reproduced by anything here; every fixture below is synthetic layout.)

Suppression runs against the completed assignment list — after
``finalized.extend(duplicate_assignments)``, so unowned means unowned — which
is what makes the wider bounds safe to assert here. That is not the end of
``_finalize_media``: ``_reconcile_media_ids`` renumbers the survivors after it,
then the ``VAL_CAPTION_OWNERSHIP`` scan runs, then the receipt is built.
Deleting before the renumber is deliberate, and safe because the renumber
rewrites only survivors and no assignment can name a deleted figure. Two earlier versions asked a *preview* ``assign_captions`` run whether
anything would claim a badge, and neither preview could be made to agree with
production — it scored against pre-reset figure ids, and its veto replay
discarded assignments production keeps. Reading ownership off the real run
removes the question: same candidates, same vetoes, same ids, same matcher.
"""

from __future__ import annotations

import pytest


def _region(index, label, content="", bbox=None, image_b64=None):
    value = {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox,
    }
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def _parse(pages, **kwargs):
    from bibr.structure.pdf_parser import PDFParser

    return PDFParser(pages, **kwargs).parse()


def _real_figure(index, page_label="figure-1"):
    """A full-column float nothing about this filter may touch."""
    return _region(index, "chart", bbox=[80, 200, 900, 700], image_b64=page_label)


# Campaign-measured badge geometry: heights 35/36/36/66, plus a wide flat box
# at the campaign's reported maximum area. Every one clears the old
# height <= 24 / area <= 5_000 bounds. Note that none of them isolates the AREA
# bound — the shortest is 33 tall, which already clears the old ``<= 24``, so at
# HEAD every entry was rejected for the height reason whatever its area.
# ``test_area_bound_alone_admits_a_wide_flat_badge`` below covers the area bound
# on its own. Areas here run 5256..5760; the two 5742 boxes are not the largest.
_MEASURED_BADGES = [
    pytest.param([76, 62, 232, 97], id="height-35"),
    pytest.param([740, 40, 900, 76], id="height-36"),
    pytest.param([64, 905, 210, 941], id="height-36-footer"),
    pytest.param([62, 118, 149, 184], id="height-66"),
    pytest.param([70, 60, 244, 93], id="height-33-max-area-5742"),  # 174 x 33 = 5742 exactly
]


@pytest.mark.parametrize("bbox", _MEASURED_BADGES)
def test_measured_front_page_badges_are_suppressed(bbox):
    contents = _parse([[_region(0, "image", bbox=bbox)], [_real_figure(0)]])

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]
    assert [figure.figure_id for figure in contents.figures] == [1]


@pytest.mark.parametrize("bbox", _MEASURED_BADGES)
def test_measured_badges_are_suppressed_when_labelled_chart(bbox):
    """``LABEL_TREATMENT`` routes chart and image alike; the old gate saw only image."""
    contents = _parse([[_region(0, "chart", bbox=bbox)], [_real_figure(0)]])

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]


@pytest.mark.parametrize("bbox", _MEASURED_BADGES)
def test_measured_badges_are_suppressed_even_with_an_extracted_crop(bbox):
    """``FIGURE_IMAGES`` decides whether a crop rides along, not whether the
    region is scholarly — the old ``image_b64 is None`` conjunct conflated the two."""
    contents = _parse([[_region(0, "image", bbox=bbox, image_b64="badge-crop")], [_real_figure(0)]])

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]


def test_area_bound_alone_admits_a_wide_flat_badge():
    """220 x 24 = 5280: height 24 satisfies the old ``<= 24`` bound exactly, so
    only the widened area bound (5280 > the old ``<= 5_000``) can decide it.
    Every entry in ``_MEASURED_BADGES`` is taller than 24 and would be rejected
    at HEAD for the height reason whatever the area bound said."""
    contents = _parse([[_region(0, "image", bbox=[70, 60, 290, 84])], [_real_figure(0)]])

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]


def test_repeated_page_footer_marks_are_suppressed_past_the_front_page():
    """``10.37651_aujlps`` repeats a footer decoration on 14 pages; page-one-only
    suppression could not reach any of them."""
    pages = [[_region(0, "image", bbox=[64, 905, 210, 941])] for _ in range(14)]
    pages[6].append(_real_figure(1))

    contents = _parse(pages)

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]


def test_outer_margin_mark_is_suppressed_past_the_front_page():
    contents = _parse(
        [
            [_real_figure(0)],
            [_region(0, "image", bbox=[930, 300, 985, 360])],
        ]
    )

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]


def test_captioned_small_front_page_figure_survives():
    """A caption is proof the region is a float, whatever its geometry."""
    contents = _parse(
        [
            [
                _region(0, "image", bbox=[62, 118, 149, 184], image_b64="tiny-real-figure"),
                _region(
                    1,
                    "figure_title",
                    "Figure 1. Study flow diagram.",
                    bbox=[60, 190, 400, 210],
                ),
            ]
        ]
    )

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["tiny-real-figure"]
    assert contents.figures[0].caption == "Figure 1. Study flow diagram."


def test_badge_sized_body_figure_on_a_later_page_survives():
    """Past the front page, badge geometry alone is not enough — only page
    furniture (footer band, outer margin) qualifies."""
    contents = _parse(
        [
            [],
            [],
            [_region(0, "image", bbox=[300, 400, 460, 466], image_b64="small-body-figure")],
        ]
    )

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["small-body-figure"]


def test_full_column_front_page_figure_survives_uncaptioned():
    contents = _parse([[_region(0, "chart", bbox=[80, 300, 900, 700], image_b64="big")]])

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["big"]


def test_suppression_keeps_figure_ids_dense():
    """Deletion happens just before ``_reconcile_media_ids``, which renumbers
    the survivors, so no exported ``figure:N`` points at a deleted decoration
    and no id is skipped."""
    contents = _parse(
        [
            [
                _region(0, "image", bbox=[76, 62, 232, 97]),
                _region(1, "chart", bbox=[80, 200, 900, 700], image_b64="one"),
            ],
            [_region(0, "chart", bbox=[80, 200, 900, 700], image_b64="two")],
        ]
    )

    assert [figure.figure_id for figure in contents.figures] == [1, 2]
    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["one", "two"]


def _receipt_row(contents, caption_id):
    receipt = contents.caption_assignment_receipt
    for candidate, assignment in zip(receipt.candidates, receipt.assignments, strict=True):
        if candidate.caption_id == caption_id:
            return candidate.text, assignment.object_id, assignment.reasons
    raise AssertionError(f"{caption_id} missing from the receipt")


# ``_finalize_media`` drops every candidate in ``_non_caption_candidate_reasons``
# from ``matching_candidates`` and then forces it to ``object_id=None``, so such
# a candidate owns nothing in the finished assignment list and cannot shield a
# badge. Round 1 spared these badges because its preview scored the candidate
# before that filter ran.
_NON_CAPTION_NEIGHBOURS = [
    pytest.param("https://doi.org/10.1234/abcd", "doi_only_evidence", id="bare-doi-line"),
    pytest.param("Author manuscript", "publisher_noise", id="author-manuscript-stamp"),
    pytest.param("p < .05", "table_note", id="table-note"),
]


@pytest.mark.parametrize(("text", "reason"), _NON_CAPTION_NEIGHBOURS)
def test_badge_next_to_a_non_caption_line_is_still_suppressed(text, reason):
    """The canonical Crossmark layout: a "Check for updates" badge printed
    beside the bare-DOI line. The neighbouring region is already ruled a
    non-caption, so nothing can ever own the badge."""
    contents = _parse(
        [
            [
                _region(0, "image", bbox=[76, 62, 232, 97]),
                _region(1, "figure_title", text, bbox=[76, 105, 400, 125]),
            ],
            [_real_figure(0)],
        ]
    )

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]
    assert _receipt_row(contents, "caption:1") == (text, None, (reason,))


def test_badge_whose_only_claim_crosses_a_heading_is_suppressed():
    """The heading barrier veto runs after assignment, so a caption on the far
    side of a heading ends at ``object_id=None`` however well it scored.

    The receipt keeps the real reason. Measured on this layout: at HEAD the
    trailing reason is ``heading_boundary``; round 2 deleted the badge before
    the matcher ever saw it, so the caption had no candidate target left and
    the receipt degraded to ``("unmatched",)``. Reading ownership off the real
    run restores the diagnostic.
    """
    contents = _parse(
        [
            [
                _region(0, "image", bbox=[76, 62, 232, 97]),
                _region(1, "paragraph_title", "Introduction", bbox=[76, 105, 400, 125]),
                _region(2, "figure_title", "Figure 1. Something.", bbox=[76, 130, 400, 150]),
            ],
            [_real_figure(0)],
        ]
    )

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]
    assert contents.figures[0].caption is None
    text, object_id, reasons = _receipt_row(contents, "caption:1")
    assert (text, object_id) == ("Figure 1. Something.", None)
    assert reasons[-1] == "heading_boundary"


def test_a_non_caption_line_does_not_shield_a_footer_mark_past_the_front_page():
    contents = _parse(
        [
            [_real_figure(0)],
            [
                _region(0, "image", bbox=[64, 905, 210, 941]),
                _region(1, "figure_title", "Author manuscript", bbox=[64, 945, 400, 965]),
            ],
        ]
    )

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]


# Panel group, a heading, then two small front-page figures and one explicit
# caption. The generated 800-layout family around this shape is where both
# preview designs lost real crops; this is the hand-written member of it.
_PANEL_FAMILY_PAGES = [
    [
        _region(0, "image", bbox=[100, 60, 500, 120], image_b64="panel-a"),
        _region(1, "image", bbox=[60, 130, 460, 190], image_b64="panel-b"),
        _region(2, "image", bbox=[100, 200, 500, 260], image_b64="panel-c"),
        _region(3, "figure_title", "(a)", bbox=[60, 270, 200, 290]),
        _region(4, "figure_title", "(b)", bbox=[60, 295, 200, 315]),
        _region(5, "figure_title", "(c)", bbox=[60, 320, 200, 340]),
        _region(6, "figure_title", "Figure 1. Panels.", bbox=[60, 345, 900, 365]),
        _region(7, "paragraph_title", "Methods", bbox=[60, 375, 900, 395]),
        _region(8, "image", bbox=[60, 405, 210, 440], image_b64="W"),
        _region(9, "image", bbox=[80, 480, 170, 505], image_b64="Z"),
        _region(10, "figure_title", "Figure 2. Real.", bbox=[80, 530, 780, 550]),
    ],
    [_region(0, "chart", bbox=[80, 200, 900, 700], image_b64="big")],
]


def test_a_small_figure_below_a_panel_group_keeps_its_own_caption():
    """Measured: HEAD gives "Figure 2. Real." to crop ``W``; round 2 gave it to
    ``Z`` and deleted ``W`` outright.

    Round 2's preview vetoed the panel-label candidate that had won the only
    edge to ``W``, so ``W`` looked unowned and was deleted before the real
    matcher ran — and the real caption then landed on ``Z``. Nothing about
    that sequence is visible to a preview; only the finished assignment shows
    that ``W`` is owned.
    """
    contents = _parse(_PANEL_FAMILY_PAGES)

    by_caption = {
        figure.caption: [part.image_b64 for part in figure.parts] for figure in contents.figures
    }
    assert by_caption["Figure 2. Real."] == ["W"]
    assert by_caption["Figure 1. Panels."] == ["panel-a", "panel-b", "panel-c"]
    assert by_caption[None] == ["big"]


def test_suppression_reads_captions_that_are_already_attached(monkeypatch):
    """Pins the ordering the redesign depends on, and the invariant it buys.

    ``_suppress_unowned_decoration_figures`` runs after the assignment loop has
    written ``PaperFigure.caption``, so "unowned" and "uncaptioned" are the same
    set at that instant — the check below is only meaningful because of the
    ordering, and both halves are asserted. Round 2 called the method before the
    id reset, when every ``caption`` was still ``None``, so ``already_captioned``
    was empty there.
    """
    from bibr.structure import parse_media

    original = parse_media.MediaHandlersMixin._suppress_unowned_decoration_figures
    calls = []

    def spy(self, *args, **kwargs):
        before = list(self.figures)
        result = original(self, *args, **kwargs)
        survivors = {id(figure) for figure in self.figures}
        calls.append(
            {
                "dropped": [figure.caption for figure in before if id(figure) not in survivors],
                "already_captioned": [
                    figure.caption for figure in before if figure.caption is not None
                ],
            }
        )
        return result

    monkeypatch.setattr(parse_media.MediaHandlersMixin, "_suppress_unowned_decoration_figures", spy)

    _parse(_PANEL_FAMILY_PAGES)

    assert len(calls) == 1
    assert calls[0]["dropped"] == [None]  # crop ``Z``, uncaptioned
    assert sorted(calls[0]["already_captioned"]) == ["Figure 1. Panels.", "Figure 2. Real."]


def test_a_badge_that_wins_a_caption_survives_captioned():
    """The honest limit of the redesign, stated as a test rather than a claim.

    A decoration-shaped region is a live matcher target, so when it is the only
    plausible owner of an explicit caption it wins, becomes owned, and is kept —
    caption and all. Suppression never sees it. The bound this method enforces
    is exactly "unowned", and nothing stronger.
    """
    contents = _parse(
        [
            [
                _region(0, "image", bbox=[76, 62, 232, 97], image_b64="badge-shaped"),
                _region(1, "figure_title", "Figure 1. Inset.", bbox=[76, 105, 400, 125]),
            ]
        ]
    )

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["badge-shaped"]
    assert contents.figures[0].caption == "Figure 1. Inset."


def test_badge_on_a_sliced_front_page_is_suppressed():
    """``_is_front_page`` follows ``--pages``; the geometry gate rides with it."""
    pages = [[] for _ in range(4)]
    pages.append([_region(0, "image", bbox=[62, 118, 149, 184])])
    pages.append([_real_figure(0)])

    contents = _parse(pages, first_page_index=4)

    assert [figure.parts[0].image_b64 for figure in contents.figures] == ["figure-1"]
