"""Multi-panel figures are one figure, and separately captioned figures stay apart (#132).

Geometry is modelled on the reported papers (0..1000 page space); the text is
synthetic.
"""

from __future__ import annotations

import pytest

from bibr.structure.pdf_parser import PDFParser

_BODY = (
    "The intervention reduced the primary outcome in every subgroup that was "
    "analysed, and the effect persisted at the final follow-up visit."
)


def _region(index, label, content="", bbox=None, image_b64=None):
    value = {"index": index, "label": label, "content": content, "bbox_2d": bbox}
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def _figures(pages):
    return [
        (figure.label, figure.caption, len(figure.parts))
        for figure in PDFParser(pages).parse().figures
    ]


def test_lettered_panel_grid_with_more_boxes_than_letters_is_one_figure():
    """A preprint's figure page: eight panel letters, ten boxes (one panel is
    read as two), caption below. Each box used to be its own figure, the
    letters their captions."""
    letters = [("d", [28, 117, 47, 133]), ("b", [434, 118, 455, 137]), ("c", [28, 348, 48, 363])]
    letters += [
        ("d", [401, 343, 422, 362]),
        ("e", [719, 335, 737, 351]),
        ("g", [386, 594, 406, 613]),
    ]
    boxes = [
        [35, 112, 411, 329],
        [438, 133, 582, 275],
        [586, 144, 754, 273],
        [757, 142, 923, 274],
        [18, 346, 449, 572],
        [486, 341, 671, 570],
        [23, 597, 455, 820],
        [481, 610, 681, 801],
        [680, 344, 976, 778],
        [440, 290, 920, 330],
    ]
    page = [_region(i, "figure_title", text, box) for i, (text, box) in enumerate(letters)]
    page += [_region(10 + i, "image", bbox=box, image_b64=f"p{i}") for i, box in enumerate(boxes)]
    page.append(
        _region(30, "figure_title", "Fig. 2 — Histology guided analysis", [40, 847, 773, 873])
    )

    # Bare letters add nothing to the caption.
    assert _figures([page]) == [("2", "Fig. 2 — Histology guided analysis", 10)]


def test_panels_inside_a_whole_figure_box_and_a_caption_above_are_one_figure():
    """A figure page whose layout has one box around the whole figure and a
    box per panel, captioned above (a report's figure pages)."""
    panels = [
        [121, 156, 473, 316],
        [112, 148, 806, 893],
        [478, 155, 805, 307],
        [108, 333, 474, 485],
        [478, 339, 804, 476],
        [109, 496, 474, 654],
        [479, 520, 803, 641],
        [122, 670, 805, 863],
        [124, 673, 476, 860],
        [477, 670, 803, 854],
    ]
    first = [_region(0, "figure_title", "Figure 1. Adjusted hazard ratios", [101, 85, 897, 138])]
    first += [_region(1 + i, "chart", bbox=box, image_b64=f"p{i}") for i, box in enumerate(panels)]
    second = [
        _region(0, "figure_title", "Figure 2 Hazard ratios (log scale)", [80, 152, 894, 219]),
        _region(1, "chart", bbox=[87, 227, 787, 854], image_b64="f2"),
    ]

    assert _figures([first, second]) == [
        ("1", "Figure 1. Adjusted hazard ratios", 1),
        ("2", "Figure 2 Hazard ratios (log scale)", 1),
    ]


def test_unlettered_image_grid_over_its_caption_is_one_figure():
    boxes = [
        [100 + 280 * (i % 3), 150 + 230 * (i // 3), 360 + 280 * (i % 3), 360 + 230 * (i // 3)]
        for i in range(6)
    ]
    page = [_region(i, "image", bbox=box, image_b64=f"g{i}") for i, box in enumerate(boxes)]
    page.append(_region(6, "figure_title", "Figure 3. Example stimuli.", [100, 610, 900, 630]))

    assert _figures([page]) == [("3", "Figure 3. Example stimuli.", 6)]


def test_side_by_side_figures_with_their_own_captions_stay_two():
    page = [
        _region(0, "chart", bbox=[60, 100, 480, 400], image_b64="left"),
        _region(1, "figure_title", "Figure 1. Left panel results.", [60, 410, 480, 430]),
        _region(2, "chart", bbox=[520, 100, 940, 400], image_b64="right"),
        _region(3, "figure_title", "Figure 2. Right panel results.", [520, 410, 940, 430]),
    ]

    assert _figures([page]) == [
        ("1", "Figure 1. Left panel results.", 1),
        ("2", "Figure 2. Right panel results.", 1),
    ]


def test_stacked_figures_with_their_own_captions_stay_two():
    page = [
        _region(0, "chart", bbox=[100, 80, 900, 380], image_b64="top"),
        _region(1, "figure_title", "Figure 4. Accuracy by block.", [100, 390, 900, 410]),
        _region(2, "chart", bbox=[100, 440, 900, 740], image_b64="bottom"),
        _region(3, "figure_title", "Figure 5. Loss by block.", [100, 750, 900, 770]),
        _region(4, "text", _BODY, [100, 800, 900, 860]),
    ]

    assert _figures([page]) == [
        ("4", "Figure 4. Accuracy by block.", 1),
        ("5", "Figure 5. Loss by block.", 1),
    ]


def test_caption_between_two_panel_pairs_keeps_the_pair_above_it():
    """Two figures of two side-by-side charts each, captions below. The first
    caption sits between the pairs, scores both alike and abstained; the
    upper pair was then merged into the second figure."""
    page = [
        _region(1, "chart", bbox=[119, 65, 522, 277], image_b64="a1"),
        _region(2, "chart", bbox=[525, 70, 931, 275], image_b64="a2"),
        _region(
            3, "figure_title", "Fig. 9. Training curves on the first dataset.", [255, 290, 870, 304]
        ),
        _region(4, "chart", bbox=[120, 344, 521, 554], image_b64="b1"),
        _region(5, "chart", bbox=[524, 346, 931, 553], image_b64="b2"),
        _region(
            6,
            "figure_title",
            "Fig. 10. Training curves on the second dataset.",
            [256, 568, 834, 582],
        ),
        _region(7, "text", _BODY, [256, 771, 933, 797]),
    ]
    earlier = [
        [_region(0, "chart", bbox=[100, 100, 900, 500], image_b64=f"e{n}")] for n in range(10)
    ]

    assert _figures([*earlier, page])[-2:] == [
        ("9", "Fig. 9. Training curves on the first dataset.", 2),
        ("10", "Fig. 10. Training curves on the second dataset.", 2),
    ]


def test_logo_beside_a_numbered_figure_stays_out_of_it():
    """A small journal logo in the page head is not a panel of the figure
    printed below the body text."""
    page = [
        _region(0, "image", bbox=[865, 83, 913, 120], image_b64="logo"),
        _region(1, "text", _BODY, [80, 150, 900, 300]),
        _region(2, "image", bbox=[100, 330, 900, 700], image_b64="figure"),
        _region(3, "figure_title", "Figure 1. Study procedure.", [100, 710, 900, 730]),
    ]

    figures = _figures([[], [], page])

    assert ("1", "Figure 1. Study procedure.", 1) in figures


def _provisional_targets(monkeypatch, pages):
    """The figure targets the caption solve sees: (object id, page)."""
    from bibr.structure import parse_media

    original = parse_media.assign_captions
    seen = []

    def spy(captions, targets, **kwargs):
        seen.extend(
            (target.object_id, target.page_number)
            for target in targets
            if target.object_type == "figure"
        )
        return original(captions, targets, **kwargs)

    monkeypatch.setattr(parse_media, "assign_captions", spy)
    PDFParser(pages).parse()
    return seen


def test_a_badge_on_a_page_without_captions_does_not_shift_figure_ids(monkeypatch):
    """The number bonus compares "Figure 1" with the provisional id; a front-page
    badge that can own no caption must not take id 1."""
    front = [
        _region(0, "image", bbox=[80, 40, 230, 90], image_b64="badge"),
        _region(1, "text", _BODY, [80, 150, 900, 300]),
    ]
    page = [
        _region(0, "image", bbox=[100, 100, 900, 400], image_b64="figure"),
        _region(1, "figure_title", "Figure 1. Study procedure.", [100, 410, 900, 430]),
    ]

    targets = dict(_provisional_targets(monkeypatch, [front, page]))

    assert targets == {"figure:1": 2, "figure:2": 1}


def test_a_badge_on_a_page_with_a_caption_keeps_its_place(monkeypatch):
    front = [
        _region(0, "image", bbox=[80, 40, 230, 90], image_b64="badge"),
        _region(1, "image", bbox=[100, 300, 900, 600], image_b64="figure"),
        _region(2, "figure_title", "Figure 1. Study procedure.", [100, 610, 900, 630]),
    ]

    targets = _provisional_targets(monkeypatch, [front])

    assert targets == [("figure:1", 1), ("figure:2", 1)]


def test_a_figure_that_owns_its_panel_caption_takes_no_second_caption():
    """An uncaptioned figure earlier in the paper shifts the provisional ids, so
    the lettered pair captioned "Figure 2" holds id 3; the number bonus must not
    hand it "Figure 3" across the figure that caption is printed under."""
    stray = [
        _region(0, "text", _BODY, [100, 100, 900, 200]),
        _region(1, "image", bbox=[100, 300, 900, 600], image_b64="stray"),
    ]
    first = [
        _region(0, "image", bbox=[100, 100, 900, 400], image_b64="one"),
        _region(1, "figure_title", "Figure 1. Overview.", [100, 410, 900, 430]),
    ]
    second = [
        _region(0, "image", bbox=[128, 97, 599, 245], image_b64="a"),
        _region(1, "figure_title", "(a)", [349, 255, 371, 267]),
        _region(2, "chart", bbox=[602, 99, 878, 261], image_b64="b"),
        _region(3, "figure_title", "(b)", [752, 256, 774, 267]),
        _region(4, "figure_title", "Figure 2. Filter circuit.", [75, 278, 919, 304]),
        _region(5, "image", bbox=[127, 323, 878, 737], image_b64="hardware"),
        _region(6, "figure_title", "Figure 3. Hardware parts.", [232, 755, 770, 768]),
    ]

    figures = {
        figure.caption: [part.image_b64 for part in figure.parts]
        for figure in PDFParser([stray, first, second]).parse().figures
    }

    assert figures["Figure 2. Filter circuit."] == ["a", "b"]
    assert figures["Figure 3. Hardware parts."] == ["hardware"]


def _caption_above_page(number, first_title, second_title):
    return [
        _region(
            0, "figure_title", f"Figure {number}: Effect on outcome {number}", [190, 89, 808, 109]
        ),
        _region(1, "figure_title", first_title, [441, 120, 559, 137]),
        _region(2, "chart", bbox=[337, 148, 656, 323], image_b64=f"{number}a"),
        _region(3, "figure_title", second_title, [425, 329, 575, 345]),
        _region(4, "chart", bbox=[337, 356, 657, 532], image_b64=f"{number}b"),
    ]


@pytest.mark.parametrize(
    ("first_title", "second_title"),
    [("A", "B"), ("(a) Women", "(b) Men"), ("Panel A: Women", "Panel B: Men")],
)
def test_caption_above_lettered_panels_does_not_take_the_previous_page(first_title, second_title):
    """With captions printed above their figures, the previous page's panels
    come before "Figure n" in reading order; each page keeps its own caption."""
    pages = [_caption_above_page(n, first_title, second_title) for n in (1, 2, 3)]

    figures = {
        figure.label: sorted({part.page_number for part in figure.parts})
        for figure in PDFParser(pages).parse().figures
    }

    assert figures == {"1": [1], "2": [2], "3": [3]}


def test_a_scheme_beside_a_figure_keeps_its_own_float():
    page = [
        _region(0, "image", bbox=[60, 100, 480, 400], image_b64="scheme"),
        _region(1, "figure_title", "Scheme 1. Synthesis route.", [60, 410, 480, 430]),
        _region(2, "text", _BODY, [60, 450, 480, 600]),
        _region(3, "image", bbox=[520, 100, 940, 400], image_b64="figure"),
        _region(4, "figure_title", "Figure 1. Yield by solvent.", [520, 410, 940, 430]),
    ]

    figures = {
        figure.caption: [part.image_b64 for part in figure.parts]
        for figure in PDFParser([page]).parse().figures
    }

    assert figures["Figure 1. Yield by solvent."] == ["figure"]
    assert figures["Scheme 1. Synthesis route."] == ["scheme"]
