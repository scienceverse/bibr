"""A caption the layout model split into several regions is parsed as one (#134).

Geometry is modelled on the reported papers (0..1000 page space); the text is
synthetic.
"""

from __future__ import annotations

from bibr.structure.pdf_parser import PDFParser

_PARAGRAPH = (
    "The cohort was followed for a median of five years, and outcomes were "
    "recorded at every scheduled visit by staff unaware of the allocation."
)


def _region(index, label, content="", bbox=None, image_b64=None):
    value = {"index": index, "label": label, "content": content, "bbox_2d": bbox}
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def _parse(pages):
    parser = PDFParser(pages)
    contents = parser.parse()
    body = [entry.text for entry in parser.assembler.entries]
    return contents, body, [note[0] for note in parser._footnotes]


def test_two_column_caption_under_a_full_width_figure_is_one_caption():
    """A journal review's figure: the caption runs in both columns under the
    figure, read as two caption regions; the right half went to the body."""
    page = [
        _region(3, "image", bbox=[97, 102, 906, 521], image_b64="fig"),
        _region(
            4,
            "figure_title",
            "Figure 1. Transvenous embolization of the lesion. (A) Angiogram before "
            "treatment with the draining vein marked by the black",
            [94, 543, 479, 641],
        ),
        _region(
            5,
            "figure_title",
            "arrow and the catheter tip by the white arrow). The coil was deployed distally.",
            [509, 543, 893, 630],
        ),
        _region(6, "text", _PARAGRAPH, [84, 671, 346, 741]),
    ]

    contents, body, _notes = _parse([page])

    (figure,) = contents.figures
    assert figure.caption.startswith("Figure 1. Transvenous embolization")
    assert figure.caption.endswith("The coil was deployed distally.")
    assert not any("catheter tip" in text for text in body)


def test_right_column_half_read_as_body_text_joins_the_caption():
    """A case series' figure: the left half is a caption region, the right
    half a text region; it became a body paragraph."""
    page = [
        _region(8, "image", bbox=[197, 322, 800, 797], image_b64="fig"),
        _region(
            9,
            "figure_title",
            "Figure 1. (A) Head computed tomography scan showing a mass in the lateral "
            "ventricle and (B) follow-up",
            [193, 806, 478, 883],
        ),
        _region(
            10,
            "text",
            "magnetic resonance imaging after 2 months revealing enhancement in the tumor bed.",
            [509, 806, 797, 883],
        ),
    ]

    contents, body, _notes = _parse([[], [], [], [], page])

    (figure,) = contents.figures
    assert figure.caption.endswith("enhancement in the tumor bed.")
    assert not any("tumor bed" in text for text in body)


def test_bare_label_above_its_title_line_is_one_caption():
    page = [
        _region(0, "image", bbox=[100, 100, 900, 500], image_b64="fig"),
        _region(1, "figure_title", "FIGURE 2", [100, 510, 180, 525]),
        _region(
            2, "vision_footnote", "Flow of participants through the trial", [100, 528, 600, 545]
        ),
    ]

    contents, _body, notes = _parse([page])

    (figure,) = contents.figures
    assert "Flow of participants through the trial" in figure.caption
    assert not any("Flow of participants" in note for note in notes)


def test_body_paragraph_under_a_finished_caption_stays_body():
    page = [
        _region(0, "image", bbox=[100, 100, 900, 500], image_b64="fig"),
        _region(1, "figure_title", "Figure 2. Flow of participants.", [100, 510, 600, 525]),
        _region(2, "text", _PARAGRAPH, [100, 530, 900, 600]),
    ]

    contents, body, _notes = _parse([page])

    assert contents.figures[0].caption == "Figure 2. Flow of participants."
    assert any(text.startswith("The cohort was followed") for text in body)


def test_capitalised_heading_under_an_unfinished_caption_is_not_joined():
    """An appendix: a caption without a full stop, then the next appendix's
    heading read as a caption region directly under it."""
    page = [
        _region(0, "image", bbox=[119, 105, 479, 284], image_b64="fig"),
        _region(1, "figure_title", "Figure D5: A larger woven basket", [119, 290, 479, 305]),
        _region(2, "figure_title", "Appendix E\nTextiles and Clothing", [119, 310, 479, 340]),
    ]

    contents, _body, _notes = _parse([page])

    assert contents.figures[0].caption == "Figure D5: A larger woven basket"


def test_two_captions_side_by_side_stay_apart():
    page = [
        _region(0, "image", bbox=[60, 100, 480, 400], image_b64="left"),
        _region(1, "image", bbox=[520, 100, 940, 400], image_b64="right"),
        _region(2, "figure_title", "Figure 1. Left panel results by", [60, 410, 480, 430]),
        _region(3, "figure_title", "Figure 2. Right panel results.", [520, 410, 940, 430]),
    ]

    contents, _body, _notes = _parse([page])

    assert sorted(figure.caption for figure in contents.figures) == [
        "Figure 1. Left panel results by",
        "Figure 2. Right panel results.",
    ]
