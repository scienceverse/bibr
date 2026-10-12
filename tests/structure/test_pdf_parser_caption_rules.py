"""Small caption rules: captions read as notes, label grammar gaps (#136).

Geometry is modelled on the reported papers (0..1000 page space); the text is
synthetic.
"""

from __future__ import annotations

import pytest

from bibr.structure.pdf_parser import PDFParser

_TABLE = "| Group | n |\n|---|---|\n| Treated | 95 |"


def _region(index, label, content="", bbox=None, image_b64=None):
    value = {"index": index, "label": label, "content": content, "bbox_2d": bbox}
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def _parse(pages):
    parser = PDFParser(pages)
    return parser, parser.parse()


def test_full_table_caption_read_as_a_note_captions_its_table():
    """A trial report: the layout model read the table's caption, printed
    under it, as a note; the table was exported uncaptioned."""
    page = [
        _region(1, "table", _TABLE, [204, 161, 549, 624]),
        _region(
            2,
            "vision_footnote",
            "Table 1: Characteristics of the intention-to-treat population at baseline",
            [207, 631, 545, 644],
        ),
        _region(3, "vision_footnote", "Data are n (%). HR=hazard ratio.", [207, 650, 545, 662]),
    ]

    parser, contents = _parse([page])

    (table,) = contents.tables
    assert (
        table.caption == "Table 1: Characteristics of the intention-to-treat population at baseline"
    )
    assert [note[0] for note in parser._footnotes] == ["Data are n (%). HR=hazard ratio."]


def test_caption_shaped_note_without_a_float_of_its_kind_stays_a_note():
    page = [
        _region(0, "text", "Results were consistent across sites.", [100, 100, 900, 150]),
        _region(
            1, "vision_footnote", "Table 2: values are adjusted for age.", [100, 900, 900, 915]
        ),
    ]

    parser, contents = _parse([page])

    assert contents.tables == []
    assert [note[0] for note in parser._footnotes] == ["Table 2: values are adjusted for age."]


@pytest.mark.parametrize("caption", ["Table Il", "T able |"])
def test_garbled_table_caption_never_captions_a_figure(caption):
    """A conference paper's scan: "Table II" read as "Table Il" was typed as
    a figure caption and took the figure under it from its own caption."""
    page = [
        _region(0, "figure_title", caption, [269, 600, 319, 611]),
        _region(1, "table", _TABLE, [106, 622, 466, 684]),
        _region(2, "image", bbox=[130, 700, 450, 870], image_b64="fig9"),
        _region(3, "figure_title", "Fig.9. Filter method for haze removal", [156, 887, 419, 900]),
    ]

    _parser, contents = _parse([page])

    (figure,) = contents.figures
    assert figure.caption == "Fig.9. Filter method for haze removal"
    assert figure.label == "9"


def test_fig_dot_label_without_a_space_is_a_caption_in_body_text():
    """Old journal scans print "Fig.2." with no space after the dot; read as
    body text, such a caption stayed in the body."""
    page = [
        _region(0, "image", bbox=[73, 360, 527, 700], image_b64="fig2"),
        _region(
            1,
            "text",
            "Fig.2. Temperature-shift experiments. a, Temperature-sensitive periods of the mutant.",
            [73, 76, 527, 347],
        ),
    ]

    _parser, contents = _parse([page])

    (figure,) = contents.figures
    assert figure.label == "2"
    assert figure.caption.startswith("Fig.2. Temperature-shift experiments.")


def test_body_sentence_opening_with_a_figure_mention_stays_body():
    page = [
        _region(0, "image", bbox=[73, 360, 527, 700], image_b64="fig2"),
        _region(1, "text", "Fig.2 shows the temperature-sensitive periods.", [73, 76, 527, 100]),
    ]

    parser, contents = _parse([page])

    assert [entry.text for entry in parser.assembler.entries] == [
        "Fig.2 shows the temperature-sensitive periods."
    ]


def test_named_supplementary_figure_caption_labels_its_figure():
    page = [
        _region(0, "chart", bbox=[69, 120, 937, 770], image_b64="sup"),
        _region(
            1,
            "figure_title",
            "Sup. Fig. PT - Proximal tubule injury states in relation to proteinuria",
            [69, 788, 937, 835],
        ),
    ]

    _parser, contents = _parse([page])

    assert [(figure.label, len(figure.parts)) for figure in contents.figures] == [("PT", 1)]


def test_panel_letter_inside_a_single_captioned_figure_does_not_take_its_caption():
    """A supplementary figure page: one figure box, a panel letter printed
    inside it, the caption under it. The letter outscored the caption."""
    page = [
        _region(1, "figure_title", "C", [575, 114, 592, 130]),
        _region(2, "image", bbox=[63, 116, 908, 823], image_b64="sup"),
        _region(4, "figure_title", "Sup. Fig. PSEUDOTIME -", [58, 852, 358, 877]),
        _region(5, "figure_title", "e", [588, 480, 607, 496]),
    ]

    _parser, contents = _parse([page])

    (figure,) = contents.figures
    assert figure.caption.startswith("Sup. Fig. PSEUDOTIME")
