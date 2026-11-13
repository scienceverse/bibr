"""Captions printed in other languages reach their floats."""

import pytest

from bibr.structure.float_labels import caption_label
from bibr.structure.pdf_parser import PDFParser

_TABLE_HTML = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"


def _region(label, content, bbox):
    return {"index": 0, "label": label, "native_label": label, "content": content, "bbox_2d": bbox}


def _parse(pages):
    parser = PDFParser(pages)
    contents = parser.parse()
    texts = [text for text, _, _, needs_seg, _ in parser._deferred_texts if needs_seg]
    parser.apply_segmentation(contents, [[text] for text in texts])
    parser.create_content_sections(contents)
    return contents


def _body(contents):
    float_sections = {table.section_id for table in contents.tables} | {
        figure.section_id for figure in contents.figures
    }
    return [s.text for s in contents.sentences if s.section_id not in float_sections]


@pytest.mark.parametrize(
    "caption",
    [
        "Tabel 1: Criteria van patiëntgerichtheid.",
        "Tabla 1. Características de la muestra.",
        "Tableau 1 : Caractéristiques des participants.",
        "Tabela 1. Características da amostra.",
        "Tabelle 1: Stichprobenbeschreibung.",
        "Tabella 1. Caratteristiche del campione.",
    ],
)
def test_a_table_caption_in_another_language_captions_its_table(caption):
    contents = _parse(
        [
            [
                _region("text", "Some body text before the table.", [100, 100, 900, 200]),
                _region("figure_title", caption, [100, 300, 900, 330]),
                _region("table", _TABLE_HTML, [100, 340, 900, 600]),
            ]
        ]
    )
    assert [table.caption for table in contents.tables] == [caption]
    assert contents.tables[0].label == "1"
    assert caption not in _body(contents)


def test_a_bare_table_label_in_another_language_composes_with_its_title():
    contents = _parse(
        [
            [
                _region("figure_title", "Tabel 2", [100, 300, 400, 320]),
                _region("text", "Aspecten van de relatie", [100, 322, 600, 340]),
                _region("table", _TABLE_HTML, [100, 350, 900, 600]),
            ]
        ]
    )
    assert [table.caption for table in contents.tables] == ["Tabel 2 Aspecten van de relatie"]


def test_a_figure_caption_in_another_language_read_as_body_text_captions_its_figure():
    contents = _parse(
        [
            [
                _region("image", "", [100, 100, 900, 500]),
                _region("text", "Figuur 1. Probleemboom van de analyse.", [100, 510, 900, 540]),
                _region("text", "Body text after the figure.", [100, 600, 900, 700]),
            ]
        ]
    )
    assert [figure.caption for figure in contents.figures] == [
        "Figuur 1. Probleemboom van de analyse."
    ]
    assert contents.figures[0].label == "1"


@pytest.mark.parametrize(
    ("caption", "kind", "label"),
    [
        ("Tabela 1. Dados", "table", "1"),
        ("TABELA 1. DADOS", "table", "1"),
        ("TABELLE 2", "table", "2"),
        ("Tableau 3 : Résultats", "table", "3"),
        ("Abb. 3: Ergebnisse", "figure", "3"),
        ("Abbildung 4. Ergebnisse", "figure", "4"),
        ("Figura 2. Resultados", "figure", "2"),
        ("FIGUUR 5", "figure", "5"),
        ("Tabla 1. Resultados", "figure", None),
        ("Figura 2. Resultados", "table", None),
        ("SUPPLEMENTARY FIGURE LEGENDS", "figure", None),
        ("TABELAS ANEXAS", "table", None),
    ],
)
def test_caption_label_reads_captions_in_other_languages(caption, kind, label):
    assert caption_label(caption, kind) == label
