import pytest

gr = pytest.importorskip("gradio")  # demo is gated behind gradio
if not hasattr(gr, "Blocks"):
    pytest.skip("gradio not fully installed", allow_module_level=True)

from bibr.demo.local_app import (
    _build_figures_html,
    _build_notes_md,
    _build_paper_card_html,
    _build_references_data,
    _build_tables_html,
    _build_text_html,
    _error_html,
    _explain_failure,
)


def test_paper_card_shows_extracted_metadata_as_text():
    """A crafted title, author, keyword or abstract must not become markup in the card."""
    result = {
        "metadata": {
            "title": "Evil <img src=https://attacker.example/q.png> <script>alert(1)</script>",
            "abstract": '<a href="javascript:alert(1)">click</a>',
            "journal": "<b>Journal</b>",
            "paper_type": "<b>article</b>",
            "oecd_l1": "Social <i>sciences</i>",
            "oecd_l2": "Psychology",
            "keywords": ["<u>k</u>", "Parkinson's disease"],
        },
        "authors": [{"given": "<svg onload=x>", "family": "Doe", "corresponding": True}],
    }
    html = _build_paper_card_html(result)
    for raw in ("<img", "<script", '<a href="javascript', "<b>", "<i>", "<u>", "<svg"):
        assert raw not in html
    assert "Evil &lt;img" in html
    assert "&lt;svg onload=x&gt; Doe<sup>*</sup>" in html
    assert "Parkinson&#x27;s disease" in html  # shown as typed once the browser decodes it
    assert "Social &lt;i&gt;sciences&lt;/i&gt; › Psychology" in html


def test_paper_card_doi_link_only_points_at_doi_org():
    """The DOI comes from the document: its link is percent-encoded under https://doi.org/."""
    html = _build_paper_card_html(
        {"metadata": {"title": "T", "doi": '10.1/x" onmouseover="alert(1) <b>'}}
    )
    assert 'href="https://doi.org/10.1/x%22%20onmouseover%3D%22alert%281%29%20%3Cb%3E"' in html
    assert 'onmouseover="' not in html
    assert "<b>" not in html


def test_paper_card_leads_with_what_a_reader_wants():
    html = _build_paper_card_html(
        {
            "metadata": {
                "title": "Ageing and memory",
                "journal": "Memory",
                "published_date": "2020-03",
                "volume": "12",
                "issue": "3",
                "first_page": "5",
                "last_page": "9",
                "abstract": "We tested memory.",
            },
            "authors": [{"given": "Ada", "family": "Vole"}, {"literal": "The Group"}],
            "bib": [{}, {}],
            "text": [{}] * 50,
        }
    )
    assert html.index("Ageing and memory") < html.index("Ada Vole, The Group")
    assert "<em>Memory</em> · 2020 · vol. 12 (3) · pp. 5–9" in html
    assert "We tested memory." in html
    assert "2 references" in html
    assert "sentence" not in html  # an internal count, not shown


def test_references_doi_cell_is_an_escaped_markdown_link():
    rows = _build_references_data(
        {"bib": [{"bib_id": 1, "title": "T", "doi": "10.1/a)(b", "year": 2020}]}
    )
    assert rows[0][5] == "[10\\.1/a\\)\\(b](https://doi.org/10.1/a%29%28b)"
    assert _build_references_data({"bib": [{"bib_id": 2, "doi": None}]})[0][5] == ""


def test_notes_escape_the_fast_mode_message():
    notes = _build_notes_md(
        {
            "extraction": {
                "warnings": [
                    {
                        "code": "LLM_SKIPPED",
                        "message": "Fast mode made no LLM calls. "
                        "Not produced: ![x](https://attacker.example/p.png).",
                    }
                ]
            }
        }
    )
    assert "](" not in notes
    assert "https://" not in notes


def test_error_card_escapes_the_file_name():
    html = _error_html(
        _explain_failure(
            RuntimeError("x"), file_name="<img src=x onerror=1>.pdf", text_layer_only=True
        )
    )
    assert "<img" not in html
    assert "&lt;img src=x onerror=1&gt;.pdf" in html


def test_text_html_escapes_script_tags():
    result = {
        "text": [{"section_id": 1, "paragraph_id": 1, "text": "<script>alert(1)</script>"}],
        "sections": [{"section_id": 1, "level": 1, "header": "Intro<script>x</script>"}],
    }
    html = _build_text_html(result)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Intro&lt;script&gt;x&lt;/script&gt;" in html


def test_text_html_handles_null_section_id():
    """Root-section sentences export with section_id=None (v10.2 schema).

    The renderer groups sentences by section_id and sorts the groups; a mix of
    None (root) and int ids must not raise ``TypeError: '<' not supported
    between instances of 'int' and 'NoneType'``.
    """
    result = {
        "text": [
            {"section_id": None, "paragraph_id": 1, "text": "Front matter line."},
            {"section_id": 2, "paragraph_id": 1, "text": "Body in section two."},
            {"section_id": 1, "paragraph_id": 1, "text": "Body in section one."},
        ],
        "sections": [
            {"section_id": 1, "level": 1, "header": "One"},
            {"section_id": 2, "level": 1, "header": "Two"},
        ],
    }
    html = _build_text_html(result)  # must not raise
    assert "Front matter line." in html
    assert "Body in section one." in html
    assert "Body in section two." in html


def test_table_html_escapes_cell_content():
    result = {
        "table": [
            {
                "table_id": 1,
                "contents": [
                    ["<b>h1</b>", "h2"],
                    ["<img src=x onerror=1>", "ok"],
                ],
            }
        ]
    }
    html = _build_tables_html(result)
    assert "<img src=x onerror=1>" not in html
    assert "&lt;img src=x onerror=1&gt;" in html
    assert "&lt;b&gt;h1&lt;/b&gt;" in html


def test_table_caption_is_shown_and_escaped():
    html = _build_tables_html(
        {"table": [{"label": "2", "caption": "Tabel 2: <b>Means</b>", "contents": [["a"]]}]}
    )
    assert "Tabel 2: &lt;b&gt;Means&lt;/b&gt;" in html
    assert "Table 2." not in html  # the caption already carries its label
    html = _build_tables_html({"table": [{"label": "3", "caption": "Means", "contents": []}]})
    assert "<strong>Table 3.</strong> Means" in html


def test_figure_images_carry_escaped_alt_text():
    html = _build_figures_html(
        {"figure": [{"label": "1", "caption": 'A "quoted" <b>chart</b>', "image": "data:x"}]},
        images_enabled=True,
    )
    assert 'alt="A &quot;quoted&quot; &lt;b&gt;chart&lt;/b&gt;"' in html


def test_text_html_shows_a_caption_in_its_float_section():
    result = {
        "text": [
            {"text_id": 1, "section_id": 1, "paragraph_id": 1, "text": "Body."},
            {"text_id": 2, "section_id": None, "paragraph_id": 9, "text": "Table 1. <b>Means</b>"},
        ],
        "sections": [{"section_id": 1, "level": 1, "header": "Methods"}],
        "table": [{"table_id": 1, "text_id": 2, "section_id": 1}],
    }
    html = _build_text_html(result)
    assert html.index("Methods") < html.index("Table 1.")
    assert '<p class="bibr-caption">Table 1. &lt;b&gt;Means&lt;/b&gt;</p>' in html
