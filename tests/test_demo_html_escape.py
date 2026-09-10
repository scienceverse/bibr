import pytest

gr = pytest.importorskip("gradio")  # demo is gated behind gradio
if not hasattr(gr, "Blocks"):
    pytest.skip("gradio not fully installed", allow_module_level=True)

from bibr.demo.local_app import _build_tables_html, _build_text_html


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
