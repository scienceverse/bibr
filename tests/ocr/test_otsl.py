"""Regression coverage for Paddle OTSL table decoding."""

import pytest

from bibr.ocr.otsl import check_otsl_completeness, decode_otsl


@pytest.mark.parametrize(
    "raw",
    [
        "<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>",
        "<fcel>A<lcel><nl><ucel><xcel><nl>",
        "  <fcel>A<ecel><nl>\n",
    ],
)
def test_check_otsl_completeness_accepts_closed_rectangular_grid(raw):
    result = check_otsl_completeness(raw)

    assert result.complete is True
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", "empty"),
        ("plain text", "no_grid_cells"),
        ("<fcel>A<fcel>B", "missing_terminal_nl"),
        ("<fcel>A<fcel>B<nl>trailing", "missing_terminal_nl"),
        ("<fcel>A<fcel>B<nl><fcel>C<nl>", "ragged_rows"),
        ("<lcel><nl>", "malformed_structure"),
    ],
)
def test_check_otsl_completeness_rejects_incomplete_raw_output(raw, reason):
    result = check_otsl_completeness(raw)

    assert result.complete is False
    assert reason in result.reasons


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>",
            "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
        ),
        (
            "<ecel><fcel>A<ecel><nl>",
            "<table><tr><td></td><td>A</td><td></td></tr></table>",
        ),
        (
            "<fcel>A<fcel>B",
            "<table><tr><td>A</td><td>B</td></tr></table>",
        ),
        (
            "<fcel>A<nl><fcel>B<fcel>C<nl>",
            "<table><tr><td>A</td><td></td></tr><tr><td>B</td><td>C</td></tr></table>",
        ),
        (
            "<fcel>A<lcel><fcel>B<nl>",
            '<table><tr><td colspan="2">A</td><td>B</td></tr></table>',
        ),
        (
            "<fcel>A<fcel>B<nl><ucel><fcel>C<nl>",
            '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>',
        ),
        (
            "<fcel>A<lcel><nl><ucel><xcel><nl>",
            '<table><tr><td rowspan="2" colspan="2">A</td></tr><tr></tr></table>',
        ),
        (
            '<fcel>line one\nline two & <tag> "quoted" \\alpha<nl>',
            "<table><tr><td>line one<br>line two &amp; &lt;tag&gt; &quot;quoted&quot; \\alpha</td></tr></table>",
        ),
        (
            r"<fcel>line one\n line two<nl>",
            "<table><tr><td>line one<br> line two</td></tr></table>",
        ),
        (
            "<fcel>line one\r\nline two\rline three<nl>",
            "<table><tr><td>line one<br>line two<br>line three</td></tr></table>",
        ),
    ],
)
def test_decode_otsl_renders_canonical_rectangular_html(raw, expected):
    result = decode_otsl(raw)

    assert result.html == expected
    assert result.warnings == ()


@pytest.mark.parametrize(
    "raw",
    [
        "<lcel><nl>",
        "<ucel><nl>",
        "<fcel>A<fcel>B<nl><ucel><xcel><nl>",
        "<fcel>A<lcel><nl><ucel><fcel>B<nl>",
    ],
)
def test_decode_otsl_malformed_spans_degrade_to_unmerged_text_preserving_html(raw):
    result = decode_otsl(raw)

    assert result.warnings
    assert result.warnings[0].startswith("Malformed Paddle OTSL:")
    assert "<table>" in result.html
    assert "rowspan" not in result.html
    assert "colspan" not in result.html


def test_decode_otsl_malformed_input_keeps_all_anchor_text():
    result = decode_otsl("<fcel>A<fcel>B<nl><ucel><xcel><nl>")

    assert result.html == "<table><tr><td>A</td><td>B</td></tr><tr><td></td><td></td></tr></table>"


@pytest.mark.parametrize(
    "command",
    [r"\nu", r"\nabla", r"\neq", r"\newcommand"],
)
def test_decode_otsl_preserves_latex_commands_starting_with_n(command):
    result = decode_otsl(f"<fcel>{command}<nl>")

    assert result.html == f"<table><tr><td>{command}</td></tr></table>"


def test_decode_otsl_converts_literal_encoded_newline_before_non_command_text():
    result = decode_otsl(r"<fcel>line one\n line two\n2<nl>")

    assert result.html == "<table><tr><td>line one<br> line two<br>2</td></tr></table>"


_SPANNED = '<table><tr><td colspan="2">A</td></tr><tr><td>B</td><td>C</td></tr></table>'


@pytest.mark.parametrize(
    "raw",
    [
        "<fcel>A<lcel><nl><fcel>B<fcel>C<nl>",
        "<fcel>A<lcel><nl><fcel>B<fcel>C<nl>\n",
        "\n<fcel>A<lcel><nl><fcel>B<fcel>C<nl>",
        "  <fcel>A<lcel><nl><fcel>B<fcel>C<nl>  \n",
    ],
)
def test_decode_otsl_ignores_surrounding_whitespace(raw):
    """check_otsl_completeness normalises with .strip() and decode_otsl did
    not, so a single trailing newline — routine from OpenAI-compatible chat
    completions, i.e. the default PaddleOCR-VL path — became a phantom cell
    and the colspan was lost."""
    assert decode_otsl(raw).html == _SPANNED
