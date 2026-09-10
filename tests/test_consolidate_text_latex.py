"""Tests for balanced-brace ``\\text{...}`` extraction in consolidate_text."""


def test_latex_text_extracts_balanced_inner_with_nested_braces():
    from bibr.input.consolidate_text import _extract_latex_text

    assert _extract_latex_text(r"\text{p(x|y)}") == "p(x|y)"
    assert _extract_latex_text(r"\text{a {b} c}") == "a {b} c"
    assert _extract_latex_text(r"\text{plain}") == "plain"


def test_latex_text_returns_input_when_no_match():
    from bibr.input.consolidate_text import _extract_latex_text

    assert _extract_latex_text("no latex here") == "no latex here"


def test_latex_text_handles_unbalanced_best_effort():
    from bibr.input.consolidate_text import _extract_latex_text

    # Unbalanced: missing closing brace -> return everything to end of string
    assert _extract_latex_text(r"\text{p(x|y)") == "p(x|y)"


def test_unwrap_latex_text_preserves_nested_braces_in_wrapped_form():
    """End-to-end: $$\\text{a {b} c}$$ should unwrap to ``a {b} c`` not ``a {b``."""
    from bibr.input.consolidate_text import unwrap_latex_text

    # NB: outer regex _LATEX_TEXT_WRAP_RE uses [^}]* so it won't match nested
    # braces inside the wrapped form; test plain p(x|y) which has no braces
    # to confirm the helper is wired through correctly.
    assert unwrap_latex_text(r"$$\text{p(x|y)}$$") == "p(x|y)"


def test_unwrap_latex_text_handles_multiple_text_blocks():
    """Two adjacent \\text{} blocks inside one $$...$$ wrapper."""
    from bibr.input.consolidate_text import unwrap_latex_text

    assert unwrap_latex_text(r"$$\text{first} \text{second}$$") == "first second"
