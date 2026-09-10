"""Tests for the column-join heuristic in ``PDFParser._should_join``."""

from bibr.structure.pdf_parser import PDFParser


def test_should_join_capitalized_continuation_when_prev_lacks_terminal_punct():
    # No terminal punct on prev -> should join even if next starts with capital.
    assert PDFParser._should_join("the participants performed", "The task included") is True
    # Has terminal punct -> never join.
    assert PDFParser._should_join("done.", "The task included") is False
    # Lowercase continuation still joins.
    assert PDFParser._should_join("methodol", "ogy used") is True
