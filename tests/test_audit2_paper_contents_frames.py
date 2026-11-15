"""Every DataFrame view of PaperContents has its columns, empty or not."""

import pytest

from bibr.paper_contents import PaperContents, PaperEquation, PaperSection, PaperSentence


def _contents(**kwargs):
    return PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={}, **kwargs)


@pytest.mark.parametrize(
    ("view", "columns"),
    [
        ("links_df", ["url", "link_text", "text_id"]),
        ("text_df", ["text_id", "section_id", "paragraph_id", "text", "page_number"]),
        ("equations_df", ["text_id", "grp_id", "lhs", "df", "comp", "rhs"]),
        (
            "sentences_df",
            [
                "text_id",
                "section_id",
                "paragraph_id",
                "text",
                "section_name",
                "section_type",
                "page_number",
            ],
        ),
    ],
)
def test_an_empty_view_has_the_columns_of_a_full_one(view, columns):
    frame = getattr(_contents(), view)
    assert frame.empty
    assert list(frame.columns) == columns


def test_a_full_view_keeps_its_rows():
    contents = _contents(
        equations=[PaperEquation(text_id=1, grp_id=1, lhs="t", comp="=", rhs="2.1")],
    )
    contents.sections = [PaperSection(section_id=1, header="Results", level=1, parent_section_id=0)]
    contents.sentences = [
        PaperSentence(text_id=1, text="t = 2.1.", section_id=1, paragraph_id=1, page_number=2)
    ]
    assert contents.sentences_df.to_dict("records")[0]["section_name"] == "Results"
    assert contents.equations_df.to_dict("records") == [
        {"text_id": 1, "grp_id": 1, "lhs": "t", "df": "", "comp": "=", "rhs": "2.1"}
    ]
