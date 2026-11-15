from __future__ import annotations

from bibr.extract.front_role_features import FrontRegion
from bibr.front_tagger.contract import (
    BIO_TAGS,
    FEATURE_DIM,
    FEATURE_KEYS,
    WORD_RE,
    build_tagger_input,
)

from .helpers import regions, tagger_input


def test_marks_glued_to_names_split_off():
    assert WORD_RE.findall("Smith1,2, O'Neil* and Jean-Pierre") == [
        "Smith",
        "1",
        ",",
        "2",
        ",",
        "O'Neil",
        "*",
        "and",
        "Jean-Pierre",
    ]


def test_cjk_characters_are_words_of_their_own():
    assert WORD_RE.findall("张伟，李娜") == ["张", "伟", "，", "李", "娜"]
    assert WORD_RE.findall("Müller über") == ["Müller", "über"]


def test_regions_join_with_newlines_and_offsets_index_the_text():
    inp = tagger_input(("doc_title", "A study\nof things"), ("text", "Ann Lee1"))
    assert inp.text == "A study of things\nAnn Lee1"
    assert [inp.text[w.start : w.end] for w in inp.words] == [w.text for w in inp.words]
    assert [w.region for w in inp.words] == [0, 0, 0, 0, 1, 1, 1]
    assert all(len(row) == FEATURE_DIM == len(FEATURE_KEYS) for row in inp.features)


def test_non_text_regions_and_later_pages_are_left_out():
    rows = [
        *regions(("image", "Figure text"), ("doc_title", "Title")),
        FrontRegion(page=2, index=0, label="text", text="Body", x0=0, y0=0, x1=1, y1=0.1),
    ]
    inp = build_tagger_input(rows)
    assert inp.text == "Title"


def test_layout_label_is_one_hot():
    inp = tagger_input(("header", "Running head"), ("weird_label", "x"))
    header_col = FEATURE_KEYS.index("label_header")
    other_col = FEATURE_KEYS.index("label_other")
    assert inp.features[0][header_col] == 1.0
    assert inp.features[-1][other_col] == 1.0 and inp.features[-1][header_col] == 0.0


def test_tag_set_is_bio_over_fields():
    assert BIO_TAGS[0] == "O"
    assert "B-FURNITURE" in BIO_TAGS and "I-AFF_MARK" in BIO_TAGS
    assert len(BIO_TAGS) % 2 == 1
