"""Unit tests for the pure BIO-span decoder used by the v4 ref parser.

No model/weights required: ``decode_bio_spans`` is a pure function over
(tags, offsets, text).
"""

from bibr.ner.decode import decode_bio_spans, map_fields_to_paper_ref


def test_single_field_multi_token_span():
    text = "Smith J"
    tags = ["B-AUTHOR", "I-AUTHOR"]
    offsets = [(0, 5), (6, 7)]
    assert decode_bio_spans(tags, offsets, text) == {"AUTHOR": "Smith J"}


def test_b_tag_closes_previous_field():
    text = "Smith 2020"
    tags = ["B-AUTHOR", "B-YEAR"]
    offsets = [(0, 5), (6, 10)]
    assert decode_bio_spans(tags, offsets, text) == {"AUTHOR": "Smith", "YEAR": "2020"}


def test_o_tag_closes_current_field():
    text = "Smith , 2020"
    tags = ["B-AUTHOR", "O", "B-YEAR"]
    offsets = [(0, 5), (6, 7), (8, 12)]
    assert decode_bio_spans(tags, offsets, text) == {"AUTHOR": "Smith", "YEAR": "2020"}


def test_i_tag_with_mismatched_field_starts_new_span():
    # Defensive: an I-TITLE while inside AUTHOR begins a fresh TITLE span.
    text = "Smith Title"
    tags = ["B-AUTHOR", "I-TITLE"]
    offsets = [(0, 5), (6, 11)]
    assert decode_bio_spans(tags, offsets, text) == {"AUTHOR": "Smith", "TITLE": "Title"}


def test_trailing_field_without_closing_o_is_captured():
    text = "A Great Title"
    tags = ["B-TITLE", "I-TITLE", "I-TITLE"]
    offsets = [(0, 1), (2, 7), (8, 13)]
    assert decode_bio_spans(tags, offsets, text) == {"TITLE": "A Great Title"}


def test_disjoint_spans_same_field_joined_with_single_space():
    text = "Smith and Jones"
    tags = ["B-AUTHOR", "O", "B-AUTHOR"]
    offsets = [(0, 5), (6, 9), (10, 15)]
    assert decode_bio_spans(tags, offsets, text) == {"AUTHOR": "Smith Jones"}


def test_whitespace_only_span_is_dropped():
    text = "Smith   "
    tags = ["B-AUTHOR", "B-TITLE"]
    offsets = [(0, 5), (5, 8)]  # second span is only spaces
    assert decode_bio_spans(tags, offsets, text) == {"AUTHOR": "Smith"}


def test_empty_inputs_return_empty_dict():
    assert decode_bio_spans([], [], "") == {}


# --- map_fields_to_paper_ref ---------------------------------------------------


def test_maps_giant_field_names_to_paper_ref_names():
    raw = {"TITLE": "A Study", "AUTHOR": "Smith, J.", "CONTAINER": "Journal of X"}
    assert map_fields_to_paper_ref(raw) == {
        "title": "A Study",
        "authors": "Smith, J.",
        "container": "Journal of X",
    }


def test_year_is_parsed_to_int():
    assert map_fields_to_paper_ref({"YEAR": "2020"}) == {"year": 2020}


def test_year_strips_surrounding_punctuation():
    assert map_fields_to_paper_ref({"YEAR": "(2019)."}) == {"year": 2019}


def test_non_numeric_year_is_dropped():
    assert map_fields_to_paper_ref({"YEAR": "in press"}) == {}


def test_pages_map_to_first_and_last_page():
    raw = {"PAGES": "100", "PAGE_RANGE_END": "115"}
    assert map_fields_to_paper_ref(raw) == {"first_page": "100", "last_page": "115"}


def test_page_range_start_alone_maps_to_first_page():
    assert map_fields_to_paper_ref({"PAGE_RANGE_START": "100"}) == {"first_page": "100"}


def test_page_range_start_wins_over_pages_regardless_of_order():
    assert map_fields_to_paper_ref({"PAGES": "100-115", "PAGE_RANGE_START": "100"}) == {
        "first_page": "100"
    }
    assert map_fields_to_paper_ref({"PAGE_RANGE_START": "100", "PAGES": "100-115"}) == {
        "first_page": "100"
    }


def test_pages_and_page_range_start_collision_logs(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="bibr.ner.decode"):
        map_fields_to_paper_ref({"PAGES": "100-115", "PAGE_RANGE_START": "100"})
    assert any("PAGE_RANGE_START" in rec.message for rec in caplog.records)


def test_bio_sequence_with_both_pages_and_page_range_start():
    text = "pp. 100-115 100"
    tags = ["O", "B-PAGES", "B-PAGE_RANGE_START"]
    offsets = [(0, 3), (4, 11), (12, 15)]
    raw = decode_bio_spans(tags, offsets, text)
    assert raw == {"PAGES": "100-115", "PAGE_RANGE_START": "100"}
    assert map_fields_to_paper_ref(raw) == {"first_page": "100"}

    tags_rev = ["O", "B-PAGE_RANGE_START", "B-PAGES"]
    raw_rev = decode_bio_spans(tags_rev, offsets, text)
    assert raw_rev == {"PAGE_RANGE_START": "100-115", "PAGES": "100"}
    assert map_fields_to_paper_ref(raw_rev) == {"first_page": "100-115"}


# --- dashed span left in the PAGE_RANGE_END slot -------------------------------


def test_dashed_span_in_end_slot_is_split():
    # "Virulence, 5(1), 20-26." — the tagger tags the whole span PAGE_RANGE_END
    # and leaves "20" as O, so first_page would otherwise be null.
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "20-26"}) == {
        "first_page": "20",
        "last_page": "26",
    }


def test_dashed_span_in_end_slot_handles_en_and_em_dashes():
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "41–49"}) == {
        "first_page": "41",
        "last_page": "49",
    }
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "41—49"}) == {
        "first_page": "41",
        "last_page": "49",
    }
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "41--49"}) == {
        "first_page": "41",
        "last_page": "49",
    }


def test_dashed_span_in_end_slot_keeps_page_labels():
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "S13-S20"}) == {
        "first_page": "S13",
        "last_page": "S20",
    }


def test_dashed_span_in_end_slot_agreeing_first_page_is_trimmed():
    raw = {"PAGE_RANGE_START": "20", "PAGE_RANGE_END": "20-26"}
    assert map_fields_to_paper_ref(raw) == {"first_page": "20", "last_page": "26"}


def test_dashed_span_in_end_slot_disagreeing_first_page_is_left_alone():
    # Never overwrite a first_page the tagger asserted independently.
    raw = {"PAGE_RANGE_START": "18", "PAGE_RANGE_END": "20-26"}
    assert map_fields_to_paper_ref(raw) == {"first_page": "18", "last_page": "20-26"}


def test_bare_end_page_is_untouched():
    # Model-tagging gap (no span to split); leave it exactly as tagged.
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "26"}) == {"last_page": "26"}


def test_non_range_end_slot_is_untouched():
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "e0123456"}) == {"last_page": "e0123456"}
    assert map_fields_to_paper_ref({"PAGE_RANGE_END": "12-15-18"}) == {"last_page": "12-15-18"}


def test_dashed_span_split_from_bio_sequence():
    text = "Virulence, 5(1), 20-26."
    tags = ["O", "O", "B-PAGE_RANGE_END"]
    offsets = [(0, 10), (11, 15), (17, 22)]
    raw = decode_bio_spans(tags, offsets, text)
    assert raw == {"PAGE_RANGE_END": "20-26"}
    assert map_fields_to_paper_ref(raw) == {"first_page": "20", "last_page": "26"}


def test_unknown_field_is_ignored():
    # ARXIV/PMID/etc. have no PaperReference target → dropped.
    assert map_fields_to_paper_ref({"ARXIV": "2101.00001", "DOI": "10.1/x"}) == {"doi": "10.1/x"}


def test_empty_fields_map_to_empty_dict():
    assert map_fields_to_paper_ref({}) == {}
