"""Recovery of complete leading objects from a truncated JSON array."""

from bibr.utils.json_salvage import salvage_array_objects

_A = '{"index": 1, "title": "A"}'
_B = '{"index": 2, "title": "B"}'


def test_complete_array_returns_all_objects():
    text = f'{{"references": [{_A}, {_B}]}}'
    objs = salvage_array_objects(text, "references")
    assert [o["index"] for o in objs] == [1, 2]
    assert objs[1]["title"] == "B"


def test_truncated_mid_object_keeps_complete_prefix():
    text = f'{{"references": [{_A}, {_B}, {{"index": 3, "title": "C"'
    objs = salvage_array_objects(text, "references")
    assert [o["index"] for o in objs] == [1, 2]


def test_truncated_mid_string_keeps_complete_prefix():
    # The third object's string value is cut off before its closing quote.
    text = f'{{"references": [{_A}, {{"index": 2, "title": "Bee and the un'
    objs = salvage_array_objects(text, "references")
    assert [o["index"] for o in objs] == [1]


def test_trailing_comma_after_last_complete_object():
    text = f'{{"references": [{_A}, {_B}, '
    objs = salvage_array_objects(text, "references")
    assert [o["index"] for o in objs] == [1, 2]


def test_brace_in_string_does_not_break_matching():
    tricky = '{"index": 1, "title": "a { nested } brace"}'
    text = f'{{"references": [{tricky}, {{"index": 2'
    objs = salvage_array_objects(text, "references")
    assert [o["title"] for o in objs] == ["a { nested } brace"]


def test_escaped_quote_in_string():
    tricky = '{"index": 1, "title": "quote \\" here"}'
    text = f'{{"references": [{tricky}, {{"index": 2, "title": "trunc'
    objs = salvage_array_objects(text, "references")
    assert objs[0]["title"] == 'quote " here'


def test_garbage_input_returns_empty():
    assert salvage_array_objects("not json at all", "references") == []
    assert salvage_array_objects('{"other": [1, 2]}', "references") == []
    assert salvage_array_objects('{"references":', "references") == []


def test_empty_array_returns_empty():
    assert salvage_array_objects('{"references": []}', "references") == []
