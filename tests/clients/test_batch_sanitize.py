from bibr.clients.batch import _sanitize_strict


def _walk(node):
    """Yield every dict in a nested JSON-schema structure."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_object_gets_additional_properties_false_and_required_all_keys():
    schema = {
        "type": "object",
        "title": "Thing",
        "properties": {
            "a": {"type": "string", "title": "A", "default": "x"},
            "b": {"type": "integer"},
        },
    }
    out = _sanitize_strict(schema)
    assert out["additionalProperties"] is False
    assert set(out["required"]) == {"a", "b"}


def test_drops_title_and_default_recursively():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "title": "A", "default": "x"}},
    }
    out = _sanitize_strict(schema)
    for node in _walk(out):
        assert "title" not in node
        assert "default" not in node


def test_preserves_nullable_enum_anyof_and_recurses_items():
    schema = {
        "type": "object",
        "properties": {
            "kind": {"anyOf": [{"enum": ["x", "y"], "type": "string"}, {"type": "null"}]},
            "tags": {
                "type": "array",
                "items": {"type": "object", "properties": {"t": {"type": "string"}}},
            },
        },
    }
    out = _sanitize_strict(schema)
    kind = out["properties"]["kind"]
    assert kind["anyOf"][0]["enum"] == ["x", "y"]
    assert kind["anyOf"][1]["type"] == "null"
    item = out["properties"]["tags"]["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["t"]
