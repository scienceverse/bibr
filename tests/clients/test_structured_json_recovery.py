"""Bounded recovery of one complete outer JSON object (ported from PR #8).

Synthetic malformed envelopes must not select a valid nested value, and the
recovery never completes truncated JSON.
"""

import json

import pytest

from bibr.clients.structured_json import StructuredResponseError, recover_structured_object
from bibr.schemas import TitleKeywordsLLM

MALFORMED = (
    r'{"title":"A printed study","abstract":"Measured \(7 \\pm 2\) units.","keywords":["growth"]}'
)


def test_outer_object_recovers_literal_math_backslashes_without_changing_values():
    recovered = recover_structured_object(f"```json\n{MALFORMED}\n```", TitleKeywordsLLM)

    assert recovered.value.title == "A printed study"
    assert recovered.value.abstract == r"Measured \(7 \pm 2\) units."
    assert recovered.value.keywords == ["growth"]
    assert recovered.repaired_backslashes == 2


def test_valid_escapes_and_unicode_are_not_reinterpreted():
    text = 'A quote " here; slash \\; tab\t; newline\n; Greek α.'
    recovered = recover_structured_object(json.dumps({"title": text}), TitleKeywordsLLM)

    assert recovered.value.title == text
    assert recovered.repaired_backslashes == 0


@pytest.mark.parametrize("prose", ["before", "after", "both"])
def test_one_explicit_fence_with_only_explanatory_prose_preserves_whole_object(prose):
    raw = '```json\n{"title":"A printed study","keywords":["growth"]}\n```'
    if prose in {"before", "both"}:
        raw = "The following values are printed in the document.\n\n" + raw
    if prose in {"after", "both"}:
        raw += "\n\nNo additional values were supplied."

    recovered = recover_structured_object(raw, TitleKeywordsLLM)

    assert recovered.value.title == "A printed study"
    assert recovered.value.keywords == ["growth"]
    assert recovered.repaired_backslashes == 0


@pytest.mark.parametrize(
    "outside",
    [
        '{"title":"Competing value"}',
        '["competing array"]',
        '{"unfinished":',
        '```json\n{"title":"Competing fence"}\n```',
        "```text\nExplanation\n```",
        "~~~json\n{}\n~~~",
    ],
)
@pytest.mark.parametrize("side", ["before", "after"])
def test_prose_fence_rejects_other_structured_material_anywhere(outside, side):
    fence = '```json\n{"title":"A printed study","keywords":["growth"]}\n```'
    raw = outside + "\n" + fence if side == "before" else fence + "\n" + outside

    with pytest.raises(StructuredResponseError):
        recover_structured_object(raw, TitleKeywordsLLM)


@pytest.mark.parametrize(
    "raw",
    [
        'Explanation\n```\n{"title":"Unnamed fence"}\n```',
        'Explanation\n```json\n{"title":"Truncated"\n```',
        'Explanation\n```json\n{"payload":{"title":"Nested"}}\n```',
        'Explanation\n```json\n["keywords only"]\n```',
        'Explanation\n```json\n{"title":"First", "title":"Second"}\n```',
    ],
)
def test_prose_fence_does_not_relax_outer_object_or_schema_guards(raw):
    with pytest.raises(StructuredResponseError):
        recover_structured_object(raw, TitleKeywordsLLM)


@pytest.mark.parametrize(
    "content",
    [
        '["growth"]',
        '{"abstract":"unfinished, "keywords":["growth"]}',
        '{"abstract":"unfinished',
        '{"abstract":"bad \\uXYZ1", "keywords":["growth"]}',
        '{"title":"first"}{"title":"second"}',
        'Note: {"title":"A printed study"}',
        '```json\n{"title":"first"}\n```\n```json\n{"title":"second"}\n```',
        '{"title":"first", "title":"second"}',
        '{"payload":{"title":"nested"}}',
        '{"TitleKeywordsLLM":["growth"]}',
        '{"TitleKeywordsLLM":{"payload":{"title":"nested"}}}',
        '{"title":NaN}',
        '{"title":{"not":"a string"}}',
    ],
)
def test_ambiguous_invalid_or_nested_content_is_never_silently_accepted(content):
    with pytest.raises(StructuredResponseError):
        recover_structured_object(content, TitleKeywordsLLM)


@pytest.mark.parametrize("fault", ["size", "depth", "repair-count"])
def test_recovery_is_bounded(fault):
    if fault == "size":
        content = json.dumps({"title": "x" * 262_144})
    elif fault == "depth":
        content = '{"keywords":' + "[" * 65 + "0" + "]" * 65 + "}"
    else:
        content = '{"title":"' + r"\(" * 129 + '"}'
    with pytest.raises(StructuredResponseError):
        recover_structured_object(content, TitleKeywordsLLM)
