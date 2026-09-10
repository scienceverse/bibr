"""Real replay failure shapes must not become successful, empty metadata."""

import json

import pytest
from pydantic import ValidationError

from bibr.schemas import CoreMetadataLLM, PaperClassificationLLM, TitleKeywordsLLM


@pytest.mark.parametrize("model", [TitleKeywordsLLM, CoreMetadataLLM, PaperClassificationLLM])
@pytest.mark.parametrize("wrapped", [False, True])
def test_generated_schema_is_rejected_as_response(model, wrapped):
    payload = model.model_json_schema()
    if wrapped:
        payload = {model.__name__: payload}
    with pytest.raises(ValidationError, match="Schema echo"):
        model.model_validate_json(json.dumps(payload))


def test_values_nested_in_schema_properties_are_not_silently_lost():
    with pytest.raises(ValidationError, match="Schema echo"):
        TitleKeywordsLLM.model_validate(
            {
                "type": "object",
                "title": "TitleKeywordsLLM",
                "properties": {"title": "A real title", "abstract": "Actual prose", "keywords": []},
            }
        )


def test_truncated_schema_without_type_is_rejected():
    with pytest.raises(ValidationError, match="Schema echo"):
        TitleKeywordsLLM.model_validate({"properties": {"abstract": {"type": "string"}}})


@pytest.mark.parametrize("wrapped", [False, True])
def test_real_metadata_and_explicit_absence_keep_existing_contract(wrapped):
    payload = {"title": "Properties of JSON Schema", "abstract": None, "keywords": []}
    if wrapped:
        payload = {"TitleKeywordsLLM": payload}
    result = TitleKeywordsLLM.model_validate(payload)
    assert result.title == "Properties of JSON Schema"
    assert result.abstract is None
    assert result.keywords == []


def test_optional_empty_metadata_is_still_constructible():
    assert TitleKeywordsLLM().title is None
