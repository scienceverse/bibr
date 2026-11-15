"""LLM response schemas: a list field answered with one string."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bibr.schemas import (
    CoreMetadataLLM,
    FundingEntryLLM,
    TitleKeywordsLLM,
    TitleKeywordsPointerLLM,
)


@pytest.mark.parametrize("model", [TitleKeywordsLLM, TitleKeywordsPointerLLM, CoreMetadataLLM])
def test_a_keywords_string_is_one_keyword_not_one_per_character(model):
    # A model whose JSON schema is advisory (small local models) answers a
    # list field with a string; iterating it made every character a keyword.
    parsed = model.model_validate({"title": "A Paper", "authors": [], "keywords": "working memory"})
    assert parsed.keywords == ["working memory"]


@pytest.mark.parametrize("model", [TitleKeywordsLLM, TitleKeywordsPointerLLM, CoreMetadataLLM])
def test_a_placeholder_keywords_string_is_no_keyword(model):
    assert (
        model.model_validate({"title": "A Paper", "authors": [], "keywords": "string"}).keywords
        == []
    )


def test_an_award_ids_string_is_one_award_id():
    parsed = FundingEntryLLM.model_validate({"funder": "NSF", "award_ids": "R01-MH12345"})
    assert parsed.award_ids == ["R01-MH12345"]


def test_an_award_ids_number_is_a_validation_error_not_a_type_error():
    # A ValidationError is what Instructor re-asks on; a TypeError failed the call.
    with pytest.raises(ValidationError):
        FundingEntryLLM.model_validate({"funder": "NSF", "award_ids": 834861})
