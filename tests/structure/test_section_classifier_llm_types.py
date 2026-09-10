"""The LLM classifier's advertised section types come from one source of truth.

The pydantic field description and the prompt body previously drifted apart
(the field description omitted "title" while the prompt allowed it).
"""

from bibr.structure import section_classifier as sc


def test_llm_type_descriptions_are_valid_canonical_values():
    values = [v for v, _ in sc._LLM_SECTION_TYPE_DESCRIPTIONS]
    assert set(values) <= sc._VALID_SECTION_VALUES
    assert len(values) == len(set(values))
    # The prompt has always allowed returning "title"; the field description
    # must advertise it too.
    assert "title" in values
    assert "unknown" in values


def test_llm_field_description_and_prompt_block_cover_same_types():
    values = [v for v, _ in sc._LLM_SECTION_TYPE_DESCRIPTIONS]
    field_desc = sc._llm_section_type_field_description()
    prompt_block = sc._llm_section_type_prompt_block()
    for value in values:
        assert value in field_desc
        assert f"- {value}:" in prompt_block
