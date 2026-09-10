from pathlib import Path

from bibr.clients.prompts import (
    _CLASSIFICATION_PROMPT,
    _PAPER_TYPE_DISAMBIGUATION,
    PROMPTS,
    prompt_text,
)
from bibr.schemas import PaperTypeLabel

_GOLDEN = Path(__file__).parent / "data" / "classification_prompt.golden.txt"


def test_classification_prompt_unchanged_byte_for_byte():
    # The taxonomy extraction must NOT alter the production prompt.
    assert _GOLDEN.read_text(encoding="utf-8") == _CLASSIFICATION_PROMPT


def test_shared_taxonomy_is_substring_of_both_prompts():
    assert _PAPER_TYPE_DISAMBIGUATION.strip()
    assert _PAPER_TYPE_DISAMBIGUATION in _CLASSIFICATION_PROMPT
    label_text = prompt_text(PROMPTS["paper_type_label"].build_user(title="T", abstract="A"))
    assert _PAPER_TYPE_DISAMBIGUATION in label_text


def test_label_spec_shape():
    spec = PROMPTS["paper_type_label"]
    assert spec.response_model is PaperTypeLabel
    parts = spec.build_user(title="My Title", abstract="My abstract body")
    text = prompt_text(parts)
    assert "My Title" in text
    assert "My abstract body" in text
