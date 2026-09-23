"""Example files any producer or reader of the export format can test against.

``tests/fixtures/schema_conformance/`` holds three sets:

- ``valid/``: exports the strict schema (what bibr writes) must accept;
- ``invalid/``: each breaks exactly one rule, and both schemas must reject it;
- ``reader_valid/``: exports from a later 12.x writer, which the strict schema
  rejects and the reader schema must accept.

Each file is checked against the published JSON Schema documents and the
Python models, so the two stay in agreement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bibr.export import PaperExport, PaperExportReader
from bibr.export.schema_artifact import schema_file_name

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "schema_conformance"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schema"


_STRICT_ONLY = {
    "unknown_root_key",
    "root_validation_block",
    "removed_author_affiliation",
    "section_type_off_vocabulary",
    # A 12.0 writer always writes every column; the reader leaves keys optional
    # so it can read a later minor's output, which may add columns.
    "dropped_column",
}

# Rules only the published strict schema states. "Required" there means the
# key is always *present*; the Python models fill a missing key's default, so
# code can build a row without spelling out every null.
_SCHEMA_ONLY = {"dropped_column"}


def _files(kind: str) -> list[Path]:
    return sorted((ROOT / kind).glob("*.json"))


def _validator(*, reader: bool):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMA_DIR / schema_file_name(reader=reader)).read_text())
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize("path", _files("valid"), ids=lambda p: p.stem)
def test_valid_examples_pass_both_schemas(path):
    payload = json.loads(path.read_text())
    assert not list(_validator(reader=False).iter_errors(payload))
    assert not list(_validator(reader=True).iter_errors(payload))
    PaperExport.model_validate(payload)


@pytest.mark.parametrize("path", _files("invalid"), ids=lambda p: p.stem)
def test_invalid_examples_fail_both_schemas(path):
    payload = json.loads(path.read_text())
    assert list(_validator(reader=False).iter_errors(payload))
    if path.stem not in _SCHEMA_ONLY:
        with pytest.raises(ValidationError):
            PaperExport.model_validate(payload)
    # The reader tolerates unknown keys and enum values it does not know yet,
    # so those examples are failures for the strict schema only.
    if path.stem not in _STRICT_ONLY:
        assert list(_validator(reader=True).iter_errors(payload))
        with pytest.raises(ValidationError):
            PaperExportReader.model_validate(payload)


@pytest.mark.parametrize("path", _files("reader_valid"), ids=lambda p: p.stem)
def test_newer_minor_examples_pass_only_the_reader(path):
    payload = json.loads(path.read_text())
    assert list(_validator(reader=False).iter_errors(payload))
    assert not list(_validator(reader=True).iter_errors(payload))
    PaperExportReader.model_validate(payload)


def test_the_full_example_has_every_root_key_in_order():
    """``valid/full.json`` is a real export of the shared demo paper."""
    example = json.loads((ROOT / "valid" / "full.json").read_text())
    assert list(example) == list(PaperExport.model_fields)
    # carries every v12 feature
    assert example["xref"][0]["start"] is not None
    assert example["extraction"]["pages"] and example["affiliation_match"]
    assert example["funding_match"] and example["metadata_match"][0]["funder"]
    assert example["extraction"]["warnings"][0]["code"] == "STATEMENT_LEXICAL_FALLBACK"
    assert example["footnote"] and example["figure"][0]["text_id"] is not None


def _regenerate_full_example() -> None:
    """Rewrite ``valid/full.json`` from the shared demo paper (run this module)."""
    from bibr.export.json_export import _export_paper_payload
    from bibr.extract.statement_scan import lexical_fallback_warning
    from bibr.paper_contents import PaperFigurePart
    from tests.export.conftest import _demo_paper, as_parsed

    paper = _demo_paper(with_refs=True)
    as_parsed(paper)
    paper.contents.sentences[
        1
    ].text = "It replicated prior work [1], t(28) = 3.42, see Table 1 and data."
    paper.contents.sections[1].classification_score = 0.97
    paper.contents.sections[1].classification_source = "exact_alias"
    paper.contents.page_sizes = {1: (612.0, 792.0), 2: (612.0, 792.0)}
    paper.contents.figures[0].parts = [
        PaperFigurePart(page_number=2, bbox=(96.0, 410.0, 512.0, 688.0), image_b64=None)
    ]
    paper.processing_warnings = [lexical_fallback_warning("data_availability")]
    payload = _export_paper_payload(paper)
    payload["extraction"]["completed_at"] = "2026-09-22T12:00:00Z"
    (ROOT / "valid" / "full.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    _regenerate_full_example()
