"""NuExtract-native semantic schema contract regression tests."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from typing import Annotated, Any, ClassVar

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from pydantic import AliasChoices, AliasPath, BaseModel, Field

from bibr.clients import nuextract
from bibr.clients.nuextract import template_for_model
from bibr.schemas import (
    AffiliationLLM,
    AuthorContributionLLM,
    AuthorLLM,
    AuthorsLLM,
    CitationMatch,
    CitationResolutionResult,
    CoreMetadataLLM,
    EquationComponentLLM,
    EquationExtractionResult,
    FrontMatterResult,
    FrontMatterSegment,
    FundingEntryLLM,
    PaperClassificationLLM,
    PaperReferenceList,
    PaperReferenceLLM,
    PaperTypeLabel,
    RefAnchors,
    ResearchIntegrityLLM,
    TitleKeywordsLLM,
)
from bibr.structure.section_classifier import (
    _LLM_SECTION_TYPE_DESCRIPTIONS,
    SectionClassification,
    SectionClassificationResult,
)

PAPER_TYPES = [
    "empirical",
    "review",
    "meta-analysis",
    "case-study",
    "commentary",
    "corrigendum",
    "erratum",
    "retraction",
]
OECD_DOMAINS = [
    "Natural Sciences",
    "Engineering and Technology",
    "Medical and Health Sciences",
    "Agricultural and Veterinary Sciences",
    "Social Sciences",
    "Humanities and the Arts",
]
OECD_SUBDOMAINS = [
    "Mathematics",
    "Computer and Information Sciences",
    "Physical Sciences",
    "Chemical Sciences",
    "Earth and Related Environmental Sciences",
    "Biological Sciences",
    "Civil Engineering",
    "Electrical Engineering, Electronic Engineering, Information Engineering",
    "Mechanical Engineering",
    "Chemical Engineering",
    "Materials Engineering",
    "Medical Engineering",
    "Environmental Engineering",
    "Environmental Biotechnology",
    "Industrial Biotechnology",
    "Nano-technology",
    "Basic Medicine",
    "Clinical Medicine",
    "Health Sciences",
    "Medical Biotechnology",
    "Agriculture, Forestry, and Fisheries",
    "Animal and Dairy Science",
    "Veterinary Science",
    "Agricultural Biotechnology",
    "Psychology and Cognitive Sciences",
    "Economics and Business",
    "Education",
    "Sociology",
    "Law",
    "Political Science",
    "Social and Economic Geography",
    "Media and Communications",
    "History and Archaeology",
    "Languages and Literature",
    "Philosophy, Ethics and Religion",
    "Arts (arts, history of arts, performing arts, music)",
]
BIB_TYPES = [
    "journal_article",
    "book",
    "book_chapter",
    "dataset",
    "software",
    "preprint",
    "conference_paper",
    "report",
    "thesis",
    "other",
]

TITLE_KEYWORDS_LEAVES = {
    "title": "verbatim-string",
    "abstract": "verbatim-string",
    "keywords[]": "verbatim-string",
    "journal": "verbatim-string",
    "volume": "string",
    "issue": "string",
    "first_page": "string",
    "last_page": "string",
    "issn": "verbatim-string",
    "publisher": "verbatim-string",
    "published": "date",
    "license": "string",
}
AUTHORS_LEAVES = {
    "authors[].given": "string",
    "authors[].family": "string",
    "authors[].affiliation": "string",
    "authors[].email": "email-address",
    "authors[].corresponding": "boolean",
    "authors[].orcid": "verbatim-string",
}
CLASSIFICATION_LEAVES = {
    "oecd_domain": OECD_DOMAINS,
    "oecd_subdomain": OECD_SUBDOMAINS,
    "paper_type": PAPER_TYPES,
}
REFERENCE_LEAVES = {
    "references[].title": "verbatim-string",
    "references[].first_page": "string",
    "references[].volume": "string",
    "references[].authors": "verbatim-string",
    "references[].year": "integer",
    "references[].container": "verbatim-string",
    "references[].year_suffix": "string",
    "references[].doi": "verbatim-string",
    "references[].bib_type": BIB_TYPES,
    "references[].last_page": "string",
    "references[].issue": "string",
    "references[].editors": "verbatim-string",
    "references[].publisher": "verbatim-string",
    "references[].url": "url",
    "references[].date": "date",
    "references[].edition": "verbatim-string",
    "references[].version": "verbatim-string",
    "references[].is_in_press": "boolean",
    "references[].index": "integer",
}

EXPECTED = {
    TitleKeywordsLLM: TITLE_KEYWORDS_LEAVES,
    AuthorsLLM: AUTHORS_LEAVES,
    PaperClassificationLLM: CLASSIFICATION_LEAVES,
    CoreMetadataLLM: {
        **TITLE_KEYWORDS_LEAVES,
        **AUTHORS_LEAVES,
        **CLASSIFICATION_LEAVES,
    },
    PaperReferenceList: REFERENCE_LEAVES,
    RefAnchors: {"anchors[]": "verbatim-string"},
    CitationResolutionResult: {
        "matches[].text_id": "integer",
        "matches[].citation_text": "verbatim-string",
        "matches[].bib_id": "integer",
    },
    ResearchIntegrityLLM: {
        "funding[].funder": "verbatim-string",
        "funding[].award_ids[]": "verbatim-string",
        "contributions[].author": "verbatim-string",
        "contributions[].roles[]": "verbatim-string",
        "affiliations[].index": "integer",
        "affiliations[].institution": "verbatim-string",
        "affiliations[].department": "verbatim-string",
        "affiliations[].city": "verbatim-string",
        "affiliations[].country": "verbatim-string",
    },
    EquationExtractionResult: {
        "equations[].sentence_index": "integer",
        "equations[].lhs": "verbatim-string",
        "equations[].df": "verbatim-string",
        "equations[].comp": "verbatim-string",
        "equations[].rhs": "verbatim-string",
    },
    PaperTypeLabel: {
        "paper_type": PAPER_TYPES,
        "confidence": "number",
    },
    FrontMatterResult: {
        "segments[].first_text_id": "integer",
        "segments[].section_type": ["abstract", "intro", "keywords", "metadata"],
    },
    SectionClassificationResult: {
        "classifications[].header": "verbatim-string",
        "classifications[].section_type": [
            value for value, _description in _LLM_SECTION_TYPE_DESCRIPTIONS
        ],
    },
}

TITLE_NULLABLE = {
    "title",
    "abstract",
    "journal",
    "volume",
    "issue",
    "first_page",
    "last_page",
    "issn",
    "publisher",
    "published",
    "license",
}
AUTHOR_NULLABLE = {
    "authors[].given",
    "authors[].family",
    "authors[].affiliation",
    "authors[].email",
    "authors[].orcid",
}
CLASSIFICATION_NULLABLE = {"oecd_domain", "oecd_subdomain", "paper_type"}
REFERENCE_NULLABLE = {
    path
    for path in REFERENCE_LEAVES
    if path not in {"references[].index", "references[].is_in_press"}
}
EXPECTED_NULLABLE = {
    TitleKeywordsLLM: TITLE_NULLABLE,
    AuthorsLLM: AUTHOR_NULLABLE,
    PaperClassificationLLM: CLASSIFICATION_NULLABLE,
    CoreMetadataLLM: TITLE_NULLABLE | AUTHOR_NULLABLE | CLASSIFICATION_NULLABLE,
    PaperReferenceList: REFERENCE_NULLABLE,
    RefAnchors: set(),
    CitationResolutionResult: {"matches[].bib_id"},
    ResearchIntegrityLLM: {
        "affiliations[].institution",
        "affiliations[].department",
        "affiliations[].city",
        "affiliations[].country",
    },
    EquationExtractionResult: {
        "equations[].sentence_index",
        "equations[].lhs",
        "equations[].df",
        "equations[].comp",
        "equations[].rhs",
    },
    PaperTypeLabel: {"paper_type", "confidence"},
    FrontMatterResult: set(),
    SectionClassificationResult: set(),
}

POLICY_MODELS = {
    TitleKeywordsLLM,
    AuthorsLLM,
    AuthorLLM,
    PaperClassificationLLM,
    CoreMetadataLLM,
    PaperReferenceList,
    PaperReferenceLLM,
    RefAnchors,
    CitationResolutionResult,
    CitationMatch,
    ResearchIntegrityLLM,
    FundingEntryLLM,
    AuthorContributionLLM,
    AffiliationLLM,
    EquationExtractionResult,
    EquationComponentLLM,
    PaperTypeLabel,
    FrontMatterResult,
    FrontMatterSegment,
    SectionClassification,
    SectionClassificationResult,
}


def flatten_template_leaves(value: Any, path: str = "") -> dict[str, str | list[str]]:
    if isinstance(value, dict):
        assert value, f"empty object template branch at {path or '<root>'}"
        flattened: dict[str, str | list[str]] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            flattened.update(flatten_template_leaves(child, child_path))
        return flattened
    if isinstance(value, list):
        assert value, f"empty list template branch at {path or '<root>'}"
        if len(value) > 1 and all(isinstance(item, str) for item in value):
            return {path: value}
        assert len(value) == 1
        return flatten_template_leaves(value[0], f"{path}[]")
    assert isinstance(value, str)
    return {path: value}


def native_contract_for_model(model: type[BaseModel]):
    function = getattr(nuextract, "native_contract_for_model", None)
    assert function is not None, "native_contract_for_model is not implemented"
    return function(model)


def policy(**kwargs):
    policy_type = getattr(nuextract, "NuExtractSchemaPolicy", None)
    assert policy_type is not None, "NuExtractSchemaPolicy is not implemented"
    return policy_type(**kwargs)


@pytest.mark.parametrize(
    ("model", "expected"),
    EXPECTED.items(),
    ids=[model.__name__ for model in EXPECTED],
)
def test_native_contract_has_exact_semantic_leaves(model, expected):
    assert flatten_template_leaves(template_for_model(model)) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    EXPECTED_NULLABLE.items(),
    ids=[model.__name__ for model in EXPECTED_NULLABLE],
)
def test_native_contract_has_exact_nullable_paths(model, expected):
    assert native_contract_for_model(model).nullable_paths == frozenset(expected)


def test_registered_prompt_inventory_is_exact():
    from bibr.clients.prompts import PROMPTS

    assert [(name, spec.response_model) for name, spec in PROMPTS.items()] == [
        ("title_keywords", TitleKeywordsLLM),
        ("authors", AuthorsLLM),
        ("classification", PaperClassificationLLM),
        ("core_metadata", CoreMetadataLLM),
        ("references_parse", PaperReferenceList),
        ("references_parse_chunk", PaperReferenceList),
        ("references_segment", RefAnchors),
        ("citation_resolution", CitationResolutionResult),
        ("research_integrity", ResearchIntegrityLLM),
        ("equations", EquationExtractionResult),
        ("paper_type_label", PaperTypeLabel),
    ]
    assert set(EXPECTED) == {spec.response_model for spec in PROMPTS.values()} | {
        FrontMatterResult,
        SectionClassificationResult,
    }


def test_every_root_and_nested_response_model_has_an_explicit_policy():
    assert all("nuextract_policy" in model.__dict__ for model in POLICY_MODELS)


def test_native_contract_excludes_exactly_downstream_fields():
    absolute_exclusions = {f"authors[].{field}" for field in AuthorLLM.nuextract_policy.exclude} | {
        f"references[].{field}" for field in PaperReferenceLLM.nuextract_policy.exclude
    }
    assert absolute_exclusions == {
        "references[].bib_id",
        "references[].text_id",
        "references[].match",
    }
    # Author runtime fields are already absent from the shared generation schema.
    assert not {"author_id", "role"} & AuthorLLM.model_json_schema()["properties"].keys()
    assert all(
        not model.nuextract_policy.exclude
        for model in POLICY_MODELS - {AuthorLLM, PaperReferenceLLM}
    )


def test_native_projection_does_not_change_shared_pydantic_schema():
    before = copy.deepcopy(CoreMetadataLLM.model_json_schema())
    first = native_contract_for_model(CoreMetadataLLM)
    second = native_contract_for_model(CoreMetadataLLM)

    assert CoreMetadataLLM.model_json_schema() == before
    assert first == second
    assert first.template is not second.template
    assert first.wire_schema is not second.wire_schema
    assert "nuextract_policy" not in json.dumps(before)


def test_mutating_returned_contract_does_not_change_later_contract():
    first = native_contract_for_model(AuthorsLLM)
    first.template["authors"][0]["given"] = "changed"
    first.wire_schema["properties"]["authors"]["items"]["properties"]["given"] = {}

    second = native_contract_for_model(AuthorsLLM)
    assert second.template["authors"][0]["given"] == "string"
    assert second.wire_schema["properties"]["authors"]["items"]["properties"]["given"]


def test_importing_schemas_does_not_import_numind():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bibr.schemas; "
            "assert not any(name == 'numind' or name.startswith('numind.') for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("model", "expected"),
    EXPECTED.items(),
    ids=[model.__name__ for model in EXPECTED],
)
def test_wire_schema_is_closed_and_requires_every_projected_property(model, expected):
    contract = native_contract_for_model(model)
    nullable_paths = set()

    def visit(node, path=""):
        non_null = [branch for branch in node.get("anyOf", []) if branch.get("type") != "null"]
        null_count = sum(branch.get("type") == "null" for branch in node.get("anyOf", []))
        if len(non_null) == 1 and null_count == 1:
            nullable_paths.add(path)
            visit(non_null[0], path)
            return
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node.get("properties", {}))
            for name, child in node.get("properties", {}).items():
                child_path = f"{path}.{name}" if path else name
                visit(child, child_path)
        elif node.get("type") == "array":
            visit(node["items"], f"{path}[]")

    visit(contract.wire_schema)
    assert nullable_paths == contract.nullable_paths
    assert flatten_template_leaves(contract.template) == expected


def test_wire_schema_keeps_nullable_fields_required_but_accepts_null():
    schema = native_contract_for_model(TitleKeywordsLLM).wire_schema
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    instance = {
        "title": None,
        "abstract": None,
        "keywords": [],
        "journal": None,
        "volume": None,
        "issue": None,
        "first_page": None,
        "last_page": None,
        "issn": None,
        "publisher": None,
        "published": None,
        "license": None,
    }

    validator.validate(instance)
    missing = dict(instance)
    missing.pop("abstract")
    with pytest.raises(ValidationError, match="required"):
        validator.validate(missing)


def test_wire_schema_accepts_bare_publication_year_while_template_uses_date():
    contract = native_contract_for_model(TitleKeywordsLLM)
    instance = {
        "title": None,
        "abstract": None,
        "keywords": [],
        "journal": None,
        "volume": None,
        "issue": None,
        "first_page": None,
        "last_page": None,
        "issn": None,
        "publisher": None,
        "published": "2024",
        "license": None,
    }

    Draft202012Validator(contract.wire_schema, format_checker=FormatChecker()).validate(instance)
    assert contract.template["published"] == "date"


def test_nested_policies_are_discovered_through_annotated_nullable_array_items():
    class Leaf(BaseModel):
        value: str | None
        nuextract_policy: ClassVar = policy(semantics={"value": "verbatim-string"})

    class Root(BaseModel):
        leaves: Annotated[list[Leaf | None], Field(description="nested")]
        nuextract_policy: ClassVar = policy()

    contract = native_contract_for_model(Root)
    assert contract.template == {"leaves": [{"value": "verbatim-string"}]}
    assert contract.nullable_paths == frozenset({"leaves[]", "leaves[].value"})


def test_nested_policy_paths_follow_pydantic_property_aliases():
    class Leaf(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy(semantics={"value": "verbatim-string"})

    class Root(BaseModel):
        leaves: list[Leaf] = Field(alias="wire_leaves")
        nuextract_policy: ClassVar = policy()

    contract = native_contract_for_model(Root)
    assert contract.template == {"wire_leaves": [{"value": "verbatim-string"}]}


def test_nested_policy_paths_follow_string_validation_aliases():
    class Leaf(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy(semantics={"value": "verbatim-string"})

    class Root(BaseModel):
        leaves: list[Leaf] = Field(validation_alias="wire_leaves")
        nuextract_policy: ClassVar = policy()

    contract = native_contract_for_model(Root)
    assert contract.template == {"wire_leaves": [{"value": "verbatim-string"}]}


def test_nested_policy_paths_follow_first_direct_alias_choice():
    class Leaf(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy(semantics={"value": "verbatim-string"})

    class Root(BaseModel):
        leaves: list[Leaf] = Field(
            validation_alias=AliasChoices(
                AliasPath("payload", "wire_leaves"),
                "wire_leaves",
                "legacy_leaves",
            )
        )
        nuextract_policy: ClassVar = policy()

    contract = native_contract_for_model(Root)
    assert contract.template == {"wire_leaves": [{"value": "verbatim-string"}]}


@pytest.mark.parametrize(
    "validation_alias",
    [
        AliasPath("payload", "wire_leaves"),
        AliasChoices(
            AliasPath("payload", "wire_leaves"),
            AliasPath("legacy", "leaves"),
        ),
    ],
)
def test_nested_policy_paths_use_field_name_when_validation_aliases_are_only_paths(
    validation_alias,
):
    class Leaf(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy(semantics={"value": "verbatim-string"})

    class Root(BaseModel):
        leaves: list[Leaf] = Field(validation_alias=validation_alias)
        nuextract_policy: ClassVar = policy()

    contract = native_contract_for_model(Root)
    assert contract.template == {"leaves": [{"value": "verbatim-string"}]}


def test_serialization_alias_does_not_change_default_validation_schema_path():
    class Leaf(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy(semantics={"value": "verbatim-string"})

    class Root(BaseModel):
        leaves: list[Leaf] = Field(serialization_alias="serialized_leaves")
        nuextract_policy: ClassVar = policy()

    contract = native_contract_for_model(Root)
    assert contract.template == {"leaves": [{"value": "verbatim-string"}]}


def test_nullable_type_shorthand_is_retained_only_on_wire(monkeypatch):
    class Synthetic(BaseModel):
        value: str | None
        nuextract_policy: ClassVar = policy(semantics={"value": "verbatim-string"})

    schema = Synthetic.model_json_schema()
    schema["properties"]["value"] = {"type": ["string", "null"]}
    monkeypatch.setattr(Synthetic, "model_json_schema", lambda: copy.deepcopy(schema))

    contract = native_contract_for_model(Synthetic)
    assert contract.template == {"value": "verbatim-string"}
    assert contract.wire_schema["properties"]["value"]["type"] == ["string", "null"]
    assert contract.nullable_paths == frozenset({"value"})


def test_nullable_object_shorthand_stays_nullable_when_wire_object_is_closed(monkeypatch):
    class Child(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy(semantics={"value": "string"})

    class Root(BaseModel):
        child: Child | None
        nuextract_policy: ClassVar = policy()

    schema = Root.model_json_schema()
    schema["properties"]["child"] = {
        "type": ["object", "null"],
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    monkeypatch.setattr(Root, "model_json_schema", lambda: copy.deepcopy(schema))

    contract = native_contract_for_model(Root)
    child_schema = contract.wire_schema["properties"]["child"]
    assert child_schema["type"] == ["object", "null"]
    assert child_schema["required"] == ["value"]
    assert child_schema["additionalProperties"] is False
    assert contract.nullable_paths == frozenset({"child"})
    validator = Draft202012Validator(contract.wire_schema)
    validator.validate({"child": None})
    validator.validate({"child": {"value": "present"}})
    with pytest.raises(ValidationError, match="Additional properties"):
        validator.validate({"child": {"value": "present", "extra": "rejected"}})


def test_nullable_array_shorthand_recursively_closes_wire_items(monkeypatch):
    class Child(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy()

    class Root(BaseModel):
        children: list[Child] | None
        nuextract_policy: ClassVar = policy()

    schema = Root.model_json_schema()
    schema["properties"]["children"] = {
        "type": ["array", "null"],
        "items": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    monkeypatch.setattr(Root, "model_json_schema", lambda: copy.deepcopy(schema))

    contract = native_contract_for_model(Root)
    children_schema = contract.wire_schema["properties"]["children"]
    item_schema = children_schema["items"]
    assert children_schema["type"] == ["array", "null"]
    assert item_schema["required"] == ["value"]
    assert item_schema["additionalProperties"] is False
    assert contract.nullable_paths == frozenset({"children"})
    validator = Draft202012Validator(contract.wire_schema)
    validator.validate({"children": None})
    validator.validate({"children": [{"value": "present"}]})
    with pytest.raises(ValidationError, match="Additional properties"):
        validator.validate({"children": [{"value": "present", "extra": "rejected"}]})


def test_cross_level_semantic_and_choice_conflict_fails_closed():
    class Leaf(BaseModel):
        value: str
        nuextract_policy: ClassVar = policy(choices={"value": ("one", "two")})

    class Root(BaseModel):
        leaf: Leaf
        nuextract_policy: ClassVar = policy(semantics={"leaf.value": "string"})

    with pytest.raises(ValueError, match="both semantics and choices"):
        native_contract_for_model(Root)


@pytest.mark.parametrize(
    ("policy_value", "message"),
    [
        ({"semantics": {"missing": "string"}}, "missing"),
        ({"semantics": {"nested": "verbatim-string"}}, "object"),
        (
            {
                "semantics": {"value": "string"},
                "choices": {"value": ("one", "two")},
            },
            "both semantics and choices",
        ),
    ],
)
def test_invalid_policy_fails_closed(policy_value, message):
    class InvalidPolicy(BaseModel):
        value: str
        nested: dict[str, str] = {}
        nuextract_policy: ClassVar = policy(**policy_value)

    with pytest.raises(ValueError, match=message):
        native_contract_for_model(InvalidPolicy)


def test_unsupported_union_fails_closed():
    class UnsupportedUnion(BaseModel):
        value: str | int
        nuextract_policy: ClassVar = policy()

    with pytest.raises(ValueError, match="unsupported|dropped"):
        native_contract_for_model(UnsupportedUnion)


def test_converter_reported_drop_fails_closed(monkeypatch):
    class SupportedAndUnsupported(BaseModel):
        good: str
        bad: str
        nuextract_policy: ClassVar = policy(semantics={"good": "string", "bad": "string"})

    def dropping_converter(schema, *, omit_unsupported_branches):
        assert omit_unsupported_branches is True
        return {"good": "string"}, [{"path": ["properties", "bad"], "error": "synthetic"}]

    monkeypatch.setattr(
        "numind.nuextract_utils.convert_json_schema_to_nuextract_template",
        dropping_converter,
    )
    with pytest.raises(ValueError, match="dropped"):
        native_contract_for_model(SupportedAndUnsupported)


def test_silent_empty_unconstrained_object_fails_closed():
    class EmptyObject(BaseModel):
        payload: dict
        nuextract_policy: ClassVar = policy()

    with pytest.raises(ValueError, match="empty|unconstrained"):
        native_contract_for_model(EmptyObject)


def test_recursive_model_fails_closed():
    class Recursive(BaseModel):
        children: list[Recursive] = []
        nuextract_policy: ClassVar = policy()

    with pytest.raises(ValueError, match="yclic"):
        native_contract_for_model(Recursive)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_silent_converter_path_mismatch_fails_closed(monkeypatch, mutation):
    class TwoFields(BaseModel):
        first: str
        second: str
        nuextract_policy: ClassVar = policy(semantics={"first": "string", "second": "string"})

    def mismatching_converter(schema, *, omit_unsupported_branches):
        template = {"first": "string", "second": "string"}
        if mutation == "missing":
            template.pop("second")
        else:
            template["third"] = "string"
        return template, []

    monkeypatch.setattr(
        "numind.nuextract_utils.convert_json_schema_to_nuextract_template",
        mismatching_converter,
    )
    with pytest.raises(ValueError, match="mismatch"):
        native_contract_for_model(TwoFields)
