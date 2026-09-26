"""Strict NuExtract native-completion boundary tests."""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, field_validator

import bibr.clients.nuextract as nuextract
from bibr.clients.nuextract import (
    NativeSchemaContract,
    NuExtractNativeBackend,
    native_contract_for_model,
)
from bibr.clients.nuextract_schema import native_wire_value_is_valid
from bibr.config import GlobalSettings
from bibr.schemas import AuthorsLLM, PaperClassificationLLM, TitleKeywordsLLM


class NumericResponse(BaseModel):
    count: int
    score: float


class RejectingResponse(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def reject_every_value(cls, value: str) -> str:
        raise ValueError("value rejected")


def _settings(**llm_overrides: Any) -> GlobalSettings:
    settings = GlobalSettings()
    for key, value in llm_overrides.items():
        setattr(settings.llm, key, value)
    return settings


def _invalid_output_type() -> type[Exception]:
    error_type = getattr(nuextract, "NuExtractInvalidOutput", None)
    assert isinstance(error_type, type), "NuExtractInvalidOutput is not implemented"
    assert issubclass(error_type, Exception)
    return error_type


def _parse(
    raw: str,
    *,
    contract: NativeSchemaContract,
    finish_reason: str | None = "stop",
    model: str = "numind/NuExtract3-FP8",
    input_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> BaseModel:
    parser = getattr(nuextract, "parse_native_completion", None)
    assert callable(parser), "parse_native_completion is not implemented"
    return parser(
        raw,
        finish_reason=finish_reason,
        contract=contract,
        model=model,
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


@pytest.fixture(scope="module")
def title_contract() -> NativeSchemaContract:
    return native_contract_for_model(TitleKeywordsLLM)


@pytest.fixture(scope="module")
def authors_contract() -> NativeSchemaContract:
    return native_contract_for_model(AuthorsLLM)


@pytest.fixture(scope="module")
def classification_contract() -> NativeSchemaContract:
    return native_contract_for_model(PaperClassificationLLM)


@pytest.fixture(scope="module")
def numeric_contract() -> NativeSchemaContract:
    return native_contract_for_model(NumericResponse)


def valid_title_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": "A real title",
        "abstract": None,
        "keywords": ["metadata extraction"],
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
    payload.update(overrides)
    return payload


def valid_authors_payload(**author_overrides: Any) -> dict[str, Any]:
    author: dict[str, Any] = {
        "given": "Ada",
        "family": "Lovelace",
        "affiliation": None,
        "email": None,
        "corresponding": False,
        "orcid": None,
    }
    author.update(author_overrides)
    return {"authors": [author]}


def valid_classification_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "oecd_domain": "Natural Sciences",
        "oecd_subdomain": "Computer and Information Sciences",
        "paper_type": "empirical",
    }
    payload.update(overrides)
    return payload


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _assert_invalid(
    raw: str,
    category: str,
    *,
    contract: NativeSchemaContract,
    finish_reason: str | None = "stop",
) -> Exception:
    with pytest.raises(_invalid_output_type()) as raised:
        _parse(
            raw,
            finish_reason=finish_reason,
            contract=contract,
        )
    error = raised.value
    assert error.category == category
    assert error.__cause__ is None
    assert error.__context__ is None
    return error


@pytest.mark.parametrize(
    ("raw", "finish_reason", "category"),
    [
        ("", "stop", "empty"),
        (" \n\t ", "stop", "empty"),
        ("prose only", "stop", "non_json"),
        ('prefix {"title": null}', "stop", "non_json"),
        ('{"title": null} suffix', "stop", "trailing_content"),
        ('{"title": null} {"title": null}', "stop", "trailing_content"),
        ('{"title":', "stop", "truncated"),
        ('{"title": "unfinished', "stop", "truncated"),
        ('{"title": nope', "stop", "non_json"),
        ('{"title": null,}', "stop", "non_json"),
        ("<think>unfinished", "stop", "non_json"),
        ('```json\n{"title": null}', "stop", "non_json"),
        ("[]", "stop", "non_object"),
    ],
)
def test_invalid_completion_categories(
    raw: str,
    finish_reason: str,
    category: str,
    title_contract: NativeSchemaContract,
):
    _assert_invalid(
        raw,
        category,
        contract=title_contract,
        finish_reason=finish_reason,
    )


@pytest.mark.parametrize(
    "raw",
    [
        "t",
        "tr",
        "tru",
        "f",
        "fa",
        "fal",
        "fals",
        "n",
        "nu",
        "nul",
        "-",
        "1.",
        "1e",
        "1e+",
        "1e-",
        "-2E",
        "-2E+",
        "-2E-",
        '{"value": t',
        '{"value": fals',
        '{"value": nul',
        '{"value": -',
        '{"value": 1.',
        '{"value": 1e',
        '{"value": 1e+',
    ],
)
def test_recognized_incomplete_literals_and_numbers_are_truncated(
    raw: str,
    title_contract: NativeSchemaContract,
):
    _assert_invalid(raw, "truncated", contract=title_contract)


@pytest.mark.parametrize(
    ("raw", "category"),
    [
        ("truth", "non_json"),
        ('{"value": truth', "non_json"),
        ('{"value": --', "non_json"),
        ("1efoo", "trailing_content"),
        ("1.2.3", "trailing_content"),
    ],
)
def test_malformed_tokens_are_not_misclassified_as_truncated(
    raw: str,
    category: str,
    title_contract: NativeSchemaContract,
):
    _assert_invalid(raw, category, contract=title_contract)


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "true",
        "42",
        "4.2",
        '"a string"',
    ],
)
def test_all_non_object_json_roots_are_rejected(
    raw: str,
    title_contract: NativeSchemaContract,
):
    _assert_invalid(raw, "non_object", contract=title_contract)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda raw: raw,
        lambda raw: f" \n{raw}\t ",
        lambda raw: f"```json\n{raw}\n```",
        lambda raw: f"```\n{raw}\n```",
        lambda raw: f"<think>private reasoning {{not output}}</think>\n{raw}",
        lambda raw: f" \n<think>\nprivate reasoning\n</think>\n```json\n{raw}\n```\n ",
    ],
)
def test_complete_supported_wrappers_parse_the_entire_valid_object(
    wrap,
    title_contract: NativeSchemaContract,
):
    result = _parse(wrap(_dump(valid_title_payload())), contract=title_contract)

    assert isinstance(result, TitleKeywordsLLM)
    assert result.title == "A real title"
    assert result.published == "2024"


@pytest.mark.parametrize(
    ("raw", "category"),
    [
        ("<think>one</think><think>two</think>{}", "non_json"),
        ("prefix <think>one</think>{}", "non_json"),
        ("{}<think>one</think>", "trailing_content"),
        ("<think>one</think>prefix {}", "non_json"),
        ("</think>{}", "non_json"),
        ("<think>one</THINK>{}", "non_json"),
        ("```json\n{}\n```\ntrailing", "non_json"),
        ("leading\n```json\n{}\n```", "non_json"),
        ("```json\n{}\n```\n```json\n{}\n```", "trailing_content"),
        ("```python\n{}\n```", "non_json"),
        ("```json {} ```", "non_json"),
        ("```\n{}\n", "non_json"),
        ("{}\n```", "trailing_content"),
    ],
)
def test_misplaced_repeated_or_malformed_wrappers_are_rejected(
    raw: str,
    category: str,
    title_contract: NativeSchemaContract,
):
    _assert_invalid(raw, category, contract=title_contract)


def test_finish_reason_length_wins_over_empty_valid_and_invalid_payloads(
    title_contract: NativeSchemaContract,
):
    for raw in ("", _dump(valid_title_payload()), "not JSON"):
        _assert_invalid(
            raw,
            "truncated",
            contract=title_contract,
            finish_reason="length",
        )


def test_extra_root_keys_are_rejected(title_contract: NativeSchemaContract):
    _assert_invalid(
        _dump(valid_title_payload(unrequested="secret")),
        "schema_invalid",
        contract=title_contract,
    )


def test_missing_root_keys_are_rejected(title_contract: NativeSchemaContract):
    payload = valid_title_payload()
    del payload["license"]

    _assert_invalid(_dump(payload), "schema_invalid", contract=title_contract)


@pytest.mark.parametrize(
    "overrides",
    [
        {"keywords": "not-a-list"},
        {"title": 12},
        {"keywords": [12]},
        {"published": []},
        {"license": {}},
    ],
)
def test_wrong_root_container_or_scalar_types_are_rejected(
    overrides: dict[str, Any],
    title_contract: NativeSchemaContract,
):
    _assert_invalid(
        _dump(valid_title_payload(**overrides)),
        "schema_invalid",
        contract=title_contract,
    )


def test_nested_keys_must_be_exact(authors_contract: NativeSchemaContract):
    missing = valid_authors_payload()
    del missing["authors"][0]["orcid"]
    extra = valid_authors_payload(unrequested="secret")

    _assert_invalid(_dump(missing), "schema_invalid", contract=authors_contract)
    _assert_invalid(_dump(extra), "schema_invalid", contract=authors_contract)


@pytest.mark.parametrize(
    "payload",
    [
        {"authors": {}},
        {"authors": ["not-an-object"]},
        valid_authors_payload(given=[]),
        valid_authors_payload(corresponding=1),
    ],
)
def test_wrong_nested_containers_and_scalars_are_rejected(
    payload: dict[str, Any],
    authors_contract: NativeSchemaContract,
):
    _assert_invalid(_dump(payload), "schema_invalid", contract=authors_contract)


def test_null_is_accepted_only_where_wire_schema_allows_it(
    title_contract: NativeSchemaContract,
):
    allowed = _parse(
        _dump(valid_title_payload(title=None, abstract=None)),
        contract=title_contract,
    )
    assert isinstance(allowed, TitleKeywordsLLM)
    assert allowed.abstract is None

    _assert_invalid(
        _dump(valid_title_payload(keywords=None)),
        "schema_invalid",
        contract=title_contract,
    )


def test_enum_membership_is_left_to_the_model_layer(
    classification_contract: NativeSchemaContract,
):
    """The wire gate owns the shape of an enum slot; the model owns its members.

    Every enum-backed field already has a lenient ``mode="before"`` validator
    that canonicalises a near miss and degrades an unrecognised label. Failing
    at the wire instead would throw the whole completion away — including every
    field the model got right — over one word the next layer would have fixed.
    """
    accepted = _parse(
        _dump(valid_classification_payload(paper_type="empirical")),
        contract=classification_contract,
    )
    assert accepted.paper_type == "empirical"

    repaired = _parse(
        _dump(valid_classification_payload(paper_type="Empirical")),
        contract=classification_contract,
    )
    assert repaired.paper_type == "empirical"

    degraded = _parse(
        _dump(valid_classification_payload(paper_type="haruspicy")),
        contract=classification_contract,
    )
    assert degraded.paper_type is None

    # Shape is still enforced: a string-typed enum slot must carry a string.
    _assert_invalid(
        _dump(valid_classification_payload(paper_type=12)),
        "schema_invalid",
        contract=classification_contract,
    )


def test_exact_scalar_and_list_item_template_echoes_are_rejected(
    title_contract: NativeSchemaContract,
):
    _assert_invalid(
        _dump(valid_title_payload(title="verbatim-string")),
        "schema_invalid",
        contract=title_contract,
    )
    _assert_invalid(
        _dump(valid_title_payload(keywords=["verbatim-string"])),
        "schema_invalid",
        contract=title_contract,
    )


def test_placeholder_substrings_in_real_values_are_not_rejected(
    title_contract: NativeSchemaContract,
):
    raw_title = "Why the verbatim-string token is not metadata"
    result = _parse(
        _dump(
            valid_title_payload(
                title=raw_title,
                keywords=["not merely a verbatim-string"],
            )
        ),
        contract=title_contract,
    )

    assert result.title == raw_title
    assert result.keywords == ["not merely a verbatim-string"]


def test_choice_template_lists_are_enums_not_placeholder_echoes(
    classification_contract: NativeSchemaContract,
):
    result = _parse(
        _dump(valid_classification_payload()),
        contract=classification_contract,
    )

    assert result.paper_type == "empirical"


def test_duplicate_root_and_nested_keys_are_non_json(
    title_contract: NativeSchemaContract,
    authors_contract: NativeSchemaContract,
):
    root_raw = _dump(valid_title_payload()).replace(
        '"title":"A real title"',
        '"title":"first","title":"second"',
    )
    nested_raw = _dump(valid_authors_payload()).replace(
        '"given":"Ada"',
        '"given":"first","given":"second"',
    )

    _assert_invalid(root_raw, "non_json", contract=title_contract)
    _assert_invalid(nested_raw, "non_json", contract=authors_contract)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_are_non_json(
    constant: str,
    numeric_contract: NativeSchemaContract,
):
    _assert_invalid(
        f'{{"count":1,"score":{constant}}}',
        "non_json",
        contract=numeric_contract,
    )


def test_overflowed_json_float_is_non_json(numeric_contract: NativeSchemaContract):
    _assert_invalid(
        '{"count":1,"score":1e309}',
        "non_json",
        contract=numeric_contract,
    )


def test_decoder_integer_digit_limit_is_a_safe_non_json_failure(
    numeric_contract: NativeSchemaContract,
):
    raw = "9" * 5000

    _assert_invalid(raw, "non_json", contract=numeric_contract)


def test_deep_json_resource_failure_is_a_safe_non_json_failure(
    title_contract: NativeSchemaContract,
):
    raw = "[" * 900 + "0" + "]" * 900

    _assert_invalid(raw, "non_json", contract=title_contract)


@pytest.mark.parametrize(
    "raw",
    [
        '{"count":true,"score":1.0}',
        '{"count":1,"score":false}',
        '{"count":1.0,"score":1.0}',
    ],
)
def test_booleans_and_integral_floats_do_not_satisfy_numeric_wire_types(
    raw: str,
    numeric_contract: NativeSchemaContract,
):
    _assert_invalid(raw, "schema_invalid", contract=numeric_contract)


def test_integer_is_allowed_for_number_but_not_coerced_for_integer(
    numeric_contract: NativeSchemaContract,
):
    result = _parse(
        '{"count":1,"score":2}',
        contract=numeric_contract,
    )

    assert result.count == 1
    assert result.score == 2.0


def test_unknown_wire_schema_shapes_fail_closed():
    contract = NativeSchemaContract(
        response_model=NumericResponse,
        template={"count": "integer", "score": "number"},
        wire_schema={"type": "mystery"},
        nullable_paths=frozenset(),
    )

    _assert_invalid(
        '{"count":1,"score":2}',
        "schema_invalid",
        contract=contract,
    )


@pytest.mark.parametrize(
    "mutate_schema",
    [
        lambda schema: schema.pop("required"),
        lambda schema: schema.__setitem__("required", ["count"]),
        lambda schema: schema.__setitem__("required", [1, "score"]),
        lambda schema: schema.__setitem__("required", [["count"], "score"]),
        lambda schema: schema.pop("additionalProperties"),
        lambda schema: schema.__setitem__("additionalProperties", True),
    ],
)
def test_object_wire_schema_must_be_explicitly_and_exactly_closed(
    mutate_schema,
    numeric_contract: NativeSchemaContract,
):
    wire_schema = json.loads(json.dumps(numeric_contract.wire_schema))
    mutate_schema(wire_schema)
    contract = NativeSchemaContract(
        response_model=numeric_contract.response_model,
        template=numeric_contract.template,
        wire_schema=wire_schema,
        nullable_paths=numeric_contract.nullable_paths,
    )

    _assert_invalid(
        '{"count":1,"score":2}',
        "schema_invalid",
        contract=contract,
    )


def test_arbitrarily_large_integer_is_a_valid_number_without_float_conversion(
    numeric_contract: NativeSchemaContract,
):
    value = {"count": 1, "score": 10**400}

    assert native_wire_value_is_valid(
        value,
        wire_schema=numeric_contract.wire_schema,
        template=numeric_contract.template,
    )


def test_final_pydantic_failure_is_safe_and_does_not_retain_raw_context():
    unique_secret = "UNIQUE-PYDANTIC-RAW-SECRET-9142"
    contract = native_contract_for_model(RejectingResponse)
    raw = json.dumps({"value": unique_secret})

    error = _assert_invalid(raw, "schema_invalid", contract=contract)

    assert unique_secret not in str(error)
    assert unique_secret not in repr(error)
    assert error.__context__ is None
    assert error.__cause__ is None


def test_safe_diagnostic_contains_complete_numeric_usage_without_raw_content(
    title_contract: NativeSchemaContract,
):
    raw = "UNIQUE-RAW-SECRET-82051 is not JSON"
    expected_digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    with pytest.raises(_invalid_output_type()) as raised:
        _parse(
            raw,
            finish_reason="stop",
            contract=title_contract,
            model="numind/NuExtract3-FP8",
            input_tokens=101,
            completion_tokens=17,
            total_tokens=118,
            cached_input_tokens=23,
        )

    error = raised.value
    assert error.category == "non_json"
    assert error.model == "numind/NuExtract3-FP8"
    assert error.finish_reason == "stop"
    assert error.response_chars == len(raw)
    assert error.response_sha256 == expected_digest
    assert error.input_tokens == 101
    assert error.completion_tokens == 17
    assert error.total_tokens == 118
    assert error.cached_input_tokens == 23
    assert error.response_model == "TitleKeywordsLLM"
    for rendered in (str(error), repr(error)):
        assert raw not in rendered
        assert "UNIQUE-RAW-SECRET-82051" not in rendered
        assert "non_json" in rendered
        assert "numind/NuExtract3-FP8" in rendered
        assert "stop" in rendered
        assert str(len(raw)) in rendered
        assert expected_digest in rendered
        assert "101" in rendered
        assert "17" in rendered
        assert "118" in rendered
        assert "23" in rendered
        assert "TitleKeywordsLLM" in rendered
    assert error.__context__ is None
    assert error.__cause__ is None


def test_safe_exception_preserves_base_exception_traceback_mutability(
    title_contract: NativeSchemaContract,
):
    with pytest.raises(_invalid_output_type()) as raised:
        _parse("not JSON", contract=title_contract)

    raised.value.__traceback__ = None
    assert raised.value.__traceback__ is None


def test_formatted_traceback_and_ordinary_logging_do_not_expose_raw_content(
    title_contract: NativeSchemaContract,
    caplog: pytest.LogCaptureFixture,
):
    unique_secret = "UNIQUE-FORMATTED-TRACEBACK-RAW-SECRET-44391"
    caught = None

    try:
        _parse(unique_secret, contract=title_contract)
    except _invalid_output_type() as error:
        caught = error
        with caplog.at_level(logging.ERROR, logger="tests.nuextract.safe"):
            logging.getLogger("tests.nuextract.safe").exception("safe NuExtract failure")

    assert caught is not None
    formatted = "".join(traceback.format_exception(type(caught), caught, caught.__traceback__))
    assert unique_secret not in formatted
    assert unique_secret not in caplog.text
    assert "safe NuExtract failure" in caplog.text
    assert caught.__cause__ is None
    assert caught.__context__ is None


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            SimpleNamespace(
                prompt_tokens=13,
                completion_tokens=5,
                total_tokens=18,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
            (13, 5, 18, 3),
        ),
        (None, (0, 0, 0, 0)),
    ],
)
async def test_backend_captures_finish_reason_and_usage_before_parsing(
    usage,
    expected: tuple[int, int, int, int],
    monkeypatch: pytest.MonkeyPatch,
):
    completion_kwargs: dict[str, Any] = {
        "choices": [
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="UNIQUE-BACKEND-RAW-SECRET"),
            )
        ]
    }
    if usage is not None:
        completion_kwargs["usage"] = usage
    completion = SimpleNamespace(**completion_kwargs)

    class _FakeCompletions:
        async def create(self, **kwargs):
            return completion

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions()),
    )
    backend = NuExtractNativeBackend(
        settings=_settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            model="numind/NuExtract3-FP8",
        )
    )
    monkeypatch.setattr(backend, "_get_client", lambda: fake_client)

    with pytest.raises(_invalid_output_type()) as raised:
        await backend.create(
            response_model=TitleKeywordsLLM,
            system="SYS",
            messages=[{"role": "user", "content": "DOCUMENT"}],
            want_completion=True,
        )

    error = raised.value
    assert error.finish_reason == "stop"
    assert (
        error.input_tokens,
        error.completion_tokens,
        error.total_tokens,
        error.cached_input_tokens,
    ) == expected
    assert "UNIQUE-BACKEND-RAW-SECRET" not in str(error)
    assert "UNIQUE-BACKEND-RAW-SECRET" not in repr(error)


# --- tolerance for contract drift NuExtract actually exhibits ---------------
#
# Measured on RenoBench (10k refs) through the production reference-parse path:
# 155 of 667 batches died, discarding 2,325 references. Two causes below; both
# were rejected by the wire gate BEFORE PaperReferenceLLM's own mode="before"
# validators -- which already normalise exactly these shapes -- could run.


@pytest.fixture(scope="module")
def reference_contract() -> NativeSchemaContract:
    from bibr.schemas import PaperReferenceList

    return native_contract_for_model(PaperReferenceList)


def _reference(contract: NativeSchemaContract, **overrides: Any) -> dict[str, Any]:
    """One reference carrying every key the closed wire schema demands.

    The gate fails closed on a missing key, so a partial payload would fail for
    the wrong reason and prove nothing about the value under test.
    """
    template = contract.template["references"][0]
    reference: dict[str, Any] = {}
    for key, spec in template.items():
        if isinstance(spec, list):
            reference[key] = spec[0]
        elif spec == "integer":
            reference[key] = 2019
        elif spec == "boolean":
            reference[key] = False
        else:
            reference[key] = "x"
    reference.update(overrides)
    return {"references": [reference]}


def test_numeric_page_locators_survive_the_wire_gate(reference_contract):
    """NuExtract emits `"first_page": 143` against its own "string" template.
    2,389 times per 10k refs -- and one of them discarded all 15 in the batch.
    """
    payload = _reference(reference_contract, first_page=143, volume=4, last_page=150)

    assert native_wire_value_is_valid(
        payload,
        wire_schema=reference_contract.wire_schema,
        template=reference_contract.template,
    )


def test_numeric_page_locators_parse_into_strings(reference_contract):
    """End to end: the gate lets them through AND pydantic normalises them, so
    the pair actually recovers the batch rather than moving where it dies."""
    payload = _reference(reference_contract, first_page=143, volume=4, last_page=150)

    result = _parse(json.dumps(payload), contract=reference_contract)

    reference = result.references[0]
    assert (reference.first_page, reference.volume, reference.last_page) == ("143", "4", "150")


def test_page_locators_keep_their_non_numeric_forms(reference_contract):
    """Coercion must not tempt anyone to redeclare these as integers: real
    locators include "S13", "e12345" and "iii"."""
    payload = _reference(reference_contract, first_page="S13", volume="iii", last_page="e12345")

    result = _parse(json.dumps(payload), contract=reference_contract)

    reference = result.references[0]
    assert (reference.first_page, reference.volume, reference.last_page) == (
        "S13",
        "iii",
        "e12345",
    )


def test_a_bib_type_outside_the_enum_reaches_the_model(reference_contract):
    """`dissertation` is not in the template enum, but normalize_bib_type is a
    mode="before" validator built to absorb exactly this. The gate must not
    discard 15 references over a value the model layer already handles."""
    payload = _reference(reference_contract, bib_type="dissertation")

    result = _parse(json.dumps(payload), contract=reference_contract)

    assert result.references[0].bib_type == "thesis"


def test_an_unrecognised_bib_type_degrades_to_other(reference_contract):
    payload = _reference(reference_contract, bib_type="haruspicy")

    result = _parse(json.dumps(payload), contract=reference_contract)

    assert result.references[0].bib_type == "other"


def test_a_non_string_bib_type_is_still_rejected(reference_contract):
    """Tolerance is for values of the right JSON *type*; an int enum member is
    the model losing the shape, not disagreeing about a label."""
    payload = _reference(reference_contract, bib_type=7)

    assert not native_wire_value_is_valid(
        payload,
        wire_schema=reference_contract.wire_schema,
        template=reference_contract.template,
    )


def test_a_boolean_is_not_a_page_number(reference_contract):
    """bool is an int subclass; a stray `true` must not become "True"."""
    payload = _reference(reference_contract, first_page=True)

    assert not native_wire_value_is_valid(
        payload,
        wire_schema=reference_contract.wire_schema,
        template=reference_contract.template,
    )


def test_the_template_placeholder_is_still_rejected_when_numeric(reference_contract):
    """The gate rejects a value equal to its own template placeholder (the
    model echoing the prompt); coercion must not open a hole in that."""
    payload = _reference(reference_contract, first_page="string")

    assert not native_wire_value_is_valid(
        payload,
        wire_schema=reference_contract.wire_schema,
        template=reference_contract.template,
    )


# --- audit S8: the native contract is built once per model, not per call ---


def _stub_backend(monkeypatch, raw: str):
    """A NuExtractNativeBackend whose client returns *raw* for every call."""
    completion = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=raw))],
        usage=None,
    )

    class _FakeCompletions:
        async def create(self, **kwargs):
            return completion

    backend = NuExtractNativeBackend(settings=_settings())
    monkeypatch.setattr(
        backend,
        "_get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions())),
    )
    return backend


async def test_create_builds_the_contract_once(monkeypatch):
    """One LLM call builds the contract once and parses the same title."""
    import bibr.clients.nuextract_schema as schema

    builds = {"n": 0}
    real_build = schema.native_contract_for_model

    def counting(model):
        builds["n"] += 1
        return real_build(model)

    monkeypatch.setattr(nuextract, "native_contract_for_model", counting)
    backend = _stub_backend(monkeypatch, json.dumps(valid_title_payload()))

    result, _completion = await backend.create(
        response_model=TitleKeywordsLLM,
        system="SYS",
        messages=[{"role": "user", "content": "DOCUMENT"}],
        want_completion=True,
    )

    assert builds["n"] == 1
    assert result.title == "A real title"


def test_repeated_contract_calls_share_one_build(monkeypatch):
    """Memoized contracts stay equal and independent: mutating one is safe."""

    class LocalTitle(BaseModel):
        title: str

    import bibr.clients.nuextract_schema as schema

    builds = {"n": 0}
    real_build = schema._build_native_contract

    def counting(model):
        builds["n"] += 1
        return real_build(model)

    monkeypatch.setattr(schema, "_build_native_contract", counting)
    # A cold cache entry: the class object is fresh to this test.
    schema._cached_native_contract.cache_clear()
    try:
        first = schema.native_contract_for_model(LocalTitle)
        second = schema.native_contract_for_model(LocalTitle)
    finally:
        schema._cached_native_contract.cache_clear()

    assert builds["n"] == 1
    assert first == second
    assert first.template is not second.template
    first.template["title"] = "changed"
    assert schema.native_contract_for_model(LocalTitle).template["title"] != "changed"


def test_contract_parse_skips_the_static_shape_recheck(monkeypatch):
    """The builder checks the wire shape once; parses only match values."""
    import bibr.clients.nuextract_schema as schema

    contract = schema.native_contract_for_model(TitleKeywordsLLM)
    assert contract.shape_checked
    raw = json.dumps(valid_title_payload())

    shapes = {"n": 0}
    real_shape = schema._wire_shape_is_supported

    def counting_shape(wire_schema, template):
        shapes["n"] += 1
        return real_shape(wire_schema, template)

    monkeypatch.setattr(schema, "_wire_shape_is_supported", counting_shape)
    for _ in range(2):
        _parse(raw, contract=contract)

    assert shapes["n"] == 0

    hand_built = NativeSchemaContract(
        response_model=contract.response_model,
        template=dict(contract.template),
        wire_schema=dict(contract.wire_schema),
        nullable_paths=contract.nullable_paths,
    )
    assert (
        nuextract.parse_native_completion(
            raw,
            finish_reason="stop",
            contract=hand_built,
            model="m",
            input_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cached_input_tokens=0,
        ).title
        == "A real title"
    )
    assert shapes["n"] > 0  # hand-assembled contracts still pay the check
