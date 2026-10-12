"""Bounded recovery of a complete JSON object without selecting nested data.

Used when Instructor rejected a finished response: a model that writes LaTeX
into a JSON string (``\\(7 \\pm 2\\)``) produces invalid backslash escapes, and
the whole call fails although every value is present. Recovery accepts only one
complete outer object, repairs a bounded number of invalid escapes, and leaves
validation to the requested schema; it never completes truncated JSON or picks
a nested value out of a malformed envelope.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


class StructuredResponseError(ValueError):
    """A finished response could not be recovered; ``category`` says why.

    The categories match :class:`~bibr.exceptions.SafeLlmDiagnostics`'
    invalid-output categories.
    """

    def __init__(self, category: str):
        super().__init__(f"structured response not recoverable: {category}")
        self.category = category


_MAX_RESPONSE_CHARS = 262_144
_MAX_ESCAPE_REPAIRS = 128
_MAX_DEPTH = 64
_LOWERCASE = frozenset("abcdefghijklmnopqrstuvwxyz")
_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\Z", re.IGNORECASE)
_PROSE_FENCE = re.compile(
    r"^```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*(?=\r?$)",
    re.IGNORECASE | re.MULTILINE,
)


def _outer_object_text(content: str) -> str:
    """Accept one explicit JSON fence without discarding competing data.

    Instructor permits explanation around a code block. Preserve that form
    only when the surrounding text cannot carry another object, array or
    fence; never search a malformed bare object for a usable nested value.
    """
    whole_fence = _FENCE.fullmatch(content)
    if whole_fence is not None:
        return whole_fence.group("body").strip()
    if content.count("```") == 2:
        fence = _PROSE_FENCE.search(content)
        if fence is not None:
            prose = content[: fence.start()] + content[fence.end() :]
            if not any(char in prose for char in "{}[]") and "~~~" not in prose:
                return fence.group("body").strip()
    if content.startswith("```"):
        raise StructuredResponseError("trailing_content")
    return content


@dataclass(frozen=True)
class RecoveredStructuredResponse:
    value: BaseModel
    repaired_backslashes: int


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredResponseError("schema_invalid")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise StructuredResponseError("schema_invalid")


def _repair_invalid_backslashes(text: str) -> tuple[str, int]:
    """Escape only a backslash JSON cannot interpret inside a string.

    Existing valid escapes, including unicode escapes, are preserved byte for
    byte, with one exception: once an invalid escape shows that the model
    wrote raw LaTeX, ``\\b \\f \\n \\r \\t`` before a lowercase letter are
    LaTeX commands too (``\\beta``, ``\\frac``, ``\\nu``, ``\\rho``,
    ``\\theta``), not a control character glued to a word. A capital still
    reads as a newline or tab starting a sentence. Quotes, commas, keys and
    incomplete strings are never repaired.
    """
    output: list[str] = []
    # Output slots of escapes that are valid JSON but read as LaTeX; filled
    # with a backslash only if the text turns out to hold raw LaTeX.
    latex_like: list[int] = []
    in_string = False
    position = repairs = 0
    depth = 0
    while position < len(text):
        char = text[position]
        if in_string and char == "\\":
            if position + 1 >= len(text):
                raise StructuredResponseError("truncated")
            following = text[position + 1]
            if following not in '"\\/bfnrtu':
                if ord(following) < 32:
                    raise StructuredResponseError("non_json")
                repairs += 1
                if repairs > _MAX_ESCAPE_REPAIRS:
                    raise StructuredResponseError("non_json")
                output.append("\\")
            elif following in "bfnrt" and text[position + 2 : position + 3] in _LOWERCASE:
                latex_like.append(len(output))
                output.append("")
            output.extend((char, following))
            position += 2
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char in "{[":
                depth += 1
                if depth > _MAX_DEPTH:
                    raise StructuredResponseError("schema_invalid")
            elif char in "}]":
                depth -= 1
        output.append(char)
        position += 1
    if in_string or depth > 0:
        raise StructuredResponseError("truncated")
    if repairs:
        for slot in latex_like:
            output[slot] = "\\"
        repairs += len(latex_like)
    return "".join(output), repairs


def recover_structured_object(
    content: str, response_model: type[BaseModel]
) -> RecoveredStructuredResponse:
    """Validate one complete outer object, optionally fixing invalid escapes.

    Accept a bare object or exactly one JSON markdown fence, with optional
    surrounding prose that contains no competing structured data. Never scan
    for a later/nested object or array, invent field values, or complete
    truncated JSON. Validation remains the requested schema's job.
    """
    if len(content) > _MAX_RESPONSE_CHARS:
        raise StructuredResponseError("non_json")
    body = content.strip()
    if not body:
        raise StructuredResponseError("empty")
    body = _outer_object_text(body)
    if not body.startswith("{"):
        raise StructuredResponseError("non_object" if body.startswith("[") else "non_json")
    body, repairs = _repair_invalid_backslashes(body)
    try:
        value = json.loads(body, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        category = "trailing_content" if exc.msg == "Extra data" else "non_json"
        raise StructuredResponseError(category) from exc
    if not isinstance(value, dict):
        raise StructuredResponseError("non_object")
    # An unrelated wrapper with only ignored keys can otherwise validate as
    # an all-default metadata model. Only the schema's explicit name envelope
    # is delegated to its existing model validator.
    fields = set(response_model.model_fields)
    if value and not fields.intersection(value):
        if set(value) != {response_model.__name__}:
            raise StructuredResponseError("schema_invalid")
        inner = value[response_model.__name__]
        if not isinstance(inner, dict) or (inner and not fields.intersection(inner)):
            raise StructuredResponseError("schema_invalid")
    try:
        result = response_model.model_validate(value)
    except ValidationError as exc:
        raise StructuredResponseError("schema_invalid") from exc
    return RecoveredStructuredResponse(result, repairs)
