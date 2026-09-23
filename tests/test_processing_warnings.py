"""The warning-code registry and the ``ProcessingWarning`` carrier.

Every code bibr writes to ``extraction.warnings[].code`` is a ``WarningCode``
member with a description, matches the pattern the export schema publishes, and
is emitted somewhere; every emission site names a member, never a string.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from bibr.export.models import WARNING_CODE_PATTERN
from bibr.processing_warnings import DESCRIPTIONS, ProcessingWarning, WarningCode

PACKAGE = Path(__file__).resolve().parents[1] / "bibr"
REGISTRY = PACKAGE / "processing_warnings.py"


def _modules() -> list[tuple[Path, ast.Module]]:
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in PACKAGE.rglob("*.py")]


def test_every_code_is_its_own_name_and_matches_the_schema_pattern():
    for code in WarningCode:
        assert code.value == code.name
        assert re.fullmatch(WARNING_CODE_PATTERN, code), code


def test_every_code_has_a_one_line_description():
    assert set(DESCRIPTIONS) == set(WarningCode)
    for code, description in DESCRIPTIONS.items():
        assert description.strip() and "\n" not in description, code


def test_emission_sites_name_registered_codes():
    """A code is always ``WarningCode.<MEMBER>``: a misspelled member would only
    fail when its (often rare) error path runs, and a string literal would
    bypass the registry."""
    referenced: set[str] = set()
    problems: list[str] = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "WarningCode"
                and path != REGISTRY
            ):
                referenced.add(node.attr)
                if node.attr not in WarningCode.__members__:
                    problems.append(f"{path.name}:{node.lineno} unknown WarningCode.{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ProcessingWarning"
            ):
                code = node.args[0] if node.args else None
                code = next((kw.value for kw in node.keywords if kw.arg == "code"), code)
                if isinstance(code, ast.Constant):
                    problems.append(f"{path.name}:{node.lineno} literal code {code.value!r}")
    assert not problems, problems
    unused = set(WarningCode.__members__) - referenced
    assert not unused, f"registered but never emitted: {sorted(unused)}"


def test_warnings_dedupe_by_value():
    first = ProcessingWarning(WarningCode.OCR_PAGE_FAILED, "page 3")
    same = ProcessingWarning("OCR_PAGE_FAILED", "page 3")
    other = ProcessingWarning(WarningCode.OCR_PAGE_FAILED, "page 4")
    assert list(dict.fromkeys([first, same, other])) == [first, other]
    assert type(first.code) is str


def test_dict_round_trip():
    warning = ProcessingWarning(
        WarningCode.LOW_TEXT_QUALITY, "text-quality score 0.31 is below 0.5"
    )
    assert warning.to_dict() == {
        "code": "LOW_TEXT_QUALITY",
        "message": "text-quality score 0.31 is below 0.5",
    }
    assert ProcessingWarning.from_dict(warning.to_dict()) == warning
    assert ProcessingWarning.from_dict(warning) is warning


@pytest.mark.parametrize(
    "value",
    [
        "LOW_TEXT_QUALITY: 0.31",
        {"code": "LOW_TEXT_QUALITY"},
        {"code": "LOW_TEXT_QUALITY", "message": 0.31},
        {"code": "LOW_TEXT_QUALITY", "message": "m", "severity": "warning"},
        None,
    ],
)
def test_from_dict_rejects_anything_but_a_code_and_a_message(value):
    with pytest.raises(ValueError, match="processing warning"):
        ProcessingWarning.from_dict(value)
