from __future__ import annotations

import importlib.util
import sys
import tomllib
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "scripts" / "ci" / "audit_dependencies.py"


def load_audit_dependencies() -> ModuleType:
    assert MODULE_PATH.is_file(), "dependency-audit helper has not been implemented"
    spec = importlib.util.spec_from_file_location("audit_dependencies", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_policy(path: Path, entries: str) -> Path:
    path.write_text(entries, encoding="utf-8")
    return path


def test_active_exception_builds_audit_ignore_argument(tmp_path: Path) -> None:
    audit = load_audit_dependencies()
    policy = write_policy(
        tmp_path / "exceptions.toml",
        """
[[exception]]
id = "CVE-2099-0001"
reason = "No fixed release exists."
exposure = "The affected parser is not reachable from untrusted input."
review_by = 2099-01-01
""",
    )

    entries = audit.load_exceptions(policy, today=date(2026, 7, 10))

    assert audit.build_command(entries) == [
        "uv",
        "run",
        "--locked",
        "pip-audit",
        "--desc",
        "on",
        "--ignore-vuln",
        "CVE-2099-0001",
    ]


@pytest.mark.parametrize("missing", ["id", "reason", "exposure", "review_by"])
def test_missing_required_metadata_fails(tmp_path: Path, missing: str) -> None:
    audit = load_audit_dependencies()
    values = {
        "id": '"CVE-2099-0001"',
        "reason": '"No fixed release exists."',
        "exposure": '"The vulnerable path is not reachable."',
        "review_by": "2099-01-01",
    }
    values.pop(missing)
    body = "[[exception]]\n" + "".join(f"{key} = {value}\n" for key, value in values.items())
    policy = write_policy(tmp_path / "exceptions.toml", body)

    with pytest.raises(ValueError, match=missing):
        audit.load_exceptions(policy, today=date(2026, 7, 10))


def test_duplicate_exception_ids_fail(tmp_path: Path) -> None:
    audit = load_audit_dependencies()
    entry = """
[[exception]]
id = "CVE-2099-0001"
reason = "No fixed release exists."
exposure = "The vulnerable path is not reachable."
review_by = 2099-01-01
"""
    policy = write_policy(tmp_path / "exceptions.toml", entry + entry)

    with pytest.raises(ValueError, match="duplicate"):
        audit.load_exceptions(policy, today=date(2026, 7, 10))


def test_expired_exception_fails(tmp_path: Path) -> None:
    audit = load_audit_dependencies()
    policy = write_policy(
        tmp_path / "exceptions.toml",
        """
[[exception]]
id = "CVE-2099-0001"
reason = "No fixed release exists."
exposure = "The vulnerable path is not reachable."
review_by = 2026-07-09
""",
    )

    with pytest.raises(ValueError, match="expired"):
        audit.load_exceptions(policy, today=date(2026, 7, 10))


def test_malformed_review_date_fails(tmp_path: Path) -> None:
    audit = load_audit_dependencies()
    policy = write_policy(
        tmp_path / "exceptions.toml",
        """
[[exception]]
id = "CVE-2099-0001"
reason = "No fixed release exists."
exposure = "The vulnerable path is not reachable."
review_by = "soon"
""",
    )

    with pytest.raises(ValueError, match="review_by"):
        audit.load_exceptions(policy, today=date(2026, 7, 10))


def test_security_tools_are_exactly_pinned() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        root_groups = tomllib.load(pyproject_file)["dependency-groups"]
    with (ROOT / "ci" / "tools" / "pyproject.toml").open("rb") as tools_file:
        tool_dependencies = tomllib.load(tools_file)["project"]["dependencies"]

    assert "security" not in root_groups
    assert tool_dependencies == [
        "actionlint-py==1.7.12.24",
        "zizmor==1.26.1",
    ]
    tools_lock = (ROOT / "ci" / "tools" / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "actionlint-py"' in tools_lock
    assert 'name = "zizmor"' in tools_lock
