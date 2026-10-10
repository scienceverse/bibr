"""Dependency audit: the vLLM pin, the locked fixes and the exception policy."""

import importlib.util
import sys
import tomllib
from datetime import date
from functools import cache
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).parents[1]

# pip-audit reports 20 advisories for vllm 0.27.0 (PYSEC-2026-3985, a
# trust_remote_code bypass, PYSEC-2026-3997/3999, ...); 0.30.0 fixes the last.
VLLM_FIXED = Version("0.30.0")
# Excepted advisory -> (package, first fixed release).
ADVISORY_FIXES = {
    "CVE-2025-3000": ("torch", Version("2.13.0")),
    "PYSEC-2026-3447": ("setuptools", Version("83.0.0")),
}


@cache
def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@cache
def _locked() -> dict[str, list[dict]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages: dict[str, list[dict]] = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    return packages


def _vllm_extra() -> tuple[Requirement, Requirement]:
    vllm, sdk = (Requirement(r) for r in _pyproject()["project"]["optional-dependencies"]["vllm"])
    return vllm, sdk


def _same(launched: str, declared: Requirement) -> bool:
    requirement = Requirement(launched)
    return (requirement.name, requirement.specifier) == (declared.name, declared.specifier)


def _policy_ids() -> set[str]:
    spec = importlib.util.spec_from_file_location(
        "audit_dependencies_for_deps", ROOT / "scripts" / "ci" / "audit_dependencies.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    policy = ROOT / "ci" / "vulnerability-exceptions.toml"
    # Both exceptions used to expire on 2026-10-17, failing the audit on every
    # PR. A fixed later date keeps the suite itself from expiring: the CI audit
    # job enforces each review date as it comes due.
    return {entry.id for entry in module.load_exceptions(policy, today=date(2026, 10, 18))}


def _env(platform: str, python: str) -> dict[str, str]:
    sys_platform, machine = {
        "linux": ("linux", "x86_64"),
        "darwin": ("darwin", "arm64"),
        "win32": ("win32", "AMD64"),
    }[platform]
    env = dict(default_environment())
    env.update(
        sys_platform=sys_platform,
        platform_machine=machine,
        python_version=python,
        python_full_version=f"{python}.0",
    )
    return env


SUPPORTED_ENVS = [
    (platform, python)
    for platform in ("linux", "darwin", "win32")
    for python in ("3.11", "3.12", "3.13", "3.14")
]


def test_vllm_extra_pins_a_release_clear_of_the_known_advisories():
    vllm, _sdk = _vllm_extra()
    (spec,) = vllm.specifier

    assert spec.operator == "=="
    assert Version(spec.version) >= VLLM_FIXED


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [("bibr.local.vllm_llm", "VllmLlmServer"), ("bibr.local.vllm_ocr", "VllmOcrServer")],
)
def test_uv_bootstrap_runs_the_audited_vllm_extra(module_name, class_name):
    # The bootstrap executes a downloaded vLLM, so it must be the same release
    # the extra pins (and the lock audits), not an older one.
    module = importlib.import_module(module_name)
    vllm, sdk = _vllm_extra()
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch.object(module.shutil, "which", return_value="/usr/bin/uv"),
    ):
        cmd = getattr(module, class_name)._resolve_launch_cmd("org/model")

    pin = cmd[cmd.index("--from") + 1]
    assert _same(pin, vllm), pin
    assert Version(pin.partition("==")[2]) >= VLLM_FIXED
    assert _same(cmd[cmd.index("--with") + 1], sdk)


def test_lock_holds_only_the_pinned_fixed_vllm():
    vllm, _sdk = _vllm_extra()
    (locked,) = (p["version"] for p in _locked()["vllm"])

    assert locked == next(iter(vllm.specifier)).version
    assert Version(locked) >= VLLM_FIXED


def test_lock_holds_no_torch_without_the_cve_2025_3000_fix():
    versions = [Version(p["version"]) for p in _locked()["torch"]]

    assert versions
    assert all(v >= ADVISORY_FIXES["CVE-2025-3000"][1] for v in versions), versions


def test_unfixed_setuptools_is_locked_only_where_vllm_caps_it():
    # vLLM caps setuptools<81 on Python >3.11, so the Linux 3.12/3.13 forks
    # cannot reach the 83.0.0 fix; every other fork must lock it.
    fixed = ADVISORY_FIXES["PYSEC-2026-3447"][1]
    capped = {("linux", "3.12"), ("linux", "3.13")}
    for package in _locked()["setuptools"]:
        if Version(package["version"]) >= fixed:
            continue
        markers = [Marker(m) for m in package.get("resolution-markers", [])]
        assert markers, f"setuptools {package['version']} is locked for every environment"
        reached = {e for e in SUPPORTED_ENVS if any(m.evaluate(_env(*e)) for m in markers)}
        assert reached <= capped, (package["version"], sorted(reached - capped))


def test_exceptions_match_what_the_lock_still_holds():
    ids = _policy_ids()
    for advisory, (name, fixed) in ADVISORY_FIXES.items():
        affected = [p["version"] for p in _locked()[name] if Version(p["version"]) < fixed]
        assert (advisory in ids) == bool(affected), (advisory, name, affected)
