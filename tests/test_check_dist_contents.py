"""Unit tests for the dist-contents allowlist checker (pure function, no build)."""

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_dist_contents import WHEEL_REQUIRED_FILES, find_violations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CLEAN_SDIST = [
    "bibr-0.3.0/PKG-INFO",
    "bibr-0.3.0/README.md",
    "bibr-0.3.0/LICENSE.md",
    "bibr-0.3.0/pyproject.toml",
    "bibr-0.3.0/.gitignore",
    "bibr-0.3.0/bibr/__init__.py",
    "bibr-0.3.0/bibr/config.py",
    "bibr-0.3.0/bibr/data/sample_paper.pdf",
]
CLEAN_WHEEL = [
    "bibr/__init__.py",
    "bibr/py.typed",
    "bibr/data/sample_paper.pdf",
    "bibr-0.3.0.dist-info/METADATA",
    "bibr-0.3.0.dist-info/RECORD",
    "bibr-0.3.0.dist-info/licenses/LICENSE.md",
]


def test_clean_dists_pass():
    assert find_violations(CLEAN_SDIST, CLEAN_WHEEL) == []


def test_typed_marker_is_required_in_wheels():
    assert {"bibr/py.typed"} == WHEEL_REQUIRED_FILES
    wheel_without_marker = [name for name in CLEAN_WHEEL if name != "bibr/py.typed"]

    violations = find_violations(CLEAN_SDIST, wheel_without_marker)

    assert violations == ["wheel: missing required member bibr/py.typed"]


def test_sdist_data_dir_flagged():
    names = [*CLEAN_SDIST, "bibr-0.3.0/data/arxiv_bert.pdf"]
    violations = find_violations(names, CLEAN_WHEEL)
    assert any("data/arxiv_bert.pdf" in v for v in violations)


def test_sdist_evaluation_and_tests_flagged():
    names = [
        *CLEAN_SDIST,
        "bibr-0.3.0/evaluation/gold/paper.json",
        "bibr-0.3.0/tests/test_bibr.py",
    ]
    violations = find_violations(names, CLEAN_WHEEL)
    assert len(violations) == 2


def test_pdf_outside_package_data_flagged_in_wheel():
    wheel = [*CLEAN_WHEEL, "bibr/extract/leaked.pdf"]
    violations = find_violations(CLEAN_SDIST, wheel)
    assert any("leaked.pdf" in v for v in violations)


def test_wheel_foreign_toplevel_flagged():
    wheel = [*CLEAN_WHEEL, "evaluation/__init__.py"]
    violations = find_violations(CLEAN_SDIST, wheel)
    assert any("evaluation/__init__.py" in v for v in violations)


def test_sdist_excludes_ci_only_projects():
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    excludes = project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "/ci" in excludes
