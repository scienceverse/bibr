"""Lazy public API visibility and Result aliases (core-api-16).

``dir(bibr)`` must list the public names before any of them is accessed,
imports must stay lazy, and the plural table aliases must resolve like the
singular schema keys.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bibr
from bibr.api import Result

FIXTURE = Path(__file__).parent / "fixtures" / "inspect_full_export.json"

PUBLIC = ("chew", "Chewer", "Result", "write_tables")


def test_dir_lists_public_api_without_prior_access():
    for name in PUBLIC:
        assert name in dir(bibr), name


def test_imports_stay_lazy():
    repo_root = str(Path(bibr.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": repo_root}
    code = "import bibr, sys; assert 'bibr.api' not in sys.modules, 'bibr.api eagerly imported'"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
    assert proc.returncode == 0, proc.stderr.decode()


def test_lazy_attributes_resolve_and_unknown_raises():
    assert bibr.Result is Result
    assert callable(bibr.chew)
    with pytest.raises(AttributeError):
        bibr.__getattr__("no_such_name")


def test_plural_table_aliases():
    result = Result(json.loads(FIXTURE.read_text()))
    assert list(result.tables) == result["table"]
    assert list(result.figures) == result["figure"]
    assert list(result.affiliations) == result["affiliation"]
    assert list(result.urls) == result["url"]
    assert list(result.equations) == result["eq"]
    for alias in ("figures", "tables", "affiliations", "urls", "equations"):
        assert alias in dir(result)
