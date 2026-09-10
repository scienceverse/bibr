"""The release matrix must collect tests on hosts without Unix signal APIs."""

import subprocess
import sys
from pathlib import Path


def test_conftest_import_and_teardown_without_killpg():
    conftest = Path(__file__).with_name("conftest.py")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy, sys; "
            "vars(os).pop('killpg', None); "
            "namespace = runpy.run_path(sys.argv[1]); "
            "assert not hasattr(os, 'killpg'); "
            "fixture = namespace['_killpg_guard'].__wrapped__(); "
            "next(fixture); "
            "assert list(fixture) == []; "
            "assert not hasattr(os, 'killpg')",
            str(conftest),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
