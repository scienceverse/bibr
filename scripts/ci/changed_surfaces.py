#!/usr/bin/env python3
"""Classify changed repository paths for GitHub Actions job selection."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterable

SURFACES = ("python", "package", "docs", "container", "workflow")

_GENERATED_DOC_INPUTS = {
    "bibr/config.py",
    "bibr/config_introspect.py",
    "bibr/export/json_export.py",
    "bibr/input/supported_files.py",
    "bibr/demo/server.py",
}
_PACKAGE_INPUTS = {
    "LICENSE.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "scripts/check_dist_contents.py",
}
_WORKFLOW_INPUTS = {
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
}


def _matches_prefix(path: str, *prefixes: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def classify_paths(paths: Iterable[str]) -> dict[str, bool]:
    """Return the CI surfaces affected by *paths*.

    Unknown root build inputs fail safe into Python/package/workflow validation. Workflow
    changes deliberately select every surface so edits to job conditions exercise the
    complete graph.
    """

    result = dict.fromkeys(SURFACES, False)
    normalized = [path.removeprefix("./").replace("\\", "/") for path in paths if path]

    if any(
        _matches_prefix(path, ".github/", "ci/", "scripts/ci/") or path in _WORKFLOW_INPUTS
        for path in normalized
    ):
        return dict.fromkeys(SURFACES, True)

    for path in normalized:
        recognized = False
        if _matches_prefix(path, "bibr/", "tests/", "evaluation/", "scripts/"):
            result["python"] = True
            recognized = True
        if _matches_prefix(path, "bibr/") or path in _PACKAGE_INPUTS:
            result["package"] = True
            recognized = True
        if (
            _matches_prefix(path, "docs/")
            or path in {"mkdocs.yml", "README.md", "LIMITATIONS.md", "LLM_POLICY.md"}
            or _matches_prefix(path, "bibr/local/cli/")
            or _matches_prefix(path, "scripts/docs_", "scripts/gen_docs_reference.py")
            or path in _GENERATED_DOC_INPUTS
        ):
            result["docs"] = True
            recognized = True
        if (
            path.startswith("Dockerfile")
            or path == ".dockerignore"
            or path.startswith("docker-compose")
            or path in {"pyproject.toml", "uv.lock"}
        ):
            result["container"] = True
            recognized = True

        if "/" not in path and not path.endswith(".md") and not recognized:
            result["python"] = True
            result["package"] = True
            result["workflow"] = True

    return result


def changed_paths(base: str, head: str) -> list[str]:
    """List paths changed between explicit Git object IDs using a merge-base diff."""

    if not base.strip() or not head.strip():
        raise ValueError("base and head SHAs must be non-empty")
    result = subprocess.run(  # noqa: S603 - arguments are passed without a shell
        [  # noqa: S607 - Git is an explicit CI runner prerequisite
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMRT",
            f"{base}...{head}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--all", action="store_true", dest="select_all")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.select_all:
        surfaces = dict.fromkeys(SURFACES, True)
    else:
        if not args.base or not args.head:
            _parser().error("--base and --head are required unless --all is supplied")
        surfaces = classify_paths(changed_paths(args.base, args.head))

    for name in SURFACES:
        print(f"{name}={str(surfaces[name]).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
