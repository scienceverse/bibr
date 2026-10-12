#!/usr/bin/env python3
"""Classify changed repository paths for GitHub Actions job selection."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable

SURFACES = ("python", "package", "docs", "container", "workflow")

_GENERATED_DOC_INPUTS = {
    "bibr/config.py",
    "bibr/config_introspect.py",
    "bibr/export/json_export.py",
    "bibr/export/models.py",
    "bibr/export/schema_artifact.py",
    "bibr/input/supported_files.py",
    "bibr/demo/server.py",
    "bibr/processing_warnings.py",
}
# Root pages the MkDocs build renders or its public-content guard scans.
_DOCS_PAGES = {
    "mkdocs.yml",
    "README.md",
    "LIMITATIONS.md",
    "LLM_POLICY.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
}
_PACKAGE_INPUTS = {
    "LICENSE.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "scripts/check_dist_contents.py",
    # The dist-contents gate imports its dataset suffixes from the tree guard.
    "scripts/check_public_tree.py",
}
_CONTAINER_FILES = ("Dockerfile", "docker-compose", "entrypoint")
_WORKFLOW_INPUTS = {
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
}


def _matches_prefix(path: str, *prefixes: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def classify_paths(paths: Iterable[str]) -> dict[str, bool]:
    """Return the CI surfaces affected by *paths*.

    Any path no rule recognises, in any directory, fails safe into Python/package/workflow
    validation. Workflow changes deliberately select every surface so edits to job
    conditions exercise the complete graph.
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
        container_file = path.startswith(_CONTAINER_FILES) or path == ".dockerignore"
        # The suite also executes the public notebooks and examples, and tests the
        # container files (compose wiring, pins, the OCR entrypoint).
        if container_file or _matches_prefix(
            path, "bibr/", "tests/", "evaluation/", "scripts/", "notebooks/", "examples/"
        ):
            result["python"] = True
            recognized = True
        if _matches_prefix(path, "bibr/") or path in _PACKAGE_INPUTS:
            result["package"] = True
            recognized = True
        if (
            _matches_prefix(path, "docs/")
            or path in _DOCS_PAGES
            or _matches_prefix(path, "bibr/local/cli/")
            or _matches_prefix(path, "scripts/docs_", "scripts/gen_docs_reference.py")
            or path in _GENERATED_DOC_INPUTS
        ):
            result["docs"] = True
            recognized = True
        # The serve image copies bibr/ and runs the segmenter prefetch while building.
        if (
            container_file
            or path in {"pyproject.toml", "uv.lock", "scripts/prefetch_segmenter.py"}
            or _matches_prefix(path, "bibr/")
        ):
            result["container"] = True
            recognized = True

        if not recognized:
            result["python"] = True
            result["package"] = True
            result["workflow"] = True

    return result


def is_commit(ref: str) -> bool:
    """Return whether *ref* names a commit present in the local repository.

    A force push reports the rewritten tip as ``github.event.before``. No ref
    reaches that commit any more, so a fresh ``fetch-depth: 0`` clone lacks it and
    ``git diff`` against it exits 128.
    """

    result = subprocess.run(  # noqa: S603 - arguments are passed without a shell
        ["git", "cat-file", "-e", f"{ref}^{{commit}}"],  # noqa: S607 - see changed_paths
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def changed_paths(base: str, head: str) -> list[str]:
    """List paths changed between explicit Git object IDs using a merge-base diff.

    A detected rename names only its destination, and quoted output hides non-ASCII
    paths from every prefix rule, so moves are listed as a deletion plus an addition
    and paths are read verbatim from NUL-separated records.
    """

    if not base.strip() or not head.strip():
        raise ValueError("base and head SHAs must be non-empty")
    result = subprocess.run(  # noqa: S603 - arguments are passed without a shell
        [  # noqa: S607 - Git is an explicit CI runner prerequisite
            "git",
            "-c",
            "core.quotePath=false",
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            "--diff-filter=ACDMRT",
            f"{base}...{head}",
        ],
        check=True,
        capture_output=True,
    )
    fields = [os.fsdecode(field) for field in result.stdout.split(b"\0")]
    paths: list[str] = []
    index = 0
    while index < len(fields) and fields[index]:
        # Copy and rename records carry a source and a destination path.
        count = 2 if fields[index][0] in "CR" else 1
        paths.extend(field for field in fields[index + 1 : index + 1 + count] if field)
        index += 1 + count
    return paths


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
        if is_commit(args.base):
            surfaces = classify_paths(changed_paths(args.base, args.head))
        else:
            # Without the base there is no diff to classify, so fail safe.
            print(
                f"::warning title=Changed surfaces unknown::base {args.base} is not a "
                "commit in this checkout (history rewritten by a force push?); "
                "selecting every surface",
                file=sys.stderr,
            )
            surfaces = dict.fromkeys(SURFACES, True)

    for name in SURFACES:
        print(f"{name}={str(surfaces[name]).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
