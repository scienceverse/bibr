"""Reject private research material from the public source tree.

In a Git checkout, inspect existing tracked files so pending deletions are
respected and ignored local work is not treated as distributable source. A
standalone source snapshot is checked in full, excluding Git metadata.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

PRIVATE_PREFIXES = ("benchmarks/", "data/", "docs/reports/", "docs/superpowers/")
PUBLIC_EVALUATION_FILES = frozenset(
    {
        "evaluation/__init__.py",
        "evaluation/evaluate.py",
        "evaluation/section_metrics.py",
        "evaluation/validation_metrics.py",
        "evaluation/README.md",
    }
)
DATASET_SUFFIXES = frozenset({".jsonl", ".parquet", ".arrow"})
PUBLIC_PDFS = frozenset(
    {
        "bibr/data/sample_paper.pdf",
        "tests/fixtures/cropbox_offset_sample.pdf",
        "tests/fixtures/native_text_sample.pdf",
        "tests/fixtures/scanned_sample.pdf",
    }
)


def find_violations(names: list[str]) -> list[str]:
    """Check source-relative paths without reading document or dataset content."""
    violations = []
    for name in sorted(set(names)):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            violations.append(f"invalid source-relative path: {name}")
            continue
        normalized = path.as_posix()
        folded = normalized.casefold()
        if folded.startswith(PRIVATE_PREFIXES):
            violations.append(f"private directory: {name}")
        elif folded.startswith("evaluation/") and normalized not in PUBLIC_EVALUATION_FILES:
            violations.append(f"unapproved evaluation file: {name}")
        elif DATASET_SUFFIXES.intersection(suffix.casefold() for suffix in path.suffixes):
            violations.append(f"dataset payload: {name}")
        elif (
            ".pdf" in [suffix.casefold() for suffix in path.suffixes]
            and normalized not in PUBLIC_PDFS
        ):
            violations.append(f"unapproved PDF: {name}")
    return violations


def source_paths(root: Path) -> list[str]:
    """Inventory a checkout index or a standalone snapshot, without Git writes."""
    if (root / ".git").exists() or (root / ".git").is_symlink():
        result = subprocess.run(  # noqa: S603 - fixed read-only Git command, no shell
            ["git", "-C", str(root), "ls-files", "--cached", "-z"],  # noqa: S607 - Git is required
            capture_output=True,
            check=True,
        )
        names = [os.fsdecode(name) for name in result.stdout.split(b"\0") if name]
        return [name for name in names if (root / name).exists() or (root / name).is_symlink()]

    names = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if name != ".git"]
        for name in directories[:]:
            path = Path(current) / name
            if path.is_symlink():
                names.append(path.relative_to(root).as_posix())
                directories.remove(name)
        names.extend(
            (Path(current) / name).relative_to(root).as_posix() for name in files if name != ".git"
        )
    return sorted(names)


def check_tree(root: Path) -> tuple[int, list[str]]:
    names = source_paths(root)
    violations = find_violations(names)
    for name in names:
        path = root / name
        if path.is_symlink():
            violations.append(f"symlinks are not approved public source assets: {name}")
        elif not path.is_file():
            violations.append(f"tracked path is not a regular source file: {name}")
    return len(names), sorted(set(violations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository checkout or clean standalone source snapshot",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"source root is not a directory: {root}")
    try:
        count, violations = check_tree(root)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Unable to inventory public source tree: {error}", file=sys.stderr)
        return 2
    for violation in violations:
        print(violation, file=sys.stderr)
    print(f"Public source tree: {count} files checked, {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
