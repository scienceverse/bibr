"""Compare bibr exports by content.

Two runs of one PDF on one build do not write the same bytes: the export says
when the run finished, how long each stage took and which build wrote it, and
model scores differ in their last digits from one CPU to the next. Everything
else is the same on the CPU path (fast mode on the text-layer route), which
makes an exact comparison a usable regression check there: a difference is a
change in what bibr read, not noise.

``canonical_export`` removes the run stamps. ``diff_exports`` lists what still
differs between two exports, treating numbers closer than a tolerance as equal.
``compare_export_dirs`` does that for two folders of exports and counts the
differences per place in the schema.

From a shell::

    python -m bibr.export.compare RUN_A RUN_B

prints the counts and exits 0 when the two folders hold the same content, 1
when they differ (or a paper is in one folder only), 2 for a bad argument.

Paths that go through an OCR or LLM server are not exactly repeatable, so this
check says little about them.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_FLOAT_TOLERANCE",
    "VOLATILE_PATHS",
    "Difference",
    "DirComparison",
    "canonical_export",
    "compare_export_dirs",
    "diff_exports",
    "main",
]

#: What a run writes about itself rather than about the paper: when it finished,
#: how long each stage took, and the build that wrote it.
VOLATILE_PATHS: tuple[tuple[str, ...], ...] = (
    ("extraction", "completed_at"),
    ("extraction", "timings"),
    ("extraction", "producer"),
)

#: Two numbers this close are one number. Classifier scores of the same build
#: differed by up to 1.6e-6 between two x86 desktops; a score that moves by more
#: than this is worth a line in the report.
DEFAULT_FLOAT_TOLERANCE = 1e-4

_MISSING = "missing"
_TYPE = "type"
_LENGTH = "length"
_NUMBER = "number"
_VALUE = "value"


@dataclass(frozen=True)
class Difference:
    """One place where two exports differ.

    ``path`` names the place by schema position, without list indexes
    (``.bib[].title``), so that the same change in many rows or many papers
    counts under one name. ``kind`` is ``value``, ``number`` (two numbers further
    apart than the tolerance), ``length`` (two lists of different length, which
    are not compared further), ``type``, or ``missing`` (a key one side lacks).
    """

    path: str
    kind: str


def canonical_export(export: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``export`` without the run stamps in ``VOLATILE_PATHS``."""
    out = copy.deepcopy(dict(export))
    for path in VOLATILE_PATHS:
        node: Any = out
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_equal(a: float, b: float, tolerance: float) -> bool:
    if a == b:
        return True
    if a != a or b != b:  # NaN: two of them are one value, one beside a number is not
        return a != a and b != b
    if isinstance(a, int) and isinstance(b, int):
        return False  # ids, offsets and counts are exact
    return abs(a - b) <= tolerance


def _walk(a: Any, b: Any, path: str, tolerance: float) -> Iterator[Difference]:
    if _is_number(a) and _is_number(b):
        if not _numbers_equal(a, b, tolerance):
            yield Difference(path, _NUMBER)
        return
    if type(a) is not type(b):
        yield Difference(path, _TYPE)
        return
    if isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                yield Difference(f"{path}.{key}", _MISSING)
            else:
                yield from _walk(a[key], b[key], f"{path}.{key}", tolerance)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield Difference(f"{path}[]", _LENGTH)
            return
        for left, right in zip(a, b, strict=True):
            yield from _walk(left, right, f"{path}[]", tolerance)
    elif a != b:
        yield Difference(path, _VALUE)


def diff_exports(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    float_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
) -> list[Difference]:
    """Where two exports of one paper differ, run stamps and score noise aside.

    Rows are compared by position. Two lists of different length give one
    ``length`` difference and their rows are not compared, so one added
    reference is one line, not a line for every reference after it.
    """
    return list(_walk(canonical_export(a), canonical_export(b), "", float_tolerance))


def _load_exports(folder: Path) -> dict[str, dict[str, Any]]:
    """The exports in ``folder`` by file name; other JSON files are left out.

    A batch output folder also holds ``run_info.json`` and may hold other JSON
    files; an export is a JSON object with ``schema_version`` and ``extraction``.
    """
    exports: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and "schema_version" in doc and "extraction" in doc:
            exports[path.name] = doc
    return exports


@dataclass
class DirComparison:
    """What ``compare_export_dirs`` found."""

    #: Papers present in both folders.
    papers: int = 0
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    #: File name to its differences, for the papers that differ.
    changed: dict[str, list[Difference]] = field(default_factory=dict)

    @property
    def identical(self) -> int:
        return self.papers - len(self.changed)

    @property
    def same(self) -> bool:
        """Both folders hold the same papers with the same content."""
        return not (self.changed or self.only_a or self.only_b)

    @property
    def changed_outside_extraction(self) -> list[str]:
        """Papers whose data differ, not only the ``extraction`` record of the run."""
        return [
            name
            for name, differences in self.changed.items()
            if any(not d.path.startswith(".extraction") for d in differences)
        ]

    def by_path(self) -> list[tuple[str, str, int, int]]:
        """``(path, kind, papers, places)``, the most widespread difference first."""
        papers: dict[tuple[str, str], set[str]] = {}
        places: dict[tuple[str, str], int] = {}
        for name, differences in self.changed.items():
            for difference in differences:
                key = (difference.path, difference.kind)
                papers.setdefault(key, set()).add(name)
                places[key] = places.get(key, 0) + 1
        return sorted(
            (
                (path, kind, len(names), places[(path, kind)])
                for (path, kind), names in papers.items()
            ),
            key=lambda row: (-row[2], row[0], row[1]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "papers": self.papers,
            "identical": self.identical,
            "changed": len(self.changed),
            "changed_outside_extraction": len(self.changed_outside_extraction),
            "only_a": self.only_a,
            "only_b": self.only_b,
            "by_path": [
                {"path": path, "kind": kind, "papers": papers, "places": places}
                for path, kind, papers, places in self.by_path()
            ],
            "changed_papers": {
                name: sorted({d.path for d in differences})
                for name, differences in sorted(self.changed.items())
            },
        }


def compare_export_dirs(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    float_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
) -> DirComparison:
    """Compare the exports of two folders, matched by file name."""
    exports_a = _load_exports(Path(dir_a))
    exports_b = _load_exports(Path(dir_b))
    result = DirComparison(
        only_a=sorted(exports_a.keys() - exports_b.keys()),
        only_b=sorted(exports_b.keys() - exports_a.keys()),
    )
    for name in sorted(exports_a.keys() & exports_b.keys()):
        result.papers += 1
        differences = diff_exports(
            exports_a[name], exports_b[name], float_tolerance=float_tolerance
        )
        if differences:
            result.changed[name] = differences
    return result


def _report(result: DirComparison, limit: int) -> str:
    lines = [
        f"{result.papers} papers in both folders: {result.identical} the same, "
        f"{len(result.changed)} differ "
        f"({len(result.changed_outside_extraction)} outside the extraction record)"
    ]
    for label, names in (("only in A", result.only_a), ("only in B", result.only_b)):
        if names:
            shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
            lines.append(f"{len(names)} {label}: {shown}")
    rows = result.by_path()
    if rows:
        lines.append(f"{'papers':>6} {'places':>7}  {'kind':<8} path")
        lines.extend(
            f"{papers:>6} {places:>7}  {kind:<8} {path}"
            for path, kind, papers, places in rows[:limit]
        )
        if len(rows) > limit:
            lines.append(f"... and {len(rows) - limit} more paths (--limit 0 lists all)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bibr.export.compare",
        description="Compare two folders of bibr exports by content, run stamps and score "
        "noise aside. Exits 0 when they hold the same content, 1 when they differ.",
    )
    parser.add_argument("dir_a", type=Path, help="folder of exports, for example a batch --out")
    parser.add_argument("dir_b", type=Path, help="folder of exports to compare with it")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_FLOAT_TOLERANCE,
        help=f"numbers this close are equal (default {DEFAULT_FLOAT_TOLERANCE})",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument(
        "--limit", type=int, default=40, help="paths to list, 0 for all (default 40)"
    )
    args = parser.parse_args(argv)
    for folder in (args.dir_a, args.dir_b):
        if not folder.is_dir():
            parser.error(f"not a folder: {folder}")
    if args.tolerance < 0:
        parser.error("--tolerance cannot be negative")

    result = compare_export_dirs(args.dir_a, args.dir_b, float_tolerance=args.tolerance)
    if not result.papers and not (result.only_a or result.only_b):
        parser.error("no bibr exports in either folder")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(_report(result, args.limit if args.limit > 0 else sys.maxsize))
    return 0 if result.same else 1


if __name__ == "__main__":
    sys.exit(main())
