"""Input discovery for ``bibr batch``.

Inputs are any mix of:

* **manifest text files** — one path per line, blank lines and ``#`` comment
  lines ignored; relative entries resolve against the manifest's directory;
* **directories** — searched recursively for the extensions bibr supports;
* **individual files**.

A file argument whose extension bibr cannot process (``.txt``, ``.lst``, ...)
is read as a manifest. Discovery never fails on a missing entry: the path is
recorded in :attr:`Discovery.missing` and the caller decides how loud to be.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bibr.input.supported_files import SUPPORTED_EXTENSIONS

_HASH_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class BatchItem:
    """One input file with its ledger/export identity."""

    path: Path
    paper_id: str
    stem: str

    @property
    def disambiguated(self) -> bool:
        """True when a stem collision forced a sha-suffixed ``paper_id``."""
        return self.paper_id != self.stem


@dataclass
class Discovery:
    """What :func:`discover_inputs` found, and what it could not resolve."""

    files: list[Path] = field(default_factory=list)
    manifests: list[Path] = field(default_factory=list)
    directories: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    empty_dirs: list[Path] = field(default_factory=list)
    unsupported: list[Path] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return len(self.missing) + len(self.unsupported) + len(self.empty_dirs)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _walk_directory(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and is_supported(p))


def parse_manifest_lines(text: str) -> list[str]:
    """Entries of a manifest: stripped, non-empty, not starting with ``#``."""
    entries: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def _read_manifest(manifest: Path, found: Discovery, seen: set[Path]) -> None:
    found.manifests.append(manifest)
    base = manifest.parent
    for entry in parse_manifest_lines(manifest.read_text(encoding="utf-8")):
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_dir():
            found.directories.append(candidate)
            _add_files(_walk_directory(candidate), found, seen, empty_source=candidate)
        elif candidate.is_file():
            _add_files([candidate], found, seen)
        else:
            found.missing.append(f"{entry} (from {manifest.name})")


def _add_files(
    paths: Sequence[Path],
    found: Discovery,
    seen: set[Path],
    *,
    empty_source: Path | None = None,
) -> None:
    if not paths and empty_source is not None:
        found.empty_dirs.append(empty_source)
        return
    for path in paths:
        if not is_supported(path):
            found.unsupported.append(path)
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        found.files.append(path)


def discover_inputs(inputs: Sequence[str | Path]) -> Discovery:
    """Resolve *inputs* (manifests, directories, files) to an ordered file list.

    Order is deterministic: inputs in the order given, manifest entries in
    file order, directory contents sorted. Duplicates (same resolved path)
    are kept once, at their first position.
    """
    found = Discovery()
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            found.directories.append(path)
            _add_files(_walk_directory(path), found, seen, empty_source=path)
        elif path.is_file():
            if is_supported(path):
                _add_files([path], found, seen)
            else:
                _read_manifest(path, found, seen)
        else:
            found.missing.append(str(raw))
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(_HASH_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def assign_paper_ids(files: Sequence[Path]) -> list[BatchItem]:
    """Map files to ``paper_id`` = stem, disambiguating stem collisions.

    Exports are written as ``<out>/<paper_id>.json``, so two inputs sharing a
    stem (compared case-insensitively — the default macOS/Windows filesystems
    fold case) would overwrite each other. Colliding files get
    ``<stem>-<sha256[:8]>``; identical bytes under the same stem additionally
    get an ordinal so every id is unique. Deterministic for a given input
    order.
    """
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        by_stem.setdefault(path.stem.casefold(), []).append(path)

    items: list[BatchItem] = []
    used: set[str] = set()
    for path in files:
        stem = path.stem
        if len(by_stem[stem.casefold()]) == 1:
            paper_id = stem
        else:
            paper_id = f"{stem}-{sha256_file(path)[:8]}"
        base = paper_id
        ordinal = 2
        while paper_id.casefold() in used:
            paper_id = f"{base}-{ordinal}"
            ordinal += 1
        used.add(paper_id.casefold())
        items.append(BatchItem(path=path, paper_id=paper_id, stem=stem))
    return items
