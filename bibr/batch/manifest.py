"""Input discovery for ``bibr batch``.

Inputs are any mix of:

* **manifest text files** — one path per line, blank lines and ``#`` comment
  lines ignored; relative entries resolve against the manifest's directory;
* **directories** — searched recursively for the extensions bibr supports;
* **individual files**.

A file argument with a manifest-like suffix (``.txt``, ``.lst``, ``.list``,
``.manifest``, or no suffix) is read as a manifest; any other file bibr
cannot process is reported as unsupported instead of being decoded.
Discovery never fails on a missing entry: the path is recorded in
:attr:`Discovery.missing` and the caller decides how loud to be.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bibr.input.supported_files import SUPPORTED_EXTENSIONS

_HASH_CHUNK = 1024 * 1024
# File suffixes read as manifests when passed as file arguments. Anything
# else bibr cannot process goes to Discovery.unsupported: a stray binary
# from a shell glob must not abort the batch with a UnicodeDecodeError, and
# prose files must not become 'not found' manifest entries.
MANIFEST_SUFFIXES = frozenset({".txt", ".lst", ".list", ".manifest", ""})
# Stems ``bibr batch`` does not give a paper as its id: ``<out>/<paper_id>.json``
# would be the runner's own ``run_info.json``.
RESERVED_IDS = frozenset({"run_info"})


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
    unreadable: list[str] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return (
            len(self.missing) + len(self.unsupported) + len(self.empty_dirs) + len(self.unreadable)
        )


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


def is_manifest_like(path: Path) -> bool:
    """Whether a file argument should be read as a manifest (see :data:`MANIFEST_SUFFIXES`)."""
    return path.suffix.lower() in MANIFEST_SUFFIXES


def _read_manifest(manifest: Path, found: Discovery, seen: set[Path]) -> None:
    try:
        text = manifest.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        # A binary or unreadable manifest-like file names itself instead of
        # aborting discovery with a path-less codec error.
        found.unreadable.append(f"{manifest} ({exc})")
        return
    found.manifests.append(manifest)
    base = manifest.parent
    for entry in parse_manifest_lines(text):
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        try:
            is_dir = candidate.is_dir()
            is_file = False if is_dir else candidate.is_file()
        except OSError:
            # An over-long line (a prose file picked up by a shell glob)
            # cannot even be stated: record it, don't abort the batch.
            found.missing.append(f"{entry} (from {manifest.name})")
            continue
        if is_dir:
            found.directories.append(candidate)
            _add_files(_walk_directory(candidate), found, seen, empty_source=candidate)
        elif is_file:
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
            elif is_manifest_like(path):
                _read_manifest(path, found, seen)
            else:
                found.unsupported.append(path)
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


def _sha8(path: Path) -> str | None:
    try:
        return sha256_file(path)[:8]
    except OSError:  # unreadable: the run reports it; the id only has to be unique
        return None


def assign_paper_ids(
    files: Sequence[Path],
    *,
    recorded: Iterable[Mapping[str, Any]] = (),
    reserved: Iterable[str] = (),
) -> list[BatchItem]:
    """Map files to ``paper_id`` = stem, disambiguating stem collisions.

    Exports are written as ``<out>/<paper_id>.json``, so two inputs sharing a
    stem (compared case-insensitively — the default macOS/Windows filesystems
    fold case) would overwrite each other. Colliding files, and a file whose
    stem is in *reserved* (``bibr batch`` passes :data:`RESERVED_IDS`), get
    ``<stem>-<sha256[:8]>``; identical bytes under the same stem additionally
    get an ordinal so every id is unique. Deterministic for a given input
    order.

    *recorded* are the ledger lines of earlier runs. A file whose path has a
    line keeps the id recorded there (a later line wins), so adding inputs
    never renames a paper that was already processed; only the newcomer that
    collides with it gets a suffix.
    """
    by_path: dict[str, str] = {}
    for entry in recorded:
        path_text, recorded_id = entry.get("path"), entry.get("paper_id")
        if isinstance(path_text, str) and isinstance(recorded_id, str) and recorded_id:
            by_path[path_text] = recorded_id

    reserved_ids = {stem.casefold() for stem in reserved}
    # A recorded id is kept only while no earlier input, and no reserved stem,
    # has it.
    used: set[str] = set(reserved_ids)
    kept: dict[int, str] = {}
    for index, path in enumerate(files):
        recorded_id = by_path.get(str(path))
        if recorded_id is not None and recorded_id.casefold() not in used:
            kept[index] = recorded_id
            used.add(recorded_id.casefold())

    by_stem: dict[str, list[Path]] = {}
    for path in files:
        by_stem.setdefault(path.stem.casefold(), []).append(path)

    items: list[BatchItem] = []
    for index, path in enumerate(files):
        stem = path.stem
        paper_id = kept.get(index)
        if paper_id is None:
            collides = len(by_stem[stem.casefold()]) > 1 or stem.casefold() in reserved_ids
            sha8 = _sha8(path) if collides else None
            paper_id = f"{stem}-{sha8}" if sha8 else stem
            base = paper_id
            ordinal = 2
            while paper_id.casefold() in used:
                paper_id = f"{base}-{ordinal}"
                ordinal += 1
            used.add(paper_id.casefold())
        items.append(BatchItem(path=path, paper_id=paper_id, stem=stem))
    return items
