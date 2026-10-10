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
import unicodedata
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
# Names Windows opens as a device, whatever follows the first dot.
WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"{port}{n}" for port in ("com", "lpt") for n in "0123456789¹²³"}
)
NAME_MAX = 255  # bytes per file name on common filesystems
# Leaves NAME_MAX room for ``<paper_id>.json``, the longest sidecar name
# (``<paper_id>.json.enrichment.json``) and an ordinal.
MAX_ID_BYTES = 200


@dataclass(frozen=True)
class BatchItem:
    """One input file with its ledger/export identity."""

    path: Path
    paper_id: str
    stem: str
    # The file's sha256 when :func:`assign_paper_ids` had to read it.
    sha256: str | None = field(default=None, compare=False, repr=False)

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


def _sha256(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:  # unreadable: the run reports it; the id only has to be unique
        return None


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def resolved_path(path: object) -> Path | None:
    """*path* (a ``Path``, or a ledger line's ``path`` text) made absolute; None if it cannot be."""
    if not isinstance(path, (str, Path)) or not str(path):
        return None
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError):  # a symlink loop, a NUL byte
        return None


def id_key(paper_id: str) -> str:
    """*paper_id* as filesystems compare names: case-insensitively (macOS,
    Windows), and with NFC and NFD spellings equal (APFS)."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", paper_id).casefold())


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8", "surrogateescape"))


def _is_device_name(paper_id: str) -> bool:
    return id_key(paper_id.split(".", 1)[0].rstrip(" ")) in WINDOWS_DEVICE_NAMES


def _with_suffix(stem: str, suffix: str) -> str:
    """``<stem><suffix>``, *stem* clipped so the id fits :data:`MAX_ID_BYTES`.

    On a Windows device name the suffix goes before the first dot
    (``nul.tar`` → ``nul-1a2b3c4d.tar``): after it, Windows would still open
    the device.
    """
    limit = max(MAX_ID_BYTES - _nbytes(suffix), 1)
    clipped = stem.encode("utf-8", "surrogateescape")[:limit].decode("utf-8", "ignore")
    if _is_device_name(clipped):
        head, dot, rest = clipped.partition(".")
        return f"{head}{suffix}{dot}{rest}"
    return f"{clipped}{suffix}"


def assign_paper_ids(
    files: Sequence[Path],
    *,
    recorded: Iterable[Mapping[str, Any]] = (),
    reserved: Iterable[str] = (),
) -> list[BatchItem]:
    """Map files to ``paper_id`` = stem, disambiguating stem collisions.

    Exports are written as ``<out>/<paper_id>.json``, so two inputs sharing a
    stem (compared as :func:`id_key` does: the default macOS/Windows
    filesystems fold case, APFS equates NFC and NFD) would overwrite each
    other. Colliding files get ``<stem>-<sha256[:8]>``, and so do a file whose
    stem is in *reserved* (``bibr batch`` passes :data:`RESERVED_IDS`) or is a
    Windows device name, and a stem longer than :data:`MAX_ID_BYTES` (clipped
    to fit); identical bytes under the same stem additionally get an ordinal
    so every id is unique. Deterministic for a given input order.

    *recorded* are the ledger lines of earlier runs; the latest line of each
    recorded id names its file. The file at that path (compared resolved; a
    different size means another file is there now), else one with the same
    sha256, keeps the id, so adding inputs never renames, or re-runs, a paper
    that was already processed. A newcomer whose stem an id was recorded under
    for other bytes gets a suffix, even when that file is not among *files*.
    """
    latest: dict[str, Mapping[str, Any]] = {}
    for entry in recorded:
        paper_id = entry.get("paper_id")
        # An id too long for its export name never had an export.
        if isinstance(paper_id, str) and paper_id and _nbytes(f"{paper_id}.json") <= NAME_MAX:
            latest.pop(paper_id, None)  # keep the order of each id's latest line
            latest[paper_id] = entry
    by_path: dict[Path, str] = {}
    by_sha: dict[str, list[str]] = {}
    recorded_shas: dict[str, set[str]] = {}
    for paper_id, entry in latest.items():
        path = resolved_path(entry.get("path"))
        if path is not None:
            by_path[path] = paper_id
        sha = entry.get("sha256")
        if isinstance(sha, str) and sha:
            by_sha.setdefault(sha, []).append(paper_id)
            recorded_shas.setdefault(id_key(paper_id), set()).add(sha)

    reserved_ids = {id_key(stem) for stem in reserved}
    # A recorded id is kept only while no earlier input, and no reserved stem,
    # has it.
    used: set[str] = set(reserved_ids)
    kept: dict[int, str] = {}
    for index, path in enumerate(files):
        resolved = resolved_path(path)
        recorded_id = by_path.get(resolved) if resolved is not None else None
        if recorded_id is None or id_key(recorded_id) in used:
            continue
        size = latest[recorded_id].get("bytes")
        if isinstance(size, int) and _size(path) not in (None, size):
            continue
        kept[index] = recorded_id
        used.add(id_key(recorded_id))

    # A moved file, or one named another way (another cwd, a manifest's
    # ``..``), keeps its id by content. Only a file of a size an unclaimed id
    # was recorded with, or whose stem an id was recorded under, is worth
    # reading for that.
    shas: dict[int, str | None] = {}
    sizes: set[int] = set()
    for paper_id, entry in latest.items():
        size = entry.get("bytes")
        if id_key(paper_id) not in used and isinstance(size, int) and entry.get("sha256"):
            sizes.add(size)
    for index, path in enumerate(files):
        if index in kept:
            continue
        if id_key(path.stem) not in recorded_shas and (not sizes or _size(path) not in sizes):
            continue
        shas[index] = sha = _sha256(path)
        for recorded_id in by_sha.get(sha or "", ()):
            if id_key(recorded_id) not in used:
                kept[index] = recorded_id
                used.add(id_key(recorded_id))
                break

    by_stem: dict[str, int] = {}
    for path in files:
        by_stem[id_key(path.stem)] = by_stem.get(id_key(path.stem), 0) + 1

    def taken(candidate: str, sha: str | None) -> bool:
        key = id_key(candidate)
        # Recorded for other bytes; a line without a sha256 cannot tell.
        elsewhere = bool(recorded_shas.get(key, set()) - {sha})
        return key in used or elsewhere or _is_device_name(candidate)

    items: list[BatchItem] = []
    for index, path in enumerate(files):
        stem = path.stem
        paper_id = kept.get(index)
        if paper_id is None:
            key = id_key(stem)
            collides = by_stem[key] > 1 or _nbytes(stem) > MAX_ID_BYTES
            if index not in shas and (collides or taken(stem, None)):
                shas[index] = _sha256(path)
            sha = shas.get(index)
            collides = collides or taken(stem, sha)
            paper_id = _with_suffix(stem, f"-{sha[:8]}" if collides and sha else "")
            base = paper_id
            ordinal = 2
            while taken(paper_id, sha):
                paper_id = _with_suffix(base, f"-{ordinal}")
                ordinal += 1
            used.add(id_key(paper_id))
        items.append(BatchItem(path=path, paper_id=paper_id, stem=stem, sha256=shas.get(index)))
    return items
