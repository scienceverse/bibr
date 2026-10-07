"""Input file collection and output path resolution for the ``chew`` command."""

import os
import sys
import tempfile
from pathlib import Path


def _resolve_single_output_path(output_path: Path, paper_path: Path) -> Path:
    """Resolve the file to write a single-paper result to.

    If ``output_path`` is an existing directory (e.g. user passed ``-o foo/``),
    write inside as ``<paper_stem>.json`` instead of trying to overwrite the
    directory itself.
    """
    if output_path.is_dir():
        return output_path / f"{paper_path.stem}.json"
    return output_path


_PDF_MAGIC = b"%PDF-"
_DOCX_MAGIC = b"PK\x03\x04"  # ZIP container — DOCX/PPTX/XLSX all share this prefix


def _zip_payload_suffix(data: bytes) -> str:
    """Distinguish supported ZIP-based stdin payloads."""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if "mimetype" in names:
                mimetype = zf.read("mimetype").decode("utf-8", errors="ignore").strip()
                if mimetype == "application/epub+zip":
                    return ".epub"
            if "word/document.xml" in names:
                return ".docx"
    except zipfile.BadZipFile:
        pass
    return ".docx"


def _suffix_for_stdin_payload(data: bytes) -> str:
    """Return a supported suffix based on the magic bytes of *data*.

    The CLI input dispatcher (``input/validate.py``) infers handling from the
    file extension, so a piped DOCX with ``.pdf`` would mis-route. Sniffing
    the first few bytes is cheap and unambiguous for the supported types.
    """
    if data[:5] == _PDF_MAGIC:
        return ".pdf"
    if data[:4] == _DOCX_MAGIC:
        return _zip_payload_suffix(data)
    head = data.lstrip()[:512]
    if head.lower().startswith((b"<!doctype html", b"<html")) or b"<body" in head.lower():
        return ".html"
    if head.startswith(b"<?xml") or head.startswith(b"<article") or b"<article" in head:
        return ".xml"
    # Fall back to .pdf so existing behaviour for unknown payloads is unchanged.
    return ".pdf"


def _collect_files(inputs: list[str]) -> tuple[list[Path], int]:
    """Resolve input arguments to a list of file paths.

    Returns ``(files, missing_count)`` — ``missing_count`` is the number of
    inputs that didn't resolve to any file (missing path / dead glob), so the
    caller can fold them into its overall error count even when other inputs
    did resolve and processing proceeds.
    """
    import atexit

    files = []
    missing_paths: list[str] = []
    empty_dirs: list[str] = []
    supported_exts = {".pdf", ".docx", ".xml", ".html", ".htm", ".epub"}
    for inp in inputs:
        # stdin support: read bytes into a temp file
        if inp == "-":
            data = sys.stdin.buffer.read()
            suffix = _suffix_for_stdin_payload(data)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
            tmp_path = Path(tmp.name)
            files.append(tmp_path)
            # Schedule deletion at interpreter exit so the temp file doesn't
            # leak — ``delete=False`` was needed because the path has to
            # outlive the ``with`` block, but nothing else cleans it up.
            atexit.register(lambda p=tmp_path: p.unlink(missing_ok=True))  # type: ignore[misc]
            continue

        p = Path(inp)
        if p.is_dir():
            dir_files = [
                child
                for child in sorted(p.iterdir())
                if child.suffix.lower() in supported_exts and child.is_file()
            ]
            if dir_files:
                files.extend(dir_files)
            else:
                empty_dirs.append(inp)
        elif p.is_file():
            files.append(p)
        else:
            # Try glob pattern
            from glob import glob

            matches = glob(inp)
            if matches:
                for m in sorted(matches):
                    mp = Path(m)
                    if mp.is_file() and mp.suffix.lower() in supported_exts:
                        files.append(mp)
            else:
                missing_paths.append(inp)

    if not files:
        from rich.console import Console
        from rich.markup import escape

        from bibr.local.cli import ui

        console = Console(stderr=True)
        if missing_paths:
            ui.error(
                console,
                f"Path not found: {escape(', '.join(missing_paths))}",
                hint="Check for typos, or pass an existing PDF/DOCX/XML/HTML/ePub "
                "file or directory.",
            )
        elif empty_dirs:
            ui.error(
                console,
                "No .pdf, .docx, .xml, .html, .htm, or .epub files in: "
                f"{escape(', '.join(empty_dirs))}",
            )
        else:
            ui.error(
                console,
                "No supported files found in the given path.",
                hint="bibr accepts .pdf, .docx, .xml (JATS), .html/.htm, and .epub files.\n"
                "\n"
                "  bibr chew paper.pdf          # single file\n"
                "  bibr chew papers/            # directory of files",
            )
        sys.exit(1)

    if missing_paths:
        from rich.console import Console
        from rich.markup import escape

        from bibr.local.cli import ui

        console = Console(stderr=True)
        for missing in missing_paths:
            ui.error(console, f"{escape(missing)}: file not found")

    return files, len(missing_paths)


def _prepare_output_path(raw: str | None, *, is_batch: bool) -> Path | None:
    """Resolve ``-o/--output`` and ensure the directory exists.

    Batch mode always writes ``<dir>/<stem>.json``, so the directory is
    always created. Never gate on ``Path.suffix`` — dotted directory names
    (``Qwen3.5-4B``) read as file suffixes and used to skip creation,
    crashing the result write.

    Single-file mode writes directly to ``output_path`` (or, if it resolves
    to an existing directory, ``<stem>.json`` inside it — see
    ``_resolve_single_output_path``), so its *parent* directory must exist
    too — e.g. ``-o deep/new/dir/out.json`` must create ``deep/new/dir``
    instead of crashing with ``FileNotFoundError`` at write time.
    """
    if raw is None:
        return None
    output_path = Path(raw)
    if is_batch:
        output_path.mkdir(parents=True, exist_ok=True)
    elif raw.endswith(("/", "\\")):
        # A trailing separator names a directory even for a single file
        # (``chew paper.xml -o results/``): create it here so the existing
        # single-output resolution below writes ``<dir>/<stem>.json``
        # instead of a FILE named ``results``. An existing directory needs
        # no creation and already resolves correctly. The input is still one
        # file named directly (not batch), so ``--paper-id`` still applies to
        # it. A FILE blocking the path raises FileExistsError (an
        # OSError), which the caller turns into a clean exit 2.
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _output_path_problem(raw: str | None, *, is_batch: bool) -> str | None:
    """Why ``-o`` cannot be written, checked without creating anything.

    The ``--dry-run`` counterpart of :func:`_prepare_output_path`: the
    directory that helper creates (or that holds the output file) must be a
    directory, or be creatable inside one, and be writable.
    """
    if raw is None:
        return None
    output_path = Path(raw)
    if is_batch or raw.endswith(("/", "\\")) or output_path.is_dir():
        directory = output_path
    else:
        directory = output_path.parent
    existing = directory
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    if not existing.is_dir():
        return f"Cannot write output {raw!r}: {existing} is not a directory"
    if not os.access(existing, os.W_OK):
        return f"Cannot write output {raw!r}: {existing} is not writable"
    return None


def _batch_output_names(stem: str) -> tuple[str, ...]:
    """The files batch ``chew -o`` writes for an input with *stem*: its export
    and the sidecars ``LocalArtifactSink`` puts beside it."""
    export = f"{stem}.json"
    return (export, f"{stem}.core.json", f"{export}.receipt.json", f"{export}.enrichment.json")


def _find_stem_collisions(files: list[Path], *, sidecars: bool = False) -> dict[str, list[Path]]:
    """Group batch input files by the output files they write, returning one
    name per group of 2+ files that would write the same file.

    Batch output writes ``<dir>/<stem>.json``, so files from different
    directories that share a stem (``a/x.pdf`` + ``b/x.pdf``) would silently
    overwrite one result with the other. With *sidecars* (a run that writes
    files, so ``LocalArtifactSink`` puts them beside each export) those count
    too: the core ``x.core.json`` of ``x.pdf`` is the export of ``x.core.pdf``.

    Names are compared case-insensitively: on the default macOS/Windows
    filesystems ``x.json`` and ``X.json`` are the same directory entry, so
    ``a/x.pdf`` + ``b/X.PDF`` collide just as surely as an exact-case match.
    """
    by_name: dict[str, list[Path]] = {}
    for f in files:
        names = _batch_output_names(f.stem) if sidecars else (f"{f.stem}.json",)
        for name in names:
            by_name.setdefault(name.casefold(), []).append(f)
    # Files sharing a stem share every sidecar name too: name the export only.
    collisions: dict[str, list[Path]] = {}
    for name, paths in by_name.items():
        if len(paths) > 1 and paths not in collisions.values():
            collisions[name] = paths
    return collisions
