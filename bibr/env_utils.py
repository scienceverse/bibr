"""Shared .env file read/write helpers."""

import os
from pathlib import Path


def read_dotenv(path: Path, *, encoding: str = "utf-8") -> dict[str, str | None]:
    """Read dotenv syntax without expanding literal ``${VAR}`` values."""
    from dotenv import dotenv_values

    return dict(dotenv_values(path, encoding=encoding, interpolate=False))


# Characters that force the value to be double-quoted so pydantic-settings
# (and python-dotenv) read it back verbatim. ``#`` would otherwise start an
# inline comment; whitespace and quote chars likewise trigger surprising
# tokenisation. ``\\`` and ``$`` need escaping to survive shell-style expansion.
_NEEDS_QUOTING = set(" \t#\"'\\$\n\r")


def _format_env_value(value: str) -> str:
    """Quote *value* for ``.env`` if it contains characters that would be
    misinterpreted by pydantic-settings / python-dotenv (``#``, whitespace,
    ``"``, ``\\``)."""
    if not value:
        return value
    if not any(c in _NEEDS_QUOTING for c in value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_env_text(path: Path, text: str) -> None:
    """Write *text* to the ``.env`` at *path*, owner-readable only when new.

    A ``.env`` holds API keys, so a new file is created 0600 whatever the
    umask, the way ``bibr config set`` (python-dotenv) creates one. An existing
    file is rewritten in place, as before: it keeps its mode, owner and hard
    links, and works in a directory the user cannot write and as a single-file
    bind mount, where a rename over it would fail. A symlinked ``.env`` is
    followed.
    """
    target = Path(os.path.realpath(path))
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        target.write_text(text, encoding="utf-8")
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def parse_env(path: Path) -> dict[str, str]:
    """Read *path* with the same parser the runtime loads it with.

    ``preset save``/``diff``/``show`` must see exactly what ``Settings`` will
    read: python-dotenv drops inline ``# comments`` and the ``export`` prefix
    and unescapes quotes, while the previous hand-rolled split kept them and
    ``preset use`` then wrote the comment back into ``.env`` as the value.
    """
    return {k: v for k, v in read_dotenv(path).items() if v is not None}


def merge_env(path: Path, new_vars: dict[str, str]) -> None:
    """Merge *new_vars* into an existing ``.env``, preserving unknown keys.

    Comments and blank lines retain their original positions. Every
    occurrence of a merged key is rewritten in place and later duplicates
    dropped, so the value the runtime reads (python-dotenv resolves
    duplicates last-wins) is the new one. A merged key spelled with an
    ``export `` prefix is rewritten to the canonical ``KEY=value`` form,
    which dotenv reads identically. New keys are appended at the end. This
    keeps any "# Section header / KEY=val" structure intact across re-runs
    of `bibr setup`.
    """
    remaining = dict(new_vars)
    written: set[str] = set()
    out_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue
        key, _, _ = stripped.partition("=")
        key = key.strip()
        lookup = key
        # Match a dotenv ``export KEY=...`` line to its bare key, the way the
        # runtime parser does.
        if lookup.startswith("export "):
            lookup = lookup[len("export ") :].strip()
        if lookup in remaining and lookup not in written:
            out_lines.append(f"{lookup}={_format_env_value(remaining[lookup])}")
            written.add(lookup)
        elif lookup in written:
            # A stale duplicate of a key already rewritten above: drop it so
            # the old value cannot keep winning under last-wins resolution.
            continue
        else:
            out_lines.append(line)

    rest = {k: v for k, v in remaining.items() if k not in written}
    if rest:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        for k, v in rest.items():
            out_lines.append(f"{k}={_format_env_value(v)}")

    out_lines.append("")
    write_env_text(path, "\n".join(out_lines))
