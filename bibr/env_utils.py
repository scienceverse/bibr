"""Shared .env file read/write helpers."""

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


def _strip_surrounding_quotes(value: str) -> str:
    """Strip a single matching pair of surrounding ``'`` or ``"`` from *value*.

    For double-quoted values, also unescape ``\\\\`` and ``\\"`` so the result
    round-trips with :func:`_format_env_value`.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    return value


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


def parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            result[key.strip()] = _strip_surrounding_quotes(value.strip())
    return result


def merge_env(path: Path, new_vars: dict[str, str]) -> None:
    """Merge *new_vars* into an existing ``.env``, preserving unknown keys.

    Comments and blank lines retain their original positions. Existing keys
    have their values overwritten in place; new keys are appended at the
    end. This keeps any "# Section header / KEY=val" structure intact across
    re-runs of `bibr setup`.
    """
    remaining = dict(new_vars)
    out_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue
        key, _, _ = stripped.partition("=")
        key = key.strip()
        if key in remaining:
            out_lines.append(f"{key}={_format_env_value(remaining.pop(key))}")
        else:
            out_lines.append(line)

    if remaining:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        for k, v in remaining.items():
            out_lines.append(f"{k}={_format_env_value(v)}")

    out_lines.append("")
    path.write_text("\n".join(out_lines), encoding="utf-8")
