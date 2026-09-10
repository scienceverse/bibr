"""Named configuration presets stored as JSON in ``~/.bibr/presets/``.

A preset is a snapshot of behavior-shaping ``.env`` keys (LLM provider,
OCR backend, rate limits, ...) — **not** secrets. ``snapshot_from_env``
filters out anything that looks like an API key, token, or password so
preset files are safe to share, commit to a dotfiles repo, etc.

JSON schema (v1)::

    {
      "schema_version": 1,
      "settings": {"LLM_PROVIDER": "google", ...}
    }

Old presets without ``schema_version`` are read as a flat dict (v0); new
writes always emit v1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from bibr.env_utils import merge_env

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

_DEFAULT_DIR = Path.home() / ".bibr" / "presets"
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_SCHEMA_VERSION = 1

# Keys that look like secrets are filtered out of snapshots. We intentionally
# match conservatively: any key whose name contains one of these markers is
# treated as a secret. Values themselves are never inspected — too easy to
# false-positive on legitimate config strings (URLs, hashes, ...).
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")

# Always-snapshot exception list: keys whose name happens to contain a marker
# but which are config, not credentials. Compared case-insensitively.
_NON_SECRET_OVERRIDES = frozenset(
    {
        # CrossRef rate-limit knob — has _LIMIT_, no marker hit; here for clarity.
        "CROSSREF_RATE_LIMIT_RPM",
        "LLM_RATE_LIMIT_RPM",
    }
)

_ACTIVE_PRESET_KEY = "BIBR_ACTIVE_PRESET"


class InvalidPresetError(Exception):
    pass


def is_secret_key(name: str) -> bool:
    """Return True if *name* should be excluded from a preset snapshot."""
    upper = name.upper()
    if upper in _NON_SECRET_OVERRIDES:
        return False
    return any(marker in upper for marker in _SECRET_MARKERS)


def redact_value(name: str, value: str, *, max_plain: int = 12) -> str:
    """Mask *value* if *name* looks like a secret.

    Short non-secrets are returned verbatim. Secrets are shown as
    ``XXXX…YY`` — the first 4 and last 2 characters around an ellipsis,
    enough to recognize the credential without exposing it. Values shorter
    than 8 characters are masked entirely.
    """
    if not is_secret_key(name) and len(value) <= max(max_plain, 1):
        return value
    if not is_secret_key(name):
        return value
    if len(value) < 8:
        return "***"
    return f"{value[:4]}…{value[-2:]}"


# Map env-var prefix → ``Settings`` attribute name. Order matters: longer
# prefixes (``OCR_VISION_``) must be checked before shorter ones (``OCR_``)
# when dispatching keys to sections.
_PREFIX_TO_SECTION: tuple[tuple[str, str], ...] = (
    ("BIBR_RESOLVER_", "resolver"),
    ("OCR_VISION_", "ocr_vision"),
    ("VLLM_MLX_", "vllm_mlx"),
    ("RAPID_MLX_", "rapid_mlx"),
    ("CROSSREF_", "crossref"),
    ("PIPELINE_", "pipeline"),
    ("LAYOUT_", "layout"),
    ("CACHE_", "cache"),
    ("REDIS_", "redis"),
    ("CORS_", "cors"),
    ("AUTH_", "auth"),
    ("METER_", "metering"),
    ("JOBS_", "jobs"),
    ("LLM_", "llm"),
    ("OCR_", "ocr"),
    ("FIG_", "fig"),
    ("CB_", "cb"),
    ("ML_", "ml"),
)


def _coerce_value(field_type: type, raw: str):
    """Coerce a string env value to the type the pydantic field expects.

    Mirrors how pydantic-settings parses ``.env`` strings, but for the
    in-memory setattr path used by ``apply_to_settings``. We handle the
    common scalar shapes; anything else is left as a string and pydantic
    validation will error out cleanly.
    """
    # ``str | None`` and similar unions: descend into the non-None arg.
    origin_args = getattr(field_type, "__args__", ())
    if origin_args:
        for arg in origin_args:
            if arg is not type(None):
                field_type = arg
                break
    if field_type is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


class PresetManager:
    def __init__(self, presets_dir: Path = _DEFAULT_DIR) -> None:
        self._dir = presets_dir

    # ------------------------------------------------------------------
    # Name / path helpers
    # ------------------------------------------------------------------

    def _validate_name(self, name: str) -> None:
        if not name or not _NAME_RE.match(name):
            raise InvalidPresetError(
                f"Invalid preset name {name!r}. "
                "Use alphanumeric characters, hyphens, dots, and underscores."
            )

    def _path(self, name: str) -> Path:
        return self._dir / f"{name}.json"

    @property
    def directory(self) -> Path:
        """Where this manager reads/writes preset files."""
        return self._dir

    # ------------------------------------------------------------------
    # JSON schema (v0 plain dict; v1 wraps in ``{"schema_version", "settings"}``)
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(data: dict[str, str]) -> dict:
        return {"schema_version": _SCHEMA_VERSION, "settings": dict(data)}

    @staticmethod
    def _unwrap(raw: dict) -> dict[str, str]:
        if isinstance(raw, dict) and "schema_version" in raw and "settings" in raw:
            return dict(raw["settings"])
        # v0 (legacy): the JSON itself is the settings dict.
        return dict(raw)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, name: str, data: dict[str, str]) -> Path:
        self._validate_name(name)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(name)
        path.write_text(json.dumps(self._wrap(data), indent=2), encoding="utf-8")
        return path

    def load(self, name: str) -> dict[str, str]:
        self._validate_name(name)
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"Preset {name!r} not found at {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._unwrap(raw)

    def delete(self, name: str) -> None:
        self._validate_name(name)
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"Preset {name!r} not found at {path}")
        path.unlink()

    def list_presets(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def exists(self, name: str) -> bool:
        self._validate_name(name)
        return self._path(name).exists()

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, name: str, env_path: Path) -> None:
        """Write the preset's settings into *env_path* and tag it active."""
        data = self.load(name)
        data[_ACTIVE_PRESET_KEY] = name
        merge_env(env_path, data)

    def apply_to_settings(self, name: str, settings: GlobalSettings) -> list[str]:
        """Mutate *settings* in place using the preset's keys.

        Returns the list of keys that could not be applied (no matching
        section field and no matching top-level field). The caller may
        log them; pydantic still re-validates on next read so ill-typed
        values raise clearly.

        This is the in-memory counterpart to :meth:`apply` — used by
        ``bibr chew --preset NAME`` so that a single run can adopt preset
        settings without writing to ``.env``.

        Dispatch order: section by env-prefix → top-level
        ``GlobalSettings`` field. The fallback catches keys like
        ``OCR_BASE_URL`` that share a section's prefix but actually live
        on the top-level model (legacy, predates the nested-settings
        refactor — see ``bibr/config.py``).
        """
        data = self.load(name)
        unknown: list[str] = []
        for key, value in data.items():
            if key == _ACTIVE_PRESET_KEY:
                continue
            section_attr, field_name = _resolve_setting(key)
            if section_attr is not None:
                section = getattr(settings, section_attr, None)
                if section is not None and hasattr(section, field_name):
                    field = type(section).model_fields.get(field_name)
                    coerced = _coerce_value(field.annotation, value) if field else value
                    setattr(section, field_name, coerced)
                    continue
                # Section matched the prefix but doesn't expose the field —
                # fall through to the top-level lookup.
            if hasattr(settings, key):
                field = type(settings).model_fields.get(key)
                coerced = _coerce_value(field.annotation, value) if field else value
                setattr(settings, key, coerced)
                continue
            unknown.append(key)
        return unknown

    # ------------------------------------------------------------------
    # Active preset bookkeeping
    # ------------------------------------------------------------------

    def get_active(self, env_path: Path) -> str | None:
        if not env_path.exists():
            return None
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{_ACTIVE_PRESET_KEY}="):
                value = stripped.split("=", 1)[1].strip()
                return value if value else None
        return None

    def deactivate(self, env_path: Path) -> bool:
        """Remove the ``BIBR_ACTIVE_PRESET`` line from *env_path*.

        Returns True if a line was removed, False if no marker was present.
        Other settings stay untouched — deactivation only clears the tag.
        """
        if not env_path.exists():
            return False
        original = env_path.read_text(encoding="utf-8")
        kept_lines: list[str] = []
        removed = False
        for line in original.splitlines():
            if line.strip().startswith(f"{_ACTIVE_PRESET_KEY}="):
                removed = True
                continue
            kept_lines.append(line)
        if not removed:
            return False
        # Preserve trailing newline behaviour.
        new_text = "\n".join(kept_lines)
        if original.endswith("\n") and not new_text.endswith("\n"):
            new_text += "\n"
        env_path.write_text(new_text, encoding="utf-8")
        return True

    # ------------------------------------------------------------------
    # Snapshot + diff
    # ------------------------------------------------------------------

    def snapshot_from_env(self, env_path: Path, *, include_secrets: bool = False) -> dict[str, str]:
        """Return ``{KEY: VALUE}`` for *env_path*, ready to be saved as a preset.

        Drops the ``BIBR_ACTIVE_PRESET`` marker and (by default) any key
        whose name looks like a credential. Pass ``include_secrets=True``
        to keep them — never the right call for files you intend to share.
        """
        from bibr.env_utils import parse_env

        out: dict[str, str] = {}
        for k, v in parse_env(env_path).items():
            if k == _ACTIVE_PRESET_KEY:
                continue
            if not include_secrets and is_secret_key(k):
                continue
            out[k] = v
        return out

    def diff_against(
        self, name: str, env_dict: dict[str, str]
    ) -> tuple[dict[str, tuple[str, str]], dict[str, str], dict[str, str]]:
        """Compare a stored preset against an env-style dict.

        Returns ``(changed, only_in_preset, only_in_env)``:

        - ``changed``         — keys present in both, value differs
                                ``{KEY: (env_value, preset_value)}``
        - ``only_in_preset``  — keys in the preset but not in *env_dict*
        - ``only_in_env``     — keys in *env_dict* but not in the preset
                                (excluding ``BIBR_ACTIVE_PRESET`` and any
                                key that was filtered out as a secret —
                                those would generate noise)
        """
        preset = self.load(name)
        env = {k: v for k, v in env_dict.items() if k != _ACTIVE_PRESET_KEY}
        changed: dict[str, tuple[str, str]] = {}
        only_in_preset: dict[str, str] = {}
        only_in_env: dict[str, str] = {}
        for k in set(preset) | set(env):
            if k in preset and k in env:
                if preset[k] != env[k]:
                    changed[k] = (env[k], preset[k])
            elif k in preset:
                only_in_preset[k] = preset[k]
            else:
                # ``only_in_env`` excludes secrets so the diff doesn't
                # alarmingly suggest "your env has a key the preset is
                # missing!" for credentials that are intentionally absent
                # from preset files.
                if not is_secret_key(k):
                    only_in_env[k] = env[k]
        return changed, only_in_preset, only_in_env


def _resolve_setting(key: str) -> tuple[str | None, str]:
    """Map an env-var name to ``(section_attr, field_name)`` on GlobalSettings.

    Returns ``(None, key)`` for top-level (un-clustered) keys; the caller
    should ``setattr(settings, key, value)`` directly.
    """
    for prefix, section_attr in _PREFIX_TO_SECTION:
        if key.startswith(prefix):
            return section_attr, key[len(prefix) :].lower()
    return None, key
