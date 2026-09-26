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
from bibr.exceptions import ConfigurationError

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

_DEFAULT_DIR = Path.home() / ".bibr" / "presets"
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_SCHEMA_VERSION = 1

# Secret filtering: known bibr settings use the settings metadata
# (``SettingDoc.is_secret``, the end-anchored rule that already excludes the
# ``*_MAX_TOKENS`` knobs, ``JOBS_KEY_PREFIX`` and ``CORS_ALLOW_CREDENTIALS``);
# anything else in the user's ``.env`` (third-party keys presets also snapshot)
# uses a conservative substring rule so a secret that merely fails the
# end-anchor — ``AWS_SECRET_ACCESS_KEY``, ``OPENROUTER_KEY``, ``DB_PASSWD``,
# ``*_CREDENTIALS``, ``LLM_API_KEY_2`` — is still excluded from the
# shareable preset JSON. Values themselves are never inspected — too easy to
# false-positive on legitimate config strings (URLs, hashes, ...).
_ACTIVE_PRESET_KEY = "BIBR_ACTIVE_PRESET"

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")

# Unknown-key exemptions: names that contain a marker but are config, not
# credentials. Compared case-insensitively.
_NON_SECRET_SUFFIXES = ("_TOKENS", "_RPM")
_NON_SECRET_KEYS = frozenset({"JOBS_KEY_PREFIX", "CORS_ALLOW_CREDENTIALS"})

_known_secret_cache: dict[str, bool] | None = None


def _known_secrets() -> dict[str, bool]:
    """Map every known bibr env name (plus aliases) to its secret flag."""
    global _known_secret_cache
    if _known_secret_cache is None:
        from bibr.config_introspect import iter_setting_docs

        known: dict[str, bool] = {}
        for doc in iter_setting_docs():
            for spelling in (doc.env_name, *doc.aliases):
                known[spelling] = doc.is_secret
                known[spelling.upper()] = doc.is_secret
        _known_secret_cache = known
    return _known_secret_cache


class InvalidPresetError(ConfigurationError):
    """A preset name is invalid, a preset file is corrupt, or its values fail validation.

    Subclasses :class:`~bibr.exceptions.ConfigurationError` (a ``BibrError``)
    so every ``bibr preset`` / ``chew --preset`` misuse renders as a clean
    error naming the preset or setting instead of a Python traceback.
    """


def effective_env_file() -> Path:
    """The ``.env`` file whose values are in effect: what presets read and write.

    Settings merge ``~/.bibr/.env`` and then ``./.env`` (or the files in
    ``BIBR_ENV_FILE``), later files overriding earlier ones, so the last one
    that exists is the file whose values win. With none present it is where
    the chain would look last; ``./.env`` when the chain is empty. ``bibr
    preset`` and the demo's preset picker both use it.
    """
    from bibr.config import _default_env_files

    chain = _default_env_files()
    existing = [path for path in chain if path.is_file()]
    if existing:
        return existing[-1].absolute()
    return (chain[-1] if chain else Path(".env")).absolute()


def is_secret_key(name: str) -> bool:
    """Return True if *name* should be excluded from a preset snapshot."""
    known = _known_secrets()
    hit = known.get(name)
    if hit is None:
        hit = known.get(name.upper())
    if hit is not None:
        return hit
    upper = name.upper()
    if upper in _NON_SECRET_KEYS:
        return False
    if upper.endswith(_NON_SECRET_SUFFIXES):
        return False
    return any(marker in upper for marker in _SECRET_MARKERS)


def redact_value(name: str, value: str) -> str:
    """Mask *value* if *name* looks like a secret.

    Non-secrets are returned verbatim. Secrets are shown as
    ``XXXX…YY`` — the first 4 and last 2 characters around an ellipsis,
    enough to recognize the credential without exposing it. Values shorter
    than 8 characters are masked entirely.
    """
    if not is_secret_key(name):
        return value
    if len(value) < 8:
        return "***"
    return f"{value[:4]}…{value[-2:]}"


def _setting_lookup() -> dict[str, tuple[str | None, str]]:
    """Map every known env-var name to ``(section_attr, field_name)``.

    Derived live from the settings models (each section's
    ``model_config['env_prefix']``, like ``config._env_prefix_for_model``)
    plus ``validation_alias`` spellings, so it cannot drift as sections are
    added — the previous hand-kept ``_PREFIX_TO_SECTION`` table was missing
    the ``ROR_`` and ``MCP_`` prefixes and aliases such as
    ``OCR_SGLANG_GPUS``. ``section_attr`` is None for top-level fields
    (``WTPSPLIT_MODEL``, ``OCR_BASE_URL``, ...); the field name is the
    model's attribute, not the env spelling.
    """
    from bibr.config import GlobalSettings
    from bibr.config_introspect import iter_setting_docs

    section_attrs: dict[str, str] = {}
    for attr, finfo in GlobalSettings.model_fields.items():
        ann = finfo.annotation
        if isinstance(ann, type):
            prefix = getattr(ann, "model_config", {}).get("env_prefix", "") or ""
            if prefix:
                section_attrs[prefix] = attr
    lookup: dict[str, tuple[str | None, str]] = {}
    for doc in iter_setting_docs():
        section_attr = section_attrs.get(doc.section)
        for spelling in (doc.env_name, *doc.aliases):
            lookup[spelling] = (section_attr, doc.field_name)
            lookup[spelling.upper()] = (section_attr, doc.field_name)
    return lookup


def _resolve_setting(key: str) -> tuple[str | None, str] | None:
    """Map an env-var name to ``(section_attr, field_name)`` on GlobalSettings.

    Returns ``None`` for unknown keys. Top-level (un-clustered) keys yield
    ``(None, field_name)`` — this also covers keys like ``OCR_BASE_URL``
    that share a section's prefix but live on the top-level model.
    """
    lookup = _setting_lookup()
    hit = lookup.get(key)
    return hit if hit is not None else lookup.get(key.upper())


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
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidPresetError(
                f"Preset {name!r} at {path} is not valid JSON: {exc}. "
                "Fix or delete the file, then retry."
            ) from exc
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
        """Write the preset's settings into *env_path* and tag it active.

        The preset's values are validated through the real settings parser
        first (see :func:`bibr.config.validate_env_overrides`), so a preset
        holding an invalid value fails naming the preset file instead of
        corrupting ``.env`` for every later run. Ownership is decided by
        difference, not by name: the preset is rejected when it adds any
        failure beyond what the current environment already reports (see
        :func:`bibr.config.baseline_problems`), which also catches
        model-level errors, alias spellings and lowercase keys that never
        mention the preset's own key.
        """
        data = self.load(name)
        if any(key != _ACTIVE_PRESET_KEY for key in data):
            from bibr.config import baseline_problems, validate_env_overrides

            candidate = {key: data[key] for key in data if key != _ACTIVE_PRESET_KEY}
            try:
                validate_env_overrides(candidate)
            except ConfigurationError as exc:
                problems = list(getattr(exc, "problems", None) or [str(exc)])
                new = [p for p in problems if p not in baseline_problems()]
                if new:
                    path = self._path(name)
                    prefixed = [f"Preset {name!r}: {problem}" for problem in new]
                    body = "\n".join(f"  - {problem}" for problem in prefixed)
                    raise InvalidPresetError(
                        f"Preset {name!r} at {path} is invalid:\n{body}\nFix the preset file.",
                        problems=prefixed,
                    ) from exc
                # Only unrelated settings are invalid — the preset adds no new
                # failure, so still write it.
        data[_ACTIVE_PRESET_KEY] = name
        merge_env(env_path, data)

    def apply_to_settings(self, name: str, settings: GlobalSettings) -> list[str]:
        """Mutate *settings* in place using the preset's keys.

        Returns the list of keys that could not be applied (no matching
        setting). The caller may log them.

        The preset is applied through the real settings parser, not raw
        ``setattr``: its keys are overlaid on the current environment and a
        fresh ``GlobalSettings`` is built (see
        :func:`bibr.config.validate_env_overrides`), so list/dict parsing,
        ``Literal`` and constraint checks, validators, aliases and the
        auto-tune model_validators all run exactly as they do for ``.env``.
        The touched sections plus any section an auto-tune changed relative
        to the no-override baseline (e.g. ``crossref`` when the preset sets
        ``BIBR_RESOLVER_URL``) are copied back onto *settings* — the way
        ``setup_wizard._reload_settings_in_place`` copies sections — so
        unrelated in-memory values are preserved. This
        also works on the ``bibr.config.Settings`` proxy, which has no
        ``model_fields`` of its own. A preset value that fails validation
        raises :class:`InvalidPresetError` naming the preset file and the
        setting; nothing is applied in that case.

        Dispatch order: section by live env-prefix mapping → top-level
        ``GlobalSettings`` field. The fallback catches keys like
        ``OCR_BASE_URL`` that share a section's prefix but actually live
        on the top-level model (legacy, predates the nested-settings
        refactor — see ``bibr/config.py``).
        """
        from bibr.config import validate_env_overrides

        data = self.load(name)
        targets: dict[str, tuple[str | None, str]] = {}
        unknown: list[str] = []
        for key in data:
            if key == _ACTIVE_PRESET_KEY:
                continue
            resolved = _resolve_setting(key)
            if resolved is None:
                unknown.append(key)
                continue
            targets[key] = resolved
        if not targets:
            return unknown
        try:
            fresh = validate_env_overrides({key: data[key] for key in targets})
        except ConfigurationError as exc:
            path = self._path(name)
            problems = list(getattr(exc, "problems", None) or [str(exc)])
            prefixed = [f"Preset {name!r}: {problem}" for problem in problems]
            body = "\n".join(f"  - {problem}" for problem in prefixed)
            raise InvalidPresetError(
                f"Preset {name!r} at {path} is invalid:\n{body}\nFix the preset file.",
                problems=prefixed,
            ) from exc
        touched_sections = {section for section, _ in targets.values() if section is not None}
        # Auto-tune model_validators can write sections the preset did not
        # touch (``BIBR_RESOLVER_URL`` raises ``crossref.enrich_concurrency``
        # 12 → 16): also copy every section the preset changed relative to
        # the no-override baseline, so a preset behaves like the same lines
        # in ``.env``. Falls back to the known cross-section edge when the
        # baseline itself cannot build (unrelated settings invalid).
        try:
            baseline = validate_env_overrides({})
        except ConfigurationError:
            baseline = None
        if baseline is not None:
            from bibr.config import GlobalSettings

            for attr in GlobalSettings.model_fields:
                if attr in touched_sections:
                    continue
                try:
                    current = getattr(fresh, attr)
                except AttributeError:
                    continue
                try:
                    other = getattr(baseline, attr)
                except AttributeError:
                    continue
                if current != other:
                    touched_sections.add(attr)
        elif "resolver" in touched_sections:
            touched_sections.add("crossref")
        for section in touched_sections:
            try:
                setattr(settings, section, getattr(fresh, section))
            except AttributeError:
                continue
        for target_section, target_field in targets.values():
            if target_section is not None:
                continue
            setattr(settings, target_field, getattr(fresh, target_field))
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
