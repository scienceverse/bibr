"""Walk the pydantic-settings model into a flat list of documented settings.

Single source for everything that needs to enumerate bibr's configuration
surface: the generated settings reference (scripts/docs_ref_core.py) and the
``bibr config`` subcommand. Import cost is just ``bibr.config``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import AliasChoices
from pydantic_settings import BaseSettings

from bibr.config import GlobalSettings

# Anchored at the end of the env name: real credential fields are always
# named "..._key" / "..._token" / "..._password" / "..._secret" (e.g.
# LLM_API_KEY, REDIS_PASSWORD). A bare substring match on "TOKEN" would also
# catch tuning knobs like LLM_MAX_TOKENS / OCR_VISION_MAX_TOKENS /
# REF_PARSE_MAX_TOKENS — those end in "_TOKENS" (plural), not "_TOKEN", so
# the end anchor excludes them without an explicit denylist.
_SECRET_RE = re.compile(r"(_KEY|_TOKEN|_PASSWORD|_SECRET)$", re.IGNORECASE)


@dataclass(frozen=True)
class SettingDoc:
    env_name: str
    section: str  # env prefix ("LLM_", ...) or "" for top-level fields
    field_name: str
    type_repr: str
    default_repr: str
    description: str
    aliases: tuple[str, ...] = field(default=())
    annotation: object = None

    @property
    def is_secret(self) -> bool:
        return bool(_SECRET_RE.search(self.env_name))


def _type_repr(annotation) -> str:
    if annotation is None:
        return "None"
    raw = getattr(annotation, "__name__", None) or str(annotation)
    # Python 3.14 gave PEP 604 unions a ``__name__`` of "Union", which collapses
    # ``str | None`` to a token that says nothing (earlier versions have no
    # ``__name__`` there and already fall through to ``str()``). Spell the union
    # out on every version so the generated reference stays identical.
    if raw == "Union":
        raw = str(annotation)
    return raw.replace("typing.", "")


def _default_repr(finfo) -> str:
    if finfo.default_factory is not None:
        return "(computed)"
    default = finfo.default
    if isinstance(default, str):
        return f'"{default}"'
    return repr(default)


def _alias_names(finfo) -> tuple[str, ...]:
    alias = finfo.validation_alias
    if alias is None:
        return ()
    if isinstance(alias, AliasChoices):
        return tuple(str(c) for c in alias.choices)
    return (str(alias),)


def _field_doc(finfo, name: str, prefix: str) -> SettingDoc:
    aliases = _alias_names(finfo)
    env_name = aliases[0] if aliases else f"{prefix}{name}".upper()
    return SettingDoc(
        env_name=env_name,
        section=prefix,
        field_name=name,
        type_repr=_type_repr(finfo.annotation),
        default_repr=_default_repr(finfo),
        description=finfo.description or "",
        aliases=aliases[1:],
        annotation=finfo.annotation,
    )


def iter_setting_docs() -> list[SettingDoc]:
    """All settings, sections first (declaration order), then top-level fields."""
    docs: list[SettingDoc] = []
    section_fields: list[tuple[str, type[BaseSettings]]] = []
    for name, finfo in GlobalSettings.model_fields.items():
        ann = finfo.annotation
        if isinstance(ann, type) and issubclass(ann, BaseSettings):
            section_fields.append((name, ann))
    for _, model in section_fields:
        prefix = model.model_config.get("env_prefix", "")
        for fname, finfo in model.model_fields.items():
            docs.append(_field_doc(finfo, fname, prefix))
    section_attach_names = {n for n, _ in section_fields}
    for fname, finfo in GlobalSettings.model_fields.items():
        if fname in section_attach_names:
            continue
        docs.append(_field_doc(finfo, fname, ""))
    return docs
