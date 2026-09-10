"""Render bibr's reference docs (settings, CLI, schema) as markdown strings.

Pure functions over live code — imported by scripts/gen_docs_reference.py at
``mkdocs build`` time and by tests. Keep this module free of mkdocs imports.
"""

from __future__ import annotations

import argparse
import typing
from types import UnionType

from pydantic import BaseModel

from bibr.config_introspect import SettingDoc, iter_setting_docs

_SECTION_TITLES = {
    "": "Top-level",
    "LLM_": "LLM",
    "OCR_": "OCR",
    "OCR_VISION_": "OCR vision",
    "FIG_": "Figure extraction",
    "LAYOUT_": "Layout detection",
    "CROSSREF_": "Crossref enrichment",
    "BIBR_RESOLVER_": "bibr-resolver",
    "CACHE_": "Cache",
    "CB_": "Circuit breaker",
    "CORS_": "CORS",
    "REDIS_": "Redis",
    "ML_": "ML models",
    "VLLM_MLX_": "vLLM-MLX",
    "RAPID_MLX_": "Rapid-MLX",
    "PIPELINE_": "Pipeline",
    "AUTH_": "HTTP auth",
    "JOBS_": "Serve jobs",
    "METER_": "Metering",
    "MCP_": "MCP endpoint (serve)",
}

_HEADER = (
    "<!-- GENERATED at mkdocs build time by scripts/gen_docs_reference.py — do not edit. -->\n"
)


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _readable_type(type_repr: str) -> str:
    """Cosmetically relabel type reprs that collapse to a bare typing token.

    ``config_introspect._type_repr()`` picks ``annotation.__name__`` first,
    which for old-style ``typing.Literal[...]`` / ``typing.Optional[...]``
    annotations returns just ``"Literal"`` / ``"Optional"`` — the choices are
    lost (they're still spelled out in the Description column). PEP 604
    unions (``str | None``) aren't affected; those already render in full.
    """
    if type_repr == "Literal":
        return "enum"
    if type_repr == "Optional":
        return "enum, optional"
    return type_repr


def _settings_table(docs: list[SettingDoc]) -> str:
    lines = ["| Variable | Type | Default | Description |", "|---|---|---|---|"]
    for d in docs:
        desc = _md_escape(d.description)
        if d.aliases:
            desc += f" (aliases: {', '.join(f'`{a}`' for a in d.aliases)})"
        # Type cell is a code span (backtick-wrapped): pipes inside a code
        # span don't need escaping for Python-Markdown's `tables` extension
        # (code-span-aware row splitter) and _md_escape would render a
        # visible literal backslash instead. Only the Description cell
        # (not inside backticks) needs pipe-escaping.
        type_cell = _readable_type(d.type_repr)
        lines.append(f"| `{d.env_name}` | `{type_cell}` | `{d.default_repr}` | {desc} |")
    return "\n".join(lines)


def render_settings_md() -> str:
    docs = iter_setting_docs()
    by_section: dict[str, list[SettingDoc]] = {}
    for d in docs:
        by_section.setdefault(d.section, []).append(d)
    parts = [
        _HEADER,
        "# Settings reference\n",
        "Configuration loads `~/.bibr/.env`, then the working directory's `.env`; "
        "environment variables override both. `BIBR_ENV_FILE` replaces this file "
        "chain (use `:` between paths on Unix, `;` on Windows, or an empty value "
        "to disable dotenv loading). Per-run options can override these defaults. "
        "Sections are namespaced by prefix; there is no global `BIBR_` prefix. "
        "The tables below show shipped defaults, not the build machine's settings. "
        "Use `bibr config show --sources` to see the live "
        "values and where each one comes from, and `bibr config path` to locate "
        "your `.env` file. See the [configuration guide](../guides/configuration.md) "
        "for presets and examples.\n",
    ]
    # Sections in walker order, "" (top-level) last for readability
    ordered = [s for s in by_section if s] + [s for s in by_section if not s]
    for section in ordered:
        title = _SECTION_TITLES.get(section, section.rstrip("_") or "Other")
        suffix = f" (`{section}*`)" if section else ""
        parts.append(f"## {title}{suffix}\n")
        parts.append(_settings_table(by_section[section]) + "\n")
    return "\n".join(parts)


def _expand_help(action: argparse.Action) -> str:
    help_text = action.help or ""
    try:
        return help_text % {"default": action.default, "prog": ""}
    except (KeyError, TypeError, ValueError):
        return help_text


def _options_table(parser: argparse.ArgumentParser) -> str | None:
    """Render the flag table, or ``None`` if the parser has nothing to document.

    A parser whose only actions are ``-h/--help`` and/or subparser routing
    (e.g. ``bibr setup``, ``bibr doctor``) has no rows to show — an
    empty header-only table reads as a rendering bug, not "no options".
    """
    lines = ["| Flag | Default | Description |", "|---|---|---|"]
    has_rows = False
    for action in parser._actions:  # noqa: SLF001 — argparse has no public tree API
        if isinstance(action, argparse._SubParsersAction) or action.dest == "help":
            continue
        has_rows = True
        flags = ", ".join(f"`{s}`" for s in action.option_strings) or f"`{action.dest}`"
        default = "" if action.default in (None, argparse.SUPPRESS) else f"`{action.default!r}`"
        lines.append(f"| {flags} | {default} | {_md_escape(_expand_help(action))} |")
    return "\n".join(lines) if has_rows else None


def _walk_commands(
    parser: argparse.ArgumentParser, prog: str, description: str = ""
) -> list[tuple[str, argparse.ArgumentParser, str]]:
    """Flatten the parser/subparser tree, resolving a description per command.

    Some subcommands (``bibr setup``, ``serve``, ``demo``, ``doctor``,
    ``preset list``, ``preset deactivate``, ...) are built with only
    ``help=`` and no ``description=`` (see ``bibr/local/cli.py``), so
    ``parser.description`` is ``None`` for them. argparse still records that
    ``help=`` text on the parent's ``_SubParsersAction._choices_actions``
    (each pseudo-action's ``.dest`` is the subcommand name, ``.help`` is the
    text) — use it as a description fallback so every command gets prose.
    """
    desc = parser.description or description
    out = [(prog, parser, desc)]
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            help_by_name = {a.dest: (a.help or "") for a in action._choices_actions}  # noqa: SLF001
            for name, sub in action.choices.items():
                out.extend(_walk_commands(sub, f"{prog} {name}", help_by_name.get(name, "")))
    return out


def render_cli_md() -> str:
    from bibr.demo.server import build_parser as demo_parser
    from bibr.local.cli import _build_parser

    delegated_help = {
        "bibr setup": (
            "The wizard handles its own options. Use `bibr setup --advanced` for "
            "the detailed provider/backend picker and `bibr setup --help` for "
            "its full help. See [Quickstart](../getting-started/quickstart.md).\n"
        ),
        "bibr serve": (
            "The server handles `--host` and `--port` in its own parser. "
            "Run `bibr serve --help` for their defaults. See "
            "[Deployment](../guides/deployment.md) for binding, authentication, "
            "and environment settings.\n"
        ),
    }
    parts = [
        _HEADER,
        "# CLI reference\n",
        "Run these commands as `uv run bibr …` from a source checkout. "
        "Tables are generated from the CLI parser; a blank default usually means "
        "the command resolves a value from configuration or hardware at runtime. "
        "Use `bibr chew paper.pdf --dry-run` to preview the resolved processing plan.\n",
    ]
    for prog, sub, desc in _walk_commands(_build_parser(), "bibr")[1:]:  # skip the root parser
        if prog == "bibr demo":
            sub = demo_parser()
            desc = sub.description
        parts.append(f"## {prog}\n")
        if desc:
            parts.append(desc + "\n")
        table = _options_table(sub)
        if table is not None:
            parts.append(table + "\n")
        if prog in delegated_help:
            parts.append(delegated_help[prog])
    return "\n".join(parts)


def _annotation_repr(annotation: object) -> str:
    """Render a field annotation as short, readable type syntax.

    A bare ``getattr(annotation, "__name__", None)`` collapses every
    parameterized generic to its origin name (``list[TextExport]`` ->
    ``"list"``) and falls back to ``str(annotation)`` for unions, which
    prints fully-qualified module paths (``bibr.export.models.
    ExtractionExport | None``). This recurses through ``get_origin``/
    ``get_args`` instead, preserving generic structure and stripping
    module paths from every leaf.
    """
    if annotation is type(None):
        return "None"
    origin = typing.get_origin(annotation)
    if origin is None:
        name = getattr(annotation, "__name__", None)
        return name if name else str(annotation).replace("typing.", "")
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    if origin in (typing.Union, UnionType):
        return " | ".join(_annotation_repr(a) for a in args)
    origin_name = getattr(origin, "__name__", str(origin))
    return f"{origin_name}[{', '.join(_annotation_repr(a) for a in args)}]"


def _nested_export_models(model: type[BaseModel]) -> list[type[BaseModel]]:
    """Walk reachable model fields in declaration order, once per model."""
    seen = {model}
    models: list[type[BaseModel]] = []

    def visit(annotation: object) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in seen:
                return
            # A parent model can validate nested forward references without
            # rebuilding the nested class's public model_fields annotations.
            annotation.model_rebuild()
            seen.add(annotation)
            models.append(annotation)
            for field in annotation.model_fields.values():
                visit(field.annotation)
        else:
            for arg in typing.get_args(annotation):
                visit(arg)

    for field in model.model_fields.values():
        visit(field.annotation)
    return models


def render_schema_md() -> str:
    from bibr.export.models import _SCHEMA_VERSION, PaperExport

    nested_models = _nested_export_models(PaperExport)
    parts = [
        _HEADER,
        f"# JSON schema (v{_SCHEMA_VERSION})\n",
        "Top-level blocks of the export produced by `bibr chew` / `POST /papers/extract`. "
        "The Pydantic model `bibr.export.models.PaperExport` is the single source "
        "of truth for output structure. Schema validation does not establish "
        "factual accuracy.\n",
        "[Download the generated JSON Schema](paper.schema.json). This uses the "
        "same schema builder as the checked-in export artifact, including its "
        "required-root-field contract.\n",
        "## Top-level blocks\n",
        "| Block | Type | Description |",
        "|---|---|---|",
    ]
    for name, finfo in PaperExport.model_fields.items():
        # Type cell is a code span (backtick-wrapped): a union's " | " does
        # NOT need escaping there — Python-Markdown's `tables` extension is
        # code-span-aware when splitting row cells, and escaping instead
        # renders a visible literal backslash in the built HTML (backslash
        # escapes aren't processed inside code spans). Only the Description
        # cell (not inside backticks) needs pipe-escaping.
        ann = _annotation_repr(finfo.annotation)
        desc = _md_escape(finfo.description or "")
        json_name = finfo.serialization_alias or finfo.alias or name
        parts.append(f"| `{json_name}` | `{ann}` | {desc} |")
    parts.extend(
        [
            "\n## Reading an export\n",
            "The root `schema_version` identifies the output schema; "
            "`extraction.bibr_version` identifies the producing package. "
            "Paper fields live in `metadata`, input identity in `source`, and "
            "telemetry in `extraction`. Table rows connect through IDs: "
            "`text.section_id` points to `section.section_id`, and `xref` links "
            "a source `text_id` to an object identified by `xref_type` and `xref_id`. "
            "Do not treat IDs as array positions.\n",
            "`bib` starts with parsed references. `bib_match` and `metadata_match` hold "
            "external enrichment separately. Explicit consolidation (`fill` or "
            "`replace`) can update reference fields and records `consolidated_fields`. "
            "[See enrichment and consolidation](../guides/configuration.md).\n",
            "Optional blocks may be absent rather than null. `extraction.regions` "
            "requires `include_regions=True` / `--regions`; "
            "per-sentence underscore fields require `include_region_meta=True` / "
            "`--region-meta`. The `validation` gate is enabled by default "
            "and can be disabled independently of the typed export builder. "
            "Inspect `extraction.warnings` and `validation` when reviewing results. "
            "The root `schema_version` key distinguishes v11 from legacy exports. "
            "For compatibility and structured reference names, see "
            "[the export overview](../guides/architecture.md).\n",
            "## Nested record fields\n",
            "These tables describe the models referenced above. Required means "
            "required by the model constructor; a nullable field may still be "
            "required. Defaults do not guarantee that a field is emitted: the "
            "exporter omits selected optional and debug fields.\n",
        ]
    )
    for model in nested_models:
        parts.extend(
            [
                f"### {model.__name__}\n",
                "| Field | Type | Required | Default |",
                "|---|---|---|---|",
            ]
        )
        for name, field in model.model_fields.items():
            json_name = field.serialization_alias or field.alias or name
            ann = _annotation_repr(field.annotation)
            required = "Yes" if field.is_required() else "No"
            if field.is_required():
                default = "—"
            elif field.default_factory is not None:
                default = "(computed)"
            else:
                default = f"`{field.default!r}`"
            parts.append(f"| `{json_name}` | `{ann}` | {required} | {default} |")
        parts.append("")
    return "\n".join(parts) + "\n"
