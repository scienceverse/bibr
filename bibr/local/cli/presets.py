"""``bibr preset`` subcommand — save/load named .env snapshots."""

import argparse
import os
import sys
from pathlib import Path

from bibr.local.cli import ui


def _run_preset(args, parser: argparse.ArgumentParser | None = None) -> None:
    """Handle ``bibr preset`` subcommands."""
    from rich.console import Console
    from rich.prompt import Confirm

    from bibr.env_utils import parse_env
    from bibr.presets import InvalidPresetError, PresetManager, redact_value

    console = Console()
    # NOTE: ``Path("") or None`` evaluates to ``Path('.')`` because Path
    # objects are always truthy. Check the env var explicitly so an unset
    # ``BIBR_PRESETS_DIR`` falls back to the PresetManager default
    # (``~/.bibr/presets/``) instead of writing presets to the cwd.
    presets_dir_str = os.environ.get("BIBR_PRESETS_DIR", "").strip()
    manager = (
        PresetManager(presets_dir=Path(presets_dir_str)) if presets_dir_str else PresetManager()
    )
    env_path = Path.cwd() / ".env"

    cmd = args.preset_command

    if cmd is None:
        # Render the argparse help instead of a hand-rolled usage line so the
        # output matches every other ``bibr X --help`` page.
        if parser is not None:
            parser.print_help()
        sys.exit(1)

    def _suggest_available(missing: str) -> None:
        names = manager.list_presets()
        ui.error(console, f"Preset [cyan]{missing}[/cyan] not found.")
        console.print(f"  [dim]Available: {', '.join(names) or '(none)'}[/dim]")

    def _print_settings(data: dict[str, str], *, dim: bool = False) -> None:
        prefix = "  [dim]" if dim else "  "
        suffix = "[/dim]" if dim else ""
        for k, v in sorted(data.items()):
            console.print(f"{prefix}{k}={redact_value(k, v)}{suffix}")

    if cmd == "list":
        presets = manager.list_presets()
        if not presets:
            console.print(
                "[dim]No presets saved yet.[/dim] "
                "Run [cyan]bibr preset save <name>[/cyan] to create one."
            )
            return
        active = manager.get_active(env_path) if env_path.exists() else None
        table = ui.minimal_table("Preset", "Active")
        for name in presets:
            marker = "[green]●[/green]" if name == active else ""
            table.add_row(name, marker)
        console.print(table)
        console.print(f"[dim]Stored in {manager.directory}[/dim]")

    elif cmd == "save":
        if not env_path.exists():
            ui.error(console, "No .env file found.", hint="Run [cyan]bibr setup[/cyan] first.")
            sys.exit(1)
        if (
            manager.exists(args.name)
            and not args.force
            and not Confirm.ask(
                f"Preset [cyan]{args.name}[/cyan] already exists. Overwrite?", default=False
            )
        ):
            console.print("[dim]Aborted.[/dim]")
            sys.exit(1)
        try:
            data = manager.snapshot_from_env(env_path)
            manager.save(args.name, data)
            ui.ok(
                console,
                f"Saved preset [cyan]{args.name}[/cyan] ({len(data)} settings; secrets excluded)",
            )
        except InvalidPresetError as e:
            ui.error(console, str(e))
            sys.exit(1)

    elif cmd == "use":
        if not env_path.exists():
            ui.error(console, "No .env file found.", hint="Run [cyan]bibr setup[/cyan] first.")
            sys.exit(1)
        try:
            manager.apply(args.name, env_path)
            data = manager.load(args.name)
            ui.ok(
                console,
                f"Applied preset [cyan]{args.name}[/cyan] to .env "
                f"(also wrote BIBR_ACTIVE_PRESET marker)",
            )
            _print_settings(data, dim=True)
        except FileNotFoundError:
            _suggest_available(args.name)
            sys.exit(1)

    elif cmd == "deactivate":
        removed = manager.deactivate(env_path)
        if removed:
            ui.ok(
                console,
                "Removed [cyan]BIBR_ACTIVE_PRESET[/cyan] from .env (other settings unchanged)",
            )
        else:
            console.print("[dim]No active preset marker in .env — nothing to do.[/dim]")

    elif cmd == "rm":
        if not args.yes and not Confirm.ask(
            f"Delete preset [cyan]{args.name}[/cyan]?", default=False
        ):
            console.print("[dim]Aborted.[/dim]")
            sys.exit(1)
        try:
            manager.delete(args.name)
            ui.ok(console, f"Deleted preset [cyan]{args.name}[/cyan]")
        except FileNotFoundError:
            _suggest_available(args.name)
            sys.exit(1)

    elif cmd == "show":
        try:
            data = manager.load(args.name)
            console.print(f"[dim]{manager._path(args.name)}[/dim]")  # noqa: SLF001
            _print_settings(data)
        except FileNotFoundError:
            _suggest_available(args.name)
            sys.exit(1)

    elif cmd == "diff":
        if not env_path.exists():
            ui.error(console, "No .env file found in current directory.")
            sys.exit(1)
        try:
            env_dict = parse_env(env_path)
            changed, only_in_preset, only_in_env = manager.diff_against(args.name, env_dict)
        except FileNotFoundError:
            _suggest_available(args.name)
            sys.exit(1)
        if not changed and not only_in_preset and not only_in_env:
            ui.ok(console, f"Preset [cyan]{args.name}[/cyan] matches .env")
            return
        if changed:
            console.print(f"[bold]Changed[/bold] ({len(changed)}):")
            for k, (env_v, pre_v) in sorted(changed.items()):
                console.print(
                    f"  {k}: [yellow]{redact_value(k, env_v)}[/yellow] "
                    f"→ [green]{redact_value(k, pre_v)}[/green]"
                )
        if only_in_preset:
            console.print(f"[bold]Only in preset[/bold] ({len(only_in_preset)}):")
            for k, v in sorted(only_in_preset.items()):
                console.print(f"  + {k}={redact_value(k, v)}")
        if only_in_env:
            console.print(f"[bold]Only in .env[/bold] ({len(only_in_env)}):")
            for k, v in sorted(only_in_env.items()):
                console.print(f"  - {k}={redact_value(k, v)}")
