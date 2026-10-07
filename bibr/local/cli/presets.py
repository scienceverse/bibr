"""``bibr preset`` subcommand — save/load named .env snapshots."""

import argparse
import sys

from bibr.local.cli import ui


def _run_preset(args, parser: argparse.ArgumentParser | None = None) -> None:
    """Handle ``bibr preset`` subcommands."""
    from rich.console import Console
    from rich.prompt import Confirm

    from bibr.env_utils import parse_env
    from bibr.presets import (
        InvalidPresetError,
        PresetManager,
        carries_credentials,
        effective_env_file,
        endpoint_changes,
        is_secret_key,
        redact_value,
    )

    console = Console()
    # ``BIBR_PRESETS_DIR``, else ``~/.bibr/presets`` (see ``default_presets_dir``).
    manager = PresetManager()
    env_path = effective_env_file()

    cmd = args.preset_command

    if cmd is None:
        # Render the argparse help instead of a hand-rolled usage line so the
        # output matches every other ``bibr X --help`` page.
        if parser is not None:
            parser.print_help()
        sys.exit(1)

    if env_path is None and cmd in ("save", "use", "deactivate", "diff"):
        # These read or write the .env in effect, and none is read now.
        from bibr.config import ENV_FILE_OVERRIDE_VAR, dotenv_disabled_by, dotenv_enable_hint

        variable = dotenv_disabled_by() or ENV_FILE_OVERRIDE_VAR
        ui.error(
            console,
            f"Dotenv loading is disabled by {variable}.",
            hint=f"{dotenv_enable_hint(variable)}, or apply a preset for one run with "
            "[cyan]bibr chew --preset NAME[/cyan].",
        )
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
        active = manager.get_active(env_path) if env_path is not None else None
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
        try:
            if (
                manager.exists(args.name)
                and not args.force
                and not Confirm.ask(
                    f"Preset [cyan]{args.name}[/cyan] already exists. Overwrite?", default=False
                )
            ):
                console.print("[dim]Aborted.[/dim]")
                sys.exit(1)
            data = manager.snapshot_from_env(env_path)
            manager.save(args.name, data)
            ui.ok(
                console,
                f"Saved preset [cyan]{args.name}[/cyan] from {env_path} "
                f"({len(data)} settings; secrets excluded)",
            )
            # Name what the secret filter took beyond the obvious keys.
            withheld = sorted(
                key
                for key, value in parse_env(env_path).items()
                if not is_secret_key(key) and carries_credentials(value)
            )
            if withheld:
                console.print(
                    f"  [dim]Left out {', '.join(withheld)}: a URL with a password or key "
                    "stays only in .env.[/dim]"
                )
        except InvalidPresetError as e:
            ui.error(console, str(e))
            sys.exit(1)

    elif cmd == "use":
        if not env_path.exists():
            ui.error(console, "No .env file found.", hint="Run [cyan]bibr setup[/cyan] first.")
            sys.exit(1)
        try:
            before = parse_env(env_path)
            manager.apply(args.name, env_path)
            data = manager.load(args.name)
            ui.ok(
                console,
                f"Applied preset [cyan]{args.name}[/cyan] to {env_path} "
                f"(also wrote BIBR_ACTIVE_PRESET marker)",
            )
            _print_settings(data, dim=True)
            if redirected := endpoint_changes(data, before):
                ui.warn(
                    console,
                    f"Preset [cyan]{args.name}[/cyan] changed {', '.join(redirected)}.",
                    hint="These decide where bibr sends requests, with the API keys in "
                    ".env, and what it launches; check them if the preset came from "
                    "someone else.",
                )
        except FileNotFoundError:
            _suggest_available(args.name)
            sys.exit(1)
        except InvalidPresetError as e:
            ui.error(console, str(e))
            sys.exit(1)

    elif cmd == "deactivate":
        removed = manager.deactivate(env_path)
        if removed:
            ui.ok(
                console,
                f"Removed [cyan]BIBR_ACTIVE_PRESET[/cyan] from {env_path} "
                "(other settings unchanged)",
            )
        else:
            console.print(f"[dim]No active preset marker in {env_path} — nothing to do.[/dim]")

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
        except InvalidPresetError as e:
            ui.error(console, str(e))
            sys.exit(1)

    elif cmd == "show":
        try:
            data = manager.load(args.name)
            console.print(f"[dim]{manager._path(args.name)}[/dim]")  # noqa: SLF001
            _print_settings(data)
        except FileNotFoundError:
            _suggest_available(args.name)
            sys.exit(1)
        except InvalidPresetError as e:
            ui.error(console, str(e))
            sys.exit(1)

    elif cmd == "diff":
        if not env_path.exists():
            ui.error(console, f"No .env file found (looked for {env_path}).")
            sys.exit(1)
        try:
            env_dict = parse_env(env_path)
            changed, only_in_preset, only_in_env = manager.diff_against(args.name, env_dict)
        except FileNotFoundError:
            _suggest_available(args.name)
            sys.exit(1)
        except InvalidPresetError as e:
            ui.error(console, str(e))
            sys.exit(1)
        if not changed and not only_in_preset and not only_in_env:
            ui.ok(console, f"Preset [cyan]{args.name}[/cyan] matches {env_path}")
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
