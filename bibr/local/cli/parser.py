"""``bibr`` CLI argument parser construction."""

import argparse


def normalize_ocr_backend(name: str | None) -> str:
    """Expand short OCR backend aliases to their full registry names.

    Thin wrapper over :func:`bibr.ocr.registry.resolve_backend_name` — the
    library-level single source of truth, shared with ``LocalPipeline``.
    """
    from bibr.ocr.registry import resolve_backend_name

    return resolve_backend_name(name)


_EXAMPLES = """\
examples:
  process papers:
    bibr chew paper.pdf                          # single file → stdout JSON
    bibr chew paper.pdf -o result.json           # single file → file
    bibr chew papers/ -o results/                # batch → directory
    cat paper.pdf | bibr chew -                  # read from stdin
    bibr chew paper.pdf --pages 1-5              # process only pages 1–5
    bibr chew paper.pdf --dry-run                # preview the resolved plan, no processing

  backends & models:
    bibr chew paper.pdf --ocr gemini             # use Gemini Flash for OCR
    bibr chew paper.pdf --ocr glm-http           # use remote GLM-OCR server
    bibr chew paper.pdf --ocr-url http://host:8080  # explicit remote OCR server
    bibr chew paper.pdf --memory aggressive      # low-RAM machines (≤8 GB)
    bibr chew paper.pdf --no-llm                 # skip downstream LLM extraction (OCR unchanged)

  references & output:
    bibr chew paper.pdf --no-crossref            # skip reference enrichment
    bibr chew paper.pdf --refs llm               # full-precision LLM ref parser (default is local NER)
    bibr chew paper.pdf --refs off               # skip reference extraction entirely
    bibr chew paper.pdf --figure-images          # include base64 figure images
    bibr chew paper.pdf --preset NAME            # apply a saved preset for this run
    bibr inspect result.json                     # summarize an extraction-output JSON

  presets & config:
    bibr preset list                             # show all saved presets
    bibr preset save NAME                        # snapshot current .env (excludes secrets)
    bibr preset use NAME                         # apply a preset to .env
    bibr preset diff NAME                        # compare a preset against current .env
    bibr preset show NAME                        # display preset contents
    bibr preset rm NAME                          # delete a preset
    bibr preset deactivate                       # clear the active-preset marker

    bibr config show --sources                   # resolved settings + where each comes from
    bibr config path                             # locate the .env file(s) bibr reads
    bibr config set LLM_PROVIDER openai          # write a setting to .env
    bibr config example --full                   # print a full .env template
"""


def _get_version() -> str:
    """Read package version from installed metadata."""
    from importlib.metadata import version

    return version("bibr")


class _BibrParser(argparse.ArgumentParser):
    """Main-parser subclass that renders the designed help screen.

    Only the *top-level* parser gets this treatment — argparse propagates the
    parser class to subparsers by default, so the override defers to the
    stock formatter for every prog other than ``bibr`` itself (subcommand
    help pages stay on argparse's formatter; they are reference material,
    not a landing page). Command rows are collected from the subparsers
    action's pseudo-actions, so the screen stays in sync as commands change.
    """

    def format_help(self) -> str:
        if self.prog != "bibr":
            return super().format_help()

        from bibr.local.cli.ui import render_main_help

        commands: list[tuple[str, str]] = []
        for action in self._actions:  # noqa: SLF001
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                commands = [
                    (pseudo.dest, pseudo.help or "")  # noqa: SLF001
                    for pseudo in action._choices_actions  # noqa: SLF001
                ]
                break
        try:
            ver = _get_version()
        except Exception:
            ver = "?"
        return render_main_help(ver, commands)


def _build_parser() -> argparse.ArgumentParser:
    parser = _BibrParser(
        prog="bibr",
        description="bibr — scientific paper metadata extraction",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bibr {_get_version()}",
    )
    sub = parser.add_subparsers(dest="command")

    chew = sub.add_parser(
        "chew",
        help="extract metadata from papers (PDF, DOCX, XML, HTML, ePub)",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    chew.add_argument(
        "input",
        nargs="*",
        help="Input file(s) or directory containing PDF/DOCX/XML/HTML/ePub files (use '-' for stdin)",
    )
    chew.add_argument(
        "--manifest",
        help="JSONL manifest carrying per-file source identity and output identity",
    )
    chew.add_argument(
        "-o",
        "--output",
        help="Output path (file for single input, directory for batch)",
    )
    chew.add_argument(
        "--memory",
        choices=["aggressive", "balanced", "keep_all"],
        default=None,
        help="Memory management mode (default: auto-detected from system RAM and CUDA VRAM)",
    )
    chew.add_argument(
        "--ocr",
        choices=[
            "paddle",
            "paddle-vllm",
            "paddle-rapid-mlx",
            "paddle-mlx-vlm",
            "paddle-http",
            "glm",
            "glm-llama",
            "glm-rapid-mlx",
            "glm-http",
            "gemini",
            "openai",
            "anthropic",
        ],
        default=None,
        help=(
            "OCR backend (default: paddle, automatically tries Paddle runtimes before the "
            "explicit GLM fallback chain)"
        ),
    )
    chew.add_argument(
        "--ocr-url",
        help="URL for external OCR server (default profile: Paddle; --ocr glm-http keeps GLM)",
    )
    chew.add_argument(
        "--ocr-model",
        help="Model path or served model name for OCR (custom aliases require --ocr-profile)",
    )
    chew.add_argument(
        "--ocr-profile",
        choices=["paddle", "glm"],
        help="OCR model family for a custom --ocr-model alias",
    )
    chew.add_argument(
        "--llm",
        choices=["cloud", "local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"],
        default=None,
        help=(
            "LLM backend: cloud, or a managed local server — 'local' auto-picks "
            "vllm-mlx/Rapid-MLX (Apple Silicon), llama.cpp (Windows/small CUDA), vLLM, "
            "or external LM Studio/llmster. "
            "Default: LLM_BACKEND "
            "setting (cloud)."
        ),
    )
    chew.add_argument(
        "--llm-provider",
        help="Override LLM_PROVIDER setting (ignored for managed local backends "
        "(--llm vllm/vllm-mlx/rapid-mlx/local))",
    )
    chew.add_argument(
        "--llm-model",
        help="Override LLM_MODEL setting (ignored for managed local backends "
        "(--llm vllm/vllm-mlx/rapid-mlx/local))",
    )
    chew.add_argument(
        "--no-crossref",
        action="store_true",
        help="Disable Crossref reference enrichment",
    )
    chew.add_argument(
        "--no-equations",
        action="store_true",
        help="Disable equation extraction",
    )
    chew.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Skip downstream LLM extraction, citation linking, equations, and Crossref. "
            "Preserves document structure and preparsed native metadata. "
            "OCR still uses the selected backend, including cloud vision providers."
        ),
    )
    chew.add_argument(
        "--refs",
        choices=["llm", "ner", "llm-chunked", "off"],
        help=(
            "Reference parser: 'ner' (default) uses the local ModernBERT-CRF model "
            "with no per-reference LLM cost (requires the ml extra). "
            "'llm' parses references with the configured LLM in batches. "
            "'llm-chunked' parses region-aligned chunks of the reference list. "
            "Choose segmentation separately with --ref-seg. "
            "'off' skips reference extraction entirely (empty bib/bib_match/xref "
            "tables; --ref-seg is ignored) while keeping core metadata, sections "
            "and equations. --no-llm also skips metadata LLM calls and equations. "
            "Overrides REF_PARSE_STRATEGY."
        ),
    )
    chew.add_argument(
        "--ref-seg",
        choices=["llm", "crf", "geom", "region"],
        help=(
            "Reference SEGMENTATION strategy (orthogonal to --refs, which selects "
            "parsing): 'geom' (default) the local geometry GBM, cascading to region "
            "anchors then LLM then CRF when geometry is absent/unconfident. The local "
            "geometry tier needs a text-layer PDF and the ml extra; 'region' segments "
            "by layout-region anchors (LLM→CRF fallback) and covers "
            "scanned PDFs with no text layer; 'llm' anchor-emit with region→CRF "
            "fallback; 'crf' the local ModernBERT-CRF segmenter. Sets "
            "REF_SEG_STRATEGY."
        ),
    )
    chew.add_argument(
        "--consolidate",
        nargs="?",
        const="fill",
        choices=["fill", "replace"],
        help=(
            "Merge accepted Crossref match data (bib_match) into the bib table. "
            "Bare flag = 'fill' (only fills missing fields); 'replace' also "
            "overwrites disagreeing ones. Modified rows get a consolidated_fields "
            "marker. No-op with --no-crossref or when no matches were found. "
            "Place after the input path (e.g. `bibr chew paper.pdf "
            "--consolidate`), or use `--consolidate=replace`."
        ),
    )
    chew.add_argument(
        "--figure-images",
        action="store_true",
        help="Include base64-encoded figure images in output (off by default)",
    )
    chew.add_argument(
        "--regions",
        action="store_true",
        help=(
            "Include the _regions debug payload (per-region layout: bbox, font, "
            "content, etc.). Off by default — not consumed by standard "
            "downstream tools like Metacheck."
        ),
    )
    chew.add_argument(
        "--region-meta",
        action="store_true",
        help=(
            "Include the per-text underscore region metadata (_bbox_2d, "
            "_font_size, _region_type, …; training/debug payload from "
            "output). Off by default."
        ),
    )
    chew.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation, for piping to jq or storage)",
    )
    chew.add_argument(
        "--pages",
        help="Page range to process (e.g., '1-5', '3')",
    )
    chew.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Force compute device",
    )
    chew.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Files per chunk in batch processing (default: auto-detected from memory mode)",
    )
    chew.add_argument(
        "--paper-id",
        help="Paper ID for single file processing",
    )
    chew.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    chew.add_argument(
        "--preset",
        help="Apply a preset before processing (does not modify .env)",
    )
    chew.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and print the full run plan (input files, OCR/LLM config, "
            "reference strategies, enrichment, memory mode, models that would "
            "need downloading, output destinations) without processing anything. "
            "No network calls, no model loads. Exits 0."
        ),
    )

    # --- setup subcommand ---
    sub.add_parser(
        "setup",
        help="interactive setup wizard",
        add_help=False,
    )

    # --- serve subcommand ---
    sub.add_parser(
        "serve",
        help="start the HTTP API (see 'bibr serve --help')",
        add_help=False,
    )

    # --- demo subcommand ---
    sub.add_parser(
        "demo",
        help="launch the interactive demo (see 'bibr demo --help')",
        add_help=False,
    )

    # --- mcp subcommand ---
    mcp_parser = sub.add_parser(
        "mcp",
        help="start the MCP server for agents (stdio)",
        description=(
            "Run a Model Context Protocol server over stdio, exposing extraction "
            "as agent tools: chew_paper / load_paper register a paper, then "
            "get_metadata, get_sections, get_text, search_text, get_references, "
            "get_reference_citations, get_tables, get_figures and save_paper "
            "query the result in slices. One warm pipeline serves the whole "
            "session, so pipeline options are fixed at start via the flags "
            "below (a subset of 'bibr chew'). Requires the 'mcp' extra. "
            "Register with e.g.: claude mcp add bibr -- uv run bibr mcp"
        ),
    )
    mcp_parser.add_argument(
        "--ocr",
        choices=[
            "paddle",
            "paddle-vllm",
            "paddle-rapid-mlx",
            "paddle-mlx-vlm",
            "paddle-http",
            "glm",
            "glm-llama",
            "glm-rapid-mlx",
            "glm-http",
            "gemini",
            "openai",
            "anthropic",
        ],
        default=None,
        help="OCR backend (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--ocr-url",
        help="URL for external OCR server (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--ocr-model",
        help="Model path or served model name for OCR (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--ocr-profile",
        choices=["paddle", "glm"],
        help="OCR model family for a custom --ocr-model alias",
    )
    mcp_parser.add_argument(
        "--llm",
        choices=["cloud", "local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"],
        default=None,
        help="LLM backend (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--refs",
        choices=["llm", "ner", "llm-chunked", "off"],
        help="Reference extraction strategy (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--ref-seg",
        choices=["llm", "crf", "geom", "region"],
        help="Reference segmentation strategy (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--no-crossref",
        action="store_true",
        help="Disable Crossref reference enrichment",
    )
    mcp_parser.add_argument(
        "--no-equations",
        action="store_true",
        help="Disable equation extraction",
    )
    mcp_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip downstream LLM extraction and Crossref; OCR remains active (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--consolidate",
        nargs="?",
        const="fill",
        choices=["fill", "replace"],
        help="Merge accepted Crossref match data into the bib table (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--figure-images",
        action="store_true",
        help="Keep base64 figure images in stored exports (available via save_paper)",
    )
    mcp_parser.add_argument(
        "--memory",
        choices=["aggressive", "balanced", "keep_all"],
        default=None,
        help="Memory management mode (default: auto-detected from system RAM and CUDA VRAM)",
    )
    mcp_parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Force compute device",
    )
    mcp_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (stderr)",
    )

    # --- doctor subcommand ---
    sub.add_parser(
        "doctor",
        help="validate your environment",
    )

    # --- inspect subcommand ---
    inspect_parser = sub.add_parser(
        "inspect",
        help="summarize an extraction-output JSON",
        description=(
            "Summarize a bibr JSON export: title/authors/DOI/paper type, "
            "section/sentence/table/figure/equation counts, reference and "
            "enrichment stats, the validation block, and per-model LLM "
            "token usage. Works on any v10.x export."
        ),
    )
    inspect_parser.add_argument("json_file", help="Path to a bibr extraction-output JSON file")

    # --- preset subcommand ---
    preset_parser = sub.add_parser(
        "preset",
        help="manage configuration presets",
        description=(
            "Save / load named .env snapshots (LLM provider, OCR backend, "
            "rate limits, ...). Secrets like API keys are never written to "
            "preset files — they stay in .env."
        ),
    )
    preset_sub = preset_parser.add_subparsers(dest="preset_command")

    preset_sub.add_parser("list", help="List all saved presets")

    preset_save = preset_sub.add_parser(
        "save",
        help="Save current .env as a named preset (excludes secrets by default)",
    )
    preset_save.add_argument("name", help="Preset name")
    preset_save.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing preset without confirmation",
    )

    preset_use = preset_sub.add_parser("use", help="Apply a preset to .env")
    preset_use.add_argument("name", help="Preset name to activate")

    preset_sub.add_parser(
        "deactivate",
        help="Remove the active-preset marker from .env (no other settings change)",
    )

    preset_rm = preset_sub.add_parser("rm", help="Delete a preset")
    preset_rm.add_argument("name", help="Preset name to delete")
    preset_rm.add_argument(
        "-y", "--yes", action="store_true", help="Skip the deletion confirmation"
    )

    preset_show = preset_sub.add_parser("show", help="Show contents of a preset")
    preset_show.add_argument("name", help="Preset name to display")

    preset_diff = preset_sub.add_parser("diff", help="Compare a preset against current .env")
    preset_diff.add_argument("name", help="Preset name to compare")

    # --- config subcommand ---
    config_parser = sub.add_parser(
        "config",
        help="inspect and edit bibr's configuration (.env)",
        description=(
            "Show resolved settings and where each one comes from, locate "
            "your .env file, set individual values, or print an example "
            ".env template. Secret values (API keys, tokens, passwords) are "
            "always redacted in output — there is no flag to un-redact them."
        ),
    )
    config_sub = config_parser.add_subparsers(dest="config_command")

    config_show = config_sub.add_parser(
        "show", help="Show current settings (non-default only, unless --all)"
    )
    config_show.add_argument(
        "--sources",
        action="store_true",
        help="Show where each value comes from (env / .env path / default)",
    )
    config_show.add_argument(
        "--all", action="store_true", help="Show every setting, including defaults"
    )

    config_sub.add_parser("path", help="Show the .env file(s) bibr reads and whether they exist")

    config_set = config_sub.add_parser("set", help="Write one setting to .env")
    config_set.add_argument("key", help="Setting name (env var), case-insensitive")
    config_set.add_argument("value", help="Value to write")

    config_example = config_sub.add_parser("example", help="Print an example .env template")
    config_example.add_argument(
        "--full",
        action="store_true",
        help="Print every setting, grouped by section, commented out",
    )

    return parser
