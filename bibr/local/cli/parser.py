"""``bibr`` CLI argument parser construction."""

import argparse
import difflib
import re
import sys
from typing import NoReturn


def normalize_ocr_backend(name: str | None) -> str:
    """Expand short OCR backend aliases to their full registry names.

    Thin wrapper over :func:`bibr.ocr.registry.resolve_backend_name` — the
    library-level single source of truth, shared with ``LocalPipeline``.
    """
    from bibr.ocr.registry import resolve_backend_name

    return resolve_backend_name(name)


_EXAMPLES = """\
examples:
  bibr chew paper.pdf -o result.json           # one paper → one JSON file
  bibr chew paper.pdf                          # JSON on stdout (progress goes to stderr)
  bibr chew papers/ -o results/                # every paper in a folder
  bibr chew paper.pdf --pages 1-5              # only pages 1–5
  bibr chew paper.pdf --dry-run                # show the plan, process nothing
  cat paper.pdf | bibr chew -                  # read from stdin

  bibr chew paper.pdf --fast                   # local models only: no LLM, no API key
  bibr chew paper.pdf --full                   # also use the configured LLM
  bibr chew paper.pdf --refs off               # skip the reference list
  bibr chew paper.pdf --refs llm               # parse references with the LLM
  bibr chew paper.pdf --crossref               # look references up in Crossref
  bibr chew scan.pdf --ocr anthropic           # read a scan with a cloud vision model
  bibr chew paper.pdf --ocr-url http://host:8080   # use an OCR server you run

next steps:
  bibr inspect result.json                     # summary of one result
  bibr references results/ -o refs.bib         # reference lists → BibTeX, RIS or CSL-JSON
  bibr tables results/ --out tables/           # results → Parquet tables
  bibr batch papers/ --out corpus/             # large corpus: resumable, with a report

Every option, setting and JSON field: https://bibr.org/reference/cli/
"""


_BATCH_EXAMPLES = """\
examples:
  bibr batch papers/ --out results/                   # local run; re-run to resume
  bibr batch manifest.txt --out results/ --refs llm   # manifest: one path per line
  bibr batch manifest.txt --out results/ --dry-run    # show the plan, run nothing
  bibr batch manifest.txt --out results/ --retry-failed --limit 50
  bibr batch manifest.txt --out results/ \\
      --serve-url https://bibr.example.org --concurrency 2 --max-concurrency 4
  bibr batch report results/                          # ledger summary (--json for JSON)

ledger: <out>/outcomes.jsonl — one JSON verdict line per attempt (status,
error_code, timings, stage times, LLM tokens, reference counts, warnings, build
sha); a local run also writes a 'started' line before each paper.
tables: <out>/tables/*.parquet — every successful paper as one Parquet file per
table, keyed by paper_id (rewritten after each run; --no-tables to skip).
"""


def _get_version() -> str:
    """Read package version from installed metadata."""
    from importlib.metadata import version

    return version("bibr")


def _safe_version() -> str:
    """Installed version, or ``'?'`` when bibr has no package metadata.

    A source-tree checkout run via ``PYTHONPATH`` (never pip-installed)
    has no ``importlib.metadata`` entry; every command, including
    ``doctor``, must still start so it can report the environment.
    """
    try:
        return _get_version()
    except Exception:
        return "?"


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
        ver = _safe_version()
        return render_main_help(ver, commands)

    def error(self, message: str) -> NoReturn:
        """Report a usage error as one styled line and a pointer to ``--help``.

        argparse's own report is the full usage synopsis (a dozen lines for
        ``bibr chew``) followed by the message; the mistake is usually one
        mistyped word, so say which, suggest the nearest real one, and name
        the help screen instead. Exit status stays 2.
        """
        from rich.console import Console
        from rich.markup import escape

        from bibr.local.cli import ui

        text, help_prog = _describe_usage_error(self, message)
        ui.error(
            Console(stderr=True),
            escape(text),
            hint=f"Run [cyan]{help_prog} --help[/cyan] to see what it accepts.",
        )
        self.exit(2)


_INVALID_CHOICE = re.compile(
    r"^argument (?P<arg>.+?): invalid choice: (?P<bad>.+?) \(choose from (?P<choices>.*)\)$"
)
_SUBCOMMAND_DESTS = ("command", "preset_command", "config_command", "ref_model_command")


def _nearest(word: str, candidates: list[str]) -> str | None:
    matches = difflib.get_close_matches(word, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _subcommand_parser(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.ArgumentParser:
    """The parser of the subcommand *argv* names, or *parser* if it names none."""
    for action in parser._actions:  # noqa: SLF001 — argparse has no public tree API
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for token in argv:
                if token in action.choices:
                    return _subcommand_parser(action.choices[token], argv)
    return parser


def _describe_usage_error(parser: argparse.ArgumentParser, message: str) -> tuple[str, str]:
    """A usage error in plain words, and the command whose help explains it."""
    invalid = _INVALID_CHOICE.match(message)
    if invalid:
        bad = invalid["bad"].strip("'\"")
        choices = [c.strip(" '\"") for c in invalid["choices"].split(",")]
        nearest = _nearest(bad, choices)
        suggestion = f" Did you mean '{nearest}'?" if nearest else ""
        if invalid["arg"] in _SUBCOMMAND_DESTS:
            return f"Unknown command '{bad}'.{suggestion}", parser.prog
        listed = "" if nearest else f" Choose from: {', '.join(choices)}."
        return (
            f"{invalid['arg']}: '{bad}' is not one of its values.{suggestion}{listed}",
            parser.prog,
        )

    target = _subcommand_parser(parser, sys.argv[1:])
    prefix = "unrecognized arguments: "
    if message.startswith(prefix):
        unknown = message[len(prefix) :].split()
        flags = [token.split("=", 1)[0] for token in unknown if token.startswith("-")]
        if flags:
            nearest = _nearest(flags[0], list(target._option_string_actions))  # noqa: SLF001
            suggestion = f" Did you mean '{nearest}'?" if nearest else ""
            return f"Unknown option '{flags[0]}'.{suggestion}", target.prog
        return f"Unexpected argument '{unknown[0]}'.", target.prog
    return message[:1].upper() + message[1:], target.prog


#: Every ``--ocr`` value. The help text groups them; argparse checks against this.
OCR_BACKEND_CHOICES = [
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
    "text-layer",
]

LLM_BACKEND_CHOICES = ["cloud", "local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"]


def _add_pipeline_options(parser: argparse.ArgumentParser) -> None:
    """Register the pipeline options shared by ``bibr chew`` and ``bibr batch``.

    One definition, so the two commands cannot drift: run mode, reference
    strategies, enrichment, output payload toggles, OCR/LLM backends, page
    range, device, chunk size and presets. ``resolve_run_config`` and
    ``_apply_runtime_settings`` read exactly these attributes.

    The options are registered in titled groups, most used first, so that
    ``--help`` reads as a short list of decisions rather than one flat wall.
    """
    mode = parser.add_argument_group(
        "run mode",
        "Default: full when an LLM is configured ('bibr setup'), otherwise fast.",
    )
    run_mode = mode.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--fast",
        action="store_true",
        help=(
            "No LLM call and no API key: local models read the title, authors, abstract "
            "and references. What a run with no LLM configured does anyway. "
            "Overrides PIPELINE_MODE."
        ),
    )
    run_mode.add_argument(
        "--full",
        action="store_true",
        help=(
            "The configured LLM also reads what the local models are unsure of. Needs an "
            "LLM (an API key or a local model; see bibr setup). Overrides PIPELINE_MODE."
        ),
    )
    run_mode.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Structure only: sections and text, but no authors, references, citation "
            "links, equations or Crossref for PDF and DOCX (JATS and HTML keep the "
            "metadata they carry). OCR still runs. To keep authors and references "
            "without an LLM, use --fast."
        ),
    )
    mode.add_argument(
        "--preset",
        help="Apply a saved preset for this run (see bibr preset; .env is not modified)",
    )

    extract = parser.add_argument_group("what to extract")
    extract.add_argument(
        "--pages",
        help="Page range to process (e.g. '1-5', '3')",
    )
    extract.add_argument(
        "--refs",
        choices=["llm", "ner", "llm-chunked", "ref-llm", "off"],
        help=(
            "How references are parsed: 'ner' (default) a local model; 'llm' the "
            "configured LLM; 'llm-chunked' the LLM on chunks of the list; 'ref-llm' a "
            "dedicated fine-tuned reference endpoint; 'off' skips references and their "
            "citation links. Overrides REF_PARSE_STRATEGY."
        ),
    )
    extract.add_argument(
        "--ref-seg",
        choices=["llm", "crf", "geom", "region"],
        help=(
            "How the reference list is split into entries (separate from --refs): "
            "'geom' (default) a local model on line geometry, which needs a PDF text "
            "layer and falls back to the others; 'region' layout regions, for scans; "
            "'llm' the configured LLM; 'crf' a local sequence model. "
            "Sets REF_SEG_STRATEGY."
        ),
    )
    crossref = extract.add_mutually_exclusive_group()
    crossref.add_argument(
        "--crossref",
        action="store_true",
        help=(
            "Look each reference up in Crossref for this run (off by default; "
            "overrides CROSSREF_ENRICH=false)"
        ),
    )
    crossref.add_argument(
        "--no-crossref",
        action="store_true",
        help="Do not look references up for this run (overrides CROSSREF_ENRICH=true)",
    )
    extract.add_argument(
        "--consolidate",
        nargs="?",
        const="fill",
        choices=["fill", "replace"],
        help=(
            "With --crossref: merge accepted matches into the reference table. 'fill' "
            "(the bare flag) only fills missing fields; 'replace' also overwrites "
            "disagreeing ones, but only from a match carrying the reference's printed "
            "DOI. Put it after the input path, or write --consolidate=replace."
        ),
    )
    extract.add_argument(
        "--no-equations",
        action="store_true",
        help="Disable equation extraction",
    )

    payload = parser.add_argument_group("extra output (off by default)")
    payload.add_argument(
        "--figure-images",
        action="store_true",
        help="Include base64-encoded figure images",
    )
    payload.add_argument(
        "--regions",
        "--include-regions",
        action="store_true",
        help="Include extraction.regions: per-region layout (bbox, font, content) for debugging",
    )
    payload.add_argument(
        "--region-meta",
        action="store_true",
        help=(
            "Include extraction.text_regions: per-sentence layout features "
            "(bbox, font size, region type) for training and debugging"
        ),
    )

    ocr = parser.add_argument_group(
        "OCR (PDF pages)",
        # Hand-wrapped: the commands use RawDescriptionHelpFormatter.
        "With no OCR setting, a fast run reads the PDF's own text layer, and a scan\n"
        "needs one of these. 'bibr doctor' shows what this machine can run.",
    )
    ocr.add_argument(
        "--ocr",
        choices=OCR_BACKEND_CHOICES,
        metavar="BACKEND",
        default=None,
        help=(
            "text-layer: no OCR model, born-digital PDFs only. paddle: PaddleOCR-VL on "
            "the runtime that fits this machine. anthropic, gemini, openai: a cloud "
            "vision model (page images leave this machine). A specific runtime: "
            "paddle-vllm, paddle-rapid-mlx, paddle-mlx-vlm, paddle-http, glm, glm-llama, "
            "glm-rapid-mlx, glm-http."
        ),
    )
    ocr.add_argument(
        "--ocr-url",
        help=(
            "URL of an OCR server you run. Connects as paddle-http unless --ocr or "
            "OCR_BACKEND names a GLM backend, then glm-http; ignored by the cloud backends"
        ),
    )
    ocr.add_argument(
        "--ocr-model",
        help="Model path or served model name for OCR (a custom alias needs --ocr-profile)",
    )
    ocr.add_argument(
        "--ocr-profile",
        choices=["paddle", "glm"],
        help="OCR model family for a custom --ocr-model alias",
    )

    llm = parser.add_argument_group("LLM (full mode)")
    llm.add_argument(
        "--llm",
        choices=LLM_BACKEND_CHOICES,
        metavar="BACKEND",
        default=None,
        help=(
            "cloud: the provider in LLM_PROVIDER. local: a server bibr starts for this "
            "machine (vllm-mlx or rapid-mlx on Apple Silicon, llama-cpp on Windows or a "
            "small NVIDIA card, vllm on a larger one); name one of those to force it. "
            "llmster: an LM Studio server you run. Default: LLM_BACKEND (cloud)."
        ),
    )
    llm.add_argument(
        "--llm-provider",
        help="Override LLM_PROVIDER (ignored by the local backends)",
    )
    llm.add_argument(
        "--llm-model",
        help="Override LLM_MODEL (ignored by the local backends)",
    )

    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--memory",
        choices=["aggressive", "balanced", "keep_all"],
        default=None,
        help=(
            "How many models stay loaded at once (default: chosen from system RAM, "
            "CUDA VRAM and the LLM backend; aggressive suits 8 GB machines)"
        ),
    )
    perf.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Force the compute device",
    )
    perf.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Files per chunk when processing several (default: from the memory mode)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = _BibrParser(
        prog="bibr",
        description="bibr — scientific papers to structured JSON",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bibr {_safe_version()}",
    )
    sub = parser.add_subparsers(dest="command")

    chew = sub.add_parser(
        "chew",
        help="extract papers to JSON (PDF, DOCX, JATS XML, HTML, ePub)",
        usage="bibr chew [options] INPUT [INPUT ...]",
        # Hand-wrapped: RawDescriptionHelpFormatter (for the examples) does not
        # re-flow the description.
        description=(
            "Extract papers to bibr's JSON format: metadata, sections and sentences,\n"
            "references and citation links, tables, figures and equations.\n"
            "\n"
            "The JSON goes to stdout unless -o is given; progress and messages go to\n"
            "stderr. With nothing configured this is a fast run: local models only, no\n"
            "LLM and no API key. Run 'bibr setup' to add an LLM, or OCR for scans."
        ),
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    chew.add_argument(
        "input",
        nargs="*",
        metavar="INPUT",
        help=(
            "Files or directories of PDF, DOCX, JATS XML, HTML or ePub papers "
            "('-' reads one file from stdin)"
        ),
    )
    chew.add_argument(
        "--sample",
        action="store_true",
        help="Process the sample paper that ships with bibr (no input needed)",
    )
    chew.add_argument(
        "--manifest",
        help="JSONL manifest carrying per-file source identity and output identity",
    )
    chew.add_argument(
        "-o",
        "--output",
        "--out",
        dest="output",
        help=(
            "Where to write: a file for one paper, a directory for several (one "
            "<name>.json each). Beside each export bibr also writes <name>.core.json "
            "(the extraction before enrichment) and <name>.json.receipt.json (a run "
            "record). Default: stdout"
        ),
    )
    _add_pipeline_options(chew)
    chew.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation, for piping to jq or storage)",
    )
    chew.add_argument(
        "--paper-id",
        help="Paper ID for single file processing",
    )
    chew.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show bibr's log (stage decisions, model loads, fallbacks); -vv adds debug detail",
    )
    chew.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the resolved plan (inputs, mode, OCR and LLM routes, reference "
            "strategies, models still to download, outputs) and process nothing: no "
            "network calls, no model loads. Exits 1 with a Blockers section when the "
            "real run would fail."
        ),
    )

    # --- batch subcommand ---
    batch = sub.add_parser(
        "batch",
        help="run a corpus with a resumable ledger (locally or via bibr serve)",
        description=(
            "Process a corpus — manifest text files (one path per line, '#' comments), "
            "directories (recursive) or files — writing <out>/<paper_id>.json per paper "
            "and an append-only <out>/outcomes.jsonl ledger (one verdict line per "
            "attempt; a local run writes a 'started' line first). "
            "Re-running the same command resumes: papers whose latest verdict is "
            "'ok' are skipped, failed ones too unless --retry-failed, everything runs "
            "again with --force. A paper that was interrupted, or refused by the serve's "
            "token, or not finished when a model override stopped the run, runs again "
            "anyway, and so does one that crashed or hit a service outage, or whose "
            "export fell back from an overridden model, until that happened three times. "
            "Without --serve-url the corpus runs through one warm "
            "local pipeline in chunks of --batch-size (the 'bibr chew' options apply); "
            "with --serve-url papers go to a bibr serve async job API with adaptive "
            "concurrency. 'bibr batch report <out>' summarizes a ledger."
        ),
        epilog=_BATCH_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batch.add_argument(
        "inputs",
        nargs="*",
        help="Manifest file(s), directories, or files — or 'report <out>'",
    )
    batch.add_argument(
        "--out",
        "-o",
        "--output",
        dest="out",
        help="Output directory (exports, outcomes.jsonl, run_info.json)",
    )
    batch.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N papers this run (after resume filtering)",
    )
    batch.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize the order (the seed is recorded in run_info.json)",
    )
    batch.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffle seed (implies --shuffle)",
    )
    batch.add_argument(
        "--deadline",
        help=(
            "Stop submitting new papers after this time: ISO-8601 "
            "(2026-09-03T06:00:00Z; a naive timestamp is local time) or epoch seconds"
        ),
    )
    batch.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Also re-run papers whose latest verdict is 'failed' (without it, an "
            "interruption or a rejected token runs again anyway, and a crash or a service "
            "outage until the paper has failed that way three times)"
        ),
    )
    batch.add_argument(
        "--force",
        action="store_true",
        help="Re-run every paper, ignoring the ledger",
    )
    batch.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the plan (inputs, resume counts, executor settings, output paths) "
            "without processing anything"
        ),
    )
    batch.add_argument(
        "--report",
        action="store_true",
        help="Only print the ledger report for --out (same as 'bibr batch report <out>')",
    )
    batch.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of a table",
    )
    batch.add_argument(
        "--no-tables",
        action="store_true",
        help=(
            "Do not write <out>/tables/: the corpus as one Parquet file per table, "
            "rebuilt from every successful paper at the end of each run"
        ),
    )
    batch.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show bibr's log (stage decisions, model loads, fallbacks); -vv adds debug detail",
    )
    _add_pipeline_options(batch)
    remote = batch.add_argument_group("remote executor (bibr serve job API)")
    remote.add_argument(
        "--serve-url",
        help="Base URL of a bibr serve — selects the remote executor",
    )
    remote.add_argument(
        "--token",
        help=(
            "Bearer token (default: AUTH_API_KEY or BIBR_SERVE_TOKEN from the "
            "environment, else AUTH_API_KEY from .env)"
        ),
    )
    remote.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help=(
            "Send the bearer token over plain http:// to a public host. Loopback and "
            "private-network hosts (LAN, tailnet, single-label names) never need it; "
            "plain http to a LAN host is allowed with a warning."
        ),
    )
    remote.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Initial number of jobs in flight (default: 2)",
    )
    remote.add_argument(
        "--min-concurrency",
        type=int,
        default=1,
        help="In-flight floor the runner drops to on a 429 (default: 1)",
    )
    remote.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="In-flight ceiling it grows back to after successes (default: 4)",
    )
    remote.add_argument(
        "--poll-timeout",
        type=float,
        default=2400.0,
        help="Per-paper wall clock for a job to finish, in seconds (default: 2400)",
    )
    remote.add_argument(
        "--poll-interval",
        type=float,
        default=4.0,
        help="Seconds between job status polls (default: 4)",
    )
    remote.add_argument(
        "--retries",
        type=int,
        default=3,
        help=(
            "Retries per paper for transient failures — 502/503/504, connection "
            "errors, a job failed by an upstream OCR/LLM outage (default: 3)"
        ),
    )
    remote.add_argument(
        "--ready-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for GET /ready before giving up (default: 900)",
    )
    remote.add_argument(
        "--form",
        action="append",
        default=[],
        metavar="K=V",
        help="Extra multipart form field for POST /papers/jobs (repeatable)",
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
            "query the result in slices, and export_references / save_references "
            "give its reference list as CSL-JSON, BibTeX or RIS. One warm "
            "pipeline serves the whole session, so pipeline options are fixed at "
            "start via the flags below (a subset of 'bibr chew'). The file tools "
            "reach only the "
            "--allow-dir directories (default: the current directory). Requires "
            "the 'mcp' extra. Register with e.g.: claude mcp add bibr -- uv run bibr mcp "
            "(installed with uv tool install 'bibr[mcp]': claude mcp add bibr -- bibr mcp)"
        ),
    )
    mcp_parser.add_argument(
        "--allow-dir",
        action="append",
        metavar="DIR",
        help=(
            "Directory chew_paper, load_paper, save_paper and save_references may "
            "access, with symlinks resolved; repeat for several (default: the "
            "current directory)"
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
            "text-layer",
        ],
        default=None,
        help="OCR backend (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--ocr-url",
        help=(
            "URL of an external OCR server (as for 'bibr chew': paddle-http unless --ocr or "
            "OCR_BACKEND names a GLM backend)"
        ),
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
        choices=["llm", "ner", "llm-chunked", "ref-llm", "off"],
        help="Reference extraction strategy (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--ref-seg",
        choices=["llm", "crf", "geom", "region"],
        help="Reference segmentation strategy (as for 'bibr chew')",
    )
    mcp_crossref = mcp_parser.add_mutually_exclusive_group()
    mcp_crossref.add_argument(
        "--crossref",
        action="store_true",
        help="Enable Crossref reference enrichment (off by default; as for 'bibr chew')",
    )
    mcp_crossref.add_argument(
        "--no-crossref",
        action="store_true",
        help="Disable Crossref reference enrichment (as for 'bibr chew')",
    )
    mcp_parser.add_argument(
        "--no-equations",
        action="store_true",
        help="Disable equation extraction",
    )
    mcp_run_mode = mcp_parser.add_mutually_exclusive_group()
    mcp_run_mode.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: no LLM call and no API key, local models only (as for 'bibr chew'). "
        "What the server does with no LLM configured, without the flag.",
    )
    mcp_run_mode.add_argument(
        "--full",
        action="store_true",
        help="Full mode: the configured LLM reads what the local models are unsure of "
        "(as for 'bibr chew')",
    )
    mcp_run_mode.add_argument(
        "--no-llm",
        action="store_true",
        help="Structure only: skip downstream LLM extraction and Crossref; OCR remains active "
        "(as for 'bibr chew'). For authors and references without an LLM, use --fast.",
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
        help=(
            "Memory management mode (default: auto-detected from system RAM, CUDA VRAM "
            "and the LLM backend)"
        ),
    )
    mcp_parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Force compute device",
    )
    mcp_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show bibr's log on stderr; -vv adds debug detail",
    )

    # --- doctor subcommand ---
    ref_model_about = (
        "The reference adapter is an optional fine-tuned Qwen3.5-0.8B LoRA adapter that "
        "reviews the local parser's reference records, served from an endpoint you run. "
        "Most setups do not need it. See https://bibr.org/guides/reference-model/"
    )
    ref_model = sub.add_parser(
        "ref-model",
        help="download the pinned fine-tuned reference adapter",
        description=ref_model_about,
    )
    ref_commands = ref_model.add_subparsers(dest="ref_model_command", required=True)
    ref_download = ref_commands.add_parser(
        "download",
        help="verify and install the reference adapter",
        description=(
            "Download the pinned adapter bundle (about 84 MB), verify its checksum and "
            "file manifest, and install it into a new directory. Re-running verifies an "
            "existing installation; other files are never overwritten. " + ref_model_about
        ),
    )
    ref_download.add_argument(
        "--output",
        "-o",
        "--out",
        dest="output",
        required=True,
        type=str,
        help="Directory for the adapter (created; an existing installation there is verified)",
    )

    sub.add_parser(
        "doctor",
        help="check that this machine and configuration can run bibr",
        description=(
            "Check that bibr can run here: Python and system libraries, the .env files "
            "it reads, the run mode, the LLM and OCR routes that mode needs, the compute "
            "device and the model runtimes. Each line is ok, a warning or a failure with "
            "the command that fixes it. Processes no paper; exits 1 when a check fails."
        ),
    )

    # --- tables subcommand ---
    tables_parser = sub.add_parser(
        "tables",
        help="write JSON exports as Parquet tables (one file per table)",
        description=(
            "Write bibr JSON exports as one Parquet file per table — paper, author, "
            "affiliation, text, section, bib, xref, figure, table, eq, the *_match "
            "tables and the extraction_* processing lists — each row keyed by "
            "paper_id. Column types come from the export schema, so every file has "
            "the same columns however many papers it holds. Directories are "
            "searched recursively for *.json, leaving out what 'bibr chew -o' writes "
            "beside each export x.json (x.core.json, x.json.receipt.json, "
            "x.json.enrichment.json, _quarantine/); other JSON files and exports of "
            "another schema major are skipped."
        ),
    )
    tables_parser.add_argument(
        "inputs", nargs="+", help="Export JSON files or directories containing them"
    )
    tables_parser.add_argument(
        "--out",
        "-o",
        "--output",
        dest="out",
        required=True,
        help="Output directory for the .parquet files",
    )

    # --- references subcommand ---
    references_parser = sub.add_parser(
        "references",
        help="write the reference lists of JSON exports as CSL-JSON, BibTeX or RIS",
        description=(
            "Write the references of bibr JSON exports as one file for a reference "
            "manager: CSL-JSON (Zotero, Mendeley, Pandoc), BibTeX or RIS. Values are "
            "as printed; what a reference does not print is filled from the record "
            "Crossref enrichment matched to it, but only when the reference check "
            "found the two agree (bib_check status matched). A note on each reference "
            "says what the check found when it is not matched, and names a retraction "
            "or withdrawal of its record. Cite keys are unique across the file. "
            "Directories are searched recursively for *.json as for 'bibr tables'."
        ),
    )
    references_parser.add_argument(
        "inputs", nargs="+", help="Export JSON files or directories containing them"
    )
    references_parser.add_argument(
        "--format",
        choices=["csl-json", "bibtex", "ris"],
        default=None,
        help="Output format (default: from the output file's extension: .json, .bib, .ris)",
    )
    references_parser.add_argument(
        "-o",
        "--out",
        "--output",
        dest="out",
        default=None,
        help="Output file (default: standard output, which needs --format)",
    )
    references_parser.add_argument(
        "--prefer",
        choices=["printed", "record"],
        default="printed",
        help=(
            "printed (default): only what the reference printed; "
            "record: a matched record's values first, gaps such as a missing DOI "
            "filled from it, with a note naming those values (needs --crossref "
            "results; a record whose check did not match is never used)"
        ),
    )
    references_parser.add_argument(
        "--no-check-note",
        action="store_true",
        help="Leave out the note saying what the reference check found",
    )

    # --- inspect subcommand ---
    inspect_parser = sub.add_parser(
        "inspect",
        help="summarize an extraction-output JSON",
        description=(
            "Summarize a bibr JSON export: title/authors/DOI/paper type, "
            "section/sentence/table/figure/equation counts, reference and "
            "enrichment stats, the validation block, and per-model LLM "
            "token usage."
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
