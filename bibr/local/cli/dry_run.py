"""``--dry-run``: resolve and print the plan without processing anything.

Every helper below only reads already-resolved ``ResolvedRunConfig`` fields
or ``Settings`` values (or performs cheap platform/RAM probes identical to
ones the real run already does via ``resolve_run_config``/``_default_memory_mode``)
— none of them construct a pipeline, backend client, or HTTP/HF-hub session.
"""

from pathlib import Path

from bibr.local.cli import ui
from bibr.local.cli.inputs import _resolve_single_output_path
from bibr.local.cli.run_config import (
    ResolvedRunConfig,
    _managed_llm_model,
    _managed_llm_weight_repo,
    _preflight_local_backend,
    _preflight_ocr_runtime,
    _preflight_opencv,
)
from bibr.ocr.profiles import GLM_SERVED_MODEL_ALIAS

# Mirrors what ``resolve_served_model`` resolves for each HTTP backend, so the
# preview cannot drift from what the run actually asks the server for.
_OCR_HTTP_DEFAULT_SERVED_NAME = {
    "glm-http": GLM_SERVED_MODEL_ALIAS,
    "paddle-http": "paddle-ocr-vl-1.6",
    "serve-http": GLM_SERVED_MODEL_ALIAS,
}

# Field on ``Settings.ocr`` each local in-process OCR engine falls back to
# when no explicit ``--ocr-model`` is given — mirrors the ``model_path or
# Settings.ocr.<field>`` line in that backend class's own ``__init__``
# (bibr/local/ocr.py). Backends not listed here
# either have no local weights (HTTP/vision) or are handled separately.
_OCR_LOCAL_WEIGHT_FIELD = {
    "paddle-vllm": "paddle_model",
    "paddle-rapid-mlx": "paddle_rapid_mlx_model",
    "paddle-mlx-vlm": "paddle_mlx_model",
    "glm-rapid-mlx": "rapid_mlx_model",
    "glm-llama": "llama_cpp_model",
}

# Best-effort approximate download sizes for the dry-run preview. Not exact —
# just enough to warn a user before a multi-GB pull. Grounded where possible
# in in-repo documentation (cited below); entries left out still say "will
# download" (a download is happening either way) but with "(size unknown)"
# rather than guessing a figure.
_APPROX_MODEL_SIZES: dict[str, str] = {
    # bibr/local/segmenter.py module docstring: "Tiny model (~0.1 GB)".
    "segment-any-text/sat-6l-sm": "0.1 GB",
    # MiniLM two-head classifier, ~22M params (bibr/config.py comment).
    "scienceverse/bibr-section-classifier": "90 MB",
    # Frozen sklearn GBM bundle (joblib) — tabular model, negligible size.
    "scienceverse/bibr-geom-segmenter-v1": "1 MB",
    # answerdotai/ModernBERT-base (~150M params) encoder + CRF head.
    "scienceverse/bibr-segmenter-v1": "300 MB",
    "scienceverse/bibr-parser-v4-5-gold": "300 MB",
    # GLM-OCR weight repos — same figures as _OCR_BACKEND_DOWNLOAD_NOTES in
    # bibr/setup_wizard.py (bf16 ~2.7 GB, llama.cpp Q8_0 GGUF quant smaller).
    # Bare repo id — the ":Q8_0" quant suffix is stripped before lookup (see
    # _hf_cache_status's ``bare_repo_id``).
    "THUDM/GLM-OCR": "2.7 GB",
    "PaddlePaddle/PaddleOCR-VL-1.6": "1.8 GB",
    "olragon/PaddleOCR-VL-1.6-8bit": "1.0 GB",
    "mlx-community/GLM-OCR-bf16": "2.7 GB",
    "mlx-community/GLM-OCR-8bit": "1.4 GB",
    "ggml-org/GLM-OCR-GGUF": "1-2 GB",
    "mlx-community/Qwen3.5-4B-MLX-4bit": "2-3 GB",
    "qwen3.5-4b-4bit": "2-3 GB",
}

_HF_CACHE_REPO_ALIASES: dict[str, str] = {
    "qwen3.5-4b-4bit": "mlx-community/Qwen3.5-4B-MLX-4bit",
}


def _dry_run_ocr_model(config: ResolvedRunConfig) -> tuple[str, str | None]:
    """Return ``(model label to display, repo id to HF-cache-check or None)``.

    The repo id is ``None`` for backends that manage their own weights
    remotely (HTTP servers, cloud vision providers) — nothing local to check.
    """
    from bibr.config import Settings
    from bibr.ocr.registry import resolve_backend_candidates

    backend = config.ocr_backend
    if backend == "paddle":
        candidate = resolve_backend_candidates(backend, Settings)[0]
        # Served identity vs weight repo: the first candidate's display model
        # may be a server alias (paddle-vllm advertises
        # ``Settings.ocr.paddle_served_model``), while the HF cache holds the
        # repo its launcher loads (``Settings.ocr.paddle_model``). Map through
        # the same per-backend weight field the explicit-backend path uses so
        # the cache check cannot miss on the alias.
        weight_field = _OCR_LOCAL_WEIGHT_FIELD.get(candidate.backend)
        weight_repo = (
            config.ocr_model or getattr(Settings.ocr, weight_field)
            if weight_field is not None
            else candidate.model
        )
        return candidate.model, weight_repo
    if backend == "paddle-http":
        return config.ocr_model or Settings.ocr.paddle_served_model, None
    if backend in _OCR_HTTP_DEFAULT_SERVED_NAME:
        from bibr.config import snapshot_settings
        from bibr.ocr.profiles import resolve_served_model

        snapshot = snapshot_settings()
        served = config.ocr_model or resolve_served_model(
            requested_backend=backend,
            concrete_backend=backend,
            settings=snapshot,
            explicit_profile=config.ocr_profile or snapshot.ocr.profile,
        )
        return served, None
    if backend in ("gemini", "openai", "anthropic"):
        return config.ocr_model or Settings.ocr_vision.model, None
    field = _OCR_LOCAL_WEIGHT_FIELD.get(backend)
    if field is None:
        return config.ocr_model or "(unknown)", None
    repo = config.ocr_model or getattr(Settings.ocr, field)
    return repo, repo


def _dry_run_ocr_candidates(config: ResolvedRunConfig):
    """Return the ordered runtime identities without importing or starting one."""
    from bibr.config import Settings
    from bibr.ocr.registry import resolve_backend_candidates

    return resolve_backend_candidates(config.ocr_backend, Settings)


def _dry_run_device_label(config: ResolvedRunConfig) -> str:
    """Device the real run would select — same probe ``bibr doctor`` reports."""
    if config.device:
        return config.device
    try:
        from bibr.utils.device import detect_torch_device

        return detect_torch_device()
    except ImportError:
        # Core install: the local models run on ONNX Runtime.
        from bibr.utils.onnx_providers import get_ort_providers, selected_device

        return f"{selected_device(get_ort_providers(model_name='dry-run'))} (onnxruntime)"


def _dry_run_llm_label(config: ResolvedRunConfig) -> str:
    """LLM provider/model the real run would use, or the disabled marker."""
    if config.no_llm:
        return "disabled (--no-llm)"
    from bibr.config import Settings
    from bibr.local.pipeline import LOCAL_LLM_BACKENDS

    if config.llm_backend in LOCAL_LLM_BACKENDS:
        model = _managed_llm_model(config.llm_backend, Settings)
        return f"{config.llm_backend} (managed local server, model={model})"
    return f"{Settings.llm.provider}/{Settings.llm.model}"


def _dry_run_memory_mode_label(args, config: ResolvedRunConfig) -> str:
    """Resolved memory mode plus how it was decided (mirrors ``_default_memory_mode``)."""
    if args.memory is not None:
        return f"{config.memory_mode} (explicit --memory)"
    from bibr.config import Settings

    if Settings.pipeline.memory_mode:
        return f"{config.memory_mode} (PIPELINE_MEMORY_MODE setting)"
    return f"{config.memory_mode} (auto-detected from system RAM)"


def _dry_run_enrichment_lines(config: ResolvedRunConfig, parse_strategy: str) -> list[str]:
    """Crossref on/off (with the reason), resolver state, and consolidate mode.

    Mirrors ``ResolvedRunConfig.enrichment_enabled`` / ``RunConfig.enrichment_enabled``:
    ``no_llm`` forces ``crossref=False`` inside ``LocalPipeline.__init__``
    (never mind ``--crossref``'s own value) — checked first so the displayed
    reason matches what actually turned it off; then the explicit flag, then
    ``refs=off`` (nothing to enrich), then the ``CROSSREF_ENRICH`` setting,
    which is off unless the deployment opted in.
    """
    from bibr.config import Settings

    lines = []
    setting_explicit = "enrich" in Settings.crossref.model_fields_set
    crossref_on = (
        not config.no_llm
        and parse_strategy != "off"
        and (config.crossref if config.crossref is not None else bool(Settings.crossref.enrich))
    )
    if not crossref_on:
        if config.no_llm:
            reason = "--no-llm"
        elif config.crossref is False:
            reason = "--no-crossref"
        elif parse_strategy == "off":
            reason = "refs=off"
        elif setting_explicit:
            reason = "CROSSREF_ENRICH=false"
        else:
            reason = "off by default; enable with --crossref or CROSSREF_ENRICH=true"
        lines.append(f"Crossref: disabled ({reason})")
        return lines
    lines.append(
        "Crossref: enabled (--crossref)"
        if config.crossref is True
        else "Crossref: enabled (CROSSREF_ENRICH=true)"
    )
    if Settings.resolver.url and Settings.resolver.enrich:
        lines.append(f"  resolver: {Settings.resolver.url}")
    else:
        lines.append("  resolver: disabled")
    consolidate_mode = config.consolidate or Settings.crossref.consolidate
    lines.append(f"  consolidate: {consolidate_mode}")
    return lines


def display_wtpsplit_repo_id(value: str) -> str:
    """Return the runtime-resolved wtpsplit source for dry-run output."""
    from bibr.segmenter_base import resolve_wtpsplit_model

    resolved = resolve_wtpsplit_model(value)
    if resolved.is_local:
        return f"local bundle: {resolved.model_name}"
    if resolved.hub_prefix is None:
        return resolved.model_name
    return f"{resolved.hub_prefix}/{resolved.model_name}"


def _hf_cache_status(repo_id: str) -> str:
    """'cached' / 'will download (~size)' / 'will download (size unknown)' / cache-unknown.

    Reads the local Hugging Face cache index (``scan_cache_dir`` — no
    network) when ``huggingface_hub`` is importable; otherwise reports that
    the cache state can't be determined rather than guessing.
    """
    if repo_id.startswith("local bundle: "):
        return "available locally"
    bare_repo_id = _HF_CACHE_REPO_ALIASES.get(repo_id, repo_id).split(":", 1)[0]
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return "cache state unknown (ml extra not installed)"
    from bibr.config import Settings

    cache_dir = None
    if Settings.rapid_mlx.hf_hub_cache:
        cache_dir = Settings.rapid_mlx.hf_hub_cache
    elif Settings.rapid_mlx.hf_home:
        cache_dir = str(Path(Settings.rapid_mlx.hf_home) / "hub")
    cache_model_dir = (
        Path(cache_dir) / f"models--{bare_repo_id.replace('/', '--')}" if cache_dir else None
    )
    try:
        cache_info = scan_cache_dir(cache_dir=cache_dir) if cache_dir else scan_cache_dir()
        cached_ids = {repo.repo_id for repo in cache_info.repos}
    except Exception:  # noqa: BLE001 — a broken/unreadable cache dir shouldn't crash the preview
        if cache_model_dir is not None and cache_model_dir.exists():
            return "cached"
        return "cache state unknown (could not scan HF cache)"
    if cache_model_dir is not None and cache_model_dir.exists():
        return "cached"
    if bare_repo_id in cached_ids:
        return "cached"
    size = _APPROX_MODEL_SIZES.get(bare_repo_id)
    return f"will download (~{size})" if size else "will download (size unknown)"


def _dry_run_model_specs(
    config: ResolvedRunConfig,
    seg_strategy: str,
    parse_strategy: str,
    *,
    needs_ocr: bool,
) -> list[tuple[str, str]]:
    """``(label, repo_id)`` for every model this run's resolved config implies."""
    import importlib.util

    from bibr.config import Settings

    specs: list[tuple[str, str]] = []
    if needs_ocr:
        specs.append(("layout (PP-DocLayout)", Settings.layout.model_id))
    specs.append(
        ("sentence segmenter (wtpsplit)", display_wtpsplit_repo_id(Settings.WTPSPLIT_MODEL))
    )

    # The trained MiniLM section classifier needs the ml extra and is not
    # used at all under --no-llm (lookup-only alias path — see post_parse.py
    # ``_classify_sections``).
    if (
        not config.no_llm
        and importlib.util.find_spec("torch") is not None
        and Settings.ml.section_classifier_model_id
    ):
        specs.append(("section classifier", Settings.ml.section_classifier_model_id))

    # Reference seg/parse models only load under LLM-enabled PDF/OCR runs —
    # --no-llm skips reference extraction entirely for OCR'd input (see
    # post_parse.py ``_extract_metadata_and_equations``), and --refs off
    # skips it for every input type.
    if not config.no_llm and parse_strategy != "off":
        if seg_strategy == "geom":
            specs.append(("ref segmenter (geom)", Settings.REF_GEOM_SEG_MODEL_ID))
        elif seg_strategy == "crf":
            specs.append(("ref segmenter (crf)", Settings.NER_SEG_CKPT))
        if parse_strategy == "ner":
            specs.append(("ref parser (ner)", Settings.NER_PARSER_CKPT))

    if needs_ocr:
        _, ocr_weight_repo = _dry_run_ocr_model(config)
        if ocr_weight_repo:
            specs.append(("OCR weights", ocr_weight_repo))
    if not config.no_llm:
        from bibr.local.pipeline import LOCAL_LLM_BACKENDS

        if config.llm_backend in LOCAL_LLM_BACKENDS:
            llm_weight_repo = _managed_llm_weight_repo(config.llm_backend, Settings)
            if llm_weight_repo:
                specs.append(("LLM weights", llm_weight_repo))

    return specs


def _dry_run_output_destinations(
    args,
    files: list[Path],
    *,
    is_batch: bool,
    manifest_outputs: list[Path] | None = None,
) -> list[str]:
    """Resolved output path per file, without creating any directory.

    Mirrors the derivation ``_write_chunk_results`` uses at write time
    (``_resolve_single_output_path`` for single-file, ``<dir>/<stem>.json``
    for batch) but never calls ``_prepare_output_path`` — that helper's
    ``mkdir`` is exactly the side effect a preview must not have.
    """
    if manifest_outputs is not None:
        return [
            f"{source.name} -> {destination}"
            for source, destination in zip(files, manifest_outputs, strict=True)
        ]
    if args.output is None:
        return [f"{f.name} -> stdout" for f in files] if is_batch else ["stdout (no -o given)"]
    output_path = Path(args.output)
    if is_batch:
        return [f"{f.name} -> {output_path / f'{f.stem}.json'}" for f in files]
    if args.output.endswith(("/", "\\")):
        # Same directory intent ``_prepare_output_path`` acts on at write
        # time (it creates the directory there; the preview must not).
        return [f"{files[0].name} -> {output_path / f'{files[0].stem}.json'}"]
    target = _resolve_single_output_path(output_path, files[0])
    return [f"{files[0].name} -> {target}"]


def _print_dry_run_plan(
    args,
    config: ResolvedRunConfig,
    files: list[Path],
    *,
    is_batch: bool,
    manifest_outputs: list[Path] | None = None,
    blockers: list[str] | None = None,
) -> None:
    """Print the full resolved run plan and return — the ``--dry-run`` payload.

    Lines that embed a raw file path use plain ``print()`` rather than
    ``console.print()`` — Rich hard-wraps long lines at the terminal width
    with no regard for word boundaries, which can split a long absolute path
    mid-character (same reasoning as the stem-collision guard above). Model
    rows carry status glyphs, so they go through the console with
    ``soft_wrap=True`` — markup is rendered, but long lines are left for the
    terminal to wrap instead of being split by Rich.
    """
    from rich.console import Console

    from bibr.extract.ref_extractor import _resolve_ref_strategies

    out = Console()
    seg_strategy, parse_strategy = _resolve_ref_strategies(config.ref_seg, config.refs)
    needs_ocr = any(f.suffix.lower() == ".pdf" for f in files)

    ui.brand_header(
        out,
        f"bibr chew {ui.SEP} dry run",
        subtitle="plan preview — nothing will be processed",
    )

    ui.section(out, f"Input ({len(files)})")
    for f in files[:5]:
        print(f"  {f}")
    if len(files) > 5:
        print(f"  … and {len(files) - 5} more")

    model_label, _ = _dry_run_ocr_model(config)
    ui.section(out, "Plan")
    out.print(
        ui.kv(
            "ocr",
            f"{config.ocr_backend} {ui.SEP} {model_label} {ui.SEP} {_dry_run_device_label(config)}",
        )
    )
    candidates = _dry_run_ocr_candidates(config)
    if config.ocr_backend == "paddle":
        out.print("  OCR backend: paddle (automatic)")
        for index, candidate in enumerate(candidates, start=1):
            out.print(f"  {index}. {candidate.backend} | {candidate.model} | {candidate.profile}")
    else:
        candidate = candidates[0]
        out.print(f"  OCR backend: {candidate.backend}")
    out.print(f"  OCR model: {model_label}")
    out.print(f"  OCR profile: {config.ocr_profile or candidates[0].profile}")
    out.print(ui.kv("llm", _dry_run_llm_label(config)))
    refs_value = "disabled (--no-llm)"
    if not config.no_llm:
        refs_value = f"seg {seg_strategy} {ui.SEP} parse {parse_strategy}"
    out.print(ui.kv("refs", refs_value))
    enrich_lines = _dry_run_enrichment_lines(config, parse_strategy)
    crossref_value = enrich_lines[0].split(": ", 1)[-1]
    extras = [line.strip().replace(": ", " ", 1) for line in enrich_lines[1:]]
    if extras:
        crossref_value += f" {ui.SEP} " + f" {ui.SEP} ".join(extras)
    # soft_wrap: the reason text ("off by default; enable with ...") plus a
    # resolver URL can exceed a narrow terminal, and Rich would otherwise
    # hard-wrap it mid-token.
    out.print(ui.kv("crossref", crossref_value), soft_wrap=True)
    out.print(ui.kv("memory", _dry_run_memory_mode_label(args, config)))

    ui.section(out, "Models")
    specs = _dry_run_model_specs(config, seg_strategy, parse_strategy, needs_ocr=needs_ocr)
    if not specs:
        out.print("  [dim]none[/dim]")
    for label, repo_id in specs:
        status = _hf_cache_status(repo_id)
        if status in ("cached", "available locally"):
            mark = f"[green]{ui.OK}[/green]"
        elif status.startswith("will download"):
            mark = "[yellow]↓[/yellow]"
        else:
            mark = "[dim]?[/dim]"
        out.print(f"  {mark} {label}: {repo_id} — {status}", soft_wrap=True)

    ui.section(out, "Output")
    for line in _dry_run_output_destinations(
        args,
        files,
        is_batch=is_batch,
        manifest_outputs=manifest_outputs,
    ):
        print(f"  {line}")

    if blockers:
        ui.section(out, f"Blockers ({len(blockers)})")
        for blocker in blockers:
            out.print(f"  [red]{ui.FAIL}[/red] {blocker}", soft_wrap=True)
        out.print("[dim]The real run exits 1 on these; fix them before processing.[/dim]")

    out.print("\n[dim]Dry run — no files were processed.[/dim]")


def _dry_run_cloud_credential_blocker() -> str | None:
    """Missing-key verdict for a cloud LLM without building a client.

    ``--dry-run`` previews without touching any client machinery (no
    httpx, no model loads), so it cannot call ``preflight_credentials``
    (that builds a real provider client). This mirrors each bundled
    provider adapter's key lookup verbatim — same Settings fields, same
    messages (see ``bibr/clients/providers/*.py``) — and returns ``None``
    when the real run's check would pass. Third-party providers stay the
    real run's job to vet.
    """
    from bibr.clients import providers
    from bibr.config import snapshot_settings

    # A concrete snapshot, not the lazy proxy: ``providers.get`` takes a
    # ``GlobalSettings | None`` (see ``llm._get_provider``), and the copy
    # freezes the same values the real run's credential check reads.
    effective = snapshot_settings()
    name = (effective.llm.provider or "").lower()
    try:
        providers.get(name, settings=effective)
    except ValueError as exc:
        return str(exc)
    llm = effective.llm
    if name == "google":
        if not (llm.api_key or effective.GOOGLE_API_KEY):
            return (
                "Google API key required. Set LLM_API_KEY or GOOGLE_API_KEY environment variable."
            )
    elif name == "anthropic":
        if not (llm.api_key or effective.ANTHROPIC_API_KEY):
            return (
                "Anthropic API key required. "
                "Set LLM_API_KEY or ANTHROPIC_API_KEY environment variable."
            )
    elif name == "groq":
        if not (llm.api_key or effective.GROQ_API_KEY):
            return "Groq API key required. Set LLM_API_KEY or GROQ_API_KEY environment variable."
    elif name == "openai" and not llm.api_key and not llm.base_url:
        return "OpenAI API key required. Set LLM_API_KEY environment variable."
    return None


def _dry_run_blockers(config, files: list, missing_count: int) -> list[str]:
    """Cheap preflight verdicts for ``--dry-run``: blockers the real run exits 1 on.

    Covers missing inputs, LLM credentials (provider key lookup only —
    no client is built), the managed local-LLM backend, and the PDF
    OCR/image runtime. Nothing is constructed or downloaded; every probe
    is local and cheap.
    """
    blockers: list[str] = []
    if missing_count:
        blockers.append(f"{missing_count} input(s) not found — the real run counts them as errors.")
    if not config.no_llm and config.llm_backend == "cloud":
        credential_blocker = _dry_run_cloud_credential_blocker()
        if credential_blocker is not None:
            blockers.append(credential_blocker)
    from bibr.local.pipeline import LOCAL_LLM_BACKENDS

    if not config.no_llm and config.llm_backend in LOCAL_LLM_BACKENDS:
        err = _preflight_local_backend(config.llm_backend)
        if err:
            blockers.append(err)
    if any(p.suffix.lower() == ".pdf" for p in files):
        opencv_problem = _preflight_opencv()
        if opencv_problem is not None:
            message, repair = opencv_problem
            blockers.append(f"{message} (repair with: {repair})")
        ocr_reason = _preflight_ocr_runtime(config)
        if ocr_reason is not None:
            blockers.append(ocr_reason)
    return blockers
