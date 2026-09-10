"""``bibr doctor`` — validate the local setup and environment."""

import sys
from pathlib import Path

from bibr.exceptions import ConfigurationError
from bibr.local.cli import ui
from bibr.local.cli.run_config import _managed_llm_model


def _probe_ocr_url(url: str, timeout: float = 2.0) -> bool:
    """Reachability probe for an OCR HTTP server.

    Returns True if any of the standard endpoints respond (status doesn't
    matter — even 404 means *something* is listening). Returns False on
    connection refused / DNS failure / timeout. Kept synchronous because
    doctor itself is synchronous; a tiny stdlib HTTP request avoids
    pulling httpx eagerly into the doctor path.
    """
    import http.client
    import urllib.parse

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    conn_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    base_path = parsed.path.rstrip("/")
    for path in ("/v1/models", "/health", "/"):
        conn = None
        try:
            conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
            conn.request("GET", f"{base_path}{path}" if base_path else path)
            conn.getresponse()
            return True
        except (OSError, TimeoutError, http.client.HTTPException, ValueError):
            continue
        finally:
            if conn is not None:
                conn.close()
    return False


def _opencv_unavailable_reason() -> str | None:
    try:
        import cv2
    except ImportError:
        return "opencv (cv2) not installed"
    if not hasattr(cv2, "resize"):
        return "cv2 module is incomplete (missing 'resize')"
    return None


def _paddle_mlx_vlm_status(model: str) -> tuple[bool, str]:
    """Return managed MLX-VLM launch/cache evidence without starting it."""
    import importlib.util
    import shutil

    try:
        package_available = importlib.util.find_spec("mlx_vlm.server") is not None
    except ModuleNotFoundError:
        package_available = False
    if not package_available and shutil.which("uv") is None:
        return False, "launch unavailable (mlx-vlm package and uv runner are both absent)"

    from bibr.local.cli.dry_run import _hf_cache_status

    launch = "installed mlx-vlm package" if package_available else "uv-managed mlx-vlm runner"
    return True, f"launch: {launch}; model cache: {_hf_cache_status(model)}"


def _paddle_vllm_status(model: str) -> tuple[bool, str]:
    """Return vLLM launch/cache evidence without loading Paddle weights."""
    import importlib.util
    import shutil

    package_available = importlib.util.find_spec("vllm") is not None
    uv_available = shutil.which("uv") is not None
    if not package_available and not uv_available:
        return (
            False,
            "launch unavailable; vllm package absent and uv runner unavailable; model cache not checked",
        )

    from bibr.local.cli.dry_run import _hf_cache_status

    launch = "importable vllm package" if package_available else "uv isolated vllm runner"
    return True, f"launch: {launch}; model cache: {_hf_cache_status(model)}"


def _check_paddle_rapid_mlx(backend: str, ok, warn, fail) -> None:  # noqa: ARG001
    """Report Rapid-MLX as a candidate until image OCR smoke proves it."""
    from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

    unavailable_reason = rapid_mlx_unavailable_reason()
    if unavailable_reason is not None:
        fail(
            f"OCR backend: {backend} — Rapid-MLX executable unavailable ({unavailable_reason})",
            hint="Install with: pip install 'rapid-mlx[guided]'",
        )
        return
    warn(
        f"OCR backend: {backend} — Rapid-MLX executable found, but Paddle OCR capability "
        "unverified",
        hint="Requires a real managed startup OCR smoke with an image and Paddle prompt.",
    )


def _check_paddle_http(backend: str, url: str | None, ok, warn) -> None:
    """Check the selected external Paddle server instead of declaring it local-ready."""
    label = f"OCR backend: {backend} (Paddle OCR remote"
    if not url:
        warn(f"{label}: not set)", hint="Set OCR_BASE_URL in .env")
        return
    if not _probe_ocr_url(url):
        warn(
            f"{label}: {url} — unreachable)",
            hint="Server may be offline; check OCR_BASE_URL and network",
        )
        return
    ok(f"{label}: {url})")


def _check_ocr_backend(ok, warn, fail) -> None:
    """Probe whether the configured OCR backend is importable and instantiable.

    The check stops short of actually starting a server / loading weights —
    it just verifies the necessary Python deps are present so users don't
    discover ``vllm-mlx not installed`` halfway through their first run.
    """
    import importlib.util

    try:
        from bibr.config import Settings

        backend = Settings.ocr.backend
    except Exception as e:  # noqa: BLE001
        fail(f"OCR config: {e}")
        return

    if backend == "paddle-http":
        _check_paddle_http(backend, getattr(Settings, "OCR_BASE_URL", None), ok, warn)
        return
    if backend == "paddle-rapid-mlx":
        _check_paddle_rapid_mlx(backend, ok, warn, fail)
        return
    if backend == "paddle-vllm":
        available, status = _paddle_vllm_status(Settings.ocr.paddle_model)
        reporter = warn if available else fail
        reporter(
            f"OCR backend: {backend} — {status}; Paddle OCR capability unverified",
            hint="Requires managed startup and model smoke before it can be declared ready.",
        )
        return
    if backend == "paddle-mlx-vlm":
        from bibr.ocr.registry import resolve_backend_candidates

        candidate = resolve_backend_candidates(backend, Settings)[0]
        available, status = _paddle_mlx_vlm_status(candidate.model)
        reporter = warn if available else fail
        reporter(
            f"OCR backend: {backend} — {status}; Paddle OCR capability unverified",
            hint="Requires managed startup before the fallback can be declared ready.",
        )
        return

    # ``paddle`` is an automatic startup chain, not proof that a particular
    # local engine can OCR. In particular, finding a Rapid-MLX executable does
    # not prove it accepts an image plus the Paddle prompt; only the managed
    # startup smoke can establish that capability. Keep doctor honest and name
    # the managed MLX-VLM fallback rather than declaring the chain ready.
    if backend == "paddle":
        import importlib.util

        from bibr.ocr.registry import resolve_backend_candidates

        candidates = resolve_backend_candidates(backend, Settings)
        primary = candidates[0]
        if primary.backend == "paddle-vllm":
            _available, status = _paddle_vllm_status(Settings.ocr.paddle_model)
            warn(
                "OCR backend: paddle (automatic) — paddle-vllm primary: "
                f"{status}; Paddle OCR capability unverified",
                hint="Automatic fallback continues to GLM if managed Paddle startup fails.",
            )
            return
        fallback = next(
            (candidate for candidate in candidates if candidate.backend == "paddle-mlx-vlm"),
            None,
        )
        if fallback is None:
            fallback_label = "The automatic chain retains explicit GLM fallback candidates."
        else:
            try:
                package_available = importlib.util.find_spec("mlx_vlm.server") is not None
            except ModuleNotFoundError:
                package_available = False
            if not package_available:
                fallback_label = (
                    f"Managed MLX-VLM fallback: {fallback.backend} ({fallback.model}); "
                    "package unavailable (mlx-vlm is not installed), so its model cache was not checked."
                )
            else:
                from bibr.local.cli.dry_run import _hf_cache_status

                fallback_label = (
                    f"Managed MLX-VLM fallback: {fallback.backend} ({fallback.model}); "
                    f"package available; model cache: {_hf_cache_status(fallback.model)}."
                )
        if primary.backend == "paddle-rapid-mlx":
            from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

            unavailable_reason = rapid_mlx_unavailable_reason()
            if unavailable_reason is not None:
                warn(
                    "OCR backend: paddle (automatic) — Rapid-MLX executable unavailable "
                    f"({unavailable_reason})",
                    hint=fallback_label,
                )
                return
            warn(
                "OCR backend: paddle (automatic) — Rapid-MLX executable found, "
                "but Paddle OCR capability unverified",
                hint="Requires a real managed startup OCR smoke. " + fallback_label,
            )
            return
        warn(
            f"OCR backend: paddle (automatic) — {primary.backend} package availability and "
            "model-cache readiness are unverified",
            hint=fallback_label,
        )
        return

    # Resolve auto-aliases to a concrete backend on this platform (shared
    # registry table) so the dependency check below inspects the right package.
    from bibr.ocr.registry import resolve_backend_name

    resolved = resolve_backend_name(backend)

    label = f"OCR backend: {backend}"
    if resolved != backend:
        label = f"OCR backend: {backend} → {resolved}"

    # Soft issues (e.g. CPU-only llama.cpp) are reported as warnings after the
    # OpenCV check so a broken opencv install still hard-fails first.
    soft_warn: tuple[str, str] | None = None

    # Each backend has a known dependency profile. Check the deps without
    # actually starting any server / loading any model.
    if resolved == "glm-mlx":
        # Disabled outright (bibr.local.ocr.raise_vllm_mlx_ocr_disabled): the
        # vllm-mlx --mllm path emitted corrupted text and leaks memory, so a
        # config that still names it fails here instead of at the first chew.
        fail(
            f"{label}: the vllm-mlx OCR backend is disabled (it produced corrupted text and "
            "leaks memory on bibr's workload)",
            hint="Use OCR_BACKEND=paddle (automatic selector) or OCR_BACKEND=glm-rapid-mlx "
            "(pip install 'rapid-mlx[guided]')",
        )
        return
    elif resolved == "glm-rapid-mlx":
        from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

        unavailable_reason = rapid_mlx_unavailable_reason()
        if unavailable_reason is not None:
            fail(
                f"{label}: rapid-mlx launcher not available ({unavailable_reason})",
                hint="Install with: pip install 'rapid-mlx[guided]'",
            )
            return
    elif resolved == "glm-llama":
        from bibr.local.llama_cpp import (
            cuda_steering_hint,
            find_llama_server,
            install_hint,
            probe_backend_kind,
            probe_gpu_backend,
        )

        prefix = find_llama_server()
        if prefix is None:
            fail(
                f"{label}: llama.cpp not installed",
                hint=install_hint(),
            )
            return
        gpu = probe_gpu_backend(prefix)
        if gpu is False:
            soft_warn = (
                f"{label}: llama.cpp looks CPU-only (no CUDA/Metal/Vulkan backend)",
                install_hint(),
            )
        elif gpu is True:
            label = f"{label} (GPU backend detected)"
            # The Vulkan build works on NVIDIA but a CUDA build is faster for
            # bibr's prefill-heavy OCR — surface that as a soft warning.
            steer = cuda_steering_hint(probe_backend_kind(prefix))
            if steer is not None:
                soft_warn = (
                    f"{label}: Vulkan build on an NVIDIA GPU — a CUDA build is faster",
                    steer,
                )
        else:
            label = f"{label} (llama.cpp found; GPU backend not verified)"
    elif resolved in ("glm-http", "serve-http"):
        # Remote — probe the URL with a short timeout so we catch the case
        # where the OCR host is down or the URL is stale (a misleading green
        # check is worse than a slightly slower doctor).
        ocr_url = getattr(Settings, "OCR_BASE_URL", None)
        if not ocr_url:
            warn(f"{label} (remote: not set)", hint="Set OCR_BASE_URL in .env")
            return
        if not _probe_ocr_url(ocr_url):
            warn(
                f"{label} (remote: {ocr_url} — unreachable)",
                hint="Server may be offline; check OCR_BASE_URL and network",
            )
            return
        ok(f"{label} (remote: {ocr_url})")
        return
    elif resolved in ("gemini", "openai", "anthropic"):
        # Vision-LLM OCR — uses the LLM provider keys.
        ok(f"{label}")
        return

    # Verify cv2 is functional — a broken opencv install is a common silent
    # failure (layout post-processing crashes mid-run). Only the torch layout
    # path can reach it: transformers' image processor imports cv2, while the
    # ONNX path a core install uses is Pillow/numpy throughout.
    if importlib.util.find_spec("torch") is not None:
        opencv_reason = _opencv_unavailable_reason()
        if opencv_reason is not None:
            fail(
                f"{label}: {opencv_reason}",
                hint=(
                    "Install with: uv sync --extra torch"
                    if "not installed" in opencv_reason
                    else "Reinstall: uv pip install --reinstall opencv-python-headless"
                ),
            )
            return

    if soft_warn is not None:
        msg, hint = soft_warn
        warn(msg, hint=hint)
        return

    ok(label)


def _check_device(ok, warn, fail) -> None:
    """Report the torch device the pipeline will actually use.

    A CUDA GPU that is present but unsupported by the installed torch wheel
    (e.g. Pascal sm_61 with a sm_75+ build) is the one case that must NOT be
    a green check: ``is_available()`` is True, yet every kernel launch would
    crash. The device ladder skips such GPUs; surface that here.
    """
    try:
        import torch

        from bibr.utils.device import cuda_incompatibility, detect_torch_device

        reason = cuda_incompatibility()
        device = detect_torch_device()
        if reason is not None:
            warn(
                f"Device: {device} (GPU present but unusable for PyTorch: {reason})",
                hint=(
                    "Layout/NER fall back to CPU automatically. On older cards (e.g. GTX 10-series) "
                    "use --ocr glm-llama and --llm llama-cpp for GPU inference via llama.cpp; "
                    "do not install the gpu/local-cuda extras or try a different PyTorch wheel"
                ),
            )
        elif device == "cuda":
            ok(f"Device: cuda ({torch.cuda.get_device_name(0)})")
        elif device == "mps":
            ok("Device: mps (Apple Silicon)")
        else:
            ok("Device: cpu")
    except ImportError:
        # Core install: bibr's own models run on ONNX Runtime, so report the
        # provider it would pick rather than calling a missing torch a failure.
        from bibr.utils.onnx_providers import get_ort_providers, selected_device

        device = selected_device(get_ort_providers(model_name="doctor"))
        ok(f"Device: {device} (ONNX Runtime; torch not installed)")
    except Exception as e:  # noqa: BLE001
        fail(f"Device: {e}")


def _check_ref_strategies(ok, fail) -> None:
    """Doctor check: resolved reference strategies, plus ML deps when needed."""
    import importlib.util

    from bibr.extract.extractor import _resolve_ref_strategies

    seg_strategy, parse_strategy = _resolve_ref_strategies()
    if parse_strategy == "off":
        # No segmentation or parsing runs at all — no deps to verify.
        ok("References: extraction disabled (parse=off)")
        return
    label = f"References: seg={seg_strategy}, parse={parse_strategy}"
    torch_missing = importlib.util.find_spec("torch") is None
    if seg_strategy == "crf" and torch_missing:
        # The CRF segmenter has no ONNX export; it is a torch-only opt-in.
        fail(
            f"{label}: the crf segmenter is torch-only",
            hint="Install with: uv sync --extra torch, or set REF_SEG_STRATEGY=geom",
        )
        return
    if parse_strategy == "ner" and torch_missing:
        from bibr.config import Settings

        if Settings.ml.runtime == "torch":
            fail(
                f"{label}: ML_RUNTIME=torch but torch is not installed",
                hint="Install with: uv sync --extra torch, or set ML_RUNTIME=auto",
            )
            return
        # A core install parses references through the ONNX bundle, which is
        # fetched from the Hub on first use — nothing to verify offline here.
        ok(f"{label} (ONNX Runtime)")
        return
    if seg_strategy == "geom" and importlib.util.find_spec("sklearn") is None:
        # scikit-learn is a core dependency, so this only fires on a damaged
        # environment. geom cascades to LLM seg, hence a soft failure.
        fail(
            f"{label}: geom segmenter unavailable, will cascade to LLM seg",
            hint="Reinstall bibr, or set REF_SEG_STRATEGY=llm",
        )
        return
    ok(label)


def _check_llm_local_backend(backend: str, model: str, ok, fail) -> None:
    """Report a managed local LLM backend + launcher availability.

    Local backends own their own server, so there is no API key to check —
    report the backend + model and verify the launcher is installed instead.
    """
    import importlib.util
    import shutil

    if backend == "vllm":
        vllm_installed = importlib.util.find_spec("vllm") is not None
        available = vllm_installed or shutil.which("uv") is not None
        hint = "Install with: uv sync --extra vllm, or install uv (https://astral.sh/uv)"
        if available and not vllm_installed:
            # Honest about the cost: the first chew bootstraps vLLM through uv
            # (several GB), and on Python 3.14 — where the vllm extra installs
            # nothing because vllm==0.27.0 has no 3.14 wheels — inside a
            # managed Python 3.13.
            note = "uv-managed vLLM runner; the first run downloads several GB"
            if sys.version_info >= (3, 14):
                note += (
                    " into a managed Python 3.13 (vllm has no 3.14 wheels, so "
                    "`uv sync --extra vllm` installs nothing on this interpreter)"
                )
            ok(f"LLM backend: {backend} ({note}, model={model}) — {hint}")
            return
    elif backend == "vllm-mlx":
        from bibr.local.vllm_mlx_runtime import vllm_mlx_unavailable_reason

        unavailable_reason = vllm_mlx_unavailable_reason()
        available = unavailable_reason is None
        hint = "Install with: uv sync --extra local-mlx  (or uv pip install vllm-mlx)"
    elif backend == "rapid-mlx":
        from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

        unavailable_reason = rapid_mlx_unavailable_reason()
        available = unavailable_reason is None
        hint = "Install with: pip install 'rapid-mlx[guided]'"
    elif backend == "llama-cpp":
        from bibr.local.llama_cpp import (
            cuda_steering_hint,
            find_llama_server,
            install_hint,
            probe_backend_kind,
            probe_gpu_backend,
        )

        prefix = find_llama_server()
        if prefix is None:
            fail(
                f"LLM backend: {backend} — launcher not available",
                hint=install_hint(),
            )
            return
        gpu = probe_gpu_backend(prefix)
        if gpu is False:
            # Soft: binary is present but will crawl on CPU. Doctor's ok/fail
            # API has no warn channel here, so encode the upgrade path in the
            # status line (install_hint is also on the OCR glm-llama check).
            ok(
                f"LLM backend: {backend} (CPU-only llama.cpp, model={model}) — "
                f"very slow; {install_hint()}"
            )
            return
        if gpu is True:
            # Vulkan-on-NVIDIA runs but is slower than CUDA for bibr's prefill-
            # heavy workload; encode the steering hint in the status line (no
            # warn channel here, same as the CPU-only case above).
            steer = cuda_steering_hint(probe_backend_kind(prefix))
            if steer is not None:
                ok(f"LLM backend: {backend} (Vulkan build on NVIDIA GPU, model={model}) — {steer}")
                return
            ok(f"LLM backend: {backend} (GPU, managed local server, model={model})")
            return
        ok(f"LLM backend: {backend} (managed local server, model={model})")
        return
    else:  # llmster
        available = shutil.which("lms") is not None
        hint = "Install manually: curl -fsSL https://lmstudio.ai/install.sh | bash"

    if available:
        ok(f"LLM backend: {backend} (managed local server, model={model})")
    else:
        fail(f"LLM backend: {backend} — launcher not available", hint=hint)


def _check_llm_connection(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    console,
    ok,
    warn,  # noqa: ARG001
    fail,
) -> None:
    """Ping the cloud LLM provider once to confirm the key + network work.

    On failure the raw SDK exception is redacted before display: google-genai and
    other SDKs frequently embed ``?key=<API_KEY>`` in exception URLs, and doctor
    output is routinely pasted into bug reports (audit M2).
    """
    try:
        import instructor
        from pydantic import BaseModel, Field

        class _Ping(BaseModel):
            reply: str = Field(description="Your reply")

        kwargs: dict = {}
        if provider == "google":
            kwargs["api_key"] = api_key
        elif provider == "openai":
            kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
                # Mirror OpenAIProvider: local OpenAI-compatible servers
                # need JSON_SCHEMA mode (they reject object-form tool_choice).
                kwargs["mode"] = instructor.Mode.JSON_SCHEMA
        elif provider in ("anthropic", "groq"):
            kwargs["api_key"] = api_key

        client = instructor.from_provider(f"{provider}/{model}", **kwargs)
        create_kwargs: dict = {}
        if provider == "google":
            create_kwargs["generation_config"] = {"max_tokens": 64}
        else:
            create_kwargs["max_tokens"] = 64
        with console.status(f"  [dim]contacting {provider}…[/dim]"):
            client.create(
                response_model=_Ping,
                messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                **create_kwargs,
            )
        ok("LLM connection OK")
    except Exception as e:
        from bibr.utils.redact import redact_key

        fail(
            f"LLM connection failed: {redact_key(str(e), api_key or '')}",
            hint="Check your API key and network connection",
        )


def _run_doctor() -> None:
    """Validate the bibr setup and print a diagnostic report."""
    from rich.console import Console

    console = Console()
    counts = {"ok": 0, "warn": 0, "fail": 0}

    def ok(msg: str) -> None:
        counts["ok"] += 1
        ui.ok(console, msg)

    def warn(msg: str, hint: str = "") -> None:
        counts["warn"] += 1
        ui.warn(console, msg, hint=hint)

    def fail(msg: str, hint: str = "") -> None:
        counts["fail"] += 1
        ui.fail(console, msg, hint=hint)

    ui.brand_header(console, "bibr doctor", subtitle="validating your bibr setup")

    # --- Environment (Settings-free) ----------------------------------------
    ui.section(console, "Environment")

    v = sys.version_info
    py_ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        ok(f"Python {py_ver}")
    else:
        fail(f"Python {py_ver}", hint="bibr requires Python 3.11+")

    import shutil

    if shutil.which("uv"):
        ok("uv available")
    else:
        fail("uv not on PATH", hint="Install from https://docs.astral.sh/uv/")

    # libmagic is a system library python-magic only binds to; a fresh install
    # on macOS/Linux can be missing it and every extraction dies on the first
    # file sniff (issue #64). Name it here, where users come to find it.
    from bibr.input.validate import libmagic_unavailable_reason

    libmagic_problem = libmagic_unavailable_reason()
    if libmagic_problem is None:
        ok("libmagic available")
    else:
        fail("libmagic not found", hint=libmagic_problem)

    # --- Configuration -------------------------------------------------------
    ui.section(console, "Configuration")

    if Path(".env").exists():
        ok(".env file found")
    else:
        fail(".env not found", hint="Run bibr setup or cp .env.example .env")

    # Settings validity — a bad .env value would otherwise crash every check
    # below on first access. Surface it as a failed check (naming the env
    # var(s) + allowed values) and keep running the Settings-free diagnostics.
    import bibr.config
    from bibr.local.pipeline import LOCAL_LLM_BACKENDS

    settings = bibr.config.Settings
    config_ok = True
    _llm_local = False
    api_key = provider = model = None
    try:
        _llm_local = settings.llm.backend in LOCAL_LLM_BACKENDS
    except ConfigurationError as e:
        config_ok = False
        for problem in e.problems or [str(e)]:
            fail(
                f"Config: {problem}", hint="Fix it in your .env or environment, then re-run doctor"
            )

    # --- LLM (provider + key, or managed local backend; needs Settings) ------
    if config_ok:
        ui.section(console, "LLM")

    if not config_ok:
        pass
    elif _llm_local:
        # Managed local server: report backend + model, check the launcher, and
        # skip the cloud API-key + connection tests below (nothing to reach yet).
        _check_llm_local_backend(
            settings.llm.backend,
            _managed_llm_model(settings.llm.backend, settings),
            ok,
            fail,
        )
    else:
        try:
            provider = settings.llm.provider
            model = settings.llm.model

            # Determine API key based on provider
            api_key = None
            if provider == "google":
                api_key = settings.GOOGLE_API_KEY
            elif provider == "anthropic":
                api_key = settings.ANTHROPIC_API_KEY
            elif provider == "groq":
                api_key = settings.GROQ_API_KEY
            elif provider == "openai":
                api_key = settings.llm.api_key
            elif provider == "ollama":
                api_key = "n/a"  # Ollama doesn't need a key

            if api_key:
                ok(f"LLM provider: {provider} ({model})")
            else:
                fail(
                    f"LLM provider: {provider} — no API key",
                    hint="Run bibr setup or set the key in .env",
                )
        except Exception as e:
            fail(f"LLM config: {e}", hint="Run bibr setup to configure")
            api_key = None
            provider = None
            model = None

        # LLM connection test (cloud backends only) — the one check that can
        # take seconds, so it gets a spinner.
        if api_key and provider and provider != "ollama":
            _check_llm_connection(
                provider, model, api_key, settings.llm.base_url, console, ok, warn, fail
            )
        elif provider == "ollama":
            warn("LLM connection: Ollama (not tested)", hint="Start Ollama and test manually")
        else:
            fail("LLM connection: skipped", hint="Configure an API key first")

    # --- OCR (needs Settings) -------------------------------------------------
    if config_ok:
        ui.section(console, "OCR")
        _check_ocr_backend(ok, warn, fail)

    # --- Pipeline -------------------------------------------------------------
    ui.section(console, "Pipeline")
    _check_device(ok, warn, fail)
    if config_ok:
        _check_ref_strategies(ok, fail)

    # --- Services --------------------------------------------------------------
    if config_ok:
        ui.section(console, "Services")
        try:
            redis_url = bibr.config.Settings.redis.url
            if redis_url and bibr.config.Settings.redis.password:
                ok(
                    f"Redis: configured "
                    f"({redis_url.split('@')[-1] if '@' in redis_url else redis_url})"
                )
            elif redis_url:
                warn("Redis: configured (no password)", hint="Set REDIS_PASSWORD for production")
            else:
                warn("Redis: not configured", hint="Only needed for bibr serve with caching")
        except Exception:
            warn("Redis: not configured", hint="Only needed for bibr serve with caching")

    # --- Summary ---------------------------------------------------------------
    console.print()
    ui.rule(console)
    parts = [f"[green]{counts['ok']} ok[/green]"]
    if counts["warn"]:
        n = counts["warn"]
        parts.append(f"[yellow]{n} warning{'s' if n != 1 else ''}[/yellow]")
    if counts["fail"]:
        parts.append(f"[red]{counts['fail']} failed[/red]")
    sep = f" [dim]{ui.SEP}[/dim] "
    console.print(f"  {sep.join(parts)}")

    if counts["fail"]:
        console.print("  [dim]Fix the failing checks above, then re-run bibr doctor.[/dim]")
        sys.exit(1)
