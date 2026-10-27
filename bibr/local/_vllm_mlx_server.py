"""Launcher shim for ``vllm_mlx.server`` with conservative hybrid-cache handling.

Older vllm-mlx versions monkeypatched mlx-lm's cache factory for Mamba-style
models. With current mlx-lm, ArraysCache already supports the batch operations
needed by hybrid ArraysCache + KVCache models (Qwen3-Next / NuExtract3), and
the old monkeypatch can crash those models with
``ArraysCache.__init__() missing 1 required positional argument: 'size'``.

vllm-mlx 0.4.x no-ops this path upstream, but bibr still enters through this
shim so managed OCR and LLM launchers have one stable module target and remain
safe if a stale environment resolves an older vllm-mlx.

``python -m vllm_mlx.server`` has no ``--served-model-name`` (only the
``vllm-mlx serve`` CLI does), so a local model directory would answer to its
path. The shim takes ``--served-model-name <name>`` itself and passes it to
``vllm_mlx.server.load_model``, as ``vllm-mlx serve`` does, so a pinned build
bibr downloaded is served under its Hub id.

Usage (drop-in for ``python -m vllm_mlx.server``):
    python -m bibr.local._vllm_mlx_server --model <model> --port <port> ...
"""

import importlib.util
import runpy
import sys


def _disable_legacy_mamba_patch() -> None:
    try:
        mamba_cache_spec = importlib.util.find_spec("vllm_mlx.utils.mamba_cache")
    except (ImportError, ModuleNotFoundError, ValueError):
        return
    if mamba_cache_spec is None:
        return

    from vllm_mlx.utils import mamba_cache

    def _ensure_mamba_support_noop():
        # mlx-lm >= 0.30.6: ArraysCache batches natively; the old patch breaks
        # hybrid ArraysCache + KVCache models. Mark patched without patching.
        mamba_cache._patched = True

    mamba_cache.ensure_mamba_support = _ensure_mamba_support_noop


def _pop_served_model_name(argv: list[str]) -> str | None:
    """Remove ``--served-model-name <name>`` (or ``=<name>``) from *argv*; return the name."""
    flag = "--served-model-name"
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            name = argv[index + 1]
            del argv[index : index + 2]
            return name
        if arg.startswith(flag + "="):
            del argv[index]
            return arg.split("=", 1)[1]
    return None


def _run_served_as(name: str) -> None:
    """Run vllm-mlx's server ``main()`` with its model served as *name*."""
    from vllm_mlx import server

    load_model = server.load_model

    def _load_model_served_as(*args, **kwargs):
        kwargs.setdefault("served_model_name", name)
        return load_model(*args, **kwargs)

    # main() looks load_model up in its module's globals when it runs.
    server.load_model = _load_model_served_as
    server.main()


def _main(argv: list[str]) -> None:
    _disable_legacy_mamba_patch()
    served_model_name = _pop_served_model_name(argv)
    if served_model_name is None:
        runpy.run_module("vllm_mlx.server", run_name="__main__")
    else:
        _run_served_as(served_model_name)


if __name__ == "__main__":
    _main(sys.argv)
