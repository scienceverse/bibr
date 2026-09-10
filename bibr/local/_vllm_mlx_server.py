"""Launcher shim for ``vllm_mlx.server`` with conservative hybrid-cache handling.

Older vllm-mlx versions monkeypatched mlx-lm's cache factory for Mamba-style
models. With current mlx-lm, ArraysCache already supports the batch operations
needed by hybrid ArraysCache + KVCache models (Qwen3-Next / NuExtract3), and
the old monkeypatch can crash those models with
``ArraysCache.__init__() missing 1 required positional argument: 'size'``.

vllm-mlx 0.4.x no-ops this path upstream, but bibr still enters through this
shim so managed OCR and LLM launchers have one stable module target and remain
safe if a stale environment resolves an older vllm-mlx.

Usage (drop-in for ``python -m vllm_mlx.server``):
    python -m bibr.local._vllm_mlx_server --model <model> --port <port> ...
"""

import importlib.util
import runpy


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


if __name__ == "__main__":
    _disable_legacy_mamba_patch()
    runpy.run_module("vllm_mlx.server", run_name="__main__")
