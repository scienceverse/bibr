"""Runtime probes for the managed vllm-mlx backend."""

from __future__ import annotations

import importlib.metadata
import importlib.util


def vllm_mlx_unavailable_reason() -> str | None:
    """Return why vllm-mlx cannot be launched, or ``None`` when it looks usable."""
    try:
        importlib.metadata.version("vllm-mlx")
    except importlib.metadata.PackageNotFoundError:
        return "package metadata for vllm-mlx was not found"

    try:
        server_spec = importlib.util.find_spec("vllm_mlx.server")
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return f"vllm_mlx.server could not be inspected: {exc}"
    if server_spec is None:
        return "vllm_mlx.server was not found"
    return None


def vllm_mlx_available() -> bool:
    """Whether the vllm-mlx package has a runnable server entrypoint."""
    return vllm_mlx_unavailable_reason() is None
