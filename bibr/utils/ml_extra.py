"""Clear ImportError for dependencies provided by the optional 'torch' extra.

The extra was called ``ml`` before 0.6; ``ml`` remains an alias of ``torch``
in ``pyproject.toml`` so old install commands keep working. Messages name the
canonical extra.
"""

TORCH_EXTRA = "torch"
TORCH_EXTRA_HINT = "pip install 'bibr[torch]' (or uv sync --extra torch)"


def ml_import_error(feature: str) -> ImportError:
    """Build the ImportError to raise when a 'torch'-extra dependency is missing."""
    return ImportError(f"{feature} requires the '{TORCH_EXTRA}' extra: {TORCH_EXTRA_HINT}")


def onnxruntime_import_error(feature: str) -> ImportError:
    """Build the ImportError to raise when onnxruntime (a core dependency) is missing."""
    return ImportError(
        f"{feature} requires onnxruntime, which ships with the core install: "
        "pip install onnxruntime (or reinstall bibr)"
    )
