"""Removed dead helpers stay removed (x-static-triage-6).

Each name below had no caller anywhere in bibr/, tests/, scripts/,
evaluation/ or docs/ — only its own definition. The rest of the finding's
candidates (BaseClient, usage_for_file, require_api_key, wait_closed_tasks,
PaperType, the floats wrappers, FootnoteBuffer.is_empty, resolver_healthy,
_download_snapshot copies) all have callers and are intentionally kept.
"""

import pytest


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("bibr.utils.hf_cache", "snapshot_download_no_symlink"),
        ("bibr.utils.ml_runtime", "onnxruntime_available"),
        ("bibr.utils.ml_runtime", "ONNX_TOKENIZER"),
        ("bibr.local.vllm_mlx_runtime", "vllm_mlx_available"),
        ("bibr.ner.tags", "TAG_TO_IDX"),
        ("bibr.batch.runner", "_print"),
    ],
)
def test_dead_helper_is_gone(module, name):
    import importlib

    mod = importlib.import_module(module)
    assert not hasattr(mod, name), f"{module}.{name} is back"


def test_neighbouring_helpers_still_import():
    from bibr.batch.runner import _print_plan  # noqa: F401
    from bibr.local.vllm_mlx_runtime import vllm_mlx_unavailable_reason  # noqa: F401
    from bibr.ner.tags import BIO_TAGS, SEG_TAGS  # noqa: F401
    from bibr.utils.hf_cache import hf_download_or_cached  # noqa: F401
    from bibr.utils.ml_runtime import ONNX_MODEL, torch_available  # noqa: F401
