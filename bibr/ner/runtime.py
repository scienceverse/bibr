"""Runtime selection for the NER reference parser (ONNX first, torch fallback)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_ref_parser(
    ckpt: str | Path,
    device: str | None = None,
    revision: str | None = None,
    settings=None,
):
    """Return an ``OnnxRefParser`` or a torch ``RefParser`` for ``ckpt``.

    ``ckpt`` is what ``NER_PARSER_CKPT`` accepts: a local ``.pt`` file, a local
    directory, or ``org/repo[:filename]``. The ONNX bundle is the sibling
    ``onnx/`` directory (local) or ``onnx/`` in the same repo at ``revision``.
    """
    from bibr.utils.ml_runtime import find_onnx_bundle, hub_bundle_hint, resolve_runtime

    if settings is None:
        from bibr.config import snapshot_settings

        settings = snapshot_settings()
    ckpt_str = str(ckpt)
    runtime, bundle = resolve_runtime(
        "NER reference parser",
        settings=settings,
        bundle=lambda: find_onnx_bundle(ckpt_str, revision, label="NER reference parser"),
        bundle_hint=hub_bundle_hint("NER_PARSER_CKPT", ckpt_str, revision),
    )
    if runtime == "onnx":
        from bibr.ner.parser_onnx import OnnxRefParser

        logger.info("NER parser runtime: onnx (%s)", bundle)
        return OnnxRefParser(bundle, device=device)

    import bibr.ner.parser as parser_mod

    logger.info("NER parser runtime: torch (%s@%s)", ckpt_str, revision or "main")
    return parser_mod.RefParser(ckpt, device=device, revision=revision)
