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
    from bibr.utils.model_overrides import is_model_override

    if settings is None:
        from bibr.config import snapshot_settings

        settings = snapshot_settings()
    ckpt_str = str(ckpt)
    runtime, bundle = resolve_runtime(
        "NER reference parser",
        settings=settings,
        # An override tells a network failure from a missing bundle (bibr.utils.model_overrides).
        bundle=lambda: find_onnx_bundle(
            ckpt_str,
            revision,
            label="NER reference parser",
            raise_transient=is_model_override(settings, "NER_PARSER_CKPT"),
        ),
        bundle_hint=hub_bundle_hint("NER_PARSER_CKPT", ckpt_str, revision),
    )
    if runtime == "torch":
        from bibr.ner.checkpoint import has_torch_checkpoint

        # A parser published as an ONNX bundle alone (the 0.9.0 default) has no
        # torch form, so it runs on ONNX Runtime under ML_RUNTIME=torch too, as
        # the front-matter tagger does.
        if not has_torch_checkpoint(ckpt, revision):
            onnx_bundle = find_onnx_bundle(
                ckpt_str,
                revision,
                label="NER reference parser",
                raise_transient=is_model_override(settings, "NER_PARSER_CKPT"),
            )
            if onnx_bundle is not None:
                logger.info("NER parser %s has no torch checkpoint; it runs on ONNX", ckpt_str)
                runtime, bundle = "onnx", onnx_bundle
    if runtime == "onnx":
        from bibr.ner.parser_onnx import OnnxRefParser

        logger.info("NER parser runtime: onnx")
        logger.debug("NER parser bundle: %s", bundle)
        parser = OnnxRefParser(bundle, device=device)
        if is_model_override(settings, "NER_PARSER_CKPT"):
            parser.check_graph()  # a candidate's graph is probed at load; the default is not
        return parser

    import bibr.ner.parser as parser_mod

    logger.info("NER parser runtime: torch (%s@%s)", ckpt_str, revision or "main")
    return parser_mod.RefParser(
        ckpt,
        encoder_name=getattr(settings, "NER_PARSER_ENCODER", None),
        device=device,
        revision=revision,
        encoder_revision=getattr(settings, "NER_PARSER_ENCODER_REVISION", None),
    )
