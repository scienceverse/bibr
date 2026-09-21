"""Training-data capture for the reference pipeline.

Best-effort JSON snapshots of (input → output) pairs used to train the local
segmenter/parser checkpoints. Only LLM output is captured, and every record
carries a ``provenance`` block naming the provider, model, and prompt that
produced it. Both writers are gated on ``GlobalSettings.REF_TRAINING_DATA_DIR``
and never raise — a capture failure must not affect extraction.
"""

import logging

from bibr.config import GlobalSettings, snapshot_settings

logger = logging.getLogger(__name__)

# Stands in for the per-call random fence boundary so a prompt hashes the same
# on every call.
_PROMPT_BOUNDARY = "<boundary>"


def _llm_provenance(source: str, prompt: str, settings: GlobalSettings) -> dict[str, str]:
    """Identify the LLM call behind a captured record.

    ``prompt_sha256`` covers the system prompt, the user template rendered
    around empty data, and the response schema, so it changes with any edit to
    what the model was asked — including edits that leave ``bibr_version``
    unchanged.
    """
    import hashlib
    import json

    from bibr import __version__
    from bibr.clients.prompts import PROMPTS, prompt_text

    spec = PROMPTS[prompt]
    template = prompt_text(spec.build_user(boundary=_PROMPT_BOUNDARY, text=""))
    fingerprint = json.dumps(
        [spec.system, template, spec.response_model.model_json_schema()],
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "source": source,
        "provider": settings.llm.provider,
        "model": settings.llm.model,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(fingerprint.encode()).hexdigest(),
        "bibr_version": __version__,
    }


def save_ref_training_data(
    raw_text: str,
    llm_refs: list,
    *,
    settings: GlobalSettings | None = None,
) -> None:
    """Save raw bibliography text and LLM-extracted references as training data.

    Writes a JSON file per paper containing the input text, the structured
    output, and its ``provenance`` (``source="llm"``, prompt
    ``references_parse``), keyed by content hash. Only runs when
    REF_TRAINING_DATA_DIR is set.
    """
    effective = settings if settings is not None else snapshot_settings()
    output_dir = effective.REF_TRAINING_DATA_DIR
    if not output_dir:
        return

    import hashlib
    import json
    from pathlib import Path

    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Use a content hash as filename to avoid duplicates
        content_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
        filepath = out_path / f"{content_hash}.json"

        if filepath.exists():
            return

        record = {
            "input": raw_text,
            "output": [ref.model_dump(mode="json") for ref in llm_refs],
            "n_references": len(llm_refs),
            "provenance": _llm_provenance("llm", "references_parse", effective),
        }
        filepath.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        logger.debug("Saved ref training data: %s (%d refs)", filepath.name, len(llm_refs))
    except Exception as e:
        logger.warning("Failed to save ref training data: %s", e)


def save_seg_training_data(
    ref_text: str,
    ref_strings: list[str],
    *,
    settings: GlobalSettings | None = None,
) -> None:
    """Save the LLM segmentation result as CRF-segmenter training data.

    Writes a JSON file per bibliography block containing the raw block text,
    the LLM's per-reference segmentation, and its ``provenance``
    (``source="llm_anchor"``, prompt ``references_segment``), keyed by content
    hash, into a ``segmentation/`` subdir of REF_TRAINING_DATA_DIR (kept
    separate from the parser training data, which has a different schema).
    Only runs when that setting is set. This is the ``block → boundaries``
    signal the CRF segmenter learns from, so it is captured only on LLM-seg
    success — never from the geom, region-anchor, CRF, or marker-split tiers,
    whose output is the pipeline's own prediction (training on it would teach
    the segmenters their own output). Under the default ``geom`` strategy only
    blocks the cascade hands to the LLM are captured; ``REF_SEG_STRATEGY=llm``
    labels every block.
    """
    effective = settings if settings is not None else snapshot_settings()
    output_dir = effective.REF_TRAINING_DATA_DIR
    if not output_dir:
        return

    import hashlib
    import json
    from pathlib import Path

    try:
        out_path = Path(output_dir) / "segmentation"
        out_path.mkdir(parents=True, exist_ok=True)

        content_hash = hashlib.sha256(ref_text.encode()).hexdigest()[:16]
        filepath = out_path / f"{content_hash}.json"
        if filepath.exists():
            return

        record = {
            "input": ref_text,
            "segments": ref_strings,
            "n_segments": len(ref_strings),
            "provenance": _llm_provenance("llm_anchor", "references_segment", effective),
        }
        filepath.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        logger.debug("Saved seg training data: %s (%d segments)", filepath.name, len(ref_strings))
    except Exception as e:
        logger.warning("Failed to save seg training data: %s", e)
