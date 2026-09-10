"""Training-data capture for the reference pipeline.

Best-effort JSON snapshots of (input → output) pairs used to train the local
segmenter/parser checkpoints. Both writers are gated on
``GlobalSettings.REF_TRAINING_DATA_DIR`` and never raise — a capture failure must
not affect extraction.
"""

import logging

from bibr.config import GlobalSettings, snapshot_settings

logger = logging.getLogger(__name__)


def save_ref_training_data(
    raw_text: str,
    llm_refs: list,
    *,
    settings: GlobalSettings | None = None,
) -> None:
    """Save raw bibliography text and LLM-extracted references as training data.

    Writes a JSON file per paper containing the input text and structured output,
    keyed by file_hash. Only runs when REF_TRAINING_DATA_DIR is set.
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

    Writes a JSON file per bibliography block containing the raw block text
    and the LLM's per-reference segmentation, keyed by content hash, into a
    ``segmentation/`` subdir of REF_TRAINING_DATA_DIR (kept separate from the
    parser training data, which has a different schema). Only runs when that
    setting is set. This is the ``block → boundaries`` signal the CRF
    segmenter learns from; captured only on LLM-seg success, never on
    CRF fallback (which would train the model on its own output).
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
        }
        filepath.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        logger.debug("Saved seg training data: %s (%d segments)", filepath.name, len(ref_strings))
    except Exception as e:
        logger.warning("Failed to save seg training data: %s", e)
