"""Prefetch the configured wtpsplit-lite model using runtime resolution rules."""

from __future__ import annotations

import argparse
from pathlib import Path

from bibr.segmenter_base import materialize_hub_snapshot, resolve_wtpsplit_model

_BASE_TOKENIZER_REPO = "FacebookAI/xlm-roberta-base"


def _prefetch_remote_to_cache(
    repo_id: str, cache_dir: Path, revision: str | None = None
) -> tuple[str, str]:
    """Download the files wtpsplit-lite opens, returning local model/tokenizer dirs."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    download_kwargs = {"cache_dir": str(cache_dir)}
    # The revision pins the segmenter repo only; the base tokenizer fallback
    # below is a different repo.
    model_kwargs = {**download_kwargs, **({"revision": revision} if revision else {})}
    onnx_path = Path(hf_hub_download(repo_id, "model_optimized.onnx", **model_kwargs))
    config_path = Path(hf_hub_download(repo_id, "config.json", **model_kwargs))
    if config_path.parent != onnx_path.parent:
        raise RuntimeError("Prefetched wtpsplit model files resolved to different snapshots")

    try:
        tokenizer_path = Path(hf_hub_download(repo_id, "tokenizer.json", **model_kwargs))
    except RemoteEntryNotFoundError:
        tokenizer_path = Path(
            hf_hub_download(_BASE_TOKENIZER_REPO, "tokenizer.json", **download_kwargs)
        )
    return str(onnx_path.parent), str(tokenizer_path.parent)


def prefetch_segmenter(model: str, *, cache_dir: Path | None, revision: str | None = None) -> None:
    """Instantiate a CPU ORT segmenter so its model assets are cached."""
    from wtpsplit_lite import SaT

    resolved = resolve_wtpsplit_model(model, revision=revision)
    model_name = resolved.model_name
    hub_prefix = resolved.hub_prefix
    tokenizer_name_or_path = resolved.tokenizer_name_or_path
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not resolved.is_local:
            model_name, tokenizer_name_or_path = _prefetch_remote_to_cache(
                resolved.repo_id, cache_dir, resolved.revision
            )
            hub_prefix = None
    elif not resolved.is_local and resolved.revision is not None:
        # Same path the runtime segmenter takes for a pinned Hub model, so the
        # baked cache holds exactly the snapshot it will ask for.
        model_name, pinned_tokenizer = materialize_hub_snapshot(resolved.repo_id, resolved.revision)
        hub_prefix = None
        if pinned_tokenizer is not None:
            tokenizer_name_or_path = pinned_tokenizer

    kwargs: dict[str, object] = {
        "hub_prefix": hub_prefix,
        "ort_providers": ["CPUExecutionProvider"],
    }
    if tokenizer_name_or_path is not None:
        kwargs["tokenizer_name_or_path"] = tokenizer_name_or_path
    SaT(model_name, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Short wtpsplit name, full Hugging Face ID, or local bundle")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache directory")
    parser.add_argument(
        "--revision",
        help="Hub revision to pin (default: the audited commit for sat-6l-sm, else main)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    prefetch_segmenter(args.model, cache_dir=args.cache_dir, revision=args.revision)
    print(f"Sentence segmenter ready: {args.model}")


if __name__ == "__main__":
    main()
