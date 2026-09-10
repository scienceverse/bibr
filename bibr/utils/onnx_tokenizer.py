"""Batch tokenisation for the ONNX text models with the ``tokenizers`` library.

The ONNX classifiers and the NER parser ship the exact ``tokenizer.json`` the
torch path loads through ``transformers.AutoTokenizer``; ``tokenizers`` is the
Rust library underneath that wrapper, so ids, masks and offsets are identical
without importing transformers (or torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bibr.utils.text import strip_lone_surrogates


@dataclass
class TokenizedBatch:
    input_ids: np.ndarray  # (B, L) int64, right-padded
    attention_mask: np.ndarray  # (B, L) int64
    offsets: list[list[tuple[int, int]]]  # per row, unpadded token offsets


def load_tokenizer(bundle_dir: str | Path, manifest: dict[str, Any]):
    """Open the bundle's ``tokenizer.json`` with truncation and padding armed."""
    from tokenizers import Tokenizer

    path = Path(bundle_dir) / manifest.get("tokenizer_file", "tokenizer.json")
    if not path.is_file():
        raise FileNotFoundError(f"ONNX bundle {bundle_dir} has no tokenizer file {path.name}")
    tokenizer = Tokenizer.from_file(str(path))
    pad_id = manifest.get("pad_token_id")
    pad_token = manifest.get("pad_token", "[PAD]")
    if pad_id is None:
        pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError(f"ONNX bundle {bundle_dir}: cannot resolve the pad token id")
    tokenizer.enable_padding(pad_id=int(pad_id), pad_token=pad_token)
    max_length = manifest.get("max_length")
    if max_length:
        tokenizer.enable_truncation(int(max_length))
    return tokenizer


def require_added_token(tokenizer, token: str, *, label: str) -> None:
    """Fail loudly when a bundle's tokenizer does not know a template marker.

    Both classifier templates put a literal ``[SEP]`` *inside* the input string.
    A ``tokenizer.json`` that carries the special tokens only in its vocabulary,
    and not in ``added_tokens``, lets the lowercasing normalizer reach the
    marker and splits it into ``UNK`` fragments — every prediction then shifts,
    with nothing raised. ``transformers`` hides the difference because its
    wrapper re-registers the specials, so the ONNX path checks it at load.
    """
    from bibr.exceptions import ConfigurationError

    ids = tokenizer.encode(token, add_special_tokens=False).ids
    if len(ids) != 1:
        raise ConfigurationError(
            f"ONNX bundle for the {label} ships a tokenizer that splits {token!r} into "
            f"{len(ids)} tokens; it has to be registered as an added special token, or "
            f"the input template's separators are read as text."
        )


def encode_batch(
    tokenizer,
    texts: list[str],
    *,
    max_length: int | None = None,
    add_special_tokens: bool = True,
) -> TokenizedBatch:
    """Tokenize ``texts`` into right-padded int64 arrays (plus per-row offsets).

    ``max_length`` re-arms truncation for this call when it differs from the
    bundle default; lone surrogates from OCR garbage are stripped up front (the
    Rust tokenizer rejects them).
    """
    if max_length is not None:
        current = tokenizer.truncation
        if current is None or current.get("max_length") != max_length:
            tokenizer.enable_truncation(int(max_length))
    clean = [strip_lone_surrogates(t) if isinstance(t, str) else str(t) for t in texts]
    encodings = tokenizer.encode_batch(clean, add_special_tokens=add_special_tokens)
    ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
    mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
    offsets = [
        [tuple(o) for o, m in zip(e.offsets, e.attention_mask, strict=True) if m] for e in encodings
    ]
    return TokenizedBatch(input_ids=ids, attention_mask=mask, offsets=offsets)
