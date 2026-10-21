"""ONNX Runtime front-matter tagger: encoder emissions + numpy CRF.

The bundle is an ``onnx/`` directory like the reference parser's: the manifest
(``bibr_onnx.json``) carries the tag set, the feature keys the model was
trained on and the CRF parameters; ``model.onnx`` maps ``input_ids``,
``attention_mask`` and ``token_features`` to ``emissions``.

The input string is tokenized without special tokens. Each subtoken takes the
feature row of the first word its offsets overlap (a zero row between
regions), and a word's tag and confidence are those of the first subtoken that
starts in it. A word no subtoken starts in (``","`` folded into ``"Smith,"``,
most kanji under a SentencePiece vocabulary) takes the ``I-`` form of the tag
of the subtoken that covers it, the label training gives it. Inputs longer
than the bundle's ``max_length`` run in overlapping windows; every token keeps
the emissions of the window it sits deepest in, and one CRF pass decodes the
stitched sequence.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bibr.front_tagger.contract import CONTRACT_VERSION, FEATURE_KEYS, TaggerInput
from bibr.ner.crf_numpy import marginals, viterbi_decode
from bibr.utils.ml_runtime import ONNX_MODEL, read_onnx_manifest

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)


def _words_of_tokens(offsets: list[tuple[int, int]], inp: TaggerInput) -> list[list[int]]:
    """Indexes of the words each token overlaps, in order; empty for whitespace
    and region breaks."""
    out: list[list[int]] = []
    w = 0
    words = inp.words
    for start, end in offsets:
        while w < len(words) and words[w].end <= start:
            w += 1
        hit: list[int] = []
        k = w
        while k < len(words) and words[k].start < end:
            if words[k].end > start:
                hit.append(k)
            k += 1
        out.append(hit)
    return out


def _inside(tag: str) -> str:
    return tag if tag == "O" else f"I-{tag[2:]}"


def _word_tags(
    words_of: list[list[int]], token_tags: list[str], token_conf: list[float], n_words: int
) -> tuple[list[str], list[float]]:
    """Word tags from token tags: a word's own first subtoken decides; a word
    only covered by a token that starts in an earlier word continues it."""
    tags = ["O"] * n_words
    conf = [1.0] * n_words
    own: set[int] = set()
    covered: set[int] = set()
    for hit, tag, p in zip(words_of, token_tags, token_conf, strict=True):
        if not hit:
            continue
        first = hit[0]
        if first not in own:
            own.add(first)
            tags[first], conf[first] = tag, p
        for w in hit[1:]:
            if w not in own and w not in covered:
                covered.add(w)
                tags[w], conf[w] = _inside(tag), p
    return tags, conf


def _windows(length: int, size: int) -> list[tuple[int, int]]:
    if length <= size:
        return [(0, length)]
    stride = max(1, size * 3 // 4)
    spans = []
    start = 0
    while True:
        end = min(start + size, length)
        spans.append((start, end))
        if end == length:
            return spans
        start += stride


class OnnxFrontTagger:
    runtime = "onnx"

    def __init__(self, bundle_dir: str | Path, device: str | None = None) -> None:
        from bibr.exceptions import ConfigurationError
        from bibr.utils.onnx_providers import create_session
        from bibr.utils.onnx_tokenizer import load_tokenizer

        self.bundle_dir = Path(bundle_dir)
        self.manifest = read_onnx_manifest(self.bundle_dir)
        keys = tuple(self.manifest.get("feature_keys") or ())
        if keys != FEATURE_KEYS or self.manifest.get("contract_version") != CONTRACT_VERSION:
            raise ConfigurationError(
                f"Front-matter tagger bundle {self.bundle_dir} was trained on another input "
                f"contract (version {self.manifest.get('contract_version')!r}); this bibr "
                f"builds version {CONTRACT_VERSION}"
            )
        self.tags: list[str] = [str(t) for t in self.manifest["bio_tags"]]
        crf = self.manifest["crf"]
        self.start = np.asarray(crf["start_transitions"], dtype=np.float32)
        self.end = np.asarray(crf["end_transitions"], dtype=np.float32)
        self.transitions = np.asarray(crf["transitions"], dtype=np.float32)
        if self.transitions.shape != (len(self.tags), len(self.tags)):
            raise ConfigurationError(
                f"Front-matter tagger bundle {self.bundle_dir}: transitions shape "
                f"{self.transitions.shape} does not match {len(self.tags)} tags"
            )
        self.max_length = int(self.manifest.get("max_length", 4096))
        self.tokenizer = load_tokenizer(self.bundle_dir, self.manifest)
        # Whole-input encoding: the windows below do the cutting.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.session, self.device = create_session(
            self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL),
            device=device,
            model_name="front-matter tagger (ONNX)",
        )
        self.version = str(self.manifest.get("model_version") or self.bundle_dir)

    def _emissions(self, ids: np.ndarray, feats: np.ndarray) -> np.ndarray:
        feeds = {
            "input_ids": ids[None, :].astype(np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "token_features": feats[None, :, :].astype(np.float32),
        }
        (emissions,) = self.session.run(["emissions"], feeds)
        return np.asarray(emissions, dtype=np.float32)[0]

    def tag(self, inp: TaggerInput) -> tuple[list[str], list[float]]:
        """One BIO tag and its posterior per word of ``inp``."""
        if not inp.words:
            return [], []
        encoding = self.tokenizer.encode(inp.text, add_special_tokens=False)
        ids = np.asarray(encoding.ids, dtype=np.int64)
        words_of = _words_of_tokens(list(encoding.offsets), inp)
        feats = np.zeros((len(ids), len(FEATURE_KEYS)), dtype=np.float32)
        rows = np.asarray(inp.features, dtype=np.float32)
        for t, hit in enumerate(words_of):
            if hit:
                feats[t] = rows[hit[0]]

        emissions = np.zeros((len(ids), len(self.tags)), dtype=np.float32)
        depth = np.full(len(ids), -1, dtype=np.int64)
        for lo, hi in _windows(len(ids), self.max_length):
            window = self._emissions(ids[lo:hi], feats[lo:hi])
            for offset in range(hi - lo):
                t = lo + offset
                here = min(offset, hi - lo - 1 - offset)
                if here > depth[t]:
                    depth[t] = here
                    emissions[t] = window[offset]

        crf = {
            "start_transitions": self.start,
            "end_transitions": self.end,
            "transitions": self.transitions,
        }
        path = viterbi_decode(emissions[None], np.ones((1, len(ids)), dtype=bool), **crf)[0]
        posterior = marginals(emissions, **crf)

        return _word_tags(
            words_of,
            [self.tags[k] for k in path],
            [float(posterior[t, k]) for t, k in enumerate(path)],
            len(inp.words),
        )


_CACHE: dict[tuple[str, str | None], OnnxFrontTagger | None] = {}
_CACHE_LOCK = threading.Lock()


def load_front_tagger(settings: GlobalSettings) -> OnnxFrontTagger | None:
    """The configured tagger, or ``None`` when it is off or cannot be loaded.

    Like the region-role classifier, the tagger is optional: a missing bundle
    logs once and front matter goes to the LLM as before.
    """
    from bibr.utils.ml_runtime import find_onnx_bundle

    ml = settings.ml
    model_id = getattr(ml, "front_tagger_model_id", None)
    if getattr(ml, "front_tagger_mode", "off") == "off" or not model_id:
        return None
    revision = getattr(ml, "front_tagger_revision", None) or None
    key = (str(model_id), revision)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    tagger: OnnxFrontTagger | None = None
    try:
        bundle = find_onnx_bundle(str(model_id), revision, label="front-matter tagger")
        if bundle is None:
            logger.warning(
                "Front-matter tagger %s has no ONNX bundle; front matter goes to the LLM",
                model_id,
            )
        else:
            tagger = OnnxFrontTagger(bundle)
            logger.info("Front-matter tagger loaded from %s", bundle)
    except Exception as exc:  # noqa: BLE001 - optional model, never fatal
        logger.warning(
            "Front-matter tagger %s unavailable (%s); front matter goes to the LLM",
            model_id,
            exc,
        )
        from bibr.utils.transient import is_transient_network_error

        if is_transient_network_error(exc):
            return None
    with _CACHE_LOCK:
        _CACHE[key] = tagger
    return tagger


def reset_front_tagger_cache() -> None:
    """Test hook: forget cached taggers (and cached failures)."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = ["OnnxFrontTagger", "load_front_tagger", "reset_front_tagger_cache"]
