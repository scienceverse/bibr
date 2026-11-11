"""Text reference segmenter: an ONNX token tagger over the reference-section text.

An opt-in replacement for the CRF segmenter (``NER_SEG_CKPT``), the last
resort of every reference segmentation strategy. Off by default
(``REF_TEXT_SEG_MODEL_ID`` unset). When it is set, it takes the CRF tier's
place: it reads the same text (the reference section as bibr joins it), its
references go through the same receipt and recovery, and the CRF is not
loaded at all. It needs only onnxruntime and ``tokenizers``.

The model tags every token ``O`` (before the first reference), ``B-REF``
(opens a reference) or ``I-REF`` (inside one), and inference follows the
procedure the model was trained and selected with:

1. the text is tokenised with the bundle's ``tokenizer.json``, without special
   tokens, keeping each token's character offsets;
2. the token ids are cut into windows ``[start, min(start + window, n))`` with
   ``start = 0, stride, 2 * stride, ...`` until a window reaches the end; each
   window is one call of the graph (batch 1, no padding, attention all ones);
3. each window's tags are the per-token argmax of its logits; token ``i`` of a
   window of ``L`` tokens has trust ``min(i, L - 1 - i)``, and every token
   keeps the tag of the window where its trust is highest (the first such
   window on a tie);
4. a reference starts at the first non-space character of every ``B-REF``
   token, and at the first non-space character of the text; each runs to the
   next start, and is stripped (empty ones are dropped). ``O`` and ``I-REF``
   never start or end one.

The tier always answers for a text that is not blank: it does not decline to
the next tier as the CRF does when it finds no reference. A text in which the
model tags no token ``B-REF`` comes back as one reference, the whole list in
one block, and its :class:`TextSegmentation` says so (``no_reference_start``):
the reference extractor lists it in the attempt's ``reason_flags`` and gives
the paper a ``REF_SEG_NOT_SPLIT`` warning.

A bundle is a directory holding ``segmenter.json`` (the tags, window, stride
and tokenisation the model was trained with), ``model.onnx`` and
``tokenizer.json``, either in the directory itself or in an ``onnx/``
subdirectory, or a Hub repo with the three files in ``onnx/``, fetched at
``REF_TEXT_SEG_REVISION``. The graph takes ``input_ids`` and
``attention_mask`` (int64, batch x tokens) and returns ``logits`` (float32,
batch x tokens x 3, in the order of the manifest's ``tags``); loading runs it
on a full window of tokenised reference text (with the tokenizer's last id in
it) and on a short one to check that.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bibr.exceptions import ConfigurationError
from bibr.utils.text import strip_lone_surrogates

logger = logging.getLogger(__name__)

SEGMENTER_MANIFEST = "segmenter.json"
SEGMENTER_ONNX = "model.onnx"
SEGMENTER_TOKENIZER = "tokenizer.json"
TAGS: tuple[str, ...] = ("O", "B-REF", "I-REF")
B_REF = TAGS.index("B-REF")
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
OUTPUT_NAME = "logits"
_LABEL = "text reference segmenter"
# What the load probe tokenises: a short reference list in the shapes papers print.
_PROBE_TEXT = (
    "[1] Smith, J. A., & Müller, K. (2019). A title of the first work. Journal of "
    "Examples, 12(3), 45–67. https://doi.org/10.1000/xyz123\n"
    "[2] Lee S, Nguyễn T, García-López M. Second title: with a subtitle. Proc Natl "
    "Acad Sci USA. 2020;117:1–9.\n"
    "3. Ōtsuka, H. 2021. 第三の題名. Tokyo: Example Press, pp. 10-20.\n"
)


def segmenter_bundle_files(
    path: str | Path, revision: str | None = None
) -> tuple[Path, Path, Path]:
    """``(model.onnx, segmenter.json, tokenizer.json)`` of the bundle ``path`` names.

    ``path`` is a local directory (the files in it, or in its ``onnx/``
    subdirectory), or a Hub repo id (``org/name``) fetched at ``revision``:
    ``onnx/segmenter.json``, ``onnx/tokenizer.json``, then ``onnx/model.onnx``
    (the cache first at a pinned commit). A Hub-shaped id is never read from the
    working directory; write a local copy as a path (``./org/name``). Raises
    :class:`ConfigurationError` naming what is missing; a Hub error (a missing
    repo, revision or file, or the network) is raised as the Hub client raised it.
    """
    from bibr.utils.ml_runtime import is_hub_repo_id

    text = str(path)
    if is_hub_repo_id(text):
        from bibr.utils.hf_cache import hf_download_or_cached

        manifest = Path(hf_download_or_cached(text, f"onnx/{SEGMENTER_MANIFEST}", revision))
        tokenizer = Path(hf_download_or_cached(text, f"onnx/{SEGMENTER_TOKENIZER}", revision))
        graph = Path(hf_download_or_cached(text, f"onnx/{SEGMENTER_ONNX}", revision))
        return graph, manifest, tokenizer
    root = Path(text).expanduser()
    if not root.is_dir():
        raise ConfigurationError(
            f"{_LABEL} {text!r} is not a directory; it must hold {SEGMENTER_MANIFEST}, "
            f"{SEGMENTER_ONNX} and {SEGMENTER_TOKENIZER} (in it or in onnx/)"
        )
    names = (SEGMENTER_ONNX, SEGMENTER_MANIFEST, SEGMENTER_TOKENIZER)
    bundle = next(
        (d for d in (root, root / "onnx") if (d / SEGMENTER_MANIFEST).is_file()),
        root,
    )
    missing = [name for name in names if not (bundle / name).is_file()]
    if missing:
        raise ConfigurationError(
            f"{_LABEL} {text!r} lacks {', '.join(missing)} (the three files may sit in an "
            "onnx/ subdirectory)"
        )
    return bundle / SEGMENTER_ONNX, bundle / SEGMENTER_MANIFEST, bundle / SEGMENTER_TOKENIZER


@dataclass(frozen=True)
class SegmenterManifest:
    window: int
    stride: int


def read_segmenter_manifest(path: Path) -> SegmenterManifest:
    """Read and check ``segmenter.json``: the tags, window, stride and tokenisation."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"{_LABEL} manifest {path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{_LABEL} manifest {path} must hold a JSON object")
    problems = []
    tags = data.get("tags")
    if not isinstance(tags, list) or tuple(tags) != TAGS:
        problems.append(f"tags {tags!r} (this bibr decodes {list(TAGS)})")
    window, stride = data.get("window"), data.get("stride")
    ints = all(isinstance(v, int) and not isinstance(v, bool) for v in (window, stride))
    if not ints or not (0 < stride <= window):  # type: ignore[operator]
        problems.append(
            f"window {window!r} and stride {stride!r} (want integers with 0 < stride <= window)"
        )
    if data.get("add_special_tokens") is not False:
        problems.append(
            f"add_special_tokens {data.get('add_special_tokens')!r} (this bibr tokenises "
            "without special tokens)"
        )
    if problems:
        raise ConfigurationError(f"{path}: unusable {_LABEL} manifest: {'; '.join(problems)}")
    return SegmenterManifest(window=int(window), stride=int(stride))  # type: ignore[arg-type]


def windows(n: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Token windows ``[start, end)`` over ``n`` tokens, the last one reaching ``n``."""
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + window, n)
        out.append((start, end))
        if end >= n:
            break
        start += stride
    return out


def merge_windows(
    n: int, window_logits: list[tuple[int, int, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token tags and the softmax of the kept window's logits, by trust merge.

    ``window_logits`` holds ``(start, end, logits[end - start, tags])``. Token
    ``i`` of a window of ``L`` tokens has trust ``min(i, L - 1 - i)``; each
    token keeps the window where its trust is highest, the first on a tie.
    """
    tags = np.zeros(n, dtype=np.int64)
    probs = np.zeros((n, len(TAGS)), dtype=np.float64)
    trust = np.full(n, -1, dtype=np.int64)
    for start, end, logits in window_logits:
        length = end - start
        pred = np.asarray(logits).argmax(-1)
        z = np.asarray(logits, dtype=np.float64)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        idx = np.arange(length)
        window_trust = np.minimum(idx, length - 1 - idx)
        better = window_trust > trust[start:end]
        trust[start:end] = np.where(better, window_trust, trust[start:end])
        tags[start:end] = np.where(better, pred, tags[start:end])
        probs[start:end][better] = p[better]
    return tags, probs


def starts_from_tags(text: str, offsets: list[tuple[int, int]], tags: np.ndarray) -> list[int]:
    """Reference starts: the first non-space character of each ``B-REF`` token, increasing,
    with the first non-space character of the text always a start."""
    starts: list[int] = []
    for (a, b), tag in zip(offsets, tags.tolist(), strict=True):
        if tag != B_REF or a == b:
            continue
        s = a
        while s < b and text[s].isspace():
            s += 1
        if not starts or s > starts[-1]:
            starts.append(s)
    first = len(text) - len(text.lstrip())
    if first < len(text) and (not starts or starts[0] > first):
        starts = [first, *starts]
    return starts


def spans_from_starts(text: str, starts: list[int]) -> list[tuple[int, int]]:
    """Reference spans ``[start_i, start_{i+1})``, trailing whitespace trimmed."""
    out = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(text)
        while e > s and text[e - 1].isspace():
            e -= 1
        out.append((s, e))
    return out


@dataclass(frozen=True)
class TextSegmentation:
    """One text's segmentation and what the model said about it."""

    text: str
    segments: list[str]
    tags: np.ndarray
    starts: list[int]
    tokens: int
    windows: int
    #: P(B-REF) of each token tagged B-REF (the forced first start has none).
    start_probs: np.ndarray
    #: ``1 - 2 * sum(min(p, 1 - p)) / starts`` over every token's P(B-REF), uncalibrated.
    expected_entry_precision: float
    #: The model tagged no token ``B-REF``: the one segment is the whole text, not split.
    no_reference_start: bool = False

    def receipt_flags(self) -> tuple[str, ...]:
        """The model's figures for the receipt's ``reason_flags``."""
        flags = [f"tokens={self.tokens}", f"windows={self.windows}", f"starts={len(self.starts)}"]
        if self.tokens:
            flags.append(f"expected_entry_precision={self.expected_entry_precision:.4f}")
        if len(self.start_probs):
            flags.append(f"start_prob_min={float(self.start_probs.min()):.4f}")
        if self.no_reference_start:
            flags.append("no_reference_start")
        return tuple(flags)


class OnnxTextSegmenter:
    """Loads a text segmenter bundle and splits reference text; safe to share across threads."""

    def __init__(self, path: str | Path, revision: str | None = None) -> None:
        from tokenizers import Tokenizer

        from bibr.utils.onnx_providers import create_session

        self.onnx_path, manifest_path, tokenizer_path = segmenter_bundle_files(path, revision)
        manifest = read_segmenter_manifest(manifest_path)
        self.window, self.stride = manifest.window, manifest.stride
        try:
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise ConfigurationError(
                f"{_LABEL} tokenizer {tokenizer_path} is unreadable ({type(exc).__name__}: {exc})"
            ) from exc
        # Exactly what the model was trained on: no truncation, no padding.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.files: tuple[Path, ...] = (self.onnx_path, manifest_path, tokenizer_path)
        self._session, _device = create_session(self.onnx_path, device="cpu", model_name=_LABEL)
        inputs = [i.name for i in self._session.get_inputs()]
        outputs = [o.name for o in self._session.get_outputs()]
        if set(inputs) != {INPUT_IDS, ATTENTION_MASK} or OUTPUT_NAME not in outputs:
            raise ConfigurationError(
                f"{self.onnx_path}: graph inputs {inputs} / outputs {outputs}; want "
                f"{INPUT_IDS!r} and {ATTENTION_MASK!r} in and {OUTPUT_NAME!r} out"
            )
        self._lock = threading.Lock()
        self._check_graph()

    def _run(self, ids: np.ndarray) -> np.ndarray:
        x = np.asarray(ids, dtype=np.int64)[None]
        with self._lock:
            out = self._session.run([OUTPUT_NAME], {INPUT_IDS: x, ATTENTION_MASK: np.ones_like(x)})
        return np.asarray(out[0])

    def _probe_ids(self) -> list[np.ndarray]:
        """Token ids in the two window lengths a paper sends, one with the tokenizer's last id.

        The ids of a short reference list through the bundle's tokenizer, repeated to
        the manifest's ``window`` (a long list's windows), and cut to a few tokens (a
        short list, or a long one's last window): a graph with fewer positions than
        the window, one fixed to a single length, or a vocabulary smaller than the
        tokenizer's, fails on one of them as it would on a paper.
        """
        from bibr.utils.onnx_probe import short_length

        ids, _offsets = self.encode(_PROBE_TEXT)
        if not len(ids):
            raise ConfigurationError(
                f"{self.onnx_path}: the tokenizer gives no tokens for a reference list"
            )
        full = np.resize(ids, self.window)
        last = int(self._tokenizer.get_vocab_size(with_added_tokens=False)) - 1
        if last >= 0:
            full[-1] = last
        short = ids[: min(short_length(self.window), len(ids))]
        return [full, short] if len(short) else [full]

    def _check_graph(self) -> None:
        """Run the graph on each probe window; it must give one finite logit per tag per token."""
        for probe in self._probe_ids():
            shape = f"(1, {len(probe)}) window of tokenised reference text"
            try:
                out = self._run(probe)
            except Exception as exc:
                raise ConfigurationError(
                    f"{self.onnx_path}: the graph does not run on a {shape} "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            if out.shape != (1, len(probe), len(TAGS)):
                raise ConfigurationError(
                    f"{self.onnx_path}: the graph returns {OUTPUT_NAME} of shape {out.shape} "
                    f"for a {shape}; want (1, {len(probe)}, {len(TAGS)}), one logit per tag "
                    f"{list(TAGS)}"
                )
            if not np.isfinite(out).all():
                raise ConfigurationError(
                    f"{self.onnx_path}: the graph returns non-finite {OUTPUT_NAME} for a {shape}"
                )

    def encode(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Token ids and character offsets, without special tokens."""
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        return np.asarray(enc.ids, dtype=np.int64), [tuple(o) for o in enc.offsets]

    def tag(self, text: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], int]:
        """``(tags, kept softmax, offsets, windows)`` for ``text`` (already surrogate-free)."""
        ids, offsets = self.encode(text)
        spans = windows(len(ids), self.window, self.stride)
        logits = [(a, b, self._run(ids[a:b])[0]) for a, b in spans]
        tags, probs = merge_windows(len(ids), logits)
        return tags, probs, offsets, len(spans)

    def segment_text(self, text: str) -> TextSegmentation:
        """Split ``text`` into references; empty for an empty or blank text.

        A text the model opens no reference in is one segment, flagged
        ``no_reference_start``.
        """
        text = strip_lone_surrogates(text or "")
        if not text.strip():
            return TextSegmentation(text, [], np.zeros(0, np.int64), [], 0, 0, np.zeros(0), 0.0)
        tags, probs, offsets, n_windows = self.tag(text)
        if not len(tags):
            return TextSegmentation(text, [], tags, [], 0, n_windows, np.zeros(0), 0.0)
        starts = starts_from_tags(text, offsets, tags)
        segments = [text[s:e].strip() for s, e in spans_from_starts(text, starts)]
        segments = [s for s in segments if s]
        p_start = probs[:, B_REF]
        opened = p_start[tags == B_REF]
        doubt = float(np.minimum(p_start, 1.0 - p_start).sum())
        score = max(0.0, 1.0 - 2.0 * doubt / max(1, len(starts)))
        return TextSegmentation(
            text,
            segments,
            tags,
            starts,
            len(tags),
            n_windows,
            opened,
            score,
            no_reference_start=not len(opened),
        )

    def segment(self, text: str) -> list[str]:
        """Reference strings of ``text``, as the CRF segmenter's ``segment`` returns them."""
        return self.segment_text(text).segments


__all__ = [
    "SEGMENTER_MANIFEST",
    "SEGMENTER_ONNX",
    "SEGMENTER_TOKENIZER",
    "TAGS",
    "OnnxTextSegmenter",
    "TextSegmentation",
    "merge_windows",
    "read_segmenter_manifest",
    "segmenter_bundle_files",
    "spans_from_starts",
    "starts_from_tags",
    "windows",
]
