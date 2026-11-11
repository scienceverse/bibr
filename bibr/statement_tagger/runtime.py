"""Statement tagger runtime: ordered text blocks in, typed statements out.

A small encoder with one sigmoid output per statement type reads each block of
a paper with the block before it, decides which types the block belongs to,
and adjacent blocks of one type are joined into one statement. This module is
self-contained on purpose: it needs numpy, onnxruntime and tokenizers, nothing
else from bibr, so any caller with a paper's blocks in reading order (bibr's
post-parse rows, another tool's paragraphs) gets the same answer from the same
bundle::

    from bibr.statement_tagger.runtime import StatementTagger

    tagger = StatementTagger("path/to/bundle")   # the directory, or its onnx/
    for st in tagger.tag(["Funding", "This work was supported by ...", ...]):
        print(st.type, st.start, st.end, st.score)

**Blocks.** A block is one paragraph-like unit of text in reading order: a
heading line is a block of its own, a paragraph is a block. The model was
trained on layout regions and text-layer line groups of that size; much
longer units (a whole section) or much shorter ones (single lines) make it
less accurate.

**Bundle.** A directory ``onnx/`` (or that directory itself) holding
``bibr_onnx.json``, ``model.onnx`` and ``tokenizer.json``. The manifest
(:data:`MANIFEST`) names the model (``"model": "statement_tagger"``), its
version, the types in output order with one threshold each, the token budget
and the input rule. The graph takes ``input_ids`` and ``attention_mask``
(int64, batch x tokens) and returns ``probs`` (batch x types, sigmoid applied).
Loading runs it once on two dummy blocks and checks the output shape.

**Input rule** (``manifest["input"]``): each block is encoded as the tokenizer
pair (context, text), where the context is the first ``context_chars``
characters of the previous block's text (empty for the first block), truncated
longest-first to ``max_length`` tokens. Both texts are normalised as in
training: NFKC, soft hyphens and zero-width characters removed, whitespace
collapsed (:func:`normalize_block_text`).

**Output rule.** A block has type *t* when its probability reaches the
threshold of *t* (a block may have several types: a funding sentence inside
acknowledgments). Each run of consecutive blocks of one type is one
:class:`TaggedStatement`; its score is the highest block probability of the
type in the run.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

MODEL = "statement_tagger"
MANIFEST = "bibr_onnx.json"
MANIFEST_SCHEMA_VERSION = 1
INPUT_RULE = "pair:previous_block"
INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAME = "probs"
DEFAULT_BATCH = 32

_WS = re.compile(r"\s+")
_INVISIBLE = str.maketrans("", "", "­​﻿")


class StatementTaggerError(ValueError):
    """The bundle is missing, unreadable, or not a statement tagger this code can run."""


@dataclass(frozen=True)
class TaggedStatement:
    """One statement: a run of consecutive blocks of one type.

    ``start`` and ``end`` are block positions in the input list, ``end``
    inclusive; ``score`` is the highest probability of ``type`` in the run.
    """

    type: str
    start: int
    end: int
    score: float


def normalize_block_text(text: str | None) -> str:
    """A block's text as the model saw it in training (NFKC, invisible marks out, one space)."""
    text = unicodedata.normalize("NFKC", text or "").translate(_INVISIBLE)
    return _WS.sub(" ", text).strip()


def bundle_dir(path: str | Path) -> Path:
    """The directory holding the manifest: ``path/onnx`` if it has one, else ``path``."""
    path = Path(path).expanduser()
    for candidate in (path / "onnx", path):
        if (candidate / MANIFEST).is_file():
            return candidate
    raise StatementTaggerError(
        f"statement tagger bundle {str(path)!r} not found: want {MANIFEST}, model.onnx and "
        "tokenizer.json in the directory or in its onnx/ subdirectory"
    )


def _member(directory: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name:
        raise StatementTaggerError(f"{directory}: manifest file name {name!r} is not a file name")
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise StatementTaggerError(f"{directory}: manifest file name {name!r} leaves the bundle")
    return directory / Path(*pure.parts)


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """Read and check a statement tagger manifest; raises :class:`StatementTaggerError`."""
    directory = Path(directory)
    path = directory / MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StatementTaggerError(
            f"statement tagger manifest {path} is unreadable: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise StatementTaggerError(f"{path} must hold a JSON object")
    problems = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        problems.append(
            f"schema_version {manifest.get('schema_version')!r} (this code reads "
            f"{MANIFEST_SCHEMA_VERSION})"
        )
    if manifest.get("model") != MODEL:
        problems.append(f"model {manifest.get('model')!r} (want {MODEL!r})")
    labels = manifest.get("labels")
    thresholds = manifest.get("thresholds")
    if not (isinstance(labels, list) and labels and all(isinstance(x, str) for x in labels)):
        problems.append("labels must be a non-empty list of type names")
    elif len(set(labels)) != len(labels):
        problems.append("labels repeat a type")
    elif not isinstance(thresholds, dict) or set(thresholds) != set(labels):
        problems.append("thresholds must give one number per label")
    else:
        for label in labels:
            value = thresholds[label]
            if not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0.0 < float(value) <= 1.0
            ):
                problems.append(f"threshold of {label} is {value!r} (want a number in (0, 1])")
    rule = manifest.get("input") or {}
    if not isinstance(rule, dict) or rule.get("rule") != INPUT_RULE:
        problems.append(f"input rule {rule!r} (this code runs {INPUT_RULE!r})")
    else:
        for key in ("max_length", "context_chars"):
            value = rule.get(key)
            if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
                problems.append(f"input.{key} is {value!r} (want a positive integer)")
    if problems:
        raise StatementTaggerError(f"{path}: not a usable statement tagger: {'; '.join(problems)}")
    return manifest


def bundle_files(path: str | Path) -> list[Path]:
    """The manifest and the model and tokenizer files it names (they may not exist)."""
    directory = bundle_dir(path)
    manifest = read_manifest(directory)
    return [
        directory / MANIFEST,
        _member(directory, manifest.get("model_file", "model.onnx")),
        _member(directory, manifest.get("tokenizer_file", "tokenizer.json")),
    ]


def join_runs(
    block_types: Sequence[frozenset[str] | set[str]],
    probs: np.ndarray,
    labels: Sequence[str],
) -> list[TaggedStatement]:
    """Runs of consecutive blocks sharing a type, one statement per run and type.

    Ordered by first block, then by the type's position in ``labels``.
    """
    out: list[TaggedStatement] = []
    for column, label in enumerate(labels):
        start = None
        for index in range(len(block_types) + 1):
            inside = index < len(block_types) and label in block_types[index]
            if inside and start is None:
                start = index
            elif not inside and start is not None:
                score = float(np.max(probs[start:index, column])) if len(probs) else 1.0
                out.append(TaggedStatement(label, start, index - 1, score))
                start = None
    order = {label: i for i, label in enumerate(labels)}
    out.sort(key=lambda s: (s.start, order[s.type]))
    return out


class StatementTagger:
    """A loaded statement tagger bundle; safe to share across threads.

    ``session_options`` is an ``onnxruntime.SessionOptions`` to open the graph
    with (a caller's thread policy); without it ``threads`` sets the intra-op
    thread count (``None``: onnxruntime's default). CPU only.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        threads: int | None = None,
        session_options: Any = None,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self.directory = bundle_dir(path)
        manifest = read_manifest(self.directory)
        self.manifest = manifest
        self.labels: tuple[str, ...] = tuple(manifest["labels"])
        self.thresholds = np.asarray(
            [float(manifest["thresholds"][label]) for label in self.labels], dtype=np.float32
        )
        rule = manifest["input"]
        self.max_length = int(rule["max_length"])
        self.context_chars = int(rule["context_chars"])
        self.version = str(manifest.get("model_version") or "")
        self.batch_size = max(1, int(batch_size))
        model_path = _member(self.directory, manifest.get("model_file", "model.onnx"))
        tokenizer_path = _member(self.directory, manifest.get("tokenizer_file", "tokenizer.json"))
        missing = [p.name for p in (model_path, tokenizer_path) if not p.is_file()]
        if missing:
            raise StatementTaggerError(f"statement tagger bundle {self.directory} lacks {missing}")
        self.files: tuple[Path, ...] = (self.directory / MANIFEST, model_path, tokenizer_path)

        from tokenizers import Tokenizer

        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise StatementTaggerError(f"{tokenizer_path} is not a tokenizer: {exc}") from exc
        pad_id = int(manifest.get("pad_token_id", 0))
        tokenizer.enable_truncation(max_length=self.max_length, strategy="longest_first")
        tokenizer.enable_padding(pad_id=pad_id, pad_token=str(manifest.get("pad_token", "[PAD]")))
        self._tokenizer = tokenizer

        import onnxruntime as ort

        if session_options is None:
            session_options = ort.SessionOptions()
            if threads is not None:
                session_options.intra_op_num_threads = max(1, int(threads))
                session_options.inter_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(model_path), session_options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise StatementTaggerError(
                f"{model_path} does not open as an ONNX graph: {exc}"
            ) from exc
        inputs = [i.name for i in self._session.get_inputs()]
        outputs = [o.name for o in self._session.get_outputs()]
        if set(inputs) != set(INPUT_NAMES) or OUTPUT_NAME not in outputs:
            raise StatementTaggerError(
                f"{model_path}: graph inputs {inputs} / outputs {outputs}; want inputs "
                f"{list(INPUT_NAMES)} and output {OUTPUT_NAME!r}"
            )
        self._lock = threading.Lock()
        self._check_graph()

    def _check_graph(self) -> None:
        """Run the graph on two dummy blocks: it must give one probability per type."""
        try:
            probs = self.probabilities(["Funding", "This work was supported by a grant."])
        except Exception as exc:
            raise StatementTaggerError(
                f"{self.directory}: the graph does not run on two dummy blocks "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if probs.shape != (2, len(self.labels)) or not np.all(np.isfinite(probs)):
            raise StatementTaggerError(
                f"{self.directory}: the graph returns {OUTPUT_NAME} of shape {probs.shape} for "
                f"2 blocks; want (2, {len(self.labels)}) finite values"
            )

    def pairs(self, texts: Sequence[str]) -> list[tuple[str, str]]:
        """The (context, text) pair each block is encoded as."""
        clean = [normalize_block_text(t) for t in texts]
        return [
            (clean[i - 1][: self.context_chars] if i else "", text) for i, text in enumerate(clean)
        ]

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Per block, the probability of each type in :attr:`labels` order (blocks x types)."""
        pairs = self.pairs(texts)
        out = np.zeros((len(pairs), len(self.labels)), dtype=np.float32)
        if not pairs:
            return out
        # Length-sorted batches pad less; results go back in block order.
        order = np.argsort([len(c) + len(t) for c, t in pairs], kind="stable")
        for begin in range(0, len(order), self.batch_size):
            idx = order[begin : begin + self.batch_size]
            encoded = self._tokenizer.encode_batch([pairs[j] for j in idx])
            ids = np.asarray([e.ids for e in encoded], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in encoded], dtype=np.int64)
            with self._lock:
                probs = self._session.run(
                    [OUTPUT_NAME], {"input_ids": ids, "attention_mask": mask}
                )[0]
            out[idx] = np.asarray(probs, dtype=np.float32)
        return out

    def block_types(self, probs: np.ndarray) -> list[frozenset[str]]:
        """The types each block has: the ones whose probability reaches their threshold."""
        hits = np.asarray(probs) >= self.thresholds[None, :]
        return [frozenset(self.labels[j] for j in np.flatnonzero(row)) for row in hits]

    def tag(self, texts: Sequence[str]) -> list[TaggedStatement]:
        """The statements in a paper's ordered blocks."""
        probs = self.probabilities(texts)
        return join_runs(self.block_types(probs), probs, self.labels)


__all__ = [
    "MANIFEST",
    "MODEL",
    "StatementTagger",
    "StatementTaggerError",
    "TaggedStatement",
    "bundle_dir",
    "bundle_files",
    "join_runs",
    "normalize_block_text",
    "read_manifest",
]
