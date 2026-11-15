"""Load probes of the ONNX models: runs on the shapes of input a paper sends.

A model override (:mod:`bibr.utils.model_overrides`) whose graph opens but does
not run on a paper's input (too few positions for the window the runtime
sends, an input fixed to one batch size or one length, a vocabulary smaller
than its tokenizer's, another label count than its manifest lists) would
otherwise be found only at inference, after papers were written with a
fallback. The ONNX loaders run such a model at load on inputs built the way
the runtime builds them, in at least two shapes: the full length the runtime
can send and a short one, and, where the runtime batches, two batch sizes
padded to their longest row. Each output shape is checked against the
manifest; a failed probe is a load failure. A shipped default is not probed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bibr.exceptions import ConfigurationError

#: Reference strings in the shapes papers print, for the reference parser's probe.
PROBE_REFERENCES: tuple[str, ...] = (
    "Smith, J. A., & Müller, K. (2019). A title of the first work. Journal of Examples, "
    "12(3), 45–67. https://doi.org/10.1000/xyz123",
    "Lee S, Nguyễn T, García-López M. Second title: with a subtitle. Proc Natl Acad Sci "
    "USA. 2020;117:1–9.",
    "Ōtsuka, H. 2021. 第三の題名. Tokyo: Example Press, pp. 10-20.",
)

#: Running text in the shape of a paper's body, repeated where a probe needs a full window.
PROBE_BODY = (
    "Participants were recruited from two university campuses (N = 128; 64 women, mean "
    "age 21.4 years, SD = 3.2) and gave written informed consent. The study was approved "
    "by the local ethics committee (protocol 2019-12). Each session lasted about 45 "
    "minutes; stimuli were presented on a 24-inch monitor at 60 Hz, and responses were "
    "recorded with a standard keyboard. We report how we determined our sample size, all "
    "data exclusions, all manipulations, and all measures in the study [12, 13]. "
)

#: The most tokens of the short input each probe sends besides the full-length one.
SHORT_PROBE_TOKENS = 24


def short_length(full: int) -> int:
    """The token length of a probe's short input when its full one is ``full`` tokens."""
    return max(1, min(SHORT_PROBE_TOKENS, full // 2))


def cut_short(ids: np.ndarray, mask: np.ndarray, full: int) -> tuple[np.ndarray, np.ndarray]:
    """A tokenised batch cut to :func:`short_length` positions, the shape of a shorter text.

    Cutting the arrays leaves the tokenizer's truncation as the bundle set it.
    """
    n = min(np.asarray(ids).shape[1], short_length(full))
    return np.asarray(ids)[:, :n], np.asarray(mask)[:, :n]


def last_token_id(tokenizer: Any) -> int:
    """The tokenizer's largest regular-vocabulary id, which a paper's text can reach."""
    return int(tokenizer.get_vocab_size(with_added_tokens=False)) - 1


def with_last_token(ids: np.ndarray, mask: np.ndarray, tokenizer: Any) -> np.ndarray:
    """``ids`` with the tokenizer's last id at the last attended position of the longest row.

    A graph whose embedding table is smaller than the tokenizer's vocabulary
    fails on it, as it would on a paper whose text reaches that id.
    """
    ids = np.array(ids, dtype=np.int64, copy=True)
    last = last_token_id(tokenizer)
    if last < 0 or ids.size == 0:
        return ids
    lengths = np.asarray(mask).sum(axis=1)
    row = int(lengths.argmax())
    if lengths[row] > 0:
        ids[row, int(lengths[row]) - 1] = last
    return ids


def run_probe(path: str | Path, shape: str, run: Callable[[], Any]) -> Any:
    """``run()``, or a :class:`ConfigurationError` saying the graph does not run on ``shape``."""
    try:
        return run()
    except Exception as exc:
        raise ConfigurationError(
            f"{path}: the graph does not run on {shape} ({type(exc).__name__}: {exc})"
        ) from exc


def check_output(
    path: str | Path,
    name: str,
    value: Any,
    want: Sequence[int | None],
    shape: str,
    why: str,
) -> np.ndarray:
    """``value`` as an array of shape ``want`` (``None`` is any size) with finite values.

    ``why`` says where the expected sizes come from (the manifest's labels).
    """
    out = np.asarray(value)
    fits = out.ndim == len(want) and all(
        size is None or have == size for have, size in zip(out.shape, want, strict=True)
    )
    if not fits:
        wanted = "(" + ", ".join("*" if size is None else str(size) for size in want) + ")"
        raise ConfigurationError(
            f"{path}: the graph returns {name} of shape {out.shape} for {shape}; want "
            f"{wanted}, {why}"
        )
    if not np.isfinite(out).all():
        raise ConfigurationError(f"{path}: the graph returns non-finite {name} for {shape}")
    return out


def check_crf(
    path: str | Path, tags: Sequence[str], start: Any, end: Any, transitions: Any
) -> None:
    """The manifest's CRF: finite start and end scores per tag, a tags x tags transition matrix."""
    n = len(tags)
    for name, values, want in (
        ("start_transitions", start, (n,)),
        ("end_transitions", end, (n,)),
        ("transitions", transitions, (n, n)),
    ):
        array = np.asarray(values)
        if array.shape != want or not np.isfinite(array).all():
            raise ConfigurationError(
                f"{path}: the manifest's crf.{name} has shape {array.shape}; want {want} "
                f"finite values for its {n} tags"
            )


__all__ = [
    "PROBE_BODY",
    "PROBE_REFERENCES",
    "SHORT_PROBE_TOKENS",
    "check_crf",
    "check_output",
    "cut_short",
    "last_token_id",
    "run_probe",
    "short_length",
    "with_last_token",
]
