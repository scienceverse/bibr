"""Viterbi decoding for a linear-chain CRF in numpy.

Mirrors ``torchcrf.CRF._viterbi_decode`` (batch-first emissions, a boolean
mask whose first column is all True, ``start_transitions`` added to the first
emission, ``end_transitions`` to the last valid step, first-maximum tie
breaking) so the ONNX reference parser decodes exactly like the torch one.
"""

from __future__ import annotations

import numpy as np


def viterbi_decode(
    emissions: np.ndarray,
    mask: np.ndarray,
    *,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
    transitions: np.ndarray,
) -> list[list[int]]:
    """Best tag path per row.

    Args:
        emissions: ``(B, L, T)`` float scores.
        mask: ``(B, L)`` bool/int; ``mask[:, 0]`` must be all True and each row
            must be a prefix (right padding), as torchcrf requires.
        start_transitions: ``(T,)``.
        end_transitions: ``(T,)``.
        transitions: ``(T, T)`` ``transitions[i, j]`` = score of ``i → j``.

    Returns one list of tag ids per row, trimmed to the row's mask length.
    """
    emissions = np.asarray(emissions, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    start = np.asarray(start_transitions, dtype=np.float32)
    end = np.asarray(end_transitions, dtype=np.float32)
    trans = np.asarray(transitions, dtype=np.float32)
    if emissions.ndim != 3 or mask.ndim != 2 or emissions.shape[:2] != mask.shape:
        raise ValueError("emissions must be (B, L, T) and mask (B, L)")
    if mask.shape[1] == 0:
        return [[] for _ in range(mask.shape[0])]
    if not mask[:, 0].all():
        raise ValueError("mask of the first timestep must all be on")

    batch, length, _num_tags = emissions.shape
    score = start[None, :] + emissions[:, 0]  # (B, T)
    history: list[np.ndarray] = []
    for i in range(1, length):
        # (B, T_prev, 1) + (T_prev, T_next) + (B, 1, T_next)
        next_score = score[:, :, None] + trans[None, :, :] + emissions[:, i][:, None, :]
        best_prev = next_score.argmax(axis=1)  # (B, T_next) first max wins
        best_score = np.take_along_axis(next_score, best_prev[:, None, :], axis=1)[:, 0, :]
        step_on = mask[:, i][:, None]
        score = np.where(step_on, best_score, score)
        history.append(best_prev)

    score = score + end[None, :]
    seq_ends = mask.sum(axis=1) - 1

    best_tags_list: list[list[int]] = []
    for idx in range(batch):
        best_last = int(score[idx].argmax())
        best_tags = [best_last]
        for hist in reversed(history[: int(seq_ends[idx])]):
            best_last = int(hist[idx][best_tags[-1]])
            best_tags.append(best_last)
        best_tags.reverse()
        best_tags_list.append(best_tags)
    return best_tags_list
