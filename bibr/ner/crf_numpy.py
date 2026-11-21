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
        mask: ``(B, L)`` bool/int; each row must be a prefix (right padding)
            and, unless it is all off (decoded as ``[]``), start on, as torchcrf
            requires.
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
    # A text the tokenizer reduces to no tokens (a BERT normalizer drops
    # zero-width characters) is an all-off row in a padded batch.
    empty = ~mask.any(axis=1)
    if not (mask[:, 0] | empty).all():
        raise ValueError("mask of the first timestep must all be on")

    batch, length, num_tags = emissions.shape
    score = start[None, :] + emissions[:, 0]  # (B, T)
    if batch == 1 and bool(mask.all()):
        # Fast path for single unpadded sequences (e.g. front_tagger, L=1000..3000):
        # 2D (T, T) ops, no 3D broadcast, no per-step np.where, single history array.
        sc = score[0]
        em = emissions[0]
        history_1 = np.empty((length - 1, num_tags), dtype=np.intp)
        tag_idx = np.arange(num_tags)
        for i in range(1, length):
            cand = sc[:, None] + trans
            best_prev = cand.argmax(axis=0)
            sc = cand[best_prev, tag_idx] + em[i]
            history_1[i - 1] = best_prev
        best_last = int((sc + end).argmax())
        best_tags = [best_last]
        for i in range(length - 2, -1, -1):
            best_last = int(history_1[i, best_last])
            best_tags.append(best_last)
        best_tags.reverse()
        return [best_tags]

    history = np.empty((length - 1, batch, num_tags), dtype=np.intp)
    batch_idx = np.arange(batch)[:, None]
    tag_idx = np.arange(num_tags)[None, :]
    all_on = mask.all(axis=0)
    trans_3d = trans[None, :, :]
    for i in range(1, length):
        # Hoist emissions[:, i] out of the (B, T, T) broadcast: adding a constant
        # per next-tag column does not change argmax over previous tags.
        cand = score[:, :, None] + trans_3d
        best_prev = cand.argmax(axis=1)  # (B, T_next) first max wins
        best_score = cand[batch_idx, best_prev, tag_idx] + emissions[:, i]
        if all_on[i]:
            score = best_score
        else:
            score = np.where(mask[:, i][:, None], best_score, score)
        history[i - 1] = best_prev

    score = score + end[None, :]
    seq_ends = mask.sum(axis=1) - 1

    best_tags_list: list[list[int]] = []
    for idx in range(batch):
        if empty[idx]:
            best_tags_list.append([])
            continue
        best_last = int(score[idx].argmax())
        best_tags = [best_last]
        for i in range(int(seq_ends[idx]) - 1, -1, -1):
            best_last = int(history[i, idx, best_last])
            best_tags.append(best_last)
        best_tags.reverse()
        best_tags_list.append(best_tags)
    return best_tags_list


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = values.max(axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.log(np.exp(values - peak).sum(axis=axis)) + np.squeeze(peak, axis=axis)


def marginals(
    emissions: np.ndarray,
    *,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
    transitions: np.ndarray,
) -> np.ndarray:
    """Per-step tag posteriors ``(L, T)`` of one unpadded sequence (forward-backward).

    ``marginals(...)[i, t]`` is the probability that step ``i`` carries tag
    ``t`` under the CRF, summed over every path; the tagger reads it as the
    confidence of the Viterbi tag.
    """
    emissions = np.asarray(emissions, dtype=np.float64)
    if emissions.ndim != 2:
        raise ValueError("emissions must be (L, T)")
    length = emissions.shape[0]
    if length == 0:
        return np.zeros_like(emissions)
    start = np.asarray(start_transitions, dtype=np.float64)
    end = np.asarray(end_transitions, dtype=np.float64)
    trans = np.asarray(transitions, dtype=np.float64)

    # Precompute shifted transition exponentials so each forward/backward step
    # does T scalar exps + a (T, T) BLAS gemv instead of T*T exps. Any entry
    # whose dot product underflows (<= 1e-300) falls back to exact logsumexp.
    t_col_max = trans.max(axis=0)
    t_col_safe = np.where(np.isfinite(t_col_max), t_col_max, 0.0)
    exp_trans_col = np.ascontiguousarray(np.exp(trans - t_col_safe[None, :]))
    t_row_max = trans.max(axis=1)
    t_row_safe = np.where(np.isfinite(t_row_max), t_row_max, 0.0)
    exp_trans_row = np.ascontiguousarray(np.exp(trans - t_row_safe[:, None]))

    alpha = np.empty_like(emissions)
    alpha[0] = start + emissions[0]
    for i in range(1, length):
        prev = alpha[i - 1]
        m = float(prev.max())
        if np.isfinite(m):
            s = np.exp(prev - m) @ exp_trans_col
            if float(s.min()) > 1e-300:
                alpha[i] = np.log(s) + (m + t_col_safe) + emissions[i]
            else:
                step = np.empty_like(prev)
                good = s > 1e-300
                if good.any():
                    step[good] = np.log(s[good]) + (m + t_col_safe[good])
                bad = ~good
                step[bad] = _logsumexp(prev[:, None] + trans[:, bad], axis=0)
                alpha[i] = step + emissions[i]
        else:
            alpha[i] = _logsumexp(prev[:, None] + trans, axis=0) + emissions[i]

    beta = np.empty_like(emissions)
    beta[-1] = end
    for i in range(length - 2, -1, -1):
        nxt = emissions[i + 1] + beta[i + 1]
        m = float(nxt.max())
        if np.isfinite(m):
            s = exp_trans_row @ np.exp(nxt - m)
            if float(s.min()) > 1e-300:
                beta[i] = np.log(s) + (m + t_row_safe)
            else:
                step = np.empty_like(nxt)
                good = s > 1e-300
                if good.any():
                    step[good] = np.log(s[good]) + (m + t_row_safe[good])
                bad = ~good
                step[bad] = _logsumexp(trans[bad, :] + nxt[None, :], axis=1)
                beta[i] = step
        else:
            beta[i] = _logsumexp(trans + nxt[None, :], axis=1)

    log_z = _logsumexp(alpha[-1] + end, axis=0)
    return np.exp(alpha + beta - log_z)
