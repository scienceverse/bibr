"""Numerical parity tests for Fix 2B (CRF marginals, Viterbi fast path, tokenizer need_offsets)."""

from __future__ import annotations

import numpy as np

from bibr.ner.crf_numpy import _logsumexp, marginals, viterbi_decode
from bibr.utils.ml_runtime import read_onnx_manifest
from bibr.utils.onnx_tokenizer import encode_batch, load_tokenizer


def _naive_marginals(
    em: np.ndarray, *, start: np.ndarray, end: np.ndarray, trans: np.ndarray
) -> np.ndarray:
    em = np.asarray(em, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    alpha = np.empty_like(em)
    alpha[0] = start + em[0]
    for i in range(1, len(em)):
        alpha[i] = _logsumexp(alpha[i - 1][:, None] + trans, axis=0) + em[i]
    beta = np.empty_like(em)
    beta[-1] = end
    for i in range(len(em) - 2, -1, -1):
        beta[i] = _logsumexp(trans + (em[i + 1] + beta[i + 1])[None, :], axis=1)
    log_z = _logsumexp(alpha[-1] + end, axis=0)
    return np.exp(alpha + beta - log_z)


def test_marginals_matches_naive_logsumexp_including_underflow_and_inf() -> None:
    rng = np.random.default_rng(7)
    L, T = 200, 43
    em = rng.standard_normal((L, T)).astype(np.float32) * 2.0
    st = rng.standard_normal(T).astype(np.float32)
    en = rng.standard_normal(T).astype(np.float32)
    tr = rng.standard_normal((T, T)).astype(np.float32) * 0.5

    # Standard case
    got = marginals(em, start_transitions=st, end_transitions=en, transitions=tr)
    ref = _naive_marginals(em, start=st, end=en, trans=tr)
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)

    # Extreme negative transitions (forces underflow fallback path)
    tr_ext = tr.copy()
    tr_ext[:, :15] = -1000.0
    tr_ext[:15, :] = -1000.0
    got_ext = marginals(em, start_transitions=st, end_transitions=en, transitions=tr_ext)
    ref_ext = _naive_marginals(em, start=st, end=en, trans=tr_ext)
    np.testing.assert_allclose(got_ext, ref_ext, atol=1e-10, rtol=1e-10)

    # -inf masked transitions (forbidden BIO pairs)
    tr_inf = tr.copy()
    tr_inf[0, 1] = -np.inf
    tr_inf[2, :] = -np.inf
    tr_inf[2, 2] = 0.0
    got_inf = marginals(em, start_transitions=st, end_transitions=en, transitions=tr_inf)
    ref_inf = _naive_marginals(em, start=st, end=en, trans=tr_inf)
    np.testing.assert_allclose(got_inf, ref_inf, atol=1e-10, rtol=1e-10)


def test_viterbi_single_and_batched_match_each_other_and_handle_empty_rows() -> None:
    rng = np.random.default_rng(19)
    B, L, T = 5, 40, 17
    em = rng.standard_normal((B, L, T)).astype(np.float32)
    st = rng.standard_normal(T).astype(np.float32)
    en = rng.standard_normal(T).astype(np.float32)
    tr = rng.standard_normal((T, T)).astype(np.float32)
    mask = np.ones((B, L), dtype=bool)
    mask[1, 15:] = False
    mask[3, :] = False  # all-off empty row

    batched = viterbi_decode(em, mask, start_transitions=st, end_transitions=en, transitions=tr)
    assert batched[3] == []
    for b in (0, 1, 2, 4):
        seq_len = int(mask[b].sum())
        single = viterbi_decode(
            em[b : b + 1, :seq_len],
            np.ones((1, seq_len), dtype=bool),
            start_transitions=st,
            end_transitions=en,
            transitions=tr,
        )[0]
        assert batched[b] == single


def test_encode_batch_need_offsets_false_matches_ids_and_mask() -> None:
    from tests import test_onnx_runtime_core as core

    bdir = core._bundle("section")
    tok = load_tokenizer(bdir, read_onnx_manifest(bdir))
    texts = ["1. Intro [SEP] Some body text.", "2. Methods [SEP] More details here."]
    full = encode_batch(tok, texts, max_length=32, need_offsets=True)
    fast = encode_batch(tok, texts, max_length=32, need_offsets=False)
    np.testing.assert_array_equal(fast.input_ids, full.input_ids)
    np.testing.assert_array_equal(fast.attention_mask, full.attention_mask)
    assert fast.offsets == []
