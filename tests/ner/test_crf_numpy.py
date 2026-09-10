"""numpy Viterbi decode must reproduce torchcrf's ``CRF.decode`` exactly."""

from __future__ import annotations

import numpy as np
import pytest

from bibr.ner.crf_numpy import viterbi_decode


def _random_case(rng, batch, length, tags):
    emissions = rng.normal(size=(batch, length, tags)).astype(np.float32)
    lengths = rng.integers(1, length + 1, size=batch)
    lengths[0] = length
    mask = np.zeros((batch, length), dtype=bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    start = rng.normal(size=tags).astype(np.float32)
    end = rng.normal(size=tags).astype(np.float32)
    trans = rng.normal(size=(tags, tags)).astype(np.float32)
    # A few hard constraints, like the BIO masks the parser applies.
    trans[rng.integers(0, tags, size=5), rng.integers(0, tags, size=5)] = -10000.0
    start[rng.integers(0, tags, size=2)] = -10000.0
    return emissions, mask, start, end, trans


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_torchcrf(seed):
    torch = pytest.importorskip("torch")
    torchcrf = pytest.importorskip("torchcrf")

    rng = np.random.default_rng(seed)
    emissions, mask, start, end, trans = _random_case(rng, batch=4, length=9, tags=6)
    crf = torchcrf.CRF(6, batch_first=True)
    with torch.no_grad():
        crf.start_transitions.copy_(torch.from_numpy(start))
        crf.end_transitions.copy_(torch.from_numpy(end))
        crf.transitions.copy_(torch.from_numpy(trans))
    expected = crf.decode(torch.from_numpy(emissions), mask=torch.from_numpy(mask))
    got = viterbi_decode(
        emissions, mask, start_transitions=start, end_transitions=end, transitions=trans
    )
    assert got == expected


def test_lengths_follow_mask():
    rng = np.random.default_rng(9)
    emissions, mask, start, end, trans = _random_case(rng, batch=3, length=5, tags=4)
    got = viterbi_decode(
        emissions, mask, start_transitions=start, end_transitions=end, transitions=trans
    )
    assert [len(row) for row in got] == mask.sum(axis=1).tolist()


def test_rejects_masked_first_step():
    with pytest.raises(ValueError, match="first timestep"):
        viterbi_decode(
            np.zeros((1, 2, 3), dtype=np.float32),
            np.array([[False, True]]),
            start_transitions=np.zeros(3),
            end_transitions=np.zeros(3),
            transitions=np.zeros((3, 3)),
        )


def test_single_step_sequence():
    emissions = np.array([[[0.1, 5.0, -1.0]]], dtype=np.float32)
    got = viterbi_decode(
        emissions,
        np.array([[True]]),
        start_transitions=np.zeros(3),
        end_transitions=np.array([0.0, -100.0, 0.0]),  # end penalty flips the answer
        transitions=np.zeros((3, 3)),
    )
    assert got == [[0]]
