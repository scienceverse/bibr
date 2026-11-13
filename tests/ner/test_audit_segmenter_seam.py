"""Window seams of the CRF reference segmenter: a reference continued across them survives."""

from tests.ner.test_segmenter import _stubbed_segmenter


class _SeamModel:
    """Two windows whose own BIO paths are valid but disagree where their votes meet.

    window=6, stride=3 over 9 words: words 0-4 keep window 0's tags, 5-8 window 1's.
    After the +1 shift window 0 reads O B I I O O and window 1 (words 3-8)
    O B I I I I, so the stitched tags are O B I I O | I I I I.
    """

    def predict(self, ids, _attention):
        first = ids[0][0] == 10
        return [[1, 2, 2, 0, 0, 0]] if first else [[1, 2, 2, 2, 2, 2]]


def test_reference_continued_past_a_window_seam_is_kept(monkeypatch):
    words = [f"w{i}" for i in range(9)]
    segmenter = _stubbed_segmenter(monkeypatch, 6, 3, _SeamModel())

    assert segmenter.segment(" ".join(words)) == ["w1 w2 w3", "w5 w6 w7 w8"]
