"""Reference segmentation behavior without a downloaded checkpoint."""

import pytest


def test_segmenter_keeps_open_final_reference():
    pytest.importorskip("torch")
    from bibr.ner.segmenter import RefSegmenter

    class Tokenizer:
        def __call__(self, _text, **_kwargs):
            return {
                "input_ids": [101, 11, 12, 102],
                "offset_mapping": [(0, 0), (0, 5), (6, 14), (0, 0)],
            }

    class Model:
        def predict(self, _ids, _attention):
            # RefSegmenter shifts the predictions right by one token.
            return [[1, 2, 0, 0]]

    segmenter = object.__new__(RefSegmenter)
    segmenter.tokenizer = Tokenizer()
    segmenter.model = Model()
    segmenter.device = "cpu"
    segmenter.window = 2048
    segmenter.stride = 1536

    assert segmenter.segment("Smith 2020 ref") == ["Smith 2020 ref"]


def test_segmenter_scrubs_lone_surrogates_before_tokenizing():
    pytest.importorskip("torch")
    from bibr.ner.segmenter import RefSegmenter

    class Tokenizer:
        def __init__(self):
            self.seen_text = None

        def __call__(self, text, **_kwargs):
            self.seen_text = text
            if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
                raise TypeError(
                    "TextEncodeInput must be Union[TextInputSequence, "
                    "Tuple[InputSequence, InputSequence]]"
                )
            return {
                "input_ids": [101, 11, 12, 102],
                "offset_mapping": [(0, 0), (0, 5), (6, 14), (0, 0)],
            }

    class Model:
        def predict(self, _ids, _attention):
            return [[1, 2, 0, 0]]

    segmenter = object.__new__(RefSegmenter)
    segmenter.tokenizer = Tokenizer()
    segmenter.model = Model()
    segmenter.device = "cpu"
    segmenter.window = 2048
    segmenter.stride = 1536

    refs = segmenter.segment("Smith 2020 re\udce9")

    assert refs == ["Smith 2020 re�"]
    assert all(not (0xD800 <= ord(char) <= 0xDFFF) for char in segmenter.tokenizer.seen_text)
