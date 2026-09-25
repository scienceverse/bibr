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


# ---------------------------------------------------------------------------
# local-runtimes sweep: multi-window merge coverage without torch (ml-4)
# ---------------------------------------------------------------------------


def _stubbed_segmenter(window, stride, model):
    """RefSegmenter with stubbed torch/transformers so tests run anywhere.

    Uses setdefault: a real torch install (developer machines, CI with
    models) is never shadowed — the stubs only fill the gaps in minimal
    environments.
    """
    import sys
    import types

    torch_stub = sys.modules.get("torch")
    if torch_stub is None:
        torch_stub = types.ModuleType("torch")
        torch_stub.tensor = lambda data, **kwargs: data
        torch_stub.ones_like = lambda data: data
        torch_stub.load = lambda *args, **kwargs: {}
        torch_stub.long = "long"
        torch_nn = types.ModuleType("torch.nn")

        class _Module:
            pass

        torch_nn.Module = _Module
        torch_stub.nn = torch_nn
        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = torch_nn
    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = object
        sys.modules["transformers"] = transformers_stub
    for name in ("bibr.ner.checkpoint", "bibr.ner.model"):
        if name not in sys.modules:
            sibling = types.ModuleType(name)
            sibling.resolve_checkpoint = lambda *args, **kwargs: ""
            sibling.EncoderCRFModel = object
            sys.modules[name] = sibling

    from bibr.ner.segmenter import RefSegmenter

    segmenter = object.__new__(RefSegmenter)
    segmenter.tokenizer = _WordTokenizer()
    segmenter.model = model
    segmenter.device = "cpu"
    segmenter.window = window
    segmenter.stride = stride
    return segmenter


class _WordTokenizer:
    """One token per whitespace-separated word, with real char offsets."""

    def __call__(self, text, **_kwargs):
        input_ids = []
        offsets = []
        pos = 0
        token = 10
        for word in text.split(" "):
            start = text.index(word, pos)
            end = start + len(word)
            input_ids.append(token)
            offsets.append((start, end))
            token += 1
            pos = end
        return {"input_ids": input_ids, "offset_mapping": offsets}


class _LocalZeroBRefModel:
    """Emits B-REF at local position 0 of every window, O elsewhere.

    After the segmenter's +1 shift the B-REF lands on local position 1, so
    overlapping windows vote on shared tokens and the interior-trust rule
    (with first-window-wins ties) decides — exactly the merge path under
    test.
    """

    def predict(self, ids, _attention):
        window = ids[0] if isinstance(ids, list) else ids
        return [[1 if i == 0 else 0 for i in range(len(window))]]


class _AllBRefModel:
    """Every position starts a reference: maximal boundary coverage."""

    def predict(self, ids, _attention):
        window = ids[0] if isinstance(ids, list) else ids
        return [[1] * len(window)]


def test_window_merge_trust_votes_and_ties():
    """Overlapping windows vote by interior trust; ties keep the first (ml-4).

    window=6, stride=3 over 10 words. Hand-computed: only absolute position 1
    wins B-REF (trust 1 from window 0); every tied overlap (positions 4, 7)
    keeps the earlier window's O vote under the strict-> rule; and since O
    terminates a reference, the single B-REF yields exactly one ref.
    """
    words = [f"w{i}" for i in range(10)]
    text = " ".join(words)
    segmenter = _stubbed_segmenter(6, 3, _LocalZeroBRefModel())
    refs = segmenter.segment(text)
    assert refs == ["w1"]


def test_window_merge_drops_no_boundaries():
    """A B-REF-everywhere model over many windows returns every word but the
    first (ml-4).

    Absolute position 0 only ever sees local position 0 votes, which the +1
    shift maps to O — so the first word can never open a reference. Every
    other boundary survives all windows.
    """
    words = [f"ref{i}" for i in range(25)]
    text = " ".join(words)
    segmenter = _stubbed_segmenter(8, 6, _AllBRefModel())
    assert segmenter.segment(text) == words[1:]


def test_window_size_does_not_change_constant_model_output():
    """A position-constant fake segments identically at any window size (ml-4).

    The shift applies uniformly to every window, so windowing only changes
    coverage, never the per-position tag mapping.
    """
    words = [f"ref{i}" for i in range(25)]
    text = " ".join(words)
    small = _stubbed_segmenter(8, 6, _AllBRefModel()).segment(text)
    big = _stubbed_segmenter(2048, 1536, _AllBRefModel()).segment(text)
    assert small == big == words[1:]
