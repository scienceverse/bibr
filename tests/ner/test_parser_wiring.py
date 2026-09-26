"""RefParser.parse wiring tests, with tokenizer/model stubbed.

Bypasses RefParser.__init__ (no ModernBERT / checkpoint download) and injects
stubs so we can assert the two things that broke the v4 checkpoint on the old
v1 path: ``add_special_tokens=False`` and the absence of the +1 tag shift.
"""

import pytest

torch = pytest.importorskip("torch")

from bibr.ner.parser import RefParser
from bibr.ner.tags import BIO_TAGS


class _StubTokenizer:
    def __init__(self, offsets):
        self._offsets = offsets
        self.calls: list[dict] = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        n = len(self._offsets)
        return {
            "input_ids": torch.zeros(1, n, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
            "offset_mapping": torch.tensor([self._offsets]),
        }


class _StubModel:
    feature_dim = 8

    def __init__(self, pred_indices):
        self._preds = pred_indices
        self.token_features_seen = "unset"

    def predict(self, input_ids, attention_mask, token_features=None):  # noqa: ARG002
        self.token_features_seen = token_features
        return [self._preds]


def _make_parser(offsets, tag_names):
    parser = RefParser.__new__(RefParser)
    parser.device = "cpu"
    parser.max_seq_len = 256
    parser.tokenizer = _StubTokenizer(offsets)
    parser.model = _StubModel([BIO_TAGS.index(t) for t in tag_names])
    return parser


def test_parse_uses_no_special_tokens_and_no_shift():
    text = "Smith J 2020"
    offsets = [(0, 5), (6, 7), (8, 12)]
    parser = _make_parser(offsets, ["B-AUTHOR", "I-AUTHOR", "B-YEAR"])

    result = parser.parse(text)

    # No shift: a +1 shift would yield {"authors": "J 2020"} and no year.
    assert result == {"authors": "Smith J", "year": 2020}
    assert parser.tokenizer.calls[0]["add_special_tokens"] is False


def test_parse_passes_zeros_token_features():
    # The v4 checkpoint's feature_proj.weight is ~0 but feature_proj.bias is a
    # learned constant added to every token during training. Passing None drops
    # that residual (train/inference mismatch); parse must pass a zeros feature
    # tensor so the bias is re-applied. Values are inert (weight ~0), so zeros
    # exactly reproduces the training forward path.
    parser = _make_parser([(0, 5)], ["B-AUTHOR"])
    parser.parse("Smith")
    tf = parser.model.token_features_seen
    assert tf is not None
    assert tf.shape == (1, 1, parser.model.feature_dim)
    assert bool((tf == 0).all())


def test_parse_empty_string_short_circuits():
    parser = _make_parser([(0, 1)], ["B-AUTHOR"])
    assert parser.parse("   ") == {}


def test_parse_scrubs_lone_surrogates_before_tokenizing():
    class _SurrogateRejectingTokenizer(_StubTokenizer):
        def __call__(self, text, **kwargs):
            self.seen_text = text
            if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
                raise TypeError(
                    "TextEncodeInput must be Union[TextInputSequence, "
                    "Tuple[InputSequence, InputSequence]]"
                )
            return super().__call__(text, **kwargs)

    parser = RefParser.__new__(RefParser)
    parser.device = "cpu"
    parser.max_seq_len = 256
    parser.tokenizer = _SurrogateRejectingTokenizer([(0, 5), (6, 7), (8, 12)])
    parser.model = _StubModel([BIO_TAGS.index(tag) for tag in ["B-AUTHOR", "I-AUTHOR", "B-YEAR"]])

    result = parser.parse("Sm\udce9th J 2020")

    assert result.get("year") == 2020
    decoded = "".join(str(value) for value in result.values())
    assert decoded and all(not (0xD800 <= ord(char) <= 0xDFFF) for char in decoded)
    assert all(not (0xD800 <= ord(char) <= 0xDFFF) for char in parser.tokenizer.seen_text)


class _BatchStubTokenizer:
    """Batch-aware stub: maps each text to fixed offsets, right-pads to max len.

    Embeds a per-text sentinel id in ``input_ids[:, 0]`` so the model stub can
    align predictions by identity regardless of batch order — this is what lets
    the tests exercise internal length-sorting (sort for the forward pass,
    unsort the results).
    """

    def __init__(self, text_to_offsets):
        self._map = text_to_offsets
        self._ids = {t: i + 1 for i, t in enumerate(text_to_offsets)}
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        per = [self._map[t] for t in texts]
        max_len = max(len(o) for o in per)
        b = len(texts)
        input_ids = torch.zeros(b, max_len, dtype=torch.long)
        attn = torch.zeros(b, max_len, dtype=torch.long)
        offs = torch.zeros(b, max_len, 2, dtype=torch.long)
        for i, (t, o) in enumerate(zip(texts, per, strict=True)):
            input_ids[i, 0] = self._ids[t]
            attn[i, : len(o)] = 1
            for j, (s, e) in enumerate(o):
                offs[i, j, 0], offs[i, j, 1] = s, e
        return {"input_ids": input_ids, "attention_mask": attn, "offset_mapping": offs}


class _BatchStubModel:
    feature_dim = 8

    def __init__(self, id_to_preds):
        self._by_id = id_to_preds
        self.token_features_seen = "unset"

    def predict(self, input_ids, attention_mask, token_features=None):
        # Mimic torchcrf.decode: one sequence per row, trimmed to its mask
        # length, looked up by the row's sentinel id (order-independent).
        self.token_features_seen = token_features
        return [
            self._by_id[int(input_ids[i, 0])][: int(attention_mask[i].sum())]
            for i in range(input_ids.shape[0])
        ]


def _make_batch_parser(text_to_offsets, text_to_tags):
    ids = {t: i + 1 for i, t in enumerate(text_to_offsets)}
    parser = RefParser.__new__(RefParser)
    parser.device = "cpu"
    parser.max_seq_len = 256
    parser.tokenizer = _BatchStubTokenizer(text_to_offsets)
    parser.model = _BatchStubModel(
        {ids[t]: [BIO_TAGS.index(x) for x in tags] for t, tags in text_to_tags.items()}
    )
    return parser


def test_parse_batch_decodes_each_ref_with_padding():
    # Two refs of different token lengths → exercises right-padding + per-ref
    # offset alignment (the batch must decode each row against its own offsets).
    t1, t2 = "Smith J", "Doe A 2021"
    offsets = {t1: [(0, 5), (6, 7)], t2: [(0, 3), (4, 5), (6, 10)]}
    tags = {t1: ["B-AUTHOR", "I-AUTHOR"], t2: ["B-AUTHOR", "I-AUTHOR", "B-YEAR"]}
    parser = _make_batch_parser(offsets, tags)

    results = parser.parse_batch([t1, t2])

    assert results == [{"authors": "Smith J"}, {"authors": "Doe A", "year": 2021}]
    # Single forward pass over the padded batch, with zeros token_features.
    tf = parser.model.token_features_seen
    assert tf.shape == (2, 3, parser.model.feature_dim)
    assert bool((tf == 0).all())


def test_parse_batch_preserves_input_order_under_length_sorting():
    # Inputs in DESCENDING length: internal length-sorting reorders the forward
    # pass for efficiency but results must come back in input order. batch_size=1
    # forces each into its own (sorted-order) chunk to stress the unsort.
    long_t, short_t = "Alpha Beta Gamma Delta", "Xi"
    offsets = {long_t: [(0, 5), (6, 10), (11, 16), (17, 22)], short_t: [(0, 2)]}
    tags = {long_t: ["B-AUTHOR"] + ["I-AUTHOR"] * 3, short_t: ["B-AUTHOR"]}
    parser = _make_batch_parser(offsets, tags)

    results = parser.parse_batch([long_t, short_t], batch_size=1)

    assert results == [{"authors": "Alpha Beta Gamma Delta"}, {"authors": "Xi"}]


def test_parse_batch_handles_empty_strings_like_parse():
    t = "Smith J"
    parser = _make_batch_parser({t: [(0, 5), (6, 7)]}, {t: ["B-AUTHOR", "I-AUTHOR"]})
    # Empty/whitespace entries map to {} (parse() short-circuits them) and must
    # not consume a model slot or shift the others.
    assert parser.parse_batch(["", t, "   "]) == [{}, {"authors": "Smith J"}, {}]


def test_parse_batch_empty_list_returns_empty():
    parser = _make_batch_parser({"x": [(0, 1)]}, {"x": ["B-AUTHOR"]})
    assert parser.parse_batch([]) == []


@pytest.mark.slow
@pytest.mark.network  # real Hub checkpoint by design; opts out of the socket guard.
def test_parse_batch_matches_individual_real_model():
    """Batching is a pure perf optimization: it must not change outputs.

    Real checkpoint, varied lengths + styles + an empty entry — catches any
    padding-side / offset / mask misalignment the stubs can't.
    """
    parser = RefParser("scienceverse/bibr-parser-v4-5-gold", device="cpu")
    refs = [
        "Tulving, E. (1985). Memory and consciousness. Canadian Psychology, 26(1), 1-12.",
        "Wongvibulsin S, Habeos EE, et al. Digital health. J Med Internet Res 2021; 23: e18773.",
        "Kahneman, D. (2011). Thinking, fast and slow. Farrar, Straus and Giroux.",
        "",
    ]
    # batch_size=2 forces multiple chunks over the 4 inputs.
    batched = parser.parse_batch(refs, batch_size=2)
    individual = [parser.parse(r) for r in refs]
    assert batched == individual
