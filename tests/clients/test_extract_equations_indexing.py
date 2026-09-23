"""Tests for LLMClient.extract_equations sentence_index handling.

Each batch sentence is sent as ``[i] (text_id=...)`` and the model reports the
``sentence_index`` of every component. Only an index sent in that batch
identifies the source sentence; anything else must be dropped rather than
attributed to a guessed sentence.
"""

import logging
from unittest import mock

import pytest

from bibr.clients.llm import LLMClient
from bibr.schemas import EquationComponentLLM, EquationExtractionResult

_BATCH = [(7, "The first effect held (t: 2.1, p: .03)."), (9, "The second (r: .45; 12 vs 17).")]


@pytest.fixture
def client():
    c = LLMClient()
    c._limiter = mock.Mock()
    c._limiter.acquire = mock.AsyncMock()
    return c


def _component(sentence_index, rhs, lhs="p"):
    # model_construct skips validation so non-integer indices reach the client
    # the way an unvalidated backend response would.
    return EquationComponentLLM.model_construct(
        sentence_index=sentence_index, lhs=lhs, df="", comp="=", rhs=rhs
    )


async def _extract(client, components, batch=_BATCH):
    fake = EquationExtractionResult.model_construct(equations=components)
    with mock.patch.object(client, "_invoke_structured", new=mock.AsyncMock(return_value=fake)):
        return await client.extract_equations(batch, file_hash="h")


async def test_valid_indices_map_to_their_sentences(client):
    eqs = await _extract(client, [_component(0, ".03"), _component(1, ".45", lhs="r")])

    assert [(eq.text_id, eq.lhs, eq.rhs) for eq in eqs] == [(7, "p", ".03"), (9, "r", ".45")]


@pytest.mark.parametrize(
    "bad_index",
    [None, 2, -1, 9, "1", 1.5, True],
    ids=["missing", "out-of-range", "negative", "text-id", "string", "float", "bool"],
)
async def test_invalid_index_is_dropped_and_valid_items_kept(client, bad_index):
    eqs = await _extract(client, [_component(bad_index, ".05"), _component(1, ".45", lhs="r")])

    assert [(eq.text_id, eq.lhs, eq.rhs) for eq in eqs] == [(9, "r", ".45")]


async def test_dropped_components_are_logged_and_counted(client, caplog):
    components = [_component(None, ".05"), _component(5, ".05"), _component(0, ".03")]

    with caplog.at_level(logging.DEBUG, logger="bibr.clients.llm"):
        eqs = await _extract(client, components)

    assert [eq.text_id for eq in eqs] == [7]
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert sum("invalid sentence_index" in message for message in debug) == 2
    assert any(
        "extracted 1 equation components" in r.getMessage()
        and "dropped 2 with an invalid sentence_index" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
    )


@pytest.mark.parametrize("absent", [False, True], ids=["null", "absent"])
async def test_missing_index_in_single_sentence_batch_maps_to_that_sentence(client, absent):
    if absent:
        component = EquationComponentLLM(lhs="p", comp="=", rhs=".03")
    else:
        component = _component(None, ".03")

    eqs = await _extract(client, [component], batch=_BATCH[:1])

    assert [(eq.text_id, eq.lhs, eq.rhs) for eq in eqs] == [(7, "p", ".03")]


@pytest.mark.parametrize(
    "bad_index",
    [1, -1, 7, False, "0", 0.0],
    ids=["out-of-range", "negative", "text-id", "bool", "string", "float"],
)
async def test_wrong_index_in_single_sentence_batch_is_dropped(client, bad_index):
    eqs = await _extract(client, [_component(bad_index, ".03")], batch=_BATCH[:1])

    assert eqs == []
