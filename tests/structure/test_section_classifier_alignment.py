"""M5: LLM section classifications were mapped back purely by position.

A short or reordered response shifted every classification after the first
omission onto the wrong header — silently, and for the rest of the document.
The response echoes each header, so the echo is verified before the position
is trusted.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.paper_contents import CanonicalSection
from bibr.structure.section_classifier import (
    _classify_llm_batch,
    _normalize_header_echo,
)

HEADERS = ["Introduction", "Methods", "Results", "Discussion"]


class _Cls:
    def __init__(self, header, section_type):
        self.header = header
        self.section_type = section_type


def _client(classifications):
    client = MagicMock()
    result = MagicMock()
    result.classifications = classifications
    client.invoke_structured = AsyncMock(return_value=result)
    return client


async def _classify(classifications):
    return await _classify_llm_batch(HEADERS, llm_client=_client(classifications))


def _types(results):
    return [canon for canon, _score in results]


class TestEchoVerifiedMapping:
    async def test_a_well_formed_response_maps_positionally(self):
        results = await _classify(
            [
                _Cls("Introduction", "intro"),
                _Cls("Methods", "method"),
                _Cls("Results", "results"),
                _Cls("Discussion", "discussion"),
            ]
        )

        assert _types(results) == [
            CanonicalSection.INTRODUCTION,
            CanonicalSection.METHODS,
            CanonicalSection.RESULTS,
            CanonicalSection.DISCUSSION,
        ]

    async def test_a_short_response_no_longer_shifts_every_later_header(self):
        """ "Methods" was omitted; positionally, everything after shifts up."""
        results = await _classify(
            [
                _Cls("Introduction", "intro"),
                _Cls("Results", "results"),
                _Cls("Discussion", "discussion"),
            ]
        )

        assert _types(results) == [
            CanonicalSection.INTRODUCTION,
            CanonicalSection.UNKNOWN,  # Methods went unclassified, not mislabelled
            CanonicalSection.RESULTS,
            CanonicalSection.DISCUSSION,
        ]

    async def test_a_reordered_response_is_matched_by_its_echo(self):
        results = await _classify(
            [
                _Cls("Discussion", "discussion"),
                _Cls("Introduction", "intro"),
                _Cls("Results", "results"),
                _Cls("Methods", "method"),
            ]
        )

        assert _types(results) == [
            CanonicalSection.INTRODUCTION,
            CanonicalSection.METHODS,
            CanonicalSection.RESULTS,
            CanonicalSection.DISCUSSION,
        ]

    async def test_an_unplaceable_classification_is_dropped_not_reassigned(self):
        results = await _classify(
            [
                _Cls("Introduction", "intro"),
                _Cls("A Header That Was Never Sent", "method"),
            ]
        )

        assert _types(results) == [
            CanonicalSection.INTRODUCTION,
            CanonicalSection.UNKNOWN,
            CanonicalSection.UNKNOWN,
            CanonicalSection.UNKNOWN,
        ]

    async def test_a_model_that_does_not_echo_keeps_positional_mapping(self):
        """Not every model fills the echo field; the count must still match."""
        results = await _classify(
            [
                _Cls("", "intro"),
                _Cls("", "method"),
                _Cls("", "results"),
                _Cls("", "discussion"),
            ]
        )

        assert _types(results) == [
            CanonicalSection.INTRODUCTION,
            CanonicalSection.METHODS,
            CanonicalSection.RESULTS,
            CanonicalSection.DISCUSSION,
        ]

    async def test_cosmetic_echo_differences_still_match(self):
        results = await _classify(
            [
                _Cls("  introduction ", "intro"),
                _Cls("Methods:", "method"),
                _Cls("RESULTS.", "results"),
                _Cls("Discussion", "discussion"),
            ]
        )

        assert _types(results) == [
            CanonicalSection.INTRODUCTION,
            CanonicalSection.METHODS,
            CanonicalSection.RESULTS,
            CanonicalSection.DISCUSSION,
        ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Methods", "methods"),
        ("  Methods  ", "methods"),
        ("Methods:", "methods"),
        ("Methods.", "methods"),
        ("METHODS", "methods"),
        ("Materials   and  Methods", "materials and methods"),
        (None, ""),
        ("", ""),
    ],
)
def test_echo_normalization(raw, expected):
    assert _normalize_header_echo(raw) == expected
