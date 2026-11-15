"""The city a fast run reads from an affiliation that prints a UK postcode after it."""

from __future__ import annotations

import pytest

from bibr.extract.affiliation_rules import parse_affiliation


@pytest.mark.parametrize(
    ("text", "city", "country"),
    [
        ("Department of Chemistry, Imperial College London, London SW7 2AZ, UK", "London", "UK"),
        ("Department of Psychology, University of Cambridge, Cambridge CB2 3EB, UK", "Cambridge", "UK"),
        ("School of Informatics, University of Edinburgh, Edinburgh EH8 9AB, United Kingdom",
         "Edinburgh", "United Kingdom"),
        ("Department of Physics, University of Bristol, Bristol BS8 1TL, UK", "Bristol", "UK"),
    ],
)  # fmt: skip
def test_a_uk_postcode_after_the_city_leaves_the_city(text, city, country):
    parts = parse_affiliation(text)
    assert (parts["city"], parts["country"]) == (city, country)
