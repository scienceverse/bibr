"""Alias-table additions: plural 'related works', ablation, preliminaries, etc.

Regression coverage for the section-hierarchy fix cluster (chunk 1). The
substring matcher keyed on ``\\brelated work\\b`` missed the plural
"related works" (e.g. KAN's "5 Related works"), routing it to the OOD model.
"""

from __future__ import annotations

from bibr.paper_contents import CANONICAL_SECTION_ALIASES, CanonicalSection
from bibr.structure.section_classifier import _classify_lookup, classify_header


def test_related_works_plural_in_intro_aliases():
    intro = CANONICAL_SECTION_ALIASES[CanonicalSection.INTRODUCTION]
    assert "related work" in intro  # singular already present
    assert "related works" in intro  # plural added


def test_related_works_exact_lookup_is_intro():
    section, score = _classify_lookup("related works")
    assert section == CanonicalSection.INTRODUCTION
    assert score == 1.0


def test_kan_related_works_header_classifies_intro():
    # KAN's "5 Related works" — classify_header normalizes to "related works".
    section, _score = classify_header("5 Related works")
    assert section == CanonicalSection.INTRODUCTION


def test_related_works_substring_hits_plural():
    # Word-boundary substring match must now fire on the plural form.
    section, score = _classify_lookup("related works and prior art")
    assert section == CanonicalSection.INTRODUCTION
    assert score == 0.95


def test_ablation_study_is_results():
    for header in ("ablation study", "ablation studies"):
        section, score = _classify_lookup(header)
        assert section == CanonicalSection.RESULTS, header
        assert score == 1.0


def test_preliminaries_is_intro():
    section, _score = _classify_lookup("preliminaries")
    assert section == CanonicalSection.INTRODUCTION


def test_broader_impact_is_discussion():
    section, _score = _classify_lookup("broader impact")
    assert section == CanonicalSection.DISCUSSION


def test_reproducibility_is_open_data():
    section, _score = _classify_lookup("reproducibility")
    assert section == CanonicalSection.OPEN_DATA
