"""Regression tests for CanonicalSection enum members."""

from bibr.paper_contents import CANONICAL_SECTION_ALIASES, CanonicalSection


def test_figure_member_is_named_figure():
    assert CanonicalSection.FIGURE.value == "figure"


def test_no_fig_member():
    assert not hasattr(CanonicalSection, "FIG")


def test_open_data_member_present():
    assert CanonicalSection.OPEN_DATA.value == "open_data"


def test_open_data_aliases_present():
    aliases = CANONICAL_SECTION_ALIASES.get(CanonicalSection.OPEN_DATA, [])
    assert "data availability" in aliases
    assert "open science" in aliases
    assert "code availability" in aliases
    assert "materials availability" in aliases
    assert "open practices" in aliases


def test_author_contributions_member_and_aliases():
    assert CanonicalSection.AUTHOR_CONTRIBUTIONS.value == "author_contributions"
    aliases = CANONICAL_SECTION_ALIASES.get(CanonicalSection.AUTHOR_CONTRIBUTIONS, [])
    assert "author contributions" in aliases
    assert "credit authorship contribution statement" in aliases


def test_coi_member_and_aliases():
    assert CanonicalSection.COI.value == "coi"
    aliases = CANONICAL_SECTION_ALIASES.get(CanonicalSection.COI, [])
    assert "conflict of interest" in aliases
    assert "competing interests" in aliases
    assert "declaration of conflicting interests" in aliases


def test_ethics_member_and_aliases():
    assert CanonicalSection.ETHICS.value == "ethics"
    aliases = CANONICAL_SECTION_ALIASES.get(CanonicalSection.ETHICS, [])
    assert "ethics statement" in aliases
    assert "ethical approval" in aliases
    assert "irb approval" in aliases
    assert "informed consent" in aliases


def test_appendix_member_and_aliases():
    assert CanonicalSection.APPENDIX.value == "appendix"
    aliases = CANONICAL_SECTION_ALIASES.get(CanonicalSection.APPENDIX, [])
    for expected in (
        "appendix",
        "appendices",
        "supplementary material",
        "supplemental material",
        "supporting information",
    ):
        assert expected in aliases, expected


def test_appendix_aliases_moved_out_of_endnote():
    endnote = CANONICAL_SECTION_ALIASES.get(CanonicalSection.ENDNOTE, [])
    for moved in (
        "appendix",
        "appendices",
        "supplementary material",
        "supplemental material",
        "supporting information",
    ):
        assert moved not in endnote, f"{moved!r} should have moved to APPENDIX"
    # Conclusion/future-work-flavoured aliases stay in ENDNOTE.
    assert "conclusions and future work" in endnote
    assert "extended data" in endnote


def test_appendix_alias_lookup_routes_to_appendix():
    from bibr.structure.section_classifier import _classify_lookup

    for header in ("appendix", "appendices", "supporting information"):
        section, score = _classify_lookup(header)
        assert section == CanonicalSection.APPENDIX, header
        assert score == 1.0


def test_acknowledgment_no_longer_swallows_split_aliases():
    """COI/ethics/author_contributions aliases should not also be in ACKNOWLEDGMENT."""
    ack_aliases = CANONICAL_SECTION_ALIASES.get(CanonicalSection.ACKNOWLEDGMENT, [])
    for stranger in (
        "conflict of interest",
        "ethics statement",
        "author contributions",
        "credit authorship contribution statement",
    ):
        assert stranger not in ack_aliases, f"{stranger!r} should have moved out of ACKNOWLEDGMENT"
