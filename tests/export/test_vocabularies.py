"""The export schema's closed vocabularies stay in lock-step with their sources.

``bibr/export/models.py`` spells each enum out as a ``Literal`` (it cannot
unpack a runtime list and stays import-light for readers), so these tests pin
every one to the runtime value set bibr actually produces from, through the
exporter's mapping where the published spelling differs (``open_data``,
``meta-analysis``, ``.htm``).
"""

from typing import get_args

from bibr.export import json_export, models


def test_section_types_match_canonical_section():
    from bibr.paper_contents import CanonicalSection

    exported = [json_export._EXPORT_SECTION_TYPES.get(m.value, m.value) for m in CanonicalSection]
    assert list(get_args(models.SectionTypeLiteral)) == exported


def test_bib_types_match_bib_type():
    from bibr.models import BibType

    assert list(get_args(models.BibTypeLiteral)) == [m.value for m in BibType]


def test_paper_types_cover_the_labels_and_the_unknown_fallback():
    from bibr.structure.paper_classifier import PAPER_TYPE_LABELS, PaperType

    labels = set(PAPER_TYPE_LABELS) | {m.value for m in PaperType}
    assert set(get_args(models.PaperTypeLiteral)) == {json_export._snake(v) for v in labels}


def test_oecd_labels_match_the_classifier_taxonomy():
    from bibr.structure.paper_classifier import ALL_OECD_L2_LABELS, OECD_L1_LABELS

    assert list(get_args(models.OecdL1Literal)) == OECD_L1_LABELS
    assert list(get_args(models.OecdL2Literal)) == ALL_OECD_L2_LABELS


def test_match_services_match_match_source():
    from bibr.models import MatchSource

    assert list(get_args(models.MatchServiceLiteral)) == [m.value for m in MatchSource]


def test_input_formats_are_the_supported_types_plus_unknown():
    from bibr.input.supported_files import SupportedFileType

    exported = {json_export._export_input_format(m.value.lstrip(".")) for m in SupportedFileType}
    # Every file type bibr reads has its format, and ``tei`` is the one format
    # only a converter from GROBID writes.
    assert exported | {"tei", "unknown"} == set(get_args(models.InputFormatLiteral))
    assert "unknown" not in exported


def test_xref_tiers_match_the_citation_linker():
    import re
    from pathlib import Path

    import bibr.structure.citation_linker as linker

    # A bib xref's tier is its candidate's ``style`` (or ``llm``).
    source = Path(linker.__file__).read_text()
    tiers = set(re.findall(r'(?:style|tier)="([a-z-]+)"', source))
    assert {"numeric", "paren-numeric", "flattened-superscript", "llm"} <= tiers
    assert {json_export._snake(t) for t in tiers} <= set(get_args(models.XrefTierLiteral))


def test_xref_tiers_cover_figure_and_table_resolution():
    from bibr.structure import xref_utils

    tiers = {xref_utils.LABEL_TIER, xref_utils.POSITION_TIER}
    assert {json_export._snake(t) for t in tiers} <= set(get_args(models.XrefTierLiteral))


def test_eq_comparators_match_the_equation_extractor():
    import re

    from bibr.extract.equation_extractor import _COMP_PATTERN, _normalize_comp

    spellings = re.findall(r"[^(?:|)]+", _COMP_PATTERN)
    produced = {_normalize_comp(spelling) for spelling in spellings}
    assert produced == set(get_args(models.EqCompLiteral))


def test_severities_match_issue_severity():
    from bibr.validation import IssueSeverity

    assert list(get_args(models.SeverityLiteral)) == [m.value for m in IssueSeverity]
