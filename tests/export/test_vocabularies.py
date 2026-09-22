"""The export schema's closed vocabularies stay in lock-step with their sources.

``bibr/export/models.py`` spells each enum out as a ``Literal`` (it cannot
unpack a runtime list and stays import-light for readers), so these tests pin
every one to the runtime value set bibr actually produces from.
"""

from typing import get_args

from bibr.export import models


def test_section_types_match_canonical_section():
    from bibr.paper_contents import CanonicalSection

    assert list(get_args(models.SectionTypeLiteral)) == [m.value for m in CanonicalSection]


def test_bib_types_match_bib_type():
    from bibr.models import BibType

    assert list(get_args(models.BibTypeLiteral)) == [m.value for m in BibType]


def test_paper_types_cover_the_labels_and_the_unknown_fallback():
    from bibr.structure.paper_classifier import PAPER_TYPE_LABELS, PaperType

    assert set(get_args(models.PaperTypeLiteral)) == set(PAPER_TYPE_LABELS) | {
        m.value for m in PaperType
    }


def test_oecd_labels_match_the_classifier_taxonomy():
    from bibr.structure.paper_classifier import ALL_OECD_L2_LABELS, OECD_L1_LABELS

    assert list(get_args(models.OecdL1Literal)) == OECD_L1_LABELS
    assert list(get_args(models.OecdL2Literal)) == ALL_OECD_L2_LABELS


def test_match_services_match_match_source():
    from bibr.models import MatchSource

    assert list(get_args(models.MatchServiceLiteral)) == [m.value for m in MatchSource]


def test_input_formats_are_the_supported_types_plus_unknown():
    from bibr.input.supported_files import SupportedFileType

    expected = [m.value.lstrip(".") for m in SupportedFileType] + ["unknown"]
    assert list(get_args(models.InputFormatLiteral)) == expected


def test_severities_match_issue_severity():
    from bibr.validation import IssueSeverity

    assert list(get_args(models.SeverityLiteral)) == [m.value for m in IssueSeverity]
