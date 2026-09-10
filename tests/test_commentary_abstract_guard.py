"""Compatibility tests for retired commentary abstract suppression.

Residual #1: on an abstract-less Commentary (e.g. Köster `09567976211001317`)
the OCR layout model mislabels the body-opening column as an `abstract` region,
and the concurrent title/keywords LLM call — which never sees paper_type —
copies that body text into info.abstract. Length is only a suspicion signal:
without source ownership it must not destructively remove the extracted value.
"""

import pytest

from bibr.extract.extractor import MetadataExtractor


class TestShouldSuppressCommentaryAbstract:
    """The retained compatibility helper never authorizes deletion."""

    # Lengths/keywords taken from the gold-validated corpus (38 commentaries,
    # 3 corpora): no-keywords genuine abstracts cap at 1563 chars; the two known
    # psych fabrications are 2787 and 2920 chars with no keywords.
    LONG_BODY = "In our current efforts to understand brain activity. " * 56  # ~2912 chars

    @pytest.mark.parametrize(
        "paper_type, abstract, keywords, expected",
        [
            # Long commentary text is warning-only without source provenance.
            ("commentary", "x" * 2920, [], False),
            ("commentary", "x" * 2787, [], False),
            # KEEP: genuine short commentary abstract (no keywords).
            ("commentary", "x" * 943, [], False),  # 0956797614566469
            ("commentary", "x" * 1563, [], False),  # longest no-keywords genuine (MDPI)
            # KEEP: commentary with a keywords block — even when long (6502 outlier).
            ("commentary", "x" * 992, ["a", "b"], False),  # Aknin 09567976211052476
            ("commentary", "x" * 6502, ["morality", "trolley"], False),
            # KEEP: not a commentary, regardless of length / keywords.
            ("empirical", "x" * 5000, [], False),
            ("review", "x" * 5000, [], False),
            # Edge: empty/short inputs never suppress.
            ("commentary", "", [], False),
            ("commentary", None, [], False),
        ],
    )
    def test_suppression_decision(self, paper_type, abstract, keywords, expected):
        assert (
            MetadataExtractor._should_suppress_commentary_abstract(paper_type, abstract, keywords)
            is expected
        )
