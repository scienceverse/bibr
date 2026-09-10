"""Tests for CoreMetadataExtractor._validate_classification (OECD L1/L2 + paper_type)."""

from bibr.extract.core_metadata import CoreMetadataExtractor


class TestValidateClassification:
    def test_exact_l1_and_l2(self):
        pt, l1, l2 = CoreMetadataExtractor._validate_classification(
            oecd_domain="Social Sciences",
            oecd_subdomain="Psychology and Cognitive Sciences",
            paper_type="empirical",
        )
        assert pt == "empirical"
        assert l1 == "Social Sciences"
        assert l2 == "Psychology and Cognitive Sciences"

    def test_near_miss_l2_still_fills(self):
        # Regression: L2 previously only did exact/case-insensitive matching,
        # so a near-miss LLM answer like bare "Psychology" was silently
        # discarded, driving the observed oecd_l2 fill rate to ~2%.
        pt, l1, l2 = CoreMetadataExtractor._validate_classification(
            oecd_domain="Social Sciences",
            oecd_subdomain="Psychology",
            paper_type="empirical",
        )
        assert l1 == "Social Sciences"
        assert l2 == "Psychology and Cognitive Sciences"

    def test_near_miss_l2_singular_science(self):
        pt, l1, l2 = CoreMetadataExtractor._validate_classification(
            oecd_domain="Social Sciences",
            oecd_subdomain="Cognitive Science",
            paper_type="empirical",
        )
        assert l2 == "Psychology and Cognitive Sciences"

    def test_invalid_l2_discarded_with_warning(self, caplog):
        with caplog.at_level("WARNING"):
            pt, l1, l2 = CoreMetadataExtractor._validate_classification(
                oecd_domain="Social Sciences",
                oecd_subdomain="Astrology",
                paper_type="empirical",
            )
        assert l1 == "Social Sciences"
        assert l2 == ""
        assert any(
            "not valid for L1" in record.message and "Astrology" in record.message
            for record in caplog.records
        )

    def test_no_l1_backfilled_from_subdomain(self):
        # Cross-L1 rescue: with no domain, the subdomain's canonical parent
        # backfills L1. (Previously both were discarded → ("", "").)
        pt, l1, l2 = CoreMetadataExtractor._validate_classification(
            oecd_domain=None,
            oecd_subdomain="Psychology",
            paper_type=None,
        )
        assert l1 == "Social Sciences"
        assert l2 == "Psychology and Cognitive Sciences"

    def test_cross_l1_flip_computer_science(self):
        # Motivating case: OECD files Computer Science under Natural Sciences,
        # so a model saying "Engineering and Technology" is corrected.
        pt, l1, l2 = CoreMetadataExtractor._validate_classification(
            oecd_domain="Engineering and Technology",
            oecd_subdomain="Computer Science",
            paper_type="empirical",
        )
        assert (l1, l2) == ("Natural Sciences", "Computer and Information Sciences")

    def test_cross_l1_flip_logs_info(self, caplog):
        with caplog.at_level("INFO"):
            CoreMetadataExtractor._validate_classification(
                oecd_domain="Engineering and Technology",
                oecd_subdomain="Computer Science",
                paper_type=None,
            )
        assert any(
            "reassigned OECD L1" in r.message and "Natural Sciences" in r.message
            for r in caplog.records
        )

    def test_within_l1_valid_not_flipped(self, caplog):
        with caplog.at_level("INFO"):
            pt, l1, l2 = CoreMetadataExtractor._validate_classification(
                oecd_domain="Social Sciences",
                oecd_subdomain="Psychology",
                paper_type=None,
            )
        assert (l1, l2) == ("Social Sciences", "Psychology and Cognitive Sciences")
        assert not any("reassigned OECD L1" in r.message for r in caplog.records)

    def test_no_subdomain(self):
        pt, l1, l2 = CoreMetadataExtractor._validate_classification(
            oecd_domain="Social Sciences",
            oecd_subdomain=None,
            paper_type=None,
        )
        assert l1 == "Social Sciences"
        assert l2 == ""

    def test_unknown_paper_type_discarded(self):
        pt, l1, l2 = CoreMetadataExtractor._validate_classification(
            oecd_domain=None,
            oecd_subdomain=None,
            paper_type="not-a-real-type",
        )
        assert pt == ""
