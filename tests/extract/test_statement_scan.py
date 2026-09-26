"""Lexical-anchor statement fallback: when copy_integrity_statements finds no
FUNDING/COI/ETHICS/OPEN_DATA section, scan the body for verbatim statements
that live inside mis-typed sections (run-in bold labels, method-embedded ethics,
acknowledgment-embedded funding).
"""

import pytest

from bibr.extract.statement_scan import lexical_fallback_warning, scan_statements_fallback
from bibr.models import PaperAuthor, PaperMetadata
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.processing_warnings import WarningCode


def _contents(sections_spec: list[tuple[int, str, CanonicalSection, list[str]]]) -> PaperContents:
    """Build PaperContents from ``(section_id, header, type, [sentences])`` tuples."""
    sections = [PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)]
    sentences = []
    tid = 1
    for section_id, header, section_type, sents in sections_spec:
        sections.append(
            PaperSection(
                section_id=section_id,
                header=header,
                level=1,
                parent_section_id=0,
                section_type=section_type,
            )
        )
        for s in sents:
            sentences.append(
                PaperSentence(text_id=tid, text=s, section_id=section_id, paragraph_id=tid)
            )
            tid += 1
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
    )


def _paper_authors(*names: tuple[str, str]) -> list[PaperAuthor]:
    return [
        PaperAuthor(
            author_id=index,
            given=given,
            family=family,
            affiliation="",
        )
        for index, (given, family) in enumerate(names, start=1)
    ]


class TestKeldersShape:
    """Three labeled statements inline in ONE discussion-typed section — the
    verbatim copy finds nothing (section_type is DISCUSSION), so all three must
    be recovered by the lexical scan, each capture bounded to its own sentence."""

    def _kelders(self) -> PaperContents:
        return _contents(
            [
                (
                    1,
                    "Conclusions",
                    CanonicalSection.DISCUSSION,
                    [
                        "In conclusion, the intervention improved adherence.",
                        "Declaration of conflicting interests: The author(s) declared no "
                        "potential conflicts of interest with respect to the research, "
                        "authorship, and/or publication of this article.",
                        "Funding: The author(s) received financial support for the research "
                        "from the University of Twente grant.",
                        "Ethical approval: All procedures were approved by the ethics "
                        "committee of the University of Twente.",
                    ],
                )
            ]
        )

    def test_all_three_statements_populated(self):
        contents = self._kelders()
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.coi_statement is not None
        assert "conflicts of interest" in metadata.coi_statement
        assert metadata.funding_statement is not None
        assert "financial support" in metadata.funding_statement
        assert metadata.ethics_statement is not None
        assert "ethics committee" in metadata.ethics_statement
        assert metadata.data_availability is None

    def test_capture_is_bounded_to_own_sentence(self):
        # A different-type anchor in the next sentence stops the capture: the
        # COI statement must not swallow the following Funding sentence.
        contents = self._kelders()
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert "financial support" not in metadata.coi_statement
        assert "ethics committee" not in metadata.funding_statement

    def test_warning_emitted_per_filled_field(self):
        contents = self._kelders()
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        warnings = contents.processing_warnings
        assert all(w.code == WarningCode.STATEMENT_LEXICAL_FALLBACK for w in warnings)
        assert lexical_fallback_warning("coi_statement") in warnings
        assert lexical_fallback_warning("funding_statement") in warnings
        assert lexical_fallback_warning("ethics_statement") in warnings

    def test_already_populated_field_untouched(self):
        contents = self._kelders()
        metadata = PaperMetadata(doi="", title="T", coi_statement="pre-existing")
        scan_statements_fallback(contents, metadata)
        assert metadata.coi_statement == "pre-existing"
        assert lexical_fallback_warning("coi_statement") not in contents.processing_warnings


class TestEyecolorShape:
    """Ethics buried in a METHODS section, funding buried in an ACKNOWLEDGMENT
    section — both recovered by the scan."""

    def test_ethics_from_method_and_funding_from_acknowledgment(self):
        contents = _contents(
            [
                (
                    1,
                    "Methods",
                    CanonicalSection.METHODS,
                    [
                        "Participants completed the survey.",
                        "The study was approved by the ethics committee and all "
                        "participants gave informed consent.",
                    ],
                ),
                (
                    2,
                    "Acknowledgments",
                    CanonicalSection.ACKNOWLEDGMENT,
                    [
                        "We thank the participants.",
                        "This work was supported by ERC grant #647910 KINSHIP.",
                    ],
                ),
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.ethics_statement is not None
        assert "ethics committee" in metadata.ethics_statement
        assert metadata.funding_statement is not None
        assert "ERC grant #647910" in metadata.funding_statement


class TestPrecisionGuards:
    def test_funding_requires_funder_hint(self):
        # "supported by strong evidence" is not a funding statement — no
        # funder-ish token nearby, so funding stays None.
        contents = _contents(
            [
                (
                    1,
                    "Introduction",
                    CanonicalSection.INTRODUCTION,
                    ["Previous work was supported by strong evidence."],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement is None

    def test_anchor_inside_references_ignored(self):
        # An anchor phrase appearing inside the references section must not be
        # captured as a statement.
        contents = _contents(
            [
                (
                    1,
                    "References",
                    CanonicalSection.REFERENCES,
                    [
                        "Smith, J. (2020). This work was supported by the NSF grant "
                        "12345. Journal of Things.",
                    ],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement is None

    def test_no_anchors_leaves_all_none(self):
        contents = _contents(
            [(1, "Introduction", CanonicalSection.INTRODUCTION, ["A plain sentence."])]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement is None
        assert metadata.coi_statement is None
        assert metadata.ethics_statement is None
        assert metadata.data_availability is None
        assert contents.processing_warnings == []

    def test_hrv_supported_by_not_funding(self):
        # Ambiguous "supported by" anchor with no funder cue AFTER the anchor
        # (the acronym "HRV" precedes it) must not be captured as funding.
        contents = _contents(
            [
                (
                    1,
                    "Discussion",
                    CanonicalSection.DISCUSSION,
                    [
                        "That HRV might be related to this neural circuitry, associated "
                        "with perceptions of threat and safety, would have important "
                        "implications for HRV as an index of stress and resilience if "
                        "supported by empirical data.",
                    ],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement is None

    def test_supported_by_with_funder_after_anchor_captured(self):
        # Ambiguous "supported by" anchor followed by a real funder cue must
        # still be captured.
        contents = _contents(
            [
                (
                    1,
                    "Acknowledgments",
                    CanonicalSection.ACKNOWLEDGMENT,
                    ["This research was supported by the Wellcome Trust."],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement is not None
        assert "Wellcome Trust" in metadata.funding_statement

    def test_data_availability_recovered(self):
        contents = _contents(
            [
                (
                    1,
                    "Discussion",
                    CanonicalSection.DISCUSSION,
                    ["Data availability: The data are openly available on OSF."],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.data_availability is not None
        assert "openly available" in metadata.data_availability

    @pytest.mark.parametrize(
        "text",
        [
            "Datasets are increasingly available to researchers, creating new ethical questions.",
            "Datasets are publicly available to researchers across disciplines.",
            "Data are available in many scientific fields, changing research practice.",
            "Code is freely available on modern platforms for teaching.",
        ],
    )
    def test_topical_dataset_availability_is_not_recovered(self, text: str):
        contents = _contents(
            [
                (
                    1,
                    "Discussion",
                    CanonicalSection.DISCUSSION,
                    [text],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.data_availability is None

    def test_topical_subject_receiving_funding_is_not_recovered(self):
        text = (
            "Political parties received funding from the National Science Foundation "
            "during the election."
        )
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement is None

    @pytest.mark.parametrize(
        ("text", "authors"),
        [
            ("J.W. was supported by NSF grant 123.", _paper_authors(("Jane", "Werner"))),
            (
                "Jane Doe received funding from NIH grant R01-MH123.",
                _paper_authors(("Jane", "Doe")),
            ),
            (
                "A.B. and C.D. were supported by NIH grant R01-MH123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
            ),
            (
                "Jane W. Doe was supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe")),
            ),
            ("The present study was funded by NSF grant 123.", []),
            (
                "Research reported in this publication was supported by NIH grant R01-MH123.",
                [],
            ),
        ],
    )
    def test_author_specific_funding_declaration_is_recovered(
        self, text: str, authors: list[PaperAuthor]
    ):
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == text

    def test_author_specific_funding_rejects_an_unlisted_name(self):
        text = "Jane Doe received funding from NIH grant R01-MH123."
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(
            doi="",
            title="T",
            authors=_paper_authors(("Alice", "Roe")),
        )

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement is None

    @pytest.mark.parametrize(
        ("text", "authors"),
        [
            (
                "A.M.D.L.C. was supported by NSF grant 123.",
                _paper_authors(("Ana Maria", "de la Cruz")),
            ),
            (
                "A.D.L.C. was supported by NSF grant 123.",
                _paper_authors(("Ana Maria", "de la Cruz")),
            ),
            (
                "J.W.D.J. was supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe Jr.")),
            ),
            (
                "J.D.J. was supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe Jr.")),
            ),
        ],
    )
    def test_author_initial_aliases_include_particles_suffixes_and_short_given_form(
        self, text: str, authors: list[PaperAuthor]
    ):
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == text

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            ([], None),
            (_paper_authors(("Alice", "Roe")), None),
            (
                _paper_authors(("Jane", "Doe")),
                "Funding: Jane Doe was funded by NSF grant 123.",
            ),
        ],
    )
    def test_labeled_named_funding_still_requires_a_grounded_author(
        self, authors: list[PaperAuthor], expected: str | None
    ):
        text = "Funding: Jane Doe was funded by NSF grant 123."
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            ([], None),
            (_paper_authors(("Alice", "Roe")), None),
            (_paper_authors(("Jane", "Doe")), "funded by NSF grant 123."),
        ],
    )
    def test_category_clipping_cannot_erase_named_funding_grounding(
        self, authors: list[PaperAuthor], expected: str | None
    ):
        text = "Ethics approval was obtained; Jane Doe was funded by NSF grant 123."
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "authors"),
        [
            (
                "Jane W. Doe, Jr. was supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe, Jr.")),
            ),
            (
                "Jane W. Doe, Jr. and Alice Brown were supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe, Jr."), ("Alice", "Brown")),
            ),
        ],
    )
    def test_grounded_comma_suffixes_are_not_mistaken_for_author_lists(
        self, text: str, authors: list[PaperAuthor]
    ):
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == text

    @pytest.mark.parametrize(
        ("text", "authors", "expected"),
        [
            (
                "Jane W. Doe, Jr. was supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe Jr.")),
                "Jane W. Doe, Jr. was supported by NSF grant 123.",
            ),
            (
                "Jane W. Doe Jr. was supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe, Jr.")),
                "Jane W. Doe Jr. was supported by NSF grant 123.",
            ),
            (
                "Jane W. Doe, Jr., Alice Brown were supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe, Jr."), ("Alice", "Brown")),
                "Jane W. Doe, Jr., Alice Brown were supported by NSF grant 123.",
            ),
            (
                "Jane W. Doe, Jr., Alice Brown were supported by NSF grant 123.",
                _paper_authors(("Jane W.", "Doe, Jr.")),
                None,
            ),
            (
                "i\u0307pek Doe was supported by NSF grant 123.",
                _paper_authors(("İpek", "Doe")),
                "i\u0307pek Doe was supported by NSF grant 123.",
            ),
        ],
    )
    def test_external_review_suffix_punctuation_and_list_partitioning(
        self,
        text: str,
        authors: list[PaperAuthor],
        expected: str | None,
    ):
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "authors", "expected"),
        [
            (
                "Jane and Doe were supported by NSF grant 123.",
                _paper_authors(("Jane", "Doe")),
                None,
            ),
            (
                "A. and B. were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown")),
                None,
            ),
            (
                "Jane Doe and Alice Brown were supported by NSF grant 123.",
                _paper_authors(("Jane", "Doe"), ("Alice", "Brown")),
                "Jane Doe and Alice Brown were supported by NSF grant 123.",
            ),
            (
                "Jane Doe, Alice Brown were supported by NSF grant 123.",
                _paper_authors(("Jane", "Doe"), ("Alice", "Brown")),
                "Jane Doe, Alice Brown were supported by NSF grant 123.",
            ),
            (
                "Jane Doe and Alice Brown were supported by NSF grant 123.",
                _paper_authors(("Jane", "Doe")),
                None,
            ),
        ],
    )
    def test_author_list_delimiters_cannot_be_merged_into_one_alias(
        self,
        text: str,
        authors: list[PaperAuthor],
        expected: str | None,
    ):
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "authors", "expected"),
        [
            (
                "Abel was supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Edward", "Lee")),
                None,
            ),
            (
                "ABCD was supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                None,
            ),
            (
                "Alice BrownCarol Doe were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                None,
            ),
            (
                "Alice Brown, Carol Doe were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                "Alice Brown, Carol Doe were supported by NSF grant 123.",
            ),
            (
                "Alice Brown and Carol Doe were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                "Alice Brown and Carol Doe were supported by NSF grant 123.",
            ),
            (
                "Alice Brown & Carol Doe were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                "Alice Brown & Carol Doe were supported by NSF grant 123.",
            ),
            (
                "A.B., C.D. were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                "A.B., C.D. were supported by NSF grant 123.",
            ),
            (
                "AB & CD were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                "AB & CD were supported by NSF grant 123.",
            ),
            (
                "Alice Brown, Eve Fox were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown"), ("Carol", "Doe")),
                None,
            ),
        ],
    )
    def test_external_review_author_aliases_require_raw_list_delimiters(
        self,
        text: str,
        authors: list[PaperAuthor],
        expected: str | None,
    ):
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "authors", "expected"),
        [
            (
                "Funding: Jane Doe acknowledges financial support from NSF.",
                _paper_authors(("Alice", "Brown")),
                None,
            ),
            (
                "Funding: Jane Doe acknowledges financial support from NSF.",
                _paper_authors(("Jane", "Doe")),
                "Funding: Jane Doe acknowledges financial support from NSF.",
            ),
            (
                "Funding: The authors acknowledge financial support from NSF.",
                [],
                "Funding: The authors acknowledge financial support from NSF.",
            ),
        ],
    )
    def test_labeled_funding_acknowledgments_obey_subject_grounding(
        self,
        text: str,
        authors: list[PaperAuthor],
        expected: str | None,
    ):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Funding: NSF grant 123.", "Funding: NSF grant 123."),
            ("Funding: Jane Doe thanks NSF.", None),
        ],
    )
    def test_labeled_funder_only_fallback_is_compact_and_not_person_led(
        self, text: str, expected: str | None
    ):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            (_paper_authors(("Alice", "Roe")), None),
            (
                _paper_authors(("Jane", "Doe")),
                "Jane Doe — supported by NSF grant 123.",
            ),
        ],
    )
    def test_dash_cannot_detach_a_named_subject_from_its_funding_action(
        self, authors: list[PaperAuthor], expected: str | None
    ):
        text = "Jane Doe — supported by NSF grant 123."
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "authors", "expected"),
        [
            (
                "Funding: Jane Doe received no funding.",
                _paper_authors(("Alice", "Roe")),
                None,
            ),
            (
                "Funding: Jane Doe received no funding.",
                _paper_authors(("Jane", "Doe")),
                "Funding: Jane Doe received no funding.",
            ),
            (
                "Funding: Jane Doe had no external funding.",
                _paper_authors(("Alice", "Roe")),
                None,
            ),
            (
                "Funding: Jane Doe has no funding.",
                _paper_authors(("Jane", "Doe")),
                "Funding: Jane Doe has no funding.",
            ),
            (
                "Funding: This study received no funding.",
                [],
                "Funding: This study received no funding.",
            ),
            (
                "Funding: This study had no external funding.",
                [],
                "Funding: This study had no external funding.",
            ),
            (
                "Funding: No funding was received.",
                [],
                "Funding: No funding was received.",
            ),
        ],
    )
    def test_negative_named_funding_requires_grounding_but_generic_negatives_do_not(
        self,
        text: str,
        authors: list[PaperAuthor],
        expected: str | None,
    ):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "authors", "expected"),
        [
            (
                "Jane Doe; supported by NSF grant 123.",
                _paper_authors(("Alice", "Roe")),
                None,
            ),
            (
                "Funding: Jane Doe; supported by NSF grant 123.",
                _paper_authors(("Alice", "Roe")),
                None,
            ),
            ("Public Policy; funded by NSF grant 123.", [], None),
            (
                "J.D.; A.B. were supported by NSF grant 123.",
                _paper_authors(("Alice", "Brown")),
                None,
            ),
            (
                "Jane Doe; Alice Brown were supported by NSF grant 123.",
                _paper_authors(("Jane", "Doe"), ("Alice", "Brown")),
                "Jane Doe; Alice Brown were supported by NSF grant 123.",
            ),
            (
                "Ethics approval was obtained; This work was funded by NSF grant 123.",
                [],
                "funded by NSF grant 123.",
            ),
        ],
    )
    def test_semicolon_cannot_turn_a_named_subject_into_a_subjectless_fragment(
        self,
        text: str,
        authors: list[PaperAuthor],
        expected: str | None,
    ):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "authors", "expected"),
        [
            (
                "Alice Roe was supported by NSF grant 123; "
                "This work was funded by NIH grant R01-MH123.",
                _paper_authors(("Jane", "Doe")),
                None,
            ),
            (
                "Jane Doe was supported by NSF grant 123; "
                "Alice Roe was funded by NIH grant R01-MH123.",
                _paper_authors(("Jane", "Doe")),
                None,
            ),
            (
                "This work was funded by NSF grant 123; "
                "The study was supported by NIH grant R01-MH123.",
                _paper_authors(("Jane", "Doe")),
                "This work was funded by NSF grant 123; "
                "The study was supported by NIH grant R01-MH123.",
            ),
            (
                "Jane Doe was supported by NSF grant 123; "
                "Alice Roe was funded by NIH grant R01-MH123.",
                _paper_authors(("Jane", "Doe"), ("Alice", "Roe")),
                "Jane Doe was supported by NSF grant 123; "
                "Alice Roe was funded by NIH grant R01-MH123.",
            ),
            (
                "This work was funded by NSF grant 123; "
                "Competing interests: Alice Roe received funding from NIH.",
                _paper_authors(("Jane", "Doe")),
                "This work was funded by NSF grant 123;",
            ),
        ],
    )
    def test_external_review_all_funding_clauses_obey_named_grounding(
        self,
        text: str,
        authors: list[PaperAuthor],
        expected: str | None,
    ):
        contents = _contents([(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT, [text])])
        metadata = PaperMetadata(doi="", title="T", authors=authors)

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == expected

    @pytest.mark.parametrize(
        ("text", "field", "expected"),
        [
            (
                "Policy was funded by NSF after public consultation.",
                "funding_statement",
                None,
            ),
            (
                "Public Policy was supported by the Wellcome Trust in several countries.",
                "funding_statement",
                None,
            ),
            (
                "Government Research was funded by NSF after public consultation.",
                "funding_statement",
                None,
            ),
            (
                "A waiver was granted by the ethics committee.",
                "ethics_statement",
                "A waiver was granted by the ethics committee.",
            ),
        ],
    )
    def test_external_review_capitalized_subject_and_waiver_regressions(
        self, text: str, field: str, expected: str | None
    ):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert getattr(metadata, field) == expected

    def test_topical_waiver_question_is_not_recovered(self):
        text = "The analysis examined whether a waiver was granted by the ethics committee."
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.ethics_statement is None

    @pytest.mark.parametrize(
        "text",
        [
            "This study examines projects funded by the National Science Foundation.",
            "Our research analyzes interventions supported by NIH grants.",
            "The study compares investigators who received funding from NSF.",
            "We surveyed organizations supported by the Wellcome Trust.",
        ],
    )
    def test_paper_subject_topical_funding_mentions_are_not_recovered(self, text: str):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement is None

    @pytest.mark.parametrize(
        "text",
        [
            "Raw data are available from the corresponding author on reasonable request.",
            "Anonymized data are available in the Zenodo repository.",
        ],
    )
    def test_modified_data_subject_declaration_is_recovered(self, text: str):
        contents = _contents([(1, "Declarations", CanonicalSection.ENDNOTE, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.data_availability == text

    @pytest.mark.parametrize(
        "text",
        [
            "The data show that materials used in other studies are available from the "
            "Zenodo repository.",
            "Our data indicate that code produced by competitors is available on GitHub.",
            "Data reveal that datasets generated by prior surveys are available upon request.",
        ],
    )
    def test_nested_topical_data_availability_mentions_are_not_recovered(self, text: str):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.data_availability is None

    @pytest.mark.parametrize(
        "text",
        [
            "This study examines whether procedures were approved by an ethics committee.",
            "Our research discusses informed consent as a legal concept.",
            "All participants debated informed consent during the workshop.",
        ],
    )
    def test_topical_ethics_mentions_are_not_recovered(self, text: str):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.ethics_statement is None

    @pytest.mark.parametrize(
        "text",
        [
            "This study examines authors who declare no conflicts of interest.",
            "Participants reported no conflicts of interest during interviews.",
            "The model assumes no competing interests among firms.",
            "We found no conflict of interest between the two measures.",
        ],
    )
    def test_topical_conflict_mentions_are_not_recovered(self, text: str):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.coi_statement is None

    def test_generated_dataset_available_from_repository_is_recovered(self):
        text = "The datasets generated during this study are available from the repository."
        contents = _contents([(1, "Declarations", CanonicalSection.ENDNOTE, [text])])
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.data_availability == text

    def test_bare_ethics_label_does_not_recover_topical_not_applicable_prose(self):
        contents = _contents(
            [
                (
                    1,
                    "Discussion",
                    CanonicalSection.DISCUSSION,
                    [
                        "Ethics",
                        "Not applicable to the scope of this philosophical discussion.",
                    ],
                )
            ]
        )
        contents.sentences[1].paragraph_id = contents.sentences[0].paragraph_id
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.ethics_statement is None

    def test_bare_funding_label_recovers_compact_negative_declaration(self):
        contents = _contents(
            [
                (
                    1,
                    "Declarations",
                    CanonicalSection.ENDNOTE,
                    ["Funding", "None."],
                )
            ]
        )
        contents.sentences[1].paragraph_id = contents.sentences[0].paragraph_id
        metadata = PaperMetadata(doi="", title="T")

        scan_statements_fallback(contents, metadata)

        assert metadata.funding_statement == "Funding None."

    def test_ipcp_acronym_after_supported_by_is_not_a_funder_hint(self):
        contents = _contents(
            [
                (
                    1,
                    "Discussion",
                    CanonicalSection.DISCUSSION,
                    ["The interpretation was supported by IPCP in the sensitivity analysis."],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement is None

    @pytest.mark.parametrize(
        "text",
        [
            "The intervention was supported by appropriate measures to promote IPCP.",
            "The conclusion was supported by HRV signals in the sensitivity analysis.",
            "The estimate was supported by 123 participants.",
            "The conclusion was supported by University data.",
        ],
    )
    def test_ambiguous_supported_by_adversarial_non_funders(self, text):
        contents = _contents([(1, "Discussion", CanonicalSection.DISCUSSION, [text])])
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement is None

    @pytest.mark.parametrize(
        "funder",
        ["NSF grant 123", "ERC grant 647910", "NIH award R01-MH123", "the Wellcome Trust"],
    )
    def test_supported_by_explicit_funders_remain_positive(self, funder):
        contents = _contents(
            [
                (
                    1,
                    "Acknowledgments",
                    CanonicalSection.ACKNOWLEDGMENT,
                    [f"This research was supported by {funder}."],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement == f"This research was supported by {funder}."

    def test_funding_received_from_funder_is_not_mistaken_for_received_date_boilerplate(self):
        text = "Financial support was received from the NSF."
        contents = _contents([(1, "Funding", CanonicalSection.FUNDING, [text])])
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement == text

    def test_fallback_does_not_continue_into_a_different_paragraph(self):
        contents = _contents(
            [
                (
                    1,
                    "Acknowledgments",
                    CanonicalSection.ACKNOWLEDGMENT,
                    [
                        "This work was supported by NSF grant 123.",
                        "The remainder of the article discusses unrelated results.",
                    ],
                )
            ]
        )
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.funding_statement == "This work was supported by NSF grant 123."

    def test_fallback_stops_at_publisher_license_boilerplate(self):
        contents = _contents(
            [
                (
                    1,
                    "Back matter",
                    CanonicalSection.ENDNOTE,
                    [
                        "Data availability: Data are available on OSF.",
                        "Published under a Creative Commons license by Example Publisher.",
                    ],
                )
            ]
        )
        # Put both sentences in one paragraph to prove the boilerplate boundary,
        # rather than the paragraph boundary, stops continuation.
        contents.sentences[1].paragraph_id = contents.sentences[0].paragraph_id
        metadata = PaperMetadata(doi="", title="T")
        scan_statements_fallback(contents, metadata)
        assert metadata.data_availability == "Data availability: Data are available on OSF."


def test_fallback_continues_within_the_same_paragraph():
    contents = _contents(
        [
            (
                1,
                "Acknowledgments",
                CanonicalSection.ACKNOWLEDGMENT,
                [
                    "This work was supported by NSF grant 123.",
                    "Additional support came from the Wellcome Trust.",
                ],
            )
        ]
    )
    contents.sentences[1].paragraph_id = contents.sentences[0].paragraph_id
    metadata = PaperMetadata(doi="", title="T")
    scan_statements_fallback(contents, metadata)
    assert metadata.funding_statement == (
        "This work was supported by NSF grant 123. Additional support came from the Wellcome Trust."
    )


def test_fallback_stops_when_another_category_starts_in_the_same_sentence():
    contents = _contents(
        [
            (
                1,
                "Declarations",
                CanonicalSection.ENDNOTE,
                [
                    "Funding: This work was supported by NSF grant 123. "
                    "Competing interests: The authors declare none."
                ],
            )
        ]
    )
    metadata = PaperMetadata(doi="", title="T")
    scan_statements_fallback(contents, metadata)
    assert metadata.funding_statement == "Funding: This work was supported by NSF grant 123."
    assert metadata.coi_statement == "Competing interests: The authors declare none."


def test_boundary_exemptions_stay_linear_on_a_run_on_paragraph():
    """Each licence boundary checks only the text just before it."""

    import time

    from bibr.extract.statement_scan import _boilerplate_boundary_for_field

    text = "Data are available under the license " * 2_700
    started = time.perf_counter()

    assert _boilerplate_boundary_for_field("data_availability", text) is None
    assert time.perf_counter() - started < 2.0
