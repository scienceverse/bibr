"""Pointer answers turned back into verbatim fields (LLM_POINTER_OUTPUTS).

The title/keywords call names the abstract's first and last words and bibr
copies the abstract out of the text the model was shown; the authors call lists
each affiliation once and numbers it.
"""

import logging

import pytest

from bibr.clients.pointer_outputs import (
    MAX_ABSTRACT_CHARS,
    abstract_text,
    expand_authors,
    locate_abstract,
    resolve_title_keywords,
)
from bibr.schemas import AuthorsPointerLLM, TitleKeywordsPointerLLM

_FRONT = (
    "Journal of Examples 2024, Vol. 3(1) 1-9\n"
    "A Study of Things\n"
    "Jane Doe, John Smith\n"
    "Abstract\n"
    "“Background” matters: we examined how the coordination of care for people\r\n"
    "affected by rare diseases is organ-\r\n"
    "ised across 52 clinics. Results were mixed. This study highlights ways\r\n"
    "of improving the coordination of care for people affected by rare diseases.\n"
    "Keywords: care; rare diseases\n"
    "1 Introduction\n"
    "Rare diseases are rare.\n"
)
_ABSTRACT_FIRST = "“Background” matters: we examined how the coordination"
_ABSTRACT_LAST = "the coordination of care for people affected by rare diseases."


def _copy(source: str, first: str, last: str) -> str | None:
    span = locate_abstract(source, first, last)
    return None if span is None else abstract_text(source, span)


class TestLocateAbstract:
    def test_copies_the_text_between_the_anchors(self):
        copied = _copy(_FRONT, _ABSTRACT_FIRST, _ABSTRACT_LAST)
        assert copied is not None
        assert copied.startswith("“Background” matters: we examined")
        assert copied.endswith("people affected by rare diseases.")
        assert "Keywords" not in copied

    def test_a_tidied_anchor_still_places_the_abstract(self):
        """Models straighten quotes, change case, rejoin a word broken at a
        line end and drop a final period when they quote."""
        tidy = locate_abstract(
            _FRONT,
            '"background" matters: we examined how the',
            "is organised across 52 clinics",
        )
        exact = locate_abstract(_FRONT, _ABSTRACT_FIRST, "is organ-\r\nised across 52 clinics.")
        assert tidy is not None and exact is not None
        assert (tidy.start, tidy.fuzzy) == (exact.start, False)
        assert _FRONT[tidy.start : tidy.end].endswith("52 clinics.")

    def test_a_closing_phrase_that_recurs_ends_at_the_abstract_end(self):
        """The last eight words also close a clause of the first sentence; the
        copy runs to the occurrence that ends with the anchor's own punctuation
        and a line break."""
        copied = _copy(_FRONT, _ABSTRACT_FIRST, "for people affected by rare diseases.")
        assert copied is not None
        assert "Results were mixed." in copied
        assert copied.endswith("people affected by rare diseases.")

    def test_an_ocr_typo_in_the_source_is_matched_fuzzily_to_word_edges(self):
        source = _FRONT.replace("we examined how", "we exarnined how")
        span = locate_abstract(source, _ABSTRACT_FIRST, _ABSTRACT_LAST)
        assert span is not None and span.fuzzy
        assert source[span.start :].startswith("“Background”")

    def test_unplaced_or_too_short_anchors_give_none(self):
        assert locate_abstract(_FRONT, "nothing like this is printed here", _ABSTRACT_LAST) is None
        assert locate_abstract(_FRONT, "A", _ABSTRACT_LAST) is None
        assert locate_abstract(_FRONT, _ABSTRACT_FIRST, "") is None

    def test_a_closing_before_the_opening_does_not_count(self):
        assert locate_abstract(_FRONT, "Rare diseases are rare.", "A Study of Things") is None

    def test_a_span_longer_than_any_abstract_is_rejected(self):
        filler = "word " * (MAX_ABSTRACT_CHARS // 5)
        source = f"Start of the abstract here now.\n{filler}\nEnd of the abstract here now."
        assert (
            locate_abstract(source, "Start of the abstract here", "End of the abstract here")
            is None
        )


class TestAbstractText:
    @staticmethod
    def _text(source: str) -> str:
        span = locate_abstract(source, "Objectives To study older people", "within primary care.")
        assert span is not None
        return abstract_text(source, span)

    def test_drops_sidebar_blocks_printed_inside_the_abstract(self):
        source = (
            "Objectives To study older people and their \r\nviews.\n"
            "Correspondence to\r\nTessa van Middelaar; \r\nt.vanmiddelaar@amc.uva.nl\n"
            "© Author(s) (or their \r\nemployer(s)) 2022. Re-use \r\npermitted under CC BY.\n"
            "Contact the team at team@example.org for data\n"
            "Received 26 August 2019\r\nAccepted 1 October 2019\n"
            "Design Qualitative interviews within primary care.\n"
        )
        text = self._text(source)
        assert "Correspondence" not in text and "@" not in text
        assert "©" not in text and "Received" not in text
        assert text.startswith("Objectives To study older people")
        assert "Design Qualitative interviews within primary care." in text

    def test_drops_a_conflict_of_interest_footnote_printed_inside_the_abstract(self):
        source = (
            "Objectives To study older people. Criterion validities of the measures\n"
            "(GME), GW Pharmaceuticals, Janssen and Merck. R.S.E.K. receives royalties\r\n"
            "from the BACS battery. S.B. and C.C. are employees of Zinfandel.\n"
            "were tested within primary care.\n"
        )
        text = self._text(source)
        assert "royalties" not in text and "Zinfandel" not in text
        assert "measures\nwere tested within primary care." in text

    def test_keeps_a_block_with_one_kind_of_disclosure_wording(self):
        source = (
            "Objectives To study older people who\n"
            "are employees of 12 hospitals, and the conflicts of interest they face.\n"
            "Results were stable within primary care.\n"
        )
        assert "are employees of 12 hospitals" in self._text(source)

    def test_keeps_a_wrapped_line_that_merely_starts_with_a_furniture_word(self):
        source = (
            "Objectives To study older people and the level of\n"
            "correspondence among the participants.\n"
            "Results were stable within primary care.\n"
        )
        assert "correspondence among the participants" in self._text(source)

    def test_drops_a_repeat_of_the_previous_block_from_a_doubled_text_layer(self):
        conclusion = (
            "CONCLUSION: Both depression and anxiety were weakly associated with oral "
            "health, indicating that they are potential factors influencing"
        )
        source = (
            f"Objectives To study older people. {conclusion}\n"
            f"{conclusion}\r\nhealth within primary care.\n"
        )
        text = self._text(source)
        assert text.count("CONCLUSION") == 1
        assert text.endswith("influencing\nhealth within primary care.")

    def test_strips_a_heading_and_rejoins_unicode_hyphen_line_wraps(self):
        source = (
            "ABSTRACT\nObjectives To study older people on psychi‑\r\n"
            "atric wards, aged 65 years, within primary care.\n"
        )
        span = locate_abstract(
            source, "ABSTRACT Objectives To study older people", "within primary care."
        )
        assert span is not None
        text = abstract_text(source, span)
        assert text.startswith("Objectives")
        assert "psychiatric wards, aged 65 years" in text


class TestResolveTitleKeywords:
    def test_copies_the_abstract_and_keeps_the_other_fields(self):
        answer = TitleKeywordsPointerLLM(
            title="A Study of Things",
            abstract_first_words=_ABSTRACT_FIRST,
            abstract_last_words=_ABSTRACT_LAST,
            keywords=["care", "rare diseases"],
            journal="Journal of Examples",
        )
        resolved = resolve_title_keywords(answer, _FRONT)
        assert resolved.status == "copied"
        assert resolved.response is not None
        assert resolved.response.abstract.startswith("“Background” matters")
        assert resolved.response.keywords == ["care", "rare diseases"]
        assert resolved.response.journal == "Journal of Examples"
        assert resolved.response._abstract_explicitly_absent is False

    def test_explicit_null_anchors_mean_no_abstract(self):
        answer = TitleKeywordsPointerLLM.model_validate(
            {"title": "T", "abstract_first_words": None, "abstract_last_words": None}
        )
        resolved = resolve_title_keywords(answer, _FRONT)
        assert resolved.status == "absent"
        assert resolved.response.abstract is None
        assert "abstract" in resolved.response.model_fields_set
        assert resolved.response._abstract_explicitly_absent is True

    def test_omitted_anchors_are_not_an_explicit_absence(self):
        answer = TitleKeywordsPointerLLM.model_validate({"title": "T"})
        resolved = resolve_title_keywords(answer, _FRONT)
        assert resolved.status == "absent"
        assert "abstract" not in resolved.response.model_fields_set
        assert resolved.response._abstract_explicitly_absent is False

    def test_unplaced_anchors_give_no_response(self):
        answer = TitleKeywordsPointerLLM(
            title="T",
            abstract_first_words="words that are printed nowhere at all",
            abstract_last_words=_ABSTRACT_LAST,
        )
        resolved = resolve_title_keywords(answer, _FRONT)
        assert (resolved.response, resolved.status) == (None, "unplaced")


class TestAnchorChecks:
    """Last words that mark a wrong span: the copy is not made."""

    _OPENING = "We studied how often models copy the abstract in full."

    def _resolve(self, source: str, last: str, keywords: list[str] | None = None):
        answer = TitleKeywordsPointerLLM(
            title="T",
            abstract_first_words="We studied how often models copy the",
            abstract_last_words=last,
            keywords=keywords or [],
        )
        return resolve_title_keywords(answer, source)

    @pytest.mark.parametrize(
        "rest",
        [
            " A few stop early.",
            " Third, some stop early.",
            "\r\nthe sentence goes on in the next line.",
        ],
    )
    def test_last_words_followed_by_more_of_the_paragraph_are_cut_short(self, rest):
        source = f"{self._OPENING} Most copies end well.{rest}\n1 Introduction\n"
        resolved = self._resolve(source, "Most copies end well.")
        assert (resolved.response, resolved.status) == (None, "cut_short")

    @pytest.mark.parametrize(
        "rest", [" in all clinics.", " (in all clinics) and labs.", ' "in all clinics".']
    )
    def test_last_words_that_stop_inside_a_sentence_are_cut_short(self, rest):
        source = f"{self._OPENING} Most copies end well{rest}\n"
        resolved = self._resolve(source, "Most copies end well")
        assert (resolved.response, resolved.status) == (None, "cut_short")

    @pytest.mark.parametrize(
        "rest",
        [
            "\nThe next block starts here.",
            " Keywords: abstracts; copying",
            " © 2024 The Authors.",
            " 2018 The Authors. Published by Elsevier Inc.",
            " (J Korean Med Rehabil 2022;32(4):61-72)",
        ],
    )
    def test_a_line_that_may_follow_the_abstract_is_not_a_sentence(self, rest):
        source = f"{self._OPENING} Most copies end well.{rest}\n"
        resolved = self._resolve(source, "Most copies end well.")
        assert resolved.status == "copied"
        assert resolved.response.abstract.endswith("Most copies end well.")

    @pytest.mark.parametrize(
        "heading",
        [
            "RESEARCH IN CONTEXT",
            "Keywords: abstracts; copying",
            "Statement of Relevance",
            "RELEVANTIE VOOR DE PRAKTIJK Dit artikel",
        ],
    )
    def test_last_words_past_a_box_printed_after_the_abstract_overrun(self, heading):
        source = (
            f"{self._OPENING} Most copies end well.\n{heading}\nThe box says copies end well.\n"
        )
        resolved = self._resolve(source, "The box says copies end well.")
        assert (resolved.response, resolved.status) == (None, "overran")

    def test_last_words_past_the_keyword_list_overrun(self):
        source = (
            f"{self._OPENING} Most copies end well.\n"
            "Abstracts; Copying; Language models\n"
            "Highlights: copies end well.\n"
        )
        keywords = ["Abstracts", "Copying", "Language models"]
        resolved = self._resolve(source, "Highlights: copies end well.", keywords)
        assert (resolved.response, resolved.status) == (None, "overran")

    def test_a_block_that_mentions_the_keywords_is_not_the_keyword_list(self):
        source = (
            f"{self._OPENING}\n"
            "Methods: We counted the abstracts that models copied; copying was common.\n"
        )
        resolved = self._resolve(source, "copying was common.", ["abstracts", "copying"])
        assert resolved.status == "copied"

    def test_a_fuzzy_last_anchor_stops_at_its_own_last_word(self):
        """A last anchor a letter longer than printed ("and" for "or") matches
        a window that runs into the section number printed after the abstract."""
        source = (
            f"{self._OPENING} This adds to the debate or the proposals of the Commission "
            "suffice to restore public trust.\n"
            "2 The social role of auditors\n"
        )
        last = "and the proposals of the Commission suffice to restore public trust."
        resolved = self._resolve(source, last)
        assert resolved.status == "copied_fuzzy"
        assert resolved.response.abstract.endswith("suffice to restore public trust.")


class TestExpandAuthors:
    @staticmethod
    def _answer(affiliations, *ids):
        return AuthorsPointerLLM.model_validate(
            {
                "affiliations": affiliations,
                "authors": [
                    {"given": f"A{n}", "family": f"F{n}", "affiliation_ids": author_ids}
                    for n, author_ids in enumerate(ids)
                ],
            }
        )

    def test_writes_each_authors_affiliations_back_out(self):
        answer = self._answer(["University of Twente", "North-West University"], [1, 2], [2], [])
        authors = expand_authors(answer).authors
        assert [a.affiliation for a in authors] == [
            "University of Twente; North-West University",
            "North-West University",
            "",
        ]
        assert [(a.given, a.family) for a in authors] == [("A0", "F0"), ("A1", "F1"), ("A2", "F2")]

    def test_numbers_counted_from_zero_are_shifted(self):
        authors = expand_authors(self._answer(["U1", "U2"], [0], [1, 0])).authors
        assert [a.affiliation for a in authors] == ["U1", "U2; U1"]

    def test_unknown_and_repeated_numbers_are_dropped(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bibr.clients.pointer_outputs"):
            authors = expand_authors(self._answer(["U1", "U2"], [1, 1, 7])).authors
        assert authors[0].affiliation == "U1"
        assert "outside its list of 2" in caplog.text

    def test_a_placeholder_entry_keeps_the_numbering_of_the_rest(self):
        authors = expand_authors(self._answer(["U1", "string", "U3"], [2, 3])).authors
        assert authors[0].affiliation == "U3"

    @pytest.mark.parametrize(
        "stand_in",
        [
            "Affiliation 2",
            "Affiliation 1 (A0, A1)",
            "Affiliation 3 not printed on this page",
            "N/A",
        ],
    )
    def test_a_stand_in_for_an_unprinted_affiliation_is_left_empty(self, stand_in, caplog):
        with caplog.at_level(logging.WARNING, logger="bibr.clients.pointer_outputs"):
            authors = expand_authors(self._answer(["U1", stand_in], [1, 2], [2])).authors
        assert [a.affiliation for a in authors] == ["U1", ""]
        assert "1 stand-in(s)" in caplog.text

    @pytest.mark.parametrize(
        "name",
        ["Institute for Affiliation Studies", "Affiliation 1: University of Twente", "Nonesuch"],
    )
    def test_an_institution_is_not_taken_for_a_stand_in(self, name):
        assert expand_authors(self._answer([name], [1])).authors[0].affiliation == name

    def test_numbers_without_a_list_give_no_answer(self):
        assert expand_authors(self._answer([], [1], [1, 2])) is None

    def test_no_numbers_and_no_list_is_an_answer(self):
        authors = expand_authors(self._answer([], [], [])).authors
        assert [a.affiliation for a in authors] == ["", ""]

    @pytest.mark.parametrize("field", ["email", "corresponding", "orcid"])
    def test_contact_fields_are_carried_over(self, field):
        value = {"email": "a@b.org", "corresponding": True, "orcid": "0000-0002-1825-0097"}[field]
        answer = AuthorsPointerLLM.model_validate(
            {"affiliations": [], "authors": [{"given": "A", "family": "B", field: value}]}
        )
        assert getattr(expand_authors(answer).authors[0], field) == value
