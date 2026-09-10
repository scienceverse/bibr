"""Tests for repeated-author (em-dash "ditto") resolution.

Bibliographies in Chicago / older-APA / many economics & humanities styles
replace a repeated author byline with a run of dashes ("———.", "— — —",
"——— and Fischer") meaning "same author(s) as the entry above". Both the NER
and LLM parsers extract the author span verbatim, so the dash placeholder lands
in ``authors`` untouched — useless for downstream matching.

``_resolve_repeated_authors`` expands the leading dash-run to the previous
entry's resolved authors, preserving any trailing coauthors. It uses only
in-document information (the preceding reference), never external enrichment, so
it is consistent with the ground-truth principle.
"""

from bibr.extract.ref_extractor import _resolve_repeated_authors
from bibr.paper import PaperReference


def _ref(authors, bib_id=1, title="A title"):
    return PaperReference(
        bib_id=bib_id,
        title=title,
        authors=authors,
        first_page=None,
        volume=None,
        year=None,
        container=None,
        doi=None,
    )


class TestResolveRepeatedAuthors:
    def test_bare_spaced_em_dashes_take_previous_author(self):
        refs = [_ref("Chami, R.", 1), _ref("— — —", 2)]
        _resolve_repeated_authors(refs)
        assert refs[1].authors == "Chami, R."

    def test_joined_em_dashes_take_previous_author(self):
        refs = [_ref("Chami, R.", 1), _ref("———", 2)]
        _resolve_repeated_authors(refs)
        assert refs[1].authors == "Chami, R."

    def test_dash_run_with_trailing_coauthor_is_preserved(self):
        refs = [_ref("Chami, R.", 1), _ref("——— and J. H. Fischer", 2)]
        _resolve_repeated_authors(refs)
        assert refs[1].authors == "Chami, R. and J. H. Fischer"

    def test_ascii_triple_hyphens_resolve(self):
        # OCR frequently renders an em-dash run as ASCII hyphens.
        refs = [_ref("Acemoglu, D.", 1), _ref("---", 2)]
        _resolve_repeated_authors(refs)
        assert refs[1].authors == "Acemoglu, D."

    def test_comma_ampersand_suffix_joins_without_double_space(self):
        refs = [_ref("Chami, R.", 1), _ref("———, & Smith, J.", 2)]
        _resolve_repeated_authors(refs)
        assert refs[1].authors == "Chami, R., & Smith, J."

    def test_chained_dittos_follow_the_resolved_chain(self):
        # Each ditto repeats the entry immediately above, resolved.
        refs = [
            _ref("Chami, R.", 1),
            _ref("———", 2),
            _ref("——— and J. H. Fischer", 3),
            _ref("— — —", 4),
        ]
        _resolve_repeated_authors(refs)
        assert refs[1].authors == "Chami, R."
        assert refs[2].authors == "Chami, R. and J. H. Fischer"
        assert refs[3].authors == "Chami, R. and J. H. Fischer"

    def test_first_reference_dash_run_is_left_verbatim(self):
        # No previous entry to resolve against.
        refs = [_ref("———", 1), _ref("Cordella, T.", 2)]
        _resolve_repeated_authors(refs)
        assert refs[0].authors == "———"
        assert refs[1].authors == "Cordella, T."

    def test_real_authors_are_untouched(self):
        refs = [_ref("Chami, R.", 1), _ref("Cordella, T., and E. Levy-Yeyati", 2)]
        _resolve_repeated_authors(refs)
        assert refs[0].authors == "Chami, R."
        assert refs[1].authors == "Cordella, T., and E. Levy-Yeyati"

    def test_single_hyphen_prefix_is_not_treated_as_ditto(self):
        # A one/two-char dash is not the convention; a hyphenated real name must
        # survive.
        refs = [_ref("Smith, J.", 1), _ref("-Anne Something", 2)]
        _resolve_repeated_authors(refs)
        assert refs[1].authors == "-Anne Something"

    def test_authorless_entry_breaks_the_repeat_chain(self):
        # A ditto can only repeat the entry immediately above; if that entry has
        # no author, the ditto is left verbatim rather than borrowing further up.
        refs = [_ref("Chami, R.", 1), _ref(None, 2), _ref("———", 3)]
        _resolve_repeated_authors(refs)
        assert refs[2].authors == "———"

    def test_empty_list_is_a_noop(self):
        refs: list[PaperReference] = []
        _resolve_repeated_authors(refs)
        assert refs == []
