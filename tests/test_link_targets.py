"""Joining a line break inside a printed link from the PDF's own link targets."""

from unittest.mock import Mock

import pytest

from bibr.extract.ref_line_stream import StreamLine, _entry_text
from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.structure.parse_text import TextHandlersMixin
from bibr.utils.link_targets import LinkTargets, printed_forms

TARGETS = LinkTargets(
    [
        "https://osf.io/ab3cd",
        "https://osf.io/qwzrt",
        "https://osf.io/7x9kp/",
        "https://github.com/some-Org-Team/ABC",
        "https://github.com/lab/repo/blob/main/analysis/Run_Models_Final.rmd",
        "https://osf.io/pq4rs/?view_only=0123456789abcdef0123456789abcdef",
        "https://github.com/someone/apackage",
        "https://doi.org/10.5061/dryad.1a2b3c4d",
        "https://www.example.org/data/files/table.csv",
        "https://github.com/someone/BethanyL",
    ]
)


def test_printed_forms_cover_scheme_www_and_bare_doi():
    assert printed_forms("https://doi.org/10.5061/Dryad.X1") == {
        "https://doi.org/10.5061/dryad.x1",
        "doi.org/10.5061/dryad.x1",
        "10.5061/dryad.x1",
    }
    assert "example.org/a/b" in printed_forms("http://www.example.org/a/b/")
    assert "example.org/~lab/x" in printed_forms("https://example.org/%7Elab/x")
    assert not any(" " in form for form in printed_forms("https://example.org/a%20b"))
    assert printed_forms("mailto:someone@example.org") == set()
    assert printed_forms("https://a.co") == {"https://a.co"}  # "a.co" alone is too short
    assert printed_forms("https://example.org/&amp;x=1") == {
        "https://example.org/&x=1",
        "example.org/&x=1",
    }


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("(https://osf.io/", "ab3cd),"),  # an identifier with prose after it
        ("osf.io/", "qwzrt."),  # no scheme
        ("osf.", "io/7x9kp/,"),  # host split at its dot, no scheme
        ("https://github.com/some-Org-Team/", "ABC"),  # a short word in capitals
        ("github.com/lab/repo/blob/main/analysis/Run_Models_Final.", "rmd"),  # an extension
        ("osf.io/pq4rs/?view_only=0123456789abcdef", "0123456789abcdef"),  # inside a value
        ("htt", "ps://github.com/someone/apackage"),  # inside the scheme
        ("https://doi.org/10.5061/dryad.1a2b", "3c4d"),  # inside a DOI
        ("doi:10.5061/", "dryad.1a2b3c4d."),  # a DOI link printed as "doi:"
        ("https://www.example.org/data/", "files/table.csv"),
        ("https://www.example.org/", "data/files/table.csv"),  # the link goes on past the word
    ],
)
def test_spans_a_break_inside_a_target(before, after):
    assert TARGETS.spans(before, after)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("https://osf.io/ab3cd", "results"),  # the target ends at the break
        ("https://osf.io/ab3cd),", "The"),
        ("osf.io/", "zzzzz"),  # no target goes on so
        ("the", "data"),
        ("github.com/someone/", "apackageX"),  # letters the target does not have
        ("osf.", "io"),  # agrees over too few characters
        ("", "ab3cd"),
        ("https://osf.io/", ""),
    ],
)
def test_does_not_span(before, after):
    assert not TARGETS.spans(before, after)


def test_a_target_going_on_past_the_word_needs_the_word_to_end_its_line():
    # the middle line of a link printed over three lines
    assert TARGETS.spans("https://www.example.org/", "data/files/", line_ends=True)
    assert not TARGETS.spans("https://www.example.org/", "data/files/", line_ends=False)


@pytest.mark.parametrize(
    ("before", "after", "line_ends"),
    [
        # two words of prose that spell the start of a target's host
        ("Creative", "Commons", False),
        ("Creative", "Commons.", True),
        ("The", "Guardian.", True),
        ("journals.", "PLoS", False),
        ("human", "connectome", False),
    ],
)
def test_prose_spelling_a_host_does_not_join(before, after, line_ends):
    targets = LinkTargets(
        [
            "http://creativecommons.org/licenses/by/4.0/",
            "https://www.theguardian.com/world/2024/feb/16/air-canada",
            "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0239518",
            "https://www.humanconnectome.org/study/hcp-young-adult",
        ]
    )
    assert not targets.spans(before, after, line_ends=line_ends)


def test_a_target_ending_at_the_break_beside_one_going_on_proves_nothing():
    targets = LinkTargets(["https://example.org/data", "https://example.org/data/raw/set1"])
    assert not targets.spans("https://example.org/data", "/raw/set1")
    assert targets.spans("https://example.org/data/raw/", "set1")


def test_matching_ignores_case():
    assert LinkTargets(["https://OSF.io/AB3CD"]).spans("osf.io/", "ab3cd")


def test_from_uri_dicts():
    assert LinkTargets.from_uri_dicts(None) is None
    assert LinkTargets.from_uri_dicts(Mock()) is None
    assert LinkTargets.from_uri_dicts([{"page": 1}, {"uri": None}, "x"]) is None
    targets = LinkTargets.from_uri_dicts(
        [{"page": 1, "bbox": [0, 0, 1, 1], "uri": "https://osf.io/ab3cd"}]
    )
    assert targets is not None and targets.spans("osf.io/", "ab3cd")


@pytest.mark.parametrize(
    ("text", "joined"),
    [
        (
            "preregistered (https://osf.io/\r\nab3cd), the primary analyses",
            "preregistered (https://osf.io/ab3cd), the primary analyses",
        ),
        (
            "Supplement 1 at osf.\r\nio/7x9kp/, the measures",
            "Supplement 1 at osf.io/7x9kp/, the measures",
        ),
        (
            "the package (https://github.com/some-Org-Team/\r\nABC)",
            "the package (https://github.com/some-Org-Team/ABC)",
        ),
        (
            "R Markdown. htt\r\nps://github.com/someone/apackage\r\nBadd, X.",
            "R Markdown. https://github.com/someone/apackage\r\nBadd, X.",
        ),
        (
            "at https://www.example.org/\r\ndata/files/\r\ntable.csv.",
            "at https://www.example.org/data/files/table.csv.",
        ),
    ],
)
def test_fix_ocr_artifacts_joins_wraps_a_target_runs_through(text, joined):
    assert fix_ocr_artifacts(text, links=TARGETS) == joined


def test_fix_ocr_artifacts_leaves_other_breaks_to_the_shape_rules():
    for text in (
        "see https://osf.io/ab3cd\r\nresults show",
        "see https://osf.io/\r\nresults show",
        "the first\r\nline of prose",
        "osf.io/\r\nzzzzz is not a target",
        # no URL in the text and no target through the break: the shape rules,
        # which read only text with a URL, stay out of it too
        "htt\r\nps://unrelated.example/x",
    ):
        assert fix_ocr_artifacts(text, links=TARGETS) == fix_ocr_artifacts(text)


def test_fix_ocr_artifacts_keeps_prose_that_spells_a_host_apart():
    targets = LinkTargets(["http://creativecommons.org/licenses/by/4.0/"])
    for text in (
        "under the terms of the Creative\r\nCommons Attribution License",
        "under the terms of the Creative\r\nCommons.",
    ):
        assert fix_ocr_artifacts(text, links=targets) == fix_ocr_artifacts(text)
    # the link itself, printed over two lines, joins
    assert fix_ocr_artifacts(
        "(http://creative\r\ncommons.org/licenses/by/4.0/).", links=targets
    ) == ("(http://creativecommons.org/licenses/by/4.0/).")


def test_without_targets_nothing_changes():
    text = "preregistered (https://osf.io/\r\nab3cd), the primary"
    assert fix_ocr_artifacts(text, links=None) == fix_ocr_artifacts(text)
    assert fix_ocr_artifacts(text, links=LinkTargets([])) == fix_ocr_artifacts(text)


def test_join_across_regions():
    join = TextHandlersMixin._join_parts
    assert join("The code is at https://github.com/some-Org-Team/", "ABC and", TARGETS) == (
        "The code is at https://github.com/some-Org-Team/",
        "",
    )
    # a wrap hyphen the target does not have is dropped
    assert join("at github.com/someone/Betha-", "nyL.", TARGETS) == (
        "at github.com/someone/Betha",
        "",
    )
    # no target: the space joiner, as before
    assert join("at https://osf.io/", "results show", TARGETS) == ("at https://osf.io/", " ")
    assert join("at https://github.com/some-Org-Team/", "ABC and", None)[1] == " "


def test_reference_entry_joins_a_wrapped_link():
    lines = [
        StreamLine("Someone, A. (2020). apackage. htt", 1, None, 0, "reference_content", True),
        StreamLine("ps://github.com/someone/apackage", 1, None, 0, "reference_content", False),
    ]
    assert (
        _entry_text(lines, TARGETS)
        == "Someone, A. (2020). apackage. https://github.com/someone/apackage"
    )
    assert (
        _entry_text(lines) == "Someone, A. (2020). apackage. htt ps://github.com/someone/apackage"
    )


@pytest.mark.parametrize(
    ("raw", "doi"),
    [
        # a publisher link joined whole now hands the DOI parser its query too
        ("10.12968/bjca.2019.0072?journalCode=bjca", "10.12968/bjca.2019.0072"),
        ("https://doi.org/10.1016/j.x.2020.01.001?via%3Dihub", "10.1016/j.x.2020.01.001"),
        ("10.1371/journal.pone.0123456?download=true", "10.1371/journal.pone.0123456"),
        # a "?" that opens no query, and a SICI DOI, stay
        ("10.1000/abc?def", "10.1000/abc?def"),
        (
            "10.1002/(SICI)1097-4679(199901)55:1<1::AID-JCLP1>3.0.CO;2-#",
            "10.1002/(SICI)1097-4679(199901)55:1<1::AID-JCLP1>3.0.CO;2-#",
        ),
    ],
)
def test_normalize_doi_drops_a_link_query(raw, doi):
    from bibr.utils.text import normalize_doi

    assert normalize_doi(raw) == doi
