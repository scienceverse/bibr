"""A given name equal to the family name: an LLM echo, or a name printed so."""

from __future__ import annotations

from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.front_tagger.assemble import assemble
from bibr.schemas import AuthorLLM
from tests.front_tagger.helpers import label, tagger_input


def _convert(given, family, **kwargs):
    authors = CoreMetadataExtractor._convert_llm_authors(
        [AuthorLLM(given=given, family=family)], **kwargs
    )
    return [(a.given, a.family) for a in authors]


def test_a_model_echoing_the_family_into_the_given_name_is_blanked():
    assert _convert("Wen", "Wen") == [("", "Wen")]


def test_a_tagged_name_printed_with_the_family_twice_keeps_its_given_name():
    # The tagger reads the print: "Wen Wen" is the name as printed.
    assert _convert("Wen", "Wen", tagged=True, cap=None) == [("Wen", "Wen")]


def test_a_tagged_name_still_has_the_family_echo_trimmed_off_its_given_name():
    assert _convert("Ada Kell", "Kell", tagged=True, cap=None) == [("Ada", "Kell")]


_TITLE = "Tidal patterns of the northern shelf"
_ABSTRACT = (
    "We measured tides at forty stations along the northern shelf over three winters and "
    "compared them with the model run for the same period."
)


def _tagged(inp, tags, *, rules):
    got = assemble(inp, tags, [1.0] * len(tags), rules=rules)
    authors = CoreMetadataExtractor._convert_llm_authors(got.byline.authors, cap=None, tagged=True)
    return [(a.given, a.family) for a in authors]


def test_a_one_name_author_printed_twice_is_no_name_printed_twice():
    # The byline's "Orvan" and the correspondence line's "Orvan" are read as one
    # author's two parts: two prints of one name, not a "Lin Lin".
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Orvan"),
        ("abstract", _ABSTRACT),
        ("text", "Correspondence: Orvan, orvan@shelf.example"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Orvan"),
        ("ABSTRACT", _ABSTRACT),
        ("FAMILY", "Orvan"),
        ("EMAIL", "orvan@shelf.example"),
    )
    for rules in (False, True):
        assert _tagged(inp, tags, rules=rules) == [("", "Orvan")], rules


def test_a_name_printed_with_the_family_twice_keeps_both_parts():
    inp = tagger_input(("doc_title", _TITLE), ("text", "Tavi Tavi and Ada Kell"))
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Tavi"),
        ("FAMILY", "Tavi"),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
    )
    for rules in (False, True):
        assert _tagged(inp, tags, rules=rules) == [("Tavi", "Tavi"), ("Ada", "Kell")], rules
