"""eLife's numbered component DOIs never name the article.

eLife PDFs of 2012-2017 print a DOI under the abstract, the digest, each figure
and each decision letter: the article DOI with a zero-padded three-digit
number (``10.7554/eLife.00013.001``). Under the abstract and digest it carries
no figure label, sits in the front matter and outranks the article's own DOI,
which the running footer prints.
"""

import pytest

from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence

_FOOTER = "Alegado et al. eLife 2012;1:e00013. DOI: 10.7554/eLife.00013 1 of 21"


def _contents(rows, footers=()):
    sections = [PaperSection(0, "Root", 0, None, CanonicalSection.TITLE)]
    sentences = []
    for index, (header, section_type, text, page) in enumerate(rows, start=1):
        sections.append(PaperSection(index, header, 1, 0, section_type))
        sentences.append(
            PaperSentence(
                text_id=index,
                text=text,
                section_id=index,
                paragraph_id=index,
                page_number=page,
            )
        )
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
        detected_footers=list(footers),
    )


def test_abstract_and_digest_dois_do_not_outrank_the_article_doi():
    contents = _contents(
        [
            (
                "Abstract",
                CanonicalSection.ABSTRACT,
                "Choanoflagellates are the closest living relatives of animals. "
                "DOI: http://dx.doi.org/10.7554/eLife.00013.001",
                1,
            ),
            (
                "eLife digest",
                CanonicalSection.ABSTRACT,
                "Animals evolved from single-celled ancestors. "
                "DOI: http://dx.doi.org/10.7554/eLife.00013.002",
                2,
            ),
        ],
        footers=[_FOOTER, _FOOTER, _FOOTER],
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.7554/elife.00013"
    components = [c for c in selection.candidates if c.source_kind == "sentence"]
    assert {c.rejection_reason for c in components} == {"component_candidate"}


def test_cite_as_line_beats_the_abstract_doi_on_the_same_page():
    contents = _contents(
        [
            (
                "Title",
                CanonicalSection.TITLE,
                "Cite as: eLife 2013;2:e00013 DOI: 10.7554/eLife.00013",
                1,
            ),
            (
                "Abstract",
                CanonicalSection.ABSTRACT,
                "We study rosettes. DOI: 10.7554/eLife.00013.001",
                1,
            ),
        ]
    )

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == "10.7554/elife.00013"


@pytest.mark.parametrize("doi", ["10.7554/eLife.88888.3", "10.7554/eLife.00013"])
def test_elife_article_and_version_dois_stay_articles(doi):
    contents = _contents([("Title", CanonicalSection.TITLE, f"DOI: {doi}", 1)])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == doi.casefold()
