"""A title the open-archive cover retyped takes its word breaks from the article.

The real case is a 1972 conference paper deposited in HAL. The cover prints
the deposit's title with a lost line-break hyphen ("CADMIUMAND"); the article's
own first page prints "CADMIUM-" over "AND THALLIUM-CONTAINING", and the parser
keeps that layout title out of the body as a repeat of the cover's. The
letter-mismatch case is another HAL deposit, whose legacy OCR layer misreads
"Al2O3" on the article page. Other text is invented.
"""

from bibr.extract.archive_cover import reground_title_off_archive_cover
from bibr.paper_contents import PaperSentence, RegionSummary
from bibr.schemas import AuthorLLM, CoreMetadataLLM

from .test_core_metadata_author_guards import _candidate, _contents, _extractor, _resolution

_COVER_TITLE = "CRYSTAL STRUCTURE AND PROPERTIES OF NEW CADMIUMAND THALLIUM-CONTAINING PEROVSKITES"
_ARTICLE_TITLE = (
    "CRYSTAL STRUCTURE AND PROPERTIES OF NEW CADMIUM-AND THALLIUM-CONTAINING PEROVSKITES"
)
_BYLINE = "Yu. Venevtsev, A. Kapyshev, V. Lebedev, V. Sal'Nikov, G. Zhdanov"
_HAL_NOTICE = (
    "HAL is a multi-disciplinary open access archive\r\nfor the deposit and dissemination "
    "of scientific research documents"
)
_ABSTRACT = "Abstract. - New cadmium- and thallium-containing perovskites have been synthesized."


def _printed_on_two_rows(title):
    cut = title.rfind(" ", 0, 41)
    return f"{title[:cut]}\r\n{title[cut + 1 :]}"


def _regions(cover_title, article_title, *, notice=_HAL_NOTICE):
    return [
        RegionSummary(
            page=1,
            index=0,
            label="doc_title",
            bbox=None,
            content=_printed_on_two_rows(cover_title) + f"\r\n{_BYLINE}",
        ),
        RegionSummary(page=1, index=5, label="abstract", bbox=None, content=notice),
        RegionSummary(page=2, index=1, label="header", bbox=None, content="JOURNAL DE PHYSIQUE"),
        RegionSummary(page=2, index=2, label="doc_title", bbox=None, content=article_title),
    ]


def _paper(
    cover_title=_COVER_TITLE, article_title=_ARTICLE_TITLE, *, notice=_HAL_NOTICE, body=_ABSTRACT
):
    contents = _contents()
    contents.region_summaries = _regions(cover_title, article_title, notice=notice)
    contents.sentences = [
        PaperSentence(text_id=1, text=f"{_BYLINE}.", section_id=2, paragraph_id=1, page_number=1),
        PaperSentence(text_id=2, text=body, section_id=3, paragraph_id=2, page_number=2),
    ]
    return contents


def _cover_resolution(cover_title=_COVER_TITLE):
    return _resolution(
        _candidate(
            "c1",
            f"{cover_title} {_BYLINE}",
            roles=frozenset({"title", "heading", "byline"}),
            source_kind="heading",
        ),
        _candidate("c2", "To cite this version:", roles=frozenset()),
    )


def test_title_retyped_on_an_archive_cover_takes_the_article_word_breaks():
    title, issue = reground_title_off_archive_cover(_COVER_TITLE, _cover_resolution(), _paper())

    assert title == _ARTICLE_TITLE
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"
    assert issue.evidence_ids == ("reason:title_retyped_on_archive_cover",)
    assert not issue.blocking


def test_a_page_without_the_archive_notice_is_not_a_cover():
    contents = _paper(notice="Submitted on 4 Feb 2008")

    assert reground_title_off_archive_cover(_COVER_TITLE, _cover_resolution(), contents) == (
        _COVER_TITLE,
        None,
    )


def test_an_article_title_with_other_letters_keeps_the_cover_text():
    cover = "HREM AND DIFFRACTION STUDIES OF AN Al2O3/Nb INTERFACE"
    misread = "HREM AND DIFFRACTION STUDIES OF AN A1203/Nb INTERFACE"

    assert reground_title_off_archive_cover(
        cover, _cover_resolution(cover), _paper(cover, misread)
    ) == (cover, None)


def test_a_joined_word_the_article_prints_stays_joined():
    # The article breaks "INTERACTIONS" at the line end, but its text uses the
    # word: the cover's spelling is right.
    cover = "SURFACE INTERACTIONS OF NEW CADMIUM PEROVSKITES"
    broken = "SURFACE INTER-ACTIONS OF NEW CADMIUM PEROVSKITES"
    body = "The interactions were measured at room temperature."

    assert reground_title_off_archive_cover(
        cover, _cover_resolution(cover), _paper(cover, broken, body=body)
    ) == (cover, None)


def test_a_model_title_the_cover_does_not_print_is_left_alone():
    # The model already wrote the article's form; nothing to take from the cover.
    assert reground_title_off_archive_cover(_ARTICLE_TITLE, _cover_resolution(), _paper()) == (
        _ARTICLE_TITLE,
        None,
    )


async def test_extracted_cover_title_takes_the_article_word_breaks():
    ext = _extractor(
        _cover_resolution(),
        CoreMetadataLLM(
            title=_COVER_TITLE,
            authors=[AuthorLLM(given="Yu.", family="Venevtsev")],
            keywords=[],
        ),
        contents=_paper(),
    )

    metadata = await ext.extract()

    assert metadata.title == _ARTICLE_TITLE
    assert "reason:title_retyped_on_archive_cover" in [
        evidence for issue in ext.validation_issues for evidence in issue.evidence_ids
    ]


# A psychoanalysis journal article distributed by Cairn.info. The cover's text
# layer lost one of the two hyphens the article prints in a compound, and the
# article page's layout title label went to its rubric, so its title is a
# text block. The byline and body text are invented.
_CAIRN_COVER_TITLE = (
    "Aux prises avec la destructivité : modalités transférocontre-transférentielles "
    "et aménagements du cadre"
)
_CAIRN_ARTICLE_TITLE = (
    "Aux prises avec la destructivité : modalités transféro-contre-transférentielles "
    "et aménagements du cadre"
)
_CAIRN_NOTICE = (
    "Distribution électronique Cairn.info pour Le Carnet Psy.\n"
    "Vous avez l'autorisation de reproduire cet article dans les limites des conditions."
)


def _cairn_paper(article_block=_CAIRN_ARTICLE_TITLE, *, notice=_CAIRN_NOTICE):
    contents = _contents()
    contents.region_summaries = [
        RegionSummary(page=1, index=0, label="doc_title", bbox=None, content=_CAIRN_COVER_TITLE),
        RegionSummary(page=1, index=1, label="text", bbox=None, content="Nadia Haddad"),
        RegionSummary(page=1, index=10, label="footer", bbox=None, content=notice),
        RegionSummary(page=2, index=0, label="doc_title", bbox=None, content="Recherche"),
        RegionSummary(page=2, index=1, label="text", bbox=None, content=article_block),
    ]
    contents.sentences = [
        PaperSentence(
            text_id=1,
            text="Les analysants interrogent continuellement le cadre.",
            section_id=3,
            paragraph_id=2,
            page_number=2,
        ),
    ]
    return contents


def _cairn_resolution():
    return _resolution(
        _candidate(
            "c1", _CAIRN_COVER_TITLE, roles=frozenset({"title", "heading"}), source_kind="heading"
        ),
        _candidate("c2", "Nadia Haddad", roles=frozenset({"byline"})),
    )


def test_a_distributor_cover_title_takes_the_article_hyphens():
    title, issue = reground_title_off_archive_cover(
        _CAIRN_COVER_TITLE, _cairn_resolution(), _cairn_paper()
    )

    assert title == _CAIRN_ARTICLE_TITLE
    assert issue is not None
    assert issue.evidence_ids == ("reason:title_retyped_on_archive_cover",)


def test_a_distributor_cover_without_its_notice_keeps_its_title():
    contents = _cairn_paper(notice="Date de mise en ligne : 01/02/2016")

    assert reground_title_off_archive_cover(_CAIRN_COVER_TITLE, _cairn_resolution(), contents) == (
        _CAIRN_COVER_TITLE,
        None,
    )


def test_a_hyphenated_cover_word_the_article_breaks_at_a_line_end_stays():
    # The article's hyphen after "transféro" ends a row, so it may be a
    # line-end break rather than a printed hyphen: the cover's form stays.
    broken = _CAIRN_ARTICLE_TITLE.replace("transféro-contre", "transféro-\r\ncontre")

    assert reground_title_off_archive_cover(
        _CAIRN_COVER_TITLE, _cairn_resolution(), _cairn_paper(broken)
    ) == (_CAIRN_COVER_TITLE, None)


def test_two_article_blocks_that_print_the_title_differently_keep_the_cover_text():
    # Two blocks of the article page match the cover's letters with other
    # hyphens: there is no single printing to take the breaks from.
    contents = _cairn_paper()
    contents.region_summaries.append(
        RegionSummary(
            page=2,
            index=2,
            label="text",
            bbox=None,
            content=_CAIRN_ARTICLE_TITLE.replace("transférentielles", "transfé-rentielles"),
        )
    )

    assert reground_title_off_archive_cover(_CAIRN_COVER_TITLE, _cairn_resolution(), contents) == (
        _CAIRN_COVER_TITLE,
        None,
    )
