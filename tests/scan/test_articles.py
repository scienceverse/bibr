"""Article boundaries on scanned pages (issue #146)."""

from bibr.ocr.types import OcrRegionResult
from bibr.scan.articles import describe, is_reference_heading, split_articles
from bibr.scan.page_kind import PageKind

SCAN = PageKind.SCAN
DIGITAL = PageKind.BORN_DIGITAL


def _r(label, content="", bbox=None):
    return OcrRegionResult(
        index=0, native_label=label, label=label, content=content, bbox_2d=bbox or [0, 0, 1, 1]
    )


def _pages(*pages):
    out = []
    for page in pages:
        regions = list(page)
        for i, region in enumerate(regions):
            region.index = i
        out.append(regions)
    return out


def _contents(pages):
    return [[r.content for r in page] for page in pages]


def test_born_digital_paper_is_never_split():
    pages = _pages(
        [_r("reference", "1. Old ref."), _r("doc_title", "A Paper About Things")],
        [_r("text", "Body."), _r("reference", "2. Ref.")],
        [_r("doc_title", "The Next Article Title")],
    )
    split = split_articles(pages, {0: DIGITAL, 1: DIGITAL, 2: DIGITAL})
    assert not split.changed
    assert _contents(split.pages) == _contents(pages)


def test_previous_article_tail_with_references_is_dropped():
    pages = _pages(
        [
            _r("header", "J. Physiol. 1962"),
            _r("text", "...and so we conclude."),
            _r("paragraph_title", "REFERENCES"),
            _r("reference", "Smith, A. (1950). J. Physiol. 3, 1-10."),
            _r("doc_title", "The Action Potential of the Squid Axon"),
            _r("text", "Our own introduction."),
        ],
        [_r("text", "Methods."), _r("reference", "Jones, B. (1955).")],
    )
    split = split_articles(pages, {0: SCAN, 1: SCAN})
    assert _contents(split.pages)[0] == [
        "J. Physiol. 1962",
        "The Action Potential of the Squid Axon",
        "Our own introduction.",
    ]
    assert [r.index for r in split.pages[0]] == [0, 1, 2]
    assert {d.reason for d in split.dropped} == {"previous_article"}
    assert split.title_page == 0


def test_text_before_title_without_references_is_kept():
    pages = _pages(
        [_r("text", "Journal of Things, Vol. 3"), _r("doc_title", "A Paper About Things")],
        [_r("text", "Body.")],
    )
    split = split_articles(pages, {0: SCAN, 1: SCAN})
    assert not split.changed


def test_floats_above_the_title_are_dropped():
    pages = _pages(
        [
            _r("image", "", [100, 50, 900, 300]),
            _r("figure_title", "Fig. 4. From the previous article.", [100, 310, 900, 330]),
            _r("doc_title", "A Paper About Things", [100, 400, 900, 440]),
            _r("image", "", [100, 600, 900, 800]),
        ],
    )
    split = split_articles(pages, {0: SCAN})
    assert [r.native_label for r in split.pages[0]] == ["doc_title", "image"]
    assert {d.reason for d in split.dropped} == {"float_above_title"}


def test_next_article_after_references_is_dropped():
    pages = _pages(
        [_r("doc_title", "A Paper About Things"), _r("text", "Body.")],
        [
            _r("paragraph_title", "References"),
            _r("reference", "1. A ref."),
            _r("doc_title", "Another Unrelated Article"),
            _r("text", "Its byline and abstract."),
            _r("footer", "Page 2"),
        ],
        [_r("text", "More of the next article."), _r("reference", "1. Its ref.")],
    )
    split = split_articles(pages, {0: SCAN, 1: SCAN, 2: SCAN})
    assert _contents(split.pages) == [
        ["A Paper About Things", "Body."],
        ["References", "1. A ref.", "Page 2"],
        [],
    ]
    assert split.next_article_at == (1, 2)
    assert {d.reason for d in split.dropped} == {"next_article"}


def test_doc_title_before_references_does_not_split():
    # A mislabelled section heading in the body: the paper's references are
    # not printed yet, so it cannot be the next article.
    pages = _pages(
        [_r("doc_title", "A Paper About Things"), _r("text", "Body.")],
        [_r("doc_title", "Experimental Procedures"), _r("text", "Methods.")],
        [_r("reference", "1. A ref.")],
    )
    split = split_articles(pages, {0: SCAN, 1: SCAN, 2: SCAN})
    assert not split.changed


def test_appendix_after_references_is_not_a_new_article():
    pages = _pages(
        [_r("doc_title", "A Paper About Things"), _r("reference", "1. A ref.")],
        [_r("doc_title", "Appendix A. Derivations"), _r("text", "Math.")],
    )
    split = split_articles(pages, {0: SCAN, 1: SCAN})
    assert not split.changed


def test_reply_in_a_discussion_item_is_not_a_new_article():
    pages = _pages(
        [_r("doc_title", "A Paper About Things"), _r("reference", "1. A ref.")],
        [_r("doc_title", "Reply to the Discussants"), _r("reference", "1. Reply ref.")],
    )
    for title in ("Reply to the Discussants", "Author's Reply", "Response to Smith"):
        pages[1][0].content = title
        assert not split_articles(pages, {0: SCAN, 1: SCAN}).changed, title


def test_born_digital_cover_sheet_before_a_scan():
    # A repository cover sheet repeats the title; the paper starts on the scan.
    pages = _pages(
        [_r("doc_title", "A Paper About Things"), _r("text", "Downloaded from a repository.")],
        [
            _r("text", "...end of the previous paper."),
            _r("reference", "Old, A. (1950)."),
            _r("doc_title", "A Paper About Things"),
            _r("text", "Introduction."),
        ],
        [_r("reference", "1. Our ref.")],
    )
    split = split_articles(pages, {0: DIGITAL, 1: SCAN, 2: SCAN})
    assert _contents(split.pages) == [
        ["A Paper About Things", "Downloaded from a repository."],
        ["A Paper About Things", "Introduction."],
        ["1. Our ref."],
    ]


def test_no_title_on_the_first_scanned_pages_leaves_the_paper_alone():
    pages = _pages(
        [_r("text", "Body.")],
        [_r("reference", "1. Ref.")],
        [_r("doc_title", "Another Article Entirely")],
    )
    split = split_articles(pages, {0: SCAN, 1: SCAN, 2: SCAN})
    assert not split.changed


def test_reference_headings_in_other_languages():
    for heading in ("Références", "BIBLIOGRAFÍA", "Literaturverzeichnis", "Список литературы"):
        assert is_reference_heading(heading), heading
    assert not is_reference_heading("Results")


def test_describe_names_reasons_and_pages():
    pages = _pages(
        [_r("doc_title", "A Paper About Things"), _r("reference", "1. A ref.")],
        [_r("doc_title", "Another Unrelated Article"), _r("text", "x")],
    )
    split = split_articles(pages, {0: SCAN, 1: SCAN})
    assert describe(split) == (
        "Dropped 2 regions of neighbouring articles from scanned pages: the next article (pages 2)"
    )
