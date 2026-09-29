"""Record agreement: selecting the paper's own record when dominance abstains.

All titles, names, DOIs and geometry below are invented. Each fixture is a
page shape that used to abstain -- a cover page or citation box repeating the
article, a translated record, furniture rooting a second block -- or a
multi-record page that must keep abstaining.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

import bibr.extract.front_matter as front_matter
from bibr.extract.front_matter import resolve_front_matter
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
)
from bibr.pipeline.identity import ExpectedIdentity


def _section(
    section_id: int,
    header: str,
    *,
    section_type: CanonicalSection = CanonicalSection.UNKNOWN,
    bbox: tuple[float, float, float, float] | None = None,
    page: int = 1,
) -> PaperSection:
    return PaperSection(
        section_id=section_id,
        header=header,
        level=0 if section_id == 0 else 1,
        parent_section_id=None,
        section_type=section_type,
        provenance=[Provenance(page_no=page, bbox=bbox)] if bbox is not None else [],
    )


def _row(
    text_id: int,
    text: str,
    *,
    y: float,
    label: str = "text",
    section_id: int = 0,
    page: int = 1,
) -> PaperSentence:
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=text_id,
        page_number=page,
        provenance=[Provenance(page_no=page, bbox=(60.0, y, 460.0, y + 30.0))],
        region_meta={"region_type": label, "font_size": 9.0, "font_bold": False},
    )


def _contents(
    rows: list[PaperSentence],
    *,
    sections: list[PaperSection] | None = None,
    detected_title: str | None = None,
) -> PaperContents:
    sections = sections or [_section(0, "Root")]
    return PaperContents(
        sentences=rows,
        sections=sections,
        tables=[],
        links=[],
        sections_text={section.section_id: "" for section in sections},
        detected_title=detected_title,
        region_summaries=[],
    )


def _selected_text(resolution) -> str:
    block = next(
        block for block in resolution.blocks if block.block_id == resolution.selected_block_id
    )
    by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    return "\n".join(by_id[candidate_id].raw_text for candidate_id in block.candidate_ids)


def _dominance_abstains(contents: PaperContents) -> bool:
    """Today's rule on the same blocks: the fallback only ever runs after this."""

    candidates = front_matter.collect_front_matter_candidates(contents)
    blocks = front_matter.group_front_matter_blocks(candidates)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return len(blocks) > 1 and front_matter._select_dominant_coherent_block(blocks, by_id) is None


TITLE = "Delayed Recall of Household Losses After Regional Flooding"
BYLINE = "Alma J. Example, Bruno Sample"
DOI = "10.1000/recall.1991.0410"


def _cover_page_and_article(*, cover_doi: str = DOI) -> PaperContents:
    """A publisher cover page repeating the article's title, authors and DOI."""

    return _contents(
        [
            _row(1, TITLE, y=60.0, label="doc_title"),
            _row(2, BYLINE, y=100.0),
            _row(
                3,
                "To cite this article: Example, A. J., Sample, B. (1991) Delayed recall of "
                f"household losses after regional flooding. Journal of Invented Studies. "
                f"DOI: {cover_doi}",
                y=140.0,
            ),
            _row(4, TITLE, y=60.0, label="doc_title", page=2),
            _row(5, BYLINE, y=100.0, page=2),
            _row(
                6,
                "Households were asked about their losses long after the flood.",
                y=140.0,
                label="abstract",
                page=2,
            ),
            _row(7, f"DOI: {DOI}", y=200.0, page=2),
        ],
        detected_title=TITLE,
    )


def test_cover_page_repeating_the_doi_selects_the_article_page():
    contents = _cover_page_and_article()
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selection_method == "record_agreement"
    assert resolution.selected_block_id == "front-matter-block-2"
    assert "agreeing_record:front-matter-block-1:doi" in resolution.reason_flags
    assert "multiple_plausible_blocks" not in resolution.reason_flags
    assert resolution.allowed_text_ids == frozenset({4, 5, 6, 7})
    assert issues == ()


def test_citation_box_without_a_doi_agrees_by_its_title():
    contents = _contents(
        [
            _row(1, TITLE, y=60.0, label="doc_title"),
            _row(2, "Alma J. Example and Bruno Sample", y=100.0),
            _row(
                3,
                'Recommended Citation: Example, Alma J. and Sample, Bruno, "Delayed recall of '
                'household losses after regional flooding" (1991). Institute Publications. 55.',
                y=140.0,
            ),
            _row(4, TITLE, y=60.0, label="doc_title", page=2),
            _row(5, "Alma J. Example and Bruno Sample", y=100.0, page=2),
            _row(
                6,
                "Households were asked about their losses long after the flood.",
                y=140.0,
                label="abstract",
                page=2,
            ),
        ],
        detected_title=TITLE,
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-2"
    assert "agreeing_record:front-matter-block-1:title" in resolution.reason_flags
    assert issues == ()


RU_TITLE = "Сравнительная оценка эффективности двух методов реабилитации после инсульта"
EN_TITLE = "Comparative evaluation of the effectiveness of two rehabilitation methods after stroke"


def _bilingual_records(english_byline: str) -> PaperContents:
    return _contents(
        [
            _row(1, RU_TITLE, y=60.0, label="doc_title"),
            _row(2, "Иванов И.И., Щукина Ю.В., Сидоров В.К.", y=100.0),
            _row(
                3,
                "Сравнили два метода реабилитации у пациентов после инсульта.",
                y=140.0,
                label="abstract",
            ),
            _row(4, EN_TITLE, y=300.0, label="doc_title"),
            _row(5, english_byline, y=340.0),
            _row(
                6,
                "Two rehabilitation methods were compared in patients after stroke.",
                y=380.0,
                label="abstract",
            ),
        ],
        detected_title=RU_TITLE,
    )


def test_translated_record_with_a_transliterated_byline_agrees_on_authors():
    contents = _bilingual_records("Ivanov I.I., Shchukina Yu.V., Sidorov V.K.")
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selection_method == "record_agreement"
    # The first-printed, parser-detected original stays the record.
    assert resolution.selected_block_id == "front-matter-block-1"
    assert "agreeing_record:front-matter-block-2:authors" in resolution.reason_flags
    assert issues == ()


def test_translated_title_with_different_authors_abstains():
    contents = _bilingual_records("Petrov A.A., Smirnova E.V., Volkov O.O.")

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert "multiple_plausible_blocks" in resolution.reason_flags
    assert any(flag.startswith("conflicting_records:") for flag in resolution.reason_flags)
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_translated_title_and_abstract_without_a_byline_attach_to_the_record():
    title = "Young witnesses in court: waiting rooms and participation rights"
    contents = _contents(
        [
            _row(1, title, y=60.0, label="doc_title"),
            _row(2, "Clara Example, Maria Sample", y=100.0),
            _row(
                3,
                "Court waiting rooms were designed by adults and for adults.",
                y=140.0,
                label="abstract",
            ),
            _row(
                4,
                "As salas de espera dos tribunais foram desenhadas por adultos.",
                y=300.0,
                label="abstract",
                section_id=2,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                2,
                "Jovens testemunhas em tribunal: salas de espera e direitos de participação",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 260.0, 460.0, 290.0),
            ),
        ],
        detected_title=title,
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert "attached_block:front-matter-block-2" in resolution.reason_flags
    assert issues == ()


@pytest.mark.parametrize(
    "furniture",
    [
        "a r t i c l e i n f o",
        "CITATION",
        "August 2020",
        "Yan Example: y.example@example.ac.uk",
        "Uniwersytet Przykładowy author@example.com",
    ],
)
def test_furniture_rooting_a_second_block_no_longer_vetoes(furniture):
    title = "Parental warmth and shyness in toddlerhood predict later attention"
    contents = _contents(
        [
            _row(1, title, y=60.0, label="doc_title"),
            _row(2, "Rita J. Example, Karin A. Sample", y=100.0),
            _row(
                3,
                "Shy toddlers were followed into preschool and tested twice.",
                y=300.0,
                label="abstract",
                section_id=2,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                2,
                furniture,
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 260.0, 460.0, 290.0),
            ),
        ],
        detected_title=title,
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert resolution.selection_method == "record_agreement"
    assert "attached_block:front-matter-block-2" in resolution.reason_flags
    assert title in _selected_text(resolution)
    assert issues == ()


def _banner_above_the_record(banner_doi: str) -> PaperContents:
    title = "Opinion Networks: Shared Attitudes Form a Social Information System"
    return _contents(
        [
            _row(1, "THIS IS THE ACCEPTED VERSION PRIOR TO TYPESETTING AND PROOFING", y=20.0),
            _row(2, f"Version of record: https://doi.org/{banner_doi}", y=50.0),
            _row(3, title, y=100.0, label="doc_title"),
            _row(4, "Michael Example, Anna Sample", y=140.0),
            _row(
                5,
                "Shared attitudes bind people into groups that others can read.",
                y=180.0,
                label="abstract",
            ),
            _row(6, "DOI: 10.1000/networks.2025.1", y=240.0),
        ],
        detected_title=title,
    )


def test_accepted_version_banner_with_the_papers_doi_attaches():
    contents = _banner_above_the_record("10.1000/networks.2025.1")
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-2"
    assert "attached_block:front-matter-block-1" in resolution.reason_flags
    assert issues == ()


def test_attached_block_printing_another_doi_abstains():
    contents = _banner_above_the_record("10.1000/unrelated.2019.7")

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert "doi_conflict:front-matter-block-1" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_same_title_with_different_dois_abstains():
    contents = _cover_page_and_article(cover_doi="10.1000/recall.erratum.2")

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert (
        "conflicting_records:front-matter-block-1:front-matter-block-2:doi"
        in resolution.reason_flags
    )
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def _two_abstracts(second_byline: str | None) -> PaperContents:
    rows = [
        _row(
            1,
            "Loneliness Among Older Immigrants in Urban Neighbourhoods",
            y=60.0,
            label="paragraph_title",
        ),
        _row(2, "Mei Lin, Jordan Smith", y=100.0),
        _row(3, "We interviewed older immigrants about loneliness.", y=140.0, label="abstract"),
        _row(
            4,
            "Social Technology Use and Loneliness During the Pandemic",
            y=300.0,
            label="paragraph_title",
        ),
        _row(6, "We surveyed adults about social technology use.", y=380.0, label="abstract"),
    ]
    if second_byline is not None:
        rows.insert(4, _row(5, second_byline, y=340.0))
    return _contents(rows)


def test_two_papers_by_the_same_authors_in_one_language_abstain():
    # A research group presenting two abstracts: shared surnames are not
    # evidence of one paper when the titles differ in the same language.
    contents = _two_abstracts("Mei Lin, Jordan Smith")
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert (
        "conflicting_records:front-matter-block-1:front-matter-block-2:title"
        in resolution.reason_flags
    )
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_second_title_and_abstract_in_the_records_language_abstains():
    # An abstract book whose second byline went unseen: a title and abstract
    # without a byline attach only as a translation, never in the same language.
    contents = _two_abstracts(None)
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert "unmatched_presentation:front-matter-block-2" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_compiled_abstract_book_with_affiliation_typed_bylines_abstains():
    """Uppercase titles, author lists typed as affiliations, one abstract each."""

    contents = _contents(
        [
            _row(1, "NEIGHBOURHOOD WALKING GROUPS AND LONELINESS IN LATER LIFE", y=60.0),
            _row(
                2,
                "Matthew Example, Olivia Sample, and Max Reader, Example University, Springfield",
                y=100.0,
            ),
            _row(
                3,
                "Walking groups met weekly for a year in four neighbourhoods.",
                y=140.0,
                label="abstract",
            ),
            _row(4, "SURVEY AND DIARY MEASURES OF SOCIAL CONTACT IN RETIREMENT", y=300.0),
            _row(
                5,
                "Elliot Example, Erin Sample, and Kai Reader, Example State University",
                y=340.0,
            ),
            _row(
                6,
                "Retirees reported their social contacts in a survey and a diary.",
                y=380.0,
                label="abstract",
            ),
        ]
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert any(flag.startswith("conflicting_records:") for flag in resolution.reason_flags)
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_record_without_byline_evidence_stays_abstained():
    # The long uppercase title holds a byline role only because of its
    # capitalization; the printed author line never reached front matter.
    title = (
        "SPELLING-INDUCED ERRORS AMONG THE HARDEST WORDS IN PRONUNCIATION TEACHING: LOCAL "
        "VERSUS GLOBAL PATTERNS AND WORDS COMMONLY MISREAD"
    )
    contents = _contents(
        [
            _row(1, title, y=60.0, label="doc_title"),
            _row(
                2,
                "This paper presents the results of a questionnaire study on spelling.",
                y=300.0,
                label="abstract",
                section_id=2,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                2,
                "Uniwersytet Przykładowy author@example.com",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 260.0, 460.0, 290.0),
            ),
        ],
        detected_title=title,
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert "incomplete_record" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_agreement_never_overrides_a_supplied_expected_identity(monkeypatch):
    contents = _cover_page_and_article()
    expected = ExpectedIdentity(
        queue_record_id="agreement-guard",
        expected_title="An entirely different benchmark paper title",
    )
    monkeypatch.setattr(
        front_matter,
        "_select_agreeing_record",
        lambda *args, **kwargs: pytest.fail("agreement must not run with expected identity"),
    )

    resolution, issues = resolve_front_matter(
        contents, expected_identity=expected, target_required=True
    )

    assert resolution.selected_block_id is None
    assert "expected_title_not_found" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_fast_path_selections_never_consult_agreement(monkeypatch):
    """Unique and dominant selections are exactly today's, byte for byte."""

    from tests.extract.test_front_matter import _article_with_weak_developed_seed

    single = _contents(
        [
            _row(1, TITLE, y=60.0, label="doc_title"),
            _row(2, BYLINE, y=100.0),
            _row(3, "Households were asked about their losses.", y=140.0, label="abstract"),
        ],
        detected_title=TITLE,
    )
    before = {
        "unique": resolve_front_matter(single, target_required=True),
        "dominant": resolve_front_matter(_article_with_weak_developed_seed(), target_required=True),
    }
    monkeypatch.setattr(
        front_matter,
        "_select_agreeing_record",
        lambda *args, **kwargs: pytest.fail("the fast path must not reach record agreement"),
    )
    after = {
        "unique": resolve_front_matter(single, target_required=True),
        "dominant": resolve_front_matter(_article_with_weak_developed_seed(), target_required=True),
    }

    assert before == after
    assert after["unique"][0].selection_method == "unique_block"
    assert after["dominant"][0].selection_method == "coherent_dominance"


@pytest.mark.parametrize(
    ("printed", "transliterated"),
    [
        ("Иванов И.И., Щукина Ю.В.", "Ivanov I.I., Shchukina Yu.V."),
        ("Олег Пайда", "Payda, O."),
        ("Example, A. J. and Sample, B.", BYLINE),
    ],
)
def test_surnames_match_across_scripts_and_citation_order(printed, transliterated):
    left = front_matter._person_surnames(printed)
    right = front_matter._person_surnames(transliterated)

    assert left
    assert all(any(front_matter._same_surname(a, b) for b in right) for a in left)


def _identity(
    title: str,
    *,
    citations: tuple[str, ...] = (),
    language: str | None = None,
    surnames: frozenset[str] = frozenset(),
    dois: frozenset[str] = frozenset(),
):
    return front_matter._RecordIdentity(
        block=front_matter.FrontMatterBlock(block_id="b", candidate_ids=(), title_candidate_ids=()),
        order=0,
        titles=(title,),
        citations=citations,
        language=language,
        surnames=surnames,
        dois=dois,
        byline=True,
        abstract=False,
        anatomy=False,
        doc_title=False,
        detected=None,
        cover=False,
    )


@pytest.mark.parametrize(
    ("left", "right", "agree"),
    [
        (TITLE, TITLE.upper(), True),
        # Subtitle set on its own row; a lost space and a line-break hyphen.
        (f"{TITLE}: A Two-Wave Study", TITLE, True),
        (
            "Young witnesses in court: waiting roomsand rights",
            "Young Witnesses in Court: Wait-\ning Rooms and Rights",
            True,
        ),
        # One letter misread by OCR in a long word.
        (TITLE, "Delayed Recall of Househo1d Losses After Regional Flooding", True),
        ("First Study of Community Health", "Second Study of Community Health", False),
        (
            "Effects of mindfulness training on anxiety among adults",
            "Effects of mindfulness training on anxiety among adolescents",
            False,
        ),
        (
            "Effects of exercise on sleep quality in older adults with insomnia",
            "Effects of exercise on sleep quality in younger adults with insomnia",
            False,
        ),
        # One letter apart, but in a short word: a different animal.
        (
            "Effects of chronic noise exposure in rats",
            "Effects of chronic noise exposure in cats",
            False,
        ),
        (
            "Effects of training on anxiety: Study 1",
            "Effects of training on anxiety: Study 2",
            False,
        ),
        # A title plus words is a different title, and so is a duplicated,
        # garbled text layer: neither can be told from "... with autism".
        ("Social anxiety in adolescents", "Social anxiety in adolescents with autism", False),
        (
            "Delayed recall of hou Delayed recall of household l ses losses after regional "
            "fl ing after regional flooding",
            TITLE,
            False,
        ),
        # A shared subtitle is not a shared title.
        (
            "Mindfulness in schools: A Randomized Controlled Trial",
            "Exercise in prisons: A Randomized Controlled Trial",
            False,
        ),
    ],
)
def test_title_agreement_tolerates_layout_but_not_different_titles(left, right, agree):
    assert front_matter._titles_agree(_identity(left), _identity(right)) is agree


@pytest.mark.parametrize(
    ("citation", "agree"),
    [
        (
            "To cite this article: Example, A. J. (1991). Delayed recall of household losses "
            "after regional flooding. Journal of Invented Studies, 4, 575-590.",
            True,
        ),
        (
            'Recommended Citation: Example, Alma J., "Delayed Recall of Household Losses After '
            'Regional Flooding" (1991).',
            True,
        ),
        # The cited title runs on: it is a longer, different title.
        (
            "Example, A. J. (1991). Delayed recall of household losses after regional flooding "
            "in coastal towns. Journal of Invented Studies.",
            False,
        ),
    ],
)
def test_citation_line_agrees_only_when_it_prints_the_whole_title(citation, agree):
    record = _identity(TITLE)
    cover = _identity("Household Losses Journal Supplement", citations=(citation,))

    assert front_matter._titles_agree(record, cover) is agree


@pytest.mark.parametrize(
    ("title", "language"),
    [
        (EN_TITLE, "en"),
        ("Jovens testemunhas em tribunal: salas de espera e direitos de participação", "pt"),
        (RU_TITLE, "ru"),
        ("Перші кроки реабілітації та українська практика", "uk"),
        ("Сравнительная оценка метода", "cyrillic"),
        ("Deep Learning Pipelines", None),
    ],
)
def test_title_language_reads_function_words_and_script_letters(title, language):
    assert front_matter._title_language(title) == language


# Wrong-merge guards: two different papers must never be read as one record.

LONELINESS = "Loneliness Among Older Immigrants in Urban Neighbourhoods"


def _two_articles(
    second_title: str,
    *,
    second_byline: str = "Anna Example, Ben Sample",
    label: str = "doc_title",
    detected: str | None = None,
    first_extra: tuple[str, ...] = (),
    second_extra: tuple[str, ...] = (),
) -> PaperContents:
    """Two complete records with different titles; the second is the PDF's own."""

    rows = [
        _row(1, LONELINESS, y=60.0, label=label),
        _row(2, "Mei Lin, Jordan Smith", y=100.0),
        _row(3, "We interviewed older immigrants about loneliness.", y=140.0, label="abstract"),
        *(_row(10 + index, text, y=170.0) for index, text in enumerate(first_extra)),
        _row(20, second_title, y=300.0, label=label),
        _row(21, second_byline, y=340.0),
        _row(22, "We surveyed students about their mental health.", y=380.0, label="abstract"),
        *(_row(30 + index, text, y=410.0) for index, text in enumerate(second_extra)),
    ]
    return _contents(rows, detected_title=second_title if detected is None else detected)


def _assert_abstains(contents: PaperContents, flag: str) -> None:
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert "multiple_plausible_blocks" in resolution.reason_flags
    assert flag in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


@pytest.mark.parametrize(
    "own_title",
    [
        "Mental Health of University Students During the Pandemic",
        "School Belonging and Academic Achievement in Early Adolescence",
        "25 Years of Research on Sleep and Mental Health in Students",
        "Mindful Parenting",
        "城市老年人孤独感与社会支持研究",
    ],
)
def test_a_layout_title_that_looks_like_furniture_is_still_a_competing_record(own_title):
    # Named institutions, a leading number, two words or an unspaced script do
    # not make layout's own title furniture that could attach to another paper.
    contents = _two_articles(own_title)
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert any(
        flag.startswith("conflicting_records:front-matter-block-1:front-matter-block-2:")
        for flag in resolution.reason_flags
    )
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_block_with_its_own_byline_and_abstract_must_name_the_records_authors():
    # Without layout evidence the heading names a university, so it is no
    # identity title and its block is no record; its own byline and abstract
    # still make it another paper.
    contents = _contents(
        [
            _row(1, LONELINESS, y=60.0, label="paragraph_title"),
            _row(2, "Mei Lin, Jordan Smith", y=100.0),
            _row(3, "We interviewed older immigrants about loneliness.", y=140.0, label="abstract"),
            _row(4, "Anna Example, Ben Sample", y=340.0, section_id=2),
            _row(5, "We surveyed students about sleep.", y=380.0, label="abstract", section_id=2),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                2,
                "Mental Health of University Students During the Pandemic",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 300.0, 460.0, 330.0),
            ),
        ],
    )

    _assert_abstains(contents, "unlinked_block:front-matter-block-2")


def test_a_block_naming_the_records_authors_still_attaches():
    # The same shape under a furniture heading, but the second byline repeats
    # the record's authors: a correspondence box, not another paper.
    contents = _contents(
        [
            _row(1, LONELINESS, y=60.0, label="doc_title"),
            _row(2, "Mei Lin, Jordan Smith", y=100.0),
            _row(3, "Mei Lin, Jordan Smith", y=340.0, section_id=2),
            _row(
                4,
                "We interviewed older immigrants about loneliness.",
                y=380.0,
                label="abstract",
                section_id=2,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                2,
                "CITATION",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 300.0, 460.0, 330.0),
            ),
        ],
        detected_title=LONELINESS,
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert "attached_block:front-matter-block-2" in resolution.reason_flags
    assert issues == ()


SECOND = "Social Technology Use and Loneliness During the Pandemic"
# U+2010 HYPHEN, as typeset DOIs print it; the DOI pattern stops at it.
HYPHEN = chr(0x2010)


@pytest.mark.parametrize(
    ("first_doi", "second_doi", "flag"),
    [
        # A DOI that prefixes another is a different DOI.
        ("10.1000/abc1", "10.1000/abc12", "doi"),
        # Typeset hyphens once cut both DOIs down to "10.1000/0033".
        (f"10.1000/0033{HYPHEN}2909.126.1.3", f"10.1000/0033{HYPHEN}2909.127.4.5", "doi"),
        # A proceedings volume's DOI, printed with every abstract, does not
        # join two titles that differ in one language.
        ("10.1000/meeting.2020", "10.1000/meeting.2020", "doi_title"),
    ],
)
def test_dois_join_records_only_when_equal_and_titles_do_not_conflict(first_doi, second_doi, flag):
    contents = _two_articles(
        SECOND,
        first_extra=(f"DOI: {first_doi}",),
        second_extra=(f"DOI: {second_doi}",),
    )

    _assert_abstains(
        contents, f"conflicting_records:front-matter-block-1:front-matter-block-2:{flag}"
    )


def test_a_typeset_hyphen_does_not_hide_an_equal_doi():
    contents = _cover_page_and_article(cover_doi=DOI.replace("recall.", f"recall{HYPHEN}"))
    contents.sentences[-1].text = f"DOI: {DOI.replace('recall.', 'recall-')}"

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-2"
    assert "agreeing_record:front-matter-block-1:doi" in resolution.reason_flags
    assert issues == ()


def test_funder_dois_are_not_paper_identity():
    # Both abstracts thank the same funder; the funder DOI is no shared DOI,
    # so the different titles decide.
    funder = "Funded by grant https://doi.org/10.13039/501100000780"
    _assert_abstains(
        _two_articles(SECOND, first_extra=(funder,), second_extra=(funder,)),
        "conflicting_records:front-matter-block-1:front-matter-block-2:title",
    )


def test_a_title_contained_in_a_longer_title_is_a_different_paper():
    _assert_abstains(
        _contents(
            [
                _row(1, "Social anxiety in adolescents with autism", y=60.0, label="doc_title"),
                _row(2, "Mei Lin, Jordan Smith", y=100.0),
                _row(3, "Adolescents with autism reported anxiety.", y=140.0, label="abstract"),
                _row(4, "Social anxiety in adolescents", y=300.0, label="doc_title"),
                _row(5, "Anna Example, Ben Sample", y=340.0),
                _row(6, "Adolescents reported social anxiety.", y=380.0, label="abstract"),
            ],
            detected_title="Social anxiety in adolescents",
        ),
        "conflicting_records:front-matter-block-1:front-matter-block-2:title",
    )


def test_translated_records_sharing_one_common_surname_stay_unlinked():
    # Two small teams share one surname: not enough to call the Portuguese and
    # English records one paper.
    contents = _contents(
        [
            _row(
                1,
                "Efeitos da ansiedade no desempenho escolar de crianças",
                y=60.0,
                label="doc_title",
            ),
            _row(2, "João Silva, Maria Santos", y=100.0),
            _row(3, "Avaliamos a ansiedade em crianças.", y=140.0, label="abstract"),
            _row(
                4,
                "Working memory training in older adults: a pilot study",
                y=300.0,
                label="doc_title",
            ),
            _row(5, "Ana Silva, Pedro Costa", y=340.0),
            _row(6, "We trained working memory in older adults.", y=380.0, label="abstract"),
        ],
        detected_title="Working memory training in older adults: a pilot study",
    )

    _assert_abstains(contents, "unlinked_records")


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ("zhang", "zhong", False),
        ("liang", "jiang", False),
        ("paida", "payda", True),
        ("sinyachkin", "sinjachkin", True),
        ("hernandez", "hernandes", True),
    ],
)
def test_short_surnames_compare_exactly_across_romanizations(left, right, same):
    assert front_matter._same_surname(left, right) is same


@pytest.mark.parametrize(
    ("left", "right", "agree"),
    [
        ({"silva", "santos"}, {"silva", "costa"}, False),
        ({"ivanov"}, {"ivanov", "petrova"}, True),
        ({"ivanov", "petrova", "sidorov"}, {"ivanov", "petrova", "orlov"}, True),
    ],
)
def test_translated_bylines_agree_on_every_or_at_least_two_surnames(left, right, agree):
    assert front_matter._surnames_agree(frozenset(left), frozenset(right)) is agree


def test_detected_title_on_a_block_outside_the_record_abstains():
    # The parser's title belongs to a block that is not in the agreeing group,
    # so a null metadata title would later be filled from another paper.
    contents = _contents(
        [
            _row(1, LONELINESS, y=60.0, label="doc_title"),
            _row(2, "Mei Lin, Jordan Smith", y=100.0),
            _row(3, "We interviewed older immigrants about loneliness.", y=140.0, label="abstract"),
            _row(4, "DOI: 10.1000/students.2021.4", y=340.0, section_id=2),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                2,
                "Mental Health of University Students During the Pandemic",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 300.0, 460.0, 330.0),
            ),
        ],
        detected_title="Mental Health of University Students",
    )

    _assert_abstains(contents, "detected_title_outside_record:front-matter-block-2")


def test_detected_title_on_a_translated_presentation_is_allowed():
    title = "Jovens testemunhas em tribunal: salas de espera e direitos de participação"
    contents = _contents(
        [
            _row(1, title, y=60.0, label="doc_title"),
            _row(2, "Clara Example, Maria Sample", y=100.0),
            _row(3, "As salas de espera foram desenhadas por adultos.", y=140.0, label="abstract"),
            _row(
                4,
                "Court waiting rooms were designed by adults.",
                y=340.0,
                label="abstract",
                section_id=2,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                2,
                "School Witnesses in Court: Waiting Rooms and Participation Rights",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 300.0, 460.0, 330.0),
            ),
        ],
        detected_title="School Witnesses in Court",
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert "attached_block:front-matter-block-2" in resolution.reason_flags
    assert issues == ()


def test_the_selected_record_must_hold_a_byline():
    # The original's title page prints its author on an untyped line; the
    # only byline is on the translated summary, which never stands in for it.
    contents = _contents(
        [
            _row(1, RU_TITLE, y=60.0, label="doc_title"),
            _row(2, "Иван Петров", y=100.0),
            _row(3, "Сравнили два метода реабилитации.", y=140.0, label="abstract"),
            _row(4, "DOI: 10.1000/rehab.2024.1", y=180.0),
            _row(5, EN_TITLE, y=60.0, label="doc_title", page=6),
            _row(6, "Ivan Petrov, Olga Sidorova", y=100.0, page=6),
            _row(7, "Two rehabilitation methods were compared.", y=140.0, label="abstract", page=6),
            _row(8, "DOI: 10.1000/rehab.2024.1", y=180.0, page=6),
        ],
        detected_title=RU_TITLE,
    )

    _assert_abstains(contents, "record_without_byline:front-matter-block-1")


def test_a_cover_page_without_a_byline_hands_selection_to_the_title_page():
    # The cover carries the parser's exact title but no byline; the title page
    # (title plus subtitle, same language) has one.
    contents = _contents(
        [
            _row(1, TITLE, y=60.0, label="doc_title"),
            _row(2, f"DOI: {DOI}", y=100.0),
            _row(3, f"{TITLE}: A Two-Wave Study", y=60.0, label="doc_title", page=2),
            _row(4, BYLINE, y=100.0, page=2),
            _row(5, "Households were asked about their losses.", y=140.0, label="abstract", page=2),
            _row(6, f"DOI: {DOI}", y=200.0, page=2),
        ],
        detected_title=TITLE,
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-2"
    assert issues == ()


def test_blocks_without_an_identity_title_abstain():
    contents = _contents(
        [
            _row(1, BYLINE, y=100.0, section_id=1),
            _row(2, f"DOI: {DOI}", y=140.0, section_id=1),
            _row(
                3,
                "Households were asked about their losses.",
                y=340.0,
                label="abstract",
                section_id=2,
            ),
        ],
        sections=[
            _section(0, "Root"),
            _section(
                1,
                "a r t i c l e i n f o",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 60.0, 460.0, 90.0),
            ),
            _section(
                2,
                "August 2020",
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 300.0, 460.0, 330.0),
            ),
        ],
    )

    _assert_abstains(contents, "no_record_identity")


def test_a_translation_without_a_byline_cannot_be_linked_to_the_record():
    contents = _contents(
        [
            _row(
                1,
                "Young witnesses in court: waiting rooms and participation rights",
                y=60.0,
                label="doc_title",
            ),
            _row(2, "Clara Example, Maria Sample", y=100.0),
            _row(3, "Court waiting rooms were designed by adults.", y=140.0, label="abstract"),
            _row(
                4,
                "Jovens testemunhas em tribunal: salas de espera e direitos",
                y=300.0,
                label="doc_title",
            ),
            _row(5, "As salas de espera foram desenhadas por adultos.", y=340.0, label="abstract"),
        ]
    )

    _assert_abstains(contents, "unlinked_records")


def test_an_agreement_error_abstains_as_before(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(front_matter, "_select_agreeing_record", broken)
    with caplog.at_level("WARNING", logger="bibr.extract.front_matter"):
        resolution, issues = resolve_front_matter(_cover_page_and_article(), target_required=True)

    assert resolution.selected_block_id is None
    assert resolution.selection_method == "abstained"
    assert "multiple_plausible_blocks" in resolution.reason_flags
    assert "record_agreement_error" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]
    assert "record agreement failed" in caplog.text


@pytest.mark.parametrize(
    ("text", "tail"),
    [
        (
            "NEIGHBOURHOOD WALKING GROUPS IN LATER LIFE Morgan Example, Mei Lin",
            "Morgan Example, Mei Lin",
        ),
        ("NEIGHBOURHOOD WALKING GROUPS IN LATER LIFE", None),
        ("INTRODUCTORY REMARKS ON 6r COLIFORM DEBRIS", None),
        ("Running head: WALKING GROUPS IN LATER LIFE", None),
        ("NEIGHBOURHOOD WALKING GROUPS (Freely available open access)", None),
    ],
)
def test_composite_name_tail_reads_names_after_an_uppercase_title(text, tail):
    assert front_matter._composite_name_tail(text) == tail


def _candidate(text: str, roles: set[str], *, label: str = "paragraph_title"):
    return front_matter.FrontMatterCandidate(
        candidate_id="c1",
        source_kind="heading",
        reading_order=0,
        page=1,
        bbox=None,
        region_label=label,
        font_size=None,
        font_bold=None,
        section_id=1,
        text_ids=(),
        paragraph_id=None,
        raw_text=text,
        normalized_text=front_matter._normalize_text(text),
        roles=frozenset(roles),
    )


@pytest.mark.parametrize(
    ("text", "roles", "label", "detected", "identity"),
    [
        # An institution line is not a title, unless layout or the parser says so.
        (
            "Mental Health of University Students",
            {"title", "affiliation"},
            "paragraph_title",
            None,
            False,
        ),
        ("Mental Health of University Students", {"title", "affiliation"}, "doc_title", None, True),
        (
            "Mental Health of University Students",
            {"title", "affiliation"},
            "paragraph_title",
            "Mental health of university students",
            True,
        ),
        (
            "UNIVERSITY OF EXAMPLE INSTITUTE OF SAMPLES",
            {"title", "affiliation"},
            "text",
            None,
            False,
        ),
        ("2. MATERIALS AND METHODS", {"title"}, "paragraph_title", None, False),
        ("CITATION", {"title"}, "paragraph_title", None, False),
        ("Yan Example: y.example@example.ac.uk", {"title"}, "doc_title", None, False),
        ("城市老年人孤独感与社会支持研究", {"title"}, "paragraph_title", None, True),
    ],
)
def test_identity_titles_exclude_furniture_but_trust_layout(text, roles, label, detected, identity):
    candidate = _candidate(text, roles, label=label)

    assert front_matter._is_identity_title(candidate, detected) is identity


@pytest.mark.parametrize(
    ("title", "language"),
    [
        ("Μελέτη της εργαζόμενης μνήμης σε ηλικιωμένους", "greek"),
        ("城市老年人孤独感与社会支持研究", "cjk"),
        ("دراسة الذاكرة العاملة لدى كبار السن", "arabic"),
        ("מחקר על זיכרון עבודה אצל מבוגרים", "hebrew"),
    ],
)
def test_title_language_names_other_scripts(title, language):
    assert front_matter._title_language(title) == language


def test_pathological_rows_stay_fast():
    started = time.perf_counter()
    front_matter._URL_OR_EMAIL_RE.search("城市老年人孤独感" * 8_000)
    words = "social anxiety in adolescents with autism and loneliness among older adults "
    long_title = "SURVEY OF SOCIAL CONTACT " + words * 250
    contents = _contents(
        [
            _row(1, long_title, y=60.0, label="doc_title"),
            _row(2, "Mei Lin, Jordan Smith", y=100.0),
            _row(3, "We surveyed retirees.", y=140.0, label="abstract"),
            _row(4, long_title + " again", y=300.0, label="doc_title"),
            _row(5, "Anna Example, Ben Sample", y=340.0),
            _row(6, "Retirees kept diaries.", y=380.0, label="abstract"),
        ]
    )
    resolve_front_matter(contents, target_required=True)

    # Quadratic, these took seconds; bounded, they are instant.
    assert time.perf_counter() - started < 1.0


# One record beside bare headings: the grouping split on a heading whose own
# capitals or punctuation read as a byline, so the page abstained where the
# same page with the heading folded in selects its only block.

COMMITTEE_TITLE = (
    "ETHICS AND EPIDEMIOLOGY Which Ethical Questions Do Cohort Researchers Meet? "
    "Results of a Questionnaire for Members of a National Cohort Committee"
)
# The heading of the committee's member list, printed pages after the title.
MEMBER_LIST_HEADING = (
    "*NATIONAL COHORT COMMITTEE ON EVALUATION OF LIFESTYLE FACTORS FOR DISEASE "
    "LARGE-SCALE COHORT STUDY AS OF 1993"
)


def _group_authored_record(*, second_abstract: bool = False) -> PaperContents:
    """A title with a group byline no row types as one, and a member list later on."""

    rows = [
        _row(1, "Working Group on Ethical Questions", y=150.0, section_id=1),
        _row(
            2,
            "In 1993 a questionnaire on ethical questions was mailed to the committee members.",
            y=200.0,
            label="abstract",
            section_id=2,
        ),
        _row(3, "Members of the committee are listed below.", y=120.0, section_id=3, page=5),
    ]
    if second_abstract:
        rows.append(
            _row(
                4,
                "Members answered a second questionnaire on consent for blood samples.",
                y=160.0,
                label="abstract",
                section_id=3,
                page=5,
            )
        )
    return _contents(
        rows,
        sections=[
            _section(0, "Root"),
            _section(
                1,
                COMMITTEE_TITLE,
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 60.0, 460.0, 120.0),
            ),
            _section(
                2,
                "Abstract",
                section_type=CanonicalSection.ABSTRACT,
                bbox=(60.0, 180.0, 460.0, 195.0),
            ),
            _section(
                3,
                MEMBER_LIST_HEADING,
                section_type=CanonicalSection.TITLE,
                bbox=(60.0, 60.0, 460.0, 90.0),
                page=5,
            ),
        ],
        detected_title=COMMITTEE_TITLE + "*",
    )


def test_a_member_list_heading_does_not_compete_with_the_only_record():
    # The record holds the parser's title and the abstract but no typed
    # byline; the member-list heading holds nothing a record prints.
    contents = _group_authored_record()
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selection_method == "record_agreement"
    assert resolution.selected_block_id == "front-matter-block-1"
    assert "lone_record:front-matter-block-1" in resolution.reason_flags
    assert "attached_block:front-matter-block-2" in resolution.reason_flags
    assert "Working Group on Ethical Questions" in _selected_text(resolution)
    assert issues == ()


def test_a_body_heading_does_not_compete_with_a_record_without_abstract():
    # An old article: title, author line, then the first body heading, whose
    # capitals and "AND" read as a byline. No abstract or DOI is printed.
    contents = _contents(
        [
            _row(1, "A STUDY OF THE COLIFORM GROUP OF BACILLI", y=60.0, label="doc_title"),
            _row(2, "By Alma J. Example and Bruno Sample.", y=100.0),
            _row(
                3,
                "INTRODUCTORY DISCUSSION: THE CLASSIFICATION OF COLIFORM BACILLI AND THEIR "
                "RELATIONS TO OTHER GRAM-NEGATIVE AEROBIC BACILLI",
                y=160.0,
            ),
            _row(4, "In 1885 a Gram-negative intestinal bacillus was first isolated.", y=220.0),
        ]
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert "lone_record:front-matter-block-1" in resolution.reason_flags
    assert issues == ()


def _old_article(*, page: int, text_id: int = 11, y: float = 60.0) -> list[PaperSentence]:
    """An old article's title, author line and first body heading, without an abstract."""

    return [
        _row(
            text_id, "A STUDY OF THE COLIFORM GROUP OF BACILLI", y=y, label="doc_title", page=page
        ),
        _row(text_id + 1, "By Alma J. Example and Bruno Sample.", y=y + 40.0, page=page),
        _row(
            text_id + 2,
            "INTRODUCTORY DISCUSSION: THE CLASSIFICATION OF COLIFORM BACILLI AND THEIR "
            "RELATIONS TO OTHER GRAM-NEGATIVE AEROBIC BACILLI",
            y=y + 100.0,
            page=page,
        ),
        _row(
            text_id + 3,
            "In 1885 a Gram-negative intestinal bacillus was first isolated.",
            y=y + 160.0,
            page=page,
        ),
    ]


# A publisher's download cover ahead of a scanned article's title page.
DOWNLOAD_COVER = [
    _row(1, "This article was downloaded by: [Example University Library]", y=40.0),
    _row(
        2,
        "On: 05 January 2015, At: 15:38 Publisher: Example & Sons Ltd Registered office: "
        "1 Example Street, London W1 3JH, UK",
        y=80.0,
    ),
    _row(3, "PLEASE SCROLL DOWN FOR ARTICLE", y=140.0, label="paragraph_title"),
    _row(
        4,
        "Example & Sons makes every effort to ensure the accuracy of all the information "
        "contained in the publications on our platform.",
        y=200.0,
    ),
]


def test_a_lone_record_is_read_without_the_download_cover_ahead_of_it():
    # The cover's stamp, disclaimer and registered office would read as the
    # article's date, abstract and affiliation.
    contents = _contents([*DOWNLOAD_COVER, *_old_article(page=2)])
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert "lone_record:front-matter-block-1" in resolution.reason_flags
    assert "cover_page_dropped:1" in resolution.reason_flags
    assert resolution.allowed_text_ids == frozenset({11, 12})
    selected = _selected_text(resolution)
    assert "A STUDY OF THE COLIFORM GROUP" in selected and "By Alma J. Example" in selected
    assert "Example & Sons" not in selected and "downloaded" not in selected
    assert issues == ()


def test_a_lone_record_keeps_a_cover_cue_printed_on_its_own_title_page():
    contents = _contents(
        [
            _row(1, "Downloaded from https://archive.example.org/item/1921", y=20.0),
            *_old_article(page=1),
        ]
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert "lone_record:front-matter-block-1" in resolution.reason_flags
    assert not any(flag.startswith("cover_page_dropped:") for flag in resolution.reason_flags)
    assert resolution.allowed_text_ids == frozenset({1, 11, 12})
    assert issues == ()


def test_a_second_block_with_an_abstract_of_its_own_still_competes():
    # The later block prints a title and abstract in the record's language, so
    # it may be another item whose byline went unseen: the page keeps abstaining.
    _assert_abstains(
        _group_authored_record(second_abstract=True),
        "unmatched_presentation:front-matter-block-2",
    )


@pytest.mark.parametrize(
    "second_block",
    [
        # A later heading with a byline under it: a second item's byline.
        [
            _row(13, "3. ANTIGENIC ANALYSIS OF THE STRAINS", y=160.0),
            _row(14, "By Carl D. Other and Dana E. Sample.", y=200.0),
            _row(15, "The agglutination tests were read after two hours.", y=240.0),
        ],
        # A later heading with a cover line under it: a repository's landing page.
        [
            _row(
                13,
                "INTRODUCTORY DISCUSSION: THE CLASSIFICATION OF COLIFORM BACILLI AND THEIR "
                "RELATIONS TO OTHER GRAM-NEGATIVE AEROBIC BACILLI",
                y=160.0,
            ),
            _row(14, "Downloaded from https://archive.example.org/item/1921", y=200.0),
        ],
    ],
    ids=["byline", "cover-line"],
)
def test_a_second_block_printing_a_byline_or_a_cover_line_is_not_bare(second_block):
    contents = _contents([*_old_article(page=1)[:2], *second_block])
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert "incomplete_record" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_two_linked_records_without_a_byline_are_no_lone_record():
    # A title and its translation share a DOI, and neither prints a byline.
    _assert_abstains(
        _contents(
            [
                _row(
                    1,
                    "Delayed recall of household losses after regional flooding",
                    y=60.0,
                    label="doc_title",
                ),
                _row(2, "DOI: 10.1000/recall.1991.0410", y=100.0),
                _row(
                    3,
                    "Recordação tardia das perdas das famílias após as inundações em uma região",
                    y=60.0,
                    label="doc_title",
                    page=2,
                ),
                _row(4, "DOI: 10.1000/recall.1991.0410", y=100.0, page=2),
            ]
        ),
        "incomplete_record",
    )


def test_four_complete_records_on_one_page_keep_abstaining():
    """A compiled abstract book page: each title carries its own names and abstract."""

    records = [
        (
            "I HAVE TO COPE WITH IT: THE VOICES OF OLDER MIGRANTS EXPERIENCING ISOLATION",
            "Dana Example1, and Gil Sample2",
        ),
        (
            "A NEW NORMAL: EXAMINING THE LINKS BETWEEN SOCIAL TECHNOLOGY AND LONELINESS",
            "Yun Reader1, and Nora Writer1",
        ),
        (
            "COMMUNITY OUTCOMES FROM A PSYCHOSOCIAL INTERVENTION FOR LONELY OLDER ADULTS",
            "Max Author, and Mia Editor",
        ),
        (
            "COMPARING DATA-DRIVEN AND THEORY-DRIVEN WAYS TO CLASSIFY SOCIAL RELATIONSHIPS",
            "Eli Coder1, and Kim Tester2",
        ),
    ]
    rows = []
    for index, (title, names) in enumerate(records):
        rows += [
            _row(10 * index + 1, f"{title} {names}, 1. Example University", y=60.0 + 150 * index),
            _row(
                10 * index + 2,
                f"Participants in study {index + 1} reported their social contacts.",
                y=110.0 + 150 * index,
                label="abstract",
            ),
        ]
    contents = _contents(rows)
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id is None
    assert not any(flag.startswith("lone_record:") for flag in resolution.reason_flags)
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


# A translated title and abstract under a layout title of their own, with no
# byline or DOI: the figures of a translated abstract are the original's.

SIGN_TITLE = "Cross-cultural adaptation of an occupational self-assessment for sign language"
SIGN_ABSTRACT = (
    "Objective: to adapt the instrument for sign language. Method: translation and "
    "back-translation from August 2016 to October 2017 with 24 deaf adults. Results: "
    "3 items were changed."
)
# The same abstract with only its two years for figures.
SIGN_YEARS_ABSTRACT = (
    "Objective: to adapt the instrument for sign language. Method: translation and "
    "back-translation from August 2016 to October 2017."
)
SIGN_YEARS_PT = (
    "Objetivo: adaptar o instrumento para língua de sinais. Método: tradução e "
    "retrotradução de agosto de 2016 a outubro de 2017."
)
SIGN_YEARS_ES = (
    "ADAPTACIÓN TRANSCULTURAL DE UNA AUTOEVALUACIÓN OCUPACIONAL PARA LA LENGUA DE SEÑAS",
    "Objetivo: adaptar el instrumento para la lengua de señas. Método: traducción y "
    "retrotraducción de agosto de 2016 a octubre de 2017.",
)


def _record_with_translated_abstract(
    translated_abstract: str,
    *,
    record_abstract: str = SIGN_ABSTRACT,
    record_doi: str | None = "DOI: 10.1000/sign.2019.0160",
    record_doi_label: str = "text",
    third: tuple[str, str] | None = None,
    translated_doi: str | None = None,
) -> PaperContents:
    rows = [
        _row(1, SIGN_TITLE, y=60.0, label="doc_title"),
        _row(2, "Luisa F. Example, Fabio E. Sample", y=100.0),
        _row(3, record_abstract, y=140.0, label="abstract"),
    ]
    if record_doi is not None:
        rows.append(_row(4, record_doi, y=260.0, label=record_doi_label))
    rows += [
        _row(
            5,
            "Adaptação transcultural de uma autoavaliação ocupacional para língua de sinais",
            y=60.0,
            label="doc_title",
            page=2,
        ),
        _row(6, translated_abstract, y=100.0, label="abstract", page=2),
    ]
    if translated_doi is not None:
        rows.append(_row(9, translated_doi, y=200.0, page=2))
    if third is not None:
        rows += [
            _row(7, third[0], y=300.0, label="paragraph_title", page=2),
            _row(8, third[1], y=340.0, label="abstract", page=2),
        ]
    return _contents(rows)


def test_a_translated_abstract_printing_the_records_figures_attaches():
    contents = _record_with_translated_abstract(
        "Objetivo: adaptar o instrumento para língua de sinais. Método: tradução e "
        "retrotradução de agosto de 2016 a outubro de 2017 com 24 adultos surdos. "
        "Resultados: 3 itens foram alterados."
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selection_method == "record_agreement"
    assert resolution.selected_block_id == "front-matter-block-1"
    assert "agreeing_record:front-matter-block-2:abstract_figures" in resolution.reason_flags
    assert issues == ()


def test_two_years_link_translations_only_when_a_third_language_prints_them():
    # A trilingual article: English, Portuguese and Spanish abstracts all print
    # the study's two years and nothing else. The record's citation line, with
    # its year, volume and DOI, is no abstract figure.
    contents = _record_with_translated_abstract(
        SIGN_YEARS_PT,
        record_abstract=SIGN_YEARS_ABSTRACT,
        record_doi=(
            "HOW CITED: Example LF, Sample FE. Cross-cultural adaptation of an occupational "
            "self-assessment. Example Nursing [Internet]. 2019; 28: e20990001. "
            "Available from: https://doi.org/10.1000/sign.2019.0160"
        ),
        record_doi_label="abstract",
        third=SIGN_YEARS_ES,
    )
    assert _dominance_abstains(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selected_block_id == "front-matter-block-1"
    assert "agreeing_record:front-matter-block-2:abstract_figures" in resolution.reason_flags
    assert issues == ()


@pytest.mark.parametrize(
    "third",
    [
        # A third abstract in the translation's language is no third language.
        (
            "ADAPTAÇÃO DE UM INSTRUMENTO DE AUTOAVALIAÇÃO OCUPACIONAL PARA SURDOS",
            "Objetivo: adaptar o instrumento para surdos. Método: estudo de agosto de 2016 "
            "a outubro de 2017.",
        ),
        # A third language printing other years echoes nothing.
        (
            SIGN_YEARS_ES[0],
            "Objetivo: adaptar el instrumento para la lengua de señas. Método: traducción de "
            "2014 a 2015.",
        ),
    ],
    ids=["same-language", "other-years"],
)
def test_two_years_stay_unlinked_without_a_third_language_printing_them(third):
    _assert_abstains(
        _record_with_translated_abstract(
            SIGN_YEARS_PT, record_abstract=SIGN_YEARS_ABSTRACT, third=third
        ),
        "unlinked_records",
    )


@pytest.mark.parametrize(
    ("translated_abstract", "record_abstract"),
    [
        # Other figures: another paper whose byline went unseen.
        (
            "Objetivo: avaliar a adesão ao tratamento. Método: estudo de 2018 a 2019 com 40 "
            "pacientes. Resultados: 12 abandonaram.",
            SIGN_ABSTRACT,
        ),
        # One figure the record prints among others.
        ("Objetivo: adaptar o instrumento. Método: estudo iniciado em 2016.", SIGN_ABSTRACT),
        # Some of the record's figures, not all of them.
        (SIGN_YEARS_PT, SIGN_ABSTRACT),
        # One figure, and the record's only one.
        (
            "Objetivo: adaptar o instrumento com 24 adultos surdos.",
            "Objective: to adapt the instrument with 24 deaf adults.",
        ),
        # Some of the record's figures, even three of them.
        (
            "Objetivo: adaptar o instrumento. Método: de agosto de 2016 a outubro de 2017 "
            "com 24 adultos surdos.",
            SIGN_ABSTRACT.replace("3 items were changed", "312 answers were read"),
        ),
        # Two years alone, in two languages: many papers of one period print them.
        (SIGN_YEARS_PT, SIGN_YEARS_ABSTRACT),
        # A year and a disease name: "19" is part of a name, not a figure.
        (
            "Objetivo: descrever a adesão durante a COVID-19. Método: estudo de 2020.",
            "Objective: to describe burnout during COVID-19. Method: a survey in 2020.",
        ),
        # List numbers are no figures.
        (
            "Objetivos: (1) descrever o programa; (2) comparar escolas; (3) testar uma escala.",
            "Aims: (1) to describe burnout; (2) to compare schools; (3) to test a scale.",
        ),
    ],
    ids=[
        "other-figures",
        "one-of-several",
        "some-not-all",
        "one-figure",
        "three-of-four",
        "two-years-two-languages",
        "name-digits",
        "list-numbers",
    ],
)
def test_a_translated_abstract_without_the_records_distinctive_figures_stays_unlinked(
    translated_abstract, record_abstract
):
    _assert_abstains(
        _record_with_translated_abstract(translated_abstract, record_abstract=record_abstract),
        "unlinked_records",
    )


def test_a_translation_printing_a_doi_of_its_own_is_not_linked_by_figures():
    # The record prints no DOI; the other item does, so it is another paper.
    _assert_abstains(
        _record_with_translated_abstract(
            "Objetivo: adaptar o instrumento para língua de sinais. Método: tradução e "
            "retrotradução de agosto de 2016 a outubro de 2017 com 24 adultos surdos.",
            record_doi=None,
            translated_doi="DOI: 10.1000/sign.2019.0161",
        ),
        "unlinked_records",
    )


def test_only_a_byline_less_doi_less_translation_links_to_a_record_with_a_byline():
    figures = frozenset({"2016", "2017", "24"})
    record = replace(
        _identity(SIGN_TITLE, language="en", surnames=frozenset({"example"})), figures=figures
    )
    translation = replace(
        _identity("Adaptação transcultural", language="pt"), byline=False, figures=figures
    )
    links = front_matter._translates_abstract
    assert links(translation, record, echoed_figures=frozenset())

    # A byline or a DOI of its own makes it an item of its own.
    assert not links(replace(translation, byline=True), record, echoed_figures=frozenset())
    assert not links(
        replace(translation, dois=frozenset({"10.1000/other"})), record, echoed_figures=frozenset()
    )
    # A record without a byline is no complete record to translate.
    assert not links(translation, replace(record, byline=False), echoed_figures=frozenset())


def test_figures_ignore_list_numbers_and_digits_glued_to_words_and_read_decimal_commas():
    identity = front_matter._record_identity(
        front_matter.FrontMatterBlock(block_id="b", candidate_ids=("c1",), title_candidate_ids=()),
        0,
        {
            "c1": _candidate(
                "Example1,2 reported 20,5% of 39 COVID-19 cases in 1993 (e20990001): "
                "(1) mild, 2) severe, 7 fatal.",
                {"abstract"},
                label="abstract",
            )
        },
        detected_title=None,
    )

    assert identity.figures == frozenset({"20.5", "39", "1993"})


# An uppercase title stacked above its translation: the punctuation rule for
# proceedings rows ("TITLE Name, Name") gave the upper title a byline role, so
# it rooted a block of its own and split the page's record in two.

PARALLEL_PT = (
    "PLANEJAMENTO COLABORATIVO DO VERDE URBANO NA CIDADE CONECTADA: ESTUDO DAS "
    "POSSIBILIDADES EM SOROCABA, SÃO PAULO"
)
PARALLEL_EN = (
    "COLLABORATIVE PLANNING OF URBAN GREEN IN THE CONNECTED CITY: A STUDY OF THE "
    "POSSIBILITIES IN SOROCABA, SÃO PAULO"
)


def _stacked_titles(upper: str, lower: str = PARALLEL_EN) -> PaperContents:
    """Page 1: two stacked titles and a byline; page 2: the title again and an abstract."""

    return _contents(
        [
            _row(1, upper, y=60.0, label="paragraph_title"),
            _row(2, lower, y=110.0, label="paragraph_title"),
            _row(3, "Regina Maria Exemplo", y=160.0, label="paragraph_title"),
            _row(4, "Doutoranda em Gestão Urbana, Universidade Exemplo (Brasil).", y=200.0),
            _row(5, PARALLEL_PT, y=60.0, label="doc_title", page=2),
            _row(6, "RESUMO", y=110.0, label="paragraph_title", page=2),
            _row(
                7,
                "As tecnologias digitais oferecem alternativas de inclusão dos cidadãos.",
                y=140.0,
                label="abstract",
                page=2,
            ),
        ]
    )


def _blocks(contents: PaperContents):
    return front_matter.group_front_matter_blocks(
        front_matter.collect_front_matter_candidates(contents)
    )


def test_a_title_stacked_above_its_translation_joins_the_record_below():
    contents = _stacked_titles(PARALLEL_PT)

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert len(_blocks(contents)) == 2
    assert resolution.selection_method == "record_agreement"
    assert resolution.selected_block_id == "front-matter-block-1"
    assert "agreeing_record:front-matter-block-2:title" in resolution.reason_flags
    selected = _selected_text(resolution)
    assert PARALLEL_PT in selected and PARALLEL_EN in selected
    assert issues == ()


@pytest.mark.parametrize(
    "upper",
    [
        # Two titles in one language are two titles, not a translation.
        (
            "URBAN LANDSCAPE PLANNING IN THE DIGITAL CITY: A REVIEW OF PARTICIPATORY TOOLS "
            "FOR CITIZEN INCLUSION, PARTICIPATION AND DESIGN"
        ),
        # A proceedings row prints its names after the title.
        (
            "PLANEJAMENTO COLABORATIVO DO VERDE URBANO NA CIDADE CONECTADA: ESTUDO DAS "
            "POSSIBILIDADES Regina Exemplo, Ana Amostra"
        ),
    ],
)
def test_an_upper_title_keeps_its_own_block_unless_it_is_a_nameless_translation(upper):
    assert len(_blocks(_stacked_titles(upper))) == 3


def _repaged(row: PaperSentence, page: int) -> PaperSentence:
    return replace(
        row,
        page_number=page,
        provenance=[replace(item, page_no=page) for item in row.provenance],
    )


def test_an_upper_title_keeps_its_own_block_on_another_page_or_over_a_row():
    rows = list(_stacked_titles(PARALLEL_PT).sentences)
    # The translation and byline start the next page.
    other_page = [rows[0], *(_repaged(row, 2) for row in rows[1:4]), *rows[4:]]
    # A row set between the upper title and its translation.
    between = [
        rows[0],
        _row(9, "Recebido em 12 de março de 2020 e aceito em 5 de junho de 2020", y=85.0),
        *rows[1:],
    ]

    assert len(_blocks(_contents(other_page))) == 3
    assert len(_blocks(_contents(between))) == 3


def test_a_parallel_title_holds_a_byline_role_and_no_other_record_role():
    upper = _candidate(PARALLEL_PT, {"title", "byline", "heading"})
    lower = _candidate(PARALLEL_EN, {"title", "byline", "heading"})
    assert front_matter._is_parallel_title_above(upper, lower)

    assert not front_matter._is_parallel_title_above(
        replace(upper, roles=frozenset({"title", "heading"})), lower
    )
    for role in ("abstract", "affiliation", "doi"):
        assert not front_matter._is_parallel_title_above(
            replace(upper, roles=upper.roles | {role}), lower
        )
    assert not front_matter._is_parallel_title_above(upper, replace(lower, page=2))


# The author's name alone on a row above the title: no initial, no affiliation
# mark, so no byline shape. The page's copyright line names the same person.

NAMED_TITLE = "ПЕРШІ НАРАДИ ДОСЛІДНИКІВ І ПРИКЛАДНА НАУКА (до ювілею Першої Міжнародної Наради)"
NAMED_DOI = "10.1000/example.2099.0001"


def _record_under_a_bare_name(name: str, *, holder: str = "О. Приклад") -> PaperContents:
    return _contents(
        [
            _row(1, name, y=60.0),
            _row(2, NAMED_TITLE, y=100.0, label="doc_title"),
            _row(3, f"DOI: {NAMED_DOI}\r\n© {holder}, 2024. CC BY 4.0", y=160.0),
            _row(
                4,
                "Останній конгрес перед війною відбувся в Римі, і українські делегати "
                "представили на ньому свої доповіді.",
                y=200.0,
                label="abstract",
            ),
            _row(
                5,
                "FIRST MEETINGS OF RESEARCHERS AND APPLIED SCIENCE (TO THE JUBILEE OF THE "
                "FIRST INTERNATIONAL MEETING)",
                y=60.0,
                label="paragraph_title",
                page=4,
            ),
            _row(
                6,
                "Key words: international meetings, applied science, Ivan Sample, Petro Example",
                y=120.0,
                page=4,
            ),
            _row(
                7,
                "Pryklad, O. (2024). Pershi narady doslidnykiv i prykladna nauka. "
                f"https://doi.org/{NAMED_DOI}",
                y=160.0,
                page=4,
            ),
        ],
        detected_title=NAMED_TITLE,
    )


def test_a_bare_name_row_the_copyright_line_confirms_is_the_byline():
    contents = _record_under_a_bare_name("Олена Приклад")

    resolution, issues = resolve_front_matter(contents, target_required=True)

    assert resolution.selection_method == "record_agreement"
    assert resolution.selected_block_id == "front-matter-block-1"
    assert "agreeing_record:front-matter-block-2:doi" in resolution.reason_flags
    assert "Олена Приклад" in _selected_text(resolution)
    assert issues == ()


@pytest.mark.parametrize(
    ("name", "holder"),
    [
        # The copyright line names someone else: the row is no byline.
        ("Олена Приклад", "О. Зразок"),
        # A publisher holds the copyright, not a person.
        ("Robert Francis", "Taylor & Francis"),
        # One capitalized word is no name.
        ("Приклад", "О. Приклад"),
        # A row of lowercase words is prose, not a name.
        ("Олена приклад", "О. Приклад"),
    ],
)
def test_a_bare_row_the_copyright_line_does_not_name_stays_out(name, holder):
    resolution, issues = resolve_front_matter(
        _record_under_a_bare_name(name, holder=holder), target_required=True
    )

    assert resolution.selected_block_id is None
    assert "record_without_byline:front-matter-block-1" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_bare_name_on_another_page_than_the_title_stays_out():
    rows = list(_record_under_a_bare_name("Олена Приклад").sentences)
    # The name row moves under the abstract, onto the next page.
    moved = [*rows[1:4], _repaged(replace(rows[0], text_id=8), 2), *rows[4:]]

    resolution, issues = resolve_front_matter(
        _contents(moved, detected_title=NAMED_TITLE), target_required=True
    )

    assert resolution.selected_block_id is None
    assert "record_without_byline:front-matter-block-1" in resolution.reason_flags


@pytest.mark.parametrize(
    "name_row",
    [
        # A row that already holds a role is not a bare name.
        replace(_candidate("Олена Приклад", set()), roles=frozenset({"heading"})),
        # A digit marks a date, a page or an affiliation, not a bare name.
        _candidate("Олена Приклад2", set()),
        # Every surname must be the holder's, not one of them.
        _candidate("Олена Приклад, Іван Зразок", set()),
        # A lowercase word makes the row prose, not a list of names.
        _candidate("Олена Приклад, та інші", set()),
    ],
    ids=["role", "digit", "extra-surname", "lowercase-word"],
)
def test_a_copyright_line_confirms_only_a_bare_row_of_its_holders_names(name_row):
    copyright_row = _candidate(f"DOI: {NAMED_DOI}\r\n© О. Приклад, 2024. CC BY 4.0", set())

    assert front_matter._copyright_named_row((_candidate("Олена Приклад", set()), copyright_row))
    assert front_matter._copyright_named_row((name_row, copyright_row)) is None
