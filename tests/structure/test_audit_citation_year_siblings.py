"""One author, several years: a linked year must not hide an unresolved sibling year."""

import pytest

from bibr.paper import PaperReference
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence, PaperXref
from bibr.structure import citation_linker

SECTIONS = [
    PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
    PaperSection(
        section_id=1,
        header="Intro",
        level=1,
        parent_section_id=0,
        section_type=CanonicalSection.INTRODUCTION,
    ),
]


def _ref(bib_id, authors, year):
    return PaperReference(
        bib_id=bib_id,
        title=f"Title {bib_id}",
        authors=authors,
        year=year,
        first_page=None,
        volume=None,
        container=None,
    )


# Smith 2020 is unique; Smith 2021 has two candidates, so Tier 2 cannot pick it.
REFERENCES = [
    _ref(1, "Smith, J.", 2020),
    _ref(2, "Smith, J.", 2021),
    _ref(3, "Smith, K.", 2021),
]


def _fake_llm(picks, asked):
    async def resolve(questions, references, llm_client, file_hash, failures):
        asked.extend(questions)
        return [
            PaperXref(xref_id=bib_id, xref_type="bib", contents=cite_text, text_id=text_id)
            for text_id, cite_text in questions
            for bib_id in picks
        ]

    return resolve


@pytest.mark.parametrize(
    "text",
    ["Memory improves (Smith, 2020, 2021).", "Smith (2020, 2021) found it."],
)
@pytest.mark.parametrize("picks", [(2,), (1, 2)])
async def test_ambiguous_sibling_year_goes_to_tier3(monkeypatch, text, picks):
    asked = []
    monkeypatch.setattr(citation_linker, "_resolve_with_llm", _fake_llm(picks, asked))
    sentences = [PaperSentence(text_id=1, text=text, section_id=1, paragraph_id=1)]

    xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        sentences, SECTIONS, REFERENCES, llm_client=object()
    )

    assert len(asked) == 1
    assert sorted(x.xref_id for x in xrefs) == [1, 2]
    assert sorted(c.bib_ids for c in receipt.candidates if c.accepted) == [(1,), (2,)]


async def test_rules_tier_keeps_the_unresolved_sibling_on_the_receipt():
    sentences = [
        PaperSentence(
            text_id=1, text="Memory improves (Smith, 2020, 2021).", section_id=1, paragraph_id=1
        )
    ]

    xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        sentences, SECTIONS, REFERENCES, rule_tier3=True
    )

    assert [x.xref_id for x in xrefs] == [1]
    rejected = [c for c in receipt.candidates if not c.accepted]
    assert [(c.bib_ids, c.rejection_reasons) for c in rejected] == [
        ((2, 3), ("ambiguous_same_surname_year",))
    ]
