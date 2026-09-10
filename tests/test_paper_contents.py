import pandas as pd
import pytest

from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)


@pytest.fixture
def sample_sections():
    return [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Introduction",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        ),
        PaperSection(
            section_id=2,
            header="Background",
            level=2,
            parent_section_id=1,
            section_type=CanonicalSection.INTRODUCTION,
        ),
        PaperSection(
            section_id=3,
            header="Methodology",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        ),
    ]


@pytest.fixture
def sample_sentences():
    return [
        PaperSentence(text_id=1, text="Intro sentence 1.", section_id=1, paragraph_id=1),
        PaperSentence(text_id=2, text="Intro sentence 2.", section_id=1, paragraph_id=1),
        PaperSentence(text_id=3, text="Background info.", section_id=2, paragraph_id=2),
        PaperSentence(text_id=4, text="Method step.", section_id=3, paragraph_id=3),
    ]


@pytest.fixture
def sample_contents(sample_sections, sample_sentences):
    return PaperContents(
        sentences=sample_sentences,
        sections=sample_sections,
        tables=[],
        links=[],
        sections_text={
            1: "Intro sentence 1. Intro sentence 2.",
            2: "Background info.",
            3: "Method step.",
        },
    )


def test_paper_contents_initialization(sample_contents):
    assert len(sample_contents.sentences) == 4
    assert len(sample_contents.sections) == 4
    assert len(sample_contents.sections_text) == 3


def test_text_df_structure(sample_contents):
    df = sample_contents.text_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4

    expected_cols = [
        "text_id",
        "section_id",
        "paragraph_id",
        "text",
        "page_number",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_text_df_content(sample_contents):
    df = sample_contents.text_df
    row0 = df.iloc[0]
    assert row0["text"] == "Intro sentence 1."
    assert row0["section_id"] == 1
    assert row0["paragraph_id"] == 1

    row2 = df.iloc[2]
    assert row2["text"] == "Background info."
    assert row2["section_id"] == 2


def test_sentences_df_has_section_name_and_type(sample_contents):
    """sentences_df denormalizes section identity for extractor compatibility."""
    df = sample_contents.sentences_df
    assert "section_name" in df.columns
    assert "section_type" in df.columns
    row0 = df.iloc[0]
    assert row0["section_name"] == "Introduction"
    assert row0["section_type"] == CanonicalSection.INTRODUCTION

    row2 = df.iloc[2]
    assert row2["section_name"] == "Background"
    assert row2["section_type"] == CanonicalSection.INTRODUCTION


def test_empty_contents():
    empty_contents = PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
    )
    df = empty_contents.text_df
    assert isinstance(df, pd.DataFrame)
    assert df.empty
