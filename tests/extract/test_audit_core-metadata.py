"""Core metadata on Turkish and long-s case variants, and affiliation cleaning cost.

An ignore-case pattern matches "Unıversity", "UNİVERSİTY", "ſcience" and
"APRİL", which lower() and casefold() keep apart from the ASCII keys of the
tables they are looked up in; the KeyError failed the paper's whole core
metadata. The affiliation tail patterns retried long runs from every position
inside them, and trailing punctuation and connectives were stripped one turn
at a time, which took seconds on a few thousand characters.
"""

import time
from unittest import mock

import pandas as pd
import pytest

from bibr.extract.core_metadata import (
    CoreMetadataExtractor,
    _clean_affiliation_value,
    _grounding_key,
)
from bibr.extract.front_matter import FrontMatterBlock, FrontMatterCandidate, FrontMatterResolution
from bibr.extract.metadata_precision import refine_publication_date
from bibr.paper import PaperAuthor
from bibr.paper_contents import PaperContents
from bibr.schemas import AuthorLLM, CoreMetadataLLM


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Hacettepe Unıversity", "hacettepeuniv"),
        ("HACETTEPE UNİVERSİTY", "hacettepeuniv"),
        ("Faculty of ſciences", "facultyofsci"),
        ("Natıonal Instıtute", "natlinst"),
        ("Hacettepe University", "hacettepeuniv"),
    ],
)
def test_case_variants_fold_like_their_ascii_spelling(text, key):
    assert _grounding_key(text) == key


@pytest.mark.parametrize(
    ("affiliation", "printed"),
    [
        # A Turkish journal name anywhere in the paper's text.
        ("Dept, Hacettepe University", "Hacettepe Unıversity Journal"),
        # The paper prints the dotless spelling the LLM corrected.
        (
            "Department of Physics, Hacettepe University, Ankara, Turkey",
            "Department of Physics, Hacettepe Unıversity, Ankara, Turkey",
        ),
    ],
)
def test_a_dotless_i_in_the_paper_still_grounds_the_affiliation(affiliation, printed):
    authors = [PaperAuthor(author_id=1, given="Ayse", family="Yilmaz", affiliation=affiliation)]

    dropped = CoreMetadataExtractor._normalize_author_affiliations(
        authors, pd.DataFrame({"text": [printed]})
    )

    assert dropped == []
    assert authors[0].affiliation == affiliation


@pytest.mark.parametrize(
    ("title", "kind"),
    [("CORRECTİON: A study of tides", "corrigendum"), ("Retractıon: A study", "retraction")],
)
def test_correction_notice_title_with_turkish_i(title, kind):
    assert CoreMetadataExtractor._apply_correction_notice_guard(title) == (True, kind)


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("Published: 12 APRİL 2020", "2020-04-12"),
        ("Published: 12 Aprıl 2020", "2020-04-12"),
        ("Published: 3 Auguſt 2020", "2020-08-03"),
        ("Received: 3 APRİL 2020; Published: 12 May 2020", "2020-05-12"),
    ],
)
def test_month_names_with_turkish_i_or_long_s(printed, expected):
    assert refine_publication_date("2020", printed) == expected


def _extractor(texts: list[str], llm_result: CoreMetadataLLM) -> CoreMetadataExtractor:
    candidates = tuple(
        FrontMatterCandidate(
            candidate_id=f"c{i}",
            source_kind="paragraph",
            reading_order=i,
            page=1,
            bbox=None,
            region_label="text",
            font_size=None,
            font_bold=None,
            section_id=0,
            text_ids=(i,),
            paragraph_id=i,
            raw_text=text,
            normalized_text=text.casefold(),
            roles=frozenset({"title"} if i == 1 else {"byline"} if i == 2 else ()),
        )
        for i, text in enumerate(texts, 1)
    )
    ids = tuple(candidate.candidate_id for candidate in candidates)
    resolution = FrontMatterResolution(
        candidates=candidates,
        blocks=(FrontMatterBlock("selected", ids, ("c1",)),),
        selected_block_id="selected",
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=frozenset(range(1, len(texts) + 1)),
        allowed_section_ids=frozenset({0}),
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {
            "text_id": list(range(1, len(texts) + 1)),
            "text": texts,
            "page_number": [1] * len(texts),
            "section_name": ["Front matter"] * len(texts),
        }
    )
    contents.sentences = []
    contents.sections = []
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.processing_warnings = []
    client = mock.MagicMock()
    client.extract_core_metadata = mock.AsyncMock(return_value=llm_result)
    return CoreMetadataExtractor(contents, llm_client=client, front_matter_resolution=resolution)


async def test_extract_keeps_core_metadata_of_a_turkish_paper():
    texts = [
        "Measuring tides with acoustic sensors",
        "Ayse Yilmaz",
        "Department of Physics, Hacettepe Unıversity, Ankara, Turkey",
        "Received: 3 OCAK 2020 / Accepted: 2 APRİL 2020 / Published: 11 APRİL 2020",
    ]
    llm_result = CoreMetadataLLM(
        title=texts[0],
        keywords=[],
        published="2020",
        authors=[
            AuthorLLM(
                given="Ayse",
                family="Yilmaz",
                affiliation="Department of Physics, Hacettepe University, Ankara, Turkey",
            )
        ],
    )

    metadata = await _extractor(texts, llm_result).extract()

    assert metadata.title == texts[0]
    assert metadata.published == "2020-04-11"
    assert [author.affiliation for author in metadata.authors] == [
        "Department of Physics, Hacettepe University, Ankara, Turkey"
    ]


# Each input took the old code 15 s or more (cubic or quadratic in the run);
# it now takes a few milliseconds.
@pytest.mark.parametrize(
    ("value", "clean"),
    [
        (
            "Department of Physics, " + "@." * 1200,
            "Department of Physics, " + "@." * 1199 + "@",
        ),
        ("Department of Physics, " + "a@" * 2000, None),
        ("Department of Physics " + "," * 3000 + "x", None),
        ("Department of Physics" + ", " * 10000 + "x", None),
        ("Department of Physics" + "." * 3000 + "x", None),
        ("Department of Physics" + ", e" * 16000 + "x", None),
        ("Department of Physics" + ";e." * 10000, "Department of Physics"),
    ],
    ids=["at-dot", "a-at", "commas", "comma-space", "dots", "connectives", "alternating-tail"],
)
def test_affiliation_cleaning_is_linear_in_long_runs(value, clean):
    started = time.perf_counter()
    cleaned = _clean_affiliation_value(value)
    elapsed = time.perf_counter() - started

    assert cleaned == (value if clean is None else clean)
    assert elapsed < 1.0


@pytest.mark.parametrize(
    ("value", "clean"),
    [
        (
            "Department of Physics, Example University, Oslo,jane@example.no",
            "Department of Physics, Example University",
        ),
        # The whole word goes, however many "@" it holds.
        ("Example University, Oslo, a@@b.org", "Example University, Oslo"),
        ("Example University, Oslo, x@y@z.org", "Example University, Oslo"),
        ("Example University, Oslo, @examplelab", "Example University, Oslo, @examplelab"),
        ("Example University, Oslo, user@localhost", "Example University, Oslo, user@localhost"),
        (
            "Example Institute, Washington, D.C., U.S.A.",
            "Example Institute, Washington, D.C., U.S.A.",
        ),
        ("Example Institute, U.S.A.; and", "Example Institute, U.S.A."),
        ("and Example Institute, Houston, Texas, et", "Example Institute, Houston, Texas"),
        ("Example Institute, Houston, Texas; and ; e", "Example Institute, Houston, Texas"),
        ("Example Institute, Houston, Texas; and.", "Example Institute, Houston, Texas"),
        ("Example Institute, Houston, Texas, et |", "Example Institute, Houston, Texas"),
        ("Universidad de Example y", "Universidad de Example"),
        ("Example Institute, Paris, France, et al", "Example Institute, Paris, France, et al"),
        (
            "Example Institute. Extended author information available on the last page.",
            "Example Institute",
        ),
    ],
)
def test_affiliation_cleaning_cuts_what_it_cut_before(value, clean):
    assert _clean_affiliation_value(value) == clean
