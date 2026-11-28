"""An institution word in abstract prose does not end the abstract span.

Cut down from a CC BY surgery article (PMC13342491): its structured abstract's
"Methods:" paragraph says "at a tertiary high-volume academic centre", the word
list behind the affiliation role matched "centre", and the span closed after
"Aim:". The extracted abstract was right, but it read as ungrounded and
cross-boundary against that one-sentence span, and the layout fallback would
have kept only "Aim:".
"""

from bibr.extract.front_matter import resolve_front_matter
from bibr.models import PaperMetadata
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
)
from bibr.pipeline.stages.post_parse import _finalize_abstract_and_keywords

TITLE = "How Safe Is Teaching Of Highly Specialized Rectal Surgery"
AIM = "Aim: This study assessed the impact of a surgical teaching program on outcomes."
METHODS = (
    "Methods: We performed a retrospective cohort study of consecutive patients undergoing "
    "rectal resection at a tertiary high-volume academic centre between 2014 and 2023."
)
PAGE_ONE = [Provenance(page_no=1)]
RESULTS = "Results: Complications occurred in sixteen percent of patients in both groups."


def _contents(abstract_rows: list[str]) -> PaperContents:
    sections = [
        PaperSection(0, "Root", 0, None),
        PaperSection(1, TITLE, 1, 0, CanonicalSection.TITLE, provenance=PAGE_ONE),
        PaperSection(2, "Abstract", 1, 0, CanonicalSection.ABSTRACT, provenance=PAGE_ONE),
        PaperSection(3, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
    ]
    sentences = [
        PaperSentence(1, "Simon Example, Luka Sample, Fabio Person", 1, 1, page_number=1),
        PaperSentence(
            2,
            "Department of Surgery, Lausanne University Hospital, Switzerland",
            1,
            2,
            page_number=1,
        ),
        *(
            PaperSentence(10 + index, row, 2, 10 + index, page_number=1)
            for index, row in enumerate(abstract_rows)
        ),
        PaperSentence(30, "Rising healthcare costs shape surgical training.", 3, 30, page_number=2),
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
        detected_title=TITLE,
    )


def _finalize(contents: PaperContents, *, abstract: str = "") -> tuple[PaperMetadata, list]:
    resolution, _ = resolve_front_matter(contents)
    metadata = PaperMetadata(doi="10.1/owned", title=TITLE, abstract=abstract)
    sink: list = []
    _finalize_abstract_and_keywords(
        contents, metadata, resolution=resolution, validation_issue_sink=sink
    )
    return metadata, sink


def test_the_layout_abstract_runs_past_an_institution_word():
    metadata, _ = _finalize(_contents([AIM, METHODS, RESULTS]))

    assert metadata.abstract == f"{AIM} {METHODS} {RESULTS}"


def test_an_extracted_abstract_with_an_institution_word_is_not_suspect():
    contents = _contents([AIM, METHODS, RESULTS])

    _, sink = _finalize(contents, abstract=f"{AIM}\n{METHODS}\n{RESULTS}")

    # The fixture is all abstract, so only the size heuristic may still fire.
    messages = " ".join(issue.message for issue in sink)
    assert "ungrounded" not in messages
    assert "cross_boundary" not in messages


def test_an_affiliation_line_still_ends_the_abstract():
    affiliation = "Department of Visceral Surgery, Lausanne University Hospital, Lausanne"

    metadata, _ = _finalize(_contents([AIM, affiliation, RESULTS]))

    assert metadata.abstract == AIM
