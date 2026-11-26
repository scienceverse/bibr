"""Links printed in table cells, and sentence links repaired from the PDF's targets."""

from __future__ import annotations

from urllib.parse import unquote

import pandas as pd
import pytest

from bibr.paper_contents import (
    PaperContents,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperTablePart,
    PaperURLLink,
)
from bibr.structure.table_links import add_table_links, repair_links_from_targets


def _table(table_id: int, rows: list[list[str]], *, page=None, bbox=None) -> PaperTable:
    df = pd.DataFrame(rows[1:], columns=rows[0])
    part = PaperTablePart(page_number=page, bbox=bbox, tbl_html=None, df=df)
    return PaperTable(table_id, df, "", 1, parts=[part])


def _contents(*, tables=(), sentences=(), links=(), uri_links=None) -> PaperContents:
    contents = PaperContents(
        sentences=list(sentences),
        xrefs=[],
        sections=[PaperSection(section_id=1, header="Results", level=1, parent_section_id=0)],
        tables=list(tables),
        links=list(links),
        sections_text={},
        figures=[],
        equations=[],
    )
    contents.pdf_uri_links = uri_links
    return contents


def _sentence(text_id: int, text: str) -> PaperSentence:
    return PaperSentence(text_id=text_id, text=text, section_id=1, paragraph_id=1)


def _table_links(contents: PaperContents) -> list[tuple[int | None, str]]:
    return [(link.table_id, link.url) for link in contents.links if link.table_id is not None]


def test_url_printed_in_a_cell_is_a_link_of_its_table():
    table = _table(
        7, [["Source", "Link"], ["Icons", "Pringle , K . (2026) https://biorender.com/m3o0ecx."]]
    )
    contents = _contents(tables=[table])

    add_table_links(contents)

    (link,) = contents.links
    assert (link.table_id, link.text_id, link.paragraph_id) == (7, None, None)
    assert link.url.rstrip(".") == "https://biorender.com/m3o0ecx"
    assert link.section_id == table.section_id


def test_bare_doi_in_a_cell_is_completed_and_keeps_its_printed_form():
    table = _table(1, [["Funder", "Grant DOI"], ["Wellcome", "10 . 35802/227093"]])
    contents = _contents(tables=[table])

    add_table_links(contents)

    (link,) = contents.links
    assert link.url == "https://doi.org/10.35802/227093"
    assert link.link_text == "10.35802/227093"
    assert link.completed is True


def test_header_cells_are_read_too():
    table = _table(1, [["See https://osf.io/abcde", "Value"], ["a", "1"]])
    contents = _contents(tables=[table])
    add_table_links(contents)
    assert _table_links(contents) == [(1, "https://osf.io/abcde")]


def test_a_target_is_listed_once_per_table_and_calling_again_adds_nothing():
    first = _table(1, [["A"], ["https://osf.io/abcde"], ["https://osf.io/abcde"]])
    second = _table(2, [["A"], ["https://osf.io/abcde"]])
    contents = _contents(tables=[first, second])

    add_table_links(contents)
    add_table_links(contents)

    assert _table_links(contents) == [(1, "https://osf.io/abcde"), (2, "https://osf.io/abcde")]


def test_only_a_full_url_is_read_when_bare_links_are_off():
    table = _table(1, [["A"], ["osf.io/abcde and https://osf.io/fghij"]])
    contents = _contents(tables=[table])
    add_table_links(contents, bare=False)
    assert _table_links(contents) == [(1, "https://osf.io/fghij")]


def test_plain_cells_add_no_link():
    table = _table(1, [["M", "SD"], ["3.42", "0.5"], ["n/a", "e.g. this"]])
    contents = _contents(tables=[table])
    add_table_links(contents)
    assert contents.links == []


def test_link_annotation_on_a_table_names_the_whole_target():
    # The cell wraps the link over two lines; the table reader keeps the first
    # line as a URL and loses the underscores of the second.
    table = _table(
        3,
        [
            ["Variable", "Source"],
            ["CO2", "https://data360.worldbank.org/en/indicator/"],
            ["", "OWID CB CO2 PER UNIT ENERGY"],
            ["Trade", "https://climatedata.imf.org/datasets/975b￾"],
        ],
        page=10,
        bbox=(55.0, 528.0, 942.0, 881.0),
    )
    co2 = "https://data360.worldbank.org/en/indicator/OWID_CB_CO2_PER_UNIT_ENERGY"
    trade = "https://climatedata.imf.org/datasets/975bc577fe7342c2a3651e8841959c47_0/about"
    uri_links = [
        {"page": 10, "bbox": [671.3, 603.9, 911.1, 615.3], "uri": trade},
        {"page": 10, "bbox": [671.3, 549.8, 918.4, 561.3], "uri": co2},
        {"page": 10, "bbox": [671.3, 562.5, 900.6, 573.9], "uri": co2},
        # Not on the table: another page, outside the box, and no web link.
        {"page": 11, "bbox": [671.3, 562.5, 900.6, 573.9], "uri": "https://example.org/other-page"},
        {"page": 10, "bbox": [100.0, 100.0, 300.0, 110.0], "uri": "https://example.org/above"},
        {"page": 10, "bbox": [671.3, 700.0, 900.6, 710.0], "uri": "mailto:someone@example.org"},
    ]
    contents = _contents(tables=[table], uri_links=uri_links)

    add_table_links(contents)

    assert _table_links(contents) == [(3, co2), (3, trade)]
    assert all(link.text_id is None and link.link_text is None for link in contents.links)


def test_cell_link_no_annotation_covers_is_kept_beside_the_annotated_ones():
    table = _table(
        1,
        [["A"], ["10.35802/227093"], ["https://osf.io/abcde"]],
        page=2,
        bbox=(0.0, 0.0, 500.0, 500.0),
    )
    uri_links = [{"page": 2, "bbox": [10, 10, 90, 20], "uri": "https://doi.org/10.35802/227093"}]
    contents = _contents(tables=[table], uri_links=uri_links)

    add_table_links(contents)

    assert _table_links(contents) == [
        (1, "https://doi.org/10.35802/227093"),
        (1, "https://osf.io/abcde"),
    ]


def _link(url: str, text_id: int = 1) -> PaperURLLink:
    return PaperURLLink(url=url, section_id=1, paragraph_id=1, text_id=text_id)


def _uris(*uris: str) -> list[dict]:
    return [{"page": 1, "bbox": [0, 0, 10, 10], "uri": uri} for uri in uris]


def test_hyphen_of_a_line_break_is_removed_by_the_link_target():
    printed = "https://github.com/abbie-ev-ans/korea-outbreak-risks"
    contents = _contents(
        sentences=[_sentence(1, f"Code is at {printed}.")],
        links=[_link(printed)],
        uri_links=_uris("https://github.com/abbie-evans/korea-outbreak-risks"),
    )

    repair_links_from_targets(contents)

    (link,) = contents.links
    assert link.url == "https://github.com/abbie-evans/korea-outbreak-risks"
    assert link.link_text == printed


def test_hyphen_a_line_break_dropped_is_restored():
    contents = _contents(
        sentences=[_sentence(1, "See https://www.niaaa.nih.gov/nesarciii-study for details.")],
        links=[_link("https://www.niaaa.nih.gov/nesarciii-study")],
        uri_links=_uris("https://www.niaaa.nih.gov/nesarc-iii-study"),
    )
    repair_links_from_targets(contents)
    assert contents.links[0].url == "https://www.niaaa.nih.gov/nesarc-iii-study"


def test_link_cut_at_a_line_break_is_completed_when_the_sentence_prints_the_rest():
    contents = _contents(
        sentences=[
            _sentence(
                1, "Data: https://microdata.nbs.gov.so/index.php/cata- log/50 (accessed 2025)."
            )
        ],
        links=[_link("https://microdata.nbs.gov.so/index.php/cata")],
        uri_links=_uris("https://microdata.nbs.gov.so/index.php/catalog/50"),
    )
    repair_links_from_targets(contents)
    assert contents.links[0].url == "https://microdata.nbs.gov.so/index.php/catalog/50"
    assert contents.links[0].link_text == "https://microdata.nbs.gov.so/index.php/cata"


def test_link_that_matches_a_target_is_left_alone():
    contents = _contents(
        sentences=[_sentence(1, "See https://github.com/org/repo and its docs.")],
        links=[_link("https://github.com/org/repo")],
        uri_links=_uris("https://github.com/org/repo", "https://github.com/org/repo/wiki/Home"),
    )
    repair_links_from_targets(contents)
    assert (contents.links[0].url, contents.links[0].link_text) == (
        "https://github.com/org/repo",
        None,
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://doi.org/10.5555/1f35%E2%80%90rn20",  # U+2010, as the PDF writer encoded it
        "https://doi.org/10.5555/1f35%e2%80%90rn20",
        "https://doi.org/10.5555/1f35\u2010rn20",
        "https://doi.org/10.5555/1f35%E2%80%91rn20",  # U+2011, non-breaking
        "https://doi.org/10.5555/1f35\u2011rn20",
        "https://doi.org/10.5555/1f35&#x2010;rn20",
        "https://doi.org/10.5555/1f%C2%AD35-rn\u00ad20",  # soft hyphens
    ],
)
def test_typeset_hyphen_in_a_target_does_not_replace_the_printed_hyphen(uri):
    printed = "https://doi.org/10.5555/1f35-rn20"
    contents = _contents(
        sentences=[_sentence(1, f"Data are at {printed}.")],
        links=[_link(printed)],
        uri_links=_uris(uri),
    )

    repair_links_from_targets(contents)

    (link,) = contents.links
    assert (link.url, link.link_text) == (printed, None)


def test_repaired_link_takes_the_target_with_its_typeset_hyphens_as_ascii():
    # The line break added the hyphen in "long-itudinal"; the target's own two
    # hyphens are U+2010 and U+2011, percent-encoded.
    printed = "https://example.org/long-itudinal-study/wave-2"
    contents = _contents(
        sentences=[_sentence(1, f"See {printed} for the data.")],
        links=[_link(printed)],
        uri_links=_uris("https://example.org/longitudinal%E2%80%90study/wave%e2%80%912"),
    )

    repair_links_from_targets(contents)

    (link,) = contents.links
    assert link.url == "https://example.org/longitudinal-study/wave-2"
    assert link.link_text == printed


def test_typeset_hyphen_in_a_target_on_a_table_is_ascii_in_the_link():
    table = _table(
        1, [["Source"], ["Archive of the survey"]], page=2, bbox=(0.0, 0.0, 500.0, 500.0)
    )
    uri_links = [
        {
            "page": 2,
            "bbox": [10, 10, 90, 20],
            "uri": "https://example.org/data%E2%80%90set\u2011one",
        }
    ]
    contents = _contents(tables=[table], uri_links=uri_links)

    add_table_links(contents)

    assert _table_links(contents) == [(1, "https://example.org/data-set-one")]


@pytest.mark.parametrize(
    "dash",
    ["%E2%80%92", "%E2%80%93", "\u2013", "%E2%80%94", "\u2014", "%E2%80%95", "%E2%88%92", "\u2212"],
)
def test_dash_in_a_target_is_kept(dash):
    # An address can mean an en dash (a wiki page title); only hyphens are read as "-".
    target = f"https://example.org/wiki/Alpha{dash}Beta_experiment/results"
    printed = f"https://example.org/wiki/Alpha{unquote(dash)}Beta_experi-ment/results"
    contents = _contents(
        sentences=[_sentence(1, f"See {printed} for an account.")],
        links=[_link(printed)],
        uri_links=_uris(target),
    )

    repair_links_from_targets(contents)

    (link,) = contents.links
    assert (link.url, link.link_text) == (target, printed)


def test_shorter_link_is_not_completed_unless_the_sentence_prints_the_rest():
    contents = _contents(
        sentences=[_sentence(1, "See https://github.com/org-name/repo and the wiki.")],
        links=[_link("https://github.com/org-name/repo")],
        uri_links=_uris("https://github.com/org-name/repo/wiki/Home"),
    )
    repair_links_from_targets(contents)
    assert contents.links[0].url == "https://github.com/org-name/repo"


def test_two_targets_that_differ_only_by_a_hyphen_repair_nothing():
    contents = _contents(
        sentences=[_sentence(1, "See https://example.org/data-set-one-a here.")],
        links=[_link("https://example.org/data-set-one-a")],
        uri_links=_uris("https://example.org/dataset-one-a", "https://example.org/data-setone-a"),
    )
    repair_links_from_targets(contents)
    assert contents.links[0].url == "https://example.org/data-set-one-a"


def test_table_links_and_documents_without_targets_are_left_alone():
    table_link = PaperURLLink(url="https://example.org/data-set-one", section_id=1, table_id=1)
    contents = _contents(links=[table_link], uri_links=_uris("https://example.org/dataset-one"))
    repair_links_from_targets(contents)
    assert contents.links[0].url == "https://example.org/data-set-one"

    plain = _contents(links=[_link("https://example.org/data-set-one")])
    repair_links_from_targets(plain)
    assert plain.links[0].url == "https://example.org/data-set-one"
