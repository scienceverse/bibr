"""Field repairs that must leave a correct title alone (bibr.extract.ref_field_repair)."""

import pytest

from bibr.extract.ref_field_repair import repair_ner_reference_fields


@pytest.mark.parametrize(
    ("title", "text"),
    [
        # An in-press or undated work prints no year: ", 1945" closing the
        # title is the title's own, not a "Place, YYYY" imprint.
        (
            "Memory and exile. Berlin, 1945",
            "Smith, J. (in press). Memory and exile. Berlin, 1945. Journal of History.",
        ),
        (
            "Memory and exile. Berlin, 1945",
            "Smith, J. Memory and exile. Berlin, 1945. Forthcoming.",
        ),
        (
            "The students revolt. Paris, 1968",
            "Smith, J. (n.d.). The students revolt. Paris, 1968. Retrieved from https://x.org/a",
        ),
        # ", 1914. <more title>" in an in-press work is the title's own year.
        (
            "The outbreak of war in Europe, 1914. The road to catastrophe",
            "Smith, J. (in press). The outbreak of war in Europe, 1914. The road to "
            "catastrophe. Hist Rev.",
        ),
    ],
)
def test_year_in_the_title_of_an_undated_work_stays_in_the_title(title, text):
    fields = {"authors": "Smith, J.", "title": title}
    fired = repair_ner_reference_fields(fields, text)
    assert fields["title"] == title
    assert fields.get("year") is None
    assert not {"title_imprint_tail", "title_year_tail"} & set(fired)


def test_imprint_of_a_dated_work_is_still_cut():
    text = "Doe, A. Relatório anual de atividades. Brasília, DF, 2011."
    fields = {"authors": "Doe, A.", "title": "Relatório anual de atividades. Brasília, DF, 2011"}
    fired = repair_ner_reference_fields(fields, text)
    assert fields["title"] == "Relatório anual de atividades"
    assert fields["year"] == 2011
    assert "title_imprint_tail" in fired


@pytest.mark.parametrize(
    ("fields", "text"),
    [
        # "retrieved" / "viewed" are title words here, not access phrases, and the
        # year printed inside the title is the title's.
        (
            {
                "authors": "Doe J",
                "title": "Sleep and memory, 2000 to 2015: how memories are retrieved",
                "container": "Nature",
                "year": 2019,
            },
            "Doe J. Sleep and memory, 2000 to 2015: how memories are retrieved. Nature 2019;5:1.",
        ),
        (
            {
                "authors": "Doe J",
                "title": "Impressionism, 1890: paintings viewed by visitors",
                "container": "Art J",
                "year": 2015,
            },
            "Doe J. Impressionism, 1890: paintings viewed by visitors. Art J 2015;3:1.",
        ),
    ],
)
def test_access_word_inside_the_title_keeps_the_tagged_year(fields, text):
    year = fields["year"]
    fired = repair_ner_reference_fields(fields, text)
    assert fields["year"] == year
    assert "year_not_access_date" not in fired
