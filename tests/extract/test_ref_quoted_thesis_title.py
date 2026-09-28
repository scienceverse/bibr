"""The quoted-title cut (title_quote_runon) and thesis titles that open on a quotation.

The texts are dev-set references with the fields the NER tagger emitted for
them; the guards give the tagger a title that ran on into the thesis note.
"""

from bibr.extract.ref_field_repair import repair_ner_reference_fields

TREVISAN = (
    "TREVISAN, G. “Somos as pessoas que temos de escolher, não são as outras pessoas que "
    "escolhem por nós”. Infância e cenários de participação pública: uma análise sociológica "
    "dos modos de codecisão das crianças na escola e na cidade. 2014. 524 f. Tese "
    "(Doutoramento em Estudos da Criança) – Programa de Pós-Graduação em Estudos da Criança, "
    "Universidade do Minho, Braga, 2014."
)
TREVISAN_TITLE = (
    "“Somos as pessoas que temos de escolher, não são as outras pessoas que escolhem por nós”. "
    "Infância e cenários de participação pública: uma análise sociológica dos modos de "
    "codecisão das crianças na escola e na cidade"
)


def test_thesis_title_keeps_the_subtitle_after_its_quotation():
    fields = {"authors": "TREVISAN, G", "title": TREVISAN_TITLE, "year": 2014}
    fired = repair_ner_reference_fields(fields, TREVISAN)
    assert "title_quote_runon" not in fired
    assert fields["title"] == TREVISAN_TITLE


def test_thesis_note_run_into_a_quoted_title_is_still_cut():
    text = (
        "E. C. Kerrigan, “Robust constraint satisfaction: Invariant sets and predictive "
        "control,” Ph.D. dissertation, University of Cambridge, 2001."
    )
    fields = {
        "authors": "E. C. Kerrigan",
        "title": "Robust constraint satisfaction: Invariant sets and predictive control,” "
        "Ph.D. dissertation, University of Cambridge",
        "year": 2001,
    }
    fired = repair_ner_reference_fields(fields, text)
    assert "title_quote_runon" in fired
    assert (
        fields["title"] == "Robust constraint satisfaction: Invariant sets and predictive control"
    )


def test_quoted_title_run_into_its_container_is_still_cut():
    text = (
        "Aspinall, Evie. 2022. “COVID-19 Timeline.” British Foreign Policy Group, "
        "https://doi.org/https://www. bbc.co.uk/news/uk-politics-51333314."
    )
    fields = {
        "authors": "Aspinall, Evie",
        "title": "COVID-19 Timeline.” British Foreign Policy Group",
        "year": 2022,
    }
    fired = repair_ner_reference_fields(fields, text)
    assert "title_quote_runon" in fired
    assert fields["title"] == "COVID-19 Timeline"


def test_english_these_is_not_a_thesis_note():
    # Built on a dev-set entry that cites a paper "in these proceedings" (the
    # quoted title is added): the title ran on into its container, which is
    # cut as before.
    text = (
        "Turwitt, M., Elssner, G. and Petzow, G., “Interface structure of bonded metals,” "
        "J. de Physique Colloque C4, these proceedings."
    )
    fields = {
        "authors": "Turwitt, M., Elssner, G. and Petzow, G",
        "title": "Interface structure of bonded metals,” J. de Physique Colloque C4",
    }
    fired = repair_ner_reference_fields(fields, text)
    assert "title_quote_runon" in fired
    assert fields["title"] == "Interface structure of bonded metals"
