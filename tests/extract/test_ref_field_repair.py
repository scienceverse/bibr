"""Deterministic repairs of NER reference fields (bibr.extract.ref_field_repair).

Each case is a real reference from the dev set with the fields the NER parser
emitted for it (trimmed to the fields the rule reads), so the tests pin both
the repair and the "leave well-formed fields alone" guards next to it.
"""

import pytest

from bibr.config import GlobalSettings
from bibr.extract import ref_extractor as ex
from bibr.extract.ref_field_repair import repair_ner_reference_fields


def _repair(fields: dict, text: str) -> tuple[dict, list[str]]:
    fired = repair_ner_reference_fields(fields, text)
    return fields, fired


# --- web_lead_title ------------------------------------------------------------


def test_untitled_web_reference_takes_its_name_as_title():
    text = "AWS Wavelength. https://aws.amazon.com/wavelength/. Accessed 2025-07-15."
    fields, fired = _repair({"authors": "AWS Wavelength"}, text)
    assert fields["title"] == "AWS Wavelength"
    assert fields["authors"] == "AWS Wavelength"
    assert "web_lead_title" in fired


def test_web_name_runs_past_a_short_author_span():
    text = "Ettus USRP X310. https://www.ettus.com/all-products/x310-kit/. Accessed: 2025-09-13."
    fields, _ = _repair({"authors": "Ettus USRP"}, text)
    assert fields["title"] == "Ettus USRP X310"


def test_web_name_stops_at_the_date_and_keeps_a_broken_url_out():
    text = (
        "Faktör Analizi - yunus.hacettepe.edu.tr. (n.d.). Retrieved fromhttp://yunus."
        "hacettepe.edu.tr/~tonta/courses/f all2007/sb5002/sb5002-12-faktor-analizi.pdf"
        "&p=DevEx.LB.1,5459.1"
    )
    fields, _ = _repair({"authors": "Faktör"}, text)
    assert fields["title"] == "Faktör Analizi - yunus.hacettepe.edu.tr"


def test_web_name_split_into_author_and_title_is_rejoined():
    text = (
        "Saldırganlığı Önlemeye Yönelik The Effect of Psycho-Education ... (n.d.). "
        "Retrieved fromhttp://dergisosyalbil.selcuk.edu.tr/susbed/arti cle/download/236/219"
    )
    fields, _ = _repair(
        {
            "authors": "Saldırganlığı Önlemeye Yönelik",
            "title": "The Effect of Psycho-Education ...",
        },
        text,
    )
    assert fields["title"] == "Saldırganlığı Önlemeye Yönelik The Effect of Psycho-Education"


def test_truncated_web_title_is_extended_to_the_web_phrase():
    text = (
        'Timeline - Overview for "Electrical imp... in Publications - Dimensions. Available '
        "at: https://app.dimensions.ai/analytics/publication/overview/tim eline?search_mode="
        "content&year_from=1990&year_to=2021"
    )
    fields, _ = _repair({"title": 'Timeline - Overview for "Electrical imp...'}, text)
    assert (
        fields["title"] == 'Timeline - Overview for "Electrical imp... in Publications - Dimensions'
    )


def test_web_lead_leaves_a_titled_reference_alone():
    text = (
        "Aspinall, Evie. 2022. Brexit timeline. British Foreign Policy Group. "
        "Available at: https://bfpg.co.uk/timeline"
    )
    fields = {"authors": "Aspinall, Evie", "title": "Brexit timeline", "year": 2022}
    _repair(fields, text)
    assert fields["title"] == "Brexit timeline"


def test_web_lead_does_not_swallow_a_publisher_after_a_quoted_title():
    # "“Title,” Publisher, (2018), URL": the name ends at the closing quote.
    text = (
        "“Congenital Heart Disease,” National Health Service (NHS), (2018), "
        "https://www.nhs.uk/conditions/congenital-heart-disease/."
    )
    fields = {
        "container": "National Health Service (NHS),",
        "title": "“Congenital Heart Disease",
        "year": 2018,
    }
    _, fired = _repair(fields, text)
    assert fields["title"] == "“Congenital Heart Disease"
    assert fired == []


# --- author_dash_title ---------------------------------------------------------


def test_author_span_run_through_a_dash_is_split():
    text = "LIPSZYC, Delia — Domínio Público, 14 folhas mimeografadas sem indicações."
    fields, _ = _repair({"authors": "LIPSZYC, Delia — Domínio Público"}, text)
    assert fields["authors"] == "LIPSZYC, Delia"
    assert fields["title"] == "Domínio Público"


# --- title_url_tail ------------------------------------------------------------


def test_url_at_the_end_of_the_title_is_cut():
    text = (
        "Traditional, Complementary and Integrative Medicine\n[accessed on 8 August]. "
        "Available at: https://www.who.int/health-topics/traditional-Complementary-and-"
        "integrative-medicine World Health Organization, 16 Nov, 2020."
    )
    fields = {
        "title": "Traditional, Complementary and Integrative Medicine "
        "https://www.who.int/health-topics/traditional-Complementary-and-integrative-medicine",
        "year": 2020,
    }
    _repair(fields, text)
    assert fields["title"] == "Traditional, Complementary and Integrative Medicine"


def test_word_online_inside_a_title_is_not_a_url_tail():
    text = (
        "Kuczerawy, A. (2018). The Proposed Regulation on Preventing the Dissemination of "
        "Terrorist Content Online: Safeguards and Risks for Freedom of Expression. Belgium: "
        "CDT. https://dx.doi.org/10.2139/ssrn.3296864"
    )
    title = (
        "The Proposed Regulation on Preventing the Dissemination of Terrorist Content "
        "Online: Safeguards and Risks for Freedom of Expression"
    )
    fields = {"authors": "Kuczerawy, A", "title": title, "year": 2018}
    _repair(fields, text)
    assert fields["title"] == title


def test_parenthesised_url_inside_a_name_is_kept():
    title = (
        "Surveillance, Epidemiology, and End Results (SEER) Program (www.seer.cancer.gov) "
        "SEER*Stat Database: Incidence - SEER 8 Regs Research Data"
    )
    fields = {"title": title}
    _repair(fields, title + ", Nov 2021 Sub (1975-2019).")
    assert fields["title"] == title


# --- title_quote_runon ---------------------------------------------------------


def test_quoted_title_ends_at_its_closing_quote():
    text = (
        "Aspinall, Evie. 2022. “COVID-19 Timeline.” British Foreign Policy Group, "
        "https://doi.org/https://www. bbc.co.uk/news/uk-politics-51333314."
    )
    fields = {
        "authors": "Aspinall, Evie",
        "title": "COVID-19 Timeline.” British Foreign Policy Group",
        "year": 2022,
    }
    _, fired = _repair(fields, text)
    assert fields["title"] == "COVID-19 Timeline"
    assert "title_quote_runon" in fired


def test_quotation_opening_a_title_is_not_cut():
    text = (
        "Memmert, D., & Furley, P. (2007). “I spy with my little eye!”: Breadth of "
        "attention, inattentional blindness, and tactical decision making in team sports. "
        "Journal of Sport & Exercise Psychology, 29(3)."
    )
    title = (
        "“I spy with my little eye!”: Breadth of attention, inattentional blindness, and "
        "tactical decision making in team sports"
    )
    fields = {"authors": "Memmert, D., & Furley, P", "title": title, "year": 2007}
    _repair(fields, text)
    assert fields["title"] == title


def test_quoted_phrase_before_a_tagged_container_is_not_cut():
    text = (
        "Folstein, M. F., Folstein, S. E. & McHugh, P. R. “Mini-Mental State”. A practical "
        "method for grading the cognitive state of patients for the clinician. J. Psychiatr. "
        "Res 12, 189–198 (1975)."
    )
    title = (
        "Mini-Mental State”. A practical method for grading the cognitive state of patients "
        "for the clinician"
    )
    fields = {"container": "J. Psychiatr. Res", "title": title, "year": 1975}
    _repair(fields, text)
    assert fields["title"] == title


@pytest.mark.parametrize(
    "title",
    [
        "«Biuletyn Informacyjny» żołnierzy 27 Wołyńskiej Dywizji Piechoty Armii Krajowej",
        "«Prevent the Reemergence of a New Rival». The Making of the Cheney Regional Defense "
        "Strategy, 1991-1992",
    ],
)
def test_guillemet_quotation_inside_a_title_is_not_cut(title):
    fields = {"title": title}
    _repair(fields, title + ". Available at: http://nsarchive.gwu.edu/nukevault/ebb245/.")
    assert fields["title"] == title


# --- title_dateline_tail / title_translation_bracket ---------------------------


def test_news_dateline_and_translation_leave_the_original_title():
    text = (
        "Akishev M. Vozmutiteli tikhookeanskogo spokoistviia [Disturbers of the peace of the "
        "Pacific]. Expert Online, 18.11.2011. Available at: http://expert.ru/2011/11/18/"
        "vozmutiteli-tihookeanskogo-spokojstviya/ (Accessed: 28.01.2016). (In Russian)."
    )
    fields = {
        "authors": "Akishev M",
        "title": "Vozmutiteli tikhookeanskogo spokoistviia [Disturbers of the peace of the "
        "Pacific]. Expert Online, 18.11.2011",
    }
    _, fired = _repair(fields, text)
    assert fields["title"] == "Vozmutiteli tikhookeanskogo spokoistviia"
    assert fields["container"] == "Expert Online"
    assert fields["year"] == 2011
    assert fired[:2] == ["title_dateline_tail", "title_translation_bracket"]


def test_bracketed_english_translation_is_cut_from_a_transliterated_title():
    text = (
        "V Iuzhnoi Koree poiavitsia krupneishaia v mire voennaia baza SShA [South Korea will "
        "have the world’s largest U.S. military base]. Rossiiskaia gazeta, 15.12.2015."
    )
    fields = {
        "container": "Rossiiskaia gazeta",
        "title": "V Iuzhnoi Koree poiavitsia krupneishaia v mire voennaia baza SShA [South "
        "Korea will have the world’s largest U.S. military base].",
    }
    _repair(fields, text)
    assert fields["title"] == "V Iuzhnoi Koree poiavitsia krupneishaia v mire voennaia baza SShA"


def test_gost_material_designation_is_cut():
    title = "Особенности ценностно-смысловой сферы у студентов в период обучения в вузе"
    fields = {"title": title + " [Электронный ресурс]", "year": 2019}
    _repair(fields, title + " [Электронный ресурс] // Психология и право. 2019. Том 9.")
    assert fields["title"] == title


def test_bracket_on_an_english_title_is_kept():
    title = "The effects of feedback on learning [Doctoral dissertation, University of Utah]"
    fields = {"title": title, "year": 2019}
    _repair(fields, "Smith, J. (2019). " + title + ".")
    assert fields["title"] == title


# --- title_language_note / title_responsibility --------------------------------


def test_language_note_ends_the_title():
    text = "—(1982): Netra Yog Chikitsa (Hindi). Yog Prakshikshan Kendra Publications, Delhi."
    fields = {"title": "Netra Yog Chikitsa (Hindi). Yog Prakshikshan", "year": 1982}
    _repair(fields, text)
    assert fields["title"] == "Netra Yog Chikitsa"


def test_statement_of_responsibility_is_cut():
    text = (
        "Yuldashev, A. A. The influence of altitude and ambient temperature on the output "
        "power of the asynchronous motor / A. A. Yuldashev, Z. Sh. Yuldashev // Bulletin of "
        "the Student Scientific Society. – 2014. – № 3. – P. 105–106."
    )
    fields = {
        "authors": "Yuldashev, A. A",
        "title": "The influence of altitude and ambient temperature on the output power of the "
        "asynchronous motor / A. A. Yuldashev, Z. Sh. Yuldashev",
        "year": 2014,
    }
    _repair(fields, text)
    assert fields["title"] == (
        "The influence of altitude and ambient temperature on the output power of the "
        "asynchronous motor"
    )


# --- title_year_tail / title_imprint_tail --------------------------------------


def test_year_and_thesis_notes_are_cut_from_the_title():
    text = (
        "BOIN, M. N. Chuvas e erosões no oeste paulista: uma análise climatológica aplicada. "
        "2000. 264f. Tese (Doutorado em Geociências e Meio Ambiente) – Instituto Geográfico"
    )
    fields = {
        "authors": "BOIN, M. N",
        "title": "Chuvas e erosões no oeste paulista: uma análise climatológica aplicada. 2000. "
        "264f. Tese (Doutorado em Geociências e Meio Ambiente) – Instituto Geográfico",
    }
    _repair(fields, text)
    assert (
        fields["title"] == "Chuvas e erosões no oeste paulista: uma análise climatológica aplicada"
    )
    assert fields["year"] == 2000


def test_imprint_is_cut_from_the_title_and_gives_the_year():
    title = (
        "Ministério da Saúde. Passo a Passo PSE. Programa Saúde na Escola: tecendo caminhos da "
        "intersetorialidade"
    )
    text = (
        "BRASIL. " + title + ". Brasília, DF, 2011. Disponível em: <https://bvsms.saude.gov.br/"
        "bvs/publicacoes/passo_a_passo_programa_saude_escola.pdf>. Acesso em: 14 abr. 2022."
    )
    fields = {"authors": "BRASIL", "title": title + ". Brasília, DF, 2011"}
    _repair(fields, text)
    assert fields["title"] == title
    assert fields["year"] == 2011


# --- year_not_access_date / year_from_text -------------------------------------


def test_access_year_yields_to_the_printed_publication_year():
    text = (
        "CASTRO, Procópio de. “Parque Linear: a água como destaque na revitalização de rios no "
        "espaço urbano” Jornal Manuelzão, Belo Horizonte, 31 dez. 1969. Disponível em:https://"
        "www.ecodebate.com.br/2011/03/02/parque-linear/. Acesso em: 14 abr. 2022."
    )
    fields = {"authors": "CASTRO, Procópio de", "title": "Parque Linear", "year": 2022}
    _, fired = _repair(fields, text)
    assert fields["year"] == 1969
    assert "year_not_access_date" in fired


def test_missing_year_is_taken_from_the_printed_date():
    text = (
        "MUNFORD, D; LIMA, M. E. C. De C. Ensinar Ciências por Investigação: Em que estamos de "
        "acordo? Revista ensaio, V. 9, n. 1, 72-89, jan-jun. 2007. Disponível em: <https://"
        "www.scielo.br/scielo.php?script=sci_arttext&pid=S1983-21172007000100089>. Acesso "
        "em: 15 de jun. 2020."
    )
    fields = {
        "authors": "MUNFORD, D; LIMA, M. E. C. De C",
        "title": "Ensinar Ciências por Investigação: Em que estamos de acordo? Revista ensaio, "
        "V. 9, n. 1, 72-89",
    }
    _, fired = _repair(fields, text)
    assert fields["year"] == 2007
    assert fired == ["year_from_text"]


@pytest.mark.parametrize(
    ("fields", "text", "year"),
    [
        (
            {
                "authors": "DOZORTSEV, V",
                "title": "Les Editeurs, les Producteurs de Phonogrammes et de Videogrammes et Ia "
                "Protection des Oeuvres du Domaine Public, Doe. PRS/CPÍ/DP/CEG/I/?, apresentado ao "
                "Comitê de Peritos Governamentais sobre a proteção das obras do domínio público",
            },
            "DOZORTSEV, V. — Les Editeurs, les Producteurs de Phonogrammes et de Videogrammes et "
            "Ia Protection des Oeuvres du Domaine Public, Doe. PRS/CPÍ/DP/CEG/I/?, apresentado ao "
            "Comitê de Peritos Governamentais sobre a proteção das obras do domínio público, da "
            "UNESCO, reunido em Paris de 17 a 21-01-1983, 16 págs. mimeografadas.",
            1983,
        ),
        (
            {"title": "Secretary Rumsfeld Interview", "container": "The New York Times"},
            "Secretary Rumsfeld Interview // The New York Times. 12.10.2001.",
            2001,
        ),
    ],
)
def test_missing_year_falls_back_to_a_full_numeric_date(fields, text, year):
    _, fired = _repair(fields, text)
    assert fields["year"] == year
    assert fired == ["year_from_text"]


def test_a_year_in_a_date_position_wins_over_numeric_dates():
    # The citing article's history dates leak into the last reference.
    text = (
        "SPOLIN, Viola. Improvisação para o teatro. São Paulo: Perspectiva, 2010.\n"
        "Recebido: 12/7/2025"
    )
    fields = {"authors": "SPOLIN, Viola", "title": "Improvisação para o teatro"}
    _repair(fields, text)
    assert fields["year"] == 2010


@pytest.mark.parametrize(
    ("fields", "text"),
    [
        # an arXiv identifier is not a date
        (
            {"authors": "A. Chandra, M. Hairer, and H. Shen", "title": "The dynamical model"},
            "A. Chandra, M. Hairer, and H. Shen, The dynamical model. arXiv:1808.02594.",
        ),
        # a year inside the title is the title's
        (
            {
                "authors": "R. Zheng, et al.,",
                "title": "Cancer incidence and mortality in China, 2016",
            },
            "R. Zheng, et al., Cancer incidence and mortality in China, 2016, Journal of the "
            "National Cancer Center 2 (1) (2022/03/01/2022) 1–9.",
        ),
        # a year in a name, not in a date position
        (
            {"title": "Symptoms of Coronavirus"},
            "Symptoms of Coronavirus. Coronavirus Disease 2019 (COVID-19). Available at: "
            "https://www.cdc.gov/coronavirus/2019-ncov/symptoms.html.",
        ),
        # two candidate years: ambiguous
        (
            {"authors": "KROEBER, A. L", "title": "The Speech of a Zuni Child"},
            "KROEBER, A. L. The Speech of a Zuni Child. Amtr. Anthrop., 1916,18, 529—534\n"
            "9. MACNF, J. A. Vocabularies. Ped. Sem., 1919, 26, 209-233",
        ),
        # an article-history date is not the cited work's year
        (
            {"authors": "World Health Organization", "title": "The World Health Report"},
            "World Health Organization. The World Health Report. Reducing Risks, Promoting "
            "Healthy Life, Geneva 2002.\nDate of acceptance: 29-3-2016",
        ),
        # in press: no year
        (
            {"authors": "Doe, J", "title": "A study"},
            "Doe, J. (in press). A study. Journal of Studies, 2021 special issue, 3.",
        ),
    ],
)
def test_year_is_not_guessed_without_one_clear_date(fields, text):
    _repair(fields, text)
    assert "year" not in fields


def test_vancouver_tail_is_left_to_the_finalize_backfill():
    text = (
        "Carabantes N, Grosso-Becerra MV, Thomé PE. Expression of glucose transporters in "
        "Cassiopea xamachana (Bigelow, 1892) jellyfish. Mar Biol 2024;171:54."
    )
    fields = {"authors": "Carabantes N, Grosso-Becerra MV, Thomé PE"}
    _repair(fields, text)
    assert "year" not in fields


def test_well_formed_reference_is_untouched():
    text = (
        "Holder D. Electrical impedance tomography: Methods, history, and applications. "
        "CRC Press, 2005."
    )
    fields = {
        "authors": "Holder D",
        "title": "Electrical impedance tomography: Methods, history, and applications",
        "year": 2005,
    }
    before = dict(fields)
    _, fired = _repair(fields, text)
    assert fired == []
    assert fields == before


# --- wiring --------------------------------------------------------------------


def test_ner_parse_path_applies_the_repairs(monkeypatch):
    class _Parser:
        def parse_batch(self, texts, batch_size=32):
            return [{"authors": "AWS Wavelength"} for _ in texts]

    monkeypatch.setattr(ex, "_get_ner_parser", lambda settings, memory_mode=None: _Parser())
    extractor = ex.ReferenceExtractor(contents=None, file_hash="x", settings=GlobalSettings())
    refs = extractor._parse_references_ner_aligned(
        ["AWS Wavelength. https://aws.amazon.com/wavelength/. Accessed 2025-07-15."]
    )
    assert refs[0] is not None
    assert refs[0].title == "AWS Wavelength"
