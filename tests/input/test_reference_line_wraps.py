"""Line breaks inside DOIs and URLs, and U+2010/U+2011 at a line end.

A reference printed over several lines reaches ``fix_ocr_artifacts`` CRLF-joined. Its DOI
is the key a citation is linked by, so a space left inside it loses the link. These cases
are short strings built to the shapes the text layer produces; none is a passage of an
article. Each ``OUT`` is the joined identifier, each negative case a line that follows a
complete DOI or URL and must stay on its own line.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bibr.extract.ref_line_stream import _entry_text
from bibr.input.consolidate_text import fix_ocr_artifacts

CRLF = "\r\n"


def _wrap(left: str, right: str) -> str:
    return f"{left}{CRLF}{right}"


@pytest.mark.parametrize(
    ("printed", "joined"),
    [
        # break after the registrant dot, with or without a doi.org link in the text
        (
            _wrap("doi: 10.", "1016/j.jbiosc.2013.09.015 PMID: 1"),
            "doi: 10.1016/j.jbiosc.2013.09.015 PMID: 1",
        ),
        (
            _wrap("https://doi.org/10.", "1177/193229680900300106."),
            "https://doi.org/10.1177/193229680900300106.",
        ),
        # break inside the registrant digits, or between "10" and its dot
        (_wrap("doi:10.11", "77/1525822x16639015"), "doi:10.1177/1525822x16639015"),
        (_wrap("doi:10", ".1101/gr.235499.118"), "doi:10.1101/gr.235499.118"),
        # break right after the registrant slash, whatever the case of the suffix
        (_wrap("doi:10.1038/", "Nmat2630"), "doi:10.1038/Nmat2630"),
        (_wrap("doi:10.1103/", "PhysRevLett.78.1190"), "doi:10.1103/PhysRevLett.78.1190"),
        # break after a bracketed group, or before one
        (
            _wrap("doi:10.1016/S0959-4388(96)", "80080-8 PMID: 3"),
            "doi:10.1016/S0959-4388(96)80080-8 PMID: 3",
        ),
        (_wrap("doi:10.1016/0006-8993(91)", "90927-N"), "doi:10.1016/0006-8993(91)90927-N"),
        (_wrap("doi: 10.1016/S0304-3940", "(03)00880-2"), "doi: 10.1016/S0304-3940(03)00880-2"),
        (_wrap("doi:10.1002/1521-3773", "(20010202)40:3"), "doi:10.1002/1521-3773(20010202)40:3"),
        # break after a dot or an underscore before an upper-case or digit continuation
        (_wrap("doi:10.1074/jbc.", "M705403200"), "doi:10.1074/jbc.M705403200"),
        (
            _wrap("doi:10.1097/01.", "CCM.0000155991.88802.4D"),
            "doi:10.1097/01.CCM.0000155991.88802.4D",
        ),
        (_wrap("doi:10.1158/0008-5472.", "CAN-13-1196"), "doi:10.1158/0008-5472.CAN-13-1196"),
        (_wrap("doi:10.1523/JNEUROSCI.5503-09.", "2010"), "doi:10.1523/JNEUROSCI.5503-09.2010"),
        (
            _wrap("doi:10.1093/cercor/9.7.", "722, PMID: 10554995"),
            "doi:10.1093/cercor/9.7.722, PMID: 10554995",
        ),
        (_wrap("doi:10.1162/dint_a_", "00032"), "doi:10.1162/dint_a_00032"),
        # Annual Reviews: a break after a dot loses no hyphen
        (
            _wrap("10.1146/annurev.arplant.57.", "032905.105346"),
            "10.1146/annurev.arplant.57.032905.105346",
        ),
        (
            _wrap("10.1146/annurev.ecols", "ys.37.091305.110215"),
            "10.1146/annurev.ecolsys.37.091305.110215",
        ),
        # break inside a word of the suffix
        (_wrap("doi:10.1161/STROK", "EAHA.119.025015"), "doi:10.1161/STROKEAHA.119.025015"),
        (_wrap("doi:10.3389/f", "micb.2025.1619626"), "doi:10.3389/fmicb.2025.1619626"),
        (_wrap("doi:10.1093/nar/g", "kq873"), "doi:10.1093/nar/gkq873"),
        # a printed hyphen at the line end (any of the three) inside the identifier
        (_wrap("doi:10.1007/s10995-", "020-03012-3"), "doi:10.1007/s10995-020-03012-3"),
        (_wrap("doi:10.1007/s10995‐", "020‐03012-3"), "doi:10.1007/s10995-020‐03012-3"),
        (_wrap("doi:10.1186/s12986‑", "021-00544-5"), "doi:10.1186/s12986-021-00544-5"),
    ],
)
def test_doi_wrap_is_joined(printed: str, joined: str) -> None:
    assert fix_ocr_artifacts(printed) == joined


@pytest.mark.parametrize(
    "printed",
    [
        # a complete DOI followed by a PubMed or PMC marker, an Elsevier PII or a link
        _wrap("doi:10.1016/j.cell.2016.10.027.", "PMID: 2345"),
        _wrap("doi:10.1016/j.cell.2016.10.027.", "PMC1234567 foo"),
        _wrap("doi:10.1164/rccm.200703-456SO", "PMid:17507545"),
        _wrap("https://doi.org/10.1016/j.tree.2012.10.020.", "S0169-5347(12)00293-5"),
        _wrap("doi:10.1016/j.meatsci.2023.109423.", "http://example.org/1993/37492"),
        _wrap("doi:10.1016/j.meatsci.2023.109423.", "www.example.org/a1"),
        # a complete DOI, then a capitalized word, a number list or a bracket not a tail
        _wrap("doi:10.1016/j.cell.2016.10.027", "Google Scholar"),
        _wrap("doi:10.1016/S0959-4388(96)", "Google Scholar"),
        _wrap("doi:10.1016/j.cell.2016.10.027", "PubMed Abstract"),
        # "10." in prose, not a DOI
        _wrap("Smith. 10.", "Smith"),
        _wrap("see page 10.", "1999 data"),
    ],
)
def test_text_after_a_complete_doi_is_not_joined(printed: str) -> None:
    assert fix_ocr_artifacts(printed) == printed


def test_doi_wrap_is_idempotent() -> None:
    once = fix_ocr_artifacts(_wrap("doi: 10.", "1016/S0959-4388(96)" + CRLF + "80080-8"))
    assert once == "doi: 10.1016/S0959-4388(96)80080-8"
    assert fix_ocr_artifacts(once) == once


@pytest.mark.parametrize(
    ("printed", "joined"),
    [
        # the scheme lies further back than a short window of the last characters
        (
            _wrap("https://www.example.org/en/medicines/human/EPAR/", "entresto (accessed)."),
            "https://www.example.org/en/medicines/human/EPAR/entresto (accessed).",
        ),
        # a token with path or query characters is no capitalized name
        (
            _wrap("https://data.example.gov/Medicare-Enrollment/", "Base-Provider-File/abcd-wxyz"),
            "https://data.example.gov/Medicare-Enrollment/Base-Provider-File/abcd-wxyz",
        ),
        (
            _wrap("https://www.example.org/documents/Redefining_", "obesity/en/"),
            "https://www.example.org/documents/Redefining_obesity/en/",
        ),
        (
            _wrap("https://cran.example.org/package=", "BayesFactor"),
            "https://cran.example.org/package=BayesFactor",
        ),
        # a wrap after the dot before a file extension
        (
            _wrap("http://www.example.org/stf/kenkounippon21.", "html. Accessed 29 May 2015."),
            "http://www.example.org/stf/kenkounippon21.html. Accessed 29 May 2015.",
        ),
        # a wrap inside a word of the path
        (
            _wrap(
                "https://www.example.gov.uk/government/publications/payment-by-re",
                "sults-pbr-2012-13",
            ),
            "https://www.example.gov.uk/government/publications/payment-by-results-pbr-2012-13",
        ),
        # a hyphen wrap in a URL that began on an earlier line, after a wrap bridged first
        (
            "https://www.example.org/retningslinjer/"
            + CRLF
            + "antibiotikabruk-"
            + CRLF
            + "primaerhelse",
            "https://www.example.org/retningslinjer/antibiotikabruk-primaerhelse",
        ),
    ],
)
def test_url_wrap_is_joined(printed: str, joined: str) -> None:
    assert fix_ocr_artifacts(printed) == joined


@pytest.mark.parametrize(
    "printed",
    [
        _wrap("https://www.example.org/en/medicines/human/EPAR/", "Accessed 12 June 2024."),
        _wrap("https://www.example.org/en/page", "Accessed 12 June 2024."),
        _wrap("Available at https://www.example.org/", "Smith J (2010)"),
    ],
)
def test_text_after_a_url_is_not_joined(printed: str) -> None:
    assert fix_ocr_artifacts(printed) == printed


@pytest.mark.parametrize("hyphen", ["-", "‐", "‑"])
class TestLineEndHyphenFollowsOneRule:
    """ASCII hyphen, U+2010 and U+2011 at a line end are adjudicated alike."""

    def test_syllable_hyphen_is_dropped(self, hyphen: str) -> None:
        assert fix_ocr_artifacts(f"analy{hyphen}{CRLF}sis of the data") == "analysis of the data"

    def test_compound_keeps_its_hyphen(self, hyphen: str) -> None:
        assert fix_ocr_artifacts(f"a self{hyphen}{CRLF}report study") == "a self-report study"
        assert fix_ocr_artifacts(f"subjective well{hyphen}{CRLF}being") == "subjective well-being"

    def test_hyphen_between_digits_is_kept(self, hyphen: str) -> None:
        assert fix_ocr_artifacts(f"pages 1999{hyphen}{CRLF}2004") == "pages 1999-2004"

    def test_hyphen_before_a_number_is_kept(self, hyphen: str) -> None:
        assert fix_ocr_artifacts(f"strain IL{hyphen}{CRLF}6 and") == "strain IL-6 and"

    def test_hyphen_at_the_start_of_a_url_line_is_kept(self, hyphen: str) -> None:
        printed = f"https://www.example.org/a{CRLF}{hyphen}b/c"
        assert fix_ocr_artifacts(printed) == "https://www.example.org/a-b/c"

    def test_hyphen_inside_a_doi_is_ascii(self, hyphen: str) -> None:
        printed = f"doi:10.1007/s10995{hyphen}{CRLF}020-03012-3"
        assert fix_ocr_artifacts(printed) == "doi:10.1007/s10995-020-03012-3"

    def test_a_hyphen_mid_line_is_left_alone(self, hyphen: str) -> None:
        text = f"a self{hyphen}report study"
        assert fix_ocr_artifacts(text) == text


def test_entry_text_joins_the_printed_lines_of_one_reference() -> None:
    lines = [
        "Author A. A study of things. J Things 2018;4:1-9. doi:10.",
        "1016/S0959-4388(96)",
        "80080-8 PMID: 3",
        "Another subjective well‐",
        "being measure.",
    ]
    out = _entry_text([SimpleNamespace(text=t) for t in lines])
    assert "doi:10.1016/S0959-4388(96)80080-8 PMID: 3" in out
    assert out.endswith("well-being measure.")


@pytest.mark.parametrize(
    "text",
    [
        "Smith J. A study of things. J Things 2018;4:1-9.",
        f"Smith J. A study of things.{CRLF}J Things 2018;4:1-9. PMID: 123456",
        f"Smith J. Words across{CRLF}a break. 2018;4:1-9.",
        f"Smith J. Pages 10.5{CRLF}to 10.7 were read.",
    ],
)
def test_text_without_these_patterns_is_unchanged(text: str) -> None:
    assert fix_ocr_artifacts(text) == text


# --- what the line after a DOI or URL starts, and where the rules run -------------------------

NEWLINES = ["\r\n", "\n"]
REFERENCE_FLAGS = [None, True, False]
LONG_HOST = "https://journal.example.org/jurnal/index.php/JPDP/"


@pytest.mark.parametrize("reference", REFERENCE_FLAGS)
@pytest.mark.parametrize("newline", NEWLINES)
@pytest.mark.parametrize(
    "template",
    [
        # a name after a registrant slash, a hyphenated word after a path word, a page range
        # after a bracket, a capitalized word or a numbered reference after an underscore
        "doi:10.1038/{nl}Smith, J. (2020) A study.",
        "Code at https://github.com/example-org/example-repo{nl}co-authored the software.",
        "doi:10.1000/a.b){nl}12-34 Smith J.",
        "http://example.com/x_{nl}Another Ref",
        "doi:10.1000/xyz_{nl}2. Jones K.",
        # a label, a new identifier or a copyright mark under a URL that ends on a slash
        LONG_HOST + "{nl}P-ISSN 2461-078X | E-ISSN 2654-783X",
        LONG_HOST + "{nl}E-ISSN:2461-078X",
        "www.example-research.org/Volumes/Vol30/13/{nl}DOI: 10.4054/ExampleRes.2014.30.13",
        "https://creativecommons.example.org/licenses/by-sa/4.0/{nl}@ 2026 by the authors",
        # the next reference's number after a URL
        "https://example.org/ecology/vol13/iss2/art14/{nl}112. Eppo C. European plant",
        # a label or an address under the URL of a journal's page, no path word
        "Home: https://journal.example.org/index.php/jex{nl}e-mail: editor@example.org",
        "ORCID https://orcid.org/0000-0001-7998-920X{nl}E-mail: author@example.org",
        # the word of a citation after a long link that ends on a slash
        "https://example.org/committees/inquiries/parliament-2015/evidence-and-reports/publications/{nl}accessed 30 June 2018.",
        # the name of a site on the line after a link
        "Uploaded 8 May 2020, https://video.example.org/watch?v=cLwQ-eBA2sc{nl}YouTube.com 2020e: Clips",
        # a page range that ends on "10." before a DOI link: no DOI starts there
        "Pages 1\u201310.{nl}https://doi.org/10.1000/example.2020.1",
        "Vol. 3, 4\u201310.{nl}https://doi.org/10.1000/example.2020.1",
    ],
)
def test_a_line_that_starts_something_new_is_not_joined(
    template: str, newline: str, reference: bool | None
) -> None:
    printed = template.format(nl=newline)
    assert fix_ocr_artifacts(printed, reference=reference) == printed


@pytest.mark.parametrize("reference", REFERENCE_FLAGS)
@pytest.mark.parametrize("newline", NEWLINES)
@pytest.mark.parametrize(
    ("template", "joined"),
    [
        # the registrant, in running text
        (
            "(https://doi.org/10.{nl}5555/example.2024.001).",
            "(https://doi.org/10.5555/example.2024.001).",
        ),
        ("data at doi:10.5555/{nl}ex.2024.001.", "data at doi:10.5555/ex.2024.001."),
        # a long URL: a wrap after a slash, inside a word of the path, after a query "="
        (
            "https://www.example.gov.uk/government/publications/consultation-responses/{nl}"
            "summary-of-responses.pdf",
            "https://www.example.gov.uk/government/publications/consultation-responses/"
            "summary-of-responses.pdf",
        ),
        (
            "https://example.org/docs/reform/annual_summary_of_resu{nl}lts_2012.pdf",
            "https://example.org/docs/reform/annual_summary_of_results_2012.pdf",
        ),
        (
            "http://www.example.org/repository/intro{nl}ducing-new-methods-for-wrapped-text",
            "http://www.example.org/repository/introducing-new-methods-for-wrapped-text",
        ),
        ("https://example.org/search?q={nl}Smith", "https://example.org/search?q=Smith"),
        # a capitalized path token, after a slash or an underscore
        (
            "see https://example.org/{nl}Example-Org/Example-Repo for",
            "see https://example.org/Example-Org/Example-Repo for",
        ),
        (
            "https://example.org/files/Annual_{nl}Report.pdf (accessed 2020)",
            "https://example.org/files/Annual_Report.pdf (accessed 2020)",
        ),
        (
            "https://example.org/files/annual_{nl}report_2020.pdf",
            "https://example.org/files/annual_report_2020.pdf",
        ),
        # a hyphenated slug that holds the rest of a split word, or a path character
        (
            "https://example.org/2019/02/cys{nl}tic-fibrosis-medicines-wars-in-europe/ (last visited)",
            "https://example.org/2019/02/cystic-fibrosis-medicines-wars-in-europe/ (last visited)",
        ),
        (
            "https://example.org/docs/adapter-trimm{nl}ed-from-only-the\u2013ends-of-reads.html",
            "https://example.org/docs/adapter-trimmed-from-only-the\u2013ends-of-reads.html",
        ),
        (
            "http://example.org/guides/shared-{nl}governance-programme",
            "http://example.org/guides/shared-governance-programme",
        ),
        # a file extension after the dot, a number after a digit, a bracketed host label
        ("http://example.org/pages/index.{nl}php?id=3", "http://example.org/pages/index.php?id=3"),
        (
            "https://example.org/a/b/c/d/e/f/g/h/i/j/k/16217{nl}534. Accessed 1 Feb 2023",
            "https://example.org/a/b/c/d/e/f/g/h/i/j/k/16217534. Accessed 1 Feb 2023",
        ),
    ],
)
def test_identifier_wraps_in_any_text_are_joined(
    template: str, joined: str, newline: str, reference: bool | None
) -> None:
    assert fix_ocr_artifacts(template.format(nl=newline), reference=reference) == joined


@pytest.mark.parametrize(
    ("left", "right", "joined"),
    [
        (
            "doi:10.1016/S0959-4388(96)",
            "80080-8 PMID: 3",
            "doi:10.1016/S0959-4388(96)80080-8 PMID: 3",
        ),
        ("doi:10.1016/0006-8993(91)", "90927-N", "doi:10.1016/0006-8993(91)90927-N"),
        ("doi: 10.1016/S0304-3940", "(03)00880-2", "doi: 10.1016/S0304-3940(03)00880-2"),
        ("doi: 10.1016/s1474-4422(16)", "00114-119", "doi: 10.1016/s1474-4422(16)00114-119"),
        (
            "(doi:10.3964/j.issn.1000-0593(2015)",
            "01-0151-06)",
            "(doi:10.3964/j.issn.1000-0593(2015)01-0151-06)",
        ),
        ("doi:10.1074/jbc.", "M705403200", "doi:10.1074/jbc.M705403200"),
        ("doi:10.1162/dint_a_", "00032", "doi:10.1162/dint_a_00032"),
        ("doi:10.1161/STROK", "EAHA.119.025015", "doi:10.1161/STROKEAHA.119.025015"),
        ("doi:10.1093/nar/g", "kq873", "doi:10.1093/nar/gkq873"),
        (
            "doi:10.1093/cercor/9.7.",
            "722, PMID: 10554995",
            "doi:10.1093/cercor/9.7.722, PMID: 10554995",
        ),
        (
            "10.1146/annurev.arplant.57.",
            "032905.105346",
            "10.1146/annurev.arplant.57.032905.105346",
        ),
    ],
)
class TestDoiSuffixShapesRunOnReferenceText:
    """Shapes that rejoin the suffix of a DOI run on a reference and on text of unknown kind.

    No body paragraph, caption or title of the papers measured held one, so they are left to
    reference text until they are shown to be right elsewhere.
    """

    def test_a_reference_is_joined(self, left: str, right: str, joined: str) -> None:
        assert fix_ocr_artifacts(_wrap(left, right), reference=True) == joined

    def test_text_of_unknown_kind_is_joined(self, left: str, right: str, joined: str) -> None:
        assert fix_ocr_artifacts(_wrap(left, right)) == joined

    def test_another_region_is_left_alone(self, left: str, right: str, joined: str) -> None:
        printed = _wrap(left, right)
        assert fix_ocr_artifacts(printed, reference=False) == printed


@pytest.mark.parametrize("reference", REFERENCE_FLAGS)
def test_a_page_range_after_a_bracket_is_no_doi_tail(reference: bool | None) -> None:
    for tail in ("12-34 Smith J.", "1234-1240 Smith J.", "12-34"):
        printed = _wrap("doi:10.1000/a.b)", tail)
        assert fix_ocr_artifacts(printed, reference=reference) == printed


def test_entry_text_runs_the_reference_shapes() -> None:
    lines = ["Author A. A study. doi:10.1016/S0959-4388(96)", "80080-8 PMID: 3"]
    out = _entry_text([SimpleNamespace(text=t) for t in lines])
    assert out == "Author A. A study. doi:10.1016/S0959-4388(96)80080-8 PMID: 3"


def test_a_reference_entry_keeps_a_page_range_apart_from_the_doi_link() -> None:
    lines = [
        "Author A. A study. J Things 2018;4:1\u201310.",
        "https://doi.org/10.1000/example.2020.1",
    ]
    out = _entry_text([SimpleNamespace(text=t) for t in lines])
    assert (
        out
        == "Author A. A study. J Things 2018;4:1\u201310. https://doi.org/10.1000/example.2020.1"
    )
