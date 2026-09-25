"""Tests for Crossref response schema parsing."""

import pytest


class TestCrossrefWorkItem:
    def test_minimal_doi_lookup(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {"DOI": "10.1234/test", "title": ["A Paper Title"], "type": "journal-article"}
        item = CrossrefWorkItem.from_raw(raw)
        assert item.doi == "10.1234/test"
        assert item.title == "A Paper Title"
        assert item.work_type == "journal-article"

    def test_full_response(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {
            "DOI": "10.1234/test",
            "title": ["Deep Learning"],
            "container-title": ["Nature"],
            "volume": "42",
            "issue": "3",
            "page": "100-110",
            "publisher": "Springer",
            "type": "journal-article",
            "URL": "https://doi.org/10.1234/test",
            "author": [
                {
                    "given": "Jane",
                    "family": "Smith",
                    "ORCID": "https://orcid.org/0000-0001-0000-0000",
                    "sequence": "first",
                },
                {"given": "Bob", "family": "Lee"},
            ],
            "editor": [{"given": "Ed", "family": "Itor"}],
            "issued": {"date-parts": [[2020, 5, 15]]},
            "ISSN": ["1234-5678"],
            "ISBN": ["978-0-123456-78-9"],
            "score": 18.5,
        }
        item = CrossrefWorkItem.from_raw(raw)
        assert item.doi == "10.1234/test"
        assert item.title == "Deep Learning"
        assert item.container_title == "Nature"
        assert item.volume == "42"
        assert item.issue == "3"
        assert item.page == "100-110"
        assert item.publisher == "Springer"
        assert item.work_type == "journal-article"
        assert item.url == "https://doi.org/10.1234/test"
        assert len(item.authors) == 2
        assert item.authors[0].given == "Jane"
        assert item.authors[0].family == "Smith"
        assert len(item.editors) == 1
        assert item.year == 2020
        assert item.date == "2020-05-15"
        assert item.api_score == 18.5

    def test_title_and_container_markup_is_dropped(self):
        """clients-external-enrich-6: deposited tags and entities reached bib_match."""
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {
            "title": ["Effects of elevated CO<sub>2</sub> on <i>Drosophila</i>"],
            "container-title": ["Genes &amp; Development"],
        }
        item = CrossrefWorkItem.from_raw(raw)
        assert (item.title, item.container_title) == (
            "Effects of elevated CO2 on Drosophila",
            "Genes & Development",
        )

    def test_empty_title_list(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {"title": [], "type": "journal-article"}
        item = CrossrefWorkItem.from_raw(raw)
        assert item.title is None

    def test_missing_fields_produce_none(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {}
        item = CrossrefWorkItem.from_raw(raw)
        assert item.doi is None
        assert item.title is None
        assert item.authors == []
        assert item.year is None

    def test_partial_date_parts(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {"issued": {"date-parts": [[2021]]}}
        item = CrossrefWorkItem.from_raw(raw)
        assert item.year == 2021
        assert item.date is None

    def test_year_month_date_parts(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {"issued": {"date-parts": [[2021, 3]]}}
        item = CrossrefWorkItem.from_raw(raw)
        assert item.year == 2021
        assert item.date == "2021-03"

    def test_year_with_zero_month_treated_as_year_only(self):
        """Crossref emits ``[2021, 0]`` for year-only records — formatting
        ``00`` would produce an invalid month string."""
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {"issued": {"date-parts": [[2021, 0]]}}
        item = CrossrefWorkItem.from_raw(raw)
        assert item.year == 2021
        assert item.date is None

    def test_year_month_with_zero_day_drops_day(self):
        """Crossref occasionally emits ``[year, month, 0]`` when only the
        month is known — keep month, drop the bogus day."""
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {"issued": {"date-parts": [[2021, 5, 0]]}}
        item = CrossrefWorkItem.from_raw(raw)
        assert item.year == 2021
        assert item.date == "2021-05"


class TestCrossrefIdentifiers:
    """ORCIDs, affiliation ROR IDs, funders and the license reach the match."""

    RAW = {
        "DOI": "10.1234/ids",
        "title": ["Identified"],
        "author": [
            {
                "given": "Jane",
                "family": "Smith",
                "ORCID": "http://orcid.org/0000-0002-1825-0097",
                "affiliation": [
                    {
                        "name": "University of Glasgow",
                        "id": [
                            {
                                "id": "https://ror.org/00vtgdb53",
                                "id-type": "ROR",
                                "asserted-by": "publisher",
                            }
                        ],
                    },
                    {"name": "Somewhere Else"},
                    {},
                ],
            }
        ],
        "funder": [
            {
                "DOI": "10.13039/501100000780",
                "name": "European Commission",
                "award": ["101000001", " "],
            },
            {
                "name": "Wellcome Trust",
                "id": [
                    {"id": "https://ror.org/029chgv08", "id-type": "ROR"},
                    {"id": "10.13039/100010269", "id-type": "DOI"},
                ],
            },
            {"award": ["orphan"]},
        ],
        "license": [
            {"URL": "https://www.elsevier.com/tdm/userlicense/1.0/", "content-version": "tdm"},
            {"URL": "http://creativecommons.org/licenses/by-nc/4.0/", "content-version": "vor"},
        ],
    }

    def test_parsed_from_the_raw_record(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        item = CrossrefWorkItem.from_raw(self.RAW)
        author = item.authors[0]
        assert [(a.name, a.ror) for a in author.affiliations] == [
            ("University of Glasgow", "https://ror.org/00vtgdb53"),
            ("Somewhere Else", None),
        ]
        assert [(f.name, f.funder_doi, f.ror, f.award_ids) for f in item.funders] == [
            ("European Commission", "10.13039/501100000780", None, ["101000001"]),
            ("Wellcome Trust", "10.13039/100010269", "https://ror.org/029chgv08", []),
        ]
        assert item.license_url == "http://creativecommons.org/licenses/by-nc/4.0/"

    def test_text_mining_licenses_never_stand_for_the_article(self):
        from bibr.enrich.schemas import CrossrefWorkItem

        raw = {"license": [{"URL": "https://example.org/tdm", "content-version": "tdm"}]}
        assert CrossrefWorkItem.from_raw(raw).license_url is None

    def test_carried_onto_the_match(self):
        from bibr.enrich.references import _build_match
        from bibr.enrich.schemas import CrossrefWorkItem

        match = _build_match(CrossrefWorkItem.from_raw(self.RAW), 100.0)
        (author,) = match.authors
        assert author.orcid == "https://orcid.org/0000-0002-1825-0097"
        assert author.affiliation[0].ror == "https://ror.org/00vtgdb53"
        assert match.license_url == "http://creativecommons.org/licenses/by-nc/4.0/"
        assert match.funders[1].ror == "https://ror.org/029chgv08"

    def test_search_selects_the_identifier_fields(self):
        import asyncio

        from bibr.clients.crossref import CrossrefClient
        from bibr.config import GlobalSettings

        client = CrossrefClient(settings=GlobalSettings())
        seen: dict = {}

        async def request(path, params=None):
            seen.update(params or {})
            return {"message": {"items": []}}

        async def cached(key, fetch, **kwargs):
            return await fetch()

        client._request = request
        client._cached = cached
        asyncio.run(client.search("A title"))
        assert {"author", "funder", "license"} <= set(seen["select"].split(","))


class TestPlainText:
    """Deposited title markup, including the pretty-printed layout Crossref
    returns for JATS deposits (recorded shapes)."""

    @pytest.mark.parametrize(
        ("deposited", "plain"),
        [
            ("Global C\n  <sub>2</sub>\n  H\n  <sub>6</sub>\n  maps", "Global C2H6 maps"),
            (
                "CO\n <sub>2</sub>\n and O\n <sub>3</sub>\n on N\n <sub>2</sub>\n O",
                "CO2 and O3 on N2O",
            ),
            ("CO\n <sub>2</sub>\n Emissions", "CO2 Emissions"),
            ("(PM\n  <sub>2.5</sub>\n  ) collected", "(PM2.5) collected"),
            ("Prostaglandin E\n  <sub>2</sub>\n  -Dependent", "Prostaglandin E2-Dependent"),
            ("<scp>COVID</scp>\n  \u201019 and teaching", "COVID\u201019 and teaching"),
            ("Novel\n  <i>Thermoplasmatota</i>\n  Clades", "Novel Thermoplasmatota Clades"),
            ("<i>Cis</i>\n  \u2013\n  <i>trans</i>\n  controls", "Cis\u2013trans controls"),
            ("(\n <i>Gadus morhua</i>\n )", "(Gadus morhua)"),
            (
                "<mml:math><mml:msub><mml:mi>H</mml:mi><mml:mn>2</mml:mn></mml:msub></mml:math>O",
                "H2O",
            ),
            ("p < 0.05 &amp; beyond", "p < 0.05 & beyond"),
            ("Plain title", "Plain title"),
            ("<i></i>", None),
            (None, None),
        ],
    )
    def test_plain_text(self, deposited, plain):
        from bibr.enrich.schemas import plain_text

        assert plain_text(deposited) == plain
