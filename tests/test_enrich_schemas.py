"""Tests for Crossref response schema parsing."""


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
