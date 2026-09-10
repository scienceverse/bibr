"""Tests for Pydantic LLM output schemas."""

import pytest
from pydantic import ValidationError

from bibr.models import BibType
from bibr.schemas import (
    AffiliationLLM,
    AuthorLLM,
    AuthorsLLM,
    CitationMatch,
    CitationResolutionResult,
    CoreMetadataLLM,
    FundingEntryLLM,
    PaperClassificationLLM,
    PaperReferenceList,
    PaperReferenceLLM,
    TitleKeywordsLLM,
)


def _collect_enums(node) -> list:
    """Recursively gather every ``enum`` list found under a JSON-schema node."""
    found: list = []
    if isinstance(node, dict):
        if "enum" in node:
            found.extend(node["enum"])
        for v in node.values():
            found.extend(_collect_enums(v))
    elif isinstance(node, list):
        for v in node:
            found.extend(_collect_enums(v))
    return found


class TestPaperClassificationLLM:
    def test_json_schema_has_enum_arrays(self):
        # The whole point: guided decoding must see enum grammars so invalid
        # strings (e.g. NuExtract3's "verbatim-string" DSL token) are
        # unrepresentable under JSON_SCHEMA mode.
        schema = PaperClassificationLLM.model_json_schema()
        for field, sample in [
            ("oecd_domain", "Social Sciences"),
            ("oecd_subdomain", "Psychology and Cognitive Sciences"),
            ("paper_type", "empirical"),
        ]:
            enums = _collect_enums(schema["properties"][field])
            assert sample in enums, f"{field} missing enum grammar"

    def test_verbatim_string_coerced_to_none(self):
        m = PaperClassificationLLM(
            oecd_domain="verbatim-string",
            oecd_subdomain="verbatim-string",
            paper_type="verbatim-string",
        )
        assert m.oecd_domain is None
        assert m.oecd_subdomain is None
        assert m.paper_type is None

    def test_cloud_near_miss_rescued(self):
        # Gemini-path leniency: near-miss labels fuzzy-canonicalize, not fail.
        m = PaperClassificationLLM(
            oecd_domain="social sciences",
            oecd_subdomain="Cognitive Science",
            paper_type="Empirical",
        )
        assert m.oecd_domain == "Social Sciences"
        assert m.oecd_subdomain == "Psychology and Cognitive Sciences"
        assert m.paper_type == "empirical"

    def test_subdomain_canonicalized_without_l1_context(self):
        # The BeforeValidator has no L1 context — canonicalize_oecd_l2_any.
        m = PaperClassificationLLM(
            oecd_domain="Engineering and Technology",
            oecd_subdomain="Computer Science",
        )
        assert m.oecd_subdomain == "Computer and Information Sciences"

    def test_none_passes_through(self):
        m = PaperClassificationLLM()
        assert (m.oecd_domain, m.oecd_subdomain, m.paper_type) == (None, None, None)

    def test_never_raises_on_garbage(self):
        m = PaperClassificationLLM(
            oecd_domain="Astrology",
            oecd_subdomain="Basket Weaving",
            paper_type="listicle",
        )
        assert (m.oecd_domain, m.oecd_subdomain, m.paper_type) == (None, None, None)


class TestAuthorLLM:
    @pytest.mark.parametrize("model", [AuthorsLLM, CoreMetadataLLM])
    def test_generation_schema_excludes_downstream_fields(self, model):
        schema = model.model_json_schema()["$defs"]["AuthorLLM"]
        assert "role" not in schema["properties"]
        assert "author_id" not in schema["properties"]
        # They remain available to downstream assembly and older cached responses.
        author = AuthorLLM(given="Jane", family="Doe")
        assert author.author_id is None
        assert author.role == []
        assert author.model_dump()["role"] == []

    def test_minimal(self):
        author = AuthorLLM(given="Jane", family="Doe")
        assert author.given == "Jane"
        assert author.family == "Doe"
        assert author.affiliation == ""
        assert author.email is None
        assert author.corresponding is False
        assert author.orcid is None

    def test_full(self):
        author = AuthorLLM(
            given="Jean-Pierre",
            family="van der Berg",
            affiliation="MIT",
            email="jp@mit.edu",
            corresponding=True,
            orcid="0000-0001-2345-6789",
        )
        assert author.corresponding is True
        assert author.orcid == "0000-0001-2345-6789"

    def test_literal_none_string_coerced_to_none(self):
        """LLMs occasionally emit ``"None"`` (Python repr) instead of JSON null —
        the literal string must not flow into corresponding-author email
        harvesting downstream."""
        author = AuthorLLM(given="J", family="D", email="None", orcid="null")
        assert author.email is None
        assert author.orcid is None

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AuthorLLM()


class TestCoreMetadataLLM:
    @pytest.mark.parametrize("model", [TitleKeywordsLLM, CoreMetadataLLM])
    @pytest.mark.parametrize("abstract", [None, "", "  ", "verbatim-string"])
    def test_only_explicit_json_null_is_an_abstract_refusal(self, model, abstract):
        value = model(title="Study", authors=[], abstract=abstract)
        assert value.abstract is None
        assert value._abstract_explicitly_absent is (abstract is None)
        assert "_abstract_explicitly_absent" not in value.model_dump()

    def test_basic(self):
        meta = CoreMetadataLLM(
            title="A Study",
            authors=[AuthorLLM(given="A", family="B")],
            keywords=["ml", "nlp"],
        )
        assert meta.title == "A Study"
        assert len(meta.authors) == 1
        assert meta.keywords == ["ml", "nlp"]

    def test_empty_authors_and_keywords(self):
        meta = CoreMetadataLLM(title="Test", authors=[], keywords=[])
        assert meta.authors == []
        assert meta.keywords == []

    def test_missing_title_degrades_to_none(self):
        meta = CoreMetadataLLM(authors=[], keywords=[])
        assert meta.title is None


class TestPaperReferenceLLM:
    def test_minimal(self):
        ref = PaperReferenceLLM(
            index=1,
            title="Deep Learning",
            first_page=None,
            volume=None,
            authors="Smith",
            year="2021",
            container=None,
        )
        assert ref.index == 1
        assert ref.doi is None
        assert ref.last_page is None
        assert ref.issue is None
        assert ref.publisher is None
        assert ref.editors is None
        assert ref.bib_type is None

    def test_full(self):
        ref = PaperReferenceLLM(
            index=1,
            title="Deep Learning",
            first_page="100",
            volume="42",
            authors="Smith, J.",
            year="2021",
            container="Nature",
            doi="10.1234/test",
        )
        assert ref.first_page == "100"
        assert ref.doi == "10.1234/test"

    def test_new_fields(self):
        ref = PaperReferenceLLM(
            index=1,
            title="Chapter Title",
            first_page="50",
            last_page="75",
            volume=None,
            issue="3",
            authors="Doe, J.",
            year="2022",
            container="Handbook of ML",
            publisher="Springer",
            editors="Ed. Smith",
            bib_type=BibType.BOOK_CHAPTER.value,
        )
        assert ref.last_page == "75"
        assert ref.issue == "3"
        assert ref.publisher == "Springer"
        assert ref.editors == "Ed. Smith"
        assert ref.bib_type == "book_chapter"

    def test_string_authors_accepted(self):
        """authors is now a plain string field."""
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page=None,
            volume=None,
            authors="Smith",
            year="2020",
            container=None,
        )
        assert ref.authors == "Smith"

    def test_none_authors_accepted(self):
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page=None,
            volume=None,
            authors=None,
            year="2020",
            container=None,
        )
        assert ref.authors is None

    def test_serialization_leak_in_first_page_coerced_to_none(self):
        """Regress the book-entry stringified-dict bug.

        When the LLM hallucinates a serialization fragment into a numeric
        field (e.g. ``"None,index:6,is_in_press:false,last_page:"``), the
        validator must recognize the colon as a clear leak signal and emit
        None rather than passing the garbage downstream.
        """
        ref = PaperReferenceLLM(
            index=1,
            title="Diagnostic Evaluation of Articulation",
            first_page="None,index:6,is_in_press:false,last_page:",
            volume=None,
            authors="Dodd, B.",
            year=2002,
            container=None,
        )
        assert ref.first_page is None

    def test_serialization_leak_in_volume_coerced_to_none(self):
        ref = PaperReferenceLLM(
            index=1,
            title="Predictably irrational",
            first_page=None,
            volume="None,year:2009},{authors:",
            authors="Ariely, D.",
            year=2009,
            container=None,
        )
        assert ref.volume is None

    def test_legitimate_short_page_values_preserved(self):
        """Sanity: don't over-fire the colon heuristic on real values."""
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page="Article 13",
            last_page="S42",
            volume="4",
            issue="2",
            authors="Smith",
            year=2020,
            container=None,
        )
        assert ref.first_page == "Article 13"
        assert ref.last_page == "S42"
        assert ref.volume == "4"
        assert ref.issue == "2"


class TestPaperReferenceList:
    def test_empty(self):
        rl = PaperReferenceList(references=[])
        assert rl.references == []

    def test_with_refs(self):
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page=None,
            volume=None,
            authors="A",
            year="2020",
            container=None,
        )
        rl = PaperReferenceList(references=[ref])
        assert len(rl.references) == 1


class TestCitationMatch:
    def test_resolved(self):
        m = CitationMatch(text_id=5, citation_text="[1]", bib_id=1)
        assert m.bib_id == 1

    def test_unresolved(self):
        m = CitationMatch(text_id=5, citation_text="Smith (2020)", bib_id=None)
        assert m.bib_id is None


class TestCitationResolutionResult:
    def test_empty(self):
        r = CitationResolutionResult(matches=[])
        assert r.matches == []

    def test_with_matches(self):
        matches = [
            CitationMatch(text_id=1, citation_text="[1]", bib_id=1),
            CitationMatch(text_id=2, citation_text="[2]", bib_id=None),
        ]
        r = CitationResolutionResult(matches=matches)
        assert len(r.matches) == 2


class TestBibTypeNormalization:
    """``PaperReferenceLLM.bib_type`` is plain str (was BibTypeEnum); the
    pre-validator coerces legacy/free-form input to a canonical
    :class:`bibr.models.BibType` value."""

    def test_canonical_values_pass_through(self):
        for canonical in (
            BibType.JOURNAL_ARTICLE.value,
            BibType.BOOK.value,
            BibType.BOOK_CHAPTER.value,
            BibType.CONFERENCE_PAPER.value,
            BibType.PREPRINT.value,
        ):
            ref = PaperReferenceLLM(
                index=1,
                title="T",
                first_page=None,
                volume=None,
                authors="A",
                year="2020",
                container=None,
                bib_type=canonical,
            )
            assert ref.bib_type == canonical

    def test_legacy_alias_normalized(self):
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page=None,
            volume=None,
            authors="A",
            year="2020",
            container=None,
            bib_type="article",
        )
        assert ref.bib_type == "journal_article"

    def test_unknown_falls_back_to_other(self):
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page=None,
            volume=None,
            authors="A",
            year="2020",
            container=None,
            bib_type="invalid_type",
        )
        assert ref.bib_type == "other"

    def test_used_in_pydantic_model(self):
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page=None,
            volume=None,
            authors="A",
            year="2020",
            container=None,
            bib_type=BibType.JOURNAL_ARTICLE.value,
        )
        assert ref.bib_type == "journal_article"

    def test_none_bibtype_allowed(self):
        ref = PaperReferenceLLM(
            index=1,
            title="T",
            first_page=None,
            volume=None,
            authors="A",
            year="2020",
            container=None,
        )
        assert ref.bib_type is None


class TestSoftwareReferenceTitleFix:
    """Residual #2: the LLM nondeterministically siphoned a printed
    '(Version X)' out of a software-reference title into the inherited
    ``version`` field, truncating the title (RStudio/afex). We steer the LLM to
    keep the version + medium tag in the title via the title description/prompt
    rather than excluding fields — ``version`` (software) and ``edition``
    (books, '2nd ed.') carry real data that must NOT be dropped.
    """

    def test_version_and_edition_remain_in_llm_schema(self):
        # Both fields must stay available — edition is used by books, version by
        # software/datasets. Dropping them from the schema would lose real data.
        props = PaperReferenceLLM.model_json_schema()["properties"]
        assert "version" in props
        assert "edition" in props

    def test_title_description_instructs_software_verbatim(self):
        desc = (PaperReferenceLLM.model_fields["title"].description or "").lower()
        assert "version" in desc
        assert "computer software" in desc


class TestAffiliationLLM:
    """Structured affiliation parse on ResearchIntegrityLLM."""

    def test_minimal_index_only(self):
        from bibr.schemas import AffiliationLLM

        aff = AffiliationLLM(index=1)
        assert aff.index == 1
        assert aff.institution is None
        assert aff.department is None
        assert aff.city is None
        assert aff.country is None

    def test_full(self):
        from bibr.schemas import AffiliationLLM

        aff = AffiliationLLM(
            index=2,
            institution="Univ X",
            department="Dept of Psychology",
            city="London",
            country="UK",
        )
        assert aff.institution == "Univ X"
        assert aff.department == "Dept of Psychology"
        assert aff.city == "London"
        assert aff.country == "UK"

    def test_literal_none_string_coerced_to_none(self):
        from bibr.schemas import AffiliationLLM

        aff = AffiliationLLM(
            index=1, institution="None", department="null", city="None", country="null"
        )
        assert aff.institution is None
        assert aff.department is None
        assert aff.city is None
        assert aff.country is None

    def test_missing_index_fails(self):
        from bibr.schemas import AffiliationLLM

        with pytest.raises(ValidationError):
            AffiliationLLM()

    def test_research_integrity_affiliations_none_coerced_to_empty(self):
        from bibr.schemas import ResearchIntegrityLLM

        r = ResearchIntegrityLLM.model_validate({"affiliations": None})
        assert r.affiliations == []

    def test_research_integrity_parses_affiliations(self):
        from bibr.schemas import ResearchIntegrityLLM

        r = ResearchIntegrityLLM.model_validate(
            {"affiliations": [{"index": 1, "institution": "Univ X", "country": "UK"}]}
        )
        assert r.affiliations[0].index == 1
        assert r.affiliations[0].institution == "Univ X"
        assert r.affiliations[0].country == "UK"


class TestSchemaNameUnwrap:
    """Small local models wrap structured output in the schema name
    (``{"RefAnchors": {"anchors": [...]}}``) — observed with gemma-4-e2b and
    Qwen3.5-4B via vllm-mlx, where json_schema mode is advisory. The
    before-validator unwraps that envelope instead of failing validation."""

    def test_refanchors_unwraps_schema_name_envelope(self):
        from bibr.schemas import RefAnchors

        result = RefAnchors.model_validate({"RefAnchors": {"anchors": ["Smith, J.", "Doe, A."]}})
        assert result.anchors == ["Smith, J.", "Doe, A."]

    def test_reference_list_unwraps_schema_name_envelope(self):
        result = PaperReferenceList.model_validate({"PaperReferenceList": {"references": []}})
        assert result.references == []

    def test_plain_payload_still_validates(self):
        from bibr.schemas import RefAnchors

        result = RefAnchors.model_validate({"anchors": ["Smith, J."]})
        assert result.anchors == ["Smith, J."]

    def test_wrong_key_envelope_still_fails(self):
        from bibr.schemas import RefAnchors

        with pytest.raises(ValidationError):
            RefAnchors.model_validate({"SomethingElse": {"anchors": ["Smith, J."]}})

    def test_envelope_with_non_dict_value_fails(self):
        from bibr.schemas import RefAnchors

        with pytest.raises(ValidationError):
            RefAnchors.model_validate({"RefAnchors": "not a dict"})


class TestTitleKeywordsBiblio:
    """The paper's own bibliographic self-identity fields on TitleKeywordsLLM."""

    def test_defaults_none(self):
        tk = TitleKeywordsLLM(title="A Study")
        assert tk.journal is None
        assert tk.volume is None
        assert tk.issue is None
        assert tk.first_page is None
        assert tk.last_page is None
        assert tk.issn is None
        assert tk.publisher is None
        assert tk.published is None
        assert tk.license is None

    def test_title_is_optional_in_validation_and_json_schema(self):
        tk = TitleKeywordsLLM()
        assert tk.title is None
        assert "title" not in TitleKeywordsLLM.model_json_schema().get("required", [])

    def test_merged_core_title_is_optional_in_json_schema(self):
        schema = CoreMetadataLLM.model_json_schema()
        assert "title" not in schema.get("required", [])

    def test_full(self):
        tk = TitleKeywordsLLM(
            title="A Study",
            journal="Psychological Science",
            volume="31",
            issue="1",
            first_page="65",
            last_page="74",
            issn="0956-7976",
            publisher="SAGE Publications",
            published="2020-01-01",
            license="CC BY 4.0",
        )
        assert tk.journal == "Psychological Science"
        assert tk.volume == "31"
        assert tk.issue == "1"
        assert tk.first_page == "65"
        assert tk.last_page == "74"
        assert tk.issn == "0956-7976"
        assert tk.publisher == "SAGE Publications"
        assert tk.published == "2020-01-01"
        assert tk.license == "CC BY 4.0"

    def test_numeric_locators_are_stringified(self):
        # The wire gate lets a bare number through on the short-numeric fields
        # (a whole extraction is not worth losing over an unquoted 31), so the
        # model has to be the layer that normalises it.
        tk = TitleKeywordsLLM.model_validate(
            {"title": "A Study", "volume": 31, "issue": 1, "first_page": 65, "last_page": 74}
        )
        assert tk.volume == "31"
        assert tk.issue == "1"
        assert tk.first_page == "65"
        assert tk.last_page == "74"

    def test_literal_none_strings_coerced(self):
        tk = TitleKeywordsLLM(
            title="A Study",
            journal="None",
            volume="null",
            publisher="None",
            license="null",
        )
        assert tk.journal is None
        assert tk.volume is None
        assert tk.publisher is None
        assert tk.license is None

    def test_colon_leak_dropped_from_numeric_fields(self):
        # A colon can only be a serialization-fragment leak in these short
        # numeric fields (mirrors PaperReferenceLLM.coerce_none_strings).
        tk = TitleKeywordsLLM(
            title="A Study",
            volume="31,issue:1,first_page:",
            issue="1:leak",
            first_page="65:leak",
            last_page="74:leak",
        )
        assert tk.volume is None
        assert tk.issue is None
        assert tk.first_page is None
        assert tk.last_page is None

    def test_colon_allowed_in_journal(self):
        # container-like text fields legitimately contain colons.
        tk = TitleKeywordsLLM(title="A Study", journal="Nature: Reviews")
        assert tk.journal == "Nature: Reviews"

    def test_core_metadata_carries_biblio(self):
        meta = CoreMetadataLLM(
            title="A Study",
            authors=[AuthorLLM(given="A", family="B")],
            journal="Nature",
            volume="600",
            issue="None",
        )
        assert meta.journal == "Nature"
        assert meta.volume == "600"
        assert meta.issue is None


class TestPlaceholderScrub:
    """NuExtract3 template-DSL type tokens (e.g. "verbatim-string") echoed as
    field values are noise. Scrub is EXACT whole-value match after
    strip/casefold only — never substring."""

    def test_keywords_drop_placeholder_token(self):
        tk = TitleKeywordsLLM(title="A Study", keywords=["deep learning", "verbatim-string"])
        assert tk.keywords == ["deep learning"]

    def test_keywords_legit_value_survives(self):
        # "string theory" contains the word "string" but is not an exact
        # whole-value match — must not be scrubbed.
        tk = TitleKeywordsLLM(title="A Study", keywords=["string theory", "deep learning"])
        assert tk.keywords == ["string theory", "deep learning"]

    def test_keywords_all_placeholder_tokens(self):
        tk = TitleKeywordsLLM(
            title="A Study",
            keywords=["string", "integer", "number", "boolean", "date-time", "verbatim-string"],
        )
        assert tk.keywords == []

    def test_keywords_case_and_whitespace_insensitive(self):
        tk = TitleKeywordsLLM(title="A Study", keywords=["  Verbatim-String  ", "real kw"])
        assert tk.keywords == ["real kw"]

    def test_keywords_non_str_items_dropped_defensively(self):
        tk = TitleKeywordsLLM(title="A Study", keywords=["ml", 123, None])
        assert tk.keywords == ["ml"]

    def test_title_placeholder_scrubbed_to_empty(self):
        # An explicitly emitted placeholder degrades to ""; an omitted title
        # remains None and is replaced by the authoritative layout title later.
        tk = TitleKeywordsLLM(title="verbatim-string")
        assert tk.title == ""

    def test_title_legit_value_survives(self):
        # "A number of things" contains the word "number" but is not an
        # exact whole-value match — must not be scrubbed.
        tk = TitleKeywordsLLM(title="A number of things")
        assert tk.title == "A number of things"

    def test_biblio_field_placeholder_coerced_to_none(self):
        tk = TitleKeywordsLLM(title="A Study", journal="String", publisher="Number")
        assert tk.journal is None
        assert tk.publisher is None

    def test_core_metadata_title_and_keywords_scrubbed(self):
        meta = CoreMetadataLLM(
            title="verbatim-string",
            authors=[AuthorLLM(given="A", family="B")],
            keywords=["nlp", "Boolean"],
        )
        assert meta.title == ""
        assert meta.keywords == ["nlp"]

    def test_core_metadata_biblio_field_placeholder_coerced_to_none(self):
        meta = CoreMetadataLLM(
            title="A Study",
            authors=[AuthorLLM(given="A", family="B")],
            journal="Date-Time",
        )
        assert meta.journal is None

    def test_author_fields_placeholder_scrubbed_to_empty(self):
        # given/family/affiliation are non-Optional on PaperAuthor; a placeholder
        # degrades to "" (never None) so downstream str-typed code is safe.
        author = AuthorLLM(given="verbatim-string", family="Doe", affiliation="Integer")
        assert author.given == ""
        assert author.family == "Doe"
        assert author.affiliation == ""

    def test_enveloped_biblio_placeholder_and_none_string_scrubbed(self):
        # A subclass mode="before" validator (_coerce_biblio) can run before the
        # inherited envelope-unwrap; the biblio sanitiser must still fire on the
        # {"<SchemaName>": {...}} envelope shape for BOTH the "None"-string and
        # placeholder-token cases, on both models that carry biblio fields.
        tk = TitleKeywordsLLM.model_validate(
            {"TitleKeywordsLLM": {"title": "A Study", "journal": "None", "publisher": "Number"}}
        )
        assert tk.journal is None
        assert tk.publisher is None

        meta = CoreMetadataLLM.model_validate(
            {
                "CoreMetadataLLM": {
                    "title": "A Study",
                    "authors": [{"given": "A", "family": "B"}],
                    "journal": "null",
                    "first_page": "verbatim-string",
                }
            }
        )
        assert meta.journal is None
        assert meta.first_page is None

    # --- newly covered fields (abstract, published/"date", funding, affiliation) ---

    @pytest.mark.parametrize("token", ["verbatim-string", "string", "date", "Boolean", "  Number "])
    def test_titlekeywords_abstract_placeholder_to_none(self, token):
        tk = TitleKeywordsLLM(title="A Study", abstract=token)
        assert tk.abstract is None

    def test_titlekeywords_abstract_legit_survives(self):
        tk = TitleKeywordsLLM(title="A Study", abstract="We tested a number of hypotheses.")
        assert tk.abstract == "We tested a number of hypotheses."

    @pytest.mark.parametrize("token", ["verbatim-string", "string", "date", "Boolean"])
    def test_core_metadata_abstract_placeholder_to_none(self, token):
        meta = CoreMetadataLLM(
            title="A Study", authors=[AuthorLLM(given="A", family="B")], abstract=token
        )
        assert meta.abstract is None

    @pytest.mark.parametrize("token", ["date", "Date", "  date-time  "])
    def test_titlekeywords_published_placeholder_to_none(self, token):
        tk = TitleKeywordsLLM(title="A Study", published=token)
        assert tk.published is None

    @pytest.mark.parametrize("token", ["date", "Date", "date-time"])
    def test_core_metadata_published_placeholder_to_none(self, token):
        meta = CoreMetadataLLM(
            title="A Study", authors=[AuthorLLM(given="A", family="B")], published=token
        )
        assert meta.published is None

    def test_published_legit_value_survives(self):
        tk = TitleKeywordsLLM(title="A Study", published="2020-01-01")
        assert tk.published == "2020-01-01"

    @pytest.mark.parametrize("token", ["verbatim-string", "string", "date"])
    def test_funding_funder_placeholder_to_empty(self, token):
        # funder is non-Optional; a placeholder degrades to "" so the consuming
        # extractor drops the entry (funder empty).
        entry = FundingEntryLLM(funder=token, award_ids=[])
        assert entry.funder == ""

    def test_funding_award_ids_placeholder_dropped(self):
        entry = FundingEntryLLM(funder="NSF", award_ids=["verbatim-string", "R01-123", "string"])
        assert entry.award_ids == ["R01-123"]

    def test_funding_funder_legit_survives(self):
        entry = FundingEntryLLM(funder="National Science Foundation", award_ids=["834861"])
        assert entry.funder == "National Science Foundation"
        assert entry.award_ids == ["834861"]

    @pytest.mark.parametrize("field", ["institution", "department", "city", "country"])
    @pytest.mark.parametrize("token", ["verbatim-string", "string", "date", "Boolean"])
    def test_affiliation_string_fields_placeholder_to_none(self, field, token):
        aff = AffiliationLLM.model_validate({"index": 1, field: token})
        assert getattr(aff, field) is None

    def test_affiliation_legit_values_survive(self):
        aff = AffiliationLLM.model_validate(
            {"index": 1, "institution": "University of Twente", "city": "Enschede"}
        )
        assert aff.institution == "University of Twente"
        assert aff.city == "Enschede"

    def test_date_token_registered(self):
        # "date" joins the DSL placeholder set; exercised on a biblio field.
        tk = TitleKeywordsLLM(title="A Study", journal="date")
        assert tk.journal is None
