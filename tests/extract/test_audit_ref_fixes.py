"""Reference-extraction audit fixes.

C6 (silent span drop + positional mis-tag), M1 (``failures[0]`` IndexError),
M2 (refs the LLM quietly skipped), M3 (untrusted indices feeding
segment-anchored backfills), M6 (combined "55(7)" volume), L8 (single-entry
remainder batch excluded from the all-null-author rescue), M7 (native
truncation classified as non-degenerate and unsalvageable).
"""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.exceptions import UpstreamServiceError
from bibr.extract.ref_extractor import (
    ReferenceExtractor,
    _is_degenerate_ref_failure,
    _normalize_vol_issue,
)
from bibr.ner.decode import _FIELD_TO_PAPER_REF, map_fields_to_paper_ref
from bibr.paper_contents import PaperContents
from bibr.processing_warnings import WarningCode
from bibr.schemas import PaperReference, PaperReferenceLLM

REFS = [
    "Smith, J. (2020). A study. Journal of Things, 55(7), 1-10.",
    "Doe, A. (2021). Another study. Journal of Stuff, 12(3), 20-30.",
    "Roe, B. (2022). A third study. Journal of Items, 8(1), 40-50.",
]


def _extractor():
    contents = mock.Mock(spec=PaperContents)
    contents.processing_warnings = []
    return ReferenceExtractor(contents, llm_client=mock.Mock())


def _llm_ref(index, **kw):
    base = {
        "index": index,
        "title": f"Title {index}",
        "authors": "Smith, J.",
        "year": 2020,
        "container": "Journal of Things",
        "volume": "55",
        "first_page": "1",
    }
    base.update(kw)
    return PaperReferenceLLM(**base)


def _no_recovery(self, segs):
    """Aligned NER parser that recovers nothing, correctly shaped."""
    return [None] * len(segs)


def _ner_ref(i):
    return PaperReference(
        bib_id=i,
        title=f"Ner {i}",
        authors="Ner, A.",
        year=2020,
        container="Ner Journal",
        volume=None,
        first_page=None,
    )


class TestNerFallbackIsNeverSilent:
    """C6: the span vanished with only a log line when NER also failed."""

    async def test_a_failed_ner_recovery_is_recorded_on_the_paper(self):
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(side_effect=RuntimeError("503"))
        with patch.object(
            type(ext), "_parse_references_ner_aligned", side_effect=RuntimeError("CUDA OOM")
        ):
            with pytest.raises(UpstreamServiceError):
                await ext._parse_references_llm("\n".join(REFS), REFS)

        assert any(
            w.code == WarningCode.REF_PARSE_LOST and "NER fallback also failed" in w.message
            for w in ext.contents.processing_warnings
        ), ext.contents.processing_warnings

    async def test_a_dropped_slot_does_not_shift_later_refs(self):
        """The filtered NER variant compacted survivors toward the start."""
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(side_effect=RuntimeError("503"))

        # Slot 0 yields nothing; slots 1 and 2 survive.
        def fake(self, segs):
            # The whole batch goes to NER; slot 0 yields nothing.
            if len(segs) == len(REFS):
                return [None, _ner_ref(1), _ner_ref(2)]
            return [None] * len(segs)

        with patch.object(
            type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=fake
        ):
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        assert [r.title for r in refs] == ["Ner 1", "Ner 2"]


class TestAllBatchesEmpty:
    """M1: ``failures`` can be empty when every batch "succeeds" with 0 refs."""

    async def test_empty_result_raises_upstream_not_index_error(self):
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(return_value=[])
        with patch.object(
            type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=_no_recovery
        ):
            with pytest.raises(UpstreamServiceError):
                await ext._parse_references_llm("\n".join(REFS), REFS)


class TestSkippedSegmentsAreRecovered:
    """M2: a "successful" batch that silently returned fewer refs than asked."""

    async def test_missing_refs_are_recovered_via_ner(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        # The LLM returns refs 1 and 3 only — ref 2 was quietly skipped.
        ext.llm_client.extract_references = AsyncMock(return_value=[_llm_ref(1), _llm_ref(3)])
        with patch.object(
            type(ext), "_parse_references_ner_aligned", return_value=[_ner_ref(2)]
        ) as ner:
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        ner.assert_called_once()
        assert [r.title for r in refs] == ["Title 1", "Ner 2", "Title 3"]
        assert any(
            w.code == WarningCode.REF_PARSE_NER_FALLBACK and "skipped by the LLM" in w.message
            for w in ext.contents.processing_warnings
        ), ext.contents.processing_warnings

    async def test_a_complete_batch_triggers_no_recovery(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(
            return_value=[_llm_ref(1), _llm_ref(2), _llm_ref(3)]
        )
        with patch.object(type(ext), "_parse_references_ner_aligned") as ner:
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        ner.assert_not_called()
        assert len(refs) == 3
        assert ext.contents.processing_warnings == []


class TestUntrustedIndicesSuppressSegmentBackfill:
    """M3: a positional guess pointed backfills at a neighbour's segment."""

    # Segment 2 is the only one carrying a printed DOI, so a ref pointed at
    # it inherits that DOI verbatim.
    DOI_SEGMENTS = [
        "Smith, J. (2020). A study. Journal of Things, 55(7), 1-10.",
        "Doe, A. (2021). Another study. Journal of Stuff, 12(3), 20-30. "
        "https://doi.org/10.1000/doe-2021",
        "Roe, B. (2022). A third study. Journal of Items, 8(1), 40-50.",
    ]

    async def test_an_untrusted_ref_does_not_inherit_a_neighbours_doi(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        untrusted = _llm_ref(2, doi=None, volume="12", issue=None)
        untrusted.mark_index_untrusted()
        ext.llm_client.extract_references = AsyncMock(return_value=[untrusted])
        with patch.object(
            type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=_no_recovery
        ):
            refs = await ext._parse_references_llm("\n".join(self.DOI_SEGMENTS), self.DOI_SEGMENTS)

        # Its index is a positional guess, so segment 2's DOI is not evidence
        # about this reference.
        assert refs[0].doi is None
        assert refs[0].issue is None

    async def test_a_trusted_ref_still_gets_its_backfills(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        trusted = _llm_ref(2, doi=None, volume="12", issue=None)
        ext.llm_client.extract_references = AsyncMock(return_value=[trusted])
        with patch.object(
            type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=_no_recovery
        ):
            refs = await ext._parse_references_llm("\n".join(self.DOI_SEGMENTS), self.DOI_SEGMENTS)

        assert refs[0].doi == "10.1000/doe-2021"
        assert refs[0].issue == "3"

    def test_index_is_trusted_by_default(self):
        assert _llm_ref(1).index_trusted

    def test_the_flag_stays_out_of_the_llm_facing_schema(self):
        assert "_index_trusted" not in PaperReferenceLLM.model_json_schema().get("properties", {})
        assert "index_trusted" not in _llm_ref(1).model_dump()


class TestCombinedVolumeIssue:
    """M6: the LLM path never split a combined "55(7)"."""

    async def test_the_llm_path_splits_a_combined_volume(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(
            return_value=[_llm_ref(1, volume="55(7)", issue=None)]
        )
        with patch.object(
            type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=_no_recovery
        ):
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        assert (refs[0].volume, refs[0].issue) == ("55", "7")

    def test_the_ner_path_normalizer_is_the_one_being_reused(self):
        assert _normalize_vol_issue("55(7)", None, "") == ("55", "7")


class TestAllNullAuthorRescue:
    """L8: a single-entry remainder batch was excluded from the rescue."""

    async def test_a_one_ref_batch_is_eligible_for_rescue(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 1)
        ext = _extractor()
        single = REFS[:1]
        ext.llm_client.extract_references = AsyncMock(return_value=[_llm_ref(1, authors=None)])
        with patch.object(
            type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=_no_recovery
        ):
            refs = await ext._parse_references_llm(single[0], single)

        # Rescued from the segment's own printed byline.
        assert refs[0].authors and "Smith" in refs[0].authors


class TestNativeTruncationIsDegenerate:
    """M7: the native backend reports its own truncation, not Instructor's."""

    def _invalid(self, category, raw=""):
        from bibr.clients.nuextract import NuExtractInvalidOutput

        return NuExtractInvalidOutput(
            category=category,
            model="numind/NuExtract3",
            finish_reason="length",
            response_chars=len(raw),
            response_sha256="deadbeef",
            input_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cached_input_tokens=0,
            response_model="ReferenceList",
            salvage_raw=raw,
        )

    def test_native_truncation_counts_as_degenerate(self):
        assert _is_degenerate_ref_failure(self._invalid("truncated"))

    def test_other_native_failures_do_not(self):
        assert not _is_degenerate_ref_failure(self._invalid("schema_invalid"))

    def test_the_salvage_bridge_finds_the_raw_completion(self):
        from bibr.clients.llm import incomplete_output_text

        raw = '{"references": [{"index": 1, "title": "A"}'
        assert incomplete_output_text(self._invalid("truncated", raw)) == raw

    def test_the_raw_text_stays_out_of_the_representation(self):
        """The class documents that it can cross log/metric/API boundaries."""
        exc = self._invalid("truncated", "SENSITIVE COMPLETION TEXT")

        assert "SENSITIVE" not in str(exc)
        assert "SENSITIVE" not in repr(exc)

    def test_non_truncated_categories_carry_no_raw_text(self):
        from bibr.clients.nuextract import _invalid_output

        contract = MagicMock()
        contract.response_model.__name__ = "X"
        exc = _invalid_output(
            "schema_invalid",
            raw="whatever",
            finish_reason="stop",
            contract=contract,
            model="m",
            input_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cached_input_tokens=0,
        )
        assert exc.salvage_raw == ""


class TestNerPathSetsInPress:
    """The NER tag set has no in-press concept, so the tagger drops the year
    *and* the field dict omitted the flag — under the default
    REF_PARSE_STRATEGY=ner an "(in press)" reference exported year: null,
    is_in_press: false and its in-text citation could never match."""

    def test_in_press_segment_sets_the_flag(self):
        from bibr.extract import ref_extractor

        ext = _extractor()
        ref_text = (
            "Robertson, C. E., & Van Bavel, J. J. (in press). Inside the funhouse mirror factory."
        )

        class _Parser:
            @staticmethod
            def parse_batch(segments):
                return [
                    {
                        "title": "Inside the funhouse mirror factory",
                        "authors": "Robertson, C. E.; Van Bavel, J. J.",
                    }
                    for _ in segments
                ]

        with patch.object(ref_extractor, "_get_ner_parser", return_value=_Parser()):
            aligned = ext._parse_references_ner_aligned([ref_text])

        assert aligned[0] is not None
        assert aligned[0].is_in_press is True

    def test_an_ordinary_segment_leaves_the_flag_false(self):
        from bibr.extract import ref_extractor

        ext = _extractor()

        class _Parser:
            @staticmethod
            def parse_batch(segments):
                return [
                    {"title": "A study", "authors": "Smith, J.", "year": "2020"} for _ in segments
                ]

        with patch.object(ref_extractor, "_get_ner_parser", return_value=_Parser()):
            aligned = ext._parse_references_ner_aligned([REFS[0]])

        assert aligned[0] is not None
        assert aligned[0].is_in_press is False


class TestNerPathKeepsNerOnlyFields:
    """ARXIV/PMID/SERIES/ACCESS_DATE/NOTE must survive reference assembly.

    The decoder maps all five (``test_every_tagged_field_reaches_paper_reference``
    guards that), but ``_parse_references_ner_aligned`` rebuilt the field dict
    key by key without them, so every value was dropped again before export —
    on the default REF_PARSE_STRATEGY=ner path and on the NER fallbacks of the
    LLM modes, which reuse it.
    """

    SEGMENT = (
        "Smith, J. (2020). A study of recall. Lecture Notes in Computer Science. "
        "arXiv:1803.04219. PMID: 28919116. Accessed 12 March 2020. In Russian."
    )
    NER_ONLY = {
        "arxiv": "1803.04219",
        "pmid": "28919116",
        "series": "Lecture Notes in Computer Science",
        "access_date": "Accessed 12 March 2020",
        "note": "In Russian",
    }
    PARSED = {"title": "A study of recall", "authors": "Smith, J.", **NER_ONLY}

    @staticmethod
    def _parser_returning(fields):
        class _Parser:
            @staticmethod
            def parse_batch(segments):
                return [dict(fields) for _ in segments]

        return patch("bibr.extract.ref_extractor._get_ner_parser", return_value=_Parser())

    def test_they_reach_the_reference(self):
        with self._parser_returning(self.PARSED):
            (ref,) = _extractor()._parse_references_ner([self.SEGMENT])

        assert {name: getattr(ref, name) for name in self.NER_ONLY} == self.NER_ONLY

    def test_no_field_the_decoder_emits_is_dropped(self):
        """Assembly-level twin of the decoder guard: a field added to
        ``_FIELD_TO_PAPER_REF`` fails here until assembly passes it on."""
        # Values finalize keeps as they are; every other field takes any string.
        clean = {
            "doi": "10.1000/xyz123",
            "volume": "12",
            "issue": "3",
            "first_page": "100",
            "last_page": "115",
        }
        fields = {
            name: clean.get(name, f"{name} value") for name in set(_FIELD_TO_PAPER_REF.values())
        }
        fields["year"] = 2020
        fields["year_suffix"] = "a"
        with self._parser_returning(fields):
            (ref,) = _extractor()._parse_references_ner([REFS[0]])

        assert sorted(name for name in fields if getattr(ref, name) is None) == []

    @pytest.mark.parametrize(
        ("parse", "llm_call"),
        [
            ("_parse_references_llm", "extract_references"),
            ("_parse_references_llm_chunked", "extract_references_chunk"),
        ],
    )
    async def test_the_llm_modes_ner_fallback_keeps_them(self, parse, llm_call):
        ext = _extractor()
        # No layout regions: the chunked mode groups the segments instead.
        ext.contents.region_summaries = []
        setattr(ext.llm_client, llm_call, AsyncMock(side_effect=RuntimeError("503")))
        with self._parser_returning(self.PARSED):
            refs = await getattr(ext, parse)("\n".join(REFS), REFS)

        assert len(refs) == len(REFS)
        for ref in refs:
            assert {name: getattr(ref, name) for name in self.NER_ONLY} == self.NER_ONLY


class TestNerPathKeepsYearSuffix:
    """The "a" of an author-year "2020a" reaches the reference on the NER path.

    ``year_suffix`` is what tells "(Smith, 2020a)" and "(Smith, 2020b)" apart
    when in-text citations are linked to references.
    """

    @staticmethod
    def _parse(year: str) -> PaperReference:
        decoded = map_fields_to_paper_ref({"TITLE": "A study", "AUTHOR": "Smith, J.", "YEAR": year})

        class _Parser:
            @staticmethod
            def parse_batch(segments):
                return [dict(decoded) for _ in segments]

        segment = f"Smith, J. ({year}). A study. Journal of Things, 55(7), 1-10."
        with patch("bibr.extract.ref_extractor._get_ner_parser", return_value=_Parser()):
            (ref,) = _extractor()._parse_references_ner([segment])
        return ref

    def test_the_letter_after_the_year_is_kept(self):
        ref = self._parse("2020a")
        assert (ref.year, ref.year_suffix) == (2020, "a")

    @pytest.mark.parametrize("year", ["2020", "2020-2021", "2020ab"])
    def test_no_letter_no_suffix(self, year):
        ref = self._parse(year)
        assert (ref.year, ref.year_suffix) == (2020, None)


class TestStubRefsAreNotCountedAsCovered:
    """Round 3: a ref with no title AND no authors is dropped further down.

    ``_sequence_references`` filters it out, so counting it as covered blocked
    the NER recovery for that slot and then deleted the entry outright —
    shifting every later ``bib_id`` so the printed ``[3]`` resolved to what was
    printed as ``[2]``. Nothing warned; 2 of 3 clears the under-yield checks.
    """

    async def test_a_stub_ref_slot_is_recovered_via_ner(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(
            return_value=[
                _llm_ref(1),
                _llm_ref(2, title=None, authors=None),
                _llm_ref(3),
            ]
        )
        with patch.object(
            type(ext), "_parse_references_ner_aligned", return_value=[_ner_ref(2)]
        ) as ner:
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        ner.assert_called_once()
        assert [r.title for r in refs] == ["Title 1", "Ner 2", "Title 3"]
        assert [r.bib_id for r in refs] == [1, 2, 3]

    async def test_the_stub_is_not_kept_alongside_its_replacement(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(
            return_value=[_llm_ref(1), _llm_ref(2, title=None, authors=None), _llm_ref(3)]
        )
        with patch.object(type(ext), "_parse_references_ner_aligned", return_value=[_ner_ref(2)]):
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        assert len(refs) == 3

    async def test_an_unrecoverable_stub_is_reported_not_silently_dropped(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(
            return_value=[_llm_ref(1), _llm_ref(2, title=None, authors=None), _llm_ref(3)]
        )
        with patch.object(
            type(ext), "_parse_references_ner_aligned", autospec=True, side_effect=_no_recovery
        ):
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        assert [r.title for r in refs] == ["Title 1", "Title 3"]
        assert any(
            w.code == WarningCode.REF_PARSE_NER_FALLBACK and "skipped by the LLM" in w.message
            for w in ext.contents.processing_warnings
        ), ext.contents.processing_warnings

    async def test_a_ner_recovery_that_is_itself_a_stub_is_not_kept(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(
            return_value=[_llm_ref(1), _llm_ref(2, title=None, authors=None), _llm_ref(3)]
        )
        stub_ner = PaperReference(
            bib_id=2,
            title="",
            authors=None,
            year=None,
            container=None,
            volume=None,
            first_page=None,
        )
        with patch.object(type(ext), "_parse_references_ner_aligned", return_value=[stub_ner]):
            refs = await ext._parse_references_llm("\n".join(REFS), REFS)

        assert [r.title for r in refs] == ["Title 1", "Title 3"]
