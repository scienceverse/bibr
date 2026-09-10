"""qualification_provenance — the deployment-qualification surface.

Covers the pure aggregator (``build_qualification_provenance``), the client
first-writer-wins protocol-hash accumulator, the ``on_protocol_hashes`` backend
callback, and the end-to-end export surface an external gate reads.
"""

import re
from unittest.mock import AsyncMock, MagicMock

from bibr.export.qualification_provenance import (
    DeploymentIdentity,
    build_qualification_provenance,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _identity(**over) -> DeploymentIdentity:
    base = {
        "bibr_sha": "a" * 40,
        "platform_sha": "b" * 40,
        "model_id": "numind/NuExtract3-FP8",
        "model_revision": "rev-1",
        "jinja_sha256": "c" * 64,
        "structured_backend": "nuextract-native",
        "temperature": 0.0,
        "thinking_mode": False,
    }
    base.update(over)
    return DeploymentIdentity(**base)


def _native_bucket(**over) -> dict:
    bucket = {
        "logical_calls": 1,
        "attempts": 1,
        "native_attempts": 1,
        "instructor_attempts": 0,
        "protocol_fallbacks": 0,
        "protocol_fallbacks_recovered": 0,
        "native_invalid_outputs": 0,
    }
    bucket.update(over)
    return bucket


class TestAggregator:
    def test_native_all_valid(self):
        prov = build_qualification_provenance(
            usage_by_label={"extract_title_keywords": _native_bucket()},
            protocol_hashes_by_label={},
            identity=_identity(),
        )
        assert prov is not None
        assert prov["structured_backend"] == "nuextract-native"
        assert prov["raw_native_valid"] is True
        assert prov["raw_native_category"] is None
        assert prov["fallback_attempted"] is False
        assert prov["fallback_outcome"] == "not_attempted"
        assert prov["physical_request_count"] == 1
        assert prov["logical_request_count"] == 1
        assert prov["protocol_hashes"] == {}
        # Identity passthrough.
        assert prov["bibr_sha"] == "a" * 40
        assert prov["platform_sha"] == "b" * 40
        assert prov["model_id"] == "numind/NuExtract3-FP8"
        assert prov["model_revision"] == "rev-1"
        assert prov["jinja_sha256"] == "c" * 64
        assert prov["temperature"] == 0.0
        assert prov["thinking_mode"] is False

    def test_native_one_invalid_recovered(self):
        bucket = _native_bucket(
            attempts=2,
            native_attempts=1,
            instructor_attempts=1,
            protocol_fallbacks=1,
            protocol_fallbacks_recovered=1,
            native_invalid_outputs=1,
            native_invalid_non_json=1,
        )
        prov = build_qualification_provenance(
            usage_by_label={"extract_authors": bucket},
            protocol_hashes_by_label={},
            identity=_identity(),
        )
        assert prov["raw_native_valid"] is False
        assert prov["raw_native_category"] == "non_json"
        assert prov["fallback_attempted"] is True
        assert prov["fallback_outcome"] == "recovered"
        assert prov["physical_request_count"] == 2
        assert prov["logical_request_count"] == 1

    def test_native_invalid_not_recovered(self):
        bucket = _native_bucket(
            attempts=1,
            protocol_fallbacks=1,
            protocol_fallbacks_recovered=0,
            native_invalid_outputs=1,
            native_invalid_truncated=1,
        )
        prov = build_qualification_provenance(
            usage_by_label={"extract_authors": bucket},
            protocol_hashes_by_label={},
            identity=_identity(),
        )
        assert prov["raw_native_valid"] is False
        assert prov["raw_native_category"] == "truncated"
        assert prov["fallback_attempted"] is True
        assert prov["fallback_outcome"] == "failed"

    def test_category_fixed_order_scans_first_present(self):
        # Both empty and truncated present -> "empty" wins by fixed order.
        bucket = _native_bucket(
            native_invalid_outputs=2,
            native_invalid_truncated=1,
            native_invalid_empty=1,
        )
        prov = build_qualification_provenance(
            usage_by_label={"a": bucket},
            protocol_hashes_by_label={},
            identity=_identity(),
        )
        assert prov["raw_native_category"] == "empty"

    def test_category_later_bucket_only(self):
        bucket = _native_bucket(
            native_invalid_outputs=1,
            native_invalid_schema_invalid=1,
        )
        prov = build_qualification_provenance(
            usage_by_label={"a": bucket},
            protocol_hashes_by_label={},
            identity=_identity(),
        )
        assert prov["raw_native_category"] == "schema_invalid"

    def test_instructor_arm(self):
        prov = build_qualification_provenance(
            usage_by_label={"extract_authors": {"logical_calls": 1, "attempts": 1}},
            protocol_hashes_by_label={},
            identity=_identity(structured_backend="instructor", temperature=0.7),
        )
        assert prov["structured_backend"] == "instructor"
        assert prov["raw_native_valid"] is None
        assert prov["raw_native_category"] is None
        assert prov["fallback_attempted"] is False
        assert prov["fallback_outcome"] == "not_attempted"
        assert prov["temperature"] == 0.7

    def test_empty_input_returns_none(self):
        assert (
            build_qualification_provenance(
                usage_by_label={},
                protocol_hashes_by_label={},
                identity=_identity(),
            )
            is None
        )

    def test_protocol_hashes_present_but_no_usage_still_builds(self):
        prov = build_qualification_provenance(
            usage_by_label={},
            protocol_hashes_by_label={"extract_authors": {"a": "1", "b": "2", "c": "3"}},
            identity=_identity(),
        )
        assert prov is not None
        assert prov["physical_request_count"] == 0
        assert prov["logical_request_count"] == 0

    def test_physical_logical_sums_across_labels(self):
        prov = build_qualification_provenance(
            usage_by_label={
                "a": _native_bucket(attempts=2, logical_calls=1),
                "b": _native_bucket(attempts=3, logical_calls=2),
            },
            protocol_hashes_by_label={},
            identity=_identity(),
        )
        assert prov["physical_request_count"] == 5
        assert prov["logical_request_count"] == 3

    def test_protocol_hashes_passthrough(self):
        hashes = {
            "extract_authors": {
                "request_document_sha256": "d" * 64,
                "instruction_sha256": "e" * 64,
                "converted_template_sha256": "f" * 64,
            }
        }
        prov = build_qualification_provenance(
            usage_by_label={"extract_authors": _native_bucket()},
            protocol_hashes_by_label=hashes,
            identity=_identity(),
        )
        assert prov["protocol_hashes"] == hashes


class TestProtocolHashAccumulator:
    def test_first_writer_wins_per_label(self):
        from bibr.clients.llm import LLMClient, _usage_file_hash, _usage_label

        client = LLMClient()
        client._track_usage = True
        file_token = _usage_file_hash.set("file#1")
        label_token = _usage_label.set("extract_authors")
        try:
            client._record_protocol_hashes({"k": "native"})
            # A later write under the same label (e.g. the instructor fallback)
            # must not overwrite the captured native hashes.
            client._record_protocol_hashes({"k": "instructor"})
        finally:
            _usage_label.reset(label_token)
            _usage_file_hash.reset(file_token)

        popped = client.protocol_hashes_pop_file("file#1")
        assert popped == {"extract_authors": {"k": "native"}}
        # Bucket evicted on pop.
        assert client.protocol_hashes_pop_file("file#1") == {}


class TestInstructorBackendProtocolHashes:
    async def test_on_protocol_hashes_fires_one_triplet_of_hex(self):
        from bibr.clients.llm import LLMClient
        from bibr.schemas import TitleKeywordsLLM

        client = LLMClient()
        captured: list[dict] = []

        class _FakeInstructor:
            async def create_with_completion(self, *, response_model, messages, max_retries, **kw):
                return "MODEL", "COMPLETION"

            async def create(self, *, response_model, messages, max_retries, **kw):
                return "MODEL"

        result, _ = await client._backend.create(
            response_model=TitleKeywordsLLM,
            system="SYS",
            messages=[{"role": "user", "content": "hi"}],
            want_completion=False,
            client_override=_FakeInstructor(),
            on_protocol_hashes=captured.append,
        )

        assert result == "MODEL"
        assert len(captured) == 1
        triplet = captured[0]
        assert set(triplet) == {
            "request_document_sha256",
            "instruction_sha256",
            "converted_template_sha256",
        }
        assert all(_HEX64.match(v) for v in triplet.values())


class TestExportSurfaceEndToEnd:
    async def test_native_arm_export_carries_all_gate_fields(self, monkeypatch):
        import bibr.pipeline.stages.post_parse as post_parse_module
        from bibr.clients.nuextract import (
            NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256,
            NUEXTRACT3_FP8_EXPECTED_REVISION,
        )
        from bibr.config import snapshot_settings
        from bibr.export.json_export import export_paper_to_json
        from bibr.models import PaperMetadata

        settings = snapshot_settings()
        settings.llm.structured_backend = "nuextract-native"
        settings.llm.model = "numind/NuExtract3-FP8"
        settings.llm.temperature = 0.0
        settings.llm.model_revision = None
        settings.llm.jinja_sha256 = None
        settings.pipeline.bibr_sha = "1" * 40
        settings.pipeline.platform_sha = "2" * 40

        client = MagicMock()
        client._track_usage = True
        client.resolved_structured_backend = "nuextract-native"
        client.usage_pop_file.return_value = {}
        client.usage_labels_pop_file.return_value = {
            ("extract_title_keywords", settings.llm.provider, settings.llm.model): {
                "logical_calls": 1,
                "attempts": 1,
                "native_attempts": 1,
                "native_invalid_outputs": 0,
            }
        }
        client.protocol_hashes_pop_file.return_value = {
            "extract_title_keywords": {
                "request_document_sha256": "d" * 64,
                "instruction_sha256": "e" * 64,
                "converted_template_sha256": "f" * 64,
            }
        }

        async def classify(*_a, **_k):
            return None

        async def extract(*_a, **_k):
            return PaperMetadata(doi="10.1/x", title="T")

        monkeypatch.setattr(post_parse_module, "_classify_sections", classify)
        monkeypatch.setattr(post_parse_module, "_normalize_section_structure", classify)
        monkeypatch.setattr(post_parse_module, "_extract_metadata_and_equations", extract)
        monkeypatch.setattr(post_parse_module, "_link_citations", AsyncMock())
        monkeypatch.setattr(
            "bibr.extract.research_integrity.extract_structured_integrity", AsyncMock()
        )

        paper = await post_parse_module.post_parse(
            _root_only_contents(),
            "source.pdf",
            "source-hash",
            llm_client=client,
            settings=settings,
        )

        prov = paper.qualification_provenance
        assert prov is not None
        assert prov["model_revision"] == NUEXTRACT3_FP8_EXPECTED_REVISION
        assert prov["jinja_sha256"] == NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256

        exported = export_paper_to_json(paper, validate=False)
        gate = exported["qualification_provenance"]
        required = [
            "bibr_sha",
            "platform_sha",
            "model_id",
            "model_revision",
            "jinja_sha256",
            "structured_backend",
            "thinking_mode",
            "protocol_hashes",
            "fallback_attempted",
            "fallback_outcome",
            "physical_request_count",
            "logical_request_count",
            "temperature",
            "raw_native_valid",
        ]
        for field in required:
            assert gate[field] is not None, field
        assert gate["structured_backend"] == "nuextract-native"
        assert gate["bibr_sha"] == "1" * 40
        assert gate["platform_sha"] == "2" * 40
        assert gate["raw_native_valid"] is True

    async def test_instructor_arm_reports_null_temperature(self, monkeypatch):
        # The manifest declares the instructor arm's temperature as null, and
        # the gate flags observed != requested for identity fields (temperature
        # included). So a non-native backend must report temperature=None, not
        # the ambient sampling temperature, or the provenance gate false-fails.
        import bibr.pipeline.stages.post_parse as post_parse_module
        from bibr.config import snapshot_settings
        from bibr.export.json_export import export_paper_to_json
        from bibr.models import PaperMetadata

        settings = snapshot_settings()
        settings.llm.structured_backend = "instructor"
        settings.llm.model = "numind/NuExtract3-FP8"
        settings.llm.temperature = 0.2
        settings.llm.model_revision = None
        settings.llm.jinja_sha256 = None
        settings.pipeline.bibr_sha = "1" * 40
        settings.pipeline.platform_sha = "2" * 40

        client = MagicMock()
        client._track_usage = True
        client.resolved_structured_backend = "instructor"
        client.usage_pop_file.return_value = {}
        client.usage_labels_pop_file.return_value = {
            ("extract_title_keywords", settings.llm.provider, settings.llm.model): {
                "logical_calls": 1,
                "attempts": 1,
                "instructor_attempts": 1,
            }
        }
        client.protocol_hashes_pop_file.return_value = {
            "extract_title_keywords": {
                "request_document_sha256": "d" * 64,
                "instruction_sha256": "e" * 64,
                "converted_template_sha256": "f" * 64,
            }
        }

        async def classify(*_a, **_k):
            return None

        async def extract(*_a, **_k):
            return PaperMetadata(doi="10.1/x", title="T")

        monkeypatch.setattr(post_parse_module, "_classify_sections", classify)
        monkeypatch.setattr(post_parse_module, "_normalize_section_structure", classify)
        monkeypatch.setattr(post_parse_module, "_extract_metadata_and_equations", extract)
        monkeypatch.setattr(post_parse_module, "_link_citations", AsyncMock())
        monkeypatch.setattr(
            "bibr.extract.research_integrity.extract_structured_integrity", AsyncMock()
        )

        paper = await post_parse_module.post_parse(
            _root_only_contents(),
            "source.pdf",
            "source-hash",
            llm_client=client,
            settings=settings,
        )

        gate = export_paper_to_json(paper, validate=False)["qualification_provenance"]
        # The gated axis for a non-native arm: temperature and raw_native_valid
        # are None; every common identity/protocol field stays non-None.
        assert gate["temperature"] is None
        assert gate["raw_native_valid"] is None
        assert gate["structured_backend"] == "instructor"
        for field in (
            "bibr_sha",
            "platform_sha",
            "model_id",
            "model_revision",
            "jinja_sha256",
            "structured_backend",
            "thinking_mode",
            "protocol_hashes",
            "fallback_attempted",
            "fallback_outcome",
            "physical_request_count",
            "logical_request_count",
        ):
            assert gate[field] is not None, field
        assert gate["fallback_outcome"] == "not_attempted"


def _root_only_contents():
    from bibr.paper_contents import PaperContents, PaperSection

    return PaperContents(
        sentences=[],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: ""},
        detected_title="T",
    )
