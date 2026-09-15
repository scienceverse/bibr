"""Field failure retains a blocked scoped export while other articles finish."""

import asyncio
from unittest.mock import AsyncMock

from bibr.clients.llm import LLMClient
from bibr.clients.structured import StructuredResponseError
from bibr.document_api import DocumentResult
from bibr.extract.extractor import MetadataExtractor
from bibr.models import PaperReference
from bibr.pipeline import document
from bibr.schemas import AuthorLLM, AuthorsLLM, PaperClassificationLLM, TitleKeywordsLLM
from tests.pipeline.test_document_pipeline import _pipeline


async def test_malformed_metadata_preserves_authors_and_references_in_partial_paper(monkeypatch):
    pipeline, observations = _pipeline(monkeypatch)
    continued = []
    plan = document.build_stage_plan

    class LaterStage:
        name = "later_stage"

        async def run(self, ctx):
            continued.append(ctx.file_states[0].prepared_front_matter.selected_block_id)

    monkeypatch.setattr(
        document,
        "build_stage_plan",
        lambda **kw: (
            *plan(**kw)[:2],
            LaterStage(),
            *plan(**kw)[2:],
        ),
    )

    async def metadata(scoped, *args, **kwargs):
        selected = scoped.front_matter_resolution
        first = selected.selected_block_id == "front-matter-block-1"
        observations["records"].append(selected.selected_block_id)
        client = LLMClient(settings=pipeline.settings)
        forbidden = AsyncMock(side_effect=AssertionError("No service calls in this fixture"))
        monkeypatch.setattr(client._backend, "create", forbidden)
        refs_started = asyncio.Event()
        refs_completed = asyncio.Event()

        async def title(*args, **kwargs):
            await refs_started.wait()
            if first:
                raise StructuredResponseError("non_json")
            return TitleKeywordsLLM(
                title=scoped.detected_title,
                abstract="The controlled study measured seedling development.",
            )

        monkeypatch.setattr(client, "extract_title_keywords", AsyncMock(side_effect=title))
        monkeypatch.setattr(
            client,
            "extract_authors",
            AsyncMock(
                return_value=AuthorsLLM(
                    authors=[
                        AuthorLLM(
                            given="Mara" if first else "Talia", family="Quill" if first else "Vale"
                        )
                    ]
                )
            ),
        )
        monkeypatch.setattr(
            client,
            "extract_paper_classification",
            AsyncMock(return_value=PaperClassificationLLM(paper_type="empirical")),
        )
        extractor = MetadataExtractor(
            scoped, llm_client=client, settings=pipeline.settings, front_matter_resolution=selected
        )
        monkeypatch.setattr(
            extractor.core,
            "_classify_paper",
            AsyncMock(
                return_value=(
                    "empirical",
                    "Natural Sciences",
                    "Biological Sciences",
                    0.9,
                    0.9,
                )
            ),
        )

        async def references(frame):
            refs_started.set()
            await asyncio.sleep(0.01)
            refs_completed.set()
            return [
                PaperReference(
                    bib_id=1,
                    title=frame.iloc[0].text,
                    first_page="1",
                    volume="1",
                    authors="River A.",
                    year=2020,
                    container="Forest Journal",
                    text_id=int(frame.iloc[0].text_id),
                )
            ]

        monkeypatch.setattr(extractor, "_extract_references", references)
        result = await extractor.extract_all_metadata()
        assert refs_completed.is_set()
        forbidden.assert_not_awaited()
        client.extract_title_keywords.assert_awaited_once()
        client.extract_authors.assert_awaited_once()
        kwargs["validation_issue_sink"].extend(extractor.validation_issues)
        return result

    monkeypatch.setattr("bibr.pipeline.stages.post_parse._extract_metadata_and_equations", metadata)

    payload = await asyncio.wait_for(document.process_document(pipeline, "collected.pdf"), 3)

    assert payload["status"] == "partial"
    blocked, success = payload["records"]
    assert blocked["status"] == "unresolved" and blocked["paper"] is None
    assert "VAL_METADATA_FIELD_FAILED" in blocked["reason_flags"]
    partial = blocked["partial_paper"]
    assert partial["source"] == payload["source"]
    assert partial["author"][0]["family"] == "Quill"
    assert "Study 1." in partial["bib"][0]["title"]
    assert "Study 2." not in str(partial)
    assert "Talia" not in str(partial)
    assert partial["validation"]["promotable"] is False
    assert partial["validation"]["blocking"] >= 1
    assert success["status"] == "extracted" and success["partial_paper"] is None
    assert success["paper"]["author"][0]["family"] == "Vale"
    assert "Study 2." in success["paper"]["bib"][0]["title"]
    assert continued == [success["record_id"]]
    assert len(observations["records"]) == 2
    result = DocumentResult(payload)
    assert len(result.papers) == 1 and not result.ok
    assert result.records[0].paper is None
    assert result.records[0].partial_paper.data["author"][0]["family"] == "Quill"
