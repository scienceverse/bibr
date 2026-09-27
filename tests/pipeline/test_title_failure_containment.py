"""End to end: a paper whose title/keywords response is truncated now exports.

Uses the smoke pipeline's fakes (canned layout, native text layer, stubbed LLM
transport). Before, the truncated anchor call failed the file with no export.
"""

from bibr.config import Settings
from bibr.exceptions import LlmTimeoutError, LlmTruncatedError
from tests import test_pipeline_smoke as smoke


def _pipeline(monkeypatch, llm):
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ocr, "native_text_min_chars", 4)
    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", None)
    pipeline = LocalPipeline(crossref=False)
    rm = ResourceManager(
        memory_mode=pipeline.memory_mode,
        ocr_backend=pipeline.ocr_backend,
        ocr_url=pipeline.ocr_url,
        ocr_model=pipeline.ocr_model,
        device=pipeline.device,
        settings=pipeline.settings,
        layout=smoke.FakeLayout(),
        segmenter=smoke.FakeSegmenter(),
    )
    pipeline._resources = rm
    rm._ocr = smoke.FakeOcr()
    rm._llm_client = llm
    return pipeline


def _llm_failing_title(error):
    llm = smoke._fake_llm_client()
    answer = llm._invoke_structured

    async def invoke(response_model, *args, **kwargs):
        if response_model.__name__ == "TitleKeywordsLLM":
            llm.calls.append(response_model.__name__)
            raise error
        return await answer(response_model, *args, **kwargs)

    llm._invoke_structured = invoke
    return llm


async def test_truncated_title_call_exports_the_rest_of_the_paper(tmp_path, monkeypatch):
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    llm = _llm_failing_title(
        LlmTruncatedError("Failed to extract title/keywords", cause="token limit")
    )
    pipeline = _pipeline(monkeypatch, llm)

    result = await pipeline.process_file(pdf)
    await pipeline.aclose()

    # References, authors and classification survive the failed anchor call.
    assert [row["year"] for row in result["bib"]] == [2020, 2021]
    assert [row["family"] for row in result["author"]] == ["Doe"]
    assert result["metadata"]["paper_type"] == "empirical"
    # The layout title fills the empty title, as for any null model title.
    assert result["metadata"]["title"] == smoke._TITLE
    validation = result["extraction"]["validation"]
    [issue] = [i for i in validation["issues"] if i["code"] == "VAL_METADATA_FIELD_FAILED"]
    assert issue["blocking"] is True
    assert "reason:llm_truncated" in issue["evidence_ids"]
    assert validation["promotable"] is False


async def test_title_call_timeout_still_fails_the_file(tmp_path, monkeypatch):
    import pytest

    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    error = LlmTimeoutError("Failed to extract title/keywords", cause="timed out")
    pipeline = _pipeline(monkeypatch, _llm_failing_title(error))

    with pytest.raises(LlmTimeoutError):
        await pipeline.process_file(pdf)
    await pipeline.aclose()
