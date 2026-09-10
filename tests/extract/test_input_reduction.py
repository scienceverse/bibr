"""Input pruning must retain uncertain evidence and task ownership."""

from unittest import mock

import pytest

from bibr.extract.core_metadata import render_block_context, render_title_context
from bibr.extract.equation_extractor import equation_batches, needs_equation_fallback
from tests.extract.test_core_metadata_front_matter import (
    _candidate,
    _resolution,
    make_extractor_with_captured_core_call,
)


@pytest.mark.parametrize(
    "text",
    [
        "We followed the protocol (Smith, 2020).",
        "We used the instrument (Smith et al., 2020; Jones, 2021).",
        "Collection began (2020).",
        "The workflow is illustrated (see Figure 2).",
        "We used R (version 4.2.1).",
    ],
)
def test_clear_nonstatistical_parentheses_are_skipped(text):
    assert not needs_equation_fallback(text)


@pytest.mark.parametrize(
    "text",
    [
        "Odd statistic (t: 3.42; p: .003).",
        "The KMO measure was 0.764 (>0.60).",
        "Smith (2020) found an effect (5 vs 7).",
        "The mean was (2020).",
        "An unknown format (12, 34).",
        "Observed values (0.42, 0.71).",
        "With an OCR error (Smlth, 2O20) and (12: 34).",
        "Descriptives (SD 2020).",
        "A nested expression (test(28): .04).",
        "A non-English expression (moyenne: 2020).",
        "Model fit was acceptable if CFI and TLI were above 0.90 (Byrne, 2012).",
        "The unidentified measurement was (2020).",
        "An unfamiliar statistic (CFI 2020).",
        "Average (2020) was observed.",
    ],
)
def test_uncertain_and_statistical_parentheses_are_kept(text):
    assert needs_equation_fallback(text)


def test_batches_pack_short_rows_and_keep_long_unicode_sentences_intact():
    short = [(i, "Unparsed statistic (12, 34).") for i in range(21)]
    assert list(map(len, equation_batches(short))) == [10, 10, 1]
    long = (25, "統計結果" * 800 + " (12, 34)")
    batches = equation_batches(short[:2] + [long] + short[2:4], input_tokens=100)
    assert batches == [short[:2], [long], short[2:4]]
    assert sum(batches, []) == short[:2] + [long] + short[2:4]


def _front_matter():
    return _resolution(
        _candidate("c1", "Selected paper title", roles=frozenset({"title"}), text_ids=(1,)),
        _candidate("c2", "Alice Example, Bob Scholar", roles=frozenset({"byline"}), text_ids=(2,)),
        _candidate(
            "c3",
            "1 Department of Biology, Example University",
            roles=frozenset({"affiliation"}),
            text_ids=(3,),
        ),
        _candidate("c4", "Uncertain intervening content", roles=frozenset(), text_ids=(4,)),
        _candidate(
            "c5",
            "Published by Example University, Journal of Testing, Vol. 2",
            roles=frozenset({"affiliation"}),
            text_ids=(5,),
        ),
        _candidate(
            "c6", "Complete abstract evidence", roles=frozenset({"abstract"}), text_ids=(6,)
        ),
        _candidate(
            "c7",
            "Abstract continues at Example University",
            roles=frozenset({"affiliation", "abstract"}),
            text_ids=(7,),
        ),
    )


def test_title_context_preserves_unknown_publication_and_mixed_abstract_rows():
    resolution = _front_matter()
    full = render_block_context(resolution) + "\n[Author table]\nAdditional evidence"
    result = render_title_context(resolution, full_text=full)
    assert "Alice Example" not in result
    assert "1 Department" not in result
    assert "[Author details omitted]" in result
    for text in [
        "Uncertain intervening content",
        "Published by Example University",
        "Complete abstract evidence",
        "Abstract continues at Example University",
        "Additional evidence",
    ]:
        assert text in result


@pytest.mark.parametrize("roles", [{"title"}, {"abstract"}, set()])
def test_title_context_without_clear_boundaries_keeps_everything(roles):
    resolution = _resolution(
        _candidate("c1", "Ambiguous record", roles=frozenset(roles)),
        _candidate("c2", "Alice Example", roles=frozenset({"byline"})),
    )
    full = render_block_context(resolution)
    assert render_title_context(resolution, full_text=full) == full


@pytest.mark.parametrize(
    "per_task,merged,prune",
    [(True, False, True), (False, False, True), (True, True, True), (True, False, False)],
)
async def test_title_pruning_does_not_change_author_or_merged_input(
    monkeypatch, per_task, merged, prune
):
    resolution = _front_matter()
    extractor, llm = make_extractor_with_captured_core_call(resolution, monkeypatch)
    extractor._settings.llm.per_task_context = per_task
    extractor._settings.llm.merged_core_metadata = merged
    extractor._settings.llm.title_context = prune
    # Keep this focused on the request boundary, independent of author recovery.
    monkeypatch.setattr(extractor, "_recover_empty_authors", mock.AsyncMock(return_value=[]))
    await extractor.extract()
    (title_input,), kwargs = llm.extract_core_metadata.await_args
    assert ("Alice Example" in title_input) == (not per_task or merged or not prune)
    if per_task:
        assert "Alice Example" in kwargs["authors_text"]
        assert "Complete abstract evidence" in kwargs["classification_text"]


async def test_compact_equation_indices_still_map_to_original_text_ids(monkeypatch):
    from types import SimpleNamespace

    from bibr.clients.llm import LLMClient
    from bibr.clients.prompts import prompt_text

    client = LLMClient()
    monkeypatch.setattr(client, "_acquire_rate_limit", mock.AsyncMock())
    invoke = mock.AsyncMock(
        return_value=SimpleNamespace(
            equations=[
                SimpleNamespace(
                    sentence_index=1,
                    lhs="p",
                    df="",
                    comp="<",
                    rhs=".05",
                )
            ]
        )
    )
    monkeypatch.setattr(client, "_invoke_structured", invoke)
    result = await client.extract_equations([(17, "First (12)."), (809, "Second (p: .05).")])
    assert result[0].text_id == 809
    sent = prompt_text(invoke.await_args.args[1][0]["content"])
    assert "[1] Second (p: .05)." in sent
    assert "text_id=" not in sent


def test_compact_contract_preserves_field_shapes_and_absence_semantics():
    from bibr.schemas import CompactTitleKeywordsLLM, TitleKeywordsLLM

    def structural(schema):
        if isinstance(schema, dict):
            return {k: structural(v) for k, v in schema.items() if k != "description"}
        if isinstance(schema, list):
            return [structural(v) for v in schema]
        return schema

    assert structural(CompactTitleKeywordsLLM.model_json_schema()) == structural(
        TitleKeywordsLLM.model_json_schema()
    )
    for model in (TitleKeywordsLLM, CompactTitleKeywordsLLM):
        assert model(abstract=None)._abstract_explicitly_absent
        assert not model()._abstract_explicitly_absent
    for name in ("TitleKeywordsLLM", "CompactTitleKeywordsLLM"):
        result = CompactTitleKeywordsLLM.model_validate(
            {name: {"title": "Source title", "abstract": None, "published": "null"}}
        )
        assert result.title == "Source title"
        assert result._abstract_explicitly_absent
        assert result.published is None


@pytest.mark.parametrize("compact", [False, True])
async def test_compact_metadata_contract_is_selected_only_when_enabled(monkeypatch, compact):
    from bibr.clients.llm import LLMClient
    from bibr.clients.prompts import prompt_text, title_keywords_spec
    from bibr.schemas import CompactTitleKeywordsLLM, TitleKeywordsLLM

    client = LLMClient()
    client._settings.llm.compact_metadata_prompt = compact
    monkeypatch.setattr(client, "_acquire_rate_limit", mock.AsyncMock())
    invoke = mock.AsyncMock(return_value=TitleKeywordsLLM(title="Source title"))
    monkeypatch.setattr(client, "_invoke_structured", invoke)
    await client.extract_title_keywords("Source title")
    assert invoke.await_args.args[0] is (CompactTitleKeywordsLLM if compact else TitleKeywordsLLM)
    parts = invoke.await_args.args[1][0]["content"]
    assert "Source title" in prompt_text(parts)
    assert all(part["nuextract_role"] in {"document", "instructions"} for part in parts)
    assert title_keywords_spec(compact=compact).response_model is invoke.await_args.args[0]
