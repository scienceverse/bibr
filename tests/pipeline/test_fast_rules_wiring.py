"""The local rules a fast run leans on are switched by the run, not by its callers.

Fast mode turns them on and a full run, with or without an LLM, leaves them off, so
a full run reads the page exactly as it did before the rules existed.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bibr.clients.skipped import SkippedLlmClient
from bibr.config import GlobalSettings
from bibr.run_mode import fast_settings
from tests.pipeline.test_fast_mode import (
    _ABSTRACT,
    _PAGE,
    _REFERENCES,
    _TITLE,
    PageTagger,
    _count_llm_reaches,
    fast_page_run,
)

_RULES = ("FRONT_TAGGER_RULES", "EQUATION_REPAIR_GLITCHES", "CITATION_RULE_TIER3")


def test_the_rules_are_off_in_a_default_run():
    settings = GlobalSettings()
    assert [getattr(settings, name) for name in _RULES] == [False, False, False]


def test_a_fast_run_turns_the_rules_on():
    fast = fast_settings(GlobalSettings())
    assert [getattr(fast, name) for name in _RULES] == [True, True, True]


def test_the_run_picks_the_settings_the_rules_follow():
    from bibr.local.pipeline import LocalPipeline

    pipeline = LocalPipeline(crossref=False)
    config = pipeline._config

    def flags(**changes):
        settings = pipeline._settings_for(dataclasses.replace(config, **changes))
        return [getattr(settings, name) for name in _RULES]

    assert flags(mode="fast") == [True, True, True]
    assert flags(mode="full") == [False, False, False]
    # Structure only wins over fast: it never takes the overlay.
    assert flags(mode="full", no_llm=True) == [False, False, False]
    assert flags(mode="fast", no_llm=True) == [False, False, False]
    # The overlay is a copy: the pipeline's own settings stay as they were.
    assert [getattr(pipeline.settings, name) for name in _RULES] == [False, False, False]


async def test_the_parse_stage_reads_the_rules_flag_from_the_runs_settings(monkeypatch):
    from bibr.front_tagger import runtime as runtime_mod
    from bibr.pipeline.stages import parse_segment

    seen: dict[str, bool] = {}

    def fake_tag_front_matter(tagger, regions, *, first_page_index=0, rules=False):
        seen["rules"] = rules
        return None

    monkeypatch.setattr(runtime_mod, "tag_front_matter", fake_tag_front_matter)

    class _RM:
        def ensure_front_tagger(self, settings=None):
            return object()

    await parse_segment._tag_front_matter(
        _RM(), [], first_page_index=0, settings=GlobalSettings(FRONT_TAGGER_RULES=True)
    )
    assert seen == {"rules": True}

    await parse_segment._tag_front_matter(_RM(), [], first_page_index=0, settings=GlobalSettings())
    assert seen == {"rules": False}


@pytest.mark.parametrize("rule_tier3", [True, False])
async def test_the_citation_step_hands_the_linker_the_flag(monkeypatch, rule_tier3):
    from bibr.pipeline.stages import post_parse
    from bibr.structure import citation_linker

    linker = AsyncMock(return_value=[])
    monkeypatch.setattr(citation_linker, "detect_bib_xrefs", linker)
    contents = SimpleNamespace(
        sentences=[], sections=[], xrefs=[], citation_receipt=None, processing_warnings=[]
    )

    await post_parse._link_citations(
        contents,
        SimpleNamespace(references=[]),
        "hash",
        SkippedLlmClient(),
        rule_tier3=rule_tier3,
    )

    assert linker.await_args.kwargs["rule_tier3"] is rule_tier3


# ---------------------------------------------------------------------------
# One page through the whole pipeline: what it reads in each mode.
# ---------------------------------------------------------------------------

_INSTITUTION = "Department of Psychology, University of Example, Utrecht, The Netherlands"
# The byline is followed by an institution line the tagger leaves untagged; the
# results hold a statistic whose "=" and "<" the text layer printed as "5" and ",";
# and "Smith-Lee" is a double surname that only the rules read as the listed "Smith".
_RULES_PAGE = (
    _PAGE[0],
    _PAGE[1],
    ("text", 10, _INSTITUTION),
    *_PAGE[2:5],
    ("text", 10, "Smith-Lee (2018) showed that attention matters. Jones (2019) studied span."),
    _PAGE[6],
    ("text", 10, "Recall was higher with cues, t(28) 5 3.42, p , .001."),
    ("paragraph_title", 14, "References"),
    ("reference", 10, _REFERENCES[0]),
    ("reference", 10, _REFERENCES[1]),
)


def _linked(result) -> list[tuple[str, int]]:
    return [(x["xref_type"], x["target_id"]) for x in result["xref"]]


async def test_a_fast_run_reads_the_page_with_the_rules(tmp_path, monkeypatch):
    from bibr.export.json_export import validate_export

    reached = _count_llm_reaches(monkeypatch)
    result = await fast_page_run(tmp_path, monkeypatch, tagger=PageTagger(), page=_RULES_PAGE)

    assert reached == []
    assert validate_export(result) == []
    # The statistic read through the text layer's glitches.
    assert [(e["lhs"], e["comp"], e["rhs"]) for e in result["eq"]] == [
        ("t", "=", "3.42"),
        ("p", "<", ".001"),
    ]
    # The double surname is linked to the reference of its year, with no LLM asked.
    assert _linked(result) == [("bib", 1), ("bib", 2)]
    # The title's language, and the institution line the tagger left untagged.
    assert result["metadata"]["language"] == "en"
    assert [(a["text"], a["author_ids"]) for a in result["affiliation"]] == [(_INSTITUTION, [])]
    assert result["metadata"]["title"] == _TITLE
    assert result["metadata"]["abstract"] == _ABSTRACT


async def test_a_full_run_without_an_llm_reads_the_page_without_them(tmp_path, monkeypatch):
    from bibr.export.json_export import validate_export

    result = await fast_page_run(
        tmp_path, monkeypatch, page=_RULES_PAGE, config={"mode": "full", "no_llm": True}
    )

    assert validate_export(result) == []
    assert result["extraction"]["settings"]["mode"] == "full"
    # Each reading stays where it was before the rules existed.
    assert result["eq"] == []
    assert result["xref"] == []
    assert result["metadata"]["language"] is None
    assert result["affiliation"] == []
    assert result["metadata"]["title"] == _TITLE


def _front_matter(affiliations=(), language=None, tagged=None):
    from bibr.models import PaperMetadata

    metadata = PaperMetadata(
        doi="",
        title="Memory span in young children and how it develops",
        language=language,
        affiliations=list(affiliations),
    )
    return metadata, SimpleNamespace(front_tagger_result=tagged)


@pytest.mark.parametrize(
    ("mode", "record", "expected"),
    [
        # A fast run does not export what the tagger read, but the title still may not be it.
        ("only", None, "Example Press"),
        # An input that declares its publisher keeps its own.
        ("only", "Declared Press", "Declared Press"),
        # With an LLM the record's answer stands, empty or not.
        ("primary", None, None),
        ("off", "Model Press", "Model Press"),
    ],
)
def test_the_title_decision_still_knows_the_tagged_publisher(mode, record, expected):
    from bibr.front_tagger.assemble import TaggedFrontMatter
    from bibr.pipeline.stages.post_parse import _masthead_publisher
    from bibr.schemas import TitleKeywordsLLM

    tagged = TaggedFrontMatter(
        title_keywords=TitleKeywordsLLM(title="A Title", publisher="Example Press"), byline=None
    )
    settings = SimpleNamespace(ml=SimpleNamespace(front_tagger_mode=mode))
    metadata = SimpleNamespace(publisher=record)

    publisher = _masthead_publisher(SimpleNamespace(front_tagger_result=tagged), metadata, settings)

    assert publisher == expected


def test_the_title_decision_needs_no_tagger_answer():
    from bibr.front_tagger.assemble import TaggedFrontMatter
    from bibr.pipeline.stages.post_parse import _masthead_publisher

    settings = SimpleNamespace(ml=SimpleNamespace(front_tagger_mode="only"))
    metadata = SimpleNamespace(publisher=None)
    empty = TaggedFrontMatter(title_keywords=None, byline=None)

    for tagged in (None, empty):
        contents = SimpleNamespace(front_tagger_result=tagged)
        assert _masthead_publisher(contents, metadata, settings) is None


def test_the_front_matter_rules_add_a_language_and_the_tagged_affiliations():
    from bibr.front_tagger.assemble import TaggedFrontMatter
    from bibr.pipeline.stages.post_parse import _apply_front_matter_rules

    tagged = TaggedFrontMatter(title_keywords=None, byline=None, unattached_affiliations=["Univ A"])
    metadata, contents = _front_matter(tagged=tagged)

    _apply_front_matter_rules(contents, metadata)

    assert metadata.language == "en"
    assert [a.text for a in metadata.affiliations] == ["Univ A"]
    decision = metadata._field_decisions.get("affiliations")
    assert (decision.source, decision.rule) == ("front_tagger", "extracted")


def test_the_front_matter_rules_add_no_affiliation_when_the_record_abstained():
    from bibr.front_tagger.assemble import TaggedFrontMatter
    from bibr.pipeline.stages.post_parse import _apply_front_matter_rules

    tagged = TaggedFrontMatter(title_keywords=None, byline=None, unattached_affiliations=["Univ A"])
    metadata, contents = _front_matter(tagged=tagged)

    _apply_front_matter_rules(contents, metadata, abstained=True)

    assert metadata.affiliations == []
    assert metadata._field_decisions.get("affiliations") is None


def test_the_front_matter_rules_keep_what_the_input_declared():
    from bibr.front_tagger.assemble import TaggedFrontMatter
    from bibr.models import Affiliation
    from bibr.pipeline.stages.post_parse import _apply_front_matter_rules

    declared = [Affiliation(text="Declared University")]
    tagged = TaggedFrontMatter(title_keywords=None, byline=None, unattached_affiliations=["Univ A"])
    metadata, contents = _front_matter(declared, language="nl", tagged=tagged)

    _apply_front_matter_rules(contents, metadata)

    assert metadata.language == "nl"
    assert metadata.affiliations == declared
    assert metadata._field_decisions.get("affiliations") is None


def test_the_front_matter_rules_need_no_tagger_answer():
    from bibr.pipeline.stages.post_parse import _apply_front_matter_rules

    metadata, contents = _front_matter()

    _apply_front_matter_rules(contents, metadata)

    assert metadata.language == "en"
    assert metadata.affiliations == []
    assert metadata._field_decisions.get("affiliations") is None
