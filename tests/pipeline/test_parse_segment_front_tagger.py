"""ParseSegmentStage hands the front-matter tagger the run's settings and its scans."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from bibr.config import GlobalSettings
from bibr.front_tagger.assemble import TaggedFrontMatter
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.parse_segment import ParseSegmentStage
from bibr.pipeline.state import FileState
from bibr.run_mode import fast_settings
from tests.front_tagger.helpers import label

ABSTRACT = (
    "We studied handedness in 400 families from three countries over two decades. Left "
    "handedness was more common in sons than in daughters, ran in families, and was rarer "
    "among older participants. Twin pairs agreed more often than siblings. It is heritable."
)


class _Tagger:
    """Tags whichever of its phrases a window prints, and counts the windows."""

    def __init__(self):
        self.windows = 0

    def tag(self, inp):
        self.windows += 1
        spans = (
            ("TITLE", "Handedness in families"),
            ("GIVEN", "Ana"),
            ("FAMILY", "Ruiz"),
            ("ABSTRACT", ABSTRACT),
        )
        present = [(field, phrase) for field, phrase in spans if phrase in inp.text]
        tags = label(inp, *present) if present else ["O"] * len(inp.words)
        return tags, [1.0] * len(tags)


def _page(*rows: tuple[str, str]) -> list[dict]:
    step = 900 // (len(rows) + 1)
    return [
        {
            "index": i,
            "label": kind,
            "content": text,
            "bbox_2d": [100, 50 + i * step, 900, 50 + i * step + int(step * 0.8)],
        }
        for i, (kind, text) in enumerate(rows)
    ]


def _manuscript(abstract_page: int = 0) -> list[list[dict]]:
    pages = [
        _page(("doc_title", "Handedness in families"), ("text", "Ana Ruiz")),
        _page(("text", "Author note and contact details")),
        _page(("text", "Header only")),
    ]
    pages[abstract_page] += _page(("text", "Abstract"), ("abstract", ABSTRACT))
    return pages


def _run(settings: GlobalSettings, rm, *, pages, page_kinds=None):
    fs = FileState(path=Path("x.pdf"))
    fs.ocr_regions = pages
    if page_kinds is not None:
        fs.page_kinds = page_kinds
    contents = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = contents
    parser._deferred_texts = []
    parser.assembler = MagicMock(__len__=lambda self: 0)
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(),
        settings=settings,
    )
    with patch("bibr.structure.pdf_parser.PDFParser", return_value=parser):
        asyncio.run(ParseSegmentStage().run(ctx))
    return fs, contents, ctx


def _rm(tagger):
    rm = MagicMock()
    rm.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))
    rm.ensure_front_role = MagicMock(return_value=None)
    rm.ensure_front_tagger = MagicMock(return_value=tagger)
    return rm


def _fast() -> GlobalSettings:
    return fast_settings(GlobalSettings())


def _primary() -> GlobalSettings:
    return GlobalSettings(ml={"front_tagger_mode": "primary"})


def test_the_tagger_is_given_the_runs_settings():
    rm = _rm(_Tagger())
    settings = _fast()
    fs, contents, _ = _run(settings, rm, pages=_manuscript())
    rm.ensure_front_tagger.assert_called_once_with(settings)
    assert settings.ml.front_tagger_mode == "only"
    assert contents.front_tagger_result.title_keywords.title == "Handedness in families"
    assert fs.error is None


def test_a_full_run_hands_over_its_own_settings_too():
    rm = _rm(_Tagger())
    settings = _primary()
    _, contents, _ = _run(settings, rm, pages=_manuscript())
    rm.ensure_front_tagger.assert_called_once_with(settings)
    assert contents.front_tagger_result.title_keywords.title == "Handedness in families"


def test_a_scanned_front_page_stays_with_the_llm_when_there_is_one():
    rm = _rm(_Tagger())
    _, contents, _ = _run(_primary(), rm, pages=_manuscript(), page_kinds={0: "scan"})
    rm.ensure_front_tagger.assert_not_called()
    assert not isinstance(contents.front_tagger_result, TaggedFrontMatter)


def test_a_broken_text_layer_stays_with_the_llm_when_there_is_one():
    rm = _rm(_Tagger())
    _run(_primary(), rm, pages=_manuscript(), page_kinds={0: "broken_text_layer"})
    rm.ensure_front_tagger.assert_not_called()


def test_with_the_tagger_alone_a_scanned_front_page_is_read_and_noted():
    rm = _rm(_Tagger())
    _, contents, _ = _run(_fast(), rm, pages=_manuscript(), page_kinds={0: "scan"})
    result = contents.front_tagger_result
    assert result.title_keywords.title == "Handedness in families"
    assert "scanned_front_page" in result.problems
    # The note rides on the problems, so the extractor reports it with the rest.
    assert "scanned_front_page" in result.rejection_reasons(0.9)


def test_with_the_tagger_alone_a_broken_text_layer_is_noted_the_same_way():
    rm = _rm(_Tagger())
    _, contents, _ = _run(_fast(), rm, pages=_manuscript(), page_kinds={0: "broken_text_layer"})
    assert "scanned_front_page" in contents.front_tagger_result.problems


def test_a_born_digital_front_page_carries_no_scan_note():
    for kinds in (None, {0: "born_digital"}):
        rm = _rm(_Tagger())
        _, contents, _ = _run(_fast(), rm, pages=_manuscript(), page_kinds=kinds)
        assert "scanned_front_page" not in contents.front_tagger_result.problems


def test_an_unloadable_tagger_leaves_no_answer_and_never_fails_the_paper():
    rm = _rm(None)
    fs, contents, _ = _run(_fast(), rm, pages=_manuscript(), page_kinds={0: "scan"})
    assert contents.front_tagger_result is None
    assert fs.error is None


def test_a_tagger_that_raises_never_fails_the_paper():
    rm = _rm(None)
    rm.ensure_front_tagger.side_effect = RuntimeError("boom")
    fs, contents, _ = _run(_fast(), rm, pages=_manuscript())
    assert contents.front_tagger_result is None
    assert fs.error is None


def test_rule_w_reads_an_abstract_from_a_later_window_with_the_tagger_alone():
    tagger = _Tagger()
    _, contents, _ = _run(_fast(), _rm(tagger), pages=_manuscript(abstract_page=2))
    result = contents.front_tagger_result
    assert result.title_keywords.abstract == ABSTRACT
    assert "no_abstract" not in result.problems
    assert tagger.windows == 2


def test_a_scanned_front_page_keeps_rule_w_and_its_note():
    tagger = _Tagger()
    _, contents, _ = _run(
        _fast(), _rm(tagger), pages=_manuscript(abstract_page=2), page_kinds={0: "scan"}
    )
    result = contents.front_tagger_result
    assert result.title_keywords.abstract == ABSTRACT
    assert result.problems.count("scanned_front_page") == 1
