"""A stage failure is recorded per file; it never aborts the chunk or the batch.

The stage contract says a stage sets ``fs.error`` and never raises out of
``run()``, but ``ParseSegmentStage``'s segmenter load, the identity stage and
the classifier stage had no guard, so one raise killed every file of the chunk,
and ``bibr.api``'s batch loop then abandoned every later chunk.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState


def _ctx(file_states, resources=None):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=resources or MagicMock(),
        config=RunConfig(),
    )


async def test_segmenter_load_failure_fails_the_files_not_the_chunk():
    from bibr.pipeline.stages.parse_segment import ParseSegmentStage

    resources = MagicMock()
    resources.ensure_segmenter.side_effect = OSError("model download failed")
    files = [FileState(path=Path("a.pdf")), FileState(path=Path("b.pdf"))]

    await ParseSegmentStage().run(_ctx(files, resources))

    for fs in files:
        assert fs.error_code == "parse_failed"
        assert fs.failed_stage == "parse"
        assert isinstance(fs.original_error, OSError)


async def test_identity_failure_is_confined_to_its_file():
    from bibr.pipeline.stages.identity import IdentityValidationStage

    def paper(label):
        return SimpleNamespace(
            contents=label,
            metadata=None,
            validation_issues=[],
            expected_identity=None,
            doi_selection=None,
        )

    good, bad = FileState(path=Path("good.pdf")), FileState(path=Path("bad.pdf"))
    good.paper, bad.paper = paper("good"), paper("bad")
    selection = SimpleNamespace(selected=None, issues=())

    def collect(contents, pdf_evidence=None):
        if contents == "bad":
            raise KeyError("text_id")
        return ()

    with (
        patch("bibr.pipeline.stages.identity.collect_doi_candidates", collect),
        patch("bibr.pipeline.stages.identity.select_doi_candidates", lambda *_: selection),
    ):
        await IdentityValidationStage().run(_ctx([good, bad]))

    assert good.error is None and good.doi_selection is selection
    assert bad.error_code == "identity_failed"
    assert bad.failed_stage == "identity"


async def test_classifier_startup_failure_leaves_the_files_running():
    from bibr.pipeline.stages.classifiers import ClassifierStage

    resources = MagicMock()

    async def start():
        raise RuntimeError("classifier resources are closed")

    resources.start_classifiers = start
    fs = FileState(path=Path("a.pdf"))

    await ClassifierStage().run(_ctx([fs], resources))

    assert fs.error is None


_MINIMAL_EXPORT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "schema_conformance"
    / "valid"
    / "minimal.json"
)


class _Pipeline:
    """Stub whose chunks crash, succeed, or silently skip the export."""

    memory_mode = "balanced"

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.chunks = []
        self.states = []

    async def process_chunk(self, file_states, progress=None):
        self.chunks.append([fs.path.name for fs in file_states])
        self.states.extend(file_states)
        action = self.behaviour[len(self.chunks) - 1]
        if action == "crash":
            for fs in file_states:
                fs.page_images = ["a rendered page"]
            raise RuntimeError("stage raised")
        for fs in file_states:
            if action == "ok":
                fs.result_json = {**json.loads(_MINIMAL_EXPORT.read_text()), "paper_id": "ok"}


async def test_crashed_chunk_fails_only_its_files():
    from bibr.api import ChewFailure, Result, _process_batch

    pipeline = _Pipeline(["crash", "ok"])
    results = await _process_batch(
        pipeline, [Path("a.pdf"), Path("b.pdf"), Path("c.pdf")], batch_size=2
    )

    assert pipeline.chunks == [["a.pdf", "b.pdf"], ["c.pdf"]]
    assert [type(r) for r in results] == [ChewFailure, ChewFailure, Result]
    assert all(fs.page_images is None for fs in pipeline.states[:2])
    assert results[0].error_code == "chunk_error"
    assert "stage raised" in results[0].error


async def test_file_without_export_is_reported_not_raised():
    from bibr.api import ChewFailure, _process_batch

    results = await _process_batch(_Pipeline(["skip"]), [Path("a.pdf")], batch_size=1)

    [failure] = results
    assert isinstance(failure, ChewFailure)
    assert failure.error_code == "export_failed"
    assert failure.failed_stage == "export"
