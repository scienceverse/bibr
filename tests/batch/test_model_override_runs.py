"""``bibr batch`` with a model override that does not load: the run stops, no paper fails.

Runs the real runner, ``LocalExecutor``, ``WarmChewMany``, ``Chewer`` and
``_process_batch`` over the real stage list, on small generated PDFs with a
text layer. Only the models that are not under test are fakes, injected
behind the ``ResourceManager`` seams as in ``tests/test_pipeline_smoke.py``:
the layout detector, the sentence splitter and the LLM transport. The model
under test is a real local bundle, broken first and then repaired.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bibr.batch.ledger import LEDGER_FILENAME, MODEL_OVERRIDE_FAILED, Ledger
from bibr.batch.runner import RUN_INFO_FILENAME, BatchOptions, LocalOptions, run_batch
from bibr.config import Settings
from bibr.utils.model_overrides import NOT_USED, reset_loaded_model_overrides
from tests.test_pipeline_smoke import (
    _LINES,
    FakeLayout,
    FakeOcr,
    FakeSegmenter,
    _build_pdf,
    _fake_llm_client,
)

pytest.importorskip("onnxruntime")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "onnx"
PAPERS = ("alpha", "beta", "gamma")
# What a paper never gets for a model override that does not load.
PAPER_FAILURES = {"extraction_failed", "parse_failed", "chunk_error"}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    from bibr.extract import ref_extractor
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.resources import ResourceManager

    reset_loaded_model_overrides()
    monkeypatch.setattr(ref_extractor, "_TEXT_SEGMENTER", None)
    monkeypatch.setattr(ref_extractor, "_TEXT_SEGMENTER_KEY", None)
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "deadbeef")
    monkeypatch.setattr(Settings.ocr, "native_text_min_chars", 4)
    monkeypatch.setattr(Settings.ml, "runtime", "onnx")

    class Pipeline(LocalPipeline):
        """The real pipeline with the models not under test replaced by fakes."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            resources = ResourceManager(
                memory_mode=self.memory_mode,
                ocr_backend=self.ocr_backend,
                ocr_url=self.ocr_url,
                ocr_model=self.ocr_model,
                device=self.device,
                settings=self.settings,
                layout=FakeLayout(),
                segmenter=FakeSegmenter(),
            )
            self._resources = resources
            resources._ocr = FakeOcr()
            resources._llm_client = _fake_llm_client()

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", Pipeline)
    yield
    reset_loaded_model_overrides()


@pytest.fixture
def corpus(tmp_path) -> Path:
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in PAPERS:
        lines = [
            (x, y, size, f"{text} ({name})" if y == 720 else text) for x, y, size, text in _LINES
        ]
        (papers / f"{name}.pdf").write_bytes(_build_pdf(lines))
    return papers


def _run(corpus: Path, out: Path, **chew) -> int:
    options = BatchOptions(
        inputs=[str(corpus)],
        out=out,
        local=LocalOptions(
            chew_options={"memory_mode": "balanced", "crossref": False, "mode": "full", **chew},
            batch_size=2,
        ),
        cli_options={"batch_size": 2},
        tables=False,
    )
    return run_batch(options)


def _rows(out: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in Ledger(out / LEDGER_FILENAME).read():
        latest[row["paper_id"]] = row
    return latest


def _run_info(out: Path) -> dict:
    return json.loads((out / RUN_INFO_FILENAME).read_text(encoding="utf-8"))


def _assert_stopped_on(out: Path, code: int, setting: str) -> None:
    assert code != 0
    info = _run_info(out)
    assert info["reason"] == MODEL_OVERRIDE_FAILED
    assert setting in json.dumps(info)
    rows = _rows(out)
    assert rows, "the papers it was running are in the ledger"
    assert {row.get("error_code") for row in rows.values()} == {MODEL_OVERRIDE_FAILED}
    assert not {row.get("error_code") for row in rows.values()} & PAPER_FAILURES
    assert not any(row["status"] == "ok" for row in rows.values())
    assert sorted(p.name for p in out.glob("*.json") if p.name != RUN_INFO_FILENAME) == []


def _assert_all_done(out: Path, code: int, setting: str) -> list[dict]:
    assert code == 0
    rows = _rows(out)
    assert sorted(rows) == sorted(PAPERS)
    assert {row["status"] for row in rows.values()} == {"ok"}
    info = _run_info(out)
    assert info["reason"] == "completed"
    records = {record["setting"]: record for record in info["model_overrides"]}
    assert records[setting]["loaded"] is True
    return [json.loads((out / f"{name}.json").read_text(encoding="utf-8")) for name in PAPERS]


def test_a_broken_eager_override_stops_the_run_and_a_repaired_one_runs_it(
    corpus, tmp_path, monkeypatch
):
    """The section classifier loads when its stage starts, before any paper is classified."""
    bundle = tmp_path / "models" / "section-candidate"
    shutil.copytree(FIXTURES / "section" / "onnx", bundle / "onnx")
    (bundle / "onnx" / "model.onnx").write_bytes(b"not an onnx graph")
    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", str(bundle))
    out = tmp_path / "out"

    _assert_stopped_on(out, _run(corpus, out), "ML_SECTION_CLASSIFIER_MODEL_ID")

    shutil.copy(FIXTURES / "section" / "onnx" / "model.onnx", bundle / "onnx" / "model.onnx")
    reset_loaded_model_overrides()
    exports = _assert_all_done(out, _run(corpus, out), "ML_SECTION_CLASSIFIER_MODEL_ID")
    for export in exports:
        models = {row["setting"]: row for row in export["extraction"]["settings"]["models"]}
        assert (
            models["ML_SECTION_CLASSIFIER_MODEL_ID"]["id"].partition("@")[0] == "section-candidate"
        )


def test_a_broken_lazy_override_stops_the_run_and_a_repaired_one_runs_it(
    corpus, tmp_path, monkeypatch
):
    """The text reference segmenter loads when the first paper reaches its tier."""
    from tests.extract.test_text_segmenter import _bundle

    bundle = _bundle(tmp_path / "models" / "text-seg-candidate")
    good = (bundle / "model.onnx").read_bytes()
    (bundle / "model.onnx").write_bytes(b"not an onnx graph")
    monkeypatch.setattr(Settings, "REF_TEXT_SEG_MODEL_ID", str(bundle))
    out = tmp_path / "out"

    _assert_stopped_on(out, _run(corpus, out, ref_seg_strategy="crf"), "REF_TEXT_SEG_MODEL_ID")

    (bundle / "model.onnx").write_bytes(good)
    reset_loaded_model_overrides()
    exports = _assert_all_done(
        out, _run(corpus, out, ref_seg_strategy="crf"), "REF_TEXT_SEG_MODEL_ID"
    )
    for export in exports:
        assert "text_tagger" in json.dumps(export["extraction"]["diagnostics"])


def test_a_classifier_that_opens_but_fails_its_probe_stops_before_any_paper(
    corpus, tmp_path, monkeypatch
):
    """The graph is a valid ONNX file; its manifest lists one class more than it returns."""
    bundle = tmp_path / "models" / "section-candidate"
    shutil.copytree(FIXTURES / "section" / "onnx", bundle / "onnx")
    manifest_path = bundle / "onnx" / "bibr_onnx.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_classes"] = [*manifest["label_classes"], "references"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", str(bundle))
    out = tmp_path / "out"

    _assert_stopped_on(out, _run(corpus, out), "ML_SECTION_CLASSIFIER_MODEL_ID")
    assert "type_logits of shape" in json.dumps(_run_info(out))


def test_papers_that_fell_back_before_a_stop_are_run_again_on_resume(
    corpus, tmp_path, monkeypatch, capsys
):
    """Three papers in a row on which the candidate raises: two written, then the stop.

    The paper classifier runs in the classifier resources' thread pool, so its
    error comes back to each paper through the inference batcher. The first
    chunk's two papers fall back and are written with the warning; the third
    paper stops the run. Once the model is repaired, a plain resume runs all
    three, the two written ones included.
    """
    from bibr.structure.paper_classifier_onnx import OnnxPaperClassifierModel

    bundle = tmp_path / "models" / "paper-candidate"
    shutil.copytree(FIXTURES / "paper" / "onnx", bundle / "onnx")
    monkeypatch.setattr(Settings.ml, "paper_classifier_model_id", str(bundle))
    working = OnnxPaperClassifierModel.classify_batch

    def broken(self, items):
        raise RuntimeError("the graph fails on this input")

    monkeypatch.setattr(OnnxPaperClassifierModel, "classify_batch", broken)
    out = tmp_path / "out"

    code = _run(corpus, out)
    assert code != 0
    assert _run_info(out)["reason"] == MODEL_OVERRIDE_FAILED
    assert "raised at inference on 3 papers in a row" in _run_info(out)["fatal"]
    rows = _rows(out)
    assert {pid: row["status"] for pid, row in rows.items()} == {
        "alpha": "ok",
        "beta": "ok",
        "gamma": "failed",
    }
    assert rows["gamma"]["error_code"] == MODEL_OVERRIDE_FAILED
    for name in ("alpha", "beta"):
        assert rows[name]["warnings"]["codes"]["MODEL_OVERRIDE_INFERENCE_FAILED"] == 1
    assert sorted(_run_info(out)["override_fallbacks"]) == ["alpha", "beta"]
    printed = capsys.readouterr().out
    assert "the run stopped on a model override: 1 paper not finished" in printed
    assert "2 papers fell back from an overridden model" in printed
    assert "0 failed" in printed and "1 not finished" in printed

    monkeypatch.setattr(OnnxPaperClassifierModel, "classify_batch", working)
    reset_loaded_model_overrides()
    exports = _assert_all_done(out, _run(corpus, out), "ML_PAPER_CLASSIFIER_MODEL_ID")
    resumed = [
        row
        for row in Ledger(out / LEDGER_FILENAME).read()
        if row["run_id"] != rows["alpha"]["run_id"]
    ]
    assert sorted(row["paper_id"] for row in resumed) == sorted(PAPERS)
    for export in exports:
        codes = {w["code"] for w in export["extraction"]["warnings"]}
        assert "MODEL_OVERRIDE_INFERENCE_FAILED" not in codes


def test_a_no_llm_run_neither_loads_nor_records_a_classifier_it_never_asks(
    corpus, tmp_path, monkeypatch
):
    """--no-llm types sections by the heading lookup alone: the classifiers are not asked.

    A broken candidate does not stop such a run, and a working one is not
    recorded as loaded, since it gave no answer.
    """
    bundle = tmp_path / "models" / "section-candidate"
    shutil.copytree(FIXTURES / "section" / "onnx", bundle / "onnx")
    good = (bundle / "onnx" / "model.onnx").read_bytes()
    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", str(bundle))
    for name, graph in (("working", good), ("broken", b"not an onnx graph")):
        (bundle / "onnx" / "model.onnx").write_bytes(graph)
        reset_loaded_model_overrides()
        out = tmp_path / f"out-{name}"
        code = _run(corpus, out, no_llm=True)
        assert code == 0, name
        info = _run_info(out)
        assert info["reason"] == "completed"
        (record,) = info["model_overrides"]
        assert record["setting"] == "ML_SECTION_CLASSIFIER_MODEL_ID"
        assert record["loaded"] is False and record["reason"] == NOT_USED
        assert {row["status"] for row in _rows(out).values()} == {"ok"}
