from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_manifest_resolves_paths_and_normalizes_expected_identity(tmp_path):
    from bibr.local.manifest import load_manifest

    source = tmp_path / "inputs" / "paper.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.4\nsource")
    expected_doi = "10.1234/Case.Sensitive"
    expected_hash = hashlib.sha256(expected_doi.casefold().encode()).hexdigest()
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "input_path": "inputs/paper.pdf",
                "output_path": "results/../results/paper.json",
                "queue_record_id": "record-1",
                "expected_doi": f"https://doi.org/{expected_doi}",
                "expected_doi_sha256": expected_hash,
                "expected_title": "A source-backed title",
                "target_block_hint": {"page": 1},
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "doi_required": True,
            }
        ],
    )

    [record] = load_manifest(manifest)

    assert record.input_path == source.resolve()
    assert record.output_path == (tmp_path / "results" / "paper.json").resolve()
    assert record.content_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.expected_identity.queue_record_id == "record-1"
    assert record.expected_identity.expected_doi == expected_doi.casefold()
    assert record.expected_identity.expected_doi_sha256 == expected_hash
    assert record.expected_identity.expected_title == "A source-backed title"
    assert record.expected_identity.target_block_hint == {"page": 1}
    assert record.expected_identity.doi_required is True


def test_manifest_rejects_expected_doi_hash_mismatch(tmp_path):
    from bibr.local.manifest import ManifestError, load_manifest

    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "input_path": "paper.pdf",
                "output_path": "paper.json",
                "queue_record_id": "record-1",
                "expected_doi": "10.1234/right",
                "expected_doi_sha256": "0" * 64,
            }
        ],
    )

    with pytest.raises(ManifestError, match="expected DOI hash"):
        load_manifest(manifest)


@pytest.mark.parametrize("duplicate", ["record", "output"])
def test_manifest_rejects_duplicate_record_or_normalized_output(tmp_path, duplicate):
    from bibr.local.manifest import ManifestError, load_manifest

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 b")
    rows = [
        {
            "input_path": "a.pdf",
            "output_path": "results/a.json",
            "queue_record_id": "record-a",
        },
        {
            "input_path": "b.pdf",
            "output_path": "results/b.json",
            "queue_record_id": "record-b",
        },
    ]
    if duplicate == "record":
        rows[1]["queue_record_id"] = "record-a"
    else:
        rows[1]["output_path"] = "results/sub/../a.json"
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(manifest, rows)

    with pytest.raises(ManifestError, match=f"duplicate {duplicate}"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    ("outputs", "expected_message"),
    [
        (("results/Paper.json", "results/paper.json"), "duplicate output"),
        (
            ("results/r\N{LATIN SMALL LETTER E WITH ACUTE}sume.json", "results/re\u0301sume.json"),
            "duplicate output",
        ),
    ],
    ids=["case-only", "unicode-normalization"],
)
def test_manifest_rejects_conservative_output_aliases(tmp_path, outputs, expected_message):
    from bibr.local.manifest import ManifestError, load_manifest

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 b")
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "input_path": "a.pdf",
                "output_path": outputs[0],
                "queue_record_id": "record-a",
            },
            {
                "input_path": "b.pdf",
                "output_path": outputs[1],
                "queue_record_id": "record-b",
            },
        ],
    )

    with pytest.raises(ManifestError, match=expected_message):
        load_manifest(manifest)


@pytest.mark.parametrize(
    "unsafe_output",
    ["inputs/../a.pdf", "b.pdf", "queue.jsonl"],
    ids=["same-input-with-dot-segments", "other-manifest-input", "manifest-itself"],
)
def test_manifest_rejects_outputs_that_alias_inputs_or_manifest(tmp_path, unsafe_output):
    from bibr.local.manifest import ManifestError, load_manifest

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 b")
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "input_path": "a.pdf",
                "output_path": unsafe_output,
                "queue_record_id": "record-a",
            },
            {
                "input_path": "b.pdf",
                "output_path": "safe/b.json",
                "queue_record_id": "record-b",
            },
        ],
    )

    with pytest.raises(ManifestError, match="output path"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("queue_record_id", 7),
        ("expected_doi", 1234),
        ("expected_doi_sha256", int("1" * 64)),
        ("expected_title", 1234),
        ("source_sha256", int("1" * 64)),
        ("doi_required", "false"),
    ],
)
def test_manifest_rejects_non_string_scalars_and_non_boolean_doi_required(tmp_path, field, value):
    from bibr.local.manifest import ManifestError, load_manifest

    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    row = {
        "input_path": "paper.pdf",
        "output_path": "paper.json",
        "queue_record_id": "record-1",
        field: value,
    }
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(manifest, [row])

    with pytest.raises(ManifestError, match=field):
        load_manifest(manifest)


def test_chew_requires_exactly_one_source_mode(tmp_path):
    from bibr.local.cli import _build_parser
    from bibr.local.manifest import ManifestError, resolve_cli_source_mode

    parser = _build_parser()
    neither = parser.parse_args(["chew", "--no-llm"])
    both = parser.parse_args(["chew", "paper.pdf", "--manifest", str(tmp_path / "q.jsonl")])
    manifest_only = parser.parse_args(["chew", "--manifest", str(tmp_path / "q.jsonl")])

    with pytest.raises(ManifestError, match="exactly one"):
        resolve_cli_source_mode(neither.input, neither.manifest)
    with pytest.raises(ManifestError, match="exactly one"):
        resolve_cli_source_mode(both.input, both.manifest)
    assert resolve_cli_source_mode(manifest_only.input, manifest_only.manifest) == "manifest"


async def test_manifest_rejects_global_output_flag_as_ambiguous(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "input_path": "paper.pdf",
                "output_path": "result.json",
                "queue_record_id": "record-1",
            }
        ],
    )
    args = _build_parser().parse_args(
        [
            "chew",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "ignored.json"),
            "--dry-run",
            "--no-llm",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--manifest" in error
    assert "-o/--output" in error
    assert "output_path" in error


async def test_manifest_execution_skips_global_output_preparation(tmp_path, monkeypatch):
    from bibr.local.cli import _build_parser, _run_process

    source = tmp_path / "paper.docx"
    source.write_bytes(b"docx fixture")
    destination = tmp_path / "results" / "paper.json"
    manifest = tmp_path / "queue.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "input_path": "paper.docx",
                "output_path": "results/paper.json",
                "queue_record_id": "record-1",
            }
        ],
    )

    def forbidden_prepare(*_args, **_kwargs):
        raise AssertionError("manifest mode must not prepare a global output path")

    class FakePipeline:
        def __init__(self, **_kwargs):
            pass

        async def process_chunk(self, states, **_kwargs):
            for state in states:
                state.result_json = {"queue_record_id": "record-1"}

        def llm_usage_snapshot(self):
            return {}

        async def aclose(self):
            pass

    monkeypatch.setattr("bibr.local.cli.process._prepare_output_path", forbidden_prepare)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", FakePipeline)
    args = _build_parser().parse_args(["chew", "--manifest", str(manifest), "--no-llm"])

    await _run_process(args)

    assert json.loads(destination.read_text()) == {"queue_record_id": "record-1"}


async def test_chunk_processor_preserves_repeated_manifest_rows_across_chunks(tmp_path):
    from bibr.local.cli import ChunkProcessor
    from bibr.local.manifest import ManifestRecord
    from bibr.pipeline.identity import ExpectedIdentity

    input_path = tmp_path / "paper.pdf"
    records = [
        ManifestRecord(
            input_path=input_path,
            output_path=tmp_path / f"paper-{index}.json",
            expected_identity=ExpectedIdentity(
                queue_record_id=f"queue-{index}",
                expected_doi=f"10.1234/paper-{index}",
                target_block_hint={"occurrence": index},
            ),
            content_sha256=str(index) * 64,
        )
        for index in (1, 2)
    ]
    seen = []
    pipeline = MagicMock()

    async def fake_chunk(states, **_kwargs):
        seen.extend(states)

    pipeline.process_chunk = fake_chunk
    processor = ChunkProcessor(
        pipeline=pipeline,
        paper_id=None,
        is_batch=True,
        active_stages=[],
    )

    await processor.run([records[0]], chunk_index=1, total_chunks=2, console=MagicMock())
    await processor.run([records[1]], chunk_index=2, total_chunks=2, console=MagicMock())

    assert [state.path for state in seen] == [input_path, input_path]
    assert [state.expected_identity.queue_record_id for state in seen] == ["queue-1", "queue-2"]
    assert [state.expected_identity.target_block_hint for state in seen] == [
        {"occurrence": 1},
        {"occurrence": 2},
    ]
    assert [state.manifest_output_path for state in seen] == [
        tmp_path / "paper-1.json",
        tmp_path / "paper-2.json",
    ]
    assert [state.content_sha256 for state in seen] == ["1" * 64, "2" * 64]
    assert all(state.paper_id is None for state in seen)


def test_chunk_result_writer_uses_each_manifest_output_path(tmp_path):
    from bibr.local.cli.process import _write_chunk_results
    from bibr.pipeline.state import FileState

    input_path = tmp_path / "paper.pdf"
    states = [
        FileState(
            path=input_path,
            manifest_output_path=tmp_path / "nested" / f"paper-{index}.json",
            result_json={"queue": index},
        )
        for index in (1, 2)
    ]

    processed, errors = _write_chunk_results(
        states,
        output_path=None,
        json_kwargs={"indent": 2},
        console=MagicMock(),
        is_batch=True,
        total_files=2,
        total_t0=0.0,
    )

    assert (processed, errors) == (2, 0)
    assert json.loads((tmp_path / "nested" / "paper-1.json").read_text()) == {"queue": 1}
    assert json.loads((tmp_path / "nested" / "paper-2.json").read_text()) == {"queue": 2}
