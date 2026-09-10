"""Regression tests for three CLI guards against silent data loss in ``bibr chew``.

1. Single-file ``-o deep/new/dir/out.json`` must create the parent directory
   instead of crashing with ``FileNotFoundError`` at write time.
2. Batch input whose files collide on ``.stem`` (e.g. ``a/x.pdf`` + ``b/x.pdf``)
   would silently overwrite one ``<stem>.json`` with the other — this must
   now be a hard error (exit 2) listing every collision group, raised before
   the pipeline is constructed.
3. ``--paper-id`` is only meaningful for single-file input; passing it with
   batch input (directory or multi-file glob) must now be a hard error
   (exit 2) instead of being silently discarded, also raised before the
   pipeline is constructed.

The stem-collision and --paper-id guards fire before ``LocalPipeline`` is
ever constructed, so those tests deliberately do NOT mock the pipeline layer
— if the guard didn't fire first, the test would blow up trying to build a
real pipeline (OCR/LLM/layout models), proving the ordering.
"""

from __future__ import annotations

import json

import pytest


class _FakePipeline:
    """Stand-in for ``LocalPipeline`` — construction is a no-op, and
    ``process_chunk`` marks every file as succeeded."""

    def __init__(self, **_kwargs):
        pass

    async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
        for fs in file_states:
            fs.result_json = {"info": {"title": fs.path.name}}

    async def aclose(self):
        pass

    def llm_usage_snapshot(self):
        return {}


def _disable_ocr_runtime_preflight(monkeypatch):
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)


# --- Guard 1: parent dirs for single-file -o ---------------------------------


async def test_single_file_output_creates_parent_dirs(tmp_path, monkeypatch):
    """`-o deep/new/dir/out.json` must create the missing directories, not
    crash with FileNotFoundError at write time."""
    from bibr.local.cli import _build_parser, _run_process

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    good_file = tmp_path / "paper.pdf"
    good_file.write_bytes(b"%PDF-1.4\n")
    out_file = tmp_path / "deep" / "new" / "dir" / "out.json"
    assert not out_file.parent.exists()

    args = _build_parser().parse_args(["chew", str(good_file), "-o", str(out_file), "--no-llm"])
    await _run_process(args)

    assert out_file.exists()


async def test_explicit_sink_honors_compact_and_pretty_json_formatting(tmp_path, monkeypatch):
    from bibr.local.cli import _build_parser, _run_process

    class _DurableFakePipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            from bibr.pipeline.artifacts import RunState

            for fs in file_states:
                payload = {"info": {"title": fs.path.name}}
                fs.result_json = payload
                fs.core_sha256 = fs.artifact_sink.write_core(fs, payload)
                fs.artifact_sink.materialize(fs, payload)
                fs.artifact_sink.record(
                    fs,
                    RunState.CORE_WRITTEN,
                    detail="enrichment_not_requested",
                )

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _DurableFakePipeline)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    pretty = tmp_path / "pretty.json"
    compact = tmp_path / "compact.json"

    await _run_process(
        _build_parser().parse_args(["chew", str(source), "-o", str(pretty), "--no-llm"])
    )
    await _run_process(
        _build_parser().parse_args(
            ["chew", str(source), "-o", str(compact), "--no-llm", "--compact"]
        )
    )

    pretty_text = pretty.read_text()
    compact_text = compact.read_text()
    assert json.loads(pretty_text) == json.loads(compact_text)
    assert '\n  "info": {' in pretty_text
    assert compact_text == '{"info":{"title":"paper.pdf"}}'


# --- Guard 2: batch stem collisions ------------------------------------------


async def test_batch_stem_collision_exits_2_and_lists_paths(tmp_path, monkeypatch, capsys):
    """Two files from different directories sharing a stem must fail loudly
    instead of silently overwriting one <stem>.json with the other."""
    from bibr.local.cli import _build_parser, _run_process

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "x.pdf"
    file_b = dir_b / "x.pdf"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert str(file_a) in err
    assert str(file_b) in err


async def test_batch_stem_collision_via_glob_exits_2(tmp_path, monkeypatch, capsys):
    """The collision guard must also fire when the collision comes from a
    glob expansion collecting files from sibling directories."""
    from bibr.local.cli import _build_parser, _run_process

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "x.pdf").write_bytes(b"%PDF-1.4\n")
    (dir_b / "x.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(tmp_path / "*" / "x.pdf"), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2


async def test_batch_stem_collision_across_extensions_exits_2(tmp_path, monkeypatch, capsys):
    """x.pdf + x.docx both write x.json — same overwrite, different extension."""
    from bibr.local.cli import _build_parser, _run_process

    file_a = tmp_path / "x.pdf"
    file_b = tmp_path / "x.docx"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"PK\x03\x04")

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert str(file_a) in err
    assert str(file_b) in err


async def test_batch_stem_collision_case_insensitive_exits_2(tmp_path, monkeypatch, capsys):
    """a/x.pdf + b/X.PDF collide on the default macOS/Windows filesystems,
    where x.json and X.json are the same directory entry."""
    from bibr.local.cli import _build_parser, _run_process

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "x.pdf"
    file_b = dir_b / "X.PDF"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert str(file_a) in err
    assert str(file_b) in err


async def test_batch_no_stem_collision_still_works(tmp_path, monkeypatch):
    """Sanity check: distinct stems in batch mode must not trip the guard."""
    from bibr.local.cli import _build_parser, _run_process

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"%PDF-1.4\n")
    out_dir = tmp_path / "out"

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(out_dir), "--no-llm"]
    )
    await _run_process(args)

    assert (out_dir / "a.json").exists()
    assert (out_dir / "b.json").exists()


# --- Guard 3: --paper-id rejected for batch input ----------------------------


async def test_paper_id_with_directory_input_exits_2(tmp_path, monkeypatch, capsys):
    """--paper-id with a directory of 2+ files must hard-error instead of
    being silently discarded."""
    from bibr.local.cli import _build_parser, _run_process

    (tmp_path / "one.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "two.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(tmp_path), "--paper-id", "my-id", "--no-llm"])

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "--paper-id" in err
    assert "single-file" in err


async def test_paper_id_with_multi_file_glob_exits_2(tmp_path, monkeypatch):
    """--paper-id with a multi-file glob must also hard-error (not just
    directory input)."""
    from bibr.local.cli import _build_parser, _run_process

    (tmp_path / "one.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "two.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(tmp_path / "*.pdf"), "--paper-id", "my-id", "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2


async def test_paper_id_with_single_file_still_works(tmp_path, monkeypatch):
    """--paper-id with a genuine single-file input must be unaffected."""
    from bibr.local.cli import _build_parser, _run_process

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    good_file = tmp_path / "paper.pdf"
    good_file.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(good_file), "--paper-id", "my-id", "--no-llm"])
    await _run_process(args)


async def test_paper_id_with_single_file_directory_still_works(tmp_path, monkeypatch):
    """Documents the resolved is_batch edge case: a directory that collects
    to exactly ONE file is NOT batch mode (``is_batch = len(files) > 1`` is
    purely file-count based, not directory-vs-file), so --paper-id must still
    apply rather than being rejected."""
    from bibr.local.cli import _build_parser, _run_process

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    (tmp_path / "only.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(tmp_path), "--paper-id", "my-id", "--no-llm"])
    await _run_process(args)
