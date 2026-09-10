from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.check_public_tree import check_tree, find_violations, main, source_paths


@pytest.mark.parametrize(
    "name",
    [
        "benchmarks/run.py",
        "data/source.json",
        "docs/reports/analysis.md",
        "docs/superpowers/plan.md",
        "evaluation/data.json",
        "evaluation/new_module.py",
        "evaluation/nested/evaluate.py",
        "examples/data.PARQUET",
        "examples/payload.jsonl.gz",
        "examples/payload.arrow.zst",
        "tests/fixtures/new-paper.pdf",
        "docs/paper.PDF",
        "bibr/data/another.pdf",
        "Benchmarks/run.py",
        "../outside.py",
        "docs\\reports\\analysis.md",
    ],
)
def test_unapproved_material_cannot_bypass_directory_or_payload_rules(name):
    assert find_violations([name])


def test_current_public_scorer_and_explicit_fixtures_are_allowed():
    names = [
        "evaluation/__init__.py",
        "evaluation/evaluate.py",
        "evaluation/section_metrics.py",
        "evaluation/validation_metrics.py",
        "evaluation/README.md",
        "bibr/data/sample_paper.pdf",
        "tests/fixtures/cropbox_offset_sample.pdf",
        "tests/fixtures/native_text_sample.pdf",
        "tests/fixtures/scanned_sample.pdf",
        "tests/fixtures/jats/PMC4383902.xml",
        "tests/fixtures/integrity_statements_synthetic.json",
        "README.md",
        "bibr/config.py",
    ]
    assert find_violations(names) == []


def test_checkout_checks_existing_tracked_files_only(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "README.md").write_text("public")
    (tmp_path / "ignored.jsonl").write_text("local data")

    def fake_git(command, **kwargs):
        assert command == ["git", "-C", str(tmp_path), "ls-files", "--cached", "-z"]
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0, b"README.md\0benchmarks/deleted.py\0", b"")

    monkeypatch.setattr(subprocess, "run", fake_git)
    assert source_paths(tmp_path) == ["README.md"]
    assert check_tree(tmp_path) == (1, [])


def test_standalone_snapshot_checks_untracked_payloads(tmp_path):
    (tmp_path / "README.md").write_text("public")
    (tmp_path / "records.jsonl").write_text("data")
    assert main(["--root", str(tmp_path)]) == 1
    (tmp_path / "records.jsonl").unlink()
    assert main(["--root", str(tmp_path)]) == 0


def test_allowed_filename_cannot_hide_symlink_to_private_payload(tmp_path):
    target = tmp_path / "outside.txt"
    target.write_text("not an approved fixture")
    fixture = tmp_path / "bibr" / "data" / "sample_paper.pdf"
    fixture.parent.mkdir(parents=True)
    try:
        fixture.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks are not available")
    _, violations = check_tree(tmp_path)
    assert any("symlinks" in violation for violation in violations)


def test_inventory_failure_does_not_fall_back_to_a_successful_scan(tmp_path, monkeypatch):
    (tmp_path / ".git").write_text("gitdir: unavailable")

    def failed_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, ["git", "ls-files"])

    monkeypatch.setattr(subprocess, "run", failed_git)
    assert main(["--root", str(tmp_path)]) == 2


def test_quality_job_runs_public_source_boundary_check():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["quality"]["steps"]
    assert any(step.get("run") == "python scripts/check_public_tree.py" for step in steps)
