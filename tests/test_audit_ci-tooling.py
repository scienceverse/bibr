"""Regression tests for the distribution and source-tree guards and the container files."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_dist_contents import find_violations as dist_violations  # noqa: E402

from scripts.check_public_tree import check_tree  # noqa: E402
from scripts.check_public_tree import find_violations as tree_violations  # noqa: E402

SDIST = [
    "bibr-0.7.0/PKG-INFO",
    "bibr-0.7.0/pyproject.toml",
    "bibr-0.7.0/bibr/__init__.py",
    "bibr-0.7.0/bibr/data/sample_paper.pdf",
]
WHEEL = [
    "bibr/__init__.py",
    "bibr/py.typed",
    "bibr/data/sample_paper.pdf",
    "bibr-0.7.0.dist-info/METADATA",
    "bibr-0.7.0.dist-info/RECORD",
]


# --- check_dist_contents.py -----------------------------------------------------


def test_clean_distributions_still_pass() -> None:
    assert dist_violations(SDIST, WHEEL) == []


@pytest.mark.parametrize(
    "member",
    [
        # .pth files in purelib run at every interpreter start; scripts land on PATH.
        "bibr-0.7.0.data/purelib/bibr-hook.pth",
        "bibr-0.7.0.data/scripts/bibr-helper",
        "evil-1.0.dist-info/METADATA",
        "bibr-0.6.0.dist-info/METADATA",
        "bibr_extra-1.0.dist-info/METADATA",
        "bibr-hook.pth",
    ],
)
def test_wheel_admits_only_the_package_and_its_own_metadata_directory(member: str) -> None:
    violations = dist_violations(SDIST, [*WHEEL, member])

    assert violations, member


def test_wheel_without_its_metadata_directory_is_rejected() -> None:
    wheel = [name for name in WHEEL if ".dist-info/" not in name]

    assert dist_violations(SDIST, wheel) == [
        "wheel: expected one bibr-<version>.dist-info/, found []"
    ]


@pytest.mark.parametrize(
    "member",
    [
        "bibr/extract/leaked.PDF",
        "bibr/extract/leaked.Pdf",
        "bibr/data/corpus.parquet",
        "bibr/data/rows.JSONL",
        "bibr/data/table.arrow",
        "bibr/data/rows.jsonl.gz",
    ],
)
def test_payload_checks_ignore_case_and_cover_datasets(member: str) -> None:
    assert dist_violations([*SDIST, f"bibr-0.7.0/{member}"], WHEEL)
    assert dist_violations(SDIST, [*WHEEL, member])


def test_sdist_members_must_share_the_release_root() -> None:
    violations = dist_violations([*SDIST, "other-1.0/bibr/__init__.py"], WHEEL)

    assert violations == ["sdist: unexpected member other-1.0/bibr/__init__.py"]


# --- check_public_tree.py -------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "tests/fixtures/new_paper.docx",
        "examples/paper.EPUB",
        "tests/fixtures/jats/PMC0000001.xml",
        "tests/fixtures/article.nxml",
        "docs/assets/paper.html",
        "notebooks/paper.htm",
        "tests/fixtures/papers.zip",
        "tests/fixtures/papers.tar.gz",
        "examples/papers.tgz",
        "bibr/data/sample_paper.pdf.zip",
    ],
)
def test_paper_copies_and_archives_need_explicit_approval(name: str) -> None:
    assert tree_violations([name])


def test_tracked_paper_fixtures_stay_approved() -> None:
    count, violations = check_tree(ROOT)

    assert count > 0
    assert violations == []


# --- Dockerfile.serve -----------------------------------------------------------


def dockerfile_commands() -> str:
    text = (ROOT / "Dockerfile.serve").read_text(encoding="utf-8")
    text = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text.replace("\\\n", " "))


def test_serve_image_copies_the_ci_uv_release_by_digest() -> None:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    sources = re.findall(r"COPY --from=(\S+)", dockerfile_commands())
    [uv_image] = [source for source in sources if "astral-sh/uv" in source]

    assert re.fullmatch(
        rf"ghcr\.io/astral-sh/uv:{re.escape(ci['env']['UV_VERSION'])}@sha256:[0-9a-f]{{64}}",
        uv_image,
    )


def test_serve_image_never_re_resolves_the_lock() -> None:
    syncs = re.findall(r"uv sync [^;&]*", dockerfile_commands())

    assert len(syncs) == 4
    assert all("--locked" in command for command in syncs), syncs


def test_gpu_runtime_reinstall_is_constrained_to_the_locked_versions() -> None:
    """The install bypasses `uv sync`; it shipped onnxruntime-gpu 1.28.0 against a 1.26.0 lock."""
    commands = dockerfile_commands()
    [export] = re.findall(r"uv export [^;&]*", commands)
    [install] = re.findall(r"uv pip install [^;&]*onnxruntime-gpu[^;&]*", commands)
    gpu_sync = next(
        command
        for command in re.findall(r"uv sync [^;&]*", commands)
        if "--extra gpu" in command and "--no-install-project" not in command
    )
    constraints = re.search(r"-o (\S+)", export)

    assert "--locked" in export
    assert constraints is not None
    assert f"--constraint {constraints.group(1)}" in install
    assert "==" not in install
    assert re.findall(r"--extra \S+", export) == re.findall(r"--extra \S+", gpu_sync)


# --- entrypoint-ocr.sh and docker-compose.yml -----------------------------------


def run_ocr_entrypoint(tmp_path: Path, **env: str) -> list[str]:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("needs bash")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    result = subprocess.run(  # noqa: S603 - the checked-in entrypoint with a stub server
        [bash, str(ROOT / "entrypoint-ocr.sh")],
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GLM_OCR_REVISION": "0" * 40,
            **env,
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.splitlines()


def test_ocr_server_requires_the_configured_api_key(tmp_path: Path) -> None:
    args = run_ocr_entrypoint(tmp_path, OCR_API_KEY="k" * 32)

    assert args[args.index("--api-key") + 1] == "k" * 32
    assert args[args.index("--host") + 1] == "0.0.0.0"  # noqa: S104 - asserted, not bound


@pytest.mark.parametrize("value", [None, ""])
def test_ocr_server_without_a_key_keeps_its_open_default(tmp_path: Path, value: str | None) -> None:
    env = {} if value is None else {"OCR_API_KEY": value}

    assert "--api-key" not in run_ocr_entrypoint(tmp_path, **env)


def test_compose_gives_the_ocr_server_and_its_client_the_same_key() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]

    for name in ("bibr-ocr", "bibr-serve"):
        assert "OCR_API_KEY=${OCR_API_KEY:-}" in services[name]["environment"], name
    # The split-deployment guide relies on this: the port is opt-in through an override.
    assert "ports" not in services["bibr-ocr"]
