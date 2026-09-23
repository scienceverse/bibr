from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "scripts" / "ci" / "changed_surfaces.py"
# The pre-push tip of the 2026-09-22 main rewrite, absent from any fresh clone.
REWRITTEN_SHA = "51b102f19aa5834ea631142935917ae3b849a1e2"


def load_changed_surfaces() -> ModuleType:
    assert MODULE_PATH.is_file(), "changed-surfaces helper has not been implemented"
    spec = importlib.util.spec_from_file_location("changed_surfaces", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_docs_only_avoids_python_package_and_container() -> None:
    changed_surfaces = load_changed_surfaces()
    assert changed_surfaces.classify_paths(["docs/index.md"]) == {
        "python": False,
        "package": False,
        "docs": True,
        "container": False,
        "workflow": False,
    }


def test_generated_reference_source_selects_python_package_and_docs() -> None:
    changed_surfaces = load_changed_surfaces()
    surfaces = changed_surfaces.classify_paths(["bibr/config.py"])

    assert surfaces["python"] is True
    assert surfaces["package"] is True
    assert surfaces["docs"] is True
    assert surfaces["container"] is False


def test_cli_package_source_selects_generated_docs() -> None:
    changed_surfaces = load_changed_surfaces()
    surfaces = changed_surfaces.classify_paths(["bibr/local/cli/parser.py"])

    assert surfaces["python"] is True
    assert surfaces["package"] is True
    assert surfaces["docs"] is True


def test_root_site_pages_select_docs_build() -> None:
    changed_surfaces = load_changed_surfaces()
    for path in ("LIMITATIONS.md", "LLM_POLICY.md"):
        assert changed_surfaces.classify_paths([path])["docs"] is True


def test_lockfile_selects_all_surfaces() -> None:
    changed_surfaces = load_changed_surfaces()
    assert all(changed_surfaces.classify_paths(["uv.lock"]).values())


def test_dockerfile_selects_only_container() -> None:
    changed_surfaces = load_changed_surfaces()
    assert changed_surfaces.classify_paths(["Dockerfile.serve"]) == {
        "python": False,
        "package": False,
        "docs": False,
        "container": True,
        "workflow": False,
    }


def test_workflow_change_selects_every_surface() -> None:
    changed_surfaces = load_changed_surfaces()
    assert all(changed_surfaces.classify_paths([".github/workflows/ci.yml"]).values())


def test_ci_tool_lock_selects_every_surface() -> None:
    changed_surfaces = load_changed_surfaces()

    assert all(changed_surfaces.classify_paths(["ci/tools/uv.lock"]).values())


def test_unknown_root_file_fails_safe() -> None:
    changed_surfaces = load_changed_surfaces()
    surfaces = changed_surfaces.classify_paths(["new-build-input.cfg"])

    assert surfaces["python"] is True
    assert surfaces["package"] is True
    assert surfaces["workflow"] is True


def test_unknown_root_file_fails_safe_when_other_paths_are_known() -> None:
    changed_surfaces = load_changed_surfaces()

    surfaces = changed_surfaces.classify_paths(["docs/index.md", "new-build-input.cfg"])

    assert surfaces["python"] is True
    assert surfaces["package"] is True
    assert surfaces["workflow"] is True


def test_all_cli_emits_github_output_lines() -> None:
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "python=true",
        "package=true",
        "docs=true",
        "container=true",
        "workflow=true",
    ]


def test_changed_paths_uses_merge_base_diff(monkeypatch) -> None:
    changed_surfaces = load_changed_surfaces()
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "docs/index.md\nbibr/config.py\n", "")

    monkeypatch.setattr(changed_surfaces.subprocess, "run", fake_run)

    assert changed_surfaces.changed_paths("base", "head") == [
        "docs/index.md",
        "bibr/config.py",
    ]
    assert recorded["command"] == [
        "git",
        "diff",
        "--name-only",
        "--diff-filter=ACDMRT",
        "base...head",
    ]


def _init_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a throwaway repository isolated from the developer's Git config."""

    empty_config = tmp_path / "gitconfig"
    empty_config.touch()
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(empty_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "CI",
        "GIT_AUTHOR_EMAIL": "ci@example.invalid",
        "GIT_COMMITTER_NAME": "CI",
        "GIT_COMMITTER_EMAIL": "ci@example.invalid",
    }
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, env, "init", "-q")
    return repo, env


def _git(repo: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(
        ["git", *args],  # noqa: S607 - the helper under test shells out to PATH git too
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, env: dict[str, str], path: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content\n", encoding="utf-8")
    _git(repo, env, "add", path)
    _git(repo, env, "commit", "-q", "-m", f"add {path}")
    return _git(repo, env, "rev-parse", "HEAD")


def _run_cli(repo: Path, env: dict[str, str], base: str, head: str):
    return subprocess.run(
        [sys.executable, str(MODULE_PATH), "--base", base, "--head", head],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_classifies_the_diff_when_the_base_commit_is_present(tmp_path: Path) -> None:
    repo, env = _init_repo(tmp_path)
    base = _commit(repo, env, "bibr/config.py")
    head = _commit(repo, env, "docs/index.md")

    result = _run_cli(repo, env, base, head)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "python=false",
        "package=false",
        "docs=true",
        "container=false",
        "workflow=false",
    ]
    assert result.stderr == ""


def test_cli_selects_every_surface_when_a_force_push_dropped_the_base(
    tmp_path: Path,
) -> None:
    """A force push's ``github.event.before`` is unreachable from every ref, so the
    fresh clone never fetched it; diffing against it exits 128 and failed CI."""

    repo, env = _init_repo(tmp_path)
    head = _commit(repo, env, "docs/index.md")

    result = _run_cli(repo, env, REWRITTEN_SHA, head)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "python=true",
        "package=true",
        "docs=true",
        "container=true",
        "workflow=true",
    ]
    assert f"::warning title=Changed surfaces unknown::base {REWRITTEN_SHA}" in result.stderr


def test_is_commit_rejects_missing_objects_and_non_commits(tmp_path: Path, monkeypatch) -> None:
    changed_surfaces = load_changed_surfaces()
    repo, env = _init_repo(tmp_path)
    head = _commit(repo, env, "docs/index.md")
    tree = _git(repo, env, "rev-parse", "HEAD^{tree}")
    monkeypatch.chdir(repo)

    assert changed_surfaces.is_commit(head) is True
    assert changed_surfaces.is_commit(REWRITTEN_SHA) is False
    assert changed_surfaces.is_commit(tree) is False
