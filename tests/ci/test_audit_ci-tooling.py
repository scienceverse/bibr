"""Regression tests for the CI classifier, workflow hardening and CI helper scripts."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.request import Request

import pytest
import yaml

ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GIT = shutil.which("git")
BASH = shutil.which("bash")
needs_git = pytest.mark.skipif(not GIT, reason="needs git")
# Workflow steps run under bash on Linux runners only. Windows' Git Bash and
# macOS (bash 3.2, BSD tools) lack the GNU tools they call, so the
# cross-platform suites skip them.
LINUX = sys.platform.startswith("linux")
needs_bash = pytest.mark.skipif(not BASH or not LINUX, reason="needs bash on Linux")
needs_flock = pytest.mark.skipif(
    not BASH or not LINUX or not shutil.which("flock"),
    reason="needs bash and flock on Linux",
)


class ActionsLoader(yaml.SafeLoader):
    """YAML loader that keeps the GitHub Actions `on` key a string."""


for _first, _resolvers in list(ActionsLoader.yaml_implicit_resolvers.items()):
    ActionsLoader.yaml_implicit_resolvers[_first] = [
        resolver for resolver in _resolvers if resolver[0] != "tag:yaml.org,2002:bool"
    ]


def workflow(name: str) -> dict:
    return yaml.load(
        (WORKFLOWS / name).read_text(encoding="utf-8"),
        Loader=ActionsLoader,  # noqa: S506 - ActionsLoader subclasses SafeLoader
    )


def step(workflow_name: str, job: str, name: str) -> dict:
    steps = workflow(workflow_name)["jobs"][job]["steps"]
    return next(entry for entry in steps if entry.get("name") == name)


def evaluate(expression: str, **contexts: object) -> object:
    """Evaluate a GitHub Actions expression built from && || ! == != and its functions.

    Python's `and`/`or` return an operand exactly as Actions' `&&`/`||` do, which
    is what makes `a && b || c` fall through to `c` whenever `b` is falsy.
    Hyphenated job names (`needs.build-dist`) are read as `needs.build_dist`.
    """

    body = " ".join(expression.split())
    if body.startswith("${{"):
        body = body.removeprefix("${{").removesuffix("}}")
    body = re.sub(r"!(?!=)", " not ", body.replace("&&", " and ").replace("||", " or "))
    body = re.sub(r"(?<=\.)[A-Za-z_][\w-]*", lambda m: m.group().replace("-", "_"), body)
    names = {
        "format": lambda template, *args: template.format(*args),
        "startsWith": lambda value, prefix: str(value).lower().startswith(prefix.lower()),
    }
    return eval(body, {"__builtins__": {}, **names}, contexts)  # noqa: S307 - own workflow file


def ns(**values: object) -> SimpleNamespace:
    return SimpleNamespace(**values)


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / "ci" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"audit_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_env(tmp_path: Path) -> dict[str, str]:
    config = tmp_path / "gitconfig"
    config.touch()
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "CI",
        "GIT_AUTHOR_EMAIL": "ci@example.invalid",
        "GIT_COMMITTER_NAME": "CI",
        "GIT_COMMITTER_EMAIL": "ci@example.invalid",
    }


def run_git(repo: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed test commands
        [str(GIT), *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_bash(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    # The flags GitHub uses for `shell: bash`.
    return subprocess.run(  # noqa: S603 - executes the checked-in workflow step
        [str(BASH), "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


# --- changed_surfaces.py: the diff must not hide moves or non-ASCII paths -------


def surfaces_for_commit(tmp_path: Path, change) -> list[str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = git_env(tmp_path)
    run_git(repo, env, "init", "-q")
    (repo / "bibr").mkdir()
    (repo / "bibr" / "core.py").write_text("value = 1\n" * 20, encoding="utf-8")
    run_git(repo, env, "add", "-A")
    run_git(repo, env, "commit", "-q", "-m", "base")
    base = run_git(repo, env, "rev-parse", "HEAD")
    change(repo, env)
    run_git(repo, env, "add", "-A")
    run_git(repo, env, "commit", "-q", "-m", "change")
    head = run_git(repo, env, "rev-parse", "HEAD")
    result = subprocess.run(  # noqa: S603 - the checked-in classifier
        [
            sys.executable,
            str(ROOT / "scripts/ci/changed_surfaces.py"),
            "--base",
            base,
            "--head",
            head,
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


@needs_git
def test_moving_package_code_out_of_bibr_still_selects_the_code_gates(tmp_path: Path) -> None:
    """Rename detection used to list only notebooks/core.py, which selected nothing."""

    def move(repo: Path, env: dict[str, str]) -> None:
        (repo / "notebooks").mkdir()
        run_git(repo, env, "mv", "bibr/core.py", "notebooks/core.py")

    assert surfaces_for_commit(tmp_path, move) == [
        "python=true",
        "package=true",
        "docs=false",
        "container=true",
        "workflow=false",
    ]


@needs_git
def test_non_ascii_package_path_selects_the_code_gates(tmp_path: Path) -> None:
    """core.quotePath turned bibr/naïve.py into '"bibr/na\\303\\257ve.py"'."""

    def add(repo: Path, env: dict[str, str]) -> None:
        (repo / "bibr" / "naïve.py").write_text("value = 2\n", encoding="utf-8")

    assert surfaces_for_commit(tmp_path, add) == [
        "python=true",
        "package=true",
        "docs=false",
        "container=true",
        "workflow=false",
    ]


def test_rename_records_contribute_both_paths(monkeypatch) -> None:
    changed_surfaces = load_script("changed_surfaces")
    output = b"R100\0bibr/core.py\0notebooks/core.py\0M\0docs/na\xc3\xafve.md\0"
    monkeypatch.setattr(
        changed_surfaces.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, output, b""),
    )

    assert changed_surfaces.changed_paths("base", "head") == [
        "bibr/core.py",
        "notebooks/core.py",
        "docs/naïve.md",
    ]


@pytest.mark.parametrize("path", ["history/new_analysis.py", "tools/build.py", "SECURITY.md"])
def test_unrecognised_paths_in_any_directory_fail_safe(path: str) -> None:
    surfaces = load_script("changed_surfaces").classify_paths([path])

    assert surfaces["python"] and surfaces["package"] and surfaces["workflow"]


@pytest.mark.parametrize("path", ["notebooks/python_api_demo.ipynb", "examples/platform/x.py"])
def test_executed_notebooks_and_examples_select_the_suite(path: str) -> None:
    assert load_script("changed_surfaces").classify_paths([path])["python"] is True


@pytest.mark.parametrize("path", ["CHANGELOG.md", "CONTRIBUTING.md", "bibr/processing_warnings.py"])
def test_docs_guard_and_generated_reference_inputs_select_docs(path: str) -> None:
    assert load_script("changed_surfaces").classify_paths([path])["docs"] is True


def test_dist_gate_dependency_selects_the_package_build() -> None:
    """check_dist_contents.py imports its dataset suffixes from the tree guard."""
    surfaces = load_script("changed_surfaces").classify_paths(["scripts/check_public_tree.py"])

    assert surfaces["package"] is True


@pytest.mark.parametrize(
    "path",
    [
        "bibr/segmenter_base.py",
        "scripts/prefetch_segmenter.py",
        "entrypoint.sh",
        "entrypoint-ocr.sh",
    ],
)
def test_serve_image_inputs_select_the_container_build(path: str) -> None:
    """publish-edge and the PR image build both key off the container surface."""
    assert load_script("changed_surfaces").classify_paths([path])["container"] is True


@pytest.mark.parametrize(
    "path",
    [
        "entrypoint.sh",
        "entrypoint-ocr.sh",
        "Dockerfile.serve",
        "Dockerfile.ocr",
        "docker-compose.yml",
        ".dockerignore",
    ],
)
def test_container_files_also_run_the_suite_that_tests_them(path: str) -> None:
    """The image build never runs entrypoint-ocr.sh; only the suite's tests check it."""
    surfaces = load_script("changed_surfaces").classify_paths([path])

    assert surfaces["python"] is True
    assert surfaces["container"] is True


# --- ci.yml / docker.yml / release.yml policy -----------------------------------


@pytest.mark.parametrize(("event", "cancels"), [("pull_request", True), ("push", False)])
def test_only_pull_request_runs_cancel_in_progress(event: str, cancels: bool) -> None:
    expression = workflow("ci.yml")["concurrency"]["cancel-in-progress"]

    assert evaluate(expression, github=ns(event_name=event, ref="refs/heads/main")) is cancels


def ci_cache_contexts(environment: str, ref: str) -> dict[str, object]:
    return {
        "runner": ns(environment=environment),
        "github": ns(ref=ref),
        "steps": ns(cache=ns(outputs=ns(dir="/cache.job-1"))),
    }


@pytest.mark.parametrize(
    ("environment", "ref", "expected"),
    [
        ("self-hosted", "refs/pull/7/merge", ""),
        ("self-hosted", "refs/heads/unreviewed", ""),
        ("self-hosted", "refs/heads/main", "type=local,dest=/cache.job-1/export,mode=max"),
        ("github-hosted", "refs/pull/7/merge", "type=gha,mode=max"),
    ],
)
def test_pull_requests_never_export_into_the_shared_fleet_cache(
    environment: str, ref: str, expected: str
) -> None:
    build = step("ci.yml", "container-build-scan", "Build unprivileged validation image")
    contexts = ci_cache_contexts(environment, ref)

    assert evaluate(build["with"]["cache-to"], **contexts) == expected
    if environment == "self-hosted":
        assert evaluate(build["with"]["cache-from"], **contexts) == (
            "type=local,src=/cache.job-1/read"
        )


@pytest.mark.parametrize(
    ("channel", "environment", "expected"),
    [
        ("release", "self-hosted", ""),
        ("release", "github-hosted", ""),
        ("edge", "self-hosted", "type=local,src=/cache.job-1/read"),
        ("edge", "github-hosted", "type=gha"),
    ],
)
def test_release_images_build_without_any_cache(
    channel: str, environment: str, expected: str
) -> None:
    build = step("docker.yml", "publish", "Build and push only the quarantine identity")
    snapshot = step("docker.yml", "publish", "Snapshot the shared layer cache")
    contexts = {
        "inputs": ns(channel=channel),
        "runner": ns(environment=environment),
        "steps": ns(cache=ns(outputs=ns(dir="/cache.job-1"))),
    }

    assert evaluate(build["with"]["cache-from"], **contexts) == expected
    assert evaluate(build["with"]["no-cache"], **contexts) is (channel == "release")
    assert "cache-to" not in build["with"]
    assert evaluate(snapshot["if"], **contexts) is (
        channel == "edge" and environment == "self-hosted"
    )


def cache_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "CACHE_DIR": str(tmp_path / "buildx-cache"),
        "MIN_FREE_GB": "0",
        "GITHUB_RUN_ID": "7",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_JOB": "container-build-scan",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        **extra,
    }


def prepare_cache(tmp_path: Path) -> Path:
    prepare = step("ci.yml", "container-build-scan", "Ensure the cache disk has room for the build")
    result = run_bash(prepare["run"], tmp_path, cache_env(tmp_path))
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in (tmp_path / "output").read_text().splitlines())
    return Path(outputs["dir"])


@needs_flock
def test_each_build_reads_a_snapshot_and_sweeps_only_stale_job_directories(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "buildx-cache"
    (cache / "blobs").mkdir(parents=True)
    (cache / "blobs" / "layer").write_text("layer", encoding="utf-8")
    stale = tmp_path / "buildx-cache.job-1-1-container-build-scan"
    running = tmp_path / "buildx-cache.job-2-1-container-build-scan"
    legacy = tmp_path / "buildx-cache.new"
    for directory in (stale, running, legacy):
        directory.mkdir()
    for directory in (stale, legacy):
        os.utime(directory, (0, 0))

    job_dir = prepare_cache(tmp_path)

    assert job_dir == tmp_path / "buildx-cache.job-7-1-container-build-scan"
    snapshot = job_dir / "read" / "blobs" / "layer"
    assert snapshot.stat().st_ino == (cache / "blobs" / "layer").stat().st_ino
    assert not stale.exists()
    assert not legacy.exists()
    assert running.exists()


@needs_flock
@pytest.mark.parametrize(
    ("ref", "swapped"),
    [("refs/pull/7/merge", False), ("refs/heads/unreviewed", False), ("refs/heads/main", True)],
)
def test_only_main_swaps_its_export_into_the_shared_cache(
    tmp_path: Path, ref: str, swapped: bool
) -> None:
    cache = tmp_path / "buildx-cache"
    cache.mkdir()
    (cache / "index.json").write_text("main", encoding="utf-8")
    job_dir = prepare_cache(tmp_path)
    (job_dir / "export").mkdir()
    (job_dir / "export" / "index.json").write_text(ref, encoding="utf-8")
    swap = step("ci.yml", "container-build-scan", "Swap in the freshly exported layer cache")
    env = cache_env(tmp_path, JOB_DIR=str(job_dir), BUILD_OUTCOME="success", SOURCE_REF=ref)

    result = run_bash(swap["run"], tmp_path, env)

    assert result.returncode == 0, result.stderr
    assert (cache / "index.json").read_text() == (ref if swapped else "main")
    assert not job_dir.exists()


@needs_flock
def test_cache_swap_waits_for_the_shared_lock(tmp_path: Path) -> None:
    cache = tmp_path / "buildx-cache"
    cache.mkdir()
    job_dir = prepare_cache(tmp_path)
    swap = step("ci.yml", "container-build-scan", "Swap in the freshly exported layer cache")
    env = cache_env(
        tmp_path, JOB_DIR=str(job_dir), BUILD_OUTCOME="failure", SOURCE_REF="refs/heads/main"
    )

    fcntl = pytest.importorskip("fcntl")
    with open(tmp_path / "buildx-cache.lock", "w") as lock:  # noqa: PTH123 - fcntl needs a file
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = subprocess.Popen(  # noqa: S603 - executes the checked-in workflow step
            [str(BASH), "--noprofile", "--norc", "-eo", "pipefail", "-c", swap["run"]],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.5)
            assert job_dir.exists()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    assert process.wait(timeout=30) == 0
    assert not job_dir.exists()


UV_SHIM = """#!/usr/bin/env bash
# Stands in for setup-uv's uv: the step must run its Python through uv.
[[ "$1 $2 $3" == "run --no-project python" ]] || exit 97
shift 3
exec "$SHIM_PYTHON" "$@"
"""


def with_tools(tmp_path: Path, env: dict[str, str]) -> dict[str, str]:
    bin_dir = tmp_path / "tools"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(UV_SHIM, encoding="utf-8")
    uv.chmod(0o755)
    path = os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent), env["PATH"]])
    return {**env, "PATH": path, "SHIM_PYTHON": sys.executable}


def docker_validation_repo(tmp_path: Path) -> tuple[Path, dict[str, str], str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = git_env(tmp_path)
    run_git(repo, env, "init", "-q", "-b", "main")
    (repo / "pyproject.toml").write_text('[project]\nversion = "0.7.0"\n', encoding="utf-8")
    run_git(repo, env, "add", "pyproject.toml")
    run_git(repo, env, "commit", "-q", "-m", "release 0.7.0")
    released = run_git(repo, env, "rev-parse", "HEAD")
    run_git(repo, env, "tag", "-a", "v0.7.0", "-m", "v0.7.0")
    run_git(repo, env, "update-ref", "refs/remotes/origin/main", "HEAD")
    run_git(repo, env, "checkout", "-q", "-b", "feature")
    run_git(repo, env, "commit", "-q", "--allow-empty", "-m", "unmerged")
    unmerged = run_git(repo, env, "rev-parse", "HEAD")
    return repo, env, released, unmerged


def origin_main_tag_repo(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    """An unmerged 0.7.1 commit carrying v0.7.1 and a tag named origin/main.

    Both are tags any write-access user can push. git resolves the short name
    origin/main to refs/tags/origin/main before refs/remotes/origin/main.
    """

    repo, env, _released, _unmerged = docker_validation_repo(tmp_path)
    (repo / "pyproject.toml").write_text('[project]\nversion = "0.7.1"\n', encoding="utf-8")
    run_git(repo, env, "commit", "-q", "-am", "unmerged 0.7.1")
    source = run_git(repo, env, "rev-parse", "HEAD")
    run_git(repo, env, "tag", "origin/main", source)
    run_git(repo, env, "tag", "-a", "v0.7.1", "-m", "v0.7.1", source)
    return repo, env, source


@needs_git
@needs_bash
@pytest.mark.parametrize(
    ("channel", "source", "version", "accepted"),
    [
        ("release", "released", "0.7.0", True),
        ("edge", "released", "", True),
        ("edge", "unmerged", "", False),
        ("release", "unmerged", "0.7.0", False),
        ("release", "released", "0.6.0", False),
    ],
)
def test_container_dispatch_requires_main_ancestry_version_and_tag(
    tmp_path: Path, channel: str, source: str, version: str, accepted: bool
) -> None:
    repo, env, released, unmerged = docker_validation_repo(tmp_path)
    sha = released if source == "released" else unmerged
    run_git(repo, env, "checkout", "-q", sha)
    check = step(
        "docker.yml", "publish", "Require a main commit and, for releases, its version tag"
    )
    env = with_tools(tmp_path, {**env, "CHANNEL": channel, "SOURCE_SHA": sha, "VERSION": version})

    result = run_bash(check["run"], repo, env)

    assert (result.returncode == 0) is accepted, result.stdout + result.stderr


@needs_git
@needs_bash
@pytest.mark.parametrize(("channel", "version"), [("edge", ""), ("release", "0.7.1")])
def test_container_dispatch_ignores_a_tag_named_origin_main(
    tmp_path: Path, channel: str, version: str
) -> None:
    repo, env, source = origin_main_tag_repo(tmp_path)
    check = step(
        "docker.yml", "publish", "Require a main commit and, for releases, its version tag"
    )
    env = with_tools(
        tmp_path, {**env, "CHANNEL": channel, "SOURCE_SHA": source, "VERSION": version}
    )

    result = run_bash(check["run"], repo, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"source commit {source} is not reachable from origin/main" in result.stderr


@needs_git
@needs_bash
@pytest.mark.parametrize("tag", ["deleted", "on an earlier main commit"])
def test_release_dispatch_requires_the_version_tag_on_the_source_commit(
    tmp_path: Path, tag: str
) -> None:
    repo, env, released, _unmerged = docker_validation_repo(tmp_path)
    run_git(repo, env, "checkout", "-q", "main")
    if tag == "deleted":
        run_git(repo, env, "tag", "-d", "v0.7.0")
    else:
        # A later main commit still carries version 0.7.0 but was never tagged.
        run_git(repo, env, "commit", "-q", "--allow-empty", "-m", "after the release")
        run_git(repo, env, "update-ref", "refs/remotes/origin/main", "HEAD")
    source = run_git(repo, env, "rev-parse", "HEAD")
    check = step(
        "docker.yml", "publish", "Require a main commit and, for releases, its version tag"
    )
    env = with_tools(
        tmp_path, {**env, "CHANNEL": "release", "SOURCE_SHA": source, "VERSION": "0.7.0"}
    )

    result = run_bash(check["run"], repo, env)

    assert result.returncode == 1
    assert f"tag v0.7.0 does not exist or does not point at {source}" in result.stderr


def test_container_checkout_fetches_main_and_tags_for_the_source_checks() -> None:
    steps = [entry.get("name") for entry in workflow("docker.yml")["jobs"]["publish"]["steps"]]
    checkout = step("docker.yml", "publish", "Checkout exact source")

    assert checkout["with"]["fetch-depth"] == 0
    assert steps.index("Require a main commit and, for releases, its version tag") < steps.index(
        "Build and push only the quarantine identity"
    )
    # The version check runs the release job's Python through uv, uncached.
    setup = step("docker.yml", "publish", "Install locked uv")
    assert setup["with"]["enable-cache"] in (False, "false")
    assert steps.index("Install locked uv") < steps.index(
        "Require a main commit and, for releases, its version tag"
    )


FAKE_DOCKER = """#!/usr/bin/env bash
if [[ "$1 $2 $3" == "buildx imagetools inspect" ]]; then
  case "$FAKE_PUBLISHED" in
    missing) echo "ERROR: $4: not found" >&2; exit 1 ;;
    error) echo "ERROR: failed to authorize: 503 Service Unavailable" >&2; exit 1 ;;
    *) printf 'Name:      %s\\nMediaType: x\\nDigest:    %s\\n' "$4" "$FAKE_PUBLISHED" ;;
  esac
elif [[ "$1 $2 $3" == "buildx imagetools create" ]]; then
  echo "$*" >> "$FAKE_CREATED"
fi
"""
DIGEST = "sha256:" + "a" * 64


def promote_release(
    tmp_path: Path, version: str, *, published: str = "missing", tags: tuple[str, ...] = ()
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the promotion step for *version* in a checkout whose main carries *tags*."""

    repo = tmp_path / "repo"
    repo.mkdir()
    env = git_env(tmp_path)
    run_git(repo, env, "init", "-q", "-b", "main")
    for tag in (*tags, f"v{version}"):
        run_git(repo, env, "commit", "-q", "--allow-empty", "-m", tag)
        run_git(repo, env, "tag", tag)
    run_git(repo, env, "update-ref", "refs/remotes/origin/main", "HEAD")
    # Neither a tag off main nor a pre-release names a release of the series.
    run_git(repo, env, "checkout", "-q", "-b", "unmerged")
    run_git(repo, env, "commit", "-q", "--allow-empty", "-m", "unmerged")
    run_git(repo, env, "tag", "v9.9.9")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    created = tmp_path / "created"
    output = tmp_path / "output"
    promote = step("docker.yml", "publish", "Promote the verified digest without rebuilding")
    env = {
        **env,
        "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        "REGISTRY": "ghcr.io",
        "IMAGE_NAME": "scienceverse/bibr",
        "CHANNEL": "release",
        "DIGEST": DIGEST,
        "VERSION": version,
        "GITHUB_OUTPUT": str(output),
        "FAKE_PUBLISHED": published,
        "FAKE_CREATED": str(created),
    }

    result = run_bash(promote["run"], repo, env)

    moved: list[str] = []
    if created.exists():
        moved = re.findall(r"--tag ghcr\.io/scienceverse/bibr:(\S+)", created.read_text())
        assert f"tags={','.join(moved)}" in output.read_text().splitlines()
    return result, moved


@needs_git
@needs_bash
@pytest.mark.parametrize(
    ("published", "promoted"),
    [("missing", True), (DIGEST, True), ("sha256:" + "b" * 64, False), ("error", False)],
)
def test_release_promotion_never_moves_a_published_version_tag(
    tmp_path: Path, published: str, promoted: bool
) -> None:
    result, moved = promote_release(tmp_path, "0.7.0", published=published)

    assert (result.returncode == 0) is promoted, result.stderr
    assert moved == (["v0.7.0", "0.7", "0", "latest"] if promoted else [])


@needs_git
@needs_bash
@pytest.mark.parametrize(
    ("version", "moved"),
    [
        ("0.7.2", ["v0.7.2", "0.7", "0", "latest"]),
        # A backport moves its own minor series only.
        ("0.6.4", ["v0.6.4", "0.6"]),
        # An older release that was never published keeps every series tag where it is.
        ("0.5.0", ["v0.5.0"]),
    ],
)
def test_release_promotion_moves_series_tags_only_to_their_newest_release(
    tmp_path: Path, version: str, moved: list[str]
) -> None:
    released = ("v0.5.1", "v0.6.3", "v0.7.0", "v0.7.1", "v1.0.0rc1")

    result, promoted = promote_release(tmp_path, version, tags=released)

    assert result.returncode == 0, result.stderr
    assert promoted == moved


def release_validation(
    repo: Path, env: dict[str, str], version: str, sha: str, output: Path
) -> subprocess.CompletedProcess[str]:
    script = next(
        entry["run"]
        for entry in workflow("release.yml")["jobs"]["validate"]["steps"]
        if entry.get("id") == "version"
    )
    env = {
        **env,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": f"refs/tags/v{version}",
        "GITHUB_REF_NAME": f"v{version}",
        "GITHUB_SHA": sha,
        "GITHUB_OUTPUT": str(output),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}",
    }
    return run_bash(script, repo, env)


@needs_git
@needs_bash
@pytest.mark.parametrize(
    ("version", "prerelease"),
    [("0.7.0", "false"), ("0.7.0rc1", "true"), ("0.7.0b2", "true"), ("0.7.0.dev1", "true")],
)
def test_release_tags_flag_every_pre_release(tmp_path: Path, version: str, prerelease: str) -> None:
    """Anything but X.Y.Z is flagged, so it skips docker.yml, which refuses it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = git_env(tmp_path)
    run_git(repo, env, "init", "-q", "-b", "main")
    run_git(repo, env, "commit", "-q", "--allow-empty", "-m", "main")
    run_git(repo, env, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    output = tmp_path / "outputs"
    sha = run_git(repo, env, "rev-parse", "HEAD")

    result = release_validation(repo, env, version, sha, output)

    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text().split() == [f"version={version}", f"prerelease={prerelease}"]


def release_contexts(prerelease: str, ghcr: str, container: str) -> dict[str, object]:
    return {
        "github": ns(event_name="push", ref="refs/tags/v0.7.0"),
        "vars": ns(PUBLISH_GHCR=ghcr, PUBLISH_PYPI="true"),
        "needs": ns(
            validate=ns(outputs=ns(prerelease=prerelease)),
            build_dist=ns(result="success"),
            publish_pypi=ns(result="success"),
            publish_container=ns(result=container),
        ),
        "cancelled": lambda: False,
    }


@pytest.mark.parametrize(("prerelease", "runs"), [("false", True), ("true", False)])
def test_pre_releases_skip_the_container_channel(prerelease: str, runs: bool) -> None:
    condition = workflow("release.yml")["jobs"]["publish-container"]["if"]

    assert evaluate(condition, **release_contexts(prerelease, "true", "pending")) is runs


@pytest.mark.parametrize(
    ("prerelease", "ghcr", "container", "finalized"),
    [
        ("false", "true", "success", True),
        ("false", "true", "failure", False),
        # An enabled channel that did not run blocks a final release...
        ("false", "true", "skipped", False),
        # ...but not a pre-release, which has no container, nor a disabled channel.
        ("true", "true", "skipped", True),
        ("false", "false", "skipped", True),
    ],
)
def test_github_release_waits_for_every_channel_a_release_publishes(
    prerelease: str, ghcr: str, container: str, finalized: bool
) -> None:
    condition = workflow("release.yml")["jobs"]["github-release"]["if"]

    assert evaluate(condition, **release_contexts(prerelease, ghcr, container)) is finalized


@needs_git
@needs_bash
def test_release_validation_ignores_a_tag_named_origin_main(tmp_path: Path) -> None:
    repo, env, source = origin_main_tag_repo(tmp_path)
    output = tmp_path / "outputs"

    result = release_validation(repo, env, "0.7.1", source, output)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "release commit is not reachable from origin/main" in result.stderr
    assert not output.exists()


# --- audit_dependencies.py ------------------------------------------------------


@pytest.mark.parametrize("body", ["", "# all exceptions resolved\n"])
def test_policy_without_exceptions_audits_with_no_ignores(tmp_path: Path, body: str) -> None:
    audit = load_script("audit_dependencies")
    policy = tmp_path / "exceptions.toml"
    policy.write_text(body, encoding="utf-8")

    entries = audit.load_exceptions(policy, today=date(2026, 10, 6))

    assert entries == []
    assert "--ignore-vuln" not in audit.build_command(entries)


def test_policy_with_a_non_table_exception_key_still_fails(tmp_path: Path) -> None:
    audit = load_script("audit_dependencies")
    policy = tmp_path / "exceptions.toml"
    policy.write_text('exception = "CVE-2099-0001"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[\[exception\]\]"):
        audit.load_exceptions(policy, today=date(2026, 10, 6))


# --- smoke_site.py --------------------------------------------------------------


def redirect(smoke: ModuleType, source: str, target: str) -> Request | None:
    request = Request(  # noqa: S310 - never opened
        source, headers={"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "secret"}
    )
    return smoke.SameOriginRedirect().redirect_request(request, None, 302, "Found", {}, target)


@pytest.mark.parametrize(
    "target",
    [
        "https://collector.example.net/bibr-build.txt",
        "http://preview.example.org/bibr-build.txt",
        "https://preview.example.org:8443/bibr-build.txt",
    ],
)
def test_protected_smoke_never_forwards_the_access_secret_off_origin(target: str) -> None:
    smoke = load_script("smoke_site")

    assert redirect(smoke, "https://preview.example.org/bibr-build.txt", target) is None


def test_protected_smoke_still_follows_same_origin_redirects() -> None:
    smoke = load_script("smoke_site")

    followed = redirect(
        smoke, "https://preview.example.org/bibr-build.txt", "https://preview.example.org:443/b"
    )

    assert followed is not None
    assert followed.full_url == "https://preview.example.org:443/b"
    assert followed.headers["Cf-access-client-secret"] == "secret"


def test_authorized_probe_uses_the_same_origin_redirect_handler(monkeypatch) -> None:
    smoke = load_script("smoke_site")
    handlers = []

    def build_opener(*args):
        handlers.extend(args)
        raise OSError("stop before any request")

    monkeypatch.setattr(smoke, "build_opener", build_opener)
    with pytest.raises(OSError, match="stop"):
        smoke._fetch("https://preview.example.org/x", follow_redirects=True)

    assert handlers == [smoke.SameOriginRedirect]
