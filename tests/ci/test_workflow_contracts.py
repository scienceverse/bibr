from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class ActionsLoader(yaml.SafeLoader):
    """YAML loader that does not misparse the GitHub Actions `on` key as bool."""


for first_character, resolvers in list(ActionsLoader.yaml_implicit_resolvers.items()):
    ActionsLoader.yaml_implicit_resolvers[first_character] = [
        resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"
    ]


def workflow(name: str) -> dict[str, Any]:
    loaded = yaml.load(
        (WORKFLOWS / name).read_text(encoding="utf-8"),
        Loader=ActionsLoader,  # noqa: S506 - ActionsLoader subclasses SafeLoader
    )
    assert isinstance(loaded, dict)
    return loaded


def workflow_text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def action_references(text: str) -> list[str]:
    return re.findall(r"^\s*uses:\s*([^\s#]+)", text, flags=re.MULTILINE)


def test_ci_runs_for_main_only() -> None:
    # The retired `dev` branch (February, Ray-era) used to trigger the whole
    # suite on every push; it is pruned, not maintained.
    ci = workflow("ci.yml")

    assert ci["on"]["push"]["branches"] == ["main"]
    assert ci["on"]["pull_request"]["branches"] == ["main"]
    assert "workflow_dispatch" in ci["on"]


def test_ci_has_read_only_defaults_and_stale_run_cancellation() -> None:
    ci = workflow("ci.yml")

    assert ci["permissions"] == {"contents": "read"}
    assert "github.workflow" in ci["concurrency"]["group"]
    assert "github.event.pull_request.number" in ci["concurrency"]["group"]
    assert "refs/tags/" in ci["concurrency"]["cancel-in-progress"]


def test_every_ci_job_has_a_timeout() -> None:
    jobs = workflow("ci.yml")["jobs"]

    assert jobs
    assert {
        job_id for job_id, job in jobs.items() if "uses" not in job and "timeout-minutes" not in job
    } == set()


def test_linux_runner_routing_guards_public_repositories_and_forks() -> None:
    for path in WORKFLOWS.glob("*.yml"):
        for job_id, job in workflow(path.name)["jobs"].items():
            runner = job.get("runs-on", "")
            if "matrix.os" in runner or "uses" in job:
                continue
            if (path.name, job_id) in {
                ("release.yml", "publish-pypi"),
                ("ci.yml", "deploy-production"),
            }:
                # Publishing jobs use disposable GitHub-hosted runners.
                assert runner == "ubuntu-latest"
                continue
            assert "vars.CI_RUNNER" in runner, (path.name, job_id)
            assert "ubuntu-latest" in runner, (path.name, job_id)
            if job_id == "sonar":
                assert "github.event.repository.private" in job["if"]
                assert "github.event_name == 'push'" in job["if"]
            else:
                assert "github.event.repository.private" in runner, (path.name, job_id)
                assert "github.event_name != 'pull_request'" in runner, (path.name, job_id)
                assert "head.repo.full_name == github.repository" in runner, (path.name, job_id)


def test_hosted_disk_cleanup_never_runs_on_the_fleet() -> None:
    for name in ("ci.yml", "docker.yml"):
        for job in workflow(name)["jobs"].values():
            for step in job.get("steps", []):
                if "sudo rm -rf" in step.get("run", ""):
                    assert step["if"] == "runner.environment == 'github-hosted'"


def test_primary_suite_installs_and_runs_the_mcp_extra() -> None:
    runs = [step.get("run", "") for step in workflow("ci.yml")["jobs"]["test-primary"]["steps"]]
    assert any("uv sync" in run and "--extra mcp" in run for run in runs)
    assert any("pytest" in run and "--extra mcp" in run for run in runs)


def test_dependabot_keeps_coverage_gate_without_uploading_with_missing_secrets() -> None:
    steps = workflow("ci.yml")["jobs"]["test-primary"]["steps"]
    suite = next(step for step in steps if "--cov-fail-under" in step.get("run", ""))
    assert "if" not in suite
    assert "--cov-fail-under=75" in suite["run"]
    codecov = next(step for step in steps if step.get("uses", "").startswith("codecov/"))
    assert "github.actor != 'dependabot[bot]'" in codecov["if"]
    assert "head.repo.full_name == github.repository" in codecov["if"]
    assert codecov["with"]["fail_ci_if_error"] == "true"
    artifact = next(step for step in steps if step.get("with", {}).get("path") == "coverage.xml")
    assert "if" not in artifact
    assert artifact["with"]["if-no-files-found"] == "error"


def test_wheel_smokes_use_per_job_scratch_directories() -> None:
    for name in ("ci.yml", "release.yml"):
        text = workflow_text(name)
        assert "/tmp/bibr-wheel-smoke" not in text
        assert "/tmp/bibr-release-smoke" not in text
        assert "${RUNNER_TEMP}/bibr-" in text


def test_every_external_action_is_pinned_to_a_full_sha() -> None:
    for path in WORKFLOWS.glob("*.yml"):
        for reference in action_references(path.read_text(encoding="utf-8")):
            if reference.startswith("./"):
                continue
            assert "@" in reference, f"{path.name}: unpinned action {reference}"
            _action, revision = reference.rsplit("@", 1)
            assert FULL_SHA.fullmatch(revision), f"{path.name}: mutable action {reference}"


def test_every_checkout_disables_persisted_credentials() -> None:
    offenders: list[str] = []
    for path in WORKFLOWS.glob("*.yml"):
        for job_id, job in workflow(path.name)["jobs"].items():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("actions/checkout@") and step.get(
                    "with", {}
                ).get("persist-credentials") not in (False, "false"):
                    offenders.append(f"{path.name}:{job_id}")

    assert offenders == []


def test_ci_uv_environment_commands_are_locked() -> None:
    offenders: list[str] = []
    for line in workflow_text("ci.yml").splitlines():
        stripped = line.strip()
        if ("uv sync" in stripped or "uv run" in stripped) and "--locked" not in stripped:
            offenders.append(stripped)

    assert offenders == []


def test_workflow_policy_tools_use_their_isolated_lock() -> None:
    quality = yaml.dump(workflow("ci.yml")["jobs"]["quality"])

    assert "uv sync --project ci/tools --locked" in quality
    assert "uv run --project ci/tools --locked actionlint" in quality
    assert "uv run --project ci/tools --locked zizmor" in quality


def test_ci_does_not_discard_failures() -> None:
    text = workflow_text("ci.yml")

    assert "|| true" not in text
    assert "quality-evals" not in text
    assert "Extraction quality gates" not in text


def test_ci_exposes_one_stable_required_aggregator() -> None:
    jobs = workflow("ci.yml")["jobs"]
    required = jobs["required"]
    expected_needs = {
        "changes",
        "quality",
        "test-primary",
        "core-compat",
        "core-install",
        "package",
        "contract-smokes",
        "security",
        "semgrep",
        "secrets",
        "docs-build",
        "container-build-scan",
    }

    assert required["name"] == "CI / required"
    assert "always()" in required["if"]
    assert set(required["needs"]) == expected_needs


def test_ci_runs_one_full_primary_and_boundary_compatibility() -> None:
    jobs = workflow("ci.yml")["jobs"]
    primary = jobs["test-primary"]
    compatibility = jobs["core-compat"]
    primary_text = yaml.dump(primary)

    assert "3.12" in primary_text
    assert "--extra all" in primary_text
    assert set(compatibility["strategy"]["matrix"]["python-version"]) == {"3.11", "3.14"}


def test_ci_builds_one_commit_marked_docs_artifact() -> None:
    job = workflow("ci.yml")["jobs"]["docs-build"]
    docs = yaml.dump(job)

    assert "mkdocs build --strict" in docs
    assert "bibr-build.txt" in docs
    assert "mkdocs-site-${{ github.sha }}" in docs
    assert "retention-days: 7" in docs
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )["with"]
    assert upload["if-no-files-found"] == "error"
    assert "mkdir -p site/.well-known" not in docs


def test_semgrep_is_isolated_in_an_immutable_root_container() -> None:
    semgrep = workflow("ci.yml")["jobs"]["semgrep"]

    assert semgrep["container"]["image"] == (
        "semgrep/semgrep@sha256:58889e4ef89e779e33dfd3f90c87dd1c568670584a2fee5ad376e5451e9a64e4"
    )
    assert semgrep["container"]["options"] == "--user 0"
    assert "semgrep scan" in yaml.dump(semgrep)


def test_maintenance_runs_cross_platform_core_coverage_weekly() -> None:
    maintenance = workflow("maintenance.yml")
    compatibility = maintenance["jobs"]["cross-platform-core"]

    assert "schedule" in maintenance["on"]
    assert "workflow_dispatch" in maintenance["on"]
    assert set(compatibility["strategy"]["matrix"]["os"]) == {
        "macos-14",
        "windows-latest",
    }
    assert "3.12" in yaml.dump(compatibility)


def test_secret_scan_covers_the_tree_and_the_event_commit_range() -> None:
    """gitleaks is the history-aware half of secret scanning (semgrep's
    p/secrets pack reads the tree at HEAD only). It runs on every event
    without path gating, from a digest-pinned image, and both scans fail the
    job on a finding."""
    secrets = workflow("ci.yml")["jobs"]["secrets"]
    checkout = next(step for step in secrets["steps"] if "uses" in step)
    runs = "\n".join(step.get("run", "") for step in secrets["steps"])

    assert "if" not in secrets
    assert secrets["container"]["image"].startswith("ghcr.io/gitleaks/gitleaks@sha256:")
    assert secrets["container"]["options"] == "--user 0"
    assert checkout["with"]["fetch-depth"] == 0
    assert "gitleaks dir . --config .gitleaks.toml" in runs
    assert "gitleaks git . --config .gitleaks.toml --log-opts=" in runs
    assert runs.count("--exit-code 1") == 2


def test_maintenance_names_full_mypy_as_advisory_and_rechecks_security() -> None:
    jobs = workflow("maintenance.yml")["jobs"]

    assert jobs["full-mypy-advisory"]["name"] == "Advisory / full mypy debt"
    assert "continue-on-error: true" in yaml.dump(jobs["full-mypy-advisory"])
    assert "audit_dependencies.py" in yaml.dump(jobs["security"])
    assert all("timeout-minutes" in job for job in jobs.values())


def test_dependabot_covers_uv_actions_and_docker() -> None:
    dependabot = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    ecosystems = {update["package-ecosystem"] for update in dependabot["updates"]}

    assert ecosystems == {"uv", "github-actions", "docker"}
    assert all(
        update.get("cooldown", {}).get("default-days") == 7 for update in dependabot["updates"]
    )


def test_delivery_policy_files_have_explicit_owners() -> None:
    codeowners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    owners = "@thesanogoeffect"

    assert codeowners.endswith("\n")
    assert f"/.github/workflows/ {owners}" in codeowners
    assert f"/scripts/ci/ {owners}" in codeowners
    assert f"/ci/ {owners}" in codeowners


def test_container_delivery_scans_a_registry_digest_before_promotion() -> None:
    text = workflow_text("docker.yml")
    docker = workflow("docker.yml")

    assert "workflow_call" in docker["on"]
    assert "workflow_dispatch" in docker["on"]
    assert "quarantine-${{ steps.source.outputs.sha }}" in text
    assert "provenance: mode=max" in text
    assert "sbom: true" in text
    assert "push: true" in text
    assert "${{ steps.build.outputs.digest }}" in text
    assert 'exit-code: "1"' in text
    scan = next(step for step in docker["jobs"]["publish"]["steps"] if step.get("id") == "trivy")
    assert scan["with"]["severity"] == "HIGH,CRITICAL"
    assert scan["with"]["limit-severities-for-sarif"] == "true"
    assert "docker buildx imagetools create" in text
    assert text.index("steps.build.outputs.digest") < text.index("docker buildx imagetools create")


def test_container_delivery_builds_once_and_limits_movable_tags() -> None:
    text = workflow_text("docker.yml")

    assert text.count("docker/build-push-action@") == 1
    assert "CHANNEL" in text
    assert ":edge" in text
    assert ":latest" in text
    assert "type=sha" not in text
    assert "short-sha" not in text
    assert (
        "image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ steps.build.outputs.digest }}"
        in text
    )


def test_local_image_handoff_requires_the_same_successful_digest_scan() -> None:
    steps = workflow("docker.yml")["jobs"]["publish"]["steps"]
    export_index = next(
        i
        for i, step in enumerate(steps)
        if step["name"] == "Export verified image for local deployment"
    )
    barrier_index = next(
        i
        for i, step in enumerate(steps)
        if step["name"] == "Block promotion when the digest scan failed"
    )
    export = steps[export_index]
    assert barrier_index < export_index
    assert "always()" not in export["if"]
    assert "self-hosted" in export["if"]
    assert "vars.CI_DELIVERY_DIR != ''" in export["if"]
    assert export["env"]["DIGEST"] == "${{ steps.build.outputs.digest }}"
    run = export["run"]
    assert run.index('[[ "$revision" == "$SOURCE_SHA" ]]') < run.index("docker image save")
    assert run.index("sha256sum image.tar") < run.index('mv -T "$staging" "$destination"')


def test_container_summary_passes_step_outcome_through_the_environment() -> None:
    docker = workflow("docker.yml")
    summary = next(
        step
        for step in docker["jobs"]["publish"]["steps"]
        if step.get("name") == "Summarize verified artifact"
    )

    assert summary["env"]["SCAN_OUTCOME"] == "${{ steps.trivy.outcome }}"
    assert "${{ steps.trivy.outcome }}" not in summary["run"]
    assert "$SCAN_OUTCOME" in summary["run"]


def test_main_promotes_edge_only_after_required_ci_and_container_changes() -> None:
    publish = workflow("ci.yml")["jobs"]["publish-edge"]

    assert set(publish["needs"]) == {"changes", "required"}
    assert publish["uses"] == "./.github/workflows/docker.yml"
    assert publish["with"]["channel"] == "edge"
    assert "refs/heads/main" in publish["if"]
    assert "needs.changes.outputs.container == 'true'" in publish["if"]


def test_registry_opt_in_covers_all_entrypoints_without_disabling_validation() -> None:
    for filename, job_id in (
        ("ci.yml", "publish-edge"),
        ("release.yml", "publish-container"),
        ("docker.yml", "publish"),
    ):
        assert "vars.PUBLISH_GHCR == 'true'" in workflow(filename)["jobs"][job_id]["if"]

    ci = workflow("ci.yml")["jobs"]
    assert "PUBLISH_GHCR" not in yaml.dump(ci["container-build-scan"])
    assert "container-build-scan" in ci["required"]["needs"]


def test_release_proves_tag_version_and_remote_main_reachability() -> None:
    release = workflow("release.yml")
    validation = yaml.dump(release["jobs"]["validate"])

    assert "fetch-depth: 0" in validation
    assert "origin/main" in validation
    assert "git fetch" not in validation
    assert "merge-base --is-ancestor" in validation
    assert "pyproject.toml" in validation
    assert "uv lock --check" in validation


def test_release_repeats_linux_and_cross_platform_critical_tests() -> None:
    jobs = workflow("release.yml")["jobs"]
    linux = yaml.dump(jobs["linux-tests"])
    cross_platform = jobs["cross-platform-core"]

    assert "--extra all" in linux
    assert 'pytest -m "not slow"' in linux
    assert set(cross_platform["strategy"]["matrix"]["os"]) == {
        "macos-14",
        "windows-latest",
    }


@pytest.mark.parametrize(
    ("event", "ref", "on_main", "accepted"),
    [
        ("push", "refs/tags/v0.5.0", True, True),
        ("push", "refs/tags/v0.4.0", True, False),
        ("push", "refs/tags/v0.5.0", False, False),
        ("push", "refs/heads/main", True, False),
        ("workflow_dispatch", "refs/heads/main", True, True),
        ("workflow_dispatch", "refs/heads/feature", True, False),
        ("workflow_dispatch", "refs/tags/v0.5.0", True, False),
    ],
)
def test_release_source_validation_executes_against_git_history(
    tmp_path: Path, event: str, ref: str, on_main: bool, accepted: bool
) -> None:
    git_bin = shutil.which("git")
    bash_bin = shutil.which("bash")
    if not git_bin or not bash_bin:
        pytest.skip("release shell validation needs git and bash")

    def git(*args: str) -> str:
        return subprocess.check_output(  # noqa: S603 - fixed test commands
            [git_bin, *args],
            cwd=tmp_path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Release test")
    git("config", "user.email", "release@example.invalid")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "main")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    if not on_main:
        git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "unmerged")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.5.0"\n')
    output = tmp_path / "outputs"
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": event,
        "GITHUB_REF": ref,
        "GITHUB_REF_NAME": ref.rsplit("/", 1)[-1],
        "GITHUB_SHA": git("rev-parse", "HEAD"),
        "GITHUB_OUTPUT": str(output),
    }
    steps = workflow("release.yml")["jobs"]["validate"]["steps"]
    script = next(step["run"] for step in steps if step.get("id") == "version")
    result = subprocess.run(  # noqa: S603 - exercise the checked-in release validation
        [bash_bin, "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is accepted, result.stdout + result.stderr
    if accepted:
        assert output.read_text().strip() == "version=0.5.0"
    else:
        assert not output.exists()


def test_release_builds_dist_once_and_awaits_delivery_channels_before_github_release() -> None:
    jobs = workflow("release.yml")["jobs"]
    text = workflow_text("release.yml")

    assert text.count("uv build") == 1
    assert jobs["publish-pypi"]["environment"]["name"] == "pypi"
    assert jobs["publish-pypi"]["permissions"] == {"id-token": "write"}
    assert jobs["publish-container"]["uses"] == "./.github/workflows/docker.yml"
    assert jobs["publish-container"]["with"]["channel"] == "release"
    assert set(jobs["github-release"]["needs"]) >= {
        "publish-pypi",
        "publish-container",
    }
    assert "gh release" in yaml.dump(jobs["github-release"])


def test_release_rehearsals_never_publish_and_pypi_requires_explicit_opt_in() -> None:
    release = workflow("release.yml")
    assert "workflow_dispatch" in release["on"]
    for job_id in ("publish-pypi", "publish-container", "github-release"):
        condition = release["jobs"][job_id]["if"]
        assert "github.event_name == 'push'" in condition
        assert "startsWith(github.ref, 'refs/tags/v')" in condition
    assert "vars.PUBLISH_PYPI == 'true'" in release["jobs"]["publish-pypi"]["if"]


def test_release_uploads_only_verified_artifacts_with_no_duplicate_suppression() -> None:
    jobs = workflow("release.yml")["jobs"]
    build = jobs["build-dist"]
    publish = jobs["publish-pypi"]
    assert "twine check --strict dist/*" in yaml.dump(build)
    assert set(publish["needs"]) == {"validate", "build-dist"}
    assert len(publish["steps"]) == 2
    download, upload = publish["steps"]
    assert download["uses"].startswith("actions/download-artifact@")
    assert download["with"] == {"name": "distributions-${{ github.sha }}", "path": "dist/"}
    assert upload["uses"].startswith("pypa/gh-action-pypi-publish@")
    assert upload["with"] == {"attestations": "true"}


def test_github_release_requires_successful_pypi_when_enabled_and_explicit_repository() -> None:
    job = workflow("release.yml")["jobs"]["github-release"]
    assert "needs.publish-pypi.result == 'success'" in job["if"]
    assert "vars.PUBLISH_PYPI != 'true' && needs.publish-pypi.result == 'skipped'" in job["if"]
    assert "needs.publish-container.result == 'success'" in job["if"]
    assert "vars.PUBLISH_GHCR != 'true' && needs.publish-container.result == 'skipped'" in job["if"]
    release_step = next(step for step in job["steps"] if "gh release" in step.get("run", ""))
    assert release_step["env"]["GH_REPO"] == "${{ github.repository }}"


def test_release_publishing_workflow_does_not_use_dependency_caches() -> None:
    release = workflow("release.yml")

    for job_id, job in release["jobs"].items():
        assert job.get("secrets") != "inherit", job_id
        for step in job.get("steps", []):
            if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                assert step.get("with", {}).get("enable-cache") in (False, "false"), job_id


def test_github_release_commands_resolve_the_repository_without_a_checkout() -> None:
    release_steps = workflow("release.yml")["jobs"]["github-release"]["steps"]
    commands = [step for step in release_steps if "gh release" in step.get("run", "")]

    assert commands
    assert all(step["env"]["GH_REPO"] == "${{ github.repository }}" for step in commands)


def test_reusable_container_calls_do_not_inherit_caller_secrets() -> None:
    for workflow_name, job_id in (("ci.yml", "publish-edge"), ("release.yml", "publish-container")):
        job = workflow(workflow_name)["jobs"][job_id]
        assert "secrets" not in job


def test_pages_deployment_consumes_the_verified_ci_artifact() -> None:
    job = workflow("ci.yml")["jobs"]["deploy-production"]
    deployment = yaml.dump(job)

    assert set(job["needs"]) == {"required", "docs-build"}
    assert "needs.required.result == 'success'" in job["if"]
    assert "needs.docs-build.result == 'success'" in job["if"]
    assert "mkdocs-site-${{ github.sha }}" in deployment
    assert "mkdocs build" not in deployment
    assert job["environment"]["name"] == "github-pages"
    assert job["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    actions = [step.get("uses", "").split("@")[0] for step in job["steps"]]
    assert actions.index("actions/download-artifact") < actions.index(
        "actions/upload-pages-artifact"
    )
    assert actions.index("actions/upload-pages-artifact") < actions.index("actions/deploy-pages")


def test_pages_deployment_is_main_only_and_needs_no_cloudflare_secrets() -> None:
    jobs = workflow("ci.yml")["jobs"]
    production = jobs["deploy-production"]
    text = workflow_text("ci.yml")

    assert "github.event_name == 'push'" in production["if"]
    assert "github.ref == 'refs/heads/main'" in production["if"]
    assert "pull_request_target" not in text
    assert "deploy-preview" not in jobs
    assert "cloudflare/" not in text
    assert "CLOUDFLARE_" not in text
    assert "CF_ACCESS_" not in text
    assert not (WORKFLOWS / "docs.yml").exists()


def test_pages_deployment_verifies_public_access_and_exact_revision() -> None:
    job = workflow("ci.yml")["jobs"]["deploy-production"]
    smoke = next(step for step in job["steps"] if "smoke_site.py" in step.get("run", ""))

    assert "--expected-sha=${{ github.sha }}" in smoke["run"]
    assert "--access=public" in smoke["run"]
    assert smoke["env"] == {"SITE_URL": "https://bibr.org/"}
    assert job["environment"]["url"] == smoke["env"]["SITE_URL"]
