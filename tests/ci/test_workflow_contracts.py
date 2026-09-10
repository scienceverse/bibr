from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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
    assert ".well-known/bibr-build" in docs
    assert "mkdocs-site-${{ github.sha }}" in docs
    assert "retention-days: 7" in docs
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["include-hidden-files"] in (True, "true")


def test_persistent_container_runner_is_only_selected_for_trusted_pushes() -> None:
    ci = workflow("ci.yml")

    assert ci["on"]["push"]["branches"] == ["main", "dev"]
    assert ci["jobs"]["container-build-scan"]["runs-on"] == (
        "${{ github.event_name == 'push' && vars.CONTAINER_RUNNER || 'ubuntu-latest' }}"
    )


def test_container_builds_stamp_the_exact_source_commit() -> None:
    for filename, job_id, expected_sha in (
        ("ci.yml", "container-build-scan", "${{ github.sha }}"),
        ("docker.yml", "publish", "${{ steps.source.outputs.sha }}"),
    ):
        build = next(
            step
            for step in workflow(filename)["jobs"][job_id]["steps"]
            if step.get("uses", "").startswith("docker/build-push-action@")
        )
        assert f"BIBR_BUILD_SHA={expected_sha}" in build["with"]["build-args"].splitlines()


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
    owners = "@Lakens @DeBruine @thesanogoeffect"

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


def test_release_proves_tag_version_and_remote_main_reachability() -> None:
    release = workflow("release.yml")
    validation = yaml.dump(release["jobs"]["validate"])

    assert "fetch-depth: 0" in validation
    assert "origin main" in validation
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


def test_release_builds_dist_once_and_publishes_both_channels_before_github_release() -> None:
    jobs = workflow("release.yml")["jobs"]
    text = workflow_text("release.yml")

    assert text.count("uv build") == 1
    assert jobs["publish-pypi"]["environment"] == "pypi"
    assert jobs["publish-pypi"]["permissions"] == {"id-token": "write"}
    assert jobs["publish-container"]["uses"] == "./.github/workflows/docker.yml"
    assert jobs["publish-container"]["with"]["channel"] == "release"
    assert set(jobs["github-release"]["needs"]) >= {
        "publish-pypi",
        "publish-container",
    }
    assert "gh release" in yaml.dump(jobs["github-release"])


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


def test_site_deployments_consume_the_ci_artifact_without_rebuilding() -> None:
    jobs = workflow("ci.yml")["jobs"]
    preview = yaml.dump(jobs["deploy-preview"])
    production = yaml.dump(jobs["deploy-production"])

    assert jobs["deploy-preview"]["environment"]["name"] == "bibr-site-preview"
    assert jobs["deploy-production"]["environment"]["name"] == "bibr-site-production"
    assert set(jobs["deploy-preview"]["needs"]) == {"required", "docs-build"}
    assert set(jobs["deploy-production"]["needs"]) == {"required", "docs-build"}
    for deployment in (preview, production):
        assert "mkdocs-site-${{ github.sha }}" in deployment
        assert "wranglerVersion: 4.110.0" in deployment
        assert "mkdocs build" not in deployment


def test_site_preview_is_same_repo_only_and_production_is_main_only() -> None:
    jobs = workflow("ci.yml")["jobs"]
    preview = jobs["deploy-preview"]
    production = jobs["deploy-production"]
    text = workflow_text("ci.yml")
    preview_command = next(step for step in preview["steps"] if step.get("id") == "deploy")["with"][
        "command"
    ]
    production_command = next(step for step in production["steps"] if step.get("id") == "deploy")[
        "with"
    ]["command"]

    assert "github.event.pull_request.head.repo.full_name == github.repository" in preview["if"]
    assert "--branch=pr-${{ github.event.pull_request.number }}" in preview_command
    assert "github.event_name == 'push'" in production["if"]
    assert "github.ref == 'refs/heads/main'" in production["if"]
    assert "--branch=main" in production_command
    assert "pull_request_target" not in text
    assert not (WORKFLOWS / "docs.yml").exists()


def test_site_deployments_verify_their_access_policy_and_exact_revision() -> None:
    jobs = workflow("ci.yml")["jobs"]

    for job_id in ("deploy-preview", "deploy-production"):
        job = jobs[job_id]
        smoke = next(step for step in job["steps"] if "smoke_site.py" in step.get("run", ""))
        assert "--expected-sha=${{ github.sha }}" in smoke["run"]
        if job_id == "deploy-preview":
            assert "--access=protected" in smoke["run"]
            assert "CF_ACCESS_CLIENT_ID" in smoke["env"]
            assert "CF_ACCESS_CLIENT_SECRET" in smoke["env"]
            assert smoke["env"]["SITE_URL"] == "${{ steps.deploy.outputs.deployment-url }}"
        else:
            assert "--access=public" in smoke["run"]
            assert smoke["env"] == {"SITE_URL": "https://bibr.org"}
