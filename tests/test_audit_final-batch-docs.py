"""Final review: the ``chew --dry-run`` footer does not promise an exit code
the real run does not give, and the guides describe which LLM retries count
toward the circuit breaker and how far the Redis LLM rate limit is shared."""

from __future__ import annotations

from pathlib import Path

import pytest

from bibr.utils.circuit_breaker import AsyncCircuitBreaker, CircuitState

REPO_ROOT = Path(__file__).parents[1]


def _guide(name: str) -> str:
    return " ".join((REPO_ROOT / "docs" / "guides" / name).read_text("utf-8").split())


class _ForbiddenPipeline:
    def __init__(self, *args, **kwargs):
        raise AssertionError("pipeline must not be constructed on this path")


# --- chew --dry-run: the footer under the blockers ---------------------------------


async def test_dry_run_footer_matches_the_real_run_on_an_unwritable_output(
    tmp_path, monkeypatch, capsys
):
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _ForbiddenPipeline)
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("a.xml", "b.xml"):
        (papers / name).write_text("<article/>", encoding="utf-8")
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    argv = ["chew", str(papers), "-o", str(blocked), "--no-llm"]

    with pytest.raises(SystemExit) as dry:
        await _run_process(_build_parser().parse_args([*argv, "--dry-run"]))
    preview = " ".join(capsys.readouterr().out.split())
    with pytest.raises(SystemExit) as real:
        await _run_process(_build_parser().parse_args(argv))

    assert dry.value.code == 1
    assert real.value.code == 2
    assert f"{blocked} is not a directory" in preview
    assert "The real run fails on these; fix them before processing." in preview
    assert "exits 1" not in preview  # the real run exits 2 here


# --- configuration.md: LLM retries and the circuit breaker -------------------------


class _StatusError(Exception):
    """A provider SDK error carrying its HTTP status, as openai/anthropic do."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status_code = status


@pytest.mark.parametrize(
    ("status", "counts"), [(408, True), (409, False), (429, False), (500, True)]
)
async def test_retried_llm_statuses_count_toward_the_breaker_as_documented(status, counts):
    breaker = AsyncCircuitBreaker(failure_threshold=1, failure_dedup_window=0)

    with pytest.raises(_StatusError):
        async with breaker:
            raise _StatusError(status)

    assert (breaker.state is CircuitState.OPEN) is counts


def test_configuration_guide_says_429_and_409_do_not_trip_the_breaker():
    text = _guide("configuration.md")

    assert "every retry waits for `LLM_RATE_LIMIT_RPM` and counts toward" not in text
    assert (
        "An attempt that fails with a server error, a 408, a timeout or a dropped connection "
        "counts toward the circuit breaker; a 429 or 409, like any other 4xx answer, does not"
    ) in text


# --- deployment.md: the Redis LLM rate limit is per provider and model -------------


def test_deployment_guide_scopes_the_shared_llm_rate_limit():
    text = _guide("deployment.md")

    assert "(and one LLM rate limit)" not in text
    assert "The LLM rate limit is shared per provider and model" in text
    assert "a process calling another provider or model has its own" in text
