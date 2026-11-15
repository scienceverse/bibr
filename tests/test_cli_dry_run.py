"""Tests for ``bibr chew --dry-run``.

``--dry-run`` resolves and prints the full run plan, then exits 0 with no
network calls and no model loads. The guard tests below patch every heavy
entry point a real run would touch (``LocalPipeline`` construction, httpx
client construction, and huggingface_hub download calls) to raise, proving
dry-run never reaches them — the collision-guard reuse test proves it still
composes correctly with Task 3's guards (they run *before* the dry-run
branch, so a colliding batch still hard-errors instead of previewing).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_ocr_runtime_blocker(monkeypatch):
    """Keep the plan tests independent of the host's OCR runtimes.

    The dry-run blocker list includes the OCR runtime preflight, which fails
    where no automatic runtime can start (the Windows chain is llama.cpp
    only, and CI has no llama-server). No test here is about that blocker.
    """
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4\n%dry-run fixture\n"


class _ForbiddenPipeline:
    """Stand-in for ``LocalPipeline`` that blows up if ever constructed."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("dry-run must not construct LocalPipeline")


def _neutralize_local_rapid_mlx_env(monkeypatch) -> None:
    """Keep cache-rendering tests independent of a developer's local .env."""
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "backend", "cloud")
    monkeypatch.setattr(Settings.rapid_mlx, "hf_home", None)
    monkeypatch.setattr(Settings.rapid_mlx, "hf_hub_cache", None)


# --- plan contents ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "display"),
    [
        ("sat-6l-sm", "segment-any-text/sat-6l-sm"),
        ("scienceverse/bibr-sat-science-en", "scienceverse/bibr-sat-science-en"),
    ],
)
def test_display_wtpsplit_repo_id_resolves_short_and_full_ids(value, display):
    from bibr.local.cli.dry_run import display_wtpsplit_repo_id

    assert display_wtpsplit_repo_id(value) == display


def test_display_wtpsplit_repo_id_labels_local_bundle(tmp_path):
    from bibr.local.cli.dry_run import _hf_cache_status, display_wtpsplit_repo_id

    display = display_wtpsplit_repo_id(str(tmp_path))
    assert display == f"local bundle: {tmp_path}"
    assert _hf_cache_status(display) == "available locally"


async def test_dry_run_prints_all_plan_sections_and_exits_cleanly(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run"])
    await _run_process(args)  # must return normally (exit 0) — not sys.exit

    out = capsys.readouterr().out
    assert "Input (1)" in out
    assert str(pdf) in out
    assert re.search(r"^  ocr\s+\S", out, re.MULTILINE)
    assert re.search(r"^  llm\s+\S", out, re.MULTILINE)
    assert re.search(r"^  refs\s+seg \S+ · parse \S+", out, re.MULTILINE)
    assert re.search(r"^  memory\s+\S", out, re.MULTILINE)
    assert "Models" in out
    assert "Output" in out
    assert "Dry run — no files were processed." in out


async def test_dry_run_paddle_automatic_chain_lists_ordered_exact_identities(
    tmp_path, capsys, monkeypatch
):
    """The preview must reveal automatic candidates without starting any runtime."""
    import platform
    import sys

    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--ocr", "paddle"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "OCR backend: paddle (automatic)" in out
    assert "1. paddle-rapid-mlx | olragon/PaddleOCR-VL-1.6-8bit | paddle" in out
    assert "2. paddle-mlx-vlm | olragon/PaddleOCR-VL-1.6-8bit | paddle" in out
    assert "3. glm-rapid-mlx | mlx-community/GLM-OCR-8bit | glm" in out
    assert "4. glm-llama | ggml-org/GLM-OCR-GGUF:Q8_0 | glm" in out


async def test_dry_run_url_only_uses_paddle_served_model_and_profile(tmp_path, capsys, monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr(Settings.ocr, "paddle_served_model", "paddle-ocr-vl-1.6")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(
        ["chew", str(pdf), "--dry-run", "--ocr-url", "http://ocr.example"]
    )
    await _run_process(args)

    out = capsys.readouterr().out
    assert "OCR backend: paddle-http" in out
    assert "OCR model: paddle-ocr-vl-1.6" in out
    assert "OCR profile: paddle" in out


async def test_dry_run_lists_first_five_and_more_marker(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    paths = []
    for i in range(7):
        p = tmp_path / f"paper{i}.pdf"
        p.write_bytes(_pdf_bytes())
        paths.append(p)

    args = _build_parser().parse_args(["chew", *(str(p) for p in paths), "--dry-run"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "Input (7)" in out
    for p in paths[:5]:
        assert str(p) in out
    assert "… and 2 more" in out


async def test_dry_run_no_llm_shows_disabled_marker(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--no-llm"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert re.search(r"^  llm\s+disabled \(--no-llm\)", out, re.MULTILINE)
    assert re.search(r"^  refs\s+disabled \(--no-llm\)", out, re.MULTILINE)
    # no_llm forces crossref off inside LocalPipeline.__init__ regardless of
    # --no-crossref's own value — the dry-run preview must show *that* as
    # the reason, not silently report Crossref as enabled.
    assert re.search(r"^  crossref\s+disabled \(--no-llm\)", out, re.MULTILINE)
    # --no-llm drops reference extraction entirely for OCR'd input, and the
    # trained section classifier (lookup-only path) — neither should be
    # listed as a model this run would need to download.
    assert "ref segmenter" not in out
    assert "ref parser" not in out
    assert "section classifier" not in out


async def test_dry_run_default_reports_enrichment_off_with_opt_in_hint(tmp_path, capsys):
    """CROSSREF_ENRICH is off by default: the preview says so and how to turn it on."""
    from bibr.config import Settings
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    # Assign first: pydantic records every attribute assignment in
    # ``model_fields_set``, and the "off by default" wording keys on the
    # variable never having been set.
    Settings.crossref.enrich = False
    Settings.crossref.model_fields_set.discard("enrich")

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert re.search(
        r"^  crossref\s+disabled \(off by default; enable with --crossref or CROSSREF_ENRICH=true\)",
        out,
        re.MULTILINE,
    )


async def test_dry_run_crossref_flag_enables_enrichment(tmp_path, capsys, monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    monkeypatch.setattr(Settings.crossref, "enrich", False)

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--crossref"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert re.search(r"^  crossref\s+enabled \(--crossref\)", out, re.MULTILINE)


async def test_dry_run_no_crossref_overrides_true_setting(tmp_path, capsys, monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    monkeypatch.setattr(Settings.crossref, "enrich", True)

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--no-crossref"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert re.search(r"^  crossref\s+disabled \(--no-crossref\)", out, re.MULTILINE)


async def test_dry_run_setting_true_reports_setting_as_reason(tmp_path, capsys, monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    monkeypatch.setattr(Settings.crossref, "enrich", True)

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert re.search(r"^  crossref\s+enabled \(CROSSREF_ENRICH=true\)", out, re.MULTILINE)


async def test_dry_run_refs_off_shows_off_strategies(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(
        ["chew", str(pdf), "--dry-run", "--refs", "off", "--crossref"]
    )
    await _run_process(args)

    out = capsys.readouterr().out
    assert re.search(r"^  refs\s+seg \S+ · parse off", out, re.MULTILINE)
    # refs=off leaves nothing to enrich, even with --crossref.
    assert re.search(r"^  crossref\s+disabled \(refs=off\)", out, re.MULTILINE)


# --- output destinations (no filesystem writes) -------------------------------


async def test_dry_run_output_destination_single_file(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    out_file = tmp_path / "nested" / "result.json"

    args = _build_parser().parse_args(["chew", str(pdf), "-o", str(out_file), "--dry-run"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert f"paper.pdf -> {out_file}" in out
    assert not out_file.exists()
    assert not out_file.parent.exists()  # dry-run must not create directories


async def test_dry_run_output_destination_batch(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"
    file_a.write_bytes(_pdf_bytes())
    file_b.write_bytes(_pdf_bytes())
    out_dir = tmp_path / "out"

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(out_dir), "--dry-run"]
    )
    await _run_process(args)

    out = capsys.readouterr().out
    assert f"a.pdf -> {out_dir / 'a.json'}" in out
    assert f"b.pdf -> {out_dir / 'b.json'}" in out
    assert not out_dir.exists()  # dry-run must not create the output directory


async def test_dry_run_output_destination_stdout_when_no_o(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "stdout" in out


async def test_dry_run_output_destination_batch_stdout_when_no_o(tmp_path, capsys):
    """Batch input with no -o writes each result as its own stdout JSON line
    in the real run — the preview must list per-file stdout, not a single
    generic line."""
    from bibr.local.cli import _build_parser, _run_process

    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"
    file_a.write_bytes(_pdf_bytes())
    file_b.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(file_a), str(file_b), "--dry-run"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "a.pdf -> stdout" in out
    assert "b.pdf -> stdout" in out


async def test_manifest_dry_run_lists_authoritative_per_row_destinations(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    source = tmp_path / "paper.pdf"
    source.write_bytes(_pdf_bytes())
    destinations = [tmp_path / "first" / "one.json", tmp_path / "second" / "two.json"]
    manifest = tmp_path / "queue.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "input_path": "paper.pdf",
                    "output_path": str(destination.relative_to(tmp_path)),
                    "queue_record_id": f"record-{index}",
                }
            )
            + "\n"
            for index, destination in enumerate(destinations, start=1)
        ),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        ["chew", "--manifest", str(manifest), "--dry-run", "--no-llm"]
    )

    await _run_process(args)

    out = capsys.readouterr().out
    assert f"paper.pdf -> {destinations[0]}" in out
    assert f"paper.pdf -> {destinations[1]}" in out
    assert "stdout" not in out
    assert not destinations[0].parent.exists()
    assert not destinations[1].parent.exists()


# --- composition with Task 3's guards -----------------------------------------


async def test_dry_run_still_hard_errors_on_stem_collision(tmp_path, capsys):
    """The stem-collision guard fires before the dry-run branch, so a
    colliding batch exits 2 with the collision error even under --dry-run —
    that's the reuse: no separate collision detection inside the printer."""
    from bibr.local.cli import _build_parser, _run_process

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "x.pdf"
    file_b = dir_b / "x.pdf"
    file_a.write_bytes(_pdf_bytes())
    file_b.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(tmp_path / "out"), "--dry-run"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert str(file_a) in err
    assert str(file_b) in err


async def test_dry_run_still_hard_errors_on_paper_id_batch(tmp_path, capsys):
    from bibr.local.cli import _build_parser, _run_process

    (tmp_path / "one.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "two.pdf").write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(tmp_path), "--paper-id", "my-id", "--dry-run"])

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "--paper-id" in err


# --- side-effect-free proof ----------------------------------------------------


async def test_dry_run_never_touches_pipeline_http_or_hf_downloads(tmp_path, monkeypatch):
    """Patch every heavy entry point a real run would touch to raise, then
    confirm --dry-run still completes cleanly — proving none of them fired."""
    import httpx

    from bibr.local.cli import _build_parser, _run_process

    def _forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not touch this")

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _ForbiddenPipeline)
    monkeypatch.setattr(httpx.Client, "__init__", _forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", _forbidden)

    hf_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hf_hub, "hf_hub_download", _forbidden)
    monkeypatch.setattr(hf_hub, "snapshot_download", _forbidden)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run"])
    # Would raise AssertionError if LocalPipeline/httpx/hf-hub-download fired.
    await _run_process(args)


async def test_dry_run_never_touches_pipeline_for_cloud_llm_backend(tmp_path, monkeypatch):
    """Cloud LLM credential preflight (``preflight_credentials``) constructs a
    real provider client and must not run under --dry-run either — otherwise
    a paper with no configured API key would fail before the preview prints."""
    from bibr.local.cli import _build_parser, _run_process

    def _forbidden():
        raise AssertionError("dry-run must not preflight LLM credentials")

    monkeypatch.setattr("bibr.clients.llm.preflight_credentials", _forbidden)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _ForbiddenPipeline)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run"])
    await _run_process(args)


# --- HF cache rendering --------------------------------------------------------


async def test_dry_run_hf_cache_cached_vs_will_download(tmp_path, capsys, monkeypatch):
    """A repo present in the (mocked) local HF cache renders 'cached'; one
    that resolves a known approximate size and is absent renders
    'will download (~size)'."""
    hf_hub = pytest.importorskip("huggingface_hub")
    from bibr.local.cli import _build_parser, _run_process

    _neutralize_local_rapid_mlx_env(monkeypatch)

    class _FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    class _FakeCacheInfo:
        repos = [_FakeRepo("segment-any-text/sat-6l-sm")]

    monkeypatch.setattr(hf_hub, "scan_cache_dir", lambda: _FakeCacheInfo())

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    # Pin the OCR backend so this cache-rendering assertion is independent of
    # platform defaults or settings mutated by earlier tests.
    # Neutralize the dry-run OCR-runtime blocker: --dry-run now surfaces it
    # (and exits 1), but this test asserts cache rendering, not readiness.
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)
    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--ocr", "glm-llama"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "segment-any-text/sat-6l-sm — cached" in out
    assert "will download (~" in out


async def test_dry_run_uncached_repo_without_known_size_still_signals_download(
    tmp_path, capsys, monkeypatch
):
    """A repo that's absent from the local cache but has no documented
    approximate size (e.g. the layout detector) must still say 'will
    download' — a bare 'size unknown' drops the fact a download WILL
    happen."""
    hf_hub = pytest.importorskip("huggingface_hub")
    from bibr.local.cli import _build_parser, _run_process

    _neutralize_local_rapid_mlx_env(monkeypatch)

    class _FakeCacheInfo:
        repos = []

    monkeypatch.setattr(hf_hub, "scan_cache_dir", lambda: _FakeCacheInfo())

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    # Neutralize the dry-run OCR-runtime blocker (see above): this test
    # asserts cache rendering, not readiness.
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)
    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--ocr", "glm-llama"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "PaddlePaddle/PP-DocLayoutV3_safetensors — will download (size unknown)" in out


async def test_dry_run_ocr_weight_repo_has_approximate_size(tmp_path, capsys, monkeypatch):
    """A managed OCR backend's weight repo has a documented approximate
    download size and must render it instead of 'size unknown' — no OCR weight
    repo used to be in the dry-run size table at all."""
    hf_hub = pytest.importorskip("huggingface_hub")
    from bibr.local.cli import _build_parser, _run_process

    _neutralize_local_rapid_mlx_env(monkeypatch)

    class _FakeCacheInfo:
        repos = []

    monkeypatch.setattr(hf_hub, "scan_cache_dir", lambda: _FakeCacheInfo())

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    # Pin the backend explicitly — the platform default (``default_backend()``
    # in bibr/ocr/registry.py) picks a different runtime (e.g. glm-mlx on
    # Apple Silicon), which would make this test host-dependent.
    # Neutralize the dry-run OCR-runtime blocker (see above): this test
    # asserts cache rendering, not readiness.
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)
    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--ocr", "glm-llama"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "OCR weights: ggml-org/GLM-OCR-GGUF:Q8_0 — will download (~1-2 GB)" in out


def test_pinned_llama_cpp_build_cache_status_reads_its_commit(monkeypatch):
    """A registry build pinned to a commit is cached only at that commit."""
    from bibr.local import hub_pins
    from bibr.local.cli.dry_run import _hf_cache_status

    _neutralize_local_rapid_mlx_env(monkeypatch)
    model = "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"
    # tests/conftest.py stubs the cache read to "not cached".
    assert _hf_cache_status(model) == "will download revision 0aa228e8 (~4.3 GB)"

    seen = []

    def cached(repo_id, filename, revision, cache_dir):
        seen.append((repo_id, filename, revision, cache_dir))
        return f"/hf/{filename}"

    monkeypatch.setattr(hub_pins, "_cached_file", cached)
    assert _hf_cache_status(model) == "cached (revision 0aa228e8)"
    assert seen == [
        (
            "scienceverse/bibr-extract-qwen35-4b-GGUF",
            "bibr-extract-qwen35-4b-Q8_0.gguf",
            "0aa228e8c908512ad5e5e13fa55e7e7d3180c7d3",
            None,
        )
    ]


def test_pinned_vllm_build_cache_status_needs_weights_at_its_commit(monkeypatch, tmp_path):
    import huggingface_hub.constants

    from bibr.local.cli.dry_run import _hf_cache_status
    from bibr.local.llm_models import get_model

    _neutralize_local_rapid_mlx_env(monkeypatch)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    model = "scienceverse/bibr-extract-qwen35-4b"
    size = get_model("bibr-extract-4b").variants[0].approx_download_gb
    will_download = f"will download revision 9199424d (~{size:.1f} GB)"
    snapshots = tmp_path / "models--scienceverse--bibr-extract-qwen35-4b" / "snapshots"
    assert _hf_cache_status(model) == will_download

    # The previous revision's weights (still on main) do not count.
    old = snapshots / "6ab47a4e4e8b1b76c1ab234aec66f360683ff9fc"
    old.mkdir(parents=True)
    (old / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert _hf_cache_status(model) == will_download

    pinned = snapshots / "9199424d4bd200feb247e26e4a860b63da73cbaf"
    pinned.mkdir()
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (pinned / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}})
    )
    # One shard of two is not the model.
    (pinned / shards[0]).write_bytes(b"x")
    assert _hf_cache_status(model) == will_download
    # A snapshot link whose blob was deleted is not a cached file.
    try:
        (pinned / shards[1]).symlink_to(tmp_path / "gone")
    except OSError:  # Windows without the symlink privilege
        pass
    else:
        assert _hf_cache_status(model) == will_download
        (pinned / shards[1]).unlink()

    (pinned / shards[1]).write_bytes(b"x")
    assert _hf_cache_status(model) == "cached (revision 9199424d)"


@pytest.mark.parametrize(
    ("llm_backend", "hf_hub_cache", "hf_home", "expected"),
    [
        ("vllm-mlx", "/rapid/hub-cache", None, None),
        ("vllm-mlx", None, "/rapid/home", None),
        (None, "/rapid/hub-cache", None, None),
        ("rapid-mlx", "/rapid/hub-cache", "/rapid/home", "/rapid/hub-cache"),
        ("rapid-mlx", None, "/rapid/home", "hf_home/hub"),
        ("rapid-mlx", None, None, None),
    ],
)
def test_pinned_mlx_build_cache_status_reads_the_serving_backends_cache(
    monkeypatch, llm_backend, hf_hub_cache, hf_home, expected
):
    """bibr downloads a pinned MLX build into Rapid-MLX's own cache only for
    Rapid-MLX; vllm-mlx reads the default Hugging Face cache."""
    from pathlib import Path

    from bibr.config import Settings
    from bibr.local import hub_pins
    from bibr.local.cli.dry_run import _hf_cache_status

    _neutralize_local_rapid_mlx_env(monkeypatch)
    monkeypatch.setattr(Settings.rapid_mlx, "hf_hub_cache", hf_hub_cache)
    monkeypatch.setattr(Settings.rapid_mlx, "hf_home", hf_home)
    if expected == "hf_home/hub":
        expected = str(Path("/rapid/home") / "hub")
    seen = set()

    def cached(repo_id, filename, revision, cache_dir):
        seen.add(cache_dir)
        return f"/hf/{filename}"

    monkeypatch.setattr(hub_pins, "_cached_file", cached)

    status = _hf_cache_status(
        "scienceverse/bibr-extract-qwen35-4b-mlx-8bit", llm_backend=llm_backend
    )

    assert status == "cached (revision 432b8cd8)"
    assert seen == {expected}


def test_pinned_gguf_build_cache_status_never_reads_rapid_mlxs_cache(monkeypatch):
    from bibr.config import Settings
    from bibr.local import hub_pins
    from bibr.local.cli.dry_run import _hf_cache_status

    _neutralize_local_rapid_mlx_env(monkeypatch)
    monkeypatch.setattr(Settings.rapid_mlx, "hf_hub_cache", "/rapid/hub-cache")
    seen = set()

    def cached(repo_id, filename, revision, cache_dir):
        seen.add(cache_dir)
        return f"/hf/{filename}"

    monkeypatch.setattr(hub_pins, "_cached_file", cached)

    status = _hf_cache_status(
        "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0", llm_backend="rapid-mlx"
    )

    assert status == "cached (revision 0aa228e8)"
    assert seen == {None}


@pytest.mark.parametrize("backend", ["vllm-mlx", "rapid-mlx"])
async def test_dry_run_checks_the_llm_build_for_its_backend(tmp_path, monkeypatch, backend):
    from bibr.config import Settings
    from bibr.local.cli import _build_parser, _run_process, dry_run

    _neutralize_local_rapid_mlx_env(monkeypatch)
    model = "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"
    monkeypatch.setattr(Settings.llm, "local_model", model)
    calls = []

    def status(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "cached"

    monkeypatch.setattr(dry_run, "_hf_cache_status", status)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    args = _build_parser().parse_args(
        ["chew", str(pdf), "--dry-run", "--ocr", "glm-llama", "--llm", backend]
    )

    try:
        await _run_process(args)
    except SystemExit:  # a blocker (no MLX runtime here) still prints the plan
        pass

    assert (model, {"llm_backend": backend}) in calls


async def test_dry_run_hf_cache_unknown_without_huggingface_hub(tmp_path, capsys, monkeypatch):
    """When huggingface_hub can't be imported, report cache-unknown instead
    of guessing or crashing."""
    import builtins

    from bibr.local.cli import _build_parser, _run_process

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("simulated: ml extra not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "cache state unknown (ml extra not installed)" in out


# --- run mode and the first-run download preview --------------------------------------

_MODE_ENV = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "LANGEXTRACT_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BACKEND",
    "PIPELINE_MODE",
    "OCR_BACKEND",
)


@pytest.fixture
def fresh_machine(tmp_path, monkeypatch):
    """A core install with no LLM configured and no Hugging Face cache directory at all."""
    import huggingface_hub.constants as hf_constants
    from huggingface_hub.utils import _cache_manager as cache_manager

    from bibr.config import GlobalSettings

    for var in _MODE_ENV:
        monkeypatch.delenv(var, raising=False)
    # The suite pins ML_RUNTIME=torch (tests/conftest.py); a core install runs on ONNX.
    monkeypatch.setenv("ML_RUNTIME", "auto")
    for var in (  # blanked by the suite so no test downloads a classifier
        "ML_SECTION_CLASSIFIER_MODEL_ID",
        "ML_PAPER_CLASSIFIER_MODEL_ID",
        "ML_FRONT_ROLE_MODEL_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("bibr.config.Settings", GlobalSettings())
    cache = tmp_path / "no-such-hub-cache"
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    # ``scan_cache_dir`` bound the default at import.
    monkeypatch.setattr(cache_manager, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return cache


def _plan(argv, tmp_path, capsys):
    """The ``--dry-run`` output for one PDF and *argv*."""
    import asyncio

    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", *argv])
    try:
        asyncio.run(_run_process(args))
    except SystemExit:  # a blocker still prints the plan
        pass
    return capsys.readouterr().out


def test_a_fresh_machine_with_no_hub_cache_says_every_model_will_download(fresh_machine):
    """``scan_cache_dir`` raises when the cache directory does not exist; that is an empty cache."""
    from bibr.local.cli.dry_run import _hf_cache_status

    assert not fresh_machine.exists()
    status = _hf_cache_status("scienceverse/bibr-layout-onnx")
    assert status.startswith("will download")
    assert "0.13 GB" in status


def test_a_hub_cache_that_exists_but_cannot_be_scanned_stays_unknown(fresh_machine, monkeypatch):
    import huggingface_hub as hf_hub

    from bibr.local.cli.dry_run import _hf_cache_status

    fresh_machine.mkdir()

    def broken():
        raise OSError("permission denied")

    monkeypatch.setattr(hf_hub, "scan_cache_dir", broken)

    assert _hf_cache_status("scienceverse/bibr-layout-onnx").startswith("cache state unknown")


def test_dry_run_on_a_fresh_machine_names_the_models_and_the_total(fresh_machine, tmp_path, capsys):
    out = _plan(["--ocr", "glm-llama"], tmp_path, capsys)

    assert "↓ layout (PP-DocLayout): scienceverse/bibr-layout-onnx — will download" in out
    assert "↓ sentence segmenter" in out
    assert "cache state unknown" not in out
    assert re.search(r"Would download \d+ models? \(about [\d.]+ (GB|MB)", out)
    assert "later runs use the cache" in out


def test_dry_run_with_a_warm_cache_has_no_download_total(
    fresh_machine, tmp_path, capsys, monkeypatch
):
    from bibr.local.cli import dry_run

    monkeypatch.setattr(dry_run, "_hf_cache_status", lambda *a, **k: "cached")
    out = _plan(["--ocr", "glm-llama"], tmp_path, capsys)

    assert "Would download" not in out


def test_dry_run_without_a_key_plans_a_fast_run(fresh_machine, tmp_path, capsys):
    out = _plan(["--ocr", "glm-llama"], tmp_path, capsys)

    assert re.search(r"^  mode\s+fast: no LLM calls, skips ", out, re.MULTILINE)
    assert re.search(r"^  llm\s+disabled \(fast mode\)", out, re.MULTILINE)
    assert "API key" not in out  # no credential blocker for a run that calls no LLM
    assert "Blockers" not in out


def test_dry_run_fast_lists_the_front_matter_tagger(fresh_machine, tmp_path, capsys):
    out = _plan(["--ocr", "glm-llama"], tmp_path, capsys)

    assert "front-matter tagger: scienceverse/bibr-front-tagger — will download" in out


def test_dry_run_fast_swaps_the_llm_reference_strategies_for_local_ones(
    fresh_machine, tmp_path, capsys
):
    fresh_machine  # noqa: B018 - the fixture sets up the keyless machine
    out = _plan(["--ocr", "glm-llama", "--fast"], tmp_path, capsys)

    assert not re.search(r"^  refs\s+seg llm", out, re.MULTILINE)
    assert not re.search(r"parse (llm|llm-chunked|ref-llm)\b", out)


def test_dry_run_full_names_the_llm_and_does_not_list_the_tagger(
    fresh_machine, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from bibr.config import GlobalSettings

    monkeypatch.setattr("bibr.config.Settings", GlobalSettings())
    out = _plan(["--ocr", "glm-llama"], tmp_path, capsys)

    assert re.search(r"^  mode\s+full: the configured LLM reads", out, re.MULTILINE)
    assert re.search(r"^  llm\s+\S+/\S+", out, re.MULTILINE)
    assert "front-matter tagger" not in out


def test_dry_run_full_without_a_key_still_blocks_on_the_missing_key(
    fresh_machine, tmp_path, capsys
):
    out = _plan(["--ocr", "glm-llama", "--full"], tmp_path, capsys)

    assert "Blockers" in out
    assert "API key" in out


def test_dry_run_no_llm_is_structure_only(fresh_machine, tmp_path, capsys):
    out = _plan(["--ocr", "glm-llama", "--no-llm"], tmp_path, capsys)

    assert re.search(r"^  mode\s+structure only \(--no-llm\)", out, re.MULTILINE)
    assert re.search(r"^  llm\s+disabled \(--no-llm\)", out, re.MULTILINE)
    assert "front-matter tagger" not in out
    assert "section classifier" not in out


def test_dry_run_lists_the_classifiers_the_run_uses(fresh_machine, tmp_path, capsys):
    out = _plan(["--ocr", "glm-llama"], tmp_path, capsys)

    for label in ("section classifier", "paper classifier", "front-role classifier"):
        assert label in out


# --- the same preview announced before a real run ---------------------------------------


def _config(*argv):
    from bibr.local.cli import _build_parser, resolve_run_config

    return resolve_run_config(_build_parser().parse_args(["chew", "paper.pdf", *argv]))


def test_pending_downloads_lists_only_what_the_cache_lacks(fresh_machine, monkeypatch):
    from bibr.local.cli import dry_run

    def status(repo_id, **_kwargs):
        return "cached" if "segment-any-text" in repo_id else "will download (~90 MB)"

    monkeypatch.setattr(dry_run, "_hf_cache_status", status)
    pending = dry_run.pending_downloads(_config("--ocr", "glm-llama"), [Path("paper.pdf")])

    assert pending
    assert all(row[2].startswith("will download") for row in pending)
    assert not [row for row in pending if "segment-any-text" in row[1]]


def test_pending_downloads_is_empty_when_the_cache_cannot_be_read(fresh_machine, monkeypatch):
    from bibr.local.cli import dry_run

    def broken(*_args, **_kwargs):
        raise RuntimeError("cache exploded")

    monkeypatch.setattr(dry_run, "_hf_cache_status", broken)

    assert dry_run.pending_downloads(_config("--ocr", "glm-llama"), [Path("paper.pdf")]) == []


def test_the_download_notice_gives_the_count_the_total_and_each_models_size():
    from bibr.local.cli.dry_run import format_download_notice

    lines = format_download_notice(
        [
            ("layout", "org/layout", "will download (~0.13 GB)"),
            ("segmenter", "org/sat", "will download (~0.45 GB)"),
            ("OCR weights", "org/ocr", "will download (~1-2 GB)"),
            ("mystery", "org/other", "will download (size unknown)"),
        ]
    )

    assert "4 models" in lines[0]
    assert "about 2.6 GB" in lines[0]
    assert "1 of unknown size" in lines[0]
    assert len(lines) == 5
    assert "org/layout" in lines[1]
    assert "size unknown" in lines[4]


def test_the_download_notice_says_megabytes_for_a_small_total():
    from bibr.local.cli.dry_run import format_download_notice

    (head, row) = format_download_notice([("classifier", "org/c", "will download (~90 MB)")])

    assert "1 model " in head
    assert "about 90 MB" in head
    assert "org/c" in row


def test_there_is_no_notice_when_nothing_is_pending():
    from bibr.local.cli.dry_run import format_download_notice

    assert format_download_notice([]) == []


def test_announce_downloads_prints_before_anything_moves_and_names_the_bars(
    fresh_machine, monkeypatch
):
    from rich.console import Console

    from bibr.local.cli import dry_run
    from bibr.utils import hf_cache

    monkeypatch.setattr(
        dry_run,
        "pending_downloads",
        lambda config, files: [("layout", "org/layout", "will download (~0.13 GB)")],
    )
    named = []
    monkeypatch.setattr(hf_cache, "name_download_bars", lambda enabled=True: named.append(enabled))
    console = Console(file=io.StringIO(), width=200)

    dry_run.announce_downloads(_config("--ocr", "glm-llama"), [Path("paper.pdf")], console)

    text = console.file.getvalue()
    assert "First run: downloading 1 model (about 130 MB)" in text
    assert "org/layout" in text
    assert named == [True]


def test_announce_downloads_is_silent_on_a_warm_cache(fresh_machine, monkeypatch):
    from rich.console import Console

    from bibr.local.cli import dry_run
    from bibr.utils import hf_cache

    monkeypatch.setattr(dry_run, "pending_downloads", lambda config, files: [])
    named = []
    monkeypatch.setattr(hf_cache, "name_download_bars", lambda enabled=True: named.append(enabled))
    console = Console(file=io.StringIO(), width=200)

    dry_run.announce_downloads(_config("--ocr", "glm-llama"), [Path("paper.pdf")], console)

    assert console.file.getvalue() == ""
    assert named == []
