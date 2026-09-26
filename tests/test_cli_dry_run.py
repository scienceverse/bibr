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

import json
import re

import pytest


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
