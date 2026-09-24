"""``bibr demo`` deletes uploads and downloads, and caps uploads while they arrive."""

import json
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")  # demo is gated behind gradio
if not hasattr(gr, "Blocks"):
    pytest.skip("gradio not fully installed", allow_module_level=True)

import bibr.config
import bibr.demo.local_app as local_app
import bibr.demo.server as server


@pytest.fixture
def _stub_pipeline(monkeypatch):
    # LocalPipeline is imported inside create_local_demo; patch it at the source.
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", lambda **_: object())
    # create_local_demo turns the OCR disk cache on unless CACHE_OCR was set;
    # restore the global setting afterwards.
    section = bibr.config.Settings.cache
    original = section.ocr
    was_explicit = "ocr" in section.model_fields_set
    yield
    section.ocr = original
    if not was_explicit:
        section.model_fields_set.discard("ocr")


def test_json_download_is_written_inside_gradio_temp_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))

    path = Path(local_app._write_json_file({"metadata": {"doi": "10.1/x"}}, suffix=".no-images"))

    assert path.is_relative_to(tmp_path / "gradio")
    assert path.name == "10.1_x.no-images.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"metadata": {"doi": "10.1/x"}}


def test_demo_deletes_cached_files_after_an_hour_by_default(monkeypatch, _stub_pipeline):
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)

    demo = local_app.create_local_demo(ocr_backend="glm-mlx")

    assert demo.delete_cache == (3600, 3600)


def test_demo_cache_lifetime_follows_env(monkeypatch, _stub_pipeline):
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", "120")

    assert local_app.create_local_demo(ocr_backend="glm-mlx").delete_cache == (120, 120)


def test_demo_cache_lifetime_zero_keeps_files(monkeypatch, _stub_pipeline):
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", "0")

    assert local_app.create_local_demo(ocr_backend="glm-mlx").delete_cache is None


def test_demo_launch_caps_uploads_server_side(monkeypatch):
    launched: dict = {}

    class _Demo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr(local_app, "create_local_demo", lambda **_: _Demo())
    monkeypatch.setattr(sys, "argv", ["bibr demo", "--port", "7999"])

    server.main()

    assert launched["max_file_size"] == f"{local_app._MAX_FILE_SIZE_MB}mb"
    assert launched["server_port"] == 7999
