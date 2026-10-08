"""``bibr demo`` deletes uploads and downloads, and caps uploads while they arrive."""

import json
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")  # demo is gated behind gradio
if not hasattr(gr, "Blocks"):
    pytest.skip("gradio not fully installed", allow_module_level=True)

from gradio import processing_utils, route_utils

import bibr.config
import bibr.demo.local_app as local_app
import bibr.demo.server as server


@pytest.fixture
def _stub_pipeline(monkeypatch):
    # LocalPipeline is imported inside create_local_demo; patch it at the source.
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", lambda **_: object())
    # create_local_demo may turn the OCR disk cache on; restore the global
    # setting afterwards.
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


def test_json_download_is_served_in_place_and_deleted_with_the_cache(
    monkeypatch, tmp_path, _stub_pipeline
):
    """Gradio serves the written file itself (no copy) and its cleanup removes it."""
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)
    demo = local_app.create_local_demo(ocr_backend="glm-mlx")
    button = next(b for b in demo.blocks.values() if isinstance(b, gr.DownloadButton))

    path = local_app._write_json_file({"metadata": {"doi": "10.1/x"}})
    served = processing_utils.move_files_to_cache(
        button.postprocess(path), button, postprocess=True
    )
    route_utils.delete_files_created_by_app(demo, age=None)

    assert served["path"] == path
    assert not Path(path).exists()


def test_demo_deletes_cached_files_after_an_hour_by_default(monkeypatch, _stub_pipeline):
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)

    demo = local_app.create_local_demo(ocr_backend="glm-mlx")

    # Checked every five minutes, so a file lives at most 65 minutes.
    assert demo.delete_cache == (300, 3600)


@pytest.mark.parametrize(("ttl", "expected"), [("120", (120, 120)), ("7200", (300, 7200))])
def test_demo_cache_lifetime_follows_env(monkeypatch, _stub_pipeline, ttl, expected):
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", ttl)

    assert local_app.create_local_demo(ocr_backend="glm-mlx").delete_cache == expected


def test_demo_cache_lifetime_zero_keeps_files(monkeypatch, _stub_pipeline):
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", "0")

    assert local_app.create_local_demo(ocr_backend="glm-mlx").delete_cache is None


@pytest.mark.parametrize("ttl", ["1h", "", "-60", "1.5"])
def test_demo_refuses_invalid_cache_lifetime_before_building_pipeline(monkeypatch, ttl):
    built: list = []
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", lambda **kw: built.append(kw))
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", ttl)

    with pytest.raises(ValueError, match="DEMO_CACHE_TTL_SECONDS must be a whole number"):
        local_app.create_local_demo(ocr_backend="glm-mlx")

    assert built == []


def test_bibr_demo_exits_with_a_message_on_invalid_cache_lifetime(monkeypatch, capsys):
    created: list = []
    monkeypatch.setattr(local_app, "create_local_demo", lambda **kw: created.append(kw))
    monkeypatch.setattr(sys, "argv", ["bibr demo"])
    monkeypatch.setenv("DEMO_CACHE_TTL_SECONDS", "1h")

    with pytest.raises(SystemExit) as exc:
        server.main()

    assert exc.value.code == 1
    assert "DEMO_CACHE_TTL_SECONDS must be a whole number" in capsys.readouterr().err
    assert created == []


def test_demo_launch_caps_uploads_server_side(monkeypatch):
    launched: dict = {}

    class _Demo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr(local_app, "create_local_demo", lambda **_: _Demo())
    monkeypatch.setattr(sys, "argv", ["bibr demo", "--port", "7999"])
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("DEMO_MAX_FILE_SIZE_MB", raising=False)

    server.main()

    assert launched["max_file_size"] == "10mb"
    assert launched["server_port"] == 7999
