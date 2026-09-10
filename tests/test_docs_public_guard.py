from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.docs_public_guard import (
    nav_paths,
    on_files,
    on_post_build,
    on_pre_build,
    private_content_findings,
)


def test_navigation_collects_nested_local_pages():
    assert nav_paths(
        [
            {"Home": "index.md"},
            {"Guide": ["guides/library.md"]},
            {"External": "https://example.org/"},
        ]
    ) == {"index.md", "guides/library.md"}


def _file(path, source, *, document=False, excluded=False):
    return SimpleNamespace(
        src_uri=path,
        src_dir=source,
        inclusion=SimpleNamespace(is_excluded=lambda: excluded),
        is_documentation_page=lambda: document,
    )


def test_unlisted_markdown_and_downloaded_payloads_cannot_publish(tmp_path):
    config = {"docs_dir": str(tmp_path), "nav": [{"Home": "index.md"}]}
    for file in [
        _file("unexpected.md", str(tmp_path), document=True),
        _file("assets/paper.pdf", str(tmp_path)),
        _file("assets/output.json", str(tmp_path)),
    ]:
        with pytest.raises(ValueError):
            on_files([file], config)
    files = [
        _file("index.md", str(tmp_path), document=True),
        _file("reference/paper.schema.json", None),
        _file("assets/logo.svg", str(tmp_path)),
        _file("unpublished.md", str(tmp_path), document=True, excluded=True),
    ]
    assert on_files(files, config) is files


def test_post_build_catches_generated_reference_and_search_leaks(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "settings.html").write_text(
        "<code>/home/<span>operator</span>/work/output.json</code>"
    )
    with pytest.raises(ValueError, match="local operator path"):
        on_post_build({"site_dir": str(tmp_path)})
    (reference / "settings.html").write_text("Set your output directory before processing.")
    (tmp_path / "search.json").write_text('{"source_id": "W123456789"}')
    with pytest.raises(ValueError, match="source corpus record"):
        on_post_build({"site_dir": str(tmp_path)})


def test_generic_user_owned_evaluation_paths_are_allowed():
    assert not private_content_findings(
        "Compare your own results in /path/to/gold with outputs/papers. "
        "Review metadata accuracy and processing failures."
    )


def test_public_repository_documents_do_not_reference_internal_material():
    root = Path(__file__).resolve().parents[1]
    on_pre_build({"docs_dir": str(root / "docs")})
