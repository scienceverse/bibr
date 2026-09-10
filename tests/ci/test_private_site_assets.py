from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def test_public_site_allows_crawlers() -> None:
    robots = (ROOT / "docs" / "robots.txt").read_text(encoding="utf-8")

    assert robots == "User-agent: *\nAllow: /\n"


def test_public_site_preserves_security_headers_without_blocking_indexing() -> None:
    headers = (ROOT / "docs" / "_headers").read_text(encoding="utf-8")

    assert headers.startswith("/*\n")
    assert "noindex" not in headers
    assert "nofollow" not in headers
    assert "X-Content-Type-Options: nosniff" in headers
    assert "Referrer-Policy: strict-origin-when-cross-origin" in headers
    assert "Permissions-Policy: camera=(), microphone=(), geolocation=()" in headers


def test_mkdocs_copies_site_policy_assets(tmp_path: Path) -> None:
    # mkdocs lives in the ``docs`` dependency group, which the test-running
    # installs (``--extra all``) do not pull in. The strict build itself is
    # gated by the dedicated "Build strict MkDocs artifact" CI job; this test
    # only adds the asset-copy assertions when the toolchain happens to be
    # present.
    pytest.importorskip("mkdocs")

    site_dir = tmp_path / "site"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(site_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (site_dir / "robots.txt").read_text(encoding="utf-8") == (
        ROOT / "docs" / "robots.txt"
    ).read_text(encoding="utf-8")
    assert (site_dir / "_headers").read_text(encoding="utf-8") == (
        ROOT / "docs" / "_headers"
    ).read_text(encoding="utf-8")

    # Root docs are published from their canonical sources, and private
    # engineering records remain excluded even though the nav is broader.
    assert "Accuracy and scope" in (site_dir / "limitations" / "index.html").read_text()
    llm_page = (site_dir / "llm-use" / "index.html").read_text()
    assert 'href="../guides/configuration/"' in llm_page
    assert 'href="../limitations/"' in llm_page
    assert "[!NOTE]" not in llm_page
    assert not (site_dir / "reports").exists()
    assert not (site_dir / "superpowers").exists()

    schema = json.loads((site_dir / "reference" / "paper.schema.json").read_text())
    assert {"_regions", "_native_source"} <= schema["properties"].keys()
    assert "author_id" in schema["$defs"]["AuthorExport"]["properties"]


def test_private_operations_runbooks_are_not_published() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    rendered_nav = repr(config["nav"])

    for relative_path in ("operations/private-site.md", "operations/ci-cd.md"):
        assert not (ROOT / "docs" / relative_path).exists()
        assert relative_path not in rendered_nav
