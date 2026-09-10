"""Executable checks for opt-in integration clients and the local API example."""

from __future__ import annotations

import ast
import inspect
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).parents[1]


def _no_request(*_args, **_kwargs):
    pytest.fail(
        "Missing platform credentials must fail before HTTP client construction or requests"
    )


def _notebook_code(path: str):
    notebook = json.loads((ROOT / path).read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.parametrize(
    "path",
    [
        "examples/platform/python_platform_demo.ipynb",
        "notebooks/python_api_demo.ipynb",
        "notebooks/python_library_demo.ipynb",
    ],
)
def test_public_notebooks_have_no_saved_extractions(path):
    notebook = json.loads((ROOT / path).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs")
            assert cell.get("execution_count") is None


def test_platform_notebook_requires_key_before_client(monkeypatch):
    monkeypatch.delenv("PLATFORM_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "Client", _no_request)
    with pytest.raises(ValueError, match="Set PLATFORM_API_KEY"):
        exec(_notebook_code("examples/platform/python_platform_demo.ipynb")[0], {})  # noqa: S102 - repository notebook under test


def test_local_api_notebook_sends_bearer_and_reads_current_export(monkeypatch, tmp_path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"synthetic upload")
    monkeypatch.setenv("PAPER_PATH", str(paper))
    monkeypatch.setenv("BIBR_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("BIBR_API_URL", "http://localhost:8000")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    export = {
        "paper_id": "synthetic",
        "info": {"title": "Example", "doi": None, "keywords": []},
        "author": [{"given": "Ada", "family": "Example"}],
        "section": [{"header": "Methods", "section_type": "method", "classification_score": 0.9}],
        "bib": [{"bib_id": 1, "authors": "Example A", "year": 2026, "title": "Reference"}],
    }
    requests = []

    def fake_request(url, **kwargs):
        requests.append((url, kwargs))
        body = export if url.endswith("/papers/extract") else {"status": "ready"}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_request)
    monkeypatch.setattr(httpx, "post", fake_request)
    namespace = {}
    for code in _notebook_code("notebooks/python_api_demo.ipynb"):
        exec(code, namespace)  # noqa: S102 - repository notebook with mocked HTTP
    assert namespace["data"] == export
    assert len(requests) == 3
    assert all(
        kwargs["headers"] == {"Authorization": "Bearer synthetic-test-key"}
        for _, kwargs in requests
    )
    Path(namespace["output_path"]).unlink()


@pytest.mark.parametrize("use_supplied_path", [False, True])
async def test_library_notebook_uses_public_async_api(monkeypatch, tmp_path, use_supplied_path):
    """Run every cell in a live event loop with extraction mocked at the public API."""
    import bibr

    export = json.loads((ROOT / "tests/fixtures/inspect_full_export.json").read_text())
    export["info"]["paper_type_confidence"] = None
    export["info"]["oecd_confidence"] = None
    export["table"][0]["contents"] = [["Example", "Value"], ["A", "1"]]
    if use_supplied_path:
        export["text"] = []
        export["bib_match"] = []
        expected_path = tmp_path / "caller.pdf"
        expected_path.write_bytes(b"synthetic upload")
        monkeypatch.setenv("PAPER_PATH", str(expected_path))
    else:
        monkeypatch.delenv("PAPER_PATH", raising=False)
        expected_path = ROOT / "bibr/data/sample_paper.pdf"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    result = bibr.Result(export)
    calls = []

    async def fake_achew_file(path):
        assert Path(path) == expected_path
        assert Path(path).is_file()
        calls.append("single")
        return result

    class FakeChewer:
        def __init__(self, *, memory):
            assert memory == "balanced"

        async def __aenter__(self):
            calls.append("enter")
            return self

        async def __aexit__(self, *_exc):
            calls.append("close")

        async def achew_many(self, paths):
            assert paths == [expected_path]
            assert all(Path(path).is_file() for path in paths)
            calls.append("batch")
            if use_supplied_path:
                return [bibr.ChewFailure(str(expected_path), "Example extraction failure")]
            return [result]

    monkeypatch.setattr(bibr, "achew_file", fake_achew_file)
    monkeypatch.setattr(bibr, "Chewer", FakeChewer)
    displayed = []
    namespace = {"display": displayed.append}
    for code in _notebook_code("notebooks/python_library_demo.ipynb"):
        cell = compile(code, "python_library_demo.ipynb", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        outcome = eval(cell, namespace)  # noqa: S307 - repository notebook with mocked extraction
        if inspect.isawaitable(outcome):
            await outcome

    assert calls == ["single", "enter", "batch", "close"]
    assert namespace["result"] is result
    assert json.loads(namespace["output_path"].read_text()) == result.data
    assert len(displayed) == (1 if use_supplied_path else 2)


def test_r_helper_preserves_nullable_info_and_current_media(tmp_path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")
    probe = subprocess.run(
        [rscript, "-e", 'quit(status=if (requireNamespace("jsonlite", quietly=TRUE)) 0 else 1)'],
        capture_output=True,
        check=False,
    )
    if probe.returncode:
        pytest.skip("R jsonlite is not installed")
    fixture = tmp_path / "paper.json"
    fixture.write_text(
        json.dumps(
            {
                "paper_id": "synthetic",
                "info": {"title": "Example", "doi": None, "keywords": []},
                "figure": [{"figure_id": 1}],
                "bib": [],
                "validation": {"errors": 0},
            }
        )
    )
    result = subprocess.run(
        [
            rscript,
            "-e",
            (
                "args <- commandArgs(TRUE); source(args[1]); x <- read_bibr_json(args[2]); "
                'stopifnot(x$paper_id == "synthetic", is.null(x$info$doi), '
                "nrow(x$figure) == 1, nrow(x$bib) == 0, x$validation$errors == 0)"
            ),
            str(ROOT / "scripts/read_json_response.R"),
            str(fixture),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
