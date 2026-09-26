"""The GROBID runner against a mocked GROBID server: retries, failures, manifest."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from evaluation.evaluate import load_expected_ids
from evaluation.grobid_run import GROBID_PARAMETERS, Job, discover, main, process, run
from evaluation.grobid_tei import FAILED_SUFFIX, MANIFEST_NAME, TEI_SUFFIX, convert_directory

TEI = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>
<title level="a" type="main">A synthetic title</title></titleStmt></fileDesc></teiHeader>
<text><body/><back/></text></TEI>"""


def grobid(*responses, seen=None):
    """A client whose server answers the processing endpoint with *responses* in turn."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/isalive":
            return httpx.Response(200, text="true")
        if request.url.path == "/api/version":
            return httpx.Response(200, text="0.9.1")
        if seen is not None:
            seen.append(request)
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://grobid.test")


@pytest.fixture
def pdfs(tmp_path):
    directory = tmp_path / "pdfs"
    directory.mkdir()
    for name in ("a", "b"):
        (directory / f"{name}.pdf").write_bytes(f"%PDF-1.7 synthetic {name}".encode())
    return directory


def no_sleep(_seconds):
    return None


class TestProcess:
    def test_success_writes_tei_and_digests(self, pdfs, tmp_path):
        seen = []
        record = process(
            Job("a", pdfs / "a.pdf"),
            tmp_path,
            client=grobid(httpx.Response(200, content=TEI), seen=seen),
            attempts=3,
        )
        data = (pdfs / "a.pdf").read_bytes()
        assert record["status"] == "ok" and record["attempts"] == 1 and record["error"] is None
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
        assert record["md5"] == hashlib.md5(data, usedforsecurity=False).hexdigest()
        assert record["wall_seconds"] is not None and record["http_status"] == 200
        assert (tmp_path / f"a{TEI_SUFFIX}").read_bytes() == TEI
        body = seen[0].content
        for name, value in GROBID_PARAMETERS.items():
            assert f'name="{name}"\r\n\r\n{value}\r\n'.encode() in body
        assert b'name="input"; filename="a.pdf"' in body

    def test_busy_server_is_retried(self, pdfs, tmp_path):
        slept = []
        client = grobid(httpx.Response(503), httpx.Response(200, content=TEI))
        record = process(
            Job("a", pdfs / "a.pdf"), tmp_path, client=client, attempts=3, sleep=slept.append
        )
        assert (record["status"], record["attempts"], slept) == ("ok", 2, [2.0])

    def test_connection_errors_are_retried(self, pdfs, tmp_path):
        client = grobid(httpx.ConnectError("refused"), httpx.Response(200, content=TEI))
        record = process(
            Job("a", pdfs / "a.pdf"), tmp_path, client=client, attempts=2, sleep=no_sleep
        )
        assert record["status"] == "ok" and record["attempts"] == 2

    def test_persistent_server_error_fails_with_a_marker(self, pdfs, tmp_path):
        client = grobid(httpx.Response(500, text="[GENERAL] An exception occurred"))
        record = process(
            Job("a", pdfs / "a.pdf"), tmp_path, client=client, attempts=3, sleep=no_sleep
        )
        assert (record["status"], record["attempts"], record["http_status"]) == ("failed", 3, 500)
        assert record["error"].startswith("HTTP 500")
        assert not (tmp_path / f"a{TEI_SUFFIX}").exists()
        marker = json.loads((tmp_path / f"a{FAILED_SUFFIX}").read_text())
        assert marker["paper_id"] == "a" and marker["status"] == "failed"

    def test_client_error_is_not_retried(self, pdfs, tmp_path):
        record = process(
            Job("a", pdfs / "a.pdf"),
            tmp_path,
            client=grobid(httpx.Response(400)),
            attempts=3,
            sleep=no_sleep,
        )
        assert (record["status"], record["attempts"]) == ("failed", 1)

    @pytest.mark.parametrize(
        ("response", "error", "attempts"),
        [
            (httpx.Response(204), "GROBID returned no content", 1),
            (httpx.Response(200, text="<html>proxy error</html>"), "the response is not TEI", 2),
        ],
    )
    def test_empty_or_foreign_responses_fail(self, pdfs, tmp_path, response, error, attempts):
        record = process(
            Job("a", pdfs / "a.pdf"), tmp_path, client=grobid(response), attempts=2, sleep=no_sleep
        )
        assert (record["status"], record["error"], record["attempts"]) == (
            "failed",
            error,
            attempts,
        )

    def test_missing_pdf_is_a_failure_not_a_skip(self, tmp_path):
        record = process(
            Job("gone", None), tmp_path, client=grobid(httpx.Response(200, content=TEI)), attempts=1
        )
        assert record["status"] == "failed" and record["error"] == "no PDF for this paper id"
        assert (tmp_path / f"gone{FAILED_SUFFIX}").is_file()


def test_discover_keeps_expected_ids_without_a_pdf(pdfs):
    assert discover(pdfs) == [Job("a", pdfs / "a.pdf"), Job("b", pdfs / "b.pdf")]
    assert discover(pdfs, {"b", "gone"}) == [Job("b", pdfs / "b.pdf"), Job("gone", None)]


def test_run_manifest_counts_every_paper_and_feeds_the_evaluator(pdfs, tmp_path):
    out = tmp_path / "tei"
    jobs = discover(pdfs, {"a", "b", "gone"})

    def handler(request: httpx.Request) -> httpx.Response:
        if b'filename="b.pdf"' in request.content:
            return httpx.Response(500)
        return httpx.Response(200, content=TEI)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://grobid.test")
    manifest = run(
        jobs,
        out,
        client=client,
        header={"grobid": {"version": "0.9.1"}},
        workers=2,
        attempts=2,
        sleep=no_sleep,
    )
    assert manifest["ids"] == ["a", "b", "gone"]
    assert manifest["counts"] == {"ok": 1, "failed": 2, "pending": 0}
    assert manifest["grobid"] == {"version": "0.9.1"}
    saved = json.loads((out / MANIFEST_NAME).read_text())
    assert [p["status"] for p in saved["papers"]] == ["ok", "failed", "failed"]
    assert load_expected_ids(str(out / MANIFEST_NAME)) == {"a", "b", "gone"}

    summary = convert_directory(out, tmp_path / "json")
    assert summary.converted == ["a"]
    assert set(summary.failed) == {"b", "gone"}
    payload = json.loads((tmp_path / "json" / "a.json").read_text())
    assert payload["source"]["file_name"] == "a.pdf"
    assert payload["source"]["sha256"] == saved["papers"][0]["sha256"]


def test_resume_keeps_finished_papers(pdfs, tmp_path):
    out = tmp_path / "tei"
    jobs = discover(pdfs)
    first = run(jobs, out, client=grobid(httpx.Response(200, content=TEI)), header={}, attempts=1)
    previous = {p["paper_id"]: p for p in first["papers"]}
    seen = []
    second = run(
        jobs,
        out,
        client=grobid(httpx.Response(500), seen=seen),
        header={},
        attempts=1,
        previous=previous,
    )
    assert seen == [] and second["counts"]["ok"] == 2


def _mock_clients(monkeypatch, handler):
    """Route the CLI's httpx.Client through *handler* instead of the network."""
    real = httpx.Client

    def client(**kwargs):
        return real(transport=httpx.MockTransport(handler), base_url=kwargs["base_url"])

    monkeypatch.setattr("evaluation.grobid_run.httpx.Client", client)


def test_cli_run_records_version_parameters_and_image(pdfs, tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, text="0.9.1")
        return httpx.Response(200, content=TEI if request.method == "POST" else b"true")

    _mock_clients(monkeypatch, handler)
    out = tmp_path / "tei"
    argv = ["--grobid-url", "http://grobid.test/", "--pdf-dir", str(pdfs), "--out", str(out)]
    assert main([*argv, "--grobid-image", "grobid/grobid:0.9.1-full"]) == 0
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["grobid"]["version"] == "0.9.1"
    assert manifest["grobid"]["image"] == "grobid/grobid:0.9.1-full"
    assert manifest["parameters"] == GROBID_PARAMETERS
    assert manifest["client"] == {"workers": 4, "timeout_seconds": 600.0, "attempts": 4}
    assert manifest["counts"] == {"ok": 2, "failed": 0, "pending": 0}
    # A second run into the same directory must be an explicit resume.
    with pytest.raises(SystemExit):
        main(argv)
    assert main([*argv, "--resume"]) == 0


def test_cli_reports_an_unreachable_server(pdfs, tmp_path, monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _mock_clients(monkeypatch, handler)
    argv = [
        "--grobid-url",
        "http://grobid.test",
        "--pdf-dir",
        str(pdfs),
        "--out",
        str(tmp_path / "t"),
    ]
    assert main(argv) == 2
    assert "not reachable" in capsys.readouterr().err
