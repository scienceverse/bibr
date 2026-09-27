"""Run a directory of PDFs through a GROBID server, for the GROBID benchmark arm.

Each PDF is sent to ``/api/processFulltextDocument`` with the fixed parameters
in ``GROBID_PARAMETERS``, by a bounded pool of workers, with a timeout and
retries. The output directory receives::

    <paper_id>.grobid.tei.xml      GROBID's TEI, for every paper that succeeded
    <paper_id>.grobid.failed.json  the failure record, for every paper that did not
    manifest.json                  the GROBID version, the parameters, and every
                                   paper's PDF name, SHA-256, MD5, wall time,
                                   attempts and outcome

A failed paper is never dropped: it stays in the manifest's ``ids``, which the
evaluator's ``--expected-ids`` reads, so it scores 0 on the pass-rate floors
like a bibr paper that crashed. Whatever fails one paper (an HTTP error, a
connection or decoding error, an unreadable PDF) is recorded for that paper,
and the run goes on. Full run, from a server to scores::

    uv run python -m evaluation.grobid_run --grobid-url http://localhost:8070 \\
        --pdf-dir papers/ --ids-file cohort-ids.txt --out grobid-tei/
    uv run python -m evaluation.grobid_tei --tei-dir grobid-tei/ --out grobid-json/
    uv run python -m evaluation.evaluate --results-dir grobid-json/ \\
        --gold-dirs /path/to/gold --expected-ids grobid-tei/manifest.json \\
        --output grobid-eval.json

``--ids-file`` lists the cohort bibr is scored on, so the manifest's ``ids``
are the same ``--expected-ids`` list as bibr's. ``--resume`` continues an
interrupted run over the same paper ids. It refuses other paper ids, a
server that reports another GROBID version, other request parameters, and a
``--grobid-image`` other than the one the manifest records, including one
given when the first run named none. A resume without ``--grobid-image``
keeps the recorded image.

Wall times are measured under the chosen ``--workers`` concurrency; compare
them with bibr timings taken at the same concurrency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evaluation.grobid_tei import FAILED_SUFFIX, MANIFEST_NAME, TEI_SUFFIX

RUNNER_NAME = "bibr-grobid-run"
RUNNER_VERSION = "1.0"
ENDPOINT = "/api/processFulltextDocument"

# The request parameters are part of the benchmark definition. GROBID echoes
# the ones it applied in every TEI (appInfo/label[@type="parameters"]), and
# evaluation.grobid_tei reports them per run.
GROBID_PARAMETERS: dict[str, str] = {
    # bibr is scored on what it reads off the page, before any registry lookup,
    # so GROBID runs without Crossref/biblio-glutton consolidation too.
    "consolidateHeader": "0",
    "consolidateCitations": "0",
    "consolidateFunders": "0",
    # Add the printed reference and affiliation strings as TEI notes. They do
    # not change GROBID's parse; the converter keeps them as the printed text.
    "includeRawCitations": "1",
    "includeRawAffiliations": "1",
    # Everything else is GROBID's default: every page, no sentence
    # segmentation, no coordinates, no consolidation of anything else.
}

# Worth another attempt: GROBID busy (503), a gateway or server hiccup, or a
# response that is not TEI (a proxy error page). A 4xx is the request's fault.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Job:
    """One paper to process; ``pdf`` is None when an expected id has no PDF."""

    paper_id: str
    pdf: Path | None


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_atomic(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, path)


def discover(pdf_dir: Path, ids: set[str] | None = None) -> list[Job]:
    """One job per ``*.pdf`` in *pdf_dir* (the paper id is the file stem).

    With *ids*, only those papers; an id without a PDF becomes a job that
    fails, so it stays in the manifest instead of vanishing.
    """
    pdfs: dict[str, Path] = {}
    for path in sorted(pdf_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".pdf":
            if path.stem in pdfs:
                raise ValueError(f"two PDFs share the paper id {path.stem!r}")
            pdfs[path.stem] = path
    wanted = sorted(ids) if ids is not None else sorted(pdfs)
    return [Job(paper_id, pdfs.get(paper_id)) for paper_id in wanted]


def _send(
    pdf: Path,
    record: dict[str, Any],
    tei_path: Path,
    *,
    client: httpx.Client,
    attempts: int,
    sleep: Callable[[float], None],
) -> None:
    """Post *pdf* until GROBID returns TEI or the failure is final; fill *record*."""
    data = pdf.read_bytes()
    record["bytes"] = len(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    record["md5"] = hashlib.md5(data, usedforsecurity=False).hexdigest()
    for attempt in range(1, attempts + 1):
        record["attempts"] = attempt
        retry = False
        sent = time.monotonic()
        try:
            response = client.post(
                ENDPOINT,
                files={"input": (pdf.name, data, "application/pdf")},
                data=GROBID_PARAMETERS,
            )
        except httpx.TransportError as error:
            record["error"] = f"{type(error).__name__}: {error}"
            retry = True
        except httpx.HTTPError as error:
            # A response httpx could not decode, a redirect loop: not transient.
            record["error"] = f"{type(error).__name__}: {error}"
        else:
            record["http_status"] = response.status_code
            body = response.content
            if response.status_code == 200 and body.strip() and b"<TEI" in body[:4096]:
                record["wall_seconds"] = round(time.monotonic() - sent, 3)
                _write_atomic(tei_path, body)
                record.update(status="ok", tei=tei_path.name, error=None)
                return
            if response.status_code == 200 and body.strip():
                record["error"] = "the response is not TEI"
                retry = True
            elif response.status_code in (200, 204):
                record["error"] = "GROBID returned no content"
            else:
                snippet = body[:200].decode("utf-8", "replace").strip()
                record["error"] = f"HTTP {response.status_code}: {snippet}"
                retry = response.status_code in _RETRY_STATUSES
        record["wall_seconds"] = round(time.monotonic() - sent, 3)
        if not retry or attempt == attempts:
            return
        sleep(min(60.0, 2.0**attempt))


def process(
    job: Job,
    out: Path,
    *,
    client: httpx.Client,
    attempts: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Send one PDF to GROBID; write its TEI or its failure record; return the record."""
    record: dict[str, Any] = {
        "paper_id": job.paper_id,
        "pdf": job.pdf.name if job.pdf else None,
        "bytes": None,
        "sha256": None,
        "md5": None,
        "status": "failed",
        "tei": None,
        "http_status": None,
        "attempts": 0,
        "wall_seconds": None,
        "elapsed_seconds": None,
        "error": None,
    }
    started = time.monotonic()
    tei_path = out / f"{job.paper_id}{TEI_SUFFIX}"
    failed_path = out / f"{job.paper_id}{FAILED_SUFFIX}"
    if job.pdf is None:
        record["error"] = "no PDF for this paper id"
    else:
        try:
            _send(job.pdf, record, tei_path, client=client, attempts=attempts, sleep=sleep)
        except OSError as error:
            # A PDF that cannot be read, or a TEI that cannot be written, fails
            # this paper, not the run.
            record.update(status="failed", tei=None, error=f"{type(error).__name__}: {error}")
        if record["status"] == "ok":
            failed_path.unlink(missing_ok=True)
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record["finished_at"] = _utc_now()
    if record["status"] != "ok":
        tei_path.unlink(missing_ok=True)
        failed_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def server_version(client: httpx.Client) -> str:
    """GROBID's version string; raises when the server is not alive."""
    client.get("/api/isalive", timeout=30).raise_for_status()
    response = client.get("/api/version", timeout=30)
    response.raise_for_status()
    text = response.text.strip()
    try:
        payload = json.loads(text)
    except ValueError:
        return text
    return str(payload.get("version") or text) if isinstance(payload, dict) else text


def run(
    jobs: list[Job],
    out: Path,
    *,
    client: httpx.Client,
    header: dict[str, Any],
    workers: int = 4,
    attempts: int = 4,
    previous: dict[str, dict[str, Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Process *jobs* with at most *workers* requests in flight; return the manifest.

    The manifest is rewritten after every paper, so an interrupted run can be
    resumed. Papers *previous* records as done (with their TEI present) are
    kept as they are.
    """
    out.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for job in jobs:
        done = (previous or {}).get(job.paper_id)
        if done and done.get("status") == "ok" and (out / str(done.get("tei"))).is_file():
            records[job.paper_id] = done
    manifest = {
        **header,
        "ids": sorted(job.paper_id for job in jobs),
        "counts": {},
        "papers": [],
    }
    lock = threading.Lock()

    def save() -> None:
        papers = [records[job.paper_id] for job in jobs if job.paper_id in records]
        manifest["papers"] = papers
        manifest["counts"] = {
            "ok": sum(1 for p in papers if p["status"] == "ok"),
            "failed": sum(1 for p in papers if p["status"] != "ok"),
            "pending": len(jobs) - len(papers),
        }
        manifest["updated_at"] = _utc_now()
        _write_atomic(out / MANIFEST_NAME, json.dumps(manifest, indent=2).encode("utf-8"))

    def work(job: Job) -> None:
        record = process(job, out, client=client, attempts=attempts, sleep=sleep)
        with lock:
            records[job.paper_id] = record
            save()
            print(
                f"[{len(records)}/{len(jobs)}] {job.paper_id}: {record['status']}"
                + (f" ({record['error']})" if record["error"] else ""),
                flush=True,
            )

    todo = [job for job in jobs if job.paper_id not in records]
    save()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, todo))
    manifest["finished_at"] = _utc_now()
    save()
    return manifest


def resume_conflict(prior: dict[str, Any], version: str, image: str | None) -> str | None:
    """Why the run a previous manifest records cannot be continued here, or None.

    A resumed run must use the same GROBID build and request parameters, or
    its papers would be scored as one run while two GROBIDs made them.
    """
    recorded = prior.get("grobid") or {}
    if recorded.get("version") != version:
        return (
            f"its manifest records GROBID {recorded.get('version')}, "
            f"but the server runs {version}; use a new --out directory"
        )
    # Without --grobid-image the resume keeps the recorded image; with it, the
    # image must be the recorded one, since papers made under an unnamed image
    # cannot be credited to a named one.
    if image and recorded.get("image") != image:
        named = f"the image {recorded['image']}" if recorded.get("image") else "no image"
        return f"its manifest records {named}, not {image}; use a new --out directory"
    if prior.get("parameters") != GROBID_PARAMETERS:
        return (
            "its manifest records other GROBID parameters than this runner sends; "
            "use a new --out directory"
        )
    return None


def _papers(count: int) -> str:
    return f"{count} paper" + ("" if count == 1 else "s")


def _some(ids: list[str], shown: int = 5) -> str:
    listed = ", ".join(ids[:shown])
    return listed + (f", and {len(ids) - shown} more" if len(ids) > shown else "")


def paper_set_conflict(prior: dict[str, Any], jobs: list[Job]) -> str | None:
    """Why *jobs* are not the papers a previous manifest records, or None.

    A resume over other papers would widen or narrow the cohort under one
    manifest, and drop the records of the papers it leaves out.
    """
    recorded = {str(paper_id) for paper_id in prior.get("ids") or []}
    current = {job.paper_id for job in jobs}
    if recorded == current:
        return None
    differences = []
    if added := sorted(current - recorded):
        differences.append(f"{len(added)} not in the manifest ({_some(added)})")
    if dropped := sorted(recorded - current):
        differences.append(f"{len(dropped)} left out of this run ({_some(dropped)})")
    return (
        f"its manifest lists {_papers(len(recorded))} and this run {len(current)}: "
        + "; ".join(differences)
        + "; pass the --pdf-dir and --ids-file the run started with"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--grobid-url", required=True, help="GROBID server, e.g. http://localhost:8070"
    )
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory of PDFs")
    parser.add_argument("--out", type=Path, required=True, help="Directory for TEI and manifest")
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Only these paper ids (PDF stems), one per line, '#' starting a comment "
        "line; an id without a PDF is recorded as failed rather than skipped",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Requests in flight (default 4); keep it at or below the server's "
        "concurrency setting, or GROBID answers 503 and the run spends time retrying",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="Seconds to wait for one response"
    )
    parser.add_argument(
        "--attempts", type=int, default=4, help="Tries per PDF for retryable failures"
    )
    parser.add_argument(
        "--grobid-image",
        help="Recorded verbatim in the manifest, e.g. the container image the server runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run in --out over the same paper ids: keep papers "
        "already done, redo the rest. Refused when the paper ids, the server's GROBID "
        "version or the request parameters differ from the ones the manifest records, "
        "or when --grobid-image names another image than the recorded one; without "
        "--grobid-image the recorded image is kept",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.attempts < 1 or args.timeout <= 0:
        parser.error("--workers, --attempts and --timeout must be positive")

    prior: dict[str, Any] | None = None
    previous: dict[str, dict[str, Any]] = {}
    manifest_path = args.out / MANIFEST_NAME
    if args.resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = {str(p["paper_id"]): p for p in prior.get("papers") or []}
    elif args.out.is_dir() and any(args.out.iterdir()) and not args.resume:
        parser.error(f"{args.out} is not empty; use an empty directory or --resume")

    ids = None
    if args.ids_file:
        try:
            # utf-8-sig: a byte-order mark would otherwise become part of the first id.
            lines = args.ids_file.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            parser.error(f"cannot read --ids-file {args.ids_file}: {error}")
        ids = {stripped for line in lines if (stripped := line.strip()) and stripped[0] != "#"}
    if not args.pdf_dir.is_dir():
        parser.error(f"--pdf-dir {args.pdf_dir} is not a directory")
    try:
        jobs = discover(args.pdf_dir, ids)
    except OSError as error:
        parser.error(f"cannot read --pdf-dir {args.pdf_dir}: {error}")
    except ValueError as error:
        parser.error(f"{args.pdf_dir}: {error}")
    if not jobs:
        parser.error(f"no PDFs to process in {args.pdf_dir}")
    if prior is not None and (conflict := paper_set_conflict(prior, jobs)):
        parser.error(f"cannot resume the run in {args.out}: {conflict}")

    from evaluation.evaluate import bibr_commit

    timeout = httpx.Timeout(args.timeout, connect=30.0)
    with httpx.Client(base_url=args.grobid_url.rstrip("/"), timeout=timeout) as client:
        try:
            version = server_version(client)
        except httpx.HTTPError as error:
            print(f"GROBID is not reachable at {args.grobid_url}: {error}", file=sys.stderr)
            return 2
        image = args.grobid_image
        started_at = _utc_now()
        resumed: dict[str, Any] = {}
        if prior is not None:
            if conflict := resume_conflict(prior, version, image):
                parser.error(f"cannot resume the run in {args.out}: {conflict}")
            # The manifest describes the whole run, not only its last part.
            recorded = prior.get("grobid") or {}
            image = image or recorded.get("image")
            resumed = {"resumed_at": [*(prior.get("resumed_at") or []), started_at]}
            started_at = prior.get("started_at") or started_at
        print(f"GROBID {version}; {len(jobs)} paper(s); {args.workers} worker(s)")
        header = {
            "runner": {
                "name": RUNNER_NAME,
                "version": RUNNER_VERSION,
                "bibr_commit": bibr_commit(),
            },
            "grobid": {"version": version, "image": image, "endpoint": ENDPOINT},
            "parameters": GROBID_PARAMETERS,
            "client": {
                "workers": args.workers,
                "timeout_seconds": args.timeout,
                "attempts": args.attempts,
            },
            "started_at": started_at,
            **resumed,
        }
        manifest = run(
            jobs,
            args.out,
            client=client,
            header=header,
            workers=args.workers,
            attempts=args.attempts,
            previous=previous,
        )
    counts = manifest["counts"]
    print(f"done: {counts['ok']} ok, {counts['failed']} failed; manifest: {manifest_path}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
