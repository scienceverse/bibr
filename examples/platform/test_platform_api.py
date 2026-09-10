"""
Test script for the Scienceverse platform API (submit → poll → download).

Mirrors the R sv_convert() function for debugging the async job queue.

Usage:
    python examples/platform/test_platform_api.py <file.pdf>
    python examples/platform/test_platform_api.py <file.pdf> --url http://localhost:8001
    PLATFORM_API_KEY=sv_... python examples/platform/test_platform_api.py paper.pdf
"""

import argparse
import os
import sys
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description="Test Scienceverse platform API")
    parser.add_argument("file", help="Path to document file")
    parser.add_argument(
        "--url",
        default=os.getenv("PLATFORM_API_URL", "https://platform.metacheck.app"),
        help="Platform API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("PLATFORM_API_KEY", ""),
        help="Platform API key (sv_...)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between status polls",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <filename>.json in current dir)",
    )
    parser.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Result format to download",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key. Set PLATFORM_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {args.api_key}"}
    client = httpx.Client(base_url=args.url, headers=headers, timeout=60)

    # ── 1. Health check ──────────────────────────────────────────────────────
    print(f"Platform: {args.url}")
    try:
        health = client.get("/health")
        print(f"Health:   {health.status_code} {health.json()}")
    except Exception as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ready = client.get("/ready")
        print(f"Ready:    {ready.status_code} {ready.json()}")
    except Exception as e:
        print(f"Readiness check failed: {e}")

    # ── 2. Submit job ────────────────────────────────────────────────────────
    filename = os.path.basename(args.file)
    print(f"\nSubmitting {filename} ...")

    with open(args.file, "rb") as f:
        resp = client.post("/jobs", files={"file": (filename, f)})

    if resp.status_code != 200:
        print(f"Submit failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)

    job = resp.json()
    job_id = job["job_id"]
    print(f"Job ID:   {job_id}")
    print(f"Status:   {job['status']}")

    # ── 3. Poll for completion ───────────────────────────────────────────────
    print("\nPolling ...")
    elapsed = 0.0
    last_stage = None

    while elapsed < args.timeout:
        time.sleep(args.poll_interval)
        elapsed += args.poll_interval

        status_resp = client.get(f"/jobs/{job_id}")
        status = status_resp.json()

        stage = status.get("stage", "")
        if stage != last_stage:
            print(f"  [{elapsed:6.1f}s] {status['status']}" + (f" ({stage})" if stage else ""))
            last_stage = stage

        if status["status"] == "complete":
            break
        elif status["status"] == "failed":
            print(f"\nJob failed: {status.get('stage', 'unknown error')}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"\nTimed out after {args.timeout}s (last: {status['status']})", file=sys.stderr)
        sys.exit(1)

    # ── 4. Download result ───────────────────────────────────────────────────
    print(f"\nDownloading result (format={args.format}) ...")

    result_resp = client.get(
        f"/jobs/{job_id}/result",
        params={"format": args.format},
        timeout=120,
    )

    if result_resp.status_code != 200:
        print(f"Download failed ({result_resp.status_code}): {result_resp.text}", file=sys.stderr)
        sys.exit(1)

    content = result_resp.content
    output_path = args.output or os.path.splitext(filename)[0] + ".json"

    with open(output_path, "wb") as f:
        f.write(content)

    print(f"Saved:    {output_path} ({len(content):,} bytes)")

    # ── 5. Quick peek at result ──────────────────────────────────────────────
    import json

    data = json.loads(content)
    info = data.get("info", {})
    print(f"\nTitle:        {info.get('title', 'N/A')}")
    print(f"DOI:          {info.get('doi', 'N/A')}")
    print(f"Authors:      {len(data.get('author', []))}")
    print(f"References:   {len(data.get('bib', []))}")
    print(f"Sections:     {len(data.get('section', []))}")
    print(f"Sentences:    {len(data.get('text', []))}")

    print("\nDone.")


if __name__ == "__main__":
    main()
