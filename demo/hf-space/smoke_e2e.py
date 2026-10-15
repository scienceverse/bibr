"""Run the Space's pipeline on one PDF without the UI, and report what each model did.

    python smoke_e2e.py paper.pdf [out.json]

Uses the same settings and in-process model shim as app.py. Off ZeroGPU,
``@spaces.GPU`` does nothing, so this runs on whatever torch device exists.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

started = time.monotonic()
import app  # noqa: E402 - loads the models
import zerogpu_models as zm  # noqa: E402

load_s = time.monotonic() - started


async def main(pdf: Path, out: Path) -> None:
    t0 = time.monotonic()
    paper = await app.get_pipeline().process_file(str(pdf))
    run_s = time.monotonic() - t0
    out.write_text(json.dumps(paper, ensure_ascii=False, indent=2))
    meta = paper.get("metadata") or {}
    refs = paper.get("bib") or []
    print(f"models loaded in {load_s:.0f}s; paper processed in {run_s:.0f}s")
    print("title:", meta.get("title"))
    print("authors:", len(paper.get("author") or []))
    print("references:", len(refs))
    for alias, s in zm.STATS.items():
        print(f"  {alias}: {s['requests']} requests in {s['batches']} batches, {s['seconds']:.1f}s generating")
    warnings = (paper.get("extraction") or {}).get("warnings") or []
    print("warnings:", sorted({w.get("code") for w in warnings if isinstance(w, dict)}))
    print(app.run_checks(str(out)))


if __name__ == "__main__":
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf.with_suffix(".bibr.json")
    asyncio.run(main(pdf, out))
