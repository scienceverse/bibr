"""bibr serve's real job routes with the pipeline replaced by a fixed export.

The plugin's client talks to these routes exactly as it would to a full
``bibr serve``, without the models: every upload "extracts" to the 12.3
reference fixture. Run from the repository root:

    uv run python integrations/zotero/test/contract_server.py --port 8765
    BIBR_CONTRACT_URL=http://127.0.0.1:8765 node --test integrations/zotero/test/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from bibr.config import Settings
from bibr.serve.ingress import UploadStore
from bibr.serve.jobs import MemoryJobStore, register_job_routes

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "references"
    / "enriched_check_12_3.json"
)


class FixedTracker:
    """Answers every descriptor with the fixture, as ``InferenceDispatchTracker`` would."""

    def __init__(self, result):
        self.result = result
        self.descriptors = []

    async def submit(self, descriptor, **_):
        self.descriptors.append(descriptor)
        return self.result

    async def discard(self, descriptor):
        pass


def build_app() -> FastAPI:
    app = FastAPI()
    upload_store = UploadStore.create(
        max_size=Settings.pipeline.max_file_size,
        spool_memory_bytes=1024 * 1024,
        stale_after_seconds=120,
    )
    tracker = FixedTracker(json.loads(FIXTURE.read_text()))
    register_job_routes(app, store=MemoryJobStore(), upload_store=upload_store, tracker=tracker)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port)
