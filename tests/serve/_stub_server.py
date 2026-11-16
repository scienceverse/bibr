"""Run ``BibrServer`` over the stub worker, as ``bibr serve`` runs over the pipeline.

``python -m tests.serve._stub_server <port>``: the real uvicorn server, signal
handling and worker lifecycle, with the stub's file-name-chosen outcomes
instead of the models. Used by the real-process tests of ``BibrServer.run``.
"""

from __future__ import annotations

import sys


def main(port: int) -> None:
    import logging

    from fastapi import FastAPI

    from bibr.serve.app import BibrServer, _register_health_route
    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore, register_extract_route
    from bibr.serve.worker import InferenceWorker
    from tests.serve._worker_stub import STUB_MAX_SIZE, StubBehaviour

    logging.basicConfig(level=logging.INFO)
    store = UploadStore.create(
        max_size=STUB_MAX_SIZE, spool_memory_bytes=1 << 20, stale_after_seconds=600
    )
    worker = InferenceWorker(StubBehaviour(str(store.root)), timeout=30)
    app = FastAPI()
    tracker = InferenceDispatchTracker(dispatch=worker.dispatch, store=store)
    register_extract_route(app, store, tracker)
    _register_health_route(app, worker)
    try:
        BibrServer(app, worker).run(host="127.0.0.1", port=port, log_level="info")
    finally:
        store.close_sync()


if __name__ == "__main__":
    main(int(sys.argv[1]))
