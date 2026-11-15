"""A picklable inference API for the real-process worker contract tests.

The spawned worker imports this module by name, so it holds no test state: the
behaviour is chosen by the uploaded file's name, as a client would choose it.
"""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import HTTPException

#: Upload bytes larger than this are refused by the stub's descriptor read.
STUB_MAX_SIZE = 1024 * 1024


class StubBehaviour:
    """decode/predict/encode with one outcome per file name."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.setup_device: object = "unset"

    def setup(self, device) -> None:
        self.setup_device = device

    async def decode_request(self, request: dict) -> dict:
        from bibr.serve.ingress import consume_upload_descriptor

        name = request.get("filename")
        if name == "block.pdf":
            # Blocks the worker's event loop, as a synchronous stage would.
            time.sleep(2.0)
        if name == "die.pdf":
            os._exit(3)
        content, digest = consume_upload_descriptor(self.root, request, max_size=STUB_MAX_SIZE)
        if name == "http_dict.pdf":
            raise HTTPException(
                status_code=422, detail={"message": "bad paper", "error_code": "stub_code"}
            )
        if name == "http_str.pdf":
            raise HTTPException(status_code=400, detail="Filename required")
        if name == "crash.pdf":
            raise RuntimeError("internal detail that must not reach the client")
        return {
            "name": name,
            "size": len(content),
            "sha256": digest,
            "pid": os.getpid(),
            "options": {k: v for k, v in request.items() if k in ("refs", "mode")},
        }

    async def predict(self, inputs: dict) -> dict:
        if inputs["name"] == "slow.pdf":
            # Holds the request in flight without blocking the worker's loop.
            await asyncio.sleep(3.0)
        return inputs

    async def encode_response(self, output: dict):
        return output


class FailingSetupStub(StubBehaviour):
    """A stub whose model load fails."""

    def setup(self, device) -> None:
        raise RuntimeError("model bundle missing")
