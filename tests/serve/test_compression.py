"""Response compression on the serve layer.

bibr JSON is large (hundreds of KB to MB) and highly redundant — it gzips
~8x. The serve app must negotiate gzip so clients that send
``Accept-Encoding: gzip`` (httr2, requests, browsers) get the small body.
"""

import pytest

pytest.importorskip("fastapi")


def test_build_server_has_one_gzip_middleware():
    import asyncio

    from fastapi.middleware.gzip import GZipMiddleware

    from bibr.serve.app import build_server

    server = build_server()
    try:
        gzip_layers = [
            middleware
            for middleware in server.app.user_middleware
            if middleware.cls is GZipMiddleware
        ]
        assert len(gzip_layers) == 1
    finally:
        asyncio.run(server.app.state.inference_tracker.close())
        asyncio.run(server.app.state.upload_store.close())
