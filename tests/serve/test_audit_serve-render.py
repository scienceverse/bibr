"""``render_json_response`` renders exactly what FastAPI's default rendering would."""

from __future__ import annotations

import datetime as dt
import enum
import math
from pathlib import PurePosixPath

import pytest

pytest.importorskip("fastapi")

from fastapi.encoders import jsonable_encoder  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

from bibr.serve.ingress import render_json_response  # noqa: E402


class _Kind(enum.StrEnum):
    BODY = "body"


def _fastapi_body(content: object) -> bytes:
    return JSONResponse(jsonable_encoder(content)).body


@pytest.mark.parametrize(
    "content",
    [
        {"text": [{"id": 1, "text": "ü – ∑", "page": None, "score": 0.25}], "ok": True},
        {"rows": (1, 2, (3, "x"))},
        {"kind": _Kind.BODY, "n": 10**30, "neg": -0.0},
        [{"nested": {"deep": [[], {}, ""]}}],
    ],
)
def test_json_data_renders_the_same_bytes(content):
    assert render_json_response(content, "").body == _fastapi_body(content)


def test_values_json_cannot_encode_still_go_through_jsonable_encoder():
    content = {"when": dt.date(2024, 5, 1), "path": PurePosixPath("/a/b"), "tags": {"x"}}

    assert render_json_response(content, "").body == _fastapi_body(content)


def test_a_nan_is_refused_as_before():
    with pytest.raises(ValueError):
        _fastapi_body({"x": math.nan})
    with pytest.raises(ValueError):
        render_json_response({"x": math.nan}, "")
