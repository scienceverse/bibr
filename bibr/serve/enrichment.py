"""Bounded JSON ingress for enrichment of saved paper exports."""

import asyncio
import json

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from bibr.enrich.backfill import ExportEnricher
from bibr.serve.admission import UploadAdmissionError, UploadAdmissionGate

# Bound memory before parsing a body or waiting on upstream HTTP. Independent
# from GPU extraction admission, so slow Crossref lookups do not reserve a GPU slot.
MAX_ENRICH_BODY_BYTES = 16 * 1024 * 1024
MAX_ENRICH_REFERENCES = 5000
MAX_ACTIVE_ENRICHMENTS = 4
ENRICH_BODY_TIMEOUT_SECONDS = 30


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def register_enrichment_route(app, settings) -> ExportEnricher:
    enricher = ExportEnricher(settings)
    gate = UploadAdmissionGate(MAX_ACTIVE_ENRICHMENTS)
    app.state.export_enricher = enricher

    @app.post(
        "/papers/enrich",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["paper"],
                            "additionalProperties": False,
                            "properties": {
                                "paper": {
                                    "type": "object",
                                    "description": "Saved bibr v10.6/v10.7 extraction JSON",
                                }
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Paper, replayable enrichment sidecar, version, key and status"
                },
                "413": {"description": "Body or bibliography exceeds the limit"},
                "422": {"description": "Invalid saved extraction"},
                "429": {"description": "Backfill concurrency limit reached"},
            },
        },
    )
    async def enrich(request: Request):  # pyright: ignore[reportUnusedFunction]
        try:
            with gate.admit():
                if (
                    request.headers.get("content-type", "").split(";", 1)[0].strip()
                    != "application/json"
                ):
                    raise HTTPException(415, "Expected application/json")
                body = bytearray()
                try:
                    async with asyncio.timeout(ENRICH_BODY_TIMEOUT_SECONDS):
                        async for chunk in request.stream():
                            if len(body) + len(chunk) > MAX_ENRICH_BODY_BYTES:
                                raise HTTPException(413, "Enrichment body exceeds 16 MiB")
                            body.extend(chunk)
                except TimeoutError as exc:
                    raise HTTPException(408, "Enrichment body upload timed out") from exc
                try:
                    data = json.loads(body, object_pairs_hook=_unique_object)
                    if (
                        not isinstance(data, dict)
                        or set(data) != {"paper"}
                        or not isinstance(data["paper"], dict)
                    ):
                        raise ValueError('Expected {"paper": <saved extraction JSON>}')
                    bibliography = data["paper"].get("bib")
                    if isinstance(bibliography, list) and len(bibliography) > MAX_ENRICH_REFERENCES:
                        raise HTTPException(413, "Bibliography exceeds 5000 references")
                    return JSONResponse(await enricher.enrich(data["paper"]))
                except ValidationError as exc:
                    # Pydantic's default errors contain uploaded input values.
                    return JSONResponse(
                        {
                            "detail": [
                                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                                for e in exc.errors(include_url=False)[:20]
                            ]
                        },
                        status_code=422,
                    )
                except (ValueError, UnicodeDecodeError) as exc:
                    raise HTTPException(422, str(exc)[:200]) from exc
                except RecursionError as exc:
                    raise HTTPException(422, "JSON nesting is too deep") from exc
        except UploadAdmissionError:
            return JSONResponse(
                {"detail": "Too many active enrichment requests"},
                status_code=429,
                headers={"Retry-After": "1"},
            )

    return enricher
