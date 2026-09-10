"""ValidateStage — read file bytes, compute hash, check format/corruption/encryption."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class ValidateStage:
    name = "validate"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ()
    produces = ("pdf_bytes", "file_hash", "content_sha256")

    async def run(self, ctx: PipelineContext) -> None:
        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        from bibr.exceptions import InputValidationError
        from bibr.input.validate import validate_input_file

        for fs in ctx.file_states:
            try:
                path = fs.path
                loaded_content = fs.pdf_bytes
                loaded_hash = fs.content_sha256

                def _validate(path=path, loaded_content=loaded_content, loaded_hash=loaded_hash):
                    # Bytes may be pre-populated by the caller (e.g. serve API)
                    # to avoid a tmp-file roundtrip. Fall back to disk read.
                    content = loaded_content if loaded_content is not None else path.read_bytes()
                    # A caller-provided hash is trustworthy only when it
                    # accompanies those exact preloaded bytes (serve/upload
                    # path). Manifest disk paths are reopened here, so hash
                    # the bytes actually entering validation and identity.
                    digest = (
                        loaded_hash
                        if loaded_content is not None and loaded_hash is not None
                        else hashlib.sha256(content).hexdigest()
                    )
                    file_hash = digest[:16]
                    input_file = validate_input_file(path.name, content, file_hash=file_hash)
                    return content, digest, file_hash, input_file

                content, content_sha256, file_hash, input_file = await asyncio.to_thread(_validate)
                fs.pdf_bytes = content
                fs.content_sha256 = content_sha256
                fs.file_hash = file_hash
                fs.native_validation_artifact = input_file.native_artifact
                if not input_file.is_valid:
                    code = "unknown"
                    if not input_file.is_supported:
                        msg = "Unsupported file format"
                        code = "unsupported_format"
                    elif input_file.is_corrupted:
                        msg = "File appears to be corrupted or unreadable"
                        code = "corrupted_file"
                    elif input_file.is_encrypted:
                        msg = "File is password-protected"
                        code = "encrypted_file"
                    else:
                        msg = "File could not be validated"
                    fs.set_error(msg, code=code, stage="validate")
                    fs.free_all()
            except InputValidationError as e:
                # A rejected input is a client-side 4xx, not an internal
                # failure. Flattening it into code="unknown" cost the serve
                # layer that distinction: it reported kind="processing" for
                # what ``_translate_error`` already knows how to render as
                # ``input_validation``.
                fs.set_error(str(e), code="invalid_input", stage="validate", exc=e)
                fs.free_all()
                logger.info("Input rejected for %s: %s", fs.path.name, e)
            except Exception as e:  # noqa: BLE001
                fs.set_error(str(e), code="unknown", stage="validate", exc=e)
                fs.free_all()
                logger.warning("Validation failed for %s", fs.path.name, exc_info=True)

        logger.debug("Validate stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
