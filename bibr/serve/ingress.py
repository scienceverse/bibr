"""Private, disk-backed handoff for LitServe uploads."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import os
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette.datastructures import UploadFile
from starlette.requests import Request

from bibr.serve.admission import UploadAdmission

INTERNAL_INFERENCE_PATH = "/_bibr/inference"
PUBLIC_EXTRACT_PATH = "/papers/extract"
_CHUNK_SIZE = 64 * 1024
FORM_OPTION_MAX_BYTES = 64
_FORM_OPTION_NAMES = (
    "start_page",
    "end_page",
    "include_figures",
    "include_regions",
    "crossref",
    "consolidate",
    "refs",
    "ref_seg",
)
_VALID_REF_PARSE = frozenset({"ner", "llm", "llm-chunked", "off"})
_VALID_REF_SEG = frozenset({"geom", "region", "llm", "crf"})

MULTIPART_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "additionalProperties": False,
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "start_page": {"type": "string"},
                        "end_page": {"type": "string"},
                        "include_figures": {"type": "string"},
                        "include_regions": {"type": "string"},
                        "crossref": {
                            "type": "string",
                            "description": (
                                "Boolean (true/false, 1/0, yes/no): run Crossref/resolver "
                                "reference enrichment for this request. Omit to follow the "
                                "server's CROSSREF_ENRICH setting (off by default)."
                            ),
                        },
                        "consolidate": {"type": "string"},
                        "refs": {"type": "string"},
                        "ref_seg": {"type": "string"},
                    },
                }
            }
        },
    }
}

DispatchCallable = Callable[[dict[str, object]], Awaitable[Any]]


class UploadTooLargeError(ValueError):
    pass


class EmptyUploadError(ValueError):
    pass


class UploadStorageError(OSError):
    pass


class UploadIntegrityError(RuntimeError):
    pass


class InvalidMultipartError(ValueError):
    pass


class InvalidUploadOptionError(ValueError):
    pass


def configure_multipart_spooling(spool_memory_bytes: int) -> int:
    """Restore bibr's upload spool bound after LitServe imports Starlette."""
    from starlette.formparsers import MultiPartParser

    threshold = max(1, spool_memory_bytes)
    MultiPartParser.spool_max_size = threshold
    return threshold


@dataclass(frozen=True)
class StoredUpload:
    upload_id: str
    filename: str
    size: int
    sha256_hex: str

    def to_descriptor(self, fields: Mapping[str, object]) -> dict[str, object]:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256_hex,
            **{key: value for key, value in fields.items() if value is not None},
        }


@dataclass
class UploadStore:
    root: Path
    max_size: int
    spool_memory_bytes: int
    stale_after_seconds: int
    _leased_uploads: set[str] = field(default_factory=set, init=False, repr=False)
    _lease_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        max_size: int,
        spool_memory_bytes: int,
        stale_after_seconds: int,
    ) -> UploadStore:
        root = Path(tempfile.mkdtemp(prefix="bibr-serve-upload-"))
        root.chmod(0o700)
        return cls(
            root=root,
            max_size=max_size,
            spool_memory_bytes=max(1, spool_memory_bytes),
            stale_after_seconds=stale_after_seconds,
        )

    async def persist(self, upload: UploadFile) -> StoredUpload:
        await asyncio.to_thread(self.sweep_stale)
        filename = bounded_upload_filename(upload.filename)

        upload_id = uuid.uuid4().hex
        path = self.root / upload_id
        digest = hashlib.sha256()
        size = 0
        self._lease(upload_id)
        try:
            handle = await asyncio.to_thread(path.open, "xb")
            try:
                while chunk := await upload.read(_CHUNK_SIZE):
                    size += len(chunk)
                    if size > self.max_size:
                        raise UploadTooLargeError(f"upload exceeds {self.max_size} bytes")
                    digest.update(chunk)
                    await asyncio.to_thread(handle.write, chunk)
                if size == 0:
                    raise EmptyUploadError("empty upload")
            finally:
                await asyncio.to_thread(handle.close)
        except OSError as exc:
            await asyncio.to_thread(self._remove_canonical_id, upload_id)
            if _is_storage_quota_error(exc):
                raise UploadStorageError("insufficient temporary storage") from None
            raise
        except BaseException:
            await asyncio.to_thread(self._remove_canonical_id, upload_id)
            raise

        return StoredUpload(
            upload_id=upload_id,
            filename=filename,
            size=size,
            sha256_hex=digest.hexdigest(),
        )

    async def remove(self, upload_id: str) -> None:
        await asyncio.to_thread(self._remove_canonical_id, upload_id)

    async def close(self) -> None:
        await asyncio.to_thread(self.close_sync)

    def sweep_stale(self, now: float | None = None) -> None:
        current_time = time.time() if now is None else now
        for entry in self.root.iterdir():
            try:
                if self._is_leased(entry.name):
                    continue
                age = current_time - entry.stat(follow_symlinks=False).st_mtime
                if age > self.stale_after_seconds:
                    _remove_owned_entry(entry)
            except FileNotFoundError:
                continue

    def _lease(self, upload_id: str) -> None:
        with self._lease_lock:
            if self._closed:
                raise RuntimeError("upload store is closed")
            self._leased_uploads.add(upload_id)

    def _is_leased(self, upload_id: str) -> bool:
        with self._lease_lock:
            return upload_id in self._leased_uploads

    def _remove_canonical_id(self, upload_id: object) -> None:
        if _canonical_upload_id(upload_id) is None:
            return
        try:
            _remove_owned_entry(self.root / str(upload_id))
        finally:
            with self._lease_lock:
                self._leased_uploads.discard(str(upload_id))

    def close_sync(self) -> None:
        """Synchronously close the owned root, including during failed construction."""
        with self._lease_lock:
            if self._closed:
                return
            self._closed = True
            self._leased_uploads.clear()
        self._close_owned_root()

    def _close_owned_root(self) -> None:
        try:
            entries = tuple(self.root.iterdir())
        except FileNotFoundError:
            return
        for entry in entries:
            try:
                mode = entry.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                _remove_owned_entry(entry)
        self.root.rmdir()


def _is_storage_quota_error(exc: OSError) -> bool:
    quota_errors = {errno.ENOSPC}
    if hasattr(errno, "EDQUOT"):
        quota_errors.add(errno.EDQUOT)
    return exc.errno in quota_errors


def bounded_upload_filename(filename: object) -> str:
    """Return the only caller filename representation allowed into descriptors."""
    bounded = str(filename or "")[:255]
    if not bounded:
        raise EmptyUploadError("empty filename")
    return bounded


def _validate_upload_options(values: Mapping[str, str]) -> dict[str, str]:
    # FastAPI's prior Form(None) dependency normalized an empty optional text
    # part to None. Preserve that public behavior under explicit parsing.
    bounded = {name: value for name, value in values.items() if value != ""}
    for name, value in bounded.items():
        if len(value.encode("utf-8")) > FORM_OPTION_MAX_BYTES:
            raise InvalidUploadOptionError(f"{name} exceeds the {FORM_OPTION_MAX_BYTES}-byte limit")

    def _optional_int(name: str) -> int | None:
        value = bounded.get(name)
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except ValueError:
            raise InvalidUploadOptionError(f"{name} must be an integer") from None
        if parsed < 0:
            raise InvalidUploadOptionError(f"{name} must be >= 0")
        return parsed

    start_page = _optional_int("start_page")
    end_page = _optional_int("end_page")
    if start_page is not None and end_page is not None and start_page > end_page:
        raise InvalidUploadOptionError(
            f"start_page ({start_page}) must be <= end_page ({end_page})"
        )

    for name in ("include_figures", "include_regions", "crossref"):
        value = bounded.get(name)
        if value is None:
            continue
        if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
            raise InvalidUploadOptionError(f"{name} must be a boolean (true/false, 1/0, yes/no)")

    choices = {
        "consolidate": frozenset({"fill", "replace"}),
        "refs": _VALID_REF_PARSE,
        "ref_seg": _VALID_REF_SEG,
    }
    for name, valid in choices.items():
        value = bounded.get(name)
        if value is None:
            continue
        normalized = value.lower()
        if normalized not in valid:
            if name == "consolidate":
                detail = "consolidate must be 'fill' or 'replace'"
            else:
                detail = f"{name} must be one of: " + ", ".join(sorted(valid - {""}))
            raise InvalidUploadOptionError(detail)

    return bounded


@asynccontextmanager
async def parse_multipart_request(request: Request):
    """Yield one bounded upload/options pair and always release Starlette's spool."""
    from python_multipart.exceptions import MultipartParseError
    from starlette.formparsers import MultiPartException, MultiPartParser

    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "multipart/form-data":
        raise InvalidMultipartError("Invalid multipart body")

    parser = MultiPartParser(
        request.headers,
        request.stream(),
        max_files=1,
        max_fields=len(_FORM_OPTION_NAMES),
        max_part_size=FORM_OPTION_MAX_BYTES,
    )
    form = None
    try:
        try:
            try:
                form = await parser.parse()
            except (MultiPartException, MultipartParseError):
                raise InvalidMultipartError("Invalid multipart body") from None

            upload: UploadFile | None = None
            raw_options: dict[str, str] = {}
            for name, value in form.multi_items():
                if isinstance(value, UploadFile):
                    if name != "file" or upload is not None:
                        raise InvalidMultipartError("Unexpected multipart file")
                    upload = value
                    continue
                if name not in _FORM_OPTION_NAMES:
                    # FastAPI's prior Form(None) dependency dropped unrecognized
                    # text fields. Clients depend on that: the Platform worker
                    # always sends include_region_meta, which this route has
                    # never accepted. Ignore them — max_fields and max_part_size
                    # still bound what an unknown name can cost us.
                    continue
                if name in raw_options:
                    raise InvalidMultipartError("Duplicate multipart field")
                if not isinstance(value, str):
                    raise InvalidMultipartError("Invalid multipart field")
                raw_options[name] = value

            if upload is None:
                raise EmptyUploadError("empty filename")
            options = _validate_upload_options(raw_options)
            yield upload, options
        finally:
            try:
                if form is not None:
                    await form.close()
            finally:
                # Starlette 1.3.1 can return FormData after finalizing a
                # truncated part without adding its UploadFile to that form.
                # Close every still-open parser-owned spool on every exit.
                for spool in getattr(parser, "_files_to_close_on_error", ()):
                    if not spool.closed:
                        spool.close()
    except OSError as exc:
        if _is_storage_quota_error(exc):
            raise UploadStorageError("insufficient temporary storage") from None
        raise


async def persist_multipart_request(
    request: Request,
    store: UploadStore,
) -> tuple[StoredUpload, dict[str, str]]:
    """Parse and persist one upload, closing its original spool before return."""
    from bibr.serve.admission import release_spool_slot

    stored: StoredUpload | None = None
    try:
        async with parse_multipart_request(request) as (upload, options):
            stored = await store.persist(upload)
        # The body is on disk and Starlette's spool is closed, so this request
        # no longer occupies the resource the admission gate bounds. Hand the
        # slot back before the pipeline runs, or a saturated pipeline would
        # keep rejecting uploads the server has capacity to accept.
        release_spool_slot(request)
        return stored, options
    except BaseException:
        if stored is not None:
            await store.remove(stored.upload_id)
        raise


def resolve_litserve_dispatch(app) -> DispatchCallable:
    """Return LitServe's sole private inference endpoint closure."""
    matches = [
        route for route in app.routes if getattr(route, "path", None) == INTERNAL_INFERENCE_PATH
    ]
    if len(matches) != 1 or not callable(getattr(matches[0], "endpoint", None)):
        raise RuntimeError("LitServe private inference route contract changed")
    return matches[0].endpoint


class InferenceDispatchTracker:
    """Own inference tasks and their persisted uploads across waiter cancellation."""

    def __init__(self, *, dispatch: DispatchCallable, store: UploadStore) -> None:
        self._dispatch = dispatch
        self._store = store
        self._tasks: set[asyncio.Task[Any]] = set()
        self._owned_uploads: dict[asyncio.Task[Any], object] = {}
        self._closed = False

    @property
    def outstanding(self) -> int:
        return sum(not task.done() for task in self._tasks)

    async def submit(
        self,
        descriptor: dict[str, object],
        request_state=None,
        *,
        admission: UploadAdmission | None = None,
    ) -> Any:
        if self._closed:
            await self._store.remove(descriptor.get("upload_id"))
            raise RuntimeError("inference dispatch tracker is closed")

        async def _run() -> Any:
            try:
                return await self._dispatch(descriptor)
            finally:
                await self._store.remove(descriptor.get("upload_id"))

        task = asyncio.create_task(_run(), name="bibr-inference-dispatch")
        self._tasks.add(task)
        self._owned_uploads[task] = descriptor.get("upload_id")
        release_inflight = admission.detach_inflight() if admission is not None else None

        def _release_ownership(done_task: asyncio.Task[Any]) -> None:
            self._tasks.discard(done_task)
            self._owned_uploads.pop(done_task, None)
            if release_inflight is not None:
                release_inflight()

        task.add_done_callback(_release_ownership)
        if request_state is not None:
            request_state.inference_outstanding = self.outstanding
        return await asyncio.shield(task)

    async def discard(self, descriptor: Mapping[str, object]) -> None:
        """Remove an upload descriptor that will never be dispatched."""
        await self._store.remove(descriptor.get("upload_id"))

    async def wait_closed_tasks(self) -> None:
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = tuple(self._tasks)
        upload_ids = tuple(self._owned_uploads[task] for task in tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if upload_ids:
            cleanup_results = await asyncio.gather(
                *(self._store.remove(upload_id) for upload_id in upload_ids),
                return_exceptions=True,
            )
            for result in cleanup_results:
                if isinstance(result, BaseException):
                    raise result


def register_extract_route(
    app,
    store: UploadStore,
    tracker: InferenceDispatchTracker,
) -> None:
    """Mount the public multipart ingress route around private LitServe dispatch."""
    from fastapi.responses import JSONResponse

    @app.post(PUBLIC_EXTRACT_PATH, openapi_extra=MULTIPART_OPENAPI_EXTRA)
    async def extract(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ):
        try:
            stored, options = await persist_multipart_request(request, store)
            descriptor = stored.to_descriptor(options)
            request_id = getattr(request.state, "request_id", None)
            if request_id is not None:
                # Link the worker-side extract record back to this request's
                # per-request metering record (serve-8); the handoff ignores
                # unknown keys, and decode_request re-sanitizes the value.
                descriptor["request_id"] = request_id
            return await tracker.submit(
                descriptor,
                request_state=request.state,
                admission=getattr(request.state, "upload_admission", None),
            )
        except EmptyUploadError:
            return JSONResponse({"detail": "Empty or missing file"}, status_code=400)
        except InvalidMultipartError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except InvalidUploadOptionError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except UploadTooLargeError:
            max_mib = store.max_size / 1024 / 1024
            return JSONResponse(
                {"detail": f"File too large (>{max_mib:.0f}MB)"},
                status_code=413,
            )
        except UploadStorageError:
            return JSONResponse({"detail": "Insufficient temporary storage"}, status_code=507)


def _canonical_upload_id(upload_id: object) -> str | None:
    if not isinstance(upload_id, str):
        return None
    try:
        canonical_id = uuid.UUID(upload_id).hex
    except (ValueError, AttributeError):
        return None
    return canonical_id if canonical_id == upload_id else None


def _remove_owned_entry(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except IsADirectoryError:
        path.rmdir()
    except PermissionError:
        # macOS and Windows report EPERM/EACCES when unlink sees a directory.
        # Do not mistake a regular file's permission failure for a directory.
        if not path.is_dir() or path.is_symlink():
            raise
        path.rmdir()


def consume_upload_descriptor(
    root: str | Path,
    descriptor: Mapping[str, Any],
    *,
    max_size: int,
) -> tuple[bytes, str]:
    """Read and remove one verified, private upload handoff descriptor."""
    try:
        upload_id = descriptor["upload_id"]
    except (KeyError, TypeError):
        raise UploadIntegrityError("upload handoff validation failed") from None

    canonical_id = _canonical_upload_id(upload_id)
    if canonical_id is None:
        raise UploadIntegrityError("upload handoff validation failed")

    path = Path(root) / canonical_id
    try:
        try:
            filename = descriptor["filename"]
            expected_size = descriptor["size"]
            expected_digest = descriptor["sha256"]
        except (KeyError, TypeError):
            raise UploadIntegrityError("upload handoff validation failed") from None

        if not isinstance(filename, str) or not 0 < len(filename) <= 255:
            raise UploadIntegrityError("upload handoff validation failed")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 0 < expected_size <= max_size
        ):
            raise UploadIntegrityError("upload handoff validation failed")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(char not in "0123456789abcdef" for char in expected_digest)
        ):
            raise UploadIntegrityError("upload handoff validation failed")

        # Windows has no O_NOFOLLOW. Check the entry before opening and compare
        # its identity with the handle so a replaced entry cannot be consumed.
        entry_stat = path.lstat()
        if not stat.S_ISREG(entry_stat.st_mode):
            raise UploadIntegrityError("upload handoff validation failed")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            handle_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(handle_stat.st_mode) or not os.path.samestat(
                entry_stat, handle_stat
            ):
                raise UploadIntegrityError("upload handoff validation failed")
            digest = hashlib.sha256()
            content = bytearray()
            while chunk := handle.read(_CHUNK_SIZE):
                content.extend(chunk)
                digest.update(chunk)
                if len(content) > max_size:
                    raise UploadIntegrityError("upload handoff validation failed")
        actual_digest = digest.hexdigest()
        if len(content) != expected_size or not hmac.compare_digest(actual_digest, expected_digest):
            raise UploadIntegrityError("upload handoff validation failed")
        return bytes(content), actual_digest
    except UploadIntegrityError:
        raise
    except OSError:
        raise UploadIntegrityError("upload handoff validation failed") from None
    finally:
        try:
            _remove_owned_entry(path)
        except OSError:
            raise UploadIntegrityError("upload handoff validation failed") from None
