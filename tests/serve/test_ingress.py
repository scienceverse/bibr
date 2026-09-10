import errno
import hashlib
import io
import os
import stat
import time

import pytest
from starlette.datastructures import UploadFile


def _upload(content: bytes, filename: str = "paper.pdf") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


@pytest.mark.parametrize("close_before_dispatch", [False, True])
async def test_cancelled_waiter_keeps_admission_until_dispatch_finishes(close_before_dispatch):
    import asyncio

    from bibr.serve.admission import UploadAdmissionError, UploadAdmissionGate
    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore

    entered = asyncio.Event()
    finish = asyncio.Event()

    async def dispatch(descriptor):
        entered.set()
        await finish.wait()
        return {"ok": True}

    gate = UploadAdmissionGate(1)
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    stored = await store.persist(_upload(b"paper"))

    async def submit():
        with gate.admit() as admission:
            admission.release_spool()
            return await tracker.submit(stored.to_descriptor({}), admission=admission)

    waiter = asyncio.create_task(submit())
    try:
        if close_before_dispatch:
            # submit runs first; its new dispatch task is queued after this
            # continuation, so close cancels the task before _run enters.
            await asyncio.sleep(0)
            await tracker.close()
        else:
            await asyncio.wait_for(entered.wait(), timeout=5)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert gate.inflight.active == 1
            with pytest.raises(UploadAdmissionError):
                with gate.admit():
                    pytest.fail("cancelled waiter released an active extraction slot")
            finish.set()
            await tracker.wait_closed_tasks()
        assert gate.spool.active == gate.inflight.active == 0
        assert list(store.root.iterdir()) == []
    finally:
        finish.set()
        await tracker.close()
        await asyncio.gather(waiter, return_exceptions=True)
        await store.close()


async def test_store_accepts_exact_limit_and_returns_path_free_descriptor():
    """Catches accepting a limit as exclusive or leaking a disk path downstream."""
    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    try:
        stored = await store.persist(_upload(b"x" * 10))
        descriptor = stored.to_descriptor({"refs": "off"})

        assert descriptor == {
            "upload_id": stored.upload_id,
            "filename": "paper.pdf",
            "size": 10,
            "sha256": hashlib.sha256(b"x" * 10).hexdigest(),
            "refs": "off",
        }
        assert "content" not in descriptor
        assert "path" not in descriptor
        assert (store.root / stored.upload_id).read_bytes() == b"x" * 10
    finally:
        await store.close()


async def test_store_rejects_one_byte_over_limit_and_removes_partial_file():
    """Catches retaining a partial file after the byte cap is crossed."""
    from bibr.serve.ingress import UploadStore, UploadTooLargeError

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    try:
        with pytest.raises(UploadTooLargeError):
            await store.persist(_upload(b"x" * 11))
        assert list(store.root.iterdir()) == []
    finally:
        await store.close()


async def test_store_rejects_empty_content_and_removes_created_file():
    """Catches accepting an empty PDF or leaving its empty owned entry behind."""
    from bibr.serve.ingress import EmptyUploadError, UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    try:
        with pytest.raises(EmptyUploadError, match="empty upload"):
            await store.persist(_upload(b""))
        assert list(store.root.iterdir()) == []
    finally:
        await store.close()


async def test_store_caps_filename_at_filesystem_safe_length():
    """Catches passing an attacker-controlled filename beyond common file limits."""
    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    try:
        stored = await store.persist(_upload(b"x", "a" * 5_000))
        assert stored.filename == "a" * 255
    finally:
        await store.close()


async def test_store_uses_canonical_uuid_hex_for_owned_entry_name():
    """Catches descriptors that could name paths outside the owned upload root."""
    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    try:
        stored = await store.persist(_upload(b"x"))
        assert len(stored.upload_id) == 32
        assert stored.upload_id == stored.upload_id.lower()
        assert all(char in "0123456789abcdef" for char in stored.upload_id)
    finally:
        await store.close()


@pytest.mark.skipif(os.name == "nt", reason="Windows uses ACLs, not POSIX permission bits")
async def test_store_root_is_owner_accessible_only():
    """Catches upload files being placed in a root readable by other users."""
    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    try:
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    finally:
        await store.close()


async def test_store_sanitizes_disk_quota_errors_and_removes_partial_file(monkeypatch):
    """Catches exposing storage internals or retaining data after ENOSPC."""
    from bibr.serve.ingress import UploadStorageError, UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    original_open = type(store.root).open

    class _NoSpaceWriter:
        def __init__(self, handle):
            self._handle = handle

        def write(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self):
            self._handle.close()

    def _open_with_no_space(path, *args, **kwargs):
        return _NoSpaceWriter(original_open(path, *args, **kwargs))

    monkeypatch.setattr(type(store.root), "open", _open_with_no_space)
    try:
        with pytest.raises(UploadStorageError, match="insufficient temporary storage"):
            await store.persist(_upload(b"x"))
        assert list(store.root.iterdir()) == []
    finally:
        await store.close()


async def test_sweep_stale_preserves_leased_upload_and_fresh_entries(tmp_path):
    """Catches orphan cleanup deleting a queued or dispatched upload it still owns."""
    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=10)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    try:
        leased = await store.persist(_upload(b"leased"))
        leased_path = store.root / leased.upload_id
        stale_path = store.root / ("1" * 32)
        stale_path.write_bytes(b"stale")
        fresh_path = store.root / ("2" * 32)
        fresh_path.write_bytes(b"fresh")
        stale_link = store.root / ("3" * 32)
        stale_link.symlink_to(outside)

        # Advance the sweep clock instead of changing a symlink's timestamp;
        # Windows does not support utime(..., follow_symlinks=False).
        now = time.time() + 11
        old = now - 11
        os.utime(leased_path, (old, old))
        os.utime(stale_path, (old, old))
        os.utime(fresh_path, (now, now))

        store.sweep_stale(now=now)

        assert leased_path.read_bytes() == b"leased"
        assert not stale_path.exists()
        assert fresh_path.read_bytes() == b"fresh"
        assert not stale_link.is_symlink()
        assert outside.read_bytes() == b"outside"
    finally:
        await store.close()


async def test_remove_releases_ownership_and_removes_only_canonical_entry():
    """Catches explicit cleanup retaining either the upload or its stale lease."""
    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=10)
    try:
        stored = await store.persist(_upload(b"owned"))
        path = store.root / stored.upload_id

        await store.remove(stored.upload_id)

        assert not path.exists()
        path.write_bytes(b"orphan")
        old = time.time() - 11
        os.utime(path, (old, old))
        store.sweep_stale(now=time.time())
        assert not path.exists()
    finally:
        await store.close()


async def test_close_removes_leased_entries_and_owned_root():
    """Catches shutdown leaving leased uploads or the private root behind."""
    from bibr.serve.ingress import UploadStore

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=10)
    root = store.root
    await store.persist(_upload(b"owned"))

    await store.close()

    assert not root.exists()


@pytest.mark.parametrize("upload_id", ["../escape", "A" * 32, "abc", "0" * 31])
def test_consumer_rejects_noncanonical_upload_ids(tmp_path, upload_id):
    """Catches a worker deriving a path from an untrusted descriptor ID."""
    from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor

    with pytest.raises(UploadIntegrityError):
        consume_upload_descriptor(
            tmp_path,
            {"upload_id": upload_id, "filename": "a.pdf", "size": 1, "sha256": "0" * 64},
            max_size=10,
        )


@pytest.mark.parametrize("content", [b"%PDF-1.4", b"%PDF-1.4\r\n\x1a\x00\xff"])
def test_consumer_reads_verifies_and_unlinks(tmp_path, content):
    """Catches retaining a valid descriptor file after successful consumption."""
    from bibr.serve.ingress import consume_upload_descriptor

    upload_id = "0" * 32
    path = tmp_path / upload_id
    path.write_bytes(content)
    descriptor = {
        "upload_id": upload_id,
        "filename": "a.pdf",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    actual, digest = consume_upload_descriptor(tmp_path, descriptor, max_size=100)

    assert actual == content
    assert digest == descriptor["sha256"]
    assert not path.exists()


def test_consumer_sanitizes_cleanup_failure_after_verified_content(tmp_path, monkeypatch):
    """Catches returning verified content when the owned entry could not be removed."""
    import bibr.serve.ingress as ingress

    upload_id = "6" * 32
    path = tmp_path / upload_id
    path.write_bytes(b"%PDF-1.4")

    def _fail_removal(_path):
        raise OSError(errno.EACCES, "cleanup denied")

    monkeypatch.setattr(ingress, "_remove_owned_entry", _fail_removal)

    with pytest.raises(ingress.UploadIntegrityError, match="upload handoff validation failed"):
        ingress.consume_upload_descriptor(
            tmp_path,
            {
                "upload_id": upload_id,
                "filename": "a.pdf",
                "size": 8,
                "sha256": "e16fa5d9b51928755db85b917f0297babaf22c7a47e97d9212adab56e61ba04e",
            },
            max_size=100,
        )


@pytest.mark.parametrize("without_nofollow", [False, True])
def test_consumer_rejects_symlink_and_unlinks_owned_link(tmp_path, monkeypatch, without_nofollow):
    """Catches following a substituted symlink outside the private upload root."""
    from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor

    if without_nofollow:
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    upload_id = "1" * 32
    path = tmp_path / upload_id
    target = tmp_path / "outside.pdf"
    target.write_bytes(b"%PDF-1.4")
    path.symlink_to(target)

    with pytest.raises(UploadIntegrityError):
        consume_upload_descriptor(
            tmp_path,
            {
                "upload_id": upload_id,
                "filename": "a.pdf",
                "size": 8,
                "sha256": "e16fa5d9b51928755db85b917f0297babaf22c7a47e97d9212adab56e61ba04e",
            },
            max_size=100,
        )

    assert not path.is_symlink()
    assert target.read_bytes() == b"%PDF-1.4"


def test_consumer_rejects_entry_replaced_while_opening(tmp_path, monkeypatch):
    from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor

    upload_id = "1" * 32
    path = tmp_path / upload_id
    path.write_bytes(b"%PDF-1.4")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"%PDF-1.4")
    original_open = os.open

    def replace_then_open(filename, flags):
        replacement.replace(path)
        return original_open(filename, flags)

    monkeypatch.setattr(os, "open", replace_then_open)
    with pytest.raises(UploadIntegrityError):
        consume_upload_descriptor(
            tmp_path,
            {
                "upload_id": upload_id,
                "filename": "a.pdf",
                "size": 8,
                "sha256": "e16fa5d9b51928755db85b917f0297babaf22c7a47e97d9212adab56e61ba04e",
            },
            max_size=100,
        )
    assert not path.exists()


def test_consumer_rejects_directory_replacement_and_removes_empty_owned_directory(tmp_path):
    """Catches treating a substituted directory as a regular upload file."""
    from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor

    upload_id = "2" * 32
    path = tmp_path / upload_id
    path.mkdir()

    with pytest.raises(UploadIntegrityError):
        consume_upload_descriptor(
            tmp_path,
            {
                "upload_id": upload_id,
                "filename": "a.pdf",
                "size": 1,
                "sha256": "0" * 64,
            },
            max_size=100,
        )

    assert not path.exists()


def test_consumer_rejects_size_mismatch_and_unlinks_owned_file(tmp_path):
    """Catches accepting descriptor metadata that does not match file bytes."""
    from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor

    upload_id = "3" * 32
    path = tmp_path / upload_id
    path.write_bytes(b"%PDF-1.4")

    with pytest.raises(UploadIntegrityError):
        consume_upload_descriptor(
            tmp_path,
            {
                "upload_id": upload_id,
                "filename": "a.pdf",
                "size": 7,
                "sha256": "e16fa5d9b51928755db85b917f0297babaf22c7a47e97d9212adab56e61ba04e",
            },
            max_size=100,
        )

    assert not path.exists()


def test_consumer_rejects_digest_mismatch_and_unlinks_owned_file(tmp_path):
    """Catches accepting bytes whose hash differs from the descriptor."""
    from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor

    upload_id = "4" * 32
    path = tmp_path / upload_id
    path.write_bytes(b"%PDF-1.4")

    with pytest.raises(UploadIntegrityError):
        consume_upload_descriptor(
            tmp_path,
            {"upload_id": upload_id, "filename": "a.pdf", "size": 8, "sha256": "0" * 64},
            max_size=100,
        )

    assert not path.exists()


def test_consumer_removes_canonical_file_when_descriptor_field_is_missing(tmp_path):
    """Catches cleanup being skipped after canonical ID validation succeeds."""
    from bibr.serve.ingress import UploadIntegrityError, consume_upload_descriptor

    upload_id = "5" * 32
    path = tmp_path / upload_id
    path.write_bytes(b"%PDF-1.4")

    with pytest.raises(UploadIntegrityError):
        consume_upload_descriptor(
            tmp_path,
            {"upload_id": upload_id, "filename": "a.pdf", "size": 8},
            max_size=100,
        )

    assert not path.exists()


def test_configure_multipart_spooling_restores_bound_after_litserve_import():
    """Catches LitServe resetting Starlette's spool threshold during import."""
    import litserve  # noqa: F401
    from starlette.formparsers import MultiPartParser

    from bibr.serve.ingress import configure_multipart_spooling

    assert configure_multipart_spooling(4096) == 4096
    assert MultiPartParser.spool_max_size == 4096
    assert configure_multipart_spooling(0) == 1
    assert MultiPartParser.spool_max_size == 1


async def test_cancelled_waiter_does_not_cancel_dispatch_or_leak_file():
    """Catches client cancellation propagating into inference or upload cleanup."""
    import asyncio

    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore

    entered = asyncio.Event()
    release = asyncio.Event()

    async def dispatch(descriptor):
        entered.set()
        await release.wait()
        return {"ok": True}

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    stored = await store.persist(_upload(b"payload"))
    waiter = asyncio.create_task(tracker.submit(stored.to_descriptor({})))
    try:
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert tracker.outstanding == 1
        assert (store.root / stored.upload_id).exists()

        release.set()
        await tracker.wait_closed_tasks()

        assert tracker.outstanding == 0
        assert not (store.root / stored.upload_id).exists()
    finally:
        release.set()
        await tracker.close()
        await store.close()


async def test_tracker_close_removes_upload_when_dispatch_task_never_started():
    """Catches cancellation before the dispatch coroutine can enter its finally."""
    import asyncio

    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore

    dispatch_started = False

    async def dispatch(_descriptor):
        nonlocal dispatch_started
        dispatch_started = True
        return {"ok": True}

    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    stored = await store.persist(_upload(b"payload"))
    waiter = asyncio.create_task(tracker.submit(stored.to_descriptor({})))
    try:
        # The waiter runs first and schedules its owned dispatch task behind this
        # test's continuation, giving close() a deterministic pre-start cancel.
        await asyncio.sleep(0)
        assert tracker.outstanding == 1
        assert dispatch_started is False

        await tracker.close()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert not (store.root / stored.upload_id).exists()
    finally:
        await tracker.close()
        await store.close()


@pytest.mark.parametrize("route_count", [0, 2])
def test_resolve_litserve_dispatch_requires_exactly_one_private_route(route_count):
    """Catches silently binding no private endpoint or an ambiguous duplicate."""
    from fastapi import FastAPI

    from bibr.serve.ingress import INTERNAL_INFERENCE_PATH, resolve_litserve_dispatch

    app = FastAPI()

    async def endpoint(descriptor: dict):
        return descriptor

    for _ in range(route_count):
        app.add_api_route(INTERNAL_INFERENCE_PATH, endpoint, methods=["POST"])

    with pytest.raises(RuntimeError, match="LitServe private inference route"):
        resolve_litserve_dispatch(app)


def test_resolve_litserve_dispatch_returns_registered_private_endpoint():
    """Catches wrapping or resolving a different route than LitServe registered."""
    from fastapi import FastAPI

    from bibr.serve.ingress import INTERNAL_INFERENCE_PATH, resolve_litserve_dispatch

    app = FastAPI()

    async def endpoint(descriptor: dict):
        return descriptor

    app.add_api_route(INTERNAL_INFERENCE_PATH, endpoint, methods=["POST"])

    assert resolve_litserve_dispatch(app) is endpoint


def test_extract_route_persists_descriptor_and_maps_upload_errors(monkeypatch):
    """Catches buffering bytes into dispatch or exposing raw storage errors."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStorageError,
        UploadStore,
        register_extract_route,
    )

    received = []

    async def dispatch(descriptor):
        received.append(descriptor)
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=10, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/papers/extract",
                files={"file": ("paper.pdf", b"0123456789", "application/pdf")},
                data={
                    "start_page": "1",
                    "end_page": "2",
                    "include_figures": "true",
                    "include_regions": "false",
                    "consolidate": "fill",
                    "refs": "llm",
                    "ref_seg": "geom",
                },
            )
            assert response.status_code == 200
            assert response.json() == {"ok": True}
            assert len(received) == 1
            descriptor = received[0]
            assert descriptor["filename"] == "paper.pdf"
            assert descriptor["size"] == 10
            assert "content" not in descriptor
            assert "path" not in descriptor
            assert descriptor["start_page"] == "1"
            assert descriptor["include_figures"] == "true"

            too_large = client.post(
                "/papers/extract",
                files={"file": ("paper.pdf", b"01234567890", "application/pdf")},
            )
            assert too_large.status_code == 413
            assert too_large.json() == {"detail": "File too large (>0MB)"}

            empty = client.post(
                "/papers/extract",
                files={"file": ("paper.pdf", b"", "application/pdf")},
            )
            assert empty.status_code == 400
            assert empty.json() == {"detail": "Empty or missing file"}

            empty_filename = client.post(
                "/papers/extract",
                files={"file": ("", b"x", "application/pdf")},
            )
            assert empty_filename.status_code == 400
            assert empty_filename.json() == {"detail": "Empty or missing file"}

            async def fail_persist(_upload):
                raise UploadStorageError("raw private disk detail")

            monkeypatch.setattr(store, "persist", fail_persist)
            no_storage = client.post(
                "/papers/extract",
                files={"file": ("paper.pdf", b"x", "application/pdf")},
            )
            assert no_storage.status_code == 507
            assert no_storage.json() == {"detail": "Insufficient temporary storage"}
            assert "raw private disk detail" not in no_storage.text
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


@pytest.mark.parametrize(
    "parts",
    [
        [
            ("file", ("first.pdf", b"first", "application/pdf")),
            ("file", ("second.pdf", b"second", "application/pdf")),
        ],
        [
            ("file", ("paper.pdf", b"paper", "application/pdf")),
            ("attachment", ("extra.pdf", b"extra", "application/pdf")),
        ],
        [
            ("file", ("paper.pdf", b"paper", "application/pdf")),
            ("refs", (None, "off")),
            ("refs", (None, "llm")),
        ],
        [
            ("file", ("paper.pdf", b"paper", "application/pdf")),
            # One more text part than the route accepts (len(_FORM_OPTION_NAMES)).
            *((f"extra_{index}", (None, "x")) for index in range(9)),
        ],
    ],
)
def test_extract_route_rejects_duplicate_or_extra_multipart_parts(parts):
    """Catches multipart fan-out retaining many spools or silently overriding options."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    received = []

    async def dispatch(descriptor):
        received.append(descriptor)
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/papers/extract",
            files=parts,
        )

        assert response.status_code == 400
        assert received == []
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


@pytest.mark.parametrize(
    "quota_errno",
    [errno.ENOSPC, *([errno.EDQUOT] if hasattr(errno, "EDQUOT") else [])],
)
def test_extract_route_maps_parser_disk_exhaustion_to_sanitized_507(
    monkeypatch,
    quota_errno,
):
    """Catches Starlette spool ENOSPC escaping before UploadStore can sanitize it."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    async def fail_write(_self, _data):
        raise OSError(quota_errno, "private spool path")

    monkeypatch.setattr(UploadFile, "write", fail_write)
    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=lambda _: None, store=store)
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"content", "application/pdf")},
        )

        assert response.status_code == 507
        assert response.json() == {"detail": "Insufficient temporary storage"}
        assert "private spool path" not in response.text
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


@pytest.mark.parametrize(
    "body",
    [
        b"garbage",
        b"--abc\r\nInvalid Header\r\n\r\nvalue\r\n--abc--\r\n",
        b'--wrong\r\nContent-Disposition: form-data; name="x"\r\n\r\nvalue\r\n--wrong--\r\n',
    ],
)
def test_extract_route_maps_native_multipart_parse_errors_to_400(body):
    """Catches python-multipart syntax errors escaping FastAPI as a 500."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    async def dispatch(_descriptor):
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/papers/extract",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=abc"},
        )

        assert response.status_code == 400
        # A malformed body must not masquerade as an empty upload: that
        # conflation sent a live extraction outage's diagnosis after the
        # caller's file rather than the caller's form.
        assert response.json() == {"detail": "Invalid multipart body"}
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


def test_native_multipart_parse_error_closes_partial_file_spool(monkeypatch):
    """Catches malformed bodies retaining a partially parsed Starlette spool."""
    import asyncio
    import tempfile

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette import formparsers

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    spools = []

    def capture_spool(*args, **kwargs):
        # Deliberately retain the handle so the assertion proves parser-owned
        # cleanup rather than CPython finalization.
        spool = tempfile.SpooledTemporaryFile(*args, **kwargs)  # noqa: SIM115
        spools.append(spool)
        return spool

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", capture_spool)

    async def dispatch(_descriptor):
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    body = (
        b"--abc\r\n"
        b'Content-Disposition: form-data; name="file"; filename="paper.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n"
        b"payload\r\n"
        b"--abc\r\n"
        b"Invalid Header\r\n\r\n"
        b"value\r\n"
        b"--abc--\r\n"
    )
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/papers/extract",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=abc"},
        )

        assert response.status_code == 400
        assert spools
        assert all(spool.closed for spool in spools)
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


def test_incomplete_file_part_successful_parse_closes_private_spool(monkeypatch):
    """Catches parser finalization returning without closing an incomplete file spool."""
    import asyncio
    import tempfile

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from starlette import formparsers

    from bibr.config import Settings
    from bibr.serve.auth import check_bearer
    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    spools = []
    parse_returned = []
    real_parse = formparsers.MultiPartParser.parse

    def capture_spool(*args, **kwargs):
        # Retain the handle so CPython finalization cannot satisfy the assertion.
        spool = tempfile.SpooledTemporaryFile(*args, **kwargs)  # noqa: SIM115
        spools.append(spool)
        return spool

    async def capture_successful_parse(parser):
        form = await real_parse(parser)
        parse_returned.append(form)
        return form

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", capture_spool)
    monkeypatch.setattr(formparsers.MultiPartParser, "parse", capture_successful_parse)
    monkeypatch.setattr(Settings.auth, "api_key", "authenticated-parser-regression-key")

    async def dispatch(_descriptor):
        return {"ok": True}

    app = FastAPI()

    @app.middleware("http")
    async def auth_gate(request, call_next):
        detail = check_bearer(request.headers.get("authorization"))
        if detail is not None:
            return JSONResponse({"detail": detail}, status_code=401)
        return await call_next(request)

    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    body = (
        b"--abc\r\n"
        b'Content-Disposition: form-data; name="file"; filename="paper.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n"
        b"payload without a terminating boundary"
    )
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/papers/extract",
            content=body,
            headers={
                "authorization": "Bearer authenticated-parser-regression-key",
                "content-type": "multipart/form-data; boundary=abc",
            },
        )

        assert response.status_code == 400
        assert parse_returned
        assert spools
        assert all(spool.closed for spool in spools)
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("start_page", "9" * 100_000, id="oversized-start-page"),
        ("refs", "bogus"),
    ],
)
def test_extract_route_rejects_invalid_options_before_descriptor_dispatch(field, value):
    """Catches unbounded or malformed raw options crossing the process queue."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    received = []

    async def dispatch(descriptor):
        received.append(descriptor)
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"content", "application/pdf")},
            data={field: value},
        )

        assert response.status_code == 400
        assert received == []
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


def test_extract_route_closes_starlette_upload_before_dispatch(monkeypatch):
    """Catches retaining Starlette's original spool throughout long inference."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    captured = {}
    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    real_persist = store.persist

    async def capture_upload(upload):
        captured["upload"] = upload
        return await real_persist(upload)

    async def dispatch(_descriptor):
        captured["closed_before_dispatch"] = captured["upload"].file.closed
        return {"ok": True}

    monkeypatch.setattr(store, "persist", capture_upload)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app).post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"content", "application/pdf")},
        )

        assert response.status_code == 200
        assert captured["closed_before_dispatch"] is True
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


@pytest.mark.parametrize("value", ["true", "False", "1", "0", "yes", "NO"])
def test_validate_upload_options_accepts_crossref_boolean_spellings(value):
    from bibr.serve.ingress import _validate_upload_options

    assert _validate_upload_options({"crossref": value}) == {"crossref": value}


def test_validate_upload_options_rejects_non_boolean_crossref():
    from bibr.serve.ingress import InvalidUploadOptionError, _validate_upload_options

    with pytest.raises(InvalidUploadOptionError, match="crossref must be a boolean"):
        _validate_upload_options({"crossref": "maybe"})


def test_validate_upload_options_drops_empty_crossref():
    """Absent/empty ``crossref`` means "server default", not false."""
    from bibr.serve.ingress import _validate_upload_options

    assert "crossref" not in _validate_upload_options({"crossref": ""})


def test_extract_route_passes_crossref_option_through():
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    received = []

    async def dispatch(descriptor):
        received.append(descriptor)
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/papers/extract",
                files={"file": ("paper.pdf", b"content", "application/pdf")},
                data={"crossref": "true"},
            )
            assert response.status_code == 200
            assert received[-1]["crossref"] == "true"

            rejected = client.post(
                "/papers/extract",
                files={"file": ("paper.pdf", b"content", "application/pdf")},
                data={"crossref": "sometimes"},
            )
            assert rejected.status_code == 400
            assert "crossref must be a boolean" in rejected.json()["detail"]
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


@pytest.mark.parametrize("field", ["include_figures", "include_regions", "crossref"])
def test_extract_route_treats_empty_optional_boolean_as_absent(field):
    """Catches explicit form parsing changing FastAPI Form(None) empty-value semantics."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    received = []

    async def dispatch(descriptor):
        received.append(descriptor)
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app).post(
            "/papers/extract",
            files=[
                ("file", ("paper.pdf", b"content", "application/pdf")),
                (field, (None, "")),
            ],
        )

        assert response.status_code == 200
        assert field not in received[0]
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


def test_extract_route_ignores_unrecognized_option_field():
    """Catches strict parsing 400ing clients that FastAPI Form(None) served fine.

    The Scienceverse Platform worker has always sent ``include_region_meta``,
    which this route never accepted; the FastAPI dependency dropped it silently.
    Rejecting it instead took every deployed extraction down.
    """
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    received = []

    async def dispatch(descriptor):
        received.append(descriptor)
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app).post(
            "/papers/extract",
            files=[
                ("file", ("paper.pdf", b"content", "application/pdf")),
                ("include_region_meta", (None, "true")),
                ("start_page", (None, "3")),
            ],
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert received[0]["start_page"] == "3"
        assert "include_region_meta" not in received[0]
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())


def test_extract_route_cleans_upload_if_tracker_rejects_before_task_creation():
    """Catches a closed tracker leaving a permanently leased persisted upload."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    async def dispatch(_descriptor):
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    asyncio.run(tracker.close())
    register_extract_route(app, store, tracker)
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/papers/extract",
            files={"file": ("paper.pdf", b"content", "application/pdf")},
        )

        assert response.status_code == 500
        assert list(store.root.iterdir()) == []
    finally:
        asyncio.run(store.close())


def test_extract_route_openapi_retains_multipart_contract():
    """Catches explicit parsing accidentally removing the public form schema."""
    import asyncio

    from fastapi import FastAPI

    from bibr.serve.ingress import (
        InferenceDispatchTracker,
        UploadStore,
        register_extract_route,
    )

    async def dispatch(_descriptor):
        return {"ok": True}

    app = FastAPI()
    store = UploadStore.create(max_size=100, spool_memory_bytes=4, stale_after_seconds=120)
    tracker = InferenceDispatchTracker(dispatch=dispatch, store=store)
    register_extract_route(app, store, tracker)
    try:
        operation = app.openapi()["paths"]["/papers/extract"]["post"]
        request_body = operation["requestBody"]
        assert request_body["required"] is True
        multipart = request_body["content"]["multipart/form-data"]["schema"]
        if "$ref" in multipart:
            multipart = app.openapi()["components"]["schemas"][multipart["$ref"].rsplit("/", 1)[-1]]
        assert multipart["required"] == ["file"]
        assert multipart["additionalProperties"] is False
        assert set(multipart["properties"]) == {
            "file",
            "start_page",
            "end_page",
            "include_figures",
            "include_regions",
            "crossref",
            "consolidate",
            "refs",
            "ref_seg",
        }
        assert multipart["properties"]["file"]["format"] == "binary"
        assert "CROSSREF_ENRICH" in multipart["properties"]["crossref"]["description"]
    finally:
        asyncio.run(tracker.close())
        asyncio.run(store.close())
