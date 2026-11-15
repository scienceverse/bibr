"""The ``bibr serve`` inference worker: one spawned process running the pipeline API.

The HTTP API process accepts uploads and hands each one to the worker as an
opaque descriptor (``bibr.serve.ingress``); the worker loads the models once
and runs the pipeline for many requests at a time on its own event loop. Only
descriptors and results cross the process boundary, over two one-way pipes:

- API → worker: ``("request", uid, enqueued_at, descriptor)`` and ``("stop",)``;
- worker → API: ``("ready", pid)`` or ``("setup_failed", reason)`` once, then
  ``(uid, reply)`` per request, where ``reply`` is ``("ok", result)``,
  ``("http_error", status, detail)`` or ``("error",)``.

A worker ``HTTPException`` keeps its status and detail; any other exception is
logged in the worker and answered with a generic 500, so its text never reaches
the client. A request the worker's loop picks up more than ``timeout`` seconds
after it was sent is answered 504 without running.

The API process watches the worker: when it exits, every request still waiting
on it is answered with a 503 (``inference_worker_stopped``) at once, and new
requests get a 503 until a worker is ready again. 503 because the requests in
flight are mostly bystanders of whatever killed the worker: ``bibr batch
--serve-url`` resubmits a 503 a bounded number of times, as it does a job whose
replica died, and takes a 500 as the paper's own failure. With ``restart`` a new
worker is started; without it (the default) ``on_fatal`` is called so the
server can stop and let its supervisor restart the service.

The worker is an API object with ``setup(device)``, ``decode_request``,
``predict`` and ``encode_response`` (sync or async), pickled into the child.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import multiprocessing
import multiprocessing.connection
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: The detail a request gets when the worker exits while running it.
WORKER_STOPPED_DETAIL = {
    "message": "The inference worker stopped while processing this request",
    "error_code": "inference_worker_stopped",
}
#: The detail a request gets when no worker is ready to take it.
WORKER_UNAVAILABLE_DETAIL = "Inference worker is not running"

#: How long ``stop`` waits for the worker to exit after asking it to, and again
#: after SIGTERM, before killing it.
_STOP_GRACE_SECONDS = 5.0


class WorkerStartError(RuntimeError):
    """The inference worker exited or failed its setup before it was ready."""


class InferenceWorker:
    """Own one spawned inference worker and the requests waiting on it."""

    def __init__(
        self,
        api: object,
        *,
        timeout: float | None,
        restart: bool = False,
        device: str | None = None,
    ) -> None:
        self._api = api
        self._timeout = timeout
        self._restart = restart
        self._device = device
        self.on_fatal: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future, int]] = {}
        self._process: Any = None
        self._requests: Any = None
        self._generation = 0
        self._ready = False
        self._stopping = False
        self._died = False
        self._reader: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def died(self) -> bool:
        """The worker exited unexpectedly and was not replaced."""
        return self._died

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None else None

    @property
    def outstanding(self) -> int:
        with self._lock:
            return len(self._pending)

    def start(self) -> None:
        """Spawn the worker and block until its setup has finished.

        Raises :class:`WorkerStartError` when the setup fails or the process
        exits first; the setup's own traceback is in the worker's log.
        """
        ctx = multiprocessing.get_context("spawn")
        request_reader, request_writer = ctx.Pipe(duplex=False)
        response_reader, response_writer = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_worker_main,
            args=(self._api, self._device, request_reader, response_writer, self._timeout),
            name="bibr-inference-worker",
        )
        process.start()
        # The child holds its own copies; closing ours lets each side see EOF
        # when the other exits.
        request_reader.close()
        response_writer.close()
        try:
            first = _await_first_message(response_reader, process)
        except BaseException:
            _end_process(process, request_writer)
            response_reader.close()
            raise
        if first[0] != "ready":
            _end_process(process, request_writer)
            response_reader.close()
            raise WorkerStartError(f"Inference worker setup failed: {first[1]}")
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._process = process
            self._requests = request_writer
            self._ready = True
            self._died = False
        self._reader = threading.Thread(
            target=self._read_replies,
            args=(response_reader, process, generation),
            name="bibr-inference-replies",
            daemon=True,
        )
        self._reader.start()
        logger.info("Inference worker ready (pid %d)", process.pid)

    async def dispatch(self, descriptor: dict[str, object]) -> Any:
        """Run one descriptor in the worker and return its encoded result.

        Raises ``HTTPException`` with the worker's status and detail, 500 for
        any other worker failure, and 503 when no worker is ready.
        """
        from fastapi import HTTPException

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        uid = uuid.uuid4().hex
        with self._lock:
            if not self._ready or self._stopping:
                raise HTTPException(status_code=503, detail=WORKER_UNAVAILABLE_DETAIL)
            self._pending[uid] = (loop, future, self._generation)
            requests = self._requests
        try:
            message = ("request", uid, time.monotonic(), dict(descriptor))
            try:
                await asyncio.to_thread(self._send, requests, message)
            except (OSError, EOFError, ValueError):
                raise HTTPException(status_code=503, detail=WORKER_UNAVAILABLE_DETAIL) from None
            reply = await future
        finally:
            with self._lock:
                self._pending.pop(uid, None)
        kind = reply[0]
        if kind == "ok":
            return reply[1]
        if kind == "http_error":
            raise HTTPException(status_code=reply[1], detail=reply[2])
        if kind == "worker_stopped":
            raise HTTPException(status_code=503, detail=WORKER_STOPPED_DETAIL)
        raise HTTPException(status_code=500, detail="Internal server error")

    def stop(self) -> None:
        """Ask the worker to exit, then terminate and kill it if it does not."""
        with self._lock:
            self._stopping = True
            self._ready = False
            process, requests = self._process, self._requests
        if process is not None:
            _end_process(process, requests, send_lock=self._send_lock)
        if self._reader is not None:
            self._reader.join(timeout=_STOP_GRACE_SECONDS)
        self._fail_pending(None)

    def _send(self, connection, message) -> None:
        with self._send_lock:
            connection.send(message)

    def _read_replies(self, connection, process, generation: int) -> None:
        try:
            while True:
                ready = multiprocessing.connection.wait([connection, process.sentinel])
                if connection not in ready and not connection.poll():
                    break
                try:
                    uid, reply = connection.recv()
                except (EOFError, OSError):
                    break
                self._resolve(uid, reply)
        finally:
            connection.close()
        self._worker_exited(process, generation)

    def _resolve(self, uid: str, reply: tuple) -> None:
        with self._lock:
            entry = self._pending.get(uid)
        if entry is None:
            return  # its waiter is gone
        loop, future, _ = entry
        loop.call_soon_threadsafe(_set_result, future, reply)

    def _fail_pending(self, generation: int | None) -> int:
        with self._lock:
            entries = [
                entry
                for entry in self._pending.values()
                if generation is None or entry[2] == generation
            ]
        for loop, future, _ in entries:
            try:
                loop.call_soon_threadsafe(_set_result, future, ("worker_stopped",))
            except RuntimeError:  # its loop is closed
                continue
        return len(entries)

    def _worker_exited(self, process, generation: int) -> None:
        process.join(timeout=_STOP_GRACE_SECONDS)
        with self._lock:
            if generation != self._generation:
                return
            self._ready = False
            stopping = self._stopping
        failed = self._fail_pending(generation)
        if stopping:
            return
        logger.error(
            "Inference worker (pid %s) exited with code %s; %d in-flight request(s) failed",
            process.pid,
            process.exitcode,
            failed,
        )
        if self._restart:
            try:
                self.start()
                return
            except Exception:
                logger.exception("Inference worker could not be restarted")
        with self._lock:
            self._died = True
        if self.on_fatal is not None:
            self.on_fatal()


def _set_result(future: asyncio.Future, reply: tuple) -> None:
    if not future.done():
        future.set_result(reply)


def _await_first_message(connection, process) -> tuple:
    while True:
        ready = multiprocessing.connection.wait([connection, process.sentinel])
        if connection in ready or connection.poll():
            try:
                message: tuple = connection.recv()
                return message
            except (EOFError, OSError):
                pass
        process.join()
        raise WorkerStartError(
            f"Inference worker exited during setup (exit code {process.exitcode})"
        )


def _end_process(process, requests, *, send_lock: threading.Lock | None = None) -> None:
    if requests is not None:
        # Under the dispatch threads' lock: two writers interleaved on one pipe
        # would corrupt both messages.
        with send_lock if send_lock is not None else contextlib.nullcontext():
            try:
                requests.send(("stop",))
            except (OSError, EOFError, ValueError):
                pass
    process.join(timeout=_STOP_GRACE_SECONDS)
    if process.is_alive():
        logger.warning("Inference worker (pid %s) did not stop; terminating it", process.pid)
        process.terminate()
        process.join(timeout=_STOP_GRACE_SECONDS)
    if process.is_alive():
        logger.warning("Inference worker (pid %s) did not terminate; killing it", process.pid)
        process.kill()
        process.join()
    if requests is not None:
        try:
            requests.close()
        except OSError:
            pass


# ---------------------------------------------------------------- the worker side


def _worker_main(api, device, requests, replies, timeout) -> None:
    """Entry point of the spawned worker process."""
    # Ctrl+C reaches the whole process group. The API process drains in-flight
    # requests on SIGINT and then stops this worker, so the worker must not die
    # first and fail the requests being drained.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        api.setup(device)
    except BaseException as exc:  # noqa: BLE001 - reported to the API process
        logger.exception("Inference worker setup failed")
        _send_quietly(replies, ("setup_failed", type(exc).__name__))
        replies.close()
        raise SystemExit(1) from None
    _send_quietly(replies, ("ready", os.getpid()))
    asyncio.run(_serve(api, requests, replies, timeout))


def _send_quietly(connection, message) -> bool:
    try:
        connection.send(message)
        return True
    except (OSError, EOFError, ValueError):
        return False


async def _serve(api, requests, replies, timeout) -> None:
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()
    send_lock = threading.Lock()
    tasks: set[asyncio.Task] = set()

    def receive() -> None:
        while True:
            try:
                message = requests.recv()
            except (EOFError, OSError):
                message = ("stop",)
            loop.call_soon_threadsafe(inbox.put_nowait, message)
            if message[0] == "stop":
                return

    def send(message) -> None:
        with send_lock:
            replies.send(message)

    async def reply(uid: str, answer: tuple) -> None:
        try:
            await asyncio.to_thread(send, (uid, answer))
        except (OSError, EOFError):
            return  # the API process is gone
        except Exception:  # noqa: BLE001 - an unpicklable result
            logger.exception("Inference request %s produced a result that cannot be sent", uid)
            await asyncio.to_thread(_send_locked_quietly, send, (uid, ("error",)))

    async def handle(uid: str, descriptor: dict) -> None:
        from fastapi import HTTPException

        try:
            decoded = await _call(api.decode_request, descriptor)
            predicted = await _call(api.predict, decoded)
            answer: tuple = ("ok", await _call(api.encode_response, predicted))
        except HTTPException as exc:
            answer = ("http_error", exc.status_code, exc.detail)
        except Exception:  # noqa: BLE001 - never reaches the client as text
            logger.exception("Inference request %s failed", uid)
            answer = ("error",)
        await reply(uid, answer)

    threading.Thread(target=receive, name="bibr-inference-requests", daemon=True).start()
    while True:
        message = await inbox.get()
        if message[0] == "stop":
            break
        _, uid, enqueued_at, descriptor = message
        if timeout and time.monotonic() - enqueued_at > timeout:
            logger.error(
                "Request %s waited %.0f s for the worker (timeout %s s); not run",
                uid,
                time.monotonic() - enqueued_at,
                timeout,
            )
            await reply(uid, ("http_error", 504, "Request timed out"))
            continue
        task = asyncio.create_task(handle(uid, descriptor), name=f"bibr-inference-{uid}")
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    for task in tuple(tasks):
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    replies.close()


def _send_locked_quietly(send, message) -> None:
    try:
        send(message)
    except (OSError, EOFError, ValueError):
        pass


async def _call(fn, value):
    result = fn(value)
    if inspect.isawaitable(result):
        result = await result
    return result
