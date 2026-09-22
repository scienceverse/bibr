"""Start reference enrichment's network prefetch while extraction still runs.

The enrich stage's up-front round-trips (resolver health probe and title
searches, Crossref bulk DOI lookup — see
:func:`bibr.enrich.references.prefetch_enrichment`) need nothing but the
parsed references, which are ready long before the extract stage's last LLM
calls (citation linking, structured integrity) finish. ``post_parse`` starts
that work as a task the moment references are parsed and hangs the handle on
the ``Paper`` (``paper.enrichment_prefetch``, runtime-only, never exported);
``CrossrefEnricher`` consumes it inside its own timeout budget, and every
path that does *not* enrich (per-request switch off, refs=off, an errored
file, a failed post-parse) cancels it so no task ever outlives its paper.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from bibr.config import GlobalSettings
    from bibr.enrich.references import EnrichmentPrefetch
    from bibr.paper import PaperReference

logger = logging.getLogger(__name__)

# asyncio only keeps weak references to tasks; hold ours until they finish so
# a prefetch whose paper was freed mid-flight is never garbage-collected while
# pending ("Task was destroyed but it is pending!").
_INFLIGHT: set[asyncio.Task[Any]] = set()


class PrefetchUnavailable(RuntimeError):
    """The prefetch task produced no result (it was cancelled or failed)."""


class EnrichmentPrefetchHandle:
    """A running/finished :func:`prefetch_enrichment` task plus its timing."""

    __slots__ = ("finished_at", "n_references", "started_at", "task")

    def __init__(
        self, task: asyncio.Task[EnrichmentPrefetch], *, started_at: float, n_references: int
    ) -> None:
        self.task = task
        self.started_at = started_at
        self.finished_at: float | None = None
        self.n_references = n_references
        task.add_done_callback(self._on_done)

    def _mark_finished(self) -> None:
        if self.finished_at is None:
            self.finished_at = time.monotonic()

    def _on_done(self, task: asyncio.Task[EnrichmentPrefetch]) -> None:
        self._mark_finished()
        _INFLIGHT.discard(task)
        if task.cancelled():
            return
        # Retrieve the exception so asyncio never logs "exception was never
        # retrieved" for a prefetch nobody awaited (its paper errored, or the
        # request switched enrichment off). The enricher reports its own copy.
        exc = task.exception()
        if exc is not None:
            logger.debug("Enrichment prefetch failed in the background: %s", exc)

    def done(self) -> bool:
        return self.task.done()

    @property
    def seconds(self) -> float | None:
        """Wall-clock seconds the prefetch took; ``None`` while still running."""
        if self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.started_at)

    def overlap(self, enrich_started_at: float) -> tuple[float, float]:
        """``(hidden, exposed)`` seconds of the prefetch relative to when the
        enrich stage began waiting on it. Fully hidden when it finished first."""
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        hidden = max(0.0, min(end, enrich_started_at) - self.started_at)
        exposed = max(0.0, end - max(self.started_at, enrich_started_at))
        return hidden, exposed

    def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()

    async def discard(self) -> None:
        """Cancel and wait for the task to settle; never raises on its behalf."""
        self.cancel()
        # ``asyncio.wait`` never re-raises the task's outcome; our own
        # cancellation (if any) still propagates from the await.
        await asyncio.wait([self.task])

    async def result(self) -> EnrichmentPrefetch:
        """Await the prefetch. A failed or independently cancelled prefetch
        raises :class:`PrefetchUnavailable` so the caller can fall back to the
        inline path; the *caller's own* cancellation (an enrich timeout) still
        propagates as ``CancelledError`` — and cancels the prefetch with it.

        Awaiting a task that already finished returns before its done
        callbacks have run, so the finish time is recorded here as well:
        ``seconds`` must be set once this returns.
        """
        try:
            return await self.task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            raise PrefetchUnavailable("enrichment prefetch was cancelled") from None
        except Exception as exc:
            raise PrefetchUnavailable(f"enrichment prefetch failed: {exc}") from exc
        finally:
            if self.task.done():
                self._mark_finished()


def start_enrichment_prefetch(
    references: Sequence[PaperReference], *, settings: GlobalSettings
) -> EnrichmentPrefetchHandle | None:
    """Kick off the prefetch for *references*; ``None`` when there is nothing to do."""
    if not references:
        return None
    from bibr.enrich.references import prefetch_enrichment

    started = time.monotonic()
    # Snapshot the list: the prefetch keys on the reference *objects*, which
    # the paper keeps, even if the metadata's list is later reassigned.
    task = asyncio.create_task(
        prefetch_enrichment(list(references), settings=settings),
        name=f"bibr-enrich-prefetch[{len(references)}]",
    )
    _INFLIGHT.add(task)
    return EnrichmentPrefetchHandle(task, started_at=started, n_references=len(references))


def prefetch_handle_of(paper: Any) -> EnrichmentPrefetchHandle | None:
    """The handle attached to *paper*, if any (tolerates test doubles)."""
    handle = getattr(paper, "enrichment_prefetch", None)
    return handle if isinstance(handle, EnrichmentPrefetchHandle) else None


def take_prefetch_handle(paper: Any) -> EnrichmentPrefetchHandle | None:
    """Detach and return the handle so exactly one consumer owns its outcome."""
    handle = prefetch_handle_of(paper)
    if handle is not None:
        paper.enrichment_prefetch = None
    return handle


def cancel_prefetch(paper: Any) -> bool:
    """Synchronously cancel *paper*'s prefetch (used where nothing can await)."""
    handle = prefetch_handle_of(paper)
    if handle is None:
        return False
    handle.cancel()
    return True


async def discard_prefetches(file_states: Iterable[Any]) -> int:
    """Cancel and settle every prefetch on the given file states' papers."""
    handles = []
    for fs in file_states:
        handle = take_prefetch_handle(getattr(fs, "paper", None))
        if handle is not None:
            handles.append(handle)
    for handle in handles:
        await handle.discard()
    return len(handles)


def cancel_leftover_prefetches(file_states: Iterable[Any]) -> int:
    """Cancel prefetches on papers that never reached the enrich stage."""
    return sum(1 for fs in file_states if cancel_prefetch(getattr(fs, "paper", None)))
