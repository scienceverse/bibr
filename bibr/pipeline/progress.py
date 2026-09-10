"""Progress tracking for local pipeline.

Provides a callback protocol that ``LocalPipeline.process_chunk`` uses to
report stage transitions and per-region OCR progress.  The CLI supplies a
``RichProgress`` implementation; other callers can ignore it (``NullProgress``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Stage definitions — ordered list used for "[N/total] Stage..." display
# ---------------------------------------------------------------------------

STAGES: list[str] = [
    "validate",
    "docx",
    "jats",
    "html",
    "layout",
    "ocr",
    "parse",
    "extract",
    "enrich",
    "export",
]


def stages_for_files(stages: Iterable[str], files: Iterable[Path]) -> list[str]:
    """Omit input-specific work for formats absent from a chunk."""
    extensions = {path.suffix.lower() for path in files}
    input_stages = {
        "docx": {".docx"},
        "jats": {".xml"},
        "html": {".html", ".htm", ".epub"},
        "layout": {".pdf"},
        "ocr": {".pdf"},
    }
    return [
        stage for stage in stages if stage not in input_stages or extensions & input_stages[stage]
    ]


def _stage_label(name: str) -> str:
    """Human-readable label for a stage name."""
    return {
        "validate": "Checking file",
        "docx": "Reading DOCX",
        "jats": "Reading JATS XML",
        "html": "Reading HTML/ePub",
        "layout": "Analyzing layout",
        "ocr": "Reading text (OCR)",
        "parse": "Structuring content",
        "extract": "Extracting metadata",
        "enrich": "Looking up references",
        "export": "Writing output",
    }.get(name, name.title())


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProgressTracker(Protocol):
    def stage_start(self, name: str, detail: str = "") -> None: ...
    def stage_end(self, name: str) -> None: ...
    def ocr_start(self, total_regions: int) -> None: ...
    def ocr_region_done(self) -> None: ...
    def ocr_end(self) -> None: ...


# ---------------------------------------------------------------------------
# Null implementation (default — no output)
# ---------------------------------------------------------------------------


class NullProgress:
    """No-op progress tracker."""

    def stage_start(self, name: str, detail: str = "") -> None:
        pass

    def stage_end(self, name: str) -> None:
        pass

    def ocr_start(self, total_regions: int) -> None:
        pass

    def ocr_region_done(self) -> None:
        pass

    def ocr_end(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Rich implementation — log lines for stages, live bar for OCR
# ---------------------------------------------------------------------------


class RichProgress:
    """Stage log lines + live OCR progress bar using rich.

    Parameters
    ----------
    stages : list[str] | None
        Ordered stage names for this run.  If *None*, uses the full
        ``STAGES`` list.  Pass a filtered list when some stages are
        skipped (e.g. native input needs no OCR, no enrichment).
    """

    def __init__(self, stages: list[str] | None = None):
        from rich.console import Console

        self._console = Console(stderr=True)
        self._stages = STAGES if stages is None else stages
        self._stage_index: dict[str, int] = {s: i for i, s in enumerate(self._stages)}
        self._total_stages = len(self._stages)

        # Stage timing
        self._stage_start_time: float | None = None

        # OCR bar state
        from rich.progress import Progress, TaskID

        self._ocr_progress: Progress | None = None
        self._ocr_task_id: TaskID | None = None
        self._ocr_total = 0
        self._ocr_done = 0

    def stage_start(self, name: str, detail: str = "") -> None:
        idx = self._stage_index.get(name)
        if idx is None:
            return
        num = idx + 1
        label = _stage_label(name)
        suffix = f"  {detail}" if detail else ""
        self._console.print(f"  [dim]\\[{num}/{self._total_stages}][/dim] {label}...{suffix}")
        self._stage_start_time = time.monotonic()

    def stage_end(self, name: str) -> None:  # noqa: ARG002
        if self._stage_start_time is not None:
            elapsed = time.monotonic() - self._stage_start_time
            if elapsed >= 1.0:
                self._console.print(f"        [dim]{elapsed:.1f}s[/dim]")
            self._stage_start_time = None

    def ocr_start(self, total_regions: int) -> None:
        # No regions need OCR (everything filled from native text) — skip the
        # bar to avoid showing a confusing ``0/0 regions`` line.
        if total_regions <= 0:
            self._ocr_total = 0
            self._ocr_done = 0
            self._ocr_progress = None
            self._ocr_task_id = None
            return

        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

        self._ocr_total = total_regions
        self._ocr_done = 0
        self._ocr_progress = Progress(
            TextColumn("       "),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("regions"),
            console=self._console,
        )
        self._ocr_progress.start()
        self._ocr_task_id = self._ocr_progress.add_task("OCR", total=total_regions)

    def ocr_region_done(self) -> None:
        if self._ocr_progress is not None and self._ocr_task_id is not None:
            self._ocr_done += 1
            self._ocr_progress.update(self._ocr_task_id, completed=self._ocr_done)

    def ocr_end(self) -> None:
        if self._ocr_progress is not None:
            self._ocr_progress.stop()
            self._ocr_progress = None
            self._ocr_task_id = None
