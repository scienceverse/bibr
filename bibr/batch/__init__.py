"""``bibr batch`` — a resumable corpus runner with an append-only outcome ledger.

Public surface (lazy, so importing the package stays cheap):

* :func:`discover_inputs` / :func:`assign_paper_ids` — manifest, directory and
  file inputs → :class:`BatchItem` list with collision-free paper ids.
* :class:`Ledger` — ``outcomes.jsonl`` reader/appender and resume planning.
* :func:`run_batch` / :class:`BatchOptions` — the runner (local or remote).
* :class:`RemoteExecutor` / :class:`RemoteOptions` — the ``bibr serve`` job-API client.
* :func:`compute_report` / :func:`render_report` — ledger statistics.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BatchItem",
    "BatchOptions",
    "Ledger",
    "LocalOptions",
    "RemoteExecutor",
    "RemoteOptions",
    "assign_paper_ids",
    "compute_report",
    "discover_inputs",
    "render_report",
    "run_batch",
]

_LAZY = {
    "BatchItem": ("bibr.batch.manifest", "BatchItem"),
    "assign_paper_ids": ("bibr.batch.manifest", "assign_paper_ids"),
    "discover_inputs": ("bibr.batch.manifest", "discover_inputs"),
    "Ledger": ("bibr.batch.ledger", "Ledger"),
    "BatchOptions": ("bibr.batch.runner", "BatchOptions"),
    "LocalOptions": ("bibr.batch.runner", "LocalOptions"),
    "run_batch": ("bibr.batch.runner", "run_batch"),
    "RemoteExecutor": ("bibr.batch.remote", "RemoteExecutor"),
    "RemoteOptions": ("bibr.batch.remote", "RemoteOptions"),
    "compute_report": ("bibr.batch.report", "compute_report"),
    "render_report": ("bibr.batch.report", "render_report"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value
