"""bibr — Scientific paper metadata extraction pipeline.

Public API
----------

For one-off use, :func:`chew` processes a paper in a single call:

.. code-block:: python

    import bibr

    result = bibr.chew("paper.pdf")
    result.title
    result.references.df

(Inside Jupyter or other running event loops: ``await bibr.achew(...)``.)

Directories and lists of paths run as a batch on one pipeline (models load
once) and return an order-aligned ``list[Result | ChewFailure]``:

.. code-block:: python

    results = bibr.chew("papers/")
    good = [r for r in results if r.ok]
    bibr.write_tables(good, "tables/")   # one Parquet file per table, keyed by paper_id

For repeated calls over time (notebooks, queue workers), :class:`Chewer`
keeps models warm across calls:

.. code-block:: python

    with bibr.Chewer(ocr="glm-rapid-mlx") as chewer:
        r1 = chewer.chew("a.pdf")
        r2 = chewer.chew("b.pdf")

For full control, drive :class:`LocalPipeline` directly:

.. code-block:: python

    import asyncio
    from bibr import LocalPipeline

    async def main():
        pipeline = LocalPipeline(memory_mode="balanced")
        try:
            return await pipeline.process_file("paper.pdf")
        finally:
            await pipeline.aclose()

    data = asyncio.run(main())

The returned dict matches the bibr v12.0 JSON schema (:func:`chew` wraps it in
a :class:`Result` view). Imports are lazy so the pipeline stages and ML deps
don't load at ``import bibr`` time.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bibr.api import (
        Chewer,
        ChewFailure,
        Records,
        Result,
        achew,
        achew_file,
        achew_many,
        chew,
        chew_file,
        chew_many,
    )
    from bibr.config import GlobalSettings, Settings
    from bibr.export import PaperExport
    from bibr.export.tables import write_tables
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.pipeline import Pipeline

try:
    __version__ = version("bibr")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "ChewFailure",
    "Chewer",
    "GlobalSettings",
    "LocalPipeline",
    "PaperExport",
    "Pipeline",
    "Records",
    "Result",
    "Settings",
    "__version__",
    "achew",
    "achew_file",
    "achew_many",
    "chew",
    "chew_file",
    "chew_many",
    "write_tables",
]

# One table drives the lazy API: module path per public name. ``__getattr__``
# imports on first access (so ``import bibr`` stays free of stage/ML deps)
# and ``__dir__`` reads ``__all__`` (so completion sees every name up front).
_LAZY = {
    "chew": "bibr.api",
    "achew": "bibr.api",
    "chew_file": "bibr.api",
    "achew_file": "bibr.api",
    "chew_many": "bibr.api",
    "achew_many": "bibr.api",
    "Result": "bibr.api",
    "Records": "bibr.api",
    "ChewFailure": "bibr.api",
    "Chewer": "bibr.api",
    "write_tables": "bibr.export.tables",
    "LocalPipeline": "bibr.local.pipeline",
    "Pipeline": "bibr.pipeline.pipeline",
    "PaperExport": "bibr.export",
    "Settings": "bibr.config",
    "GlobalSettings": "bibr.config",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
