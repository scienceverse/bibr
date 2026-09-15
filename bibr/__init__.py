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

The returned dict matches the bibr v11.0 JSON schema (:func:`chew` wraps it in
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
    from bibr.document_api import DocumentRecord, DocumentResult, achew_document, chew_document
    from bibr.export import DocumentExport, PaperExport
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.pipeline import Pipeline

try:
    __version__ = version("bibr")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "ChewFailure",
    "Chewer",
    "DocumentExport",
    "DocumentRecord",
    "DocumentResult",
    "GlobalSettings",
    "LocalPipeline",
    "PaperExport",
    "Pipeline",
    "Records",
    "Result",
    "Settings",
    "__version__",
    "achew",
    "achew_document",
    "achew_file",
    "achew_many",
    "chew",
    "chew_document",
    "chew_file",
    "chew_many",
]


def __getattr__(name: str) -> Any:
    if name in ("DocumentRecord", "DocumentResult", "achew_document", "chew_document"):
        from bibr import document_api

        value = getattr(document_api, name)
        globals()[name] = value
        return value
    if name == "DocumentExport":
        from bibr.export import DocumentExport

        globals()[name] = DocumentExport
        return DocumentExport
    if name in (
        "chew",
        "achew",
        "chew_file",
        "achew_file",
        "chew_many",
        "achew_many",
        "Result",
        "Records",
        "ChewFailure",
        "Chewer",
    ):
        from bibr import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    if name == "LocalPipeline":
        from bibr.local.pipeline import LocalPipeline

        globals()["LocalPipeline"] = LocalPipeline
        return LocalPipeline
    if name == "Pipeline":
        from bibr.pipeline.pipeline import Pipeline

        globals()["Pipeline"] = Pipeline
        return Pipeline
    if name == "PaperExport":
        from bibr.export import PaperExport

        globals()["PaperExport"] = PaperExport
        return PaperExport
    if name == "Settings":
        from bibr.config import Settings

        globals()["Settings"] = Settings
        return Settings
    if name == "GlobalSettings":
        from bibr.config import GlobalSettings

        globals()["GlobalSettings"] = GlobalSettings
        return GlobalSettings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
