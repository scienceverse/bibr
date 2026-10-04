"""Lossless document layer: what bibr reads from a PDF and decides about it.

The model lives in :mod:`bibr.document.model`, the harvest in
:mod:`bibr.document.harvest` (run by ``inspect_pdf`` with
``include_doc_layer``), the cache-hit rebuild and block attachment in
:mod:`bibr.document.rebuild`, read views in :mod:`bibr.document.views` and
the canonical serialisation in :mod:`bibr.document.serialize` and object ids
in :mod:`bibr.document.ids`.

The layer is internal: it is built only with ``pipeline.document_layer`` on,
changes no existing output, and is never exported (``bibr.export`` must not
import this package).
"""

from bibr.document.model import (
    INDEX_FRAME,
    LAYER_VERSION,
    Block,
    Decided,
    DocumentLayer,
    Font,
    Furniture,
    Page,
    PageColumns,
    Presence,
    RenderRecipe,
    RoleTag,
)

__all__ = [
    "INDEX_FRAME",
    "LAYER_VERSION",
    "Block",
    "Decided",
    "DocumentLayer",
    "Font",
    "Furniture",
    "Page",
    "PageColumns",
    "Presence",
    "RenderRecipe",
    "RoleTag",
]
