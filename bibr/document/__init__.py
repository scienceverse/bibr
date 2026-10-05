"""Lossless document layer: what bibr reads from a PDF and decides about it.

The model lives in :mod:`bibr.document.model`, the harvest in
:mod:`bibr.document.harvest` (run by ``inspect_pdf`` with
``include_doc_layer``), the cache-hit rebuild and block attachment in
:mod:`bibr.document.rebuild`, read views in :mod:`bibr.document.views` and
the canonical serialisation in :mod:`bibr.document.serialize` and object ids
in :mod:`bibr.document.ids`.

What the harvest reads beyond the text lives in modules of its own:
:mod:`bibr.document.destinations` (named destinations and the pdfium string
and address helpers), :mod:`bibr.document.links` (link annotations and the
class of what they point to), :mod:`bibr.document.outline` (the bookmarks)
and :mod:`bibr.document.structure` (the structure tree). Their reads call
pdfium under the caller's ``pdfium_lock``; what is decided from the reads (the
link classes, the verdict of :mod:`bibr.document.outline_guard`) does not.

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
    Link,
    OutlineEntry,
    OutlineGuard,
    Page,
    PageColumns,
    Presence,
    RenderRecipe,
    RoleTag,
    StructElem,
)

__all__ = [
    "INDEX_FRAME",
    "LAYER_VERSION",
    "Block",
    "Decided",
    "DocumentLayer",
    "Font",
    "Furniture",
    "Link",
    "OutlineEntry",
    "OutlineGuard",
    "Page",
    "PageColumns",
    "Presence",
    "RenderRecipe",
    "RoleTag",
    "StructElem",
]
