"""Data model of the lossless document layer (:data:`LAYER_VERSION`).

The layer keeps what bibr reads from a PDF and what it decides about it, so
later stages and models can use facts the string pipeline drops: glyph fonts,
sizes and baselines, superscripts, furniture, page render recipes.

Conventions:

- Geometry is in unrotated PDF user space (points, y up), with each page's
  ``crop_box`` and ``/Rotate`` kept on :class:`Page`. Convert to and from the
  layout frame with ``bibr.document.views.to_layout_bbox`` and
  ``from_layout_bbox``.
- Page indices are absolute and 0-based. Pages in other sources count from 1
  (``RegionSummary.page``, ``OutlineItem.page_no``); the layer never does.
- Glyph indices are pdfium char indices of the text page built after
  ``strip_furniture_objects`` ran, under the strip rules :data:`INDEX_FRAME`
  names.
- Per-page data is held as numpy columns (:class:`PageColumns`).
- Every derived fact carries a :class:`Decided` saying which rule or model
  made it.
- Ids are deterministic and do not move when a layer is built for another page
  range (:mod:`bibr.document.ids`): blocks ``p3.r12`` (page, post-OCR region
  index), lines ``p3.l40``, spans ``p3.sp210``, links ``p3.lk4`` (page,
  position among the page's link annotations), structure elements ``p3.st12``
  (page, position in that page's tree). An object of the whole PDF names no
  page: outline entries ``ol5`` (position in the outline, which is always
  read whole).

D1 fills the PDF-native part: glyphs, text objects, fonts, records, spans,
lines, superscript tags, furniture, render recipes and presence flags, plus
blocks attached from the post-OCR regions. D2 adds what the PDF declares
about its structure: link annotations with their targets, the structure
tree, the outline with the outline guard's verdict, page labels and the
presence flags that tell each of them apart from "absent". :class:`Suppressed`
and :class:`DecisionRecord` are declared for D3 (layout provenance) and stay
empty until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bibr.document import ids

# The layer's schema version: bump it with any change to a serialised class,
# its fields or COLUMN_DTYPES (tests/document/test_layer.py pins the pair).
LAYER_VERSION = "doclayer/3"
# The furniture strip's rule versions (bibr.ocr.native_text._find_furniture).
# The strip removes objects before the text page is built, so a rule change
# moves every glyph, span and line index on the pages it touches: bump the
# rule's version with it, and INDEX_FRAME, the key of those indexes, changes.
WATERMARK_RULE = "watermark/1"
LINE_NUMBER_RULE = "line_number/1"
INDEX_FRAME = f"post_strip:{WATERMARK_RULE}+{LINE_NUMBER_RULE}"
# The invisible-layer rule (bibr.ocr.native_text._is_invisible_text_layer_page)
# behind Page.text_source.
INVISIBLE_LAYER_RULE = "invisible_layer/1"

Box = tuple[float, float, float, float]


def as_box(values) -> Box:
    """*values* (four numbers) as a :data:`Box` of floats."""
    left, bottom, right, top = values
    return (float(left), float(bottom), float(right), float(top))


# --- PageColumns.gflags: one bit set per glyph fact ------------------------
# pdfium generated the char (an inferred space or line break).
GLYPH_GENERATED = 1 << 0
# pdfium marked a line-end hyphen (code 0x2, read as U+FFFE in the page text).
GLYPH_HYPHEN = 1 << 1
# The font's ToUnicode map failed for the char.
GLYPH_MAP_ERROR = 1 << 2
# The char's text object draws in an invisible render mode (3 or 7).
GLYPH_INVISIBLE_RENDER = 1 << 3
# pdfium leaves the char out of the page text (text index -1).
GLYPH_EXCLUDED = 1 << 4
# No tight box, loose box or origin could be read (the value is NaN).
GLYPH_NO_BOX = 1 << 5
GLYPH_NO_LOOSE_BOX = 1 << 6
GLYPH_NO_ORIGIN = 1 << 7

# --- PageColumns.obj_flags: values pdfium could not read for an object -----
OBJ_NO_MATRIX = 1 << 0
OBJ_NO_FONT_SIZE = 1 << 1
OBJ_NO_FILL = 1 << 2
OBJ_NO_STROKE = 1 << 3
# pdfium named the object for a char, but the walk of the page and its forms
# did not reach it, so its matrix is not composed with any form matrix.
OBJ_NOT_IN_WALK = 1 << 4

# The dtype of every PageColumns array, for validation and round trips. Tight
# char boxes and record centres stay float64: they are pdfium's doubles, and
# the views reproduce bibr's native text and line geometry from them exactly.
COLUMN_DTYPES: dict[str, str] = {
    # Raw stream: one row per pdfium char index.
    "cp": "<u4",
    "box": "<f8",
    "loose": "<f4",
    "origin": "<f4",
    "obj": "<i4",
    "gflags": "<u2",
    # Text objects: one row per object any char belongs to.
    "obj_font": "<i4",
    "obj_tf": "<f4",
    "obj_size_eff": "<f4",
    "obj_matrix": "<f4",
    "obj_fill": "<u4",
    "obj_stroke": "<u4",
    "obj_render_mode": "|i1",
    "obj_mcid": "<i4",
    "obj_artifact": "|b1",
    "obj_flags": "|u1",
    # Reading records: what _build_page_char_records returns, in order.
    "rec_cp": "<u4",
    "rec_cx": "<f8",
    "rec_cy": "<f8",
    "rec_newline": "|b1",
    "rec_src": "<i4",
    "rec_flags": "|u1",
    # Spans and lines over the records.
    "span_rec": "<i4",
    "span_obj": "<i4",
    "span_bbox": "<f4",
    "span_baseline": "<f4",
    "line_span": "<i4",
    "line_bbox": "<f4",
    "line_block": "<i4",
}


@dataclass(frozen=True, slots=True)
class Decided:
    """Provenance of a derived fact."""

    # layout_onnx | furniture.watermark | furniture.line_number | span_rules | ...
    component: str
    # Model sha or rule version, e.g. "superscript/1".
    version: str
    score: float | None = None
    # True only for a calibrated model probability. Rules, rule margins and
    # constant LLM scores keep the default.
    calibrated: bool = False
    # Layer ids the decision rests on.
    evidence: tuple[str, ...] = ()


@dataclass(slots=True)
class PageColumns:
    """One page's glyph, object, record, span and line columns.

    Raw stream (row = pdfium char index): ``cp`` the FPDFText_GetUnicode code,
    ``box`` the tight char box and ``loose`` the loose one (left, bottom,
    right, top), ``origin`` the glyph origin, ``obj`` the row of its text
    object (-1 for generated chars), ``gflags`` the ``GLYPH_*`` bits.

    Text objects (row = object): ``obj_font`` the index into
    :attr:`DocumentLayer.fonts` (-1 unknown), ``obj_tf`` the nominal Tf size,
    ``obj_size_eff`` the effective size Tf·√|det M| of the matrix composed
    through the forms that hold the object (use this one: Tf is 1 in many
    PDFs, with the scale in the matrix), ``obj_matrix`` that matrix's
    (a, b, c, d), ``obj_fill``/``obj_stroke`` RGBA packed as 0xRRGGBBAA,
    ``obj_render_mode`` (-1 unknown), ``obj_mcid`` the marked-content id
    (-1 none), ``obj_artifact`` inside /Artifact marked content, and
    ``obj_flags`` the ``OBJ_*`` bits.

    Records (row = what ``_build_page_char_records`` returned): the first
    code point of the char in ``rec_cp`` (a char of several code points also
    sits whole in ``rec_text``), the centre bibr's native text assigns by,
    ``rec_newline``, ``rec_src`` the pdfium index the record starts at (-1
    for a space the word-boundary repair inserted) and the ``REC_*`` flags of
    ``bibr.ocr.native_text``.

    Spans: ``span_rec`` the half-open record range, ``span_obj`` the object
    row of its first glyph, ``span_bbox`` the union of its glyphs' tight
    boxes and ``span_baseline`` its baseline along the text's up direction.
    Lines: ``line_span`` the half-open span range, ``line_bbox`` and
    ``line_block`` the index into :attr:`Page.blocks` (-1 outside every
    block, or before blocks are attached).
    """

    cp: np.ndarray
    box: np.ndarray
    loose: np.ndarray
    origin: np.ndarray
    obj: np.ndarray
    gflags: np.ndarray
    obj_font: np.ndarray
    obj_tf: np.ndarray
    obj_size_eff: np.ndarray
    obj_matrix: np.ndarray
    obj_fill: np.ndarray
    obj_stroke: np.ndarray
    obj_render_mode: np.ndarray
    obj_mcid: np.ndarray
    obj_artifact: np.ndarray
    obj_flags: np.ndarray
    rec_cp: np.ndarray
    rec_cx: np.ndarray
    rec_cy: np.ndarray
    rec_newline: np.ndarray
    rec_src: np.ndarray
    rec_flags: np.ndarray
    span_rec: np.ndarray
    span_obj: np.ndarray
    span_bbox: np.ndarray
    span_baseline: np.ndarray
    line_span: np.ndarray
    line_bbox: np.ndarray
    line_block: np.ndarray
    # Record index -> the record's char when it is not one code point.
    rec_text: dict[int, str] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in COLUMN_DTYPES)

    def record_char(self, index: int) -> str:
        text = self.rec_text.get(index)
        return text if text is not None else chr(int(self.rec_cp[index]))


@dataclass(frozen=True, slots=True)
class RenderRecipe:
    """How the layout stage rendered the page, so it can be rendered again.

    ``dpi`` is the DPI the layout render used (reduced for pages over the
    render budget; None when the page could not be rendered within it).
    ``flags`` names the pypdfium2 render options in effect. ``rgb_sha256``,
    the digest of the rendered RGB bytes, is filled once rendering from the
    recipe is wired up (D4). A re-render must open a fresh document: the
    furniture strip edits the inspected document's pages in memory.
    """

    pdfium: str
    dpi: int | None
    crop_box: Box
    rotation: int
    flags: tuple[str, ...]
    rgb_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Font:
    font_id: int
    base_name: str
    family: str
    # PDF font-descriptor flags (/Flags).
    pdf_flags: int
    weight: int
    italic_angle: int
    embedded: bool


@dataclass(frozen=True, slots=True)
class RoleTag:
    """A role given to a layer object (``target`` is its id)."""

    target: str
    # superscript | subscript (D1); running_header, page_number, ... later.
    role: str
    decided: Decided


@dataclass(frozen=True, slots=True)
class Furniture:
    """A page object removed before the text layer was read."""

    # p{page}.f{n}: the n-th object the furniture strip removed from the page.
    furniture_id: str
    page: int
    # watermark | line_number
    kind: str
    bbox_pdf: Box | None
    text: str
    decided: Decided


@dataclass(frozen=True, slots=True)
class Suppressed:
    """A layout box the layout post-processing removed (D3)."""

    page: int
    bbox_pdf: Box
    label: str
    score: float
    query: int | None
    # threshold | nms | full_page | containment | ocr_dedupe
    reason: str


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """A PDF outline (bookmark) entry, in document order (D2).

    ``idx`` is the entry's position in :attr:`DocumentLayer.outline` (its id is
    ``ol{idx}``), ``level`` the 0-based depth and ``parent`` the ``idx`` of the
    entry above. ``page`` is the 0-based target page (None when the entry names
    no page of this document) and ``x``, ``y`` the destination's position on it
    in PDF points, where the destination gives one.
    """

    idx: int
    parent: int | None
    level: int
    title: str
    page: int | None
    x: float | None
    y: float | None
    # The named destination the entry points at, if it uses one.
    dest_name: str | None

    @property
    def entry_id(self) -> str:
        return ids.outline_entry(self.idx)


@dataclass(frozen=True, slots=True)
class OutlineGuard:
    """The outline guard's verdict on the outline (D2).

    The rule version is ``decided.version``. ``decided.score`` is the share of
    the entries the guard kept that are printed on their target page, when it
    looked at the text layer (None otherwise); its evidence lists the entries
    that were not.
    """

    passed: bool
    # R1_too_few | R2_targets | R3_ungrounded; None when the outline passed.
    reject: str | None
    # (entry idx, rule) of each entry the cleanup drops before the verdict:
    # C1_blank | C2_page | C3_nav | C4_float | C5_title | C6_wrapper.
    dropped: tuple[tuple[int, str], ...]
    decided: Decided


@dataclass(frozen=True, slots=True)
class Link:
    """A link annotation and where it points (D2).

    ``link_id`` is ``p{page}.lk{n}``, ``n`` the link's position among the page's
    link annotations; a link that cannot be read leaves its number unused, so
    the others keep theirs. ``rect`` and ``quads`` (8 numbers each, the corners
    of the linked text) are in PDF points on the link's page. ``target_page``
    is the 0-based page an internal link lands on and ``target_xy`` the
    destination's position there in PDF points (a coordinate the destination
    leaves open is None); ``bibr.document.views.block_at`` finds the block it
    lands in once blocks are attached.
    """

    link_id: str
    page: int
    rect: Box
    quads: tuple[tuple[float, ...], ...]
    # dest (the annotation's /Dest) | goto | remote | uri | launch | other (an
    # action pdfium does not read) | none
    action: str
    # The URI of a URI action; the file a remote or launch action names.
    uri: str | None
    # The named destination the link points at, if it uses one, and where
    # the name came from: annot (the annotation's own /Dest) | table (the
    # document's named destination with the link's destination).
    dest_name: str | None
    name_source: str | None
    target_page: int | None
    target_xy: tuple[float | None, float | None] | None
    # bib | float | section | footnote | equation | other | external | unresolved
    target_class: str | None
    target: Decided | None
    # The spans of the text the link covers.
    source_span_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructElem:
    """An element of a tagged PDF's structure tree, as one page's tree holds it (D2).

    pdfium reads the tree page by page: an element with content on several
    pages, and each ancestor of such content, appears once per page and holds
    only that page's marked content. The id ``p{page}.st{n}`` numbers the
    page's elements in document order and ``parent`` is the element above on
    the same page. ``path`` is the element's place in the whole tree (its
    index among its parent's kids, from the root down); it is the same in
    every page's copy, so copies of one element can be grouped by it.

    ``role`` is the structure type after /RoleMap as pdfium resolves it (one
    mapping step). ``mcrs`` are the ``(page, mcid)`` pairs of the marked content
    the element holds directly: an mcid is unique only within its page's
    content stream, so the page is part of the key, and the text objects that
    carry it are those whose ``obj_mcid`` is that mcid on that page.
    """

    elem_id: str
    parent: str | None
    role: str
    mcrs: tuple[tuple[int, int], ...]
    page: int
    path: tuple[int, ...]
    alt: str | None
    actual: str | None
    lang: str | None


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """A decision with its full distribution (D3)."""

    subject: str
    decided: Decided
    chosen: str | None
    distribution: tuple[tuple[str, float], ...]
    extra: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Presence:
    """What the document has, so a consumer can tell "absent" from "not read".

    None means the fact was not read (D3 fills the rest). A fact no read page
    shows is None, not False, when some page could not be read.
    """

    has_text_layer: bool | None = None
    is_scan: bool | None = None
    has_invisible_layer: bool | None = None
    has_outline: bool | None = None
    outline_guard_pass: bool | None = None
    is_tagged: bool | None = None
    has_mcids: bool | None = None
    has_internal_links: bool | None = None
    has_named_dests: bool | None = None
    has_layout_provenance: bool | None = None
    # pdfium functions this pypdfium2 build lacks; the fields they feed are unread.
    missing_apis: tuple[str, ...] = ()


@dataclass(slots=True)
class Block:
    """A post-OCR region placed in the layer.

    Every region of a page becomes a block, in region order. ``region_index``
    is the region's position in the page's post-OCR region list, which region
    summaries carry as ``RegionSummary.index``, and the D1 id is
    ``p{page}.r{region_index}``. D3 moves the ids to the layout slot before
    OCR renumbering and fills the layout provenance and lineage.
    """

    block_id: str
    page: int
    region_index: int
    # None for a region without a layout box.
    bbox_pdf: Box | None
    label: str
    native_label: str
    layout: Decided | None = None
    class_topk: tuple[tuple[str, float], ...] = ()
    # Position in the page's region list (the regions' reading order).
    read_order: int | None = None
    # The post-OCR region text under the source it came from: native,
    # invisible_layer (the text layer of a scanned page) or ocr. Only the
    # chosen source is kept; bibr.document.views.block_text reads the text
    # layer under any block.
    text: dict[str, str] = field(default_factory=dict)
    # The key of ``text`` (None for an empty region).
    chosen: str | None = None
    # min_chars | usability | private_use | ineligible_label (D3)
    native_gate: str | None = None
    finish_reason: str | None = None
    # Indexes of the page lines assigned to the block (PageColumns.line_block).
    lines: tuple[int, ...] = ()
    source_block_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class Page:
    """One page of the layer.

    ``text_source`` describes the page's PDF text layer, not the source bibr
    used: ``native`` (born-digital), ``invisible_layer`` (a scan's hidden OCR
    layer) or ``ocr`` (no text layer). A native page whose text a corruption
    gate rejected (``min_printable_ratio``) was still OCR'd; ``Block.chosen``
    says which source each region used.

    A page the harvest failed on stays in the layer with the failure in
    ``error`` and ``cols`` None, and keeps what was read before the failure:
    its geometry, furniture and render recipe, and its ``text_source`` once
    decided (``unread`` before). A page that could not be opened or sized is
    ``unread`` with None geometry.
    """

    index: int
    # The page's label from the PDF's /PageLabels; None when it defines none.
    label: str | None
    # Geometry: None only for a page that could not be opened or sized.
    width: float | None
    height: float | None
    crop_box: Box | None
    rotation: int | None
    # native | invisible_layer | ocr | unread
    text_source: str
    # None when the page has no text layer, could not be read, or the
    # layer's columns were freed (DocumentLayer.columns_freed).
    cols: PageColumns | None
    blocks: list[Block] = field(default_factory=list)
    suppressed: list[Suppressed] = field(default_factory=list)
    furniture: list[Furniture] = field(default_factory=list)
    render: RenderRecipe | None = None
    # The invisible-layer rule's decision on a page with a text layer; its
    # score is the share of countable glyphs drawn invisibly (None when no
    # glyph counts).
    text_source_decided: Decided | None = None
    # The share of the CropBox that images cover, read when the share qualifies.
    image_coverage: float | None = None
    # Why the harvest failed on the page (also in DocumentLayer.component_errors).
    error: str | None = None


@dataclass(slots=True)
class DocumentLayer:
    version: str
    # pypdfium2 and pdfium versions the layer was read with.
    pdfium: str
    source_sha256: str
    index_frame: str
    pages: list[Page]
    fonts: list[Font]
    outline: list[OutlineEntry] = field(default_factory=list)
    # None when the outline could not be read, or the layer lacks the text of
    # some page to judge it by (see :mod:`bibr.document.outline_guard`).
    outline_guard: OutlineGuard | None = None
    links: list[Link] = field(default_factory=list)
    struct: list[StructElem] = field(default_factory=list)
    roles: list[RoleTag] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    presence: Presence = field(default_factory=Presence)
    # Failures of layer components, keyed like "harvest:3"; they never fail
    # the paper and never reach PdfInspection.component_errors.
    component_errors: dict[str, str] = field(default_factory=dict)
    # Set by free_columns: every page's cols is then None because the
    # columns were dropped, not because a page has no text layer or failed.
    columns_freed: bool = False

    def free_columns(self) -> None:
        """Drop every page's glyph columns, the bulk of the layer, and keep the rest.

        The pipeline calls it after the last stage that requires the layer.
        Blocks, furniture, roles, fonts and presence stay, and
        :attr:`columns_freed` records that the columns are gone.
        """
        for page in self.pages:
            page.cols = None
        self.columns_freed = True

    def page(self, index: int) -> Page | None:
        """The page with absolute 0-based *index*, if the layer covers it."""
        pages = self.pages
        # Pages are a contiguous run unless the layout skipped some.
        offset = index - pages[0].index if pages else -1
        if 0 <= offset < len(pages) and pages[offset].index == index:
            return pages[offset]
        for page in pages:
            if page.index == index:
                return page
        return None

    @property
    def nbytes(self) -> int:
        """Bytes held in numpy columns (the bulk of the layer)."""
        return sum(page.cols.nbytes for page in self.pages if page.cols is not None)
