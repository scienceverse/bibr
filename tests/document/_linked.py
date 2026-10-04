"""A hand-built PDF for what a document declares about its structure.

``linked_paper()`` is a six-page paper with page labels (roman, decimal, a
prefixed range and an empty one), named destinations in the /Names tree and in
the catalog's old-style /Dests, and the pages' text as the targets the
destinations point at: a figure and a table caption, a reference list, a
numbered equation and a section heading. The outline, the link annotations and
the structure tree are added by the builders below as the layer reads them.

The structure tree tags every marked-content sequence of the pages: a custom
heading type that /RoleMap turns into H1, a paragraph that crosses a page
break, an element with alternate and actual text and a language, a figure
whose marked content draws nothing, a table, and a last page with no tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.document._pdfs import PAGE_H, PAGE_W, marked, serialize_pdf, text
from tests.ocr.test_watermark_text import _width

N_PAGES = 6
# The label of each page (``/PageLabels``): lower-case roman, two decimals, a
# prefixed decimal range and an empty one.
PAGE_LABELS = ["i", "1", "2", "3", "A-1", ""]

BODY = 11.0
HEAD = 14.0
BLUE = (0.0, 0.0, 1.0)

FIGURE_CAPTION = "Figure 1: Overview of the linked pipeline."
TABLE_CAPTION = "Table 1: Results of the linked study."
REFERENCES = [
    "[1] Smith, J. (2020). A study of linked examples. Journal of Examples, 12, 1-9.",
    "[2] Doe, A. and Roe, B. (2019). Another study of links. Annals of Examples, 4, 10-20.",
]


@dataclass
class Table:
    """PDF objects numbered as they are added; object 1 is the catalog, 2 the page tree."""

    bodies: list[bytes] = field(default_factory=lambda: [b"", b""])

    def add(self, body: bytes = b"") -> int:
        self.bodies.append(body)
        return len(self.bodies)

    def fill(self, number: int, body: bytes) -> None:
        self.bodies[number - 1] = body


@dataclass(frozen=True)
class Placed:
    """A run of text drawn at (x, y), and the box it covers."""

    string: str
    x: float
    y: float
    size: float

    @property
    def right(self) -> float:
        return self.x + _width(self.string, self.size)

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x, self.y - 3.0, self.right, self.y + self.size)

    @property
    def quad(self) -> tuple[float, ...]:
        left, bottom, right, top = self.rect
        return (left, top, right, top, left, bottom, right, bottom)


def line_chunks(
    parts: list[str], x: float, y: float, linked=(), size: float = BODY
) -> tuple[list[bytes], list[Placed]]:
    """*parts* drawn one text object each, side by side from x; each part's content and place.

    The parts at the indexes in *linked* are drawn blue, as links are, which
    also makes each one a span of its own.
    """
    chunks = []
    placed = []
    for index, part in enumerate(parts):
        run = Placed(part, x, y, size)
        chunks.append(text(part, x, y, size=size, rgb=BLUE if index in linked else None))
        placed.append(run)
        x = run.right
    return chunks, placed


def line(
    parts: list[str], x: float, y: float, linked=(), size: float = BODY
) -> tuple[bytes, list[Placed]]:
    """:func:`line_chunks` as one content stream."""
    chunks, placed = line_chunks(parts, x, y, linked, size)
    return b"".join(chunks), placed


def _name(value: str) -> bytes:
    return b"(" + value.encode("latin-1") + b")"


def _array(*items: bytes) -> bytes:
    return b"[" + b" ".join(items) + b"]"


def _numbers(values) -> bytes:
    return b" ".join(f"{value:.2f}".encode() for value in values)


# --- Page contents ---------------------------------------------------------------

# Where each named destination points: (page index, view), the view being the
# destination array after the page. An XYZ y sits just above the text it names,
# as hyperref puts it. The ``view.*`` ones leave coordinates open in each way
# the PDF destination types allow.
DESTS = {
    "figure.1": (3, "/XYZ 72 662 null"),
    "table.1": (3, "/XYZ 72 582 null"),
    "cite.smith2020": (4, "/XYZ 72 704 null"),
    "section.1": (1, "/XYZ 72 724 null"),
    "section.2": (2, "/XYZ 72 722 null"),
    "equation.3": (2, "/XYZ 72 252 null"),
    "Hfootnote.1": (0, "/XYZ 72 110 null"),
    # The census treats a name like this one as a page number, not as a name to classify by.
    "page.4": (3, "/XYZ 72 582 null"),
    "old.dest": (1, "/XYZ 72 700 null"),
    "view.fit": (1, "/Fit"),
    "view.fitb": (1, "/FitB"),
    "view.fith": (1, "/FitH 500"),
    "view.fitbh": (1, "/FitBH 510"),
    "view.fitv": (1, "/FitV 100"),
    "view.fitbv": (1, "/FitBV 110"),
    "view.fitr": (1, "/FitR 10 20 300 400"),
    "view.xyz_x": (1, "/XYZ 55 null null"),
    "view.xyz_y": (1, "/XYZ null 66 null"),
    "view.xyz_open": (1, "/XYZ null null null"),
}
# A name written out of order in the /Names tree. pdfium's lookup by name stops at the
# first name that sorts after the one it wants, so it never finds this one; the table
# of named destinations lists it.
UNSORTED = {"aaa.unsorted": (3, "/XYZ 72 582 null")}
# The position each of them gives (x, y); None is a coordinate the view leaves open.
DEST_XY = {
    "figure.1": (72.0, 662.0),
    "table.1": (72.0, 582.0),
    "cite.smith2020": (72.0, 704.0),
    "section.1": (72.0, 724.0),
    "section.2": (72.0, 722.0),
    "equation.3": (72.0, 252.0),
    "Hfootnote.1": (72.0, 110.0),
    "page.4": (72.0, 582.0),
    "old.dest": (72.0, 700.0),
    "view.fit": (None, None),
    "view.fitb": (None, None),
    "view.fith": (None, 500.0),
    "view.fitbh": (None, 510.0),
    "view.fitv": (100.0, None),
    "view.fitbv": (110.0, None),
    "view.fitr": (10.0, 400.0),
    "view.xyz_x": (55.0, None),
    "view.xyz_y": (None, 66.0),
    "view.xyz_open": (None, None),
    "aaa.unsorted": (72.0, 582.0),
}
FILLER_LINES = 30
FILLER = "The study reads every page of the paper and reports what the pages declare."


def _pages() -> tuple[list[bytes], dict[str, Placed]]:
    """The six pages' content streams and the runs the links are drawn over."""
    runs: dict[str, Placed] = {}

    def draw(key_parts: dict[int, str], parts: list[str], x: float, y: float):
        content, placed = line(parts, x, y, linked=set(key_parts))
        for index, key in key_parts.items():
            runs[key] = placed[index]
        return content

    def draw_marked(key_parts: dict[int, str], parts: list[str], x: float, y: float, marks: list):
        """:func:`draw` with each part in a marked-content sequence of its own: *marks* are (tag, mcid)."""
        chunks, placed = line_chunks(parts, x, y, linked=set(key_parts))
        for index, key in key_parts.items():
            runs[key] = placed[index]
        return b"".join(
            marked(tag, chunk, mcid=mcid) for chunk, (tag, mcid) in zip(chunks, marks, strict=True)
        )

    # Page 0, label "i": the abstract, with the links of every kind.
    out = marked("Artifact", text("Linked Paper Fixture 2026", 72.0, 760.0, size=8.0))
    out += marked("Heading", text("Abstract", 72.0, 700.0, size=HEAD, font="F3"), mcid=0)
    out += marked(
        "P",
        draw(
            {1: "fig_ref", 3: "cite_ref", 5: "sec_ref"},
            [
                "This paper links a figure: see ",
                "Figure 1",
                ", ",
                "Smith et al. [1]",
                " and ",
                "Section 2",
                ".",
            ],
            72.0,
            676.0,
        ),
        mcid=1,
    )
    out += marked(
        "P",
        draw(
            {1: "note_ref", 3: "eq_ref", 5: "doi_ref"},
            ["Also ", "footnote 1", " and ", "equation (3)", " and the ", "DOI", " link."],
            72.0,
            658.0,
        ),
        mcid=2,
    )
    out += marked(
        "P",
        draw(
            {1: "remote_ref", 3: "launch_ref", 5: "named_ref"},
            ["Elsewhere: ", "supplement", ", ", "a program", ", ", "next page", "."],
            72.0,
            640.0,
        ),
        mcid=3,
    )
    out += marked(
        "P",
        draw(
            {1: "broken_ref", 3: "bare_ref", 5: "old_ref"},
            ["Odd ones: ", "broken", ", ", "bare", ", ", "old style", "."],
            72.0,
            622.0,
        ),
        mcid=4,
    )
    out += marked(
        "P",
        draw(
            {1: "array_bib_ref", 3: "array_fig_ref", 5: "fit_ref"},
            ["By array: ", "[1]", " and ", "Fig. 1", " and ", "page four", "."],
            72.0,
            604.0,
        ),
        mcid=5,
    )
    out += text("Supplement", 72.0, 300.0)
    out += marked(
        "P", text("1 This footnote is read after the page.", 72.0, 100.0, size=8.0), mcid=6
    )
    first = out

    # Page 1, label "1": the introduction.
    out = marked("Artifact", text("Linked Paper Fixture 2026", 72.0, 760.0, size=8.0))
    out += marked("Heading", text("1 Introduction", 72.0, 720.0, size=HEAD, font="F3"), mcid=0)
    # Three links, each text in its own marked content under a Link element.
    out += draw_marked(
        {1: "table_ref", 3: "second_ref", 5: "late_ref"},
        ["The introduction reads ", "Table 1", " and ", "Smith [1]", " and ", "late", "."],
        72.0,
        700.0,
        [("P", 1), ("Link", 5), ("P", 8), ("Link", 6), ("P", 9), ("Link", 7), ("P", 10)],
    )
    # A link over two lines: one run on each.
    out += marked(
        "P",
        draw({1: "wrap_a"}, ["It continues, as shown in the ", "first line of"], 72.0, 682.0)
        + draw({0: "wrap_b"}, ["the second figure", ", and ends."], 72.0, 666.0),
        mcid=2,
    )
    out += marked("H2", text("1.1 Background", 72.0, 640.0, size=12.0, font="F3"), mcid=3)
    out += marked(
        "P", text("Background text that goes on to the next page and", 72.0, 620.0), mcid=4
    )
    second = out

    # Page 2, label "2": the methods; the filler brings the text layer over the
    # outline guard's 2,000 letters and digits.
    out = marked("Artifact", text("Linked Paper Fixture 2026", 72.0, 760.0, size=8.0))
    out += marked("P", text("finishes here, then Methods begin.", 72.0, 740.0), mcid=0)
    out += marked("Heading", text("2 Methods", 72.0, 710.0, size=HEAD, font="F3"), mcid=1)
    for index in range(FILLER_LINES):
        out += marked("P", text(FILLER, 72.0, 690.0 - 14.0 * index, size=10.0), mcid=2 + index)
    out += marked("P", text("E = mc2 (3)", 72.0, 240.0), mcid=2 + FILLER_LINES)
    third = out

    # Page 3, label "3": the results with the figure and the table.
    out = marked("Artifact", text("Linked Paper Fixture 2026", 72.0, 760.0, size=8.0))
    out += marked("Heading", text("3 Results", 72.0, 720.0, size=HEAD, font="F3"), mcid=0)
    out += marked("Caption", text(FIGURE_CAPTION, 72.0, 644.0, size=10.0), mcid=1)
    out += marked("Caption", text(TABLE_CAPTION, 72.0, 564.0, size=10.0), mcid=2)
    out += marked("TD", text("Arm", 72.0, 540.0), mcid=3)
    out += marked("TD", text("Score", 200.0, 540.0), mcid=4)
    # The figure's own marked content: the picture is not drawn.
    out += marked("Figure", b"", mcid=5)
    fourth = out

    # Page 4, label "A-1": the references.
    out = marked("Artifact", text("Linked Paper Fixture 2026", 72.0, 760.0, size=8.0))
    out += marked("Heading", text("References", 72.0, 730.0, size=HEAD, font="F3"), mcid=0)
    for index, reference in enumerate(REFERENCES):
        out += marked("P", text(reference, 72.0, 690.0 - 18.0 * index, size=10.0), mcid=1 + index)
    fifth = out

    # Page 5, label "": a closing page.
    out = text("Appendix: Supplementary material", 72.0, 700.0)
    sixth = out
    return [first, second, third, fourth, fifth, sixth], runs


# --- The document -----------------------------------------------------------------


def _dest(page_ids: list[int], page: int, view: str) -> bytes:
    return b"[%d 0 R %s]" % (page_ids[page], view.encode())


def _names(page_ids: list[int]) -> bytes:
    """The /Names tree (sorted, as a name tree must be) and the catalog's old-style /Dests."""
    tree = b" ".join(
        _name(name) + b" " + _dest(page_ids, *place)
        for name, place in sorted(DESTS.items())
        if name != "old.dest"
    )
    tree += b" " + b" ".join(
        _name(name) + b" " + _dest(page_ids, *place) for name, place in UNSORTED.items()
    )
    old = _dest(page_ids, *DESTS["old.dest"])
    return b" /Names << /Dests << /Names [" + tree + b"] >> >> /Dests << /old.dest " + old + b" >>"


def _page_labels() -> bytes:
    return (
        b" /PageLabels << /Nums [0 << /S /r >> 1 << /S /D >> 4 << /S /D /P (A-) >> "
        b"5 << /P () >>] >>"
    )


TITLE = "Linked Paper Fixture 2026"


@dataclass(frozen=True)
class Bookmark:
    """An outline entry as the fixture writes it.

    *target* is how the entry points: ``("array", page, view)`` an explicit
    destination, ``("name", name)`` a named one in /Dest, ``("action", name)``
    one through a GoTo action, ``("remote", file)`` a jump into another file,
    ``("number", page)`` a destination whose page is a bare number, or
    ``("none",)``.
    """

    title: str
    level: int
    target: tuple


# The fixture's outline, and what the layer must read of it. The entries the
# guard drops are the paper's own title (C5), a float (C4), "Contents" (C3), a
# blank one (C1) and a page number (C2).
OUTLINE = [
    Bookmark(TITLE, 0, ("array", 0, "/Fit")),
    Bookmark("Abstract", 0, ("array", 0, "/XYZ 72 712 null")),
    Bookmark("1 Introduction", 0, ("name", "section.1")),
    Bookmark("1.1 Background", 1, ("array", 1, "/FitH 650")),
    Bookmark("2 Methods", 0, ("action", "section.2")),
    Bookmark("Figure 1", 1, ("array", 3, "/Fit")),
    Bookmark("3 Results", 0, ("array", 3, "/XYZ 72 736 null")),
    Bookmark("References", 0, ("array", 4, "/XYZ 72 744 null")),
    Bookmark("Contents", 0, ("array", 0, "/Fit")),
    Bookmark("", 0, ("array", 0, "/Fit")),
    Bookmark("Supplement", 0, ("remote", "supp.pdf")),
    Bookmark("Appendix", 0, ("number", 99)),
    Bookmark("12", 1, ("array", 5, "/Fit")),
]
# What each of them reads as: (parent, page, x, y, name).
OUTLINE_READ = [
    (None, 0, None, None, None),
    (None, 0, 72.0, 712.0, None),
    (None, 1, 72.0, 724.0, "section.1"),
    (2, 1, None, 650.0, None),
    (None, 2, 72.0, 722.0, "section.2"),
    (4, 3, None, None, None),
    (None, 3, 72.0, 736.0, None),
    (None, 4, 72.0, 744.0, None),
    (None, 0, None, None, None),
    (None, 0, None, None, None),
    (None, None, None, None, None),
    (None, None, None, None, None),
    (11, 5, None, None, None),
]
OUTLINE_DROPPED = (
    (0, "C5_title"),
    (5, "C4_float"),
    (8, "C3_nav"),
    (9, "C1_blank"),
    (12, "C2_page"),
)


def _pdf_string(value: str) -> bytes:
    """*value* as a PDF text string: Latin-1 when it fits, UTF-16BE otherwise."""
    try:
        raw = value.encode("latin-1")
    except UnicodeEncodeError:
        return b"<" + (b"\xfe\xff" + value.encode("utf-16-be")).hex().encode() + b">"
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def _target(target: tuple, page_ids: list[int]) -> bytes:
    kind = target[0]
    if kind == "array":
        return b" /Dest " + _dest(page_ids, target[1], target[2])
    if kind == "name":
        return b" /Dest " + _name(target[1])
    if kind == "action":
        return b" /A << /S /GoTo /D " + _name(target[1]) + b" >>"
    if kind == "remote":
        return b" /A << /S /GoToR /F " + _name(target[1]) + b" /D [0 /Fit] >>"
    if kind == "number":
        return b" /Dest [%d /Fit]" % target[1]
    return b""


def _outline(table: Table, page_ids: list[int], bookmarks: list[Bookmark], *, loop: bool) -> int:
    """The outline's root object; with *loop*, the last top-level entry's /Next is the first."""
    root = table.add()
    nodes = [table.add() for _ in bookmarks]
    parent_of: list[int | None] = []
    children: dict[int | None, list[int]] = {}
    stack: list[int] = []
    for index, bookmark in enumerate(bookmarks):
        del stack[bookmark.level :]
        parent = stack[-1] if stack else None
        parent_of.append(parent)
        children.setdefault(parent, []).append(index)
        stack.append(index)
    for index, bookmark in enumerate(bookmarks):
        parent = parent_of[index]
        siblings = children[parent]
        place = siblings.index(index)
        body = b"<< /Title " + _pdf_string(bookmark.title)
        body += b" /Parent %d 0 R" % (nodes[parent] if parent is not None else root)
        if place > 0:
            body += b" /Prev %d 0 R" % nodes[siblings[place - 1]]
        if place < len(siblings) - 1:
            body += b" /Next %d 0 R" % nodes[siblings[place + 1]]
        elif loop and parent is None:
            body += b" /Next %d 0 R" % nodes[siblings[0]]
        kids = children.get(index)
        if kids:
            body += b" /First %d 0 R /Last %d 0 R /Count %d" % (
                nodes[kids[0]],
                nodes[kids[-1]],
                len(kids),
            )
        table.fill(nodes[index], body + _target(bookmark.target, page_ids) + b" >>")
    top = children.get(None, [])
    if top:
        table.fill(
            root,
            b"<< /Type /Outlines /First %d 0 R /Last %d 0 R /Count %d >>"
            % (nodes[top[0]], nodes[top[-1]], len(top)),
        )
    else:
        table.fill(root, b"<< /Type /Outlines /Count 0 >>")
    return root


@dataclass(frozen=True)
class Tag:
    """A structure element as the fixture writes it: its type /S, its kids and its text attributes.

    A kid is a ``Tag`` or a ``(page, mcid)`` pair, the marked content it holds.
    """

    role: str
    kids: tuple = ()
    alt: str | None = None
    actual: str | None = None
    lang: str | None = None


# /RoleMap: the custom heading type is an H1.
ROLE_MAP = {"Heading": "H1"}


def _paragraphs(page: int, mcids, role: str = "P") -> list[Tag]:
    return [Tag(role, ((page, mcid),)) for mcid in mcids]


# The tree over the pages' marked content: sections, then the elements in each.
TREE = Tag(
    "Document",
    (
        Tag(
            "Sect",
            (
                Tag("Heading", ((0, 0),), lang="en-US"),
                *_paragraphs(0, range(1, 7)),
            ),
        ),
        Tag(
            "Sect",
            (
                Tag("H1", ((1, 0),)),
                Tag(
                    "P",
                    (
                        (1, 1),
                        Tag("Link", ((1, 5),)),
                        (1, 8),
                        Tag("Link", ((1, 6),)),
                        (1, 9),
                        Tag("Link", ((1, 7),)),
                        (1, 10),
                    ),
                ),
                Tag("P", ((1, 2),), lang="de-DE"),
                Tag("H2", ((1, 3),)),
                # Crosses the page break: one element, content on pages 1 and 2.
                Tag("P", ((1, 4), (2, 0))),
            ),
        ),
        Tag(
            "Sect",
            (
                Tag("Heading", ((2, 1),)),
                *_paragraphs(2, range(2, 2 + FILLER_LINES)),
                Tag("P", ((2, 2 + FILLER_LINES),), actual="E = mc squared (3)"),
            ),
        ),
        Tag(
            "Sect",
            (
                Tag("H1", ((3, 0),)),
                Tag("Figure", ((3, 5),), alt="A diagram of the linked pipeline."),
                Tag("Caption", ((3, 1),)),
                Tag("Caption", ((3, 2),)),
                Tag("Table", (Tag("TR", (Tag("TD", ((3, 3),)), Tag("TD", ((3, 4),)))),)),
            ),
        ),
        Tag(
            "Sect",
            (
                Tag("H1", ((4, 0),)),
                *_paragraphs(4, range(1, 1 + len(REFERENCES)), role="Reference"),
            ),
        ),
    ),
)


def _structure(
    table: Table, page_ids: list[int], tree: Tag, role_map: dict[str, str]
) -> tuple[int, dict[int, int]]:
    """The tree's /StructTreeRoot object, and each tagged page's /StructParents key."""
    root = table.add()
    parents: dict[int, dict[int, int]] = {}

    def emit(tag: Tag, parent: int) -> int:
        obj = table.add()
        own = next((kid[0] for kid in tag.kids if isinstance(kid, tuple)), None)
        kids = []
        for kid in tag.kids:
            if isinstance(kid, Tag):
                kids.append(b"%d 0 R" % emit(kid, obj))
                continue
            page, mcid = kid
            parents.setdefault(page, {})[mcid] = obj
            # Content on the element's own page is a bare mcid; on another, an /MCR.
            kids.append(
                b"%d" % mcid
                if page == own
                else b"<< /Type /MCR /Pg %d 0 R /MCID %d >>" % (page_ids[page], mcid)
            )
        body = b"<< /Type /StructElem /S /" + tag.role.encode() + b" /P %d 0 R" % parent
        if own is not None:
            body += b" /Pg %d 0 R" % page_ids[own]
        if len(kids) == 1:
            body += b" /K " + kids[0]
        elif kids:
            body += b" /K [" + b" ".join(kids) + b"]"
        for key, value in (("Alt", tag.alt), ("ActualText", tag.actual), ("Lang", tag.lang)):
            if value is not None:
                body += b" /" + key.encode() + b" " + _pdf_string(value)
        table.fill(obj, body + b" >>")
        return obj

    top = emit(tree, root)
    # The parent tree: for each page, the element of each mcid, by mcid.
    nums = b" ".join(
        b"%d [" % page
        + b" ".join(
            b"%d 0 R" % by_mcid[mcid] if mcid in by_mcid else b"null"
            for mcid in range(max(by_mcid) + 1)
        )
        + b"]"
        for page, by_mcid in sorted(parents.items())
    )
    mapped = b" ".join(b"/%s /%s" % (a.encode(), b.encode()) for a, b in role_map.items())
    table.fill(
        root,
        b"<< /Type /StructTreeRoot /K %d 0 R /RoleMap << " % top
        + mapped
        + b" >> /ParentTree << /Nums ["
        + nums
        + b"] >> >>",
    )
    return root, {page: page for page in parents}


@dataclass(frozen=True)
class LinkSpec:
    """A link annotation as the fixture writes it: the runs it covers and where it points.

    *target* is ``("dest", name)`` the annotation's /Dest as a name,
    ``("array", page, view)`` its /Dest as an explicit destination,
    ``("goto", name)`` a GoTo action, ``("uri", uri)``, ``("remote", file)``,
    ``("launch", file)``, ``("named", action)`` or ``("none",)``. With *quads*
    the annotation also lists the quadrilateral of each run it covers.
    """

    page: int
    runs: tuple[str, ...]
    target: tuple
    quads: bool = False


# In annotation order; the keys are the runs of ``_pages``.
LINKS = [
    LinkSpec(0, ("fig_ref",), ("dest", "figure.1")),
    LinkSpec(0, ("cite_ref",), ("goto", "cite.smith2020"), quads=True),
    LinkSpec(0, ("sec_ref",), ("dest", "section.2")),
    LinkSpec(0, ("note_ref",), ("dest", "Hfootnote.1")),
    LinkSpec(0, ("eq_ref",), ("dest", "equation.3")),
    LinkSpec(0, ("doi_ref",), ("uri", "https://doi.org/10.1000/xyz123")),
    LinkSpec(0, ("remote_ref",), ("remote", "other.pdf")),
    LinkSpec(0, ("launch_ref",), ("launch", "run.sh")),
    LinkSpec(0, ("named_ref",), ("named", "NextPage")),
    LinkSpec(0, ("broken_ref",), ("dest", "nowhere")),
    LinkSpec(0, ("bare_ref",), ("none",)),
    LinkSpec(0, ("old_ref",), ("dest", "old.dest")),
    LinkSpec(0, ("array_bib_ref",), ("array", 4, "/XYZ 72 704 null")),
    LinkSpec(0, ("array_fig_ref",), ("array", 3, "/XYZ 72 662 null")),
    LinkSpec(0, ("fit_ref",), ("array", 3, "/Fit")),
    LinkSpec(1, ("table_ref",), ("dest", "table.1")),
    LinkSpec(1, ("second_ref",), ("dest", "page.4")),
    LinkSpec(1, ("late_ref",), ("dest", "aaa.unsorted")),
    LinkSpec(1, ("wrap_a", "wrap_b"), ("goto", "figure.1"), quads=True),
]
NOTE_ANNOTATION = b"<< /Type /Annot /Subtype /Text /Rect [400 700 420 720] /Contents (A note) >>"


def _annotation(spec: LinkSpec, runs: dict[str, Placed], page_ids: list[int]) -> bytes:
    placed = [runs[key] for key in spec.runs]
    rect = (
        min(run.rect[0] for run in placed),
        min(run.rect[1] for run in placed),
        max(run.rect[2] for run in placed),
        max(run.rect[3] for run in placed),
    )
    body = b"<< /Type /Annot /Subtype /Link /Rect [" + _numbers(rect) + b"] /Border [0 0 0]"
    if spec.quads:
        body += b" /QuadPoints [" + b" ".join(_numbers(run.quad) for run in placed) + b"]"
    kind, *args = spec.target
    if kind == "dest":
        body += b" /Dest " + _name(args[0])
    elif kind == "array":
        body += b" /Dest " + _dest(page_ids, args[0], args[1])
    elif kind == "goto":
        body += b" /A << /S /GoTo /D " + _name(args[0]) + b" >>"
    elif kind == "uri":
        body += b" /A << /S /URI /URI " + _name(args[0]) + b" >>"
    elif kind == "remote":
        body += b" /A << /S /GoToR /F " + _name(args[0]) + b" /D [0 /Fit] >>"
    elif kind == "launch":
        body += b" /A << /S /Launch /F " + _name(args[0]) + b" >>"
    elif kind == "named":
        body += b" /A << /S /Named /N /" + args[0].encode() + b" >>"
    return body + b" >>"


def linked_paper(
    *,
    outline: list[Bookmark] | None = OUTLINE,
    loop: bool = False,
    tagging: str = "full",
    tree: Tag = TREE,
    role_map: dict[str, str] = ROLE_MAP,
) -> bytes:
    """The six-page paper: page labels, named destinations, an outline and the text they point at.

    *outline* are the bookmarks (None for a paper without an outline). *tagging*
    is how the paper declares its structure: ``full`` (/MarkInfo and a structure
    tree), ``tree`` (the tree alone), ``marked`` (/MarkInfo alone) or ``none``;
    the tree is *tree* with the type mapping *role_map*.
    """
    table = Table()
    fonts = b" ".join(
        b"/F%d %d 0 R"
        % (number, table.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /%s >>" % name))
        for number, name in ((1, b"Helvetica"), (2, b"Times-Roman"), (3, b"Helvetica-Bold"))
    )
    contents, runs = _pages()
    page_ids = [table.add() for _ in range(N_PAGES)]
    annots: dict[int, list[int]] = {}
    for spec in LINKS:
        annots.setdefault(spec.page, []).append(table.add(_annotation(spec, runs, page_ids)))
    annots[0].append(table.add(NOTE_ANNOTATION))
    tree_root, struct_parents = (
        _structure(table, page_ids, tree, role_map) if tagging in ("full", "tree") else (0, {})
    )
    for number, (page_id, content) in enumerate(zip(page_ids, contents, strict=True)):
        stream = table.add(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        listed = b" ".join(b"%d 0 R" % annot for annot in annots.get(number, []))
        table.fill(
            page_id,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] " % (PAGE_W, PAGE_H)
            + b"/Resources << /Font << "
            + fonts
            + b" >> >> /Contents %d 0 R" % stream
            + (b" /Annots [" + listed + b"]" if listed else b"")
            + (b" /StructParents %d" % struct_parents[number] if number in struct_parents else b"")
            + b" >>",
        )
    table.fill(
        2,
        b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % page_id for page_id in page_ids), N_PAGES),
    )
    extras = _names(page_ids) + _page_labels()
    if tagging in ("full", "marked"):
        extras += b" /MarkInfo << /Marked true >>"
    if tree_root:
        extras += b" /StructTreeRoot %d 0 R" % tree_root
    if outline is not None:
        extras += b" /Outlines %d 0 R" % _outline(table, page_ids, outline, loop=loop)
    table.fill(1, b"<< /Type /Catalog /Pages 2 0 R" + extras + b" >>")
    info = table.add(b"<< /Title " + _pdf_string(TITLE) + b" >>")
    return serialize_pdf(table.bodies, b"/Info %d 0 R" % info)
