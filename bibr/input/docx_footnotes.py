"""OOXML footnote and endnote loaders.

Reads ``footnotes.xml`` / ``endnotes.xml`` via the rels table on the main
document part (python-docx 1.x has no first-class accessor for either).
Returns ``{note_id: text}``. Built-in separator/continuation pseudo-notes are
skipped.

Footnotes and endnotes have *independent* id spaces — both start at 1 — so
callers must keep the two maps apart rather than merging them.
"""

from __future__ import annotations

FOOTNOTES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
ENDNOTES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes"
_W_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Elements that end the run of text they sit in without carrying text of their
# own. A note is flattened to one string, so without these a tab- or
# break-separated bibliography endnote fuses into "Smith, J.2020Title".
_NOTE_SEPARATORS: frozenset[str] = frozenset(
    f"{{{_W_NAMESPACE}}}{name}" for name in ("br", "tab", "cr", "p")
)


def _load_notes(doc, *, part_attr: str, reltype: str, tag: str) -> dict[str, str]:
    """Return ``{note_id: text}`` for one notes part."""
    part = getattr(doc.part, part_attr, None)
    if part is None:
        rels = getattr(doc.part, "rels", None)
        if rels:
            for rel in rels.values():
                if rel.reltype == reltype:
                    part = rel.target_part
                    break
    if part is None:
        return {}

    out: dict[str, str] = {}
    root = getattr(part, "element", None)
    if root is None:
        # Fallback for parts whose element isn't auto-parsed (Part vs StoryPart).
        # The blob comes from a user-uploaded file: parse with entity
        # resolution disabled so crafted DOCTYPEs can't leak local files (XXE)
        # or detonate an entity-expansion bomb (billion laughs).
        from lxml import etree

        parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
        try:
            root = etree.fromstring(part.blob, parser=parser)
        except Exception:
            return {}

    for note in root.iter(f"{{{_W_NAMESPACE}}}{tag}"):
        note_type = note.get(f"{{{_W_NAMESPACE}}}type")
        if note_type in ("separator", "continuationSeparator"):
            continue
        note_id = note.get(f"{{{_W_NAMESPACE}}}id")
        if note_id is None:
            continue
        parts: list[str] = []
        for node in note.iter():
            tag = node.tag
            if not isinstance(tag, str):  # comments / PIs
                continue
            if tag == f"{{{_W_NAMESPACE}}}t":
                if node.text:
                    parts.append(node.text)
            elif tag in _NOTE_SEPARATORS and parts and not parts[-1][-1].isspace():
                # Word splits runs mid-word for formatting, so only explicit
                # separators may contribute whitespace; adjacent <w:t> still
                # concatenates untouched.
                parts.append(" ")
        out[note_id] = "".join(parts).strip()
    return out


def load_footnotes(doc) -> dict[str, str]:
    """Return ``{footnote_id: text}`` for all real footnote entries.

    Returns an empty dict when the document has no footnotes part.
    """
    return _load_notes(doc, part_attr="footnotes_part", reltype=FOOTNOTES_REL, tag="footnote")


def load_endnotes(doc) -> dict[str, str]:
    """Return ``{endnote_id: text}`` for all real endnote entries.

    Endnotes were previously not read at all, so a document whose notes — or
    whose entire bibliography, the humanities convention — are endnotes rather
    than footnotes lost every one of them silently.
    """
    return _load_notes(doc, part_attr="endnotes_part", reltype=ENDNOTES_REL, tag="endnote")
