"""Hardened XML parsing for the native JATS/ePub paths, with entity resolution.

Both parsers read untrusted uploads, so they build their lxml parser with
``resolve_entities=False`` and ``no_network=True`` — the XXE and billion-laughs
defense. libxml2 then represents every *named* character entity (``&alpha;``,
``&uuml;``, ``&deg;`` …) as an ``_Entity`` node whose ``.text`` is the literal
source string, so flattening helpers such as ``"".join(el.itertext())`` emit
that raw markup into titles, author names and references. XML's five predefined
entities and every numeric reference are resolved by libxml2 regardless, which
makes the damage look plausible rather than obviously broken.

Turning resolution back on is not an option: the DTD is never fetched, so
libxml2 raises ``Entity 'alpha' not defined`` for any document that declares a
DOCTYPE and uses one. Instead the entity nodes are resolved here after the
parse, against the HTML5 named-character table — a superset of the ISO sets
(isogrk, isolat, isonum, isoamsa, mmlextra …) that JATS DTDs pull in.

Entities the table does not know keep their literal ``&name;`` source text.
That deliberately includes anything a document declares in its own internal DTD
subset: expanding those is the vector ``resolve_entities=False`` exists to close.
"""

from __future__ import annotations

from html.entities import html5, name2codepoint

from lxml import etree


def _replacement(node) -> str:
    """Character(s) for an entity node, or its literal source text if unknown."""
    name = getattr(node, "name", None)
    if not name:
        return node.text or ""
    # XML always spells the reference with the semicolon; ``name2codepoint`` is
    # the older HTML4 set, kept as a fallback for the handful html5 omits.
    resolved = html5.get(f"{name};")
    if resolved is None:
        codepoint = name2codepoint.get(name)
        resolved = chr(codepoint) if codepoint is not None else None
    if resolved is None:
        return node.text or f"&{name};"
    return resolved


def resolve_entity_refs(root) -> None:
    """Splice every unresolved named-entity node back into surrounding text, in place.

    An entity node carries no children, so its replacement plus its tail merges
    into the preceding sibling's tail — or the parent's text when it is first.
    Once the nodes are gone, every downstream text helper sees plain characters.
    """
    for node in list(root.iter(etree.Entity)):
        parent = node.getparent()
        if parent is None:  # defensive: an entity cannot be the document root
            continue
        text = _replacement(node) + (node.tail or "")
        previous = node.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + text
        else:
            parent.text = (parent.text or "") + text
        parent.remove(node)


def parse_xml(data: bytes):
    """Parse *data* with entity/network expansion off, then resolve named entities."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    root = etree.fromstring(data, parser=parser)
    resolve_entity_refs(root)
    return root
