"""Named-entity resolution for the hardened JATS/ePub XML parser.

The parser keeps ``resolve_entities=False``/``no_network=True``, so the tests
below pin both halves of the contract: standard character entities resolve,
and anything a document tries to declare or fetch for itself does not.
"""

from bibr.input.xml_entities import parse_xml

DOCTYPE = b'<!DOCTYPE a SYSTEM "external.dtd">'


def _flat(xml: bytes) -> str:
    return "".join(parse_xml(xml).itertext())


class TestResolution:
    def test_iso_and_latin1_names_resolve(self):
        assert _flat(DOCTYPE + b"<a>&alpha; &uuml; &deg; &mdash;</a>") == "α ü ° —"

    def test_adjacent_entities_resolve(self):
        assert _flat(DOCTYPE + b"<a><b>&alpha;&beta;Z</b></a>") == "αβZ"

    def test_entity_merges_around_inline_children(self):
        # Leading entity joins the parent's text; a later one joins the
        # preceding child's tail. Ordering must survive both splices.
        assert _flat(DOCTYPE + b"<a><b>p&alpha;<i>c</i>&beta;q</b></a>") == "pαcβq"

    def test_predefined_and_numeric_references_are_untouched(self):
        assert _flat(DOCTYPE + b"<a>&amp; &lt; &#945; &#x3b2;</a>") == "& < α β"

    def test_unknown_name_keeps_its_literal_source(self):
        # Dropping it would silently delete content; the raw form stays visible.
        assert _flat(DOCTYPE + b"<a>x&notarealentity;y</a>") == "x&notarealentity;y"


class TestHardening:
    """Resolution must not re-open what ``resolve_entities=False`` closes."""

    def test_internal_subset_entity_is_not_expanded(self):
        xml = b'<!DOCTYPE a [ <!ENTITY secret "EXPANDED"> ]><a>&secret;</a>'
        assert _flat(xml) == "&secret;"

    def test_external_system_entity_is_not_fetched(self):
        xml = b'<!DOCTYPE a [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]><a>&xxe;</a>'
        assert _flat(xml) == "&xxe;"

    def test_nested_entity_bomb_does_not_expand(self):
        xml = (
            b"<!DOCTYPE a [\n"
            b' <!ENTITY l0 "aaaaaaaaaa">\n'
            b' <!ENTITY l1 "&l0;&l0;&l0;&l0;&l0;&l0;&l0;&l0;&l0;&l0;">\n'
            b' <!ENTITY l2 "&l1;&l1;&l1;&l1;&l1;&l1;&l1;&l1;&l1;&l1;">\n'
            b"]><a>&l2;</a>"
        )
        assert _flat(xml) == "&l2;"
