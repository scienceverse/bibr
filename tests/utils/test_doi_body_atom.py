"""DOI-body atom extraction must not change any compiled pattern."""

import re

from bibr.extract.segment_filter import _LOCATOR_RE
from bibr.input.consolidate_text import _DOI_BODY_RE
from bibr.utils.text import _BARE_DOI_RE, DOI_BODY


def test_atom_value():
    assert DOI_BODY == r"10\.\d{4,9}/"


def test_patterns_unchanged():
    # The suffix tail deliberately excludes a trailing hyphen: a DOI wrapped
    # after one of its own hyphens would otherwise validate as a stub. The
    # DOI_BODY atom itself is still spliced in byte-for-byte, which is what
    # this module guards.
    assert _BARE_DOI_RE.pattern == r"^10\.\d{4,9}/[^\s]*[^\s-]$"
    assert _DOI_BODY_RE.pattern == r"10\.\d{4,9}/"
    assert _LOCATOR_RE.pattern == r"https?://|\b10\.\d{4,9}/"
    # IGNORECASE is part of the byte-preserving contract for the locator.
    assert _LOCATOR_RE.flags & re.IGNORECASE
