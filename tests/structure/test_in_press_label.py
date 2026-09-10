"""In-press label extraction must not change any compiled pattern."""

import re

from bibr.structure import citation_matcher as cm
from bibr.utils.text import IN_PRESS_LABEL


def test_shared_label_value():
    assert IN_PRESS_LABEL == r"in\s+press|forthcoming"


def test_matcher_patterns_unchanged():
    assert cm._IN_PRESS_RE.pattern == r"\b(in\s+press|forthcoming)\b"
    assert cm._IN_PRESS_PAREN_RE.pattern == r"\(\s*(in\s+press|forthcoming)\s*\)"
    # IGNORECASE is part of the byte-preserving contract ("In Press" must match).
    assert cm._IN_PRESS_RE.flags & re.IGNORECASE
    assert cm._IN_PRESS_PAREN_RE.flags & re.IGNORECASE
