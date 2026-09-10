"""OOXML Math Markup Language (OMML) helpers.

Flattens block-level ``<m:oMath>`` / ``<m:oMathPara>`` subtrees to plain
text so the DOCX parser can preserve inline math characters in paragraph
flow without depending on a full math renderer.
"""

from __future__ import annotations

OMML_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def omml_to_text(omath_el) -> str:
    """Concatenate ``<m:t>`` text within an OMML subtree.

    Lossy — fractions, superscripts, etc. lose structural meaning. Good
    enough to preserve symbols/letters for downstream NER and to keep
    paragraph flow coherent.
    """
    parts = [t.text for t in omath_el.iter(f"{{{OMML_NAMESPACE}}}t") if t.text]
    return "".join(parts).strip()
