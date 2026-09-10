"""OMML math flattening unit test."""

from __future__ import annotations

from defusedxml import ElementTree as ET

from bibr.input.docx_omml import OMML_NAMESPACE, omml_to_text


def test_omml_flattens_text_elements():
    xml = f"""
    <oMath xmlns="{OMML_NAMESPACE}">
        <r><t>x</t></r>
        <r><t>+</t></r>
        <r><t>1</t></r>
    </oMath>
    """
    root = ET.fromstring(xml)
    assert omml_to_text(root) == "x+1"


def test_omml_skips_non_text_elements():
    xml = f"""
    <oMath xmlns="{OMML_NAMESPACE}">
        <r><t>a</t></r>
        <fraction><numerator><r><t>2</t></r></numerator></fraction>
    </oMath>
    """
    root = ET.fromstring(xml)
    # Lossy: structural info dropped, but text within tree concatenated.
    assert omml_to_text(root) == "a2"


def test_omml_empty_returns_empty_string():
    xml = f'<oMath xmlns="{OMML_NAMESPACE}"></oMath>'
    root = ET.fromstring(xml)
    assert omml_to_text(root) == ""
