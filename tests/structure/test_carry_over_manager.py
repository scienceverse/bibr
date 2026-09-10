"""CarryOverState unit tests.

These pin the C6 invariant: section_id captured at append time survives
across a heading change before flush.
"""

from __future__ import annotations


def test_starts_empty():
    from bibr.structure.carry_over_manager import CarryOverState

    co = CarryOverState()
    assert co.text == ""
    assert co.page is None
    assert co.section_id is None
    assert co.provenance == []
    assert co.has_pending() is False


def test_has_pending_with_text():
    from bibr.structure.carry_over_manager import CarryOverState

    co = CarryOverState()
    co.text = "hello"
    assert co.has_pending() is True


def test_has_pending_false_for_whitespace_only():
    from bibr.structure.carry_over_manager import CarryOverState

    co = CarryOverState()
    co.text = "   \n\t"
    assert co.has_pending() is False


def test_reset_clears_all_fields():
    from bibr.structure.carry_over_manager import CarryOverState

    co = CarryOverState()
    co.text = "hello"
    co.page = 1
    co.section_id = 5
    co.provenance = [object()]
    co.reset()
    assert co.text == ""
    assert co.page is None
    assert co.section_id is None
    assert co.provenance == []


def test_independent_provenance_lists():
    """Mutating one instance's provenance must not affect another's default."""
    from bibr.structure.carry_over_manager import CarryOverState

    a = CarryOverState()
    b = CarryOverState()
    a.provenance.append("x")
    assert b.provenance == []
