from bibr.schemas import PaperTypeLabel


def test_valid_paper_type_and_confidence():
    m = PaperTypeLabel(paper_type="empirical", confidence=0.9)
    assert m.paper_type == "empirical"
    assert m.confidence == 0.9


def test_near_miss_paper_type_coerced_to_none():
    # lenient BeforeValidator: unknown label -> None (not a hard failure)
    m = PaperTypeLabel(paper_type="Empirical Study", confidence=0.5)
    assert m.paper_type is None


def test_defaults_are_null():
    m = PaperTypeLabel()
    assert m.paper_type is None
    assert m.confidence is None
