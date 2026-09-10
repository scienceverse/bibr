"""Section-classifier lookup tests.

Two layers of defense for the alias-lookup contract (the table is keyed on
lowercase, whitespace-trimmed strings):

1. ``_classify_lookup`` itself is case-insensitive — direct callers can pass
   non-normalized input and still get the right answer (Plan 5 Task 14).
2. Public callers (``classify_header``, ``classify_headers_batch``) normalize
   their input before calling ``_classify_lookup`` (Plan 6 Task 7) — so the
   call-site contract is also pinned.

Together these prevent silent regressions if either layer is removed.
"""

from __future__ import annotations

from bibr.paper_contents import CanonicalSection
from bibr.structure.section_classifier import _classify_lookup


def test_classify_lookup_is_case_insensitive():
    """Direct call with non-lowercased input must still match aliases."""
    sec, score = _classify_lookup("Methods")
    assert sec == CanonicalSection.METHODS
    assert score == 1.0

    sec, score = _classify_lookup("Statistical Methods")
    assert sec in {CanonicalSection.METHODS}  # longest substring wins


def test_classify_lookup_uses_compiled_regex():
    """Substring path must consult the precomputed regex, not iterate aliases."""
    import bibr.structure.section_classifier as mod

    assert hasattr(mod, "_ALIAS_SUBSTRING_RE"), "module must precompile a combined alias regex"


def test_public_callers_pass_normalized_input_to_lookup(monkeypatch):
    """``classify_headers_batch`` must normalize before calling ``_classify_lookup``."""
    import bibr.structure.section_classifier as mod

    seen: list[str] = []

    real = mod._classify_lookup

    def spy(header_text):
        seen.append(header_text)
        return real(header_text)

    monkeypatch.setattr(mod, "_classify_lookup", spy)

    out = mod.classify_headers_batch(["Methods", "  RESULTS  "])
    assert out, out
    assert all(s == s.strip().lower() for s in seen), seen


def test_classify_header_normalizes_input(monkeypatch):
    """``classify_header`` (sync single-header API) must also normalize."""
    import bibr.structure.section_classifier as mod

    seen: list[str] = []

    real = mod._classify_lookup

    def spy(header_text):
        seen.append(header_text)
        return real(header_text)

    monkeypatch.setattr(mod, "_classify_lookup", spy)

    mod.classify_header("Discussion")
    assert seen == ["discussion"], seen


async def test_printed_furniture_never_reaches_the_model_or_llm_tiers():
    """Article-type kickers and badges are not sections and are not titles.

    There is no vocabulary member for "journal furniture", so the model and LLM
    tiers are forced to guess on these rows and answer TITLE confidently — which
    then seeds a false front-matter record ("SHORT COMMUNICATION" took the DOI
    row from the real record in 10.1515_ffp-2017-0016). UNKNOWN is the honest
    answer, and front matter vetoes the row again by the same list.
    """
    import bibr.structure.section_classifier as mod

    furniture = [
        "OPEN ACCESS",
        "Research Article",
        "SHORT COMMUNICATION",
        "Check for updates",
        "Article Info",
    ]

    out = await mod.classify_headers_batch_async(furniture, llm_client=None)

    assert [section for section, *_ in out] == [CanonicalSection.UNKNOWN] * len(furniture)
    assert all(source is None for *_, source in out)


async def test_furniture_guard_does_not_swallow_real_headings():
    import bibr.structure.section_classifier as mod

    out = await mod.classify_headers_batch_async(["Methods", "References"], llm_client=None)

    assert [section for section, *_ in out] == [
        CanonicalSection.METHODS,
        CanonicalSection.REFERENCES,
    ]


def test_non_english_headings_resolve_without_the_model():
    """Translated heading aliases resolve without being mistaken for title sections."""
    for header, expected in (
        ("Список литературы", CanonicalSection.REFERENCES),
        ("Kaynakça", CanonicalSection.REFERENCES),
        ("参考文献", CanonicalSection.REFERENCES),
        ("Аннотация", CanonicalSection.ABSTRACT),
        ("Özet", CanonicalSection.ABSTRACT),
        ("Kata Kunci", CanonicalSection.KEYWORDS),
        ("Hasil dan Pembahasan", CanonicalSection.RESULTS),
        ("Pendahuluan", CanonicalSection.INTRODUCTION),
    ):
        section, score = _classify_lookup(header.lower())
        assert section == expected, header
        assert score == 1.0, header
