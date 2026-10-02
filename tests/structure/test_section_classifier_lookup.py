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
from bibr.structure.section_classifier import _classify_lookup, _classify_lookup_full


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


async def test_cover_sheet_labels_and_containers_never_reach_the_model_or_llm():
    """Preprint cover labels and grouping headings are not sections (#121, #123).

    A preprint server's cover page prints "Posted Date: ..." as a heading and
    the model typed it TITLE; a BMC "Declarations" block went to
    Acknowledgments; Lancet's "Research in context" box anchored the
    introduction. None of them names a section type.
    """
    import bibr.structure.section_classifier as mod

    headings = [
        "Posted Date: September 29th, 2026",
        "Word count",
        "Running title: Sleep and memory",
        "Manuscript Number: ABC-D-26-00123",
        "Authors",
        "Disclaimer: The manuscript is the authors' accepted version",
        "Declarations",
        "Statements and Declarations",
        "Research in context",
        "Clinical Perspective",
        "What is new?",
        "What this study adds",
    ]

    out = await mod.classify_headers_batch_async(headings, llm_client=None)

    assert [section for section, *_ in out] == [CanonicalSection.UNKNOWN] * len(headings)
    assert all(source is None for *_, source in out)
    assert mod.classify_headers_batch(headings) == [(CanonicalSection.UNKNOWN, 0.0)] * len(headings)


async def test_cover_label_guard_keeps_author_sections():
    import bibr.structure.section_classifier as mod

    out = await mod.classify_headers_batch_async(
        ["Author contributions", "Authors' contributions", "Contributors"], llm_client=None
    )

    assert [section for section, *_ in out] == [CanonicalSection.AUTHOR_CONTRIBUTIONS] * 3


def test_declaration_block_subheadings_resolve_by_alias():
    """The BMC/Springer "Declarations" block's sub-headings type by alias (#121)."""
    for header, expected in (
        ("ethics approval and consent to participate", CanonicalSection.ETHICS),
        ("consent for publication", CanonicalSection.ETHICS),
        ("availability of data and materials", CanonicalSection.OPEN_DATA),
        ("competing interests", CanonicalSection.COI),
        ("transparency declarations", CanonicalSection.COI),
        ("contributors", CanonicalSection.AUTHOR_CONTRIBUTIONS),
    ):
        section, score, trusted = _classify_lookup_full(header)
        assert section == expected, header
        assert trusted, header


def test_substring_coverage_adds_up_hits_of_the_same_type():
    # "ethics" (6) + "consent to participate" (22) cover 28 of 42 characters;
    # the longest alias alone covers only 0.52, below the trust bar.
    section, score, trusted = _classify_lookup_full("ethics approval and consent to participate")
    assert (section, score, trusted) == (CanonicalSection.ETHICS, 0.95, True)
    # A single generic hit inside a long heading stays untrusted.
    section, _score, trusted = _classify_lookup_full("a model of memory consolidation in sleep")
    assert not trusted


def test_float_lists_after_references_are_figure_sections():
    for header in ("figure legends", "figure captions", "legends to figures", "tables and figures"):
        section, score = _classify_lookup(header)
        assert (section, score) == (CanonicalSection.FIGURE, 1.0), header


def test_more_non_english_section_names_resolve_by_alias():
    """Italian, Dutch, Swedish, Portuguese, Spanish and Indonesian headings (#121, #124)."""
    for header, expected in (
        ("riassunto", CanonicalSection.ABSTRACT),
        ("samenvatting", CanonicalSection.ABSTRACT),
        ("sammanfattning", CanonicalSection.ABSTRACT),
        ("bibliografia", CanonicalSection.REFERENCES),
        ("riferimenti bibliografici", CanonicalSection.REFERENCES),
        ("referências bibliográficas", CanonicalSection.REFERENCES),
        ("literatuur", CanonicalSection.REFERENCES),
        ("introduzione", CanonicalSection.INTRODUCTION),
        ("inleiding", CanonicalSection.INTRODUCTION),
        ("materiali e metodi", CanonicalSection.METHODS),
        ("risultati", CanonicalSection.RESULTS),
        ("discussione", CanonicalSection.DISCUSSION),
        ("conclusioni", CanonicalSection.DISCUSSION),
        ("kesimpulan dan saran", CanonicalSection.DISCUSSION),
    ):
        section, score = _classify_lookup(header)
        assert (section, score) == (expected, 1.0), header
