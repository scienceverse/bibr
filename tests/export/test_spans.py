"""Character spans of xrefs, links and expressions within text[].text."""

from __future__ import annotations

from types import SimpleNamespace

from bibr.export.spans import SpanLocator, equation_span, url_span, verified_span, xref_span


def _xref(text_id, contents, start=None, end=None):
    return SimpleNamespace(text_id=text_id, contents=contents, start=start, end=end)


def test_a_recorded_span_is_used_when_it_covers_the_printed_reference():
    texts = {1: "Study 1 showed it.1 Later work2 agreed."}
    locator = SpanLocator(texts, shared=True)
    # A superscript "1": locating "1" would hit "Study 1"; the detector's span wins.
    assert xref_span(locator, texts, _xref(1, "1", 18, 19)) == (18, 19)


def test_a_stale_recorded_span_falls_back_to_locating_the_contents():
    texts = {1: "As shown in Table 2, it rose."}
    locator = SpanLocator(texts, shared=True)
    assert xref_span(locator, texts, _xref(1, "Table 2", 0, 3)) == (12, 19)


def test_whitespace_in_the_sentence_does_not_block_location():
    texts = {1: "See (Smith,  2020) for details."}
    locator = SpanLocator(texts, shared=True)
    start, end = xref_span(locator, texts, _xref(1, "(Smith, 2020)")) or (0, 0)
    assert texts[1][start:end] == "(Smith,  2020)"


def test_targets_of_one_citation_share_its_span():
    texts = {1: "Prior work [1, 2] agrees."}
    locator = SpanLocator(texts, shared=True)
    first = xref_span(locator, texts, _xref(1, "[1, 2]"))
    second = xref_span(locator, texts, _xref(1, "[1, 2]"))
    assert first == second == (11, 17)


def test_a_footnote_reference_is_never_located():
    """Its mark is not in the text, so a search for "1" would find the 1 of a
    citation or a statistic."""
    texts = {1: "It replicated prior work [1] in 2021."}
    locator = SpanLocator(texts, shared=True)
    foot = SimpleNamespace(text_id=1, contents="1", start=None, end=None, xref_type="foot")
    assert xref_span(locator, texts, foot) is None


def test_unlocatable_items_stay_null():
    texts = {1: "[equation]"}
    locator = SpanLocator(texts, shared=True)
    assert xref_span(locator, texts, _xref(1, "Table 9")) is None
    assert xref_span(locator, texts, _xref(2, "Table 9")) is None  # unknown sentence
    assert verified_span("abc", 2, 9, "c") is None


def test_links_locate_their_visible_text_or_their_wrapped_url():
    texts = {
        1: "Data are on OSF at https://osf.io/ ab12c.",
        2: "Download the data here.",
    }
    locator = SpanLocator(texts, shared=False)
    wrapped = SimpleNamespace(text_id=1, link_text=None)
    start, end = url_span(locator, wrapped, "https://osf.io/ab12c") or (0, 0)
    assert texts[1][start:end] == "https://osf.io/ ab12c"
    named = SimpleNamespace(text_id=2, link_text="data")
    assert url_span(locator, named, "https://example.org") == (13, 17)


def test_expressions_locate_across_printed_spellings():
    texts = {1: "It held, t (28)= 2.10, p <= .04, and again t(28) = 2.10."}
    locator = SpanLocator(texts, shared=False)

    def eq(lhs, df, comp, rhs):
        return SimpleNamespace(text_id=1, lhs=lhs, df=df, comp=comp, rhs=rhs)

    first = equation_span(locator, eq("t", "28", "=", "2.10"))
    p_value = equation_span(locator, eq("p", "", "≤", ".04"))
    second = equation_span(locator, eq("t", "28", "=", "2.10"))
    assert [texts[1][a:b] for a, b in (first, p_value, second)] == [
        "t (28)= 2.10",
        "p <= .04",
        "t(28) = 2.10",
    ]
    assert equation_span(locator, eq("t", "28", "=", "2.10")) is None  # no third occurrence


def test_export_emits_spans_and_fills_eq_verbatim(demo_paper):
    from bibr.export.json_export import _export_paper_payload

    sentence = demo_paper.contents.sentences[1]
    sentence.text = "It replicated prior work [1], t(28) = 3.42, see Table 1 and data."
    payload = _export_paper_payload(demo_paper)
    text = payload["text"][1]["text"]

    by_type = {x["xref_type"]: x for x in payload["xref"]}
    assert text[by_type["bib"]["start"] : by_type["bib"]["end"]] == "[1]"
    assert text[by_type["table"]["start"] : by_type["table"]["end"]] == "Table 1"
    url = payload["url"][0]
    assert text[url["start"] : url["end"]] == "data"
    eq = payload["eq"][0]
    assert eq["verbatim"] == "t(28) = 3.42" == text[eq["start"] : eq["end"]]
