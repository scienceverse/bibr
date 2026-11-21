"""Regression & exactness tests for Fix 2C (PDFium, NativeText, TextLayer, Identity)."""

from __future__ import annotations

import random
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bibr.extract.pdf_doi_evidence import read_pdf_doi_evidence
from bibr.ocr import native_text as nt
from bibr.ocr import pdf_inspection as pi
from bibr.ocr import text_layer as tl
from bibr.ocr import text_layer_quality as tlq
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.identity import _pdf_doi_evidence
from bibr.pipeline.stages.native_text import NativeTextStage
from bibr.pipeline.state import FileState
from tests.document._pdfs import band_layout, fixture_pdfs, page_count


def test_indexed_records_reconstruct_matches_plain_list_exactly():
    rng = random.Random(2026)  # noqa: S311 - deterministic synthetic test data
    raw: list[tuple[str, float, float, bool]] = []
    for line in range(25):
        y = 750.0 - line * 14.0
        for col in range(60):
            ch = rng.choice("abcde fghij 12345.,-")
            x = 50.0 + col * 8.0 + rng.uniform(-0.5, 0.5)
            raw.append((ch, x, y + rng.uniform(-1.0, 1.0), False))
        # Include single, double, and trailing newlines
        for _ in range(rng.choice([1, 1, 2])):
            raw.append(("\n", 0.0, 0.0, True))

    indexed = nt._IndexedRecords(raw)
    assert indexed == raw

    for _ in range(150):
        x0 = rng.uniform(0.0, 600.0)
        x1 = x0 + rng.uniform(0.0, 400.0)
        y0 = rng.uniform(350.0, 770.0)
        y1 = y0 + rng.uniform(0.0, 200.0)
        assert nt._reconstruct_text_from_records(
            indexed, x0, y0, x1, y1
        ) == nt._reconstruct_text_from_records(raw, x0, y0, x1, y1)

    # Mutation invalidates cached runs cleanly.
    extra = ("Z", 999.0, 999.0, False)
    indexed.append(extra)
    raw.append(extra)
    assert nt._reconstruct_text_from_records(
        indexed, 990.0, 990.0, 1010.0, 1010.0
    ) == nt._reconstruct_text_from_records(raw, 990.0, 990.0, 1010.0, 1010.0)


def test_lexical_profile_matches_naive_unicodedata_reference():
    texts = [
        "The empirical analysis of résumé, naïve, Übertragung, Łódź, and café studies "
        "shows that common Latin letters and function words occur at expected frequencies. "
        "We also include Greek αβγδ and Cyrillic текст tokens alongside ASCII prose so "
        "mixed-script counting is exercised thoroughly across more than two hundred letters.",
    ]

    def naive_is_latin(ch: str) -> bool:
        try:
            return unicodedata.name(ch).startswith("LATIN")
        except ValueError:
            return False

    def naive_base(ch: str) -> str:
        d = unicodedata.normalize("NFKD", ch)
        return (d[0] if d else ch).lower()

    tokens = [t for text in texts for t in tlq._TOKEN_RE.findall(text)]
    letters = sum(len(t) for t in tokens)
    latin = [ch for t in tokens for ch in t if naive_is_latin(ch)]
    expected_common = sum(1 for ch in latin if naive_base(ch) in tlq.COMMON_LETTERS) / len(latin)

    prof = tlq.lexical_profile(texts)
    assert prof["tokens"] == float(len(tokens))
    assert prof["common_letter_share"] == pytest.approx(expected_common, rel=0, abs=1e-15)
    assert prof["glued_letter_share"] == 0.0
    assert letters >= tlq.MIN_LEXICAL_LETTERS


def test_judge_calls_covered_regions_once(monkeypatch):
    calls = 0
    real_covered = tlq.covered_regions

    def counting_covered(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_covered(*args, **kwargs)

    monkeypatch.setattr(tlq, "covered_regions", counting_covered)
    page = tl._PageText(
        index=0,
        crop_box=(0.0, 0.0, 612.0, 792.0),
        rotation=0,
        char_count=10,
        invisible=False,
        records=[],
        chars=[("a", (10.0, 10.0, 20.0, 20.0)), ("b", (25.0, 10.0, 35.0, 20.0))],
        lines=[],
    )
    region = {"label": "text", "bbox_2d": [0.0, 0.0, 100.0, 100.0], "content": "ab"}
    res = tl._judge(page, [region], {id(region): "filled"})
    assert calls == 1
    assert res.metrics["coverage"] == 1.0
    assert res.metrics["region_coverage"] == 1.0


@pytest.mark.parametrize("name", sorted(fixture_pdfs()))
def test_inspect_pdf_inline_doi_evidence_matches_standalone(name):
    pdf = fixture_pdfs()[name]
    n_pages = page_count(pdf)
    layout = band_layout(n_pages)
    for reverse in (False, True):
        indices = list(range(n_pages))[::-1] if reverse else list(range(n_pages))
        insp = pi.inspect_pdf(
            pdf,
            layout,
            page_indices=indices,
            fill_native_text=True,
            include_outline=False,
            include_ref_geometry=True,
            min_chars=20,
            min_printable_ratio=0.85,
            reject_invisible_text_layer=True,
        )
        expected = read_pdf_doi_evidence(pdf, (1, 2))
        assert insp.pdf_doi_evidence == expected


async def test_identity_reuses_inline_doi_evidence_without_reopening_pdf(monkeypatch):
    pdf = fixture_pdfs()["linked_paper.pdf"]
    n_pages = page_count(pdf)
    layout = band_layout(n_pages)
    fs = FileState(path=Path("linked_paper.pdf"))
    fs.pdf_bytes = pdf
    fs.layout_results = layout
    fs.page_indices = list(range(n_pages))
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(spec=[]),
        config=RunConfig(),
    )
    await NativeTextStage().run(ctx)
    assert fs.pdf_doi_evidence is not None

    # Simulate post-OCR memory release: pdf_bytes and pdf_inspection are freed,
    # but pdf_doi_evidence survives until free_all().
    fs.free_pre_ocr()
    assert fs.pdf_bytes is None
    assert fs.pdf_inspection is None
    assert fs.pdf_doi_evidence is not None

    def must_not_reopen(*_args, **_kwargs):
        raise AssertionError("Identity stage must reuse inline pdf_doi_evidence")

    monkeypatch.setattr("bibr.pipeline.stages.identity.read_pdf_doi_evidence", must_not_reopen)

    # Full pages (1, 2) match
    fs.paper = SimpleNamespace(
        contents=SimpleNamespace(
            preparsed_metadata=None,
            sentences=[SimpleNamespace(page_number=1), SimpleNamespace(page_number=2)],
            region_summaries=[],
        )
    )
    ev_both = _pdf_doi_evidence(fs)
    assert ev_both == read_pdf_doi_evidence(pdf, (1, 2))

    # Subset page (1,) filters cleanly without reopening
    fs.paper = SimpleNamespace(
        contents=SimpleNamespace(
            preparsed_metadata=None,
            sentences=[SimpleNamespace(page_number=1)],
            region_summaries=[],
        )
    )
    ev_one = _pdf_doi_evidence(fs)
    assert ev_one == read_pdf_doi_evidence(pdf, (1,))

    fs.free_all()
    assert fs.pdf_doi_evidence is None


def test_text_layer_stream_defers_read_page_outside_pdfium_lock(monkeypatch):
    pdf = fixture_pdfs()["sample_paper.pdf"]
    n_pages = page_count(pdf)
    layout = band_layout(n_pages)
    lock_held = {"active": False}
    read_page_saw_lock: list[bool] = []

    real_lock = pi.pdfium_lock
    real_read_page = tl._read_page

    class SpyLock:
        def __enter__(self):
            real_lock.__enter__()
            lock_held["active"] = True
            return self

        def __exit__(self, *args):
            lock_held["active"] = False
            return real_lock.__exit__(*args)

    def spy_read_page(*args, **kwargs):
        read_page_saw_lock.append(lock_held["active"])
        return real_read_page(*args, **kwargs)

    monkeypatch.setattr(pi, "pdfium_lock", SpyLock())
    monkeypatch.setattr(tl, "_read_page", spy_read_page)

    stream = tl.TextLayerStream(list(range(n_pages)))
    insp = pi.inspect_pdf(
        pdf,
        layout,
        page_indices=list(range(n_pages)),
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=20,
        min_printable_ratio=0.85,
        reject_invisible_text_layer=True,
        text_layer=stream,
    )
    res = stream.finish(insp.layout_results)
    assert res is not None
    assert len(read_page_saw_lock) == n_pages
    assert not any(read_page_saw_lock)
