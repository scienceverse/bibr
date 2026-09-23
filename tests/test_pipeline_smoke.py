"""End-to-end smoke test: ``LocalPipeline.process_file`` over a generated 1-page PDF.

Runs the real stage list (validate → docx → layout → native_text → ocr →
llm_server → parse → extract → enrich → export) with three substitutions,
all injected behind the existing ``ResourceManager`` seams:

- layout detection: canned region boxes (no PP-DocLayoutV3 weights)
- sentence segmentation: naive period splitter (no wtpsplit download)
- LLM transport: ``LLMClient._invoke_structured`` returns canned Pydantic
  responses (no network) — prompt building, response validation, and all
  extractor post-processing still run through the production code.

The generated PDF carries a native text layer, so ``NativeTextStage``
pre-fills every region and the fake OCR backend asserts it is never called.
"""

import copy

from bibr.config import Settings
from bibr.schemas import AuthorLLM, PaperReferenceLLM

_PAGE_W, _PAGE_H = 612, 792

_TITLE = "Tiny Smoke Paper on Stubbed Pipelines"
_ABSTRACT = (
    "This study examines end-to-end smoke testing of extraction pipelines. "
    "We find that stubbed models preserve the production data flow."
)


# ---------------------------------------------------------------------------
# Minimal hand-built PDF with a native text layer
# ---------------------------------------------------------------------------


def _pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _build_pdf(lines: list[tuple[float, float, float, str]]) -> bytes:
    """Build a single-page PDF; ``lines`` are (x, baseline_y, font_size, text) in points."""
    parts = [
        f"BT /F1 {size:g} Tf 1 0 0 1 {x:g} {y:g} Tm ({_pdf_escape(text)}) Tj ET"
        for x, y, size, text in lines
    ]
    stream = "\n".join(parts).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


_LINES = [
    (72, 720, 18, _TITLE),
    (72, 700, 12, "Jane Q. Doe"),
    (72, 680, 14, "Abstract"),
    (72, 660, 10, "This study examines end-to-end smoke testing of extraction pipelines."),
    (72, 646, 10, "We find that stubbed models preserve the production data flow."),
    (72, 600, 14, "Introduction"),
    (72, 580, 10, "Smoke tests catch wiring regressions early. They must run in seconds."),
    (72, 566, 10, "We describe a fixture-driven approach for the bibr pipeline."),
    (72, 520, 14, "References"),
    (72, 500, 10, "Doe, J. (2020). Smoke testing pipelines. Journal of Tests, 5, 1-10."),
    (72, 486, 10, "doi:10.1234/jt.2020.001"),
    (72, 466, 10, "Roe, A. (2021). Stubs and fakes in CI. Testing Review, 7(2), 11-20."),
]


def _norm_bbox(x1: float, y_top: float, x2: float, y_bottom: float) -> list[int]:
    """PDF points (y-up) → normalized 0..1000 y-down bbox, as the layout detector emits."""
    return [
        round(x1 / _PAGE_W * 1000),
        round((_PAGE_H - y_top) / _PAGE_H * 1000),
        round(x2 / _PAGE_W * 1000),
        round((_PAGE_H - y_bottom) / _PAGE_H * 1000),
    ]


def _layout_region(index: int, label: str, bbox: list[int]) -> dict:
    return {
        "index": index,
        "label": label,
        "task_type": "text",
        "score": 0.99,
        "bbox_2d": bbox,
        "read_order": index,
        "content": "",
    }


_REGIONS = [
    _layout_region(0, "doc_title", _norm_bbox(60, 740, 552, 705)),
    _layout_region(1, "text", _norm_bbox(60, 706, 552, 694)),
    _layout_region(2, "paragraph_title", _norm_bbox(60, 698, 300, 672)),
    _layout_region(3, "text", _norm_bbox(60, 672, 552, 638)),
    _layout_region(4, "paragraph_title", _norm_bbox(60, 618, 300, 592)),
    _layout_region(5, "text", _norm_bbox(60, 592, 552, 558)),
    _layout_region(6, "paragraph_title", _norm_bbox(60, 538, 300, 512)),
    _layout_region(7, "reference", _norm_bbox(60, 512, 552, 456)),
]


# ---------------------------------------------------------------------------
# Fakes injected behind the ResourceManager seams
# ---------------------------------------------------------------------------


class FakeLayout:
    loaded = True

    async def detect_batch(self, images_list):
        return [copy.deepcopy(_REGIONS) for _ in images_list]


class FakeOcr:
    loaded = True

    async def recognize(self, image, prompt):
        raise AssertionError("OCR backend must not be called — native text covers all regions")


def _split_sentences(text: str) -> list[str]:
    sentences = []
    current = ""
    for ch in text:
        current += ch
        if ch == ".":
            sentences.append(current)
            current = ""
    if current.strip():
        sentences.append(current)
    return sentences


class FakeSegmenter:
    loaded = True

    async def segment_batch(self, texts):
        return [_split_sentences(t) for t in texts]


class _NoopLimiter:
    async def acquire(self):
        return None


def _fake_llm_client():
    from bibr.clients.llm import LLMClient

    class FakeLLMClient(LLMClient):
        def __init__(self):
            super().__init__()
            self._limiter = _NoopLimiter()
            self.calls: list[str] = []

        async def close(self):
            return None

        async def _invoke_structured(
            self,
            response_model,
            messages,
            system_prompt,
            reasoning_effort=None,
            client_override=None,
            max_tokens=None,
        ):
            name = response_model.__name__
            self.calls.append(name)
            if name == "TitleKeywordsLLM":
                return response_model(
                    title=_TITLE,
                    abstract=_ABSTRACT,
                    keywords=["smoke testing", "pipelines"],
                )
            if name == "AuthorsLLM":
                return response_model(
                    authors=[
                        AuthorLLM(
                            given="Jane Q",
                            family="Doe",
                            affiliation="University of Testing",
                        )
                    ]
                )
            if name == "PaperClassificationLLM":
                return response_model(
                    oecd_domain="Social Sciences",
                    oecd_subdomain="Psychology and Cognitive Sciences",
                    paper_type="empirical",
                )
            if name == "RefAnchors":
                return response_model(anchors=["Doe, J. (2020).", "Roe, A. (2021)."])
            if name == "PaperReferenceList":
                return response_model(
                    references=[
                        PaperReferenceLLM(
                            index=1,
                            authors="Doe, J.",
                            year=2020,
                            title="Smoke testing pipelines",
                            container="Journal of Tests",
                            volume="5",
                            issue=None,
                            first_page="1",
                            last_page="10",
                            doi="10.1234/jt.2020.001",
                            bib_type="journal_article",
                        ),
                        PaperReferenceLLM(
                            index=2,
                            authors="Roe, A.",
                            year=2021,
                            title="Stubs and fakes in CI",
                            container="Testing Review",
                            volume="7",
                            issue="2",
                            first_page="11",
                            last_page="20",
                            doi=None,
                            bib_type="journal_article",
                        ),
                    ]
                )
            if name == "SectionClassificationResult":
                return response_model(classifications=[])
            if name == "FrontMatterResult":
                return response_model(segments=[])
            if name == "CitationResolutionResult":
                return response_model(matches=[])
            if name == "EquationExtractionResult":
                return response_model(equations=[])
            raise AssertionError(f"Unexpected LLM call for response model {name}")

    return FakeLLMClient()


# ---------------------------------------------------------------------------
# The smoke test
# ---------------------------------------------------------------------------


async def test_local_pipeline_end_to_end_smoke(tmp_path, monkeypatch):
    from bibr.export.json_export import validate_export
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ocr, "native_text_min_chars", 4)
    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", None)

    pdf_path = tmp_path / "smoke.pdf"
    pdf_path.write_bytes(_build_pdf(_LINES))

    pipeline = LocalPipeline(crossref=False)
    rm = ResourceManager(
        memory_mode=pipeline.memory_mode,
        ocr_backend=pipeline.ocr_backend,
        ocr_url=pipeline.ocr_url,
        ocr_model=pipeline.ocr_model,
        device=pipeline.device,
        settings=pipeline.settings,
        layout=FakeLayout(),
        segmenter=FakeSegmenter(),
    )
    pipeline._resources = rm
    rm._ocr = FakeOcr()
    llm = _fake_llm_client()
    rm._llm_client = llm

    result = await pipeline.process_file(pdf_path)

    # The ready injected OCR fake has no concrete runtime identity. The fused
    # cache probe must preserve this test seam with a static Paddle identity,
    # rather than attempting managed startup or failing before native text can
    # bypass OCR.
    assert result["extraction"]["ocr"]["backend"] == "paddle"
    assert result["extraction"]["ocr"]["profile"] == "paddle"

    assert {
        "TitleKeywordsLLM",
        "AuthorsLLM",
        "PaperClassificationLLM",
        "RefAnchors",
        "PaperReferenceList",
    } <= set(llm.calls)

    assert validate_export(result) == []
    expected_keys = {
        "paper_id",
        "schema_version",
        "source",
        "metadata",
        "author",
        "text",
        "section",
        "url",
        "bib",
        "xref",
        "figure",
        "table",
        "eq",
        "bib_match",
        "extraction",
    }
    assert expected_keys <= set(result.keys())

    assert result["metadata"]["title"] == _TITLE
    assert result["metadata"]["abstract"].startswith("This study examines")
    assert result["source"]["input_format"] == "pdf"
    assert result["metadata"]["paper_type"] == "empirical"

    headers = {s["header"] for s in result["section"]}
    assert {"Abstract", "Introduction", "References"} <= headers
    section_types = {s["section_type"] for s in result["section"]}
    assert {"abstract", "intro", "references"} <= section_types

    assert result["text"], "expected non-empty sentence list"
    all_text = " ".join(t["text"] for t in result["text"])
    assert "Smoke tests catch wiring regressions early" in all_text
    assert all(t["text_id"] is not None for t in result["text"])

    assert len(result["author"]) == 1
    assert result["author"][0]["family"] == "Doe"

    assert len(result["bib"]) == 2
    assert result["bib"][0]["doi"] == "10.1234/jt.2020.001"
    assert [r["year"] for r in result["bib"]] == [2020, 2021]

    # The identity receipt names the layout region each sentence DOI was read
    # from, by its ``extraction.regions`` (page, index) key.
    [reference_doi] = [
        candidate
        for candidate in result["extraction"]["identity"]["receipt"]["candidates"]
        if candidate["normalized"] == "10.1234/jt.2020.001"
    ]
    assert (reference_doi["page"], reference_doi["region_index"]) == (1, 7)
    assert reference_doi["region_type"] == "reference"

    # The run had an LLM, so the engine is reported rather than null.
    assert result["extraction"]["llm"]["provider"]
    # Gate findings live only in ``validation.issues`` — never mirrored into
    # warnings — and this smoke run emits no genuine processing warnings.
    assert result["extraction"]["warnings"] == []
    assert "VAL_ABSTRACT_SUSPECT" in {
        i["code"] for i in result["extraction"]["validation"]["issues"]
    }

    await pipeline.aclose()
