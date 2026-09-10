"""Native PDF doc-info metadata harvest.

Born-digital PDFs often carry publisher-set document-info metadata (Title,
Keywords, a DOI in Subject/Keywords). It is free to read and, when sane,
a better backstop than nothing — but doc-info is also routinely junk
("Microsoft Word - draft3.docx", the submitting author's username), so
every field passes a guard before being offered to the fill-empty merge
(``_merge_ocr_metadata``).
"""

from bibr.input.pdf_metadata import harvest_docinfo, harvest_pdf_metadata

PAGE_TEXT = (
    "Attention Is All You Need\n"
    "Ashish Vaswani, Noam Shazeer, Niki Parmar\n"
    "Abstract: The dominant sequence transduction models are based on complex "
    "recurrent or convolutional neural networks that include an encoder and a "
    "decoder. We propose a new simple network architecture, the Transformer."
) * 2  # comfortably above the verification-text minimum


class TestTitleGuards:
    def test_title_accepted_when_printed_on_first_page(self):
        out = harvest_docinfo({"Title": "Attention Is All You Need"}, PAGE_TEXT)
        assert out.get("title") == "Attention Is All You Need"

    def test_title_rejected_when_not_on_first_page(self):
        out = harvest_docinfo({"Title": "Some Entirely Different Manuscript Name"}, PAGE_TEXT)
        assert "title" not in out

    def test_filename_junk_rejected(self):
        for junk in (
            "Microsoft Word - draft_v3.docx",
            "paper_final.tex",
            "untitled",
            "manuscript.pdf",
            "PowerPoint Presentation",
        ):
            out = harvest_docinfo({"Title": junk}, PAGE_TEXT)
            assert "title" not in out, junk

    def test_short_or_letterless_title_rejected(self):
        assert "title" not in harvest_docinfo({"Title": "Foo"}, PAGE_TEXT)
        assert "title" not in harvest_docinfo({"Title": "12345 678"}, PAGE_TEXT)

    def test_title_rejected_without_verification_text(self):
        # No usable first-page text (scanned PDF) → too risky to trust.
        out = harvest_docinfo({"Title": "Attention Is All You Need"}, "")
        assert "title" not in out

    def test_hyphenation_and_linebreaks_tolerated(self):
        # The printed title may be line-wrapped/hyphenated; token-level
        # matching must still accept it.
        page = PAGE_TEXT.replace("Attention Is All You Need", "Atten-\ntion Is All\nYou Need")
        out = harvest_docinfo({"Title": "Attention Is All You Need"}, page)
        assert out.get("title") == "Attention Is All You Need"


class TestDoiAndKeywords:
    def test_doi_harvested_from_subject(self):
        out = harvest_docinfo({"Subject": "doi:10.1234/abc.def-5"}, PAGE_TEXT)
        assert out.get("doi") == "10.1234/abc.def-5"

    def test_doi_harvested_from_keywords_field(self):
        out = harvest_docinfo({"Keywords": "https://doi.org/10.5555/j.tacl.2026.7"}, PAGE_TEXT)
        assert out.get("doi") == "10.5555/j.tacl.2026.7"

    def test_no_doi_no_key(self):
        out = harvest_docinfo({"Subject": "Machine Learning"}, PAGE_TEXT)
        assert "doi" not in out

    def test_keywords_split_on_semicolon_or_comma(self):
        out = harvest_docinfo({"Keywords": "transformers; attention; NLP"}, PAGE_TEXT)
        assert out.get("keywords") == ["transformers", "attention", "NLP"]
        out = harvest_docinfo({"Keywords": "memory, aging"}, PAGE_TEXT)
        assert out.get("keywords") == ["memory", "aging"]

    def test_junk_keywords_rejected(self):
        # Sentence-like / overlong entries are not a keyword list.
        long_entry = "this string is way too long to plausibly be a keyword " * 3
        assert "keywords" not in harvest_docinfo({"Keywords": long_entry}, PAGE_TEXT)
        assert "keywords" not in harvest_docinfo({"Keywords": ""}, PAGE_TEXT)

    def test_empty_docinfo_yields_empty_dict(self):
        assert harvest_docinfo({}, PAGE_TEXT) == {}


def _minimal_pdf_with_info(title: str, subject: str, keywords: str) -> bytes:
    """Handcraft a minimal one-page PDF carrying a doc-info dictionary."""

    def _esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        (
            f"<< /Title ({_esc(title)}) /Subject ({_esc(subject)}) /Keywords ({_esc(keywords)}) >>"
        ).encode(),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 4 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


class TestHarvestFromPdfBytes:
    def test_reads_docinfo_through_pdfium(self):
        pdf = _minimal_pdf_with_info(
            title="Attention Is All You Need",
            subject="doi:10.1234/abc.def",
            keywords="transformers; attention",
        )
        out = harvest_pdf_metadata(pdf, PAGE_TEXT)
        assert out == {
            "title": "Attention Is All You Need",
            "doi": "10.1234/abc.def",
            "keywords": ["transformers", "attention"],
        }

    def test_corrupt_pdf_yields_empty_dict(self):
        assert harvest_pdf_metadata(b"not a pdf at all", PAGE_TEXT) == {}
