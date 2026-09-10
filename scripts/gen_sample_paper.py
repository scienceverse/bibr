# /// script
# requires-python = ">=3.11"
# dependencies = ["fpdf2"]
# ///
"""Generate bibr/data/sample_paper.pdf — a synthetic 2-page "paper" used by
``bibr setup``'s smoke-test step and by the ``@pytest.mark.slow`` end-to-end
test (tests/test_setup_wizard.py).

Every author, affiliation, result, and reference below is invented for this
script; none refers to a real person, institution, or publication, so the
generated PDF is freely redistributable as part of the bibr package.

Run with ``uv run scripts/gen_sample_paper.py`` from the repo root. Output is
byte-stable across runs: the PDF creation timestamp is pinned to a fixed
constant (``_FIXED_DATE`` below) and no other run-to-run entropy (fonts are
fpdf2's built-in core fonts, never subsetted/embedded) leaks into the file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "bibr" / "data" / "sample_paper.pdf"

# Fixed instant so regenerating the PDF produces byte-identical output.
# fpdf2 hashes (content bytes + this timestamp) into the PDF's /ID, and
# writes it verbatim as /CreationDate — so pinning it is both necessary and
# sufficient for reproducibility (see fpdf.fpdf.FPDF._default_file_id).
_FIXED_DATE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

TITLE = "The Coefficient of Rodential Efficiency: A Synthetic Benchmark"
AUTHORS = "Ada Vole and Boris Capybarov"
AFFILIATION = "Department of Synthetic Studies, Institute for Fictional Research"

ABSTRACT = (
    "We introduce the Coefficient of Rodential Efficiency (CRE), a fully synthetic "
    "benchmark statistic invented for software-testing purposes. Using a simulated "
    "sample of burrowing-agent trajectories, we compute CRE across four experimental "
    "groups and compare it to a null baseline. CRE separates the groups with a "
    "reported effect size of d = 0.82. No real animals, participants, or data were "
    "involved: every figure in this document is fabricated to exercise a document "
    "processing pipeline. We conclude that the synthetic benchmark behaves as designed."
)

INTRO_PARAGRAPHS = [
    "Benchmark documents are useful for exercising a text-extraction pipeline end to "
    "end without depending on copyrighted material. This paper is one such document: "
    "a compact, invented paper with a title, two invented authors, an abstract, a "
    "methods section containing one table, a results section, a discussion section, "
    "and a bibliography of ten invented references.",
    "None of the content below describes a real study. The Coefficient of Rodential "
    "Efficiency (CRE) is a placeholder statistic chosen only so that the document "
    "reads like a scientific paper to layout and text-extraction software.",
]

METHODS_PARAGRAPH = (
    "We simulated four groups (A, B, C, and control) with twelve synthetic "
    "observations per group. Table 1 reports the simulated sample characteristics "
    "used to derive the Coefficient of Rodential Efficiency for each group."
)

TABLE_HEADER = ["Group", "N", "Mean CRE", "SD"]
TABLE_ROWS = [
    ["A", "12", "0.91", "0.14"],
    ["B", "12", "0.77", "0.18"],
    ["C", "12", "0.69", "0.21"],
    ["Control", "12", "0.42", "0.16"],
]

RESULTS_PARAGRAPH = (
    "Simulated group A showed the highest mean CRE (M = 0.91, SD = 0.14), followed by "
    "group B (M = 0.77, SD = 0.18) and group C (M = 0.69, SD = 0.21). The control "
    "group produced the lowest mean CRE (M = 0.42, SD = 0.16). A one-way comparison "
    "across the four simulated groups yielded an overall effect size of d = 0.82, "
    "consistent with the synthetic generating process used to produce this benchmark."
)

DISCUSSION_PARAGRAPH = (
    "These simulated results behave exactly as the generating script intended, which "
    "is the only claim this document makes. Because every number above was drawn from "
    "a fixed synthetic process rather than measured, no scientific conclusion should "
    "be drawn from them. The value of this document is purely instrumental: it gives "
    "a document-processing pipeline a small, redistributable, two-page PDF containing "
    "the structural elements (title, authors, abstract, sections, a table, and a "
    "reference list) that such a pipeline is expected to parse."
)

REFERENCES = [
    "Vole, A., & Capybarov, B. (2023). Synthetic coefficients for pipeline testing. "
    "Journal of Fabricated Metrics, 12(3), 145-158. https://doi.org/10.9999/jfm.2023.0145",
    "Marmot, T. Q. (2022). Burrowing-agent simulations for software benchmarks. "
    "Proceedings of the Invented Systems Conference, 4, 22-31.",
    "Capybarov, B., Vole, A., & Nutria, S. (2024). Reproducible placeholder documents "
    "for text-extraction evaluation. Fictional Data Science Review, 8(1), 1-19. "
    "https://doi.org/10.9999/fdsr.2024.0001",
    "Gopher, L. M. (2021). On the invention of benchmark statistics. Synthetic "
    "Benchmarks Quarterly, 3(2), 77-90.",
    "Vole, A. (2020). A short history of made-up coefficients. Institute for "
    "Fictional Research Technical Reports, 2020-01.",
    "Prairie, D., & Marmot, T. Q. (2023). Table extraction stress-tests using "
    "invented tabular data. Journal of Fabricated Metrics, 11(4), 301-312.",
    "Nutria, S. (2019). Two-page documents as unit-test fixtures. Proceedings of the "
    "Invented Systems Conference, 1, 5-14.",
    "Capybarov, B. (2022). Synthetic authorship: naming conventions for fictional "
    "papers. Fictional Data Science Review, 6(2), 88-97.",
    "Beaver, R. K., & Vole, A. (2024). Deterministic PDF generation for regression "
    "testing. Journal of Fabricated Metrics, 13(1), 40-52. "
    "https://doi.org/10.9999/jfm.2024.0040",
    "Marmot, T. Q., Gopher, L. M., & Prairie, D. (2021). A synthetic benchmark corpus "
    "for scientific document parsers. Synthetic Benchmarks Quarterly, 1(1), 1-10.",
]


def build_pdf() -> FPDF:
    pdf = FPDF(format="A4")
    pdf.set_creation_date(_FIXED_DATE)
    pdf.set_creator("bibr sample-paper generator (scripts/gen_sample_paper.py)")
    pdf.set_title(TITLE)
    pdf.set_author(AUTHORS)
    pdf.set_margin(20)
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.multi_cell(0, 8, TITLE, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", style="", size=11)
    pdf.multi_cell(0, 6, AUTHORS, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", style="I", size=10)
    pdf.multi_cell(0, 6, AFFILIATION, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "Abstract", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", style="", size=10)
    pdf.multi_cell(0, 5.5, ABSTRACT, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "1. Introduction", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", style="", size=10)
    for para in INTRO_PARAGRAPHS:
        pdf.multi_cell(0, 5.5, para, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "2. Methods", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", style="", size=10)
    pdf.multi_cell(0, 5.5, METHODS_PARAGRAPH, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", style="I", size=9)
    pdf.cell(
        0, 5, "Table 1. Simulated sample characteristics by group.", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.ln(1)
    pdf.set_font("Helvetica", size=9)
    with pdf.table(
        col_widths=(2, 1, 1.4, 1),
        text_align="CENTER",
        width=150,
    ) as table:
        header = table.row()
        for item in TABLE_HEADER:
            header.cell(item)
        for row_data in TABLE_ROWS:
            row = table.row()
            for item in row_data:
                row.cell(item)
    pdf.ln(3)

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "3. Results", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", style="", size=10)
    pdf.multi_cell(0, 5.5, RESULTS_PARAGRAPH, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "4. Discussion", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", style="", size=10)
    pdf.multi_cell(0, 5.5, DISCUSSION_PARAGRAPH, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 7, "References", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", style="", size=9)
    for i, ref in enumerate(REFERENCES, start=1):
        pdf.multi_cell(0, 5, f"[{i}] {ref}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.5)

    return pdf


def main() -> None:
    pdf = build_pdf()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PATH))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
