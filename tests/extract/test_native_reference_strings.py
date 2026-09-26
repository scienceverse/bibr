"""JATS ``<ref>`` strings reach the parser verbatim; HTML list items are still filtered.

A JATS ref-list delimits every reference by markup, so the post-segmentation
junk filter (short entries without a year) and the merged-reference splitter
(in-title citations) could only drop or cut real entries and shift every later
bib_id. The HTML walker, by contrast, collects every list item and text block
under a references heading, page navigation included, so its strings keep the
filter.
"""

from __future__ import annotations

import asyncio
from unittest import mock

from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.extract.ref_locator import RefLocator
from bibr.input.html_native import HtmlParser
from bibr.input.jats_native import JatsParser
from bibr.paper import PaperReference

JATS_REFS = [
    "Brown, T. (2018). Beyond Kahneman and Tversky (1979): Prospect Theory Today. "
    "Econ Review, 5, 1-20.",
    "Aristotle. Nicomachean Ethics. Trans. W. D. Ross.",
    "Doe, A. (2019). Seeing things clearly. Journal of Vision, 2, 11-20.",
    # A sentence-case title citing another work, which the merged-reference
    # splitter still cuts before "Kahneman".
    "Evans, R. (2020). Loss aversion after Kahneman and Tversky (1979). Econ Letters, 7, 3-9.",
]

JATS = (
    b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Prospects</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Introduction</title><p>Text.</p></sec></body>
  <back><ref-list><title>References</title>"""
    + b"".join(
        f'<ref id="r{i}"><mixed-citation>{ref}</mixed-citation></ref>'.encode()
        for i, ref in enumerate(JATS_REFS, start=1)
    )
    + b"""</ref-list></back>
</article>"""
)

HTML = b"""<html><head><title>Prospects</title></head><body>
<h1>Prospects</h1>
<h2>Introduction</h2><p>Text.</p>
<h2>References</h2>
<ol>
  <li>Smith, J. (2020). First reference. Journal A, 1, 1-2.</li>
  <li>Doe, J. (2021). Second reference. Journal B, 2, 3-4.</li>
</ol>
<ul><li>Download BibTeX</li><li>Copy to clipboard</li></ul>
</body></html>"""


def _contents(parser):
    contents = parser.parse()
    segments = [[entry.text] for entry in parser.assembler.entries if entry.needs_segmentation]
    parser.apply_segmentation(contents, segments)
    return contents


def _parsed_segments(contents, segments: list[str] | None = None) -> list[str]:
    """Run reference extraction with the NER parser stubbed; return what it was given.

    *segments*, when given, replaces the segmentation cascade's output.
    """
    seen: list[str] = []

    def parse(self, segments):
        seen.extend(segments)
        return [
            PaperReference(
                bib_id=i,
                title=segment,
                authors=None,
                year=None,
                container=None,
                volume=None,
                first_page=None,
            )
            for i, segment in enumerate(segments, start=1)
        ]

    ref_df = RefLocator(contents).collect_reference_rows()
    extractor = ReferenceExtractor(contents, llm_client=mock.Mock(), parse_strategy="ner")
    with mock.patch.object(ReferenceExtractor, "_parse_references_ner_aligned", parse):
        if segments is None:
            asyncio.run(extractor.extract(ref_df))
        else:
            with mock.patch.object(
                ReferenceExtractor, "_segment_references", mock.AsyncMock(return_value=segments)
            ):
                asyncio.run(extractor.extract(ref_df))
    return seen


def test_jats_reference_strings_are_parsed_verbatim():
    contents = _contents(JatsParser(JATS))

    assert contents.native_ref_strings == JATS_REFS
    assert _parsed_segments(contents) == JATS_REFS
    assert contents.processing_warnings == []


def test_html_reference_list_items_still_drop_page_navigation():
    contents = _contents(HtmlParser(HTML))

    assert contents.native_ref_strings == [
        "Smith, J. (2020). First reference. Journal A, 1, 1-2.",
        "Doe, J. (2021). Second reference. Journal B, 2, 3-4.",
        "Download BibTeX",
        "Copy to clipboard",
    ]
    assert _parsed_segments(contents) == [
        "Smith, J. (2020). First reference. Journal A, 1, 1-2.",
        "Doe, J. (2021). Second reference. Journal B, 2, 3-4.",
    ]


def test_reference_strings_from_another_tier_are_filtered_even_for_jats():
    # A JATS ref-list that yielded no strings leaves segmentation to the
    # cascade, whose output is not one reference per <ref>.
    contents = _contents(HtmlParser(HTML))
    contents.native_ref_strings = []
    contents.native_ref_strings_authoritative = True

    parsed = _parsed_segments(
        contents,
        segments=["Smith, J. (2020). First reference. Journal A, 1, 1-2.", "Download BibTeX"],
    )

    assert parsed == ["Smith, J. (2020). First reference. Journal A, 1, 1-2."]
