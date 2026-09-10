"""Tests for the per-reference DOI rescue (residual #4).

Abdi (2010) prints 'doi:10.4135/9781412961288.n83' on its final line, but the
batched LLM parser stochastically emitted doi=None. normalize_doi accepts the
DOI fine, so the loss is an LLM under-emission, not a validator rejection. The
rescue recovers a DOI from a clean doi:/doi.org token in the per-reference
segment, and — only when that strict form fails — rejoins one broken across an
OCR line wrap. A wrap continues the identifier mid-token, so the scan bridges a
gap only into a digit or a lowercase letter; anything else starts a new field
and the rescue declines rather than emitting a corrupt, non-resolving DOI.
"""

from bibr.extract.extractor import _rescue_doi_from_segment


class TestRescueDoiFromSegment:
    def test_recovers_clean_end_of_entry_chapter_doi(self):
        seg = (
            "Abdi, H., & Williams, L. J. (2010). Correspondence analysis. "
            "In N. J. Salkind (Ed.), Encyclopedia of Research Design "
            "(Vol. 1, pp. 267-278). Thousand Oaks, CA: Sage. "
            "doi:10.4135/9781412961288.n83"
        )
        assert _rescue_doi_from_segment(seg) == "10.4135/9781412961288.n83"

    def test_recovers_doi_org_url_form(self):
        seg = "Smith, J. (2020). A paper. Journal, 1, 2. https://doi.org/10.1037/abc123"
        assert _rescue_doi_from_segment(seg) == "10.1037/abc123"

    def test_rejoins_doi_wrapped_after_the_slash(self):
        seg = "Downing, P. (2001). A cortical area. Science. doi:10.1126/ science.1063414"
        assert _rescue_doi_from_segment(seg) == "10.1126/science.1063414"

    def test_rejoins_doi_wrapped_mid_suffix(self):
        seg = "Doe, A. (2019). A paper. https://doi.org/10.7717/ peerj. 1234"
        assert _rescue_doi_from_segment(seg) == "10.7717/peerj.1234"

    def test_rejoins_doi_wrapped_after_an_internal_hyphen(self):
        # The wrap falls after a hyphen the DOI genuinely contains, so the
        # whitespace-free token is "10.1037/0033-" — which, unlike the
        # wrapped-after-slash shape above, still satisfies the bare-DOI pattern.
        #
        # Serialized as a literal space, which is how this reaches the parser:
        # a newline wrap is already rejoined upstream by _bridge_doi_url_crlf_wraps,
        # but glmocr also emits a printed wrap as a space, and the space bridge
        # (_DOI_SPACE_WRAP_RE) only rejoins a continuation that starts ".<digit>".
        # A hyphen break continues with a bare digit, so it escapes every OCR-time
        # bridge and this rescue is the only layer left that can recover it.
        seg = (
            "Miller, N. E. (1994). Some thoughts. Psychological Bulletin, "
            "115(1), 102-115. doi:10.1037/0033- 2909.115.1.102"
        )
        assert _rescue_doi_from_segment(seg) == "10.1037/0033-2909.115.1.102"

    def test_declines_when_the_continuation_starts_a_new_field(self):
        # A capital after the gap is an author initial / "In:" / a container
        # title, never the rest of an identifier. "10.1126/" stays invalid and
        # the rescue declines rather than swallowing the next field.
        seg = "Downing, P. (2001). A cortical area. doi:10.1126/ Science 293, 2470."
        assert _rescue_doi_from_segment(seg) is None

    def test_strict_token_wins_over_the_wrap_scan(self):
        # The whitespace-free token is already a valid DOI, so the trailing
        # year/volume must not be joined onto it.
        seg = "Doe, A. doi:10.1234/abc 2019;12:3-5."
        assert _rescue_doi_from_segment(seg) == "10.1234/abc"

    def test_returns_none_when_no_doi_token(self):
        seg = "Bolton, P. (2012). Education: Historical statistics. Retrieved from http://x.gov"
        assert _rescue_doi_from_segment(seg) is None

    def test_returns_none_on_empty_segment(self):
        assert _rescue_doi_from_segment("") is None
        assert _rescue_doi_from_segment(None) is None


class TestRescueDoiFromPublisherUrl:
    """MDPI/DG style 'Available online:' entries print the DOI only inside a
    publisher URL path — recover it from an explicit /doi/ path marker."""

    def test_degruyter_document_path_with_pdf_suffix(self):
        seg = (
            "Mosteller, Frederick. 1995. The Tennessee Study of Class Size. "
            "Available online: https://www.degruyter.com/document/doi/"
            "10.1515/9781400851607.261/pdf (accessed on 11 April 2022)."
        )
        assert _rescue_doi_from_segment(seg) == "10.1515/9781400851607.261"

    def test_tandfonline_doi_full_path(self):
        seg = (
            "Ungar, Orit Avidov. 2020. The professional learning expectations. "
            "Available online: https://www.tandfonline.com/doi/full/"
            "10.1080/19415257.2020.1763435 (accessed on 26 July 2021)."
        )
        assert _rescue_doi_from_segment(seg) == "10.1080/19415257.2020.1763435"

    def test_wiley_legacy_abstract_suffix(self):
        seg = "http://onlinelibrary.wiley.com/doi/10.1111/1467-6419.00056/abstract"
        assert _rescue_doi_from_segment(seg) == "10.1111/1467-6419.00056"

    def test_url_fragment_stripped(self):
        seg = "http://www.tandfonline.com/doi/abs/10.1080/00036840802599784#.VP2pU1OW9I"
        assert _rescue_doi_from_segment(seg) == "10.1080/00036840802599784"

    def test_percent_encoding_decoded(self):
        # Case is preserved (DOIs are case-insensitive; we keep the printed form)
        seg = "https://doi.org/10.1016/S0731-9053%2800%2915007-8"
        assert _rescue_doi_from_segment(seg) == "10.1016/S0731-9053(00)15007-8"

    def test_plain_publisher_url_without_doi_path_declined(self):
        # No /doi/ marker — never harvest a bare 10.x token out of an
        # arbitrary URL (could be a page id, not a DOI).
        seg = "Available online: https://www.jstor.org/stable/10.2307/1912352 (accessed 2020)."
        assert _rescue_doi_from_segment(seg) is None
