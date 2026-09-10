"""Mid-token DOI line-wrap bridging.

The text sent to the parser must preserve a DOI split across lines. Hyphen-structured prefixes and a complete DOI followed by an unrelated token must keep their boundaries."""

from bibr.input.consolidate_text import _bridge_doi_midword_wraps, fix_ocr_artifacts
from bibr.utils.text import normalize_doi


class TestBridgesMidWordDoiWraps:
    def test_bridges_jneuron(self):
        out = fix_ocr_artifacts(
            "Neuron, 89(1), 221-234. https://doi.org/10.1016/j.neu\r\nron.2015.11.028"
        )
        assert "10.1016/j.neuron.2015.11.028" in out
        assert "j.neuron.2015.11.028" in out
        # The bug signature must be gone.
        assert "/neuron.2015" not in out.replace("j.neuron", "X")

    def test_bridges_jneurosci(self):
        out = _bridge_doi_midword_wraps("10.1523/jneuro\r\nsci.3440-12.2013")
        assert out == "10.1523/jneurosci.3440-12.2013"

    def test_bridges_science(self):
        out = _bridge_doi_midword_wraps("10.1126/sci\r\nence.115.2978.77")
        assert out == "10.1126/science.115.2978.77"

    def test_bridges_xge_digit_continuation(self):
        out = _bridge_doi_midword_wraps("10.1037/xge\r\n0000729")
        assert out == "10.1037/xge0000729"

    def test_bridges_evolhumbehav(self):
        out = _bridge_doi_midword_wraps("10.1016/j.evolhum\r\nbehav.2016.03.004")
        assert out == "10.1016/j.evolhumbehav.2016.03.004"

    def test_recovered_doi_normalizes_clean(self):
        joined = _bridge_doi_midword_wraps("10.1016/j.neu\r\nron.2015.11.028")
        assert normalize_doi(joined) == "10.1016/j.neuron.2015.11.028"


class TestDoesNotCorruptDois:
    def test_excludes_annurev_lost_hyphen(self):
        # 10.1146 (Annual Reviews) DOIs are annurev-{subject}-…; the text layer
        # drops the structural hyphen at the wrap, so joining would fabricate a
        # wrong DOI (annurevclinpsy…). Leave it untouched.
        out = _bridge_doi_midword_wraps("10.1146/annurev\r\nclinpsy-032210-104544")
        assert "annurevclinpsy" not in out

    def test_does_not_append_pages_to_complete_doi(self):
        # DOI ends ".x" (separator + single char) — complete; the wrap precedes
        # a page range. Must not absorb "1374".
        out = _bridge_doi_midword_wraps("10.1111/j.1469-8986.2007.00550.x\r\n1374-1385")
        assert "00550.x1374" not in out
        assert "10.1111/j.1469-8986.2007.00550.x" in out

    def test_does_not_join_complete_doi_ending_in_check_letter(self):
        out = _bridge_doi_midword_wraps("10.1017/S003329171800380X\r\ndecision making")
        assert "380Xdecision" not in out

    def test_does_not_join_complete_doi_ending_separator_letter(self):
        out = _bridge_doi_midword_wraps("10.1038/s41593-020-00742-z\r\nverbal report")
        assert "00742-zverbal" not in out

    def test_does_not_join_into_lowercase_name_particle(self):
        # DOI ends in two letters but the continuation ("van der Berg") is a new
        # reference, not a DOI continuation (no dot/digit in the token).
        out = _bridge_doi_midword_wraps("10.1234/abcdef\r\nvan der Berg, J. (2020).")
        assert "abcdefvan" not in out

    def test_does_not_join_into_capitalized_next_reference(self):
        out = _bridge_doi_midword_wraps("10.1016/j.neuron\r\nSmith, J. (2020).")
        assert "neuronSmith" not in out


class TestBridgesDotWrappedDoi:
    def test_bridges_plos_citation_dot_space(self):
        # verbatim from predictions/10.1371_journal.pone.0279511.json text[]
        text = (
            "Citation: Cintado MA´, Gonza´lez G, Ca´rcel L, De la Casa LG (2023) "
            "Unconditioned and conditioned anxiolytic effects of Sodium Valproate on "
            "flavor neophobia and fear conditioning. PLoS ONE 18(7): e0279511. "
            "https://doi.org/10.1371/journal. pone.0279511"
        )
        assert "10.1371/journal.pone.0279511" in fix_ocr_artifacts(text)

    def test_bridges_plos_marker_dot_space(self):
        # verbatim from predictions/10.1371_journal.pone.0130688.json text[]
        text = "PLoS ONE 10(6): e0130688. doi:10.1371/journal. pone.0130688"
        assert "10.1371/journal.pone.0130688" in fix_ocr_artifacts(text)

    def test_bridges_dot_newline_wrap(self):
        text = "available at 10.1371/journal.\npone.0130688 online"
        assert "10.1371/journal.pone.0130688" in fix_ocr_artifacts(text)

    def test_does_not_join_sentence_after_doi(self):
        text = "See https://doi.org/10.1234/abcd. the study found nothing."
        assert "10.1234/abcd.the" not in fix_ocr_artifacts(text)

    def test_does_not_join_capitalized_next_line(self):
        text = "https://doi.org/10.1234/abcd.\nNature Publishing Group\n"
        assert "abcd.Nature" not in fix_ocr_artifacts(text)

    def test_does_not_join_next_reference(self):
        text = "doi: 10.1371/journal.pgen.1004048. 27. Fischer B, Metzger M. p53 and TAp63."
        assert "1004048.27" not in fix_ocr_artifacts(text)

    def test_does_not_join_ocr_garbage_after_ref_doi(self):
        # verbatim from predictions/10.3390_vaccines12091066.json text[]
        text = "https://doi.org/10.1093/cid/ciab633. 5KlRMMiltiAIflfCOVID19i’fftidftfilititP 5. Kaplan,"
        assert "ciab633.5KlRMM" not in fix_ocr_artifacts(text)

    def test_bridges_uppercase_doi_tail(self):
        assert "10.1371/journal.pone.0033-295X" in fix_ocr_artifacts(
            "doi: 10.1371/journal.pone. 0033-295X"
        )

    def test_bridges_pure_digit_tail(self):
        assert "10.1371/journal.pone.0233061" in fix_ocr_artifacts(
            "doi: 10.1371/journal.pone. 0233061"
        )

    def test_does_not_join_four_digit_reference_number(self):
        assert "1004048.1027" not in fix_ocr_artifacts(
            "doi: 10.1371/journal.pgen.1004048. 1027. Fischer B."
        )

    def test_does_not_join_comma_separated_reference_number(self):
        assert "1004048.27," not in fix_ocr_artifacts(
            "doi: 10.1371/journal.pgen.1004048. 27, Fischer B."
        )

    def test_does_not_join_numbered_heading(self):
        assert "abcd.3.1" not in fix_ocr_artifacts("See https://doi.org/10.1234/abcd.\n3.1 Design")

    def test_repairs_doi_wrapped_at_two_dots(self):
        # fixpoint loop, matching the sibling bridges
        text = "https://doi.org/10.1371/journal. pone. 0279511"
        assert "10.1371/journal.pone.0279511" in fix_ocr_artifacts(text)

    def test_bridges_doi_with_four_digit_tail(self):
        # verbatim from predictions/10.1186_s12872-021-02446-z.json text[]:
        # a real Circulation DOI whose final component looks like a reference number.
        text = "https://doi.org/10.1161/01.cir.83.5. 1832."
        assert "10.1161/01.cir.83.5.1832" in fix_ocr_artifacts(text)


class TestBridgesSlashSpaceDoi:
    def test_bridges_vetrec_slash_space(self):
        # verbatim from predictions/10.1136_vr.105253.json text[]
        text = "doi:10.1136/ vetrec-2018-105253"
        assert "10.1136/vetrec-2018-105253" in fix_ocr_artifacts(text)

    def test_does_not_join_prose_after_slash(self):
        text = "either 10.1136/ or another registrant entirely"
        assert "10.1136/or" not in fix_ocr_artifacts(text)
