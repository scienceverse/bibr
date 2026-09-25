from bibr.extract import merge_split
from bibr.extract.merge_split import detect_merges, find_interior_onsets, split_merged_refs

# --- merged segments that MUST split (one onset each) ---
P_PERSONAL = (
    "Kahneman, D., & Tversky, A. (1982). The psychology of preferences. "
    "Scientific American, 246, 160-173. Kühberger, A. (1998). The influence of "
    "framing on risky decisions: A meta-analysis. Organizational Behavior and "
    "Human Decision Processes, 75, 23-55."
)
P_CORP_TERM = (
    "Shah, A., Mullainathan, S., & Shafir, E. (2012). Some consequences of having "
    "too little. Science, 338, 682-685. Social and Behavioral Sciences Team. (2015). "
    "Social and behavioral sciences team annual report. Executive Office of the President."
)
P_PERSONAL2 = (
    "Sugita, Y., & Suzuki, Y. (2003). Implicit estimation of sound-arrival time. "
    "Nature, 421, 911. Summerfield, C., & de Lange, F. P. (2014). Expectation in "
    "perceptual decision making. Nature Reviews Neuroscience, 15, 745-756."
)
P_URL_NDOT = (
    "Servick, K. (2017). GM banana shows promise against deadly fungus strain. "
    "Science. https://www.sciencemag.org/news/2017/11/gm-banana-shows-promise-"
    "against-deadly-fungus-strain United Nations. (n.d.). Global issues: Food. "
    "https://www.un.org/en/global-issues/food"
)
P_DIACRITIC = (
    "Bruce, V., & Young, A. (1986). Understanding face recognition. British Journal "
    "of Psychology, 77, 305-327. Bürkner, P.-C. (2017). brms: An R package for "
    "Bayesian multilevel models using Stan. Journal of Statistical Software, 80, 1-28."
)
P_CORP2 = (
    "U.S. Office of Management and Budget. (2010). 2010 standards for delineating "
    "metropolitan and micropolitan statistical areas. Federal Register, 75, 37246-37252. "
    "U.S. Social Security Administration. (2023). Top 10 baby names. "
    "https://www.ssa.gov/oact/babynames/"
)
MERGED = [P_PERSONAL, P_CORP_TERM, P_PERSONAL2, P_URL_NDOT, P_DIACRITIC, P_CORP2]

# --- genuine single references that MUST NOT split ---
N_ORIG_WORK = (
    "Freud, S. (1953). The interpretation of dreams (J. Strachey, Trans.). "
    "Basic Books. (Original work published 1900)."
)
N_REPRINT = (
    "Watson, J. B. (1994). Psychology as the behaviorist views it. Psychological "
    "Review, 101, 248-253. (Reprinted from Psychological Review, 1913, 20, 158-177)"
)
N_EDITION = (
    "American Psychological Association. (2020). Publication manual of the American "
    "Psychological Association (7th ed.). Author."
)
N_TITLE_YEAR = (
    "Reinhart, C. M., & Rogoff, K. S. (2009). Lessons from the 2008 financial "
    "crisis. American Economic Review, 99, 466-472."
)
N_CLEAN_SINGLE = (
    "Tversky, A., & Kahneman, D. (1981). The framing of decisions and the "
    "psychology of choice. Science, 211, 453-458."
)
# two bare anchors but a meta-word in between → must NOT split
N_META_BARE = (
    "Skinner, B. F. (1948). Superstition in the pigeon. Journal of Experimental "
    "Psychology, 38, 168-172. Reprinted in Cumulative Record (1992)."
)
# two bare anchors but the second is preceded by lowercase prose → must NOT split
N_PROSE = (
    "Author, A. (2019). A study of outcomes measured in 2010 versus baseline "
    "collected earlier (2005)."
)
SINGLES = [
    N_ORIG_WORK,
    N_REPRINT,
    N_EDITION,
    N_TITLE_YEAR,
    N_CLEAN_SINGLE,
    N_META_BARE,
    N_PROSE,
]


N_INTITLE_CORRECTION = (
    "Morey, R. D. (2008). Confidence intervals from normalized data: A correction "
    "to Cousineau (2005). Tutorials in Quantitative Methods for Psychology, 4, 61-64."
)
N_INTITLE_COMMENT = (
    "Wagenmakers, E.-J., Wetzels, R., Borsboom, D., & van der Maas, H. L. J. (2011). "
    "Why psychologists must change the way they analyze their data: The case of psi: "
    "Comment on Bem (2011). Journal of Personality and Social Psychology, 105, 426-432."
)
N_INTITLE_REPLY = (
    "Levine, L. J., Lench, H. C., Kaplan, R. L., & Safer, M. A. (2013). Like "
    "Schrodinger's cat, the impact bias is both dead and alive: Reply to Wilson and "
    "Gilbert (2013). Journal of Personality and Social Psychology, 105, 74-84."
)
N_INTITLE_COMMENT_ON = (
    "Rothstein, J. (2007). Does competition among public schools benefit students "
    "and taxpayers? A comment on Hoxby (2000). American Economic Review, 97, 2026-2037."
)
INTITLE_SINGLES = [
    N_INTITLE_CORRECTION,
    N_INTITLE_COMMENT,
    N_INTITLE_REPLY,
    N_INTITLE_COMMENT_ON,
]

# --- REAL merges whose ref1 TITLE contains a citation verb far from the 2nd date ---
# The guard must NOT suppress these (substring "responses"/"Corrections"/"A comment.").
M_VERB_RESPONSES = (
    "Spence, K. W., & Ross, L. E. (1959). A methodological study of the form and "
    "latency of eyelid responses. Journal of Experimental Psychology, 58, 376-381. "
    "Squire, L. R. (1994). Declarative and nondeclarative memory. Journal of "
    "Cognitive Neuroscience, 4, 232-243."
)
M_VERB_CORRECTIONS = (
    "Hautus, M. J. (1995). Corrections for extreme proportions and their biasing "
    "effects on estimated values of d-prime. Behavior Research Methods, 27, 46-51. "
    "Holmes, E. (2018). Mental imagery in emotion. Clinical Psychology Review, 60, 1-10."
)
M_VERB_COMMENT_TITLE = (
    "Fachin, S. (2004). Bootstrap inference on cointegrating coefficients: A comment. "
    "Economics Bulletin, 3, 1-8. Fachin, S. (2007). Long-run trends in the dynamics "
    "of expenditure. Empirical Economics, 33, 19-35."
)
MERGED_WITH_VERB_IN_TITLE = [M_VERB_RESPONSES, M_VERB_CORRECTIONS, M_VERB_COMMENT_TITLE]


# --- NUMBERED (Vancouver/Nature/IEEE) merges: year-at-END, "N." delimiter ---
# These defeat the author-date heuristic (no author lead before the 2nd date),
# but the sequential "N." markers are an unambiguous boundary. MUST split.
P_NUM_EYECOLOR = (
    "1. Zietsch, B. P., Verweij, K. J. H. & Martin, N. G. Variation in human mate "
    "choice. The American Naturalist 177, 605-616 (2011). "
    "2. Nojo, S., Tamura, S. & Ihara, Y. Human homogamy in facial characteristics. "
    "Human Nature 23, 323-334 (2012). "
    "3. Rantala, M. J. & Marcinkowska, U. M. The role of sexual imprinting. "
    "Behavioral Ecology and Sociobiology 65, 859-873 (2011)."
)
# geom-style merge starting mid-list (6,7,8); ref 7 author "ten Cate" is lowercase
P_NUM_MIDLIST = (
    "6. Bateson, P. Preferences for cousins in Japanese quail. Nature 295, 236-237 (1982). "
    "7. ten Cate, C. & Vos, D. R. Sexual Imprinting and Evolutionary Processes in Birds. "
    "Advances in the Study of Behavior 28, 1-31 (1999). "
    "8. Bereczkei, T., Gyuris, P. & Bernath, L. Homogamy, genetic similarity, and "
    "imprinting. Personality and Individual Differences 33, 677-690 (2002)."
)
NUMBERED_MERGED = [P_NUM_EYECOLOR, P_NUM_MIDLIST]

# --- single NUMBERED references that MUST NOT split (precision) ---
N_NUM_SINGLE = (
    "1. Burley, N. T. The meaning of assortative mating. Ethology and Sociobiology "
    "4, 191-203 (1983)."
)
N_NUM_VOL = (  # interior "Vol. 2." is a volume marker, not a reference onset
    "1. Burnham, K. P. & Anderson, D. R. Model selection and multimodel inference, "
    "Vol. 2. Springer (2002)."
)
N_NUM_EDITION = (  # "Edition 2." mid-reference, not preceded by a ref-ending boundary
    "1. American Psychological Association. Publication manual. Edition 2. Washington, DC (2020)."
)
NUMBERED_SINGLES = [N_NUM_SINGLE, N_NUM_VOL, N_NUM_EDITION]

NUMBERED_VANCOUVER_SINGLES = [
    (
        "1. Blair RJR. The neurobiology of psychopathic traits in youths. "
        "Nat Rev Neurosci. 2013. https://doi.org/10.1038/nrn3577."
    ),
    (
        "2. Dekkers TJ, Popma A, van Rentergem JAA, Bexkens A, Huizenga HM. "
        "Risky decision-making in attention-deficit/hyperactivity disorder. "
        "Clin Psychol Rev. 2016. https://doi.org/10.1016/j.cpr.2016.03.001."
    ),
    "in cocaine-dependent individuals. Addict Biol. 2015. https://doi.org/10.1111/adb.12143.",
    (
        "3. Rotge JY, Poitou C, Fossati P, Aron-Wisnewsky J, Oppert JM. "
        "Decision-making in obesity without eating disorders. "
        "Obes Rev. 2017. https://doi.org/10.1111/obr.12549."
    ),
]


# --- BARE-YEAR (arXiv/ACL) merges: "FirstName LastName, ... YEAR. Title." ---
# No numbered marker and no parenthesized date, so both other detectors are
# blind; the "<byline>. YEAR. <title>" shape is the only signal. These are the
# geom under-segmentation failures (BERT refs merged into one bib record). The
# strings are the real OCR reference-content blocks from arxiv_bert.json.
P_BAREYEAR_BERT = (
    "Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones. 2018. "
    "Character-level language modeling with deeper self-attention. arXiv preprint "
    "arXiv:1808.04444. Rie Kubota Ando and Tong Zhang. 2005. A framework for learning "
    "predictive structures from multiple tasks and unlabeled data. Journal of Machine "
    "Learning Research, 6(Nov):1817–1853. Luisa Bentivogli, Bernardo Magnini, Ido "
    "Dagan, Hoa Trang Dang, and Danilo Giampiccolo. 2009. The fifth PASCAL recognizing "
    "textual entailment challenge. In TAC. NIST."
)
# dense-comma "Lastname, Firstname and Lastname, Firstname." byline style
P_BAREYEAR_DENSE = (
    "Bengio, Yoshua and Glorot, Xavier. 2010. Understanding the difficulty of training "
    "deep feedforward neural networks. In Proceedings of AISTATS, pages 249-256. "
    "LeCun, Yann and Bengio, Yoshua. 1995. Convolutional networks for images, speech, "
    "and time series. The handbook of brain theory and neural networks, 3361."
)

# --- single BARE-YEAR references that MUST NOT split (precision) ---
# title itself contains a year ("the 2019 shared task") — not a byline onset
N_BAREYEAR_TITLE_YEAR = (
    "Rie Kubota Ando and Tong Zhang. 2005. A framework for the 2019 shared task. JMLR."
)
# single APA "(2019)" ref — parenthesized, not the bare shape
N_BAREYEAR_APA = "Smith, J. (2019). The title of the paper. Journal of Psychology, 12, 45-60."
# prose "In 2018. " after a real ref — "2018" opens no author byline
N_BAREYEAR_PROSE = (
    "Smith, John. 2005. A retrospective study. In 2018. The field changed dramatically."
)
# single ACL ref with initials — "K. Toutanova. 2019." is one byline, not a merge
N_BAREYEAR_INITIALS = (
    "J. Devlin, M. Chang, K. Lee, and K. Toutanova. 2019. BERT pretraining. NAACL."
)
BAREYEAR_SINGLES = [
    N_BAREYEAR_TITLE_YEAR,
    N_BAREYEAR_APA,
    N_BAREYEAR_PROSE,
    N_BAREYEAR_INITIALS,
]


def test_bareyear_merge_yields_two_interior_onsets():
    onsets = find_interior_onsets(P_BAREYEAR_BERT)
    assert len(onsets) >= 2, f"expected >=2 onsets, got {onsets}"
    assert P_BAREYEAR_BERT[onsets[0] :].startswith("Rie Kubota Ando")
    assert P_BAREYEAR_BERT[onsets[1] :].startswith("Luisa Bentivogli")


def test_bareyear_merge_splits_into_three():
    out, n_new = split_merged_refs([P_BAREYEAR_BERT])
    assert n_new == 2, f"expected +2, got +{n_new}: {out}"
    assert out[0].startswith("Rami Al-Rfou")
    assert out[1].startswith("Rie Kubota Ando")
    assert out[2].startswith("Luisa Bentivogli")


def test_bareyear_dense_comma_byline_splits():
    onsets = find_interior_onsets(P_BAREYEAR_DENSE)
    assert len(onsets) == 1, f"expected 1 onset, got {onsets}"
    assert P_BAREYEAR_DENSE[onsets[0] :].startswith("LeCun, Yann and Bengio, Yoshua")


def test_bareyear_singles_do_not_split():
    for s in BAREYEAR_SINGLES:
        assert find_interior_onsets(s) == [], f"false split on bare-year single: {s[:60]}..."


def test_numbered_merge_splits_on_sequential_markers():
    onsets = find_interior_onsets(P_NUM_EYECOLOR)
    assert len(onsets) == 2, f"expected 2 onsets, got {onsets}"
    assert P_NUM_EYECOLOR[onsets[0] :].startswith("2. Nojo")
    assert P_NUM_EYECOLOR[onsets[1] :].startswith("3. Rantala")


def test_numbered_midlist_merge_splits_into_three():
    out, n_new = split_merged_refs([P_NUM_MIDLIST])
    assert n_new == 2, f"expected +2, got +{n_new}: {out}"
    assert out[0].startswith("6. Bateson")
    assert out[1].startswith("7. ten Cate")
    assert out[2].startswith("8. Bereczkei")


def test_numbered_singles_do_not_split():
    for s in NUMBERED_SINGLES:
        assert find_interior_onsets(s) == [], f"false split on numbered single: {s[:60]}..."


# Edition numbers printed before the edition word ("2. Aufl.") that happen to
# equal the next entry number.
NUMBERED_EDITION_BIBLIOGRAPHY = [
    "1. Müller A. Lehrbuch der Inneren Medizin. 2. Aufl. Stuttgart: Thieme; 2019.",
    "2. Schmidt B. Kardiologie. Berlin: Springer; 2018.",
    "3. Weber C. Pharmakologie. 4. Auflage. München: Elsevier; 2017.",
    "4. Meyer D. Chirurgie. Köln: Deutscher Ärzteverlag; 2016.",
    "5. Hansen E. Klinisk farmakologi. 6. udg. København: Munksgaard; 2015.",
    "6. Silva F. Farmacologia clínica. 7. ed. São Paulo: Atlas; 2014.",
]


def test_numbered_edition_markers_do_not_split():
    out, n_new = split_merged_refs(NUMBERED_EDITION_BIBLIOGRAPHY)

    assert out == NUMBERED_EDITION_BIBLIOGRAPHY
    assert n_new == 0


def test_numbered_merge_with_next_author_named_like_an_edition_word_still_splits():
    merged = (
        "1. Müller A. Lehrbuch. Stuttgart: Thieme; 2019. 2. Edwards J. Cardiology. Lancet; 2018."
    )

    out, n_new = split_merged_refs([merged])

    assert n_new == 1
    assert out == [
        "1. Müller A. Lehrbuch. Stuttgart: Thieme; 2019.",
        "2. Edwards J. Cardiology. Lancet; 2018.",
    ]


def test_numbered_bibliography_does_not_split_journal_year_tails():
    candidates = detect_merges(NUMBERED_VANCOUVER_SINGLES)
    out, n_new = split_merged_refs(NUMBERED_VANCOUVER_SINGLES)

    assert candidates == []
    assert out == NUMBERED_VANCOUVER_SINGLES
    assert n_new == 0


def test_numbered_bibliography_still_splits_sequential_numbered_merge():
    refs = [P_NUM_MIDLIST, N_NUM_SINGLE]

    candidates = detect_merges(refs)
    out, n_new = split_merged_refs(refs)

    assert [candidate.index for candidate in candidates] == [0]
    assert n_new == 2
    assert out[0].startswith("6. Bateson")
    assert out[1].startswith("7. ten Cate")
    assert out[2].startswith("8. Bereczkei")
    assert out[3] == N_NUM_SINGLE


def test_merge_split_keeps_surname_with_multi_initial_second_author():
    merged = (
        "Jennison, C., & Turnbull, B. W. (2000). Group sequential methods. "
        "Jones, L. V. (1952). Test of hypotheses."
    )

    split, added = split_merged_refs([merged])

    assert added == 1
    assert split == [
        "Jennison, C., & Turnbull, B. W. (2000). Group sequential methods.",
        "Jones, L. V. (1952). Test of hypotheses.",
    ]


def test_in_title_citation_single_does_not_split():
    # a single ref whose title cites "(Author Year)" must NOT be split
    for s in INTITLE_SINGLES:
        assert find_interior_onsets(s) == [], f"false split on in-title cite: {s[:70]}..."


# A Title Case title that opens right after the reference's own date and cites
# another work. Nothing separates the two dates, so there is no reference 1
# title between them and the second date cannot start a new reference.
N_INTITLE_TITLE_CASE = [
    (
        "Brown, T. (2018). Beyond Kahneman and Tversky (1979): Prospect Theory Today. "
        "Econ Review, 5, 1-20."
    ),
    "Jones, K. (2021). Why Smith (2019) was wrong. Journal of Things, 3, 4-5.",
    # eLife HTML reference list item (the authors carry no punctuation)
    (
        "Thomas DR Zumbo BD Kwan E Schweitzer L (2014) On Johnson's (2000) relative "
        "weights method for assessing variable importance: A reanalysis Multivariate "
        "Behavioral Research 49:329–338."
    ),
]


def test_title_case_in_title_citation_does_not_split():
    for s in N_INTITLE_TITLE_CASE:
        assert find_interior_onsets(s) == [], f"false split on in-title cite: {s[:70]}..."
    out, n_new = split_merged_refs(N_INTITLE_TITLE_CASE)
    assert out == N_INTITLE_TITLE_CASE
    assert n_new == 0


def test_merge_whose_second_title_cites_a_work_splits_once():
    merged = (
        "Adams, A. (2017). Loss aversion revisited. Econ Letters, 4, 1-9. "
        "Brown, T. (2018). Beyond Kahneman and Tversky (1979): Prospect Theory Today. "
        "Econ Review, 5, 1-20."
    )

    out, n_new = split_merged_refs([merged])

    assert n_new == 1
    assert out == [
        "Adams, A. (2017). Loss aversion revisited. Econ Letters, 4, 1-9.",
        "Brown, T. (2018). Beyond Kahneman and Tversky (1979): Prospect Theory Today. "
        "Econ Review, 5, 1-20.",
    ]


def test_real_merge_with_citation_verb_in_title_still_splits():
    # ref1's title legitimately contains reply/comment/correction -> must STILL split
    for s in MERGED_WITH_VERB_IN_TITLE:
        onsets = find_interior_onsets(s)
        assert len(onsets) == 1, f"lost a real split: {onsets} for {s[:70]}..."


def test_merged_segments_yield_one_onset():
    for s in MERGED:
        onsets = find_interior_onsets(s)
        assert len(onsets) == 1, f"expected 1 onset, got {onsets} for: {s[:60]}..."
        assert 0 < onsets[0] < len(s)


def test_onset_lands_at_the_second_reference_start():
    # the split point must begin the SECOND reference, prefix intact
    o = find_interior_onsets(P_CORP2)[0]
    assert P_CORP2[o:].startswith("U.S. Social Security Administration")
    o2 = find_interior_onsets(P_URL_NDOT)[0]
    assert P_URL_NDOT[o2:].startswith("United Nations")
    o3 = find_interior_onsets(P_PERSONAL)[0]
    assert P_PERSONAL[o3:].startswith("Kühberger")


def test_single_references_yield_no_onset():
    for s in SINGLES:
        assert find_interior_onsets(s) == [], f"false split on: {s[:60]}..."


def test_detect_merges_flags_only_merged():
    cands = detect_merges([P_PERSONAL, N_CLEAN_SINGLE, P_CORP2])
    assert [c.index for c in cands] == [0, 2]
    assert all(c.offsets for c in cands)
    assert all(len(c.contexts) == len(c.offsets) for c in cands)


def test_split_separates_a_merged_pair():
    out, n_new = split_merged_refs([P_CORP2])
    assert n_new == 1
    assert len(out) == 2
    assert out[0].startswith("U.S. Office of Management and Budget")
    assert out[1].startswith("U.S. Social Security Administration")


def test_split_leaves_clean_singles_untouched():
    out, n_new = split_merged_refs(SINGLES)
    assert n_new == 0
    assert out == SINGLES


def test_split_is_one_sided_safe():
    mixed = [P_PERSONAL, N_CLEAN_SINGLE, P_CORP2, N_REPRINT]
    out, n_new = split_merged_refs(mixed)
    assert len(out) >= len(mixed)
    assert n_new == len(out) - len(mixed)


def test_split_is_best_effort_on_error(monkeypatch):
    def boom(_s):
        raise ValueError("synthetic")

    monkeypatch.setattr(merge_split, "find_interior_onsets", boom)
    out, n_new = split_merged_refs([P_CORP2, N_CLEAN_SINGLE])
    assert out == [P_CORP2, N_CLEAN_SINGLE]
    assert n_new == 0


from unittest.mock import Mock

from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.paper_contents import PaperContents
from bibr.processing_warnings import WarningCode


def _bare_extractor():
    return ReferenceExtractor(Mock(spec=PaperContents), llm_client=Mock())


def test_maybe_split_noop_when_disabled():
    ext = _bare_extractor()
    ext._settings.REF_SPLIT_MERGED_REFS = False
    refs = [P_CORP2, N_CLEAN_SINGLE]
    assert ext._maybe_split_merged(refs) == refs


def test_maybe_split_splits_and_warns_when_enabled():
    ext = _bare_extractor()
    ext._settings.REF_SPLIT_MERGED_REFS = True
    out = ext._maybe_split_merged([P_CORP2, N_CLEAN_SINGLE])
    assert len(out) == 3
    warnings = ext.contents.processing_warnings
    assert any(w.code == WarningCode.REF_SEG_MERGE_SPLIT for w in warnings)
