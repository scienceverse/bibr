"""Equation xref precision regressions.

Cover LaTeX command substrings, display formulas, URL substrings, decimal equation numbers, and enclosing punctuation."""

from bibr.paper_contents import PaperSentence
from bibr.structure.xref_utils import detect_xrefs


def _sent(text: str, *, formula: bool = False) -> list[PaperSentence]:
    return [
        PaperSentence(
            text_id=1,
            text=text,
            section_id=1,
            paragraph_id=1,
            is_display_formula=formula,
        )
    ]


def _eq_xrefs(sentences):
    return [x for x in detect_xrefs(sentences, [], []) if x.xref_type == "equation"]


class TestEquationXrefFalsePositives:
    def test_latex_leq_not_matched(self):
        # "\leq 1" contains "eq 1" — must not become an equation xref
        xrefs = _eq_xrefs(_sent(r"The share satisfies 0 \leq \alpha \leq 1 in all periods."))
        assert xrefs == []

    def test_latex_geq_glued_digit_not_matched(self):
        xrefs = _eq_xrefs(_sent(r"We selected pictures with large apertures ( f\leq4 )."))
        assert xrefs == []

    def test_url_substring_not_matched(self):
        # "osf.io/geq9x" produced contents "eq9"
        xrefs = _eq_xrefs(_sent("Code can be accessed at https://osf.io/geq9x/."))
        assert xrefs == []

    def test_display_formula_sentence_skipped(self):
        # Display formulas export as "[equation]"; nothing inside the raw
        # math (even a literal "Equation 1" annotation) is a cross-reference.
        xrefs = _eq_xrefs(
            _sent(r"P_t = P_{D,t}^{\alpha}, \text{Equation 1}, \tag{2}", formula=True)
        )
        assert xrefs == []

    def test_display_formula_skipped_for_all_xref_types(self):
        xrefs = detect_xrefs(_sent(r"0 \leq \gamma \leq 1, \tag{7}", formula=True), [], [])
        assert xrefs == []


class TestEquationXrefContentsHygiene:
    def test_decimal_equation_number_not_truncated(self):
        # Was: contents "Eq. (2" — truncated mid-number
        xrefs = _eq_xrefs(_sent("The coefficient matrix is zero (See Eq. (2.3))."))
        assert len(xrefs) == 1
        assert xrefs[0].contents == "Eq. (2.3)"
        assert xrefs[0].xref_id == 2  # major number, mirroring section xrefs

    def test_enclosing_paren_not_dragged_into_contents(self):
        # Was: contents "Equations 1, 3, 5 and 7)" — trailing ")" belongs to
        # the enclosing "(see ...)" parenthetical, not to the reference.
        xrefs = _eq_xrefs(_sent("Estimates are shown (see Equations 1, 3, 5 and 7)."))
        assert {x.xref_id for x in xrefs} == {1, 3, 5, 7}
        assert all(x.contents == "Equations 1, 3, 5 and 7" for x in xrefs)


class TestEquationXrefRecallPreserved:
    def test_equation_paren_form(self):
        xrefs = _eq_xrefs(_sent("As shown in Equation (3), growth slows."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 3
        assert xrefs[0].contents == "Equation (3)"

    def test_eq_abbrev_form(self):
        xrefs = _eq_xrefs(_sent("Substituting into Eq. 5 yields the result."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 5

    def test_eq_range_form(self):
        xrefs = _eq_xrefs(_sent("Eqs. 1-3 define the system."))
        assert {x.xref_id for x in xrefs} == {1, 2, 3}

    def test_eq_glued_paren_form(self):
        # "eq.(1)" (no space) is common in de Gruyter econ papers
        xrefs = _eq_xrefs(_sent("It is clear from eq.(1) that profits fall."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1
        assert xrefs[0].contents == "eq.(1)"

    def test_equation_list_form(self):
        xrefs = _eq_xrefs(_sent("Equations 1, 2, 5 and 6 give the equilibrium."))
        assert {x.xref_id for x in xrefs} == {1, 2, 5, 6}

    def test_lowercase_equation_form(self):
        xrefs = _eq_xrefs(_sent("Substituting equation (2) into the constraint."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 2

    def test_ocr_glued_footnote_marker_still_matches(self):
        # OCR-flattened footnote marker glued to the word: "9Equations (1)"
        # (observed in de Gruyter econ outputs) — still a real reference.
        xrefs = _eq_xrefs(_sent("9Equations (1) and (2) are still exact."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1


class TestEquationXrefUnitAndSoftwareNames:
    """structure-citations-floats-15: unit spellings and versioned software
    names must not become equation xrefs."""

    def test_co2_equivalent_unit_not_matched(self):
        # "CO2-eq. (39.1%)" produced equation 39 with contents "eq. (39.1"
        xrefs = _eq_xrefs(_sent("The flow of CO2-eq. (39.1%) in the region grows."))
        assert xrefs == []

    def test_co2_equivalent_endash_variant_not_matched(self):
        xrefs = _eq_xrefs(_sent("The flow of CO2–eq. (39.1%) in the region grows."))
        assert xrefs == []

    def test_percentage_after_number_not_matched(self):
        # Even spaced ("CO2 eq. (39.1%)"), the "%" marks a share, not an id.
        xrefs = _eq_xrefs(_sent("The flow of CO2 eq. (39.1%) in the region grows."))
        assert xrefs == []

    def test_eqs_software_version_not_matched(self):
        # "EQS 6.1 (Bentler, 2005" produced equation 6 with contents "EQS 6.1"
        xrefs = _eq_xrefs(_sent("Model fit used the software EQS 6.1 (Bentler, 2005)."))
        assert xrefs == []

    def test_eqs_integer_form_is_software_not_equation(self):
        # The finding's own paper also cites "EQS 6 structural equations
        # program manual": the integer form is the same SEM software, not an
        # equation, whatever the number shape.
        xrefs = _eq_xrefs(
            _sent(
                "Bentler, P. M. (2005). EQS 6 structural equations program manual. "
                "Encino, CA: Multivariate Software."
            )
        )
        assert xrefs == []
        xrefs = _eq_xrefs(_sent("Models were estimated in EQS 6 (Bentler, 2006)."))
        assert xrefs == []

    def test_hyphen_before_lowercase_eq_unit_not_matched(self):
        # The hyphen rule fires without a trailing "%": "CO2-eq. 3" is a
        # unit with a count, not equation 3.
        xrefs = _eq_xrefs(_sent("Emissions of 5 t CO2-eq. 3 times higher."))
        assert xrefs == []


class TestEquationXrefBareFormsPreserved:
    """Guards: real bare short forms observed in the gate192 exports and the
    JATS corpora must keep matching after the unit/software tightening."""

    def test_bare_eq_integer_form(self):
        # gate192: "eq 5"
        xrefs = _eq_xrefs(_sent("As shown in eq 5, growth slows."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 5

    def test_bare_eqs_list_form(self):
        # gate192: "eqs 4 and 8"
        xrefs = _eq_xrefs(_sent("See eqs 4 and 8 for the system."))
        assert {x.xref_id for x in xrefs} == {4, 8}

    def test_bare_eq_paren_form(self):
        # JATS corpora: "Eq (1)"
        xrefs = _eq_xrefs(_sent("As shown in Eq (1), growth slows."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1

    def test_caps_eqs_with_period_still_matches(self):
        # The software guard only fires without the period: "EQS. 6" is a
        # printed reference shape, not a version string.
        xrefs = _eq_xrefs(_sent("See EQS. 6 for the system."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 6

    def test_equation_followed_by_percent_phrase_still_matches(self):
        # The "%" guard only fires when the sign directly trails the number:
        # "Eq. 5" here is followed by words, not a share.
        xrefs = _eq_xrefs(_sent("Eq. 5 explains most of the variance in the sample."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 5

    def test_longhand_endash_range_matches_both_halves(self):
        # The hyphen rejection must not eat longhand ranges: the "–" before
        # the second "Equation" is a range dash, not a unit hyphen.
        xrefs = _eq_xrefs(_sent("We simplify the full model (Equation 5–Equation 7) using a mean."))
        assert {x.xref_id for x in xrefs} == {5, 7}

    def test_longhand_hyphen_range_matches_both_halves(self):
        # OCR and text layers often flatten the range dash to a hyphen
        # ("Equation 5-Equation 7"): the second half is a reference, not a
        # unit spelling — only a hyphen before lowercase "eq" is refused.
        xrefs = _eq_xrefs(_sent("As in Equation 5-Equation 7 above."))
        assert {x.xref_id for x in xrefs} == {5, 7}

    def test_caps_singular_eq_integer_still_matches(self):
        # The software guard drops the all-caps plural ("EQS 6") whatever
        # the number shape; the singular caps form is still a reference.
        xrefs = _eq_xrefs(_sent("See EQ 5 for the system."))
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 5
