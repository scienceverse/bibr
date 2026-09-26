"""A reference list and page text shared by the tests of reference-region
coverage: ``alnum_text_covered``, the OCR stage's reference dedup and the PDF
parser's aggregate-box shadowing."""

from __future__ import annotations

# Twelve entries. The shortest is under 5% of the list's text, so a fuzzy score
# over the whole list still passes 95 when that entry is missing.
REFERENCE_LIST = [
    "Ajzen, I. (1991). The theory of planned behavior. Organizational Behavior and "
    "Human Decision Processes, 50(2), 179-211.",
    "Baumeister, R. F., & Leary, M. R. (1995). The need to belong: Desire for "
    "interpersonal attachments as a fundamental human motivation. Psychological "
    "Bulletin, 117(3), 497-529.",
    "Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159.",
    "Deci, E. L., & Ryan, R. M. (2000). The what and why of goal pursuits. "
    "Psychological Inquiry, 11(4), 227-268.",
    "Festinger, L. (1954). A theory of social comparison processes. Human Relations, "
    "7(2), 117-140.",
    "Gross, J. J. (1998). The emerging field of emotion regulation: An integrative "
    "review. Review of General Psychology, 2(3), 271-299.",
    "Hayes, A. F. (2013). Introduction to mediation, moderation, and conditional "
    "process analysis. Guilford Press.",
    "Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision "
    "under risk. Econometrica, 47(2), 263-291.",
    "Markus, H. R., & Kitayama, S. (1991). Culture and the self: Implications for "
    "cognition, emotion, and motivation. Psychological Review, 98(2), 224-253.",
    "Open Science Collaboration. (2015). Estimating the reproducibility of "
    "psychological science. Science, 349(6251), aac4716.",
    "Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive "
    "psychology. Psychological Science, 22(11), 1359-1366.",
    "Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup "
    "conflict. In W. G. Austin & S. Worchel (Eds.), The social psychology of "
    "intergroup relations (pp. 33-47). Brooks/Cole.",
]

# Body text printed on the same page as the list.
BODY_TEXT = (
    "Participants in both studies rated how strongly they identified with the "
    "group and how often they compared their own outcomes with those of others, "
    "and every rating was reviewed over the five waves. "
)


def read_again(text: str) -> str:
    """A second OCR read of *text*: "rn" for its first "m", "l" for its first "1"."""
    return text.replace("m", "rn", 1).replace("1", "l", 1)
