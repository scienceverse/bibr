"""Line-end hyphen adjudication must work outside ASCII/English.

`_stx_alpha_keep_hyphen` used ASCII-only fragment regexes and a hardcoded
English wordfreq lookup, so for Cyrillic/Greek/Turkish/accented-Latin text the
adjudicator could never fire and the printed line-end hyphen was always
deleted.  Deletion is the destructive branch: bibr must reproduce what is on
the page unless it can prove the mark was hyphenation.
"""

from bibr.input.consolidate_text import (
    _dehyphenate_word_linewraps,
    _resolve_stx_marks,
    fix_ocr_artifacts,
)


class TestStxNonAsciiKeepsPrintedHyphen:
    """STX-marked wrap hyphens in non-English scripts must survive."""

    def test_cyrillic_compound_keeps_hyphen(self):
        # 10.17513/spno.32661: page 1 prints "ГЛИЦИЛ-" at line end; text_id 4/5
        # read "глицил-цистеинил-глутамата" mid-line, so the hyphen is printed.
        assert (
            _resolve_stx_marks("ИНОЗИНА ГЛИЦИЛ\x02ЦИСТЕИНИЛ-ГЛУТАМАТА")
            == "ИНОЗИНА ГЛИЦИЛ-ЦИСТЕИНИЛ-ГЛУТАМАТА"
        )

    def test_accented_latin_compound_keeps_hyphen(self):
        assert _resolve_stx_marks("santé\x02publique") == "santé-publique"

    def test_greek_compound_keeps_hyphen(self):
        assert _resolve_stx_marks("ανθρώπινο\x02δικαίωμα") == "ανθρώπινο-δικαίωμα"

    def test_turkish_compound_keeps_hyphen(self):
        assert _resolve_stx_marks("sağlık\x02hizmeti") == "sağlık-hizmeti"

    def test_unsupported_script_keeps_hyphen(self):
        # wordfreq ships no Georgian lexicon, so nothing is ever proven about
        # these fragments and the printed hyphen stays.
        assert _resolve_stx_marks("ქართული\x02ენა") == "ქართული-ენა"


class TestAsciiFragmentsKeepTheJoinDefault:
    """The keep-default is scoped to fragments the lexicons cannot cover.

    For plain ASCII, English's silence is informative: extending the flip
    there re-hyphenates a large share of genuine Italian/Dutch/German/French/
    Indonesian wraps whose fragments carry no diacritic.
    """

    def test_ascii_unknown_pair_still_joins(self):
        assert _resolve_stx_marks("qwrtz\x02plkjh") == "qwrtzplkjh"

    def test_italian_ascii_wrap_still_joins(self):
        assert _resolve_stx_marks("perché la ricostru\x02zione è") == "perché la ricostruzione è"


class TestStxNonAsciiStillJoinsRealHyphenation:
    """A language hint must let genuine hyphenation re-join outside English."""

    def test_russian_hyphenation_joins(self):
        assert _resolve_stx_marks("полная информа\x02ция здесь") == "полная информация здесь"

    def test_region_language_rescues_ascii_fragments(self):
        # The wrap falls between two undiacriticked syllables, so only the
        # surrounding region names the language.  Without it, English knows
        # none of the three forms and the printed hyphen would be kept.
        assert (
            _resolve_stx_marks("wypowie\x02dzenie umowy o pracę") == "wypowiedzenie umowy o pracę"
        )

    def test_region_language_does_not_override_fragment_script(self):
        # A Cyrillic fragment inside a region that also carries Polish letters
        # is still adjudicated as Cyrillic, and stays hyphenated.
        assert (
            _resolve_stx_marks("pracę ИНОЗИНА ГЛИЦИЛ\x02ЦИСТЕИНИЛ")
            == "pracę ИНОЗИНА ГЛИЦИЛ-ЦИСТЕИНИЛ"
        )


class TestStxAsciiControlsUnchanged:
    """The long-standing English adjudication must be bit-identical."""

    def test_ascii_valid_word_pair_keeps_hyphen(self):
        assert _resolve_stx_marks("risk self\x02management") == "risk self-management"

    def test_ascii_joined_word_still_joins(self):
        assert fix_ocr_artifacts("off\x02line processing") == "offline processing"

    def test_ascii_rare_joined_form_still_joins(self):
        # zipf("psychophysiological") == 1.65: below the word threshold but
        # still a lexicon opinion, so this is hyphenation, not a compound.
        assert (
            fix_ocr_artifacts("psychophysio\x02logical measures") == "psychophysiological measures"
        )

    def test_ascii_surname_wrap_still_joins(self):
        assert fix_ocr_artifacts("Kahne\x02man (2011)") == "Kahneman (2011)"


class TestDehyphenateWordLinewrapsNonAscii:
    """The literal-hyphen line-wrap path drops the same hyphens."""

    def test_cyrillic_linewrap_keeps_hyphen(self):
        assert _dehyphenate_word_linewraps("ГЛИЦИЛ-\nЦИСТЕИНИЛ") == "ГЛИЦИЛ-ЦИСТЕИНИЛ"

    def test_greek_linewrap_keeps_hyphen(self):
        assert _dehyphenate_word_linewraps("ανθρώπινο-\nδικαίωμα") == "ανθρώπινο-δικαίωμα"

    def test_russian_linewrap_hyphenation_joins(self):
        assert _dehyphenate_word_linewraps("информа-\nция") == "информация"

    def test_ascii_linewrap_still_joins(self):
        assert _dehyphenate_word_linewraps("analy-\nsis code") == "analysis code"

    def test_ascii_linewrap_compound_still_keeps_hyphen(self):
        assert _dehyphenate_word_linewraps("subjective well-\nbeing") == "subjective well-being"
