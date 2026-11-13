"""Merged-reference splitting must not cut a numbered entry at its edition statement."""

import pytest

from bibr.extract.merge_split import split_merged_refs


# The edition number equals the next entry number, and a qualifier sits
# between it and the edition word, or the edition word is spelled out.
@pytest.mark.parametrize(
    "entry",
    [
        "1. Müller A. Lehrbuch der Chirurgie. 2. überarb. Aufl. Stuttgart: Thieme; 2019.",
        "1. Müller A. Lehrbuch der Chirurgie. 2., erw. Aufl. Stuttgart: Thieme; 2019.",
        "1. Müller A. Lehrbuch der Chirurgie. 2. erw. und aktualisierte Auflage. Thieme; 2019.",
        "1. Müller A. Lehrbuch. 2. vollständig überarbeitete Auflage. Stuttgart: Thieme; 2019.",
        "1. Smith J. Clinical medicine. 2. edition. London: Elsevier; 2010.",
        "1. Smith J. Clinical medicine. 2. rev. ed. London: Elsevier; 2010.",
        "1. Dupont J. Manuel de chirurgie. 2. éd. Paris: Masson; 2001.",
        "4. Kowalski J. Podręcznik chorób. 5. wydanie. Warszawa: PZWL; 2010.",
        "2. Novák J. Učebnice interny. 3. přepracované vydání. Praha: Grada; 2012.",
        "3. Nagy P. Belgyógyászat tankönyve. 4. átdolg. kiad. Budapest: Medicina; 2001.",
        "7. Svensson A. Lärobok i medicin. 8. upplagan. Stockholm: Liber; 2015.",
        "1. García J. Manual de medicina. 2. edición. Madrid: Elsevier; 2010.",
    ],
)
def test_edition_statement_does_not_split_the_entry(entry):
    out, n_new = split_merged_refs([entry])
    assert out == [entry]
    assert n_new == 0


@pytest.mark.parametrize(
    "merged",
    [
        "1. Müller A. Lehrbuch. Stuttgart: Thieme; 2019. 2. Neumann B. Kardiologie. Berlin; 2018.",
        "1. Smith J. Medicine. London; 2010. 2. Revell K. Surgery. Paris: Masson; 2011.",
        "1. Smith J. Medicine. London; 2010. 2. Edwards J. Cardiology. Lancet; 2018.",
    ],
)
def test_next_entry_named_like_a_qualifier_still_splits(merged):
    out, n_new = split_merged_refs([merged])
    assert n_new == 1
    assert out[1].startswith("2. ")


def test_next_author_with_edition_like_initials_still_splits():
    merged = "1. Smith J. Medicine. London; 2010. 2. Verbeek ED, Korr AB. Surgery. Paris; 2011."
    out, n_new = split_merged_refs([merged])
    assert n_new == 1
    assert out[1].startswith("2. Verbeek ED")
