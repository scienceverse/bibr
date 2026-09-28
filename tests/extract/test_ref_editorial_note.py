"""An editorial note printed as its own reference-list entry is dropped.

The notes are from scanned bibliographies: one annotates the entry above it,
the other is a reviewers' note under the list heading. The NER tagger read
the first as a title.
"""

import pytest

from bibr.config import GlobalSettings
from bibr.extract import ref_extractor as ex
from bibr.extract.ref_field_repair import repair_ner_reference_fields

NOTE = (
    "(This is a series of short articles by Adler and various pupils which show in more "
    "detail the general theory.)"
)
ENTRY = "—— — and FURTMÜLLER, CARL. eds. Heilen und Bilden. Munich, Reinhardt, 1914. 398p."


REVIEWERS_NOTE = (
    "(Note.-The titles of the books and articles without original English translations by "
    "the authors have been tentatively translated by the present reviewers.)"
)


@pytest.mark.parametrize("text", [NOTE, REVIEWERS_NOTE], ids=["annotation", "reviewers-note"])
def test_parenthesised_note_loses_the_title_the_tagger_gave_it(text):
    fields = {"title": text}
    fired = repair_ner_reference_fields(fields, text)
    assert "editorial_note" in fired
    assert not fields["title"]


@pytest.mark.parametrize(
    ("text", "fields"),
    [
        (ENTRY, {"authors": "—— — and FURTMÜLLER, CARL. eds", "title": "Heilen und Bilden"}),
        # A fragment that opens on a parenthesis but goes on after it.
        (
            "(Washington, D.C.), 8 (4), 494–521. https://doi.org/10.1037/1528-3542.8.4.494",
            {"title": "(Washington, D.C.)"},
        ),
        # A note the tagger found nothing in: nothing to clear.
        (REVIEWERS_NOTE, {}),
    ],
    ids=["entry", "paren-fragment", "untagged-note"],
)
def test_other_text_is_not_cleared_as_a_note(text, fields):
    before = dict(fields)
    fired = repair_ner_reference_fields(fields, text)
    assert "editorial_note" not in fired
    assert fields.get("title") == before.get("title")
    assert fields.get("authors") == before.get("authors")


def test_ner_parse_drops_the_note_and_keeps_the_entries(monkeypatch):
    outputs = {
        ENTRY: {"authors": "—— — and FURTMÜLLER, CARL. eds", "title": "Heilen und Bilden"},
        NOTE: {"title": NOTE},
    }

    class _Parser:
        def parse_batch(self, texts, batch_size=32):
            return [dict(outputs[t]) for t in texts]

    monkeypatch.setattr(ex, "_get_ner_parser", lambda settings, memory_mode=None: _Parser())
    extractor = ex.ReferenceExtractor(contents=None, file_hash="x", settings=GlobalSettings())
    refs = extractor._parse_references_ner_aligned([ENTRY, NOTE])
    assert refs[0] is not None
    assert refs[0].title == "Heilen und Bilden"
    assert refs[1] is None
