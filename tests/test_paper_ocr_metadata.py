"""L5: OCR-supplied author names offered as doc-info author candidates.

"Family, Given" was split on the last space, which inverted the name.
"""


class TestOcrMetadataNameSplitting:
    """L5: "Family, Given" was split on the last space, inverting the name."""

    def _merged(self, name):
        from bibr.paper import doc_info_candidates

        return doc_info_candidates({"authors": [name]})["author"].value[0]

    def test_comma_form_is_not_inverted(self):
        author = self._merged("Smith, John")
        assert (author.given, author.family) == ("John", "Smith")

    def test_plain_form_still_splits_on_the_last_space(self):
        author = self._merged("John Smith")
        assert (author.given, author.family) == ("John", "Smith")

    def test_a_multi_part_given_name_survives(self):
        author = self._merged("Smith, John Quincy")
        assert (author.given, author.family) == ("John Quincy", "Smith")
