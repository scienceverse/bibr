"""L5: OCR-supplied author names merged into ``PaperMetadata``.

"Family, Given" was split on the last space, which inverted the name.
"""

from bibr.models import PaperMetadata


class TestOcrMetadataNameSplitting:
    """L5: "Family, Given" was split on the last space, inverting the name."""

    def _merged(self, name):
        from bibr.paper import _merge_ocr_metadata

        metadata = PaperMetadata(title="T", doi="")
        _merge_ocr_metadata(metadata, {"authors": [name]})
        return metadata.authors[0]

    def test_comma_form_is_not_inverted(self):
        author = self._merged("Smith, John")
        assert (author.given, author.family) == ("John", "Smith")

    def test_plain_form_still_splits_on_the_last_space(self):
        author = self._merged("John Smith")
        assert (author.given, author.family) == ("John", "Smith")

    def test_a_multi_part_given_name_survives(self):
        author = self._merged("Smith, John Quincy")
        assert (author.given, author.family) == ("John Quincy", "Smith")
