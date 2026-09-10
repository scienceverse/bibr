"""Verify ``invalidate_text_caches`` drops every derived DataFrame cache."""

from bibr.paper_contents import PaperContents, PaperURLLink


def _empty_contents() -> PaperContents:
    return PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
    )


def test_invalidate_text_caches_drops_links_df():
    pc = _empty_contents()
    pc.links = [
        PaperURLLink(
            url="http://x",
            section_id=1,
            paragraph_id=1,
            text_id=1,
            link_text="x",
        )
    ]
    _ = pc.links_df  # populate cache
    assert "links_df" in pc.__dict__

    pc.invalidate_text_caches()
    assert "links_df" not in pc.__dict__


def test_invalidate_text_caches_drops_equations_df():
    pc = _empty_contents()
    pc.equations = []
    _ = pc.equations_df
    assert "equations_df" in pc.__dict__

    pc.invalidate_text_caches()
    assert "equations_df" not in pc.__dict__


def test_invalidate_text_caches_still_drops_sentences_and_text_df():
    """Regression guard: existing caches must continue to be invalidated."""
    pc = _empty_contents()
    _ = pc.sentences_df
    _ = pc.text_df
    assert "sentences_df" in pc.__dict__
    assert "text_df" in pc.__dict__

    pc.invalidate_text_caches()
    assert "sentences_df" not in pc.__dict__
    assert "text_df" not in pc.__dict__
