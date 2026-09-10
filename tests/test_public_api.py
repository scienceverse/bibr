def test_export_models_are_public():
    from bibr.export import (
        AuthorExport,
        BibExport,
        InfoExport,
        PaperExport,
        TextExport,
    )

    assert PaperExport.__name__ == "PaperExport"
    assert InfoExport.__name__ == "InfoExport"
    assert AuthorExport.__name__ == "AuthorExport"
    assert TextExport.__name__ == "TextExport"
    assert BibExport.__name__ == "BibExport"


def test_typed_chew_entry_points_are_public():
    import bibr

    for name in ("chew_file", "achew_file", "chew_many", "achew_many", "PaperExport"):
        assert getattr(bibr, name) is not None
