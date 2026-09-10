"""LLM JSON → Pydantic LLM model → core model → export dict roundtrip.

Pins the existing behavior so the inheritance refactor cannot
silently change extraction output.
"""

from __future__ import annotations


def test_authorllm_to_paperauthor_roundtrip():
    """LLM emits an author dict; AuthorLLM validates it; downstream code
    constructs a PaperAuthor with author_id assigned positionally."""
    from bibr.models import PaperAuthor
    from bibr.schemas import AuthorLLM

    llm_payload = {
        "given": "Jane",
        "family": "Doe",
        "affiliation": "MIT",
        "email": "jane@mit.edu",
        "corresponding": True,
        "orcid": "0000-0001-2345-6789",
    }
    llm = AuthorLLM.model_validate(llm_payload)
    pa = PaperAuthor(
        author_id=1,
        given=llm.given or "",
        family=llm.family or "",
        affiliation=llm.affiliation or "",
        email=llm.email,
        corresponding=llm.corresponding,
        orcid=llm.orcid,
    )
    assert pa.author_id == 1
    assert pa.given == "Jane"
    assert pa.email == "jane@mit.edu"


def test_authorllm_null_string_coerced():
    """AuthorLLM coerces 'None'/'null' string to actual None on email/orcid."""
    from bibr.schemas import AuthorLLM

    a = AuthorLLM.model_validate(
        {
            "given": "x",
            "family": "y",
            "email": "None",
            "orcid": "null",
        }
    )
    assert a.email is None
    assert a.orcid is None


def test_referencellm_strips_citation_punctuation():
    from bibr.schemas import PaperReferenceLLM

    r = PaperReferenceLLM.model_validate(
        {
            "index": 1,
            "title": "Some title.",
            "first_page": "100",
            "volume": "10",
            "year": 2020,
            "container": "Nature,",
        }
    )
    assert r.title == "Some title"
    assert r.container == "Nature"


def test_referencellm_drops_colon_in_short_numeric_fields():
    """Defensive: LLM serialization leak detection."""
    from bibr.schemas import PaperReferenceLLM

    r = PaperReferenceLLM.model_validate(
        {
            "index": 1,
            "title": "x",
            "first_page": "None,index:6,is_in_press:false",  # leaked
            "volume": "10",
            "year": 2020,
            "container": "Nature",
        }
    )
    assert r.first_page is None  # rejected
    assert r.volume == "10"  # legitimate
