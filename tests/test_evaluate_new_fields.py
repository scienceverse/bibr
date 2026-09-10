"""tests/test_evaluate_new_fields.py"""

from evaluation.evaluate import (
    affiliation_sim,
    corresponding_acc,
    email_f1,
    extract_comparable_from_json,
    keywords_f1,
    orcid_f1,
    score_paper,
)


def test_comparable_carries_keywords_and_full_authors():
    data = {
        "info": {"title": "T", "doi": "10.1/x", "keywords": ["alpha", "beta"]},
        "author": [
            {
                "author_id": 1,
                "given": "Ada",
                "family": "Byron",
                "affiliation": "Analytical Engine Institute",
                "email": "ada@aei.example",
                "corresponding": True,
                "orcid": "0000-0001-2345-6789",
                "role": [],
            }
        ],
        "text": [],
        "section": [],
        "bib": [],
    }
    c = extract_comparable_from_json(data)
    assert c["keywords"] == ["alpha", "beta"]
    a = c["authors"][0]
    assert a["affiliation"] == "Analytical Engine Institute"
    assert a["email"] == "ada@aei.example"
    assert a["orcid"] == "0000-0001-2345-6789"
    assert a["corresponding"] is True


def test_keywords_f1():
    assert keywords_f1(["Alpha", "beta"], ["alpha", "beta"]) == 1.0
    assert keywords_f1(["alpha"], ["alpha", "beta"]) == 2 / 3  # P=1, R=0.5
    assert keywords_f1(["alpha"], []) is None


def test_email_orcid_corresponding():
    e = [
        {
            "family": "Byron",
            "given": "Ada",
            "email": "A@X.com",
            "orcid": "https://orcid.org/0000-0001-2345-6789",
            "corresponding": True,
            "affiliation": "",
        }
    ]
    g = [
        {
            "family": "Byron",
            "given": "Ada",
            "email": "a@x.com",
            "orcid": "0000-0001-2345-6789",
            "corresponding": True,
            "affiliation": "",
        }
    ]
    assert email_f1(e, g) == 1.0
    assert orcid_f1(e, g) == 1.0
    assert corresponding_acc(e, g) == 1.0
    g2 = [dict(g[0], corresponding=False)]
    assert corresponding_acc(e, g2) is None  # gold flags none
    assert email_f1(e, [dict(g[0], email="")]) is None


def test_affiliation_sim_pairs_by_family():
    e = [
        {
            "family": "Byron",
            "given": "Ada",
            "affiliation": "Analytical Engine Inst",
            "email": "",
            "orcid": "",
            "corresponding": False,
        }
    ]
    g = [
        {
            "family": "Byron",
            "given": "Ada",
            "affiliation": "Analytical Engine Institute",
            "email": "",
            "orcid": "",
            "corresponding": False,
        }
    ]
    assert affiliation_sim(e, g) > 0.85
    assert affiliation_sim(e, [dict(g[0], affiliation="")]) is None


def test_affiliation_sim_empty_gold_affiliation_does_not_steal_later_match():
    # Regression: a gold author with empty affiliation must still consume its
    # matching extraction author, so a later same-family gold author doesn't
    # steal the earlier extraction author's slot.
    g = [
        {"family": "Jia", "given": "X", "affiliation": ""},
        {"family": "Jia", "given": "Y", "affiliation": "Tsinghua University"},
    ]
    e = [
        {"family": "Jia", "given": "X", "affiliation": "Wrong Uni For X"},
        {"family": "Jia", "given": "Y", "affiliation": "Tsinghua Univ"},
    ]
    assert affiliation_sim(e, g) > 0.7


def test_score_paper_emits_new_keys():
    e = {
        "title": "T",
        "doi": "",
        "abstract": "",
        "authors": [],
        "references": [],
        "reference_count": 0,
        "keywords": [],
    }
    s = score_paper(
        e, {"title": "T", "authors": [], "references": [], "reference_count": 0, "keywords": []}
    )
    for k in ("keywords_f1", "affiliation_sim", "email_f1", "orcid_f1", "corresponding_acc"):
        assert k in s
