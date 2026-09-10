from bibr.export.usage import build_usage_export

LABELS = {
    ("extract_core", "google", "gemini-flash-lite"): {
        "calls": 2,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cached_input_tokens": 40,
    },
    ("parse_refs", "openai", "nuextract-3"): {
        "calls": 1,
        "input_tokens": 300,
        "output_tokens": 50,
        "total_tokens": 350,
        "cached_input_tokens": 0,
    },
}


def test_returns_none_when_nothing_was_tracked():
    assert build_usage_export(None) is None
    assert build_usage_export({}) is None


def test_breakdown_carries_label_provider_and_model():
    usage = build_usage_export(LABELS)
    rows = {(r["label"], r["provider"], r["model"]): r for r in usage["breakdown"]}
    assert rows[("extract_core", "google", "gemini-flash-lite")]["provider"] == "google"
    assert rows[("parse_refs", "openai", "nuextract-3")]["model"] == "nuextract-3"


def test_totals_equal_the_sum_of_breakdown_rows():
    usage = build_usage_export(LABELS)
    for field in ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        assert usage["totals"][field] == sum(r[field] for r in usage["breakdown"]), field


def test_breakdown_is_ordered_by_label_for_stable_output():
    usage = build_usage_export(LABELS)
    keys = [(r["label"], r["provider"], r["model"]) for r in usage["breakdown"]]
    assert keys == sorted(keys)


def test_unknown_extra_bucket_keys_are_not_exported():
    labels = {
        ("x", "google", "gemini-flash-lite"): {
            **LABELS[("extract_core", "google", "gemini-flash-lite")],
            "rate_limit_wait_ms": 99,
        }
    }
    usage = build_usage_export(labels)
    assert "rate_limit_wait_ms" not in usage["breakdown"][0]


def test_none_provider_and_model_do_not_break_the_sort():
    """provider/model are typed str | None; a None entry sharing a label with
    a real-engine entry must not raise TypeError from comparing str < None,
    and None must serialize as None (not "") in the row."""
    labels = {
        ("parse_refs", None, None): {
            "calls": 1,
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "cached_input_tokens": 0,
        },
        ("parse_refs", "openai", "nuextract-3"): {
            "calls": 3,
            "input_tokens": 300,
            "output_tokens": 60,
            "total_tokens": 360,
            "cached_input_tokens": 0,
        },
    }

    usage = build_usage_export(labels)

    rows = [r for r in usage["breakdown"] if r["label"] == "parse_refs"]
    assert len(rows) == 2

    none_row = next(r for r in rows if r["provider"] is None)
    assert none_row["model"] is None
    assert none_row["calls"] == 1

    openai_row = next(r for r in rows if r["provider"] == "openai")
    assert openai_row["model"] == "nuextract-3"
    assert openai_row["calls"] == 3

    for field in ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        assert usage["totals"][field] == sum(r[field] for r in usage["breakdown"]), field


def test_same_label_under_two_providers_yields_two_separated_rows():
    """A label (e.g. parse_refs) run under two engines within one file must
    not collapse into a single row — each engine's counters land in its own
    row, keyed by the full (label, provider, model) triple."""
    labels = {
        ("parse_refs", "google", "gemini-flash-lite"): {
            "calls": 2,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cached_input_tokens": 10,
        },
        ("parse_refs", "openai", "nuextract-3"): {
            "calls": 3,
            "input_tokens": 300,
            "output_tokens": 60,
            "total_tokens": 360,
            "cached_input_tokens": 0,
        },
    }

    usage = build_usage_export(labels)

    rows = [r for r in usage["breakdown"] if r["label"] == "parse_refs"]
    assert len(rows) == 2

    google_row = next(r for r in rows if r["provider"] == "google")
    openai_row = next(r for r in rows if r["provider"] == "openai")
    assert google_row["model"] == "gemini-flash-lite"
    assert google_row["calls"] == 2
    assert google_row["input_tokens"] == 100
    assert openai_row["model"] == "nuextract-3"
    assert openai_row["calls"] == 3
    assert openai_row["input_tokens"] == 300

    for field in ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        assert usage["totals"][field] == sum(r[field] for r in usage["breakdown"]), field
