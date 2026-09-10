"""Tests for LLMClient.segment_references (anchor emit)."""

import asyncio
from unittest import mock

import pytest

from bibr.clients.llm import LLMClient, _window_ref_text
from bibr.clients.prompts import prompt_text
from bibr.schemas import PaperReferenceLLM, RefAnchors


@pytest.mark.parametrize(
    "text,budget",
    [
        ("short", 200),  # fits → single window, unchanged
        ("", 200),  # empty
        ("\n".join(f"[{i}] Author{i}. Title {i}." for i in range(100)), 80),  # many lines
        ("x" * 1000, 200),  # one over-long line with NO newline → must hard-split
        ("a" * 199 + "\n" + "b" * 199, 200),  # two max-ish lines
    ],
)
def test_window_ref_text_invariants(text, budget):
    windows = _window_ref_text(text, budget)
    # Every window fits the budget (so each LLM request fits the model context).
    assert all(len(w) <= budget for w in windows), [len(w) for w in windows]
    # Lossless and order-preserving: concatenation reconstructs the input exactly.
    assert "".join(windows) == text
    # Text already within budget stays a single window (no needless fan-out).
    if len(text) <= budget:
        assert windows == [text]


@pytest.fixture
def client():
    c = LLMClient()
    # Neutralise the rate limiter so the test does not touch Redis/local timing.
    c._limiter = mock.Mock()
    c._limiter.acquire = mock.AsyncMock()
    return c


async def test_segment_references_returns_anchor_list(client):
    fake = RefAnchors(anchors=["Smith, J. (2020).", "Doe, A. (2019)."])
    with mock.patch.object(
        client, "_invoke_structured", new=mock.AsyncMock(return_value=fake)
    ) as inv:
        anchors = await client.segment_references("ref text block", file_hash="h")
    assert anchors == ["Smith, J. (2020).", "Doe, A. (2019)."]
    # The references text must be passed as data inside the user message.
    _, kwargs = inv.call_args
    messages = kwargs.get("messages") or inv.call_args.args[1]
    assert any("ref text block" in prompt_text(m["content"]) for m in messages)


async def test_segment_references_preserves_duplicate_sibling_anchors(client):
    """Two sibling references (same authors + year, e.g. 2013a/2013b) can yield
    byte-identical opening anchors. Both must survive segmentation so that
    ``anchor_snap`` can claim a distinct position for each — otherwise the two
    references silently merge into one.

    Regression: a cross-window string dedup collapsed identical anchors even in
    the common single-window case, dropping the second reference.
    """
    dup = "Lakens, D. (2013)."
    fake = RefAnchors(anchors=[dup, dup])
    with mock.patch.object(client, "_invoke_structured", new=mock.AsyncMock(return_value=fake)):
        anchors = await client.segment_references("ref text block", file_hash="h")
    assert anchors == [dup, dup]


async def test_segment_references_wraps_failure(client):
    from bibr.exceptions import UpstreamServiceError

    with mock.patch.object(
        client, "_invoke_structured", new=mock.AsyncMock(side_effect=RuntimeError("boom"))
    ):
        with pytest.raises(UpstreamServiceError):
            await client.segment_references("ref text", file_hash="h")


async def test_segment_references_typed_window_failure_wins_over_ordinary_failure(client):
    from bibr.exceptions import ProcessingError

    typed_error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
    )
    release_typed = asyncio.Event()

    async def fail_window(window):
        if window == "ordinary":
            release_typed.set()
            raise RuntimeError("ordinary sibling failure")
        await release_typed.wait()
        raise typed_error

    with (
        mock.patch("bibr.clients.llm._window_ref_text", return_value=["ordinary", "typed"]),
        mock.patch.object(client, "_segment_window", side_effect=fail_window),
    ):
        with pytest.raises(ProcessingError) as raised:
            await client.segment_references("two windows", file_hash="h")

    assert raised.value is typed_error


def _window_payload(content: str) -> str:
    """Extract the reference-text slice a window sent (between boundary markers)."""
    import re

    m = re.search(r"START ---\n(.*)\n--- [0-9a-f]+ END ---", content, re.S)
    assert m, f"no boundary-delimited ref block in payload: {content!r}"
    return m.group(1)


async def test_segment_references_windows_long_input_preserving_all_anchors():
    """A references block larger than the per-call input budget is segmented in
    multiple windows — each within budget — and EVERY anchor is preserved.

    Regression: a single capped call truncates long bibliographies (the cause of
    the 24K-context Gemma overflow), silently dropping every reference past the cap.
    """
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.llm.max_input_chars = 200  # tiny budget → forces multiple windows
    client = LLMClient(settings=settings)
    client._limiter = mock.Mock()
    client._limiter.acquire = mock.AsyncMock()

    # 20 references, one per line; total well over the 200-char budget.
    refs = [f"[{i}] Author{i}, B. ({2000 + i}). Title number {i}." for i in range(1, 21)]
    ref_text = "\n".join(refs)
    assert len(ref_text) > settings.llm.max_input_chars  # sanity: needs windowing

    seen_payloads: list[str] = []

    def fake_invoke(response_model, messages, system_prompt, **kwargs):
        content = prompt_text(messages[0]["content"])
        seen_payloads.append(content)
        # Emit one anchor ("[i]") per reference line present in THIS window.
        block = _window_payload(content)
        anchors = [
            line[: line.index("]") + 1] for line in block.splitlines() if line.startswith("[")
        ]
        return RefAnchors(anchors=anchors)

    with mock.patch.object(
        client, "_invoke_structured", new=mock.AsyncMock(side_effect=fake_invoke)
    ):
        anchors = await client.segment_references(ref_text, file_hash="h")

    # More than one window was needed.
    assert len(seen_payloads) > 1, "long input should be split into multiple windows"
    # Each window's reference-text slice stays within the budget.
    for content in seen_payloads:
        assert len(_window_payload(content)) <= settings.llm.max_input_chars
    # Every reference's anchor is present, in order, none lost to truncation.
    assert anchors == [f"[{i}]" for i in range(1, 21)]


async def test_segment_windows_by_ref_seg_window_chars_not_max_input():
    """Seg OUTPUT is bounded by ``ref_seg_window_chars`` even when the input
    budget (``max_input_chars``) is large. Regression (LIGO 118->0): a 34K refs
    block fits max_input_chars in ONE window, so all 118 anchors emit in a single
    call that exceeds the LLM hard-timeout 3x and the whole paper falls to 0 refs.
    """
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.llm.max_input_chars = 300_000  # large input budget (the default)
    settings.llm.ref_seg_window_chars = 4_000  # but seg output is bounded smaller
    client = LLMClient(settings=settings)
    client._limiter = mock.Mock()
    client._limiter.acquire = mock.AsyncMock()

    refs = [
        f"[{i}] Author{i}, B. C., Someone{i}, D. E., and Third{i}, F. ({2000 + i}). "
        f"A reasonably long reference title number {i} that pads the line out. "
        f"Journal of Testing, {i}({i}), {i * 10}-{i * 10 + 9}."
        for i in range(1, 61)
    ]
    ref_text = "\n".join(refs)
    assert len(ref_text) > settings.llm.ref_seg_window_chars  # needs windowing
    assert len(ref_text) < settings.llm.max_input_chars  # but fits input in one piece

    seen: list[str] = []

    def fake_invoke(response_model, messages, system_prompt, **kwargs):
        content = prompt_text(messages[0]["content"])
        seen.append(content)
        block = _window_payload(content)
        anchors = [
            line[: line.index("]") + 1] for line in block.splitlines() if line.startswith("[")
        ]
        return RefAnchors(anchors=anchors)

    with mock.patch.object(
        client, "_invoke_structured", new=mock.AsyncMock(side_effect=fake_invoke)
    ):
        anchors = await client.segment_references(ref_text, file_hash="ligo")

    assert len(seen) > 1, "seg must window by ref_seg_window_chars, not max_input_chars"
    for content in seen:
        assert len(_window_payload(content)) <= settings.llm.ref_seg_window_chars
    assert anchors == [f"[{i}]" for i in range(1, 61)]


async def test_segment_references_rate_limits_each_window():
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.llm.max_input_chars = 80
    settings.llm.ref_seg_window_chars = 80
    client = LLMClient(settings=settings)
    client._limiter = mock.Mock()
    client._limiter.acquire = mock.AsyncMock()

    refs = [f"[{i}] Author{i}, B. ({2000 + i}). Title {i}." for i in range(1, 8)]
    ref_text = "\n".join(refs)
    expected_windows = _window_ref_text(ref_text, settings.llm.ref_seg_window_chars)
    assert len(expected_windows) > 1

    def fake_invoke(response_model, messages, system_prompt, **kwargs):
        block = _window_payload(prompt_text(messages[0]["content"]))
        anchors = [
            line[: line.index("]") + 1] for line in block.splitlines() if line.startswith("[")
        ]
        return RefAnchors(anchors=anchors)

    with mock.patch.object(
        client, "_invoke_structured", new=mock.AsyncMock(side_effect=fake_invoke)
    ):
        anchors = await client.segment_references(ref_text, file_hash="h")

    assert anchors == [f"[{i}]" for i in range(1, 8)]
    assert client._limiter.acquire.await_count == len(expected_windows)


async def test_segment_does_not_truncate_beyond_legacy_12_window_cap():
    """A bibliography that needs more than the old 12-window cap must NOT be
    truncated now that windows are smaller (the cap was raised to cover any
    real reference list)."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.llm.max_input_chars = 60  # tiny → each ref becomes its own window
    client = LLMClient(settings=settings)
    client._limiter = mock.Mock()
    client._limiter.acquire = mock.AsyncMock()

    refs = [f"[{i}] Author{i}, B. ({2000 + i}). Title {i}." for i in range(1, 21)]
    ref_text = "\n".join(refs)

    def fake_invoke(response_model, messages, system_prompt, **kwargs):
        block = _window_payload(prompt_text(messages[0]["content"]))
        anchors = [
            line[: line.index("]") + 1] for line in block.splitlines() if line.startswith("[")
        ]
        return RefAnchors(anchors=anchors)

    with mock.patch.object(
        client, "_invoke_structured", new=mock.AsyncMock(side_effect=fake_invoke)
    ):
        anchors = await client.segment_references(ref_text, file_hash="h")

    # 20 single-ref windows > old cap of 12; none may be dropped.
    assert anchors == [f"[{i}]" for i in range(1, 21)]


class TestExtractReferencesStartIndex:
    """Batched parse renumbers returned refs from start_index."""

    def _make_llm_ref(self, idx, title):
        return PaperReferenceLLM(
            index=idx,
            title=title,
            authors="Author",
            first_page=None,
            last_page=None,
            volume=None,
            issue=None,
            year=2020,
            container=None,
            doi=None,
        )

    async def test_indices_renumbered_from_start_index(self):
        from bibr.schemas import PaperReferenceList

        client = LLMClient()
        client._limiter = mock.Mock()
        client._limiter.acquire = mock.AsyncMock()

        # Model returns local indices 1,2 — must be renumbered to 6,7.
        fake = PaperReferenceList(
            references=[self._make_llm_ref(1, "First"), self._make_llm_ref(2, "Second")]
        )
        with mock.patch.object(client, "_invoke_structured", new=mock.AsyncMock(return_value=fake)):
            refs = await client.extract_references("6. ...\n7. ...", start_index=6)
        assert [r.index for r in refs] == [6, 7]
