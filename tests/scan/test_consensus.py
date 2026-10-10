"""Two-recognizer consensus on scanned regions."""

import asyncio

from PIL import Image

from bibr.ocr.profiles import GLM_PROFILE
from bibr.scan.consensus import (
    ConsensusConfig,
    Recognizer,
    apply_consensus,
    comparable,
    disagreement,
    medoid,
)


class _Backend:
    """Answers each crop by its width, so a test maps regions to readings."""

    name = "fake"

    def __init__(self, by_width, *, fail=False):
        self.by_width = by_width
        self.fail = fail
        self.calls = 0

    async def recognize(self, image, prompt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return self.by_width[image.size[0]]

    async def shutdown(self):
        pass


def _recognizer(by_width, **kw):
    return Recognizer(role="consensus", backend=_Backend(by_width, **kw), profile=GLM_PROFILE)


def _region(index, content, x1, label="text", native=False):
    region = {
        "index": index,
        "native_label": label,
        "label": label,
        "content": content,
        "bbox_2d": [0, 0, x1, 100],
    }
    if native:
        region["_native_text_used"] = True
    return region


def _image():
    return Image.new("RGB", (1000, 1000), "white")


def _run(pages, kinds, second, escalation=None, **config):
    return asyncio.run(
        apply_consensus(
            pages,
            [_image() for _ in pages],
            list(range(len(pages))),
            kinds,
            second=second,
            escalation=escalation,
            config=ConsensusConfig(**config),
        )
    )


def test_comparable_drops_markup_and_hyphen_breaks():
    assert comparable("**Intro-  duction**\n\n# Text") == "Introduction Text"


def test_disagreement_bounds():
    assert disagreement("same text", "same  text") == 0.0
    assert disagreement("", "") == 0.0
    assert disagreement("abc", "") == 1.0


def test_medoid_prefers_the_reading_two_agree_on():
    readings = [("primary", "Tbe cat"), ("second", "The cat"), ("escalation", "The cat")]
    assert medoid(readings) in {"second", "escalation"}
    assert medoid([("primary", "a"), ("second", "b"), ("escalation", "c")]) == "escalation"


def test_only_scan_pages_and_ocr_text_regions_are_compared():
    pages = [
        [
            _region(0, "Agreed text", 100),
            _region(1, "native", 200, native=True),
            _region(2, "<table/>", 300, label="table"),
        ],
        [_region(0, "Born digital", 400)],
    ]
    second = _recognizer({100: "Agreed text", 200: "x", 300: "x", 400: "x"})
    report = _run(pages, {0: "scan", 1: "born_digital"}, second)
    assert report.compared == 1
    assert second.backend.calls == 1
    assert pages[0][0]["_ocr_consensus"]["score"] == 0.0
    assert "_ocr_consensus" not in pages[1][0]


def test_worst_disagreements_are_flagged_within_the_budget():
    pages = [[_region(i, f"Region number {i}", 100 + i) for i in range(20)]]
    readings = {100 + i: f"Region number {i}" for i in range(20)}
    readings[100] = "Rcgion nurnber O"  # mild
    readings[101] = "completely different"  # worst
    report = _run(pages, {0: "scan"}, _recognizer(readings), escalate_share=0.05)
    assert report.compared == 20
    assert report.disagreeing == 2
    assert report.escalated == 1  # ceil(0.05 * 20)
    assert pages[0][1]["_ocr_consensus"]["escalated"] is True
    assert "escalated" not in pages[0][0]["_ocr_consensus"]
    # No escalation recognizer: the primary reading stands.
    assert pages[0][1]["content"] == "Region number 1"
    assert report.replaced == 0


def test_empty_primary_takes_the_second_reading():
    pages = [[_region(0, "", 100)]]
    report = _run(pages, {0: "scan"}, _recognizer({100: "Recovered text"}))
    assert pages[0][0]["content"] == "Recovered text"
    assert pages[0][0]["_ocr_consensus"]["chosen"] == "second"
    assert report.replaced == 1


def test_escalation_keeps_the_medoid_reading():
    pages = [[_region(0, "Tbe qnick hrown f0x", 100)]]
    second = _recognizer({100: "The quick brown fox"})
    escalation = _recognizer({100: "The quick brown fox."})
    escalation.role = "escalation"
    report = _run(pages, {0: "scan"}, second, escalation)
    assert pages[0][0]["content"] in {"The quick brown fox", "The quick brown fox."}
    assert pages[0][0]["_raw_ocr_content"] == "Tbe qnick hrown f0x"
    assert report.escalated == report.replaced == 1


def test_a_failing_second_recognizer_leaves_the_primary_reading():
    pages = [[_region(0, "Primary text", 100)]]
    report = _run(pages, {0: "scan"}, _recognizer({}, fail=True))
    assert report.compared == 0
    assert report.failed == 1
    assert pages[0][0]["content"] == "Primary text"
    assert "_ocr_consensus" not in pages[0][0]


def _ocr_settings(**ocr):
    from bibr.config import snapshot_settings

    settings = snapshot_settings()
    return settings.model_copy(update={"ocr": settings.ocr.model_copy(update=ocr)})


def test_start_recognizers_is_off_by_default():
    from bibr.scan.consensus import start_recognizers

    assert asyncio.run(start_recognizers(_ocr_settings())) is None


def test_recognizers_build_http_clients_with_their_own_profile(monkeypatch):
    from bibr.local.ocr import HttpOcrClient, PaddleHttpOcrClient
    from bibr.scan.consensus import start_recognizers

    async def ready(self):
        return None

    monkeypatch.setattr(HttpOcrClient, "wait_for_server", ready)
    monkeypatch.setattr(PaddleHttpOcrClient, "wait_for_server", ready)
    settings = _ocr_settings(
        consensus_backend="glm-http",
        consensus_url="http://127.0.0.1:9",
        escalation_backend="paddle-http",
        escalation_url="http://127.0.0.1:10",
    )
    second, escalation = asyncio.run(start_recognizers(settings))
    assert isinstance(second.backend, HttpOcrClient)
    assert second.profile.name == "glm"
    assert isinstance(escalation.backend, PaddleHttpOcrClient)
    assert escalation.profile.name == "paddle"


def test_an_unreachable_consensus_server_disables_consensus(monkeypatch):
    from bibr.local.ocr import HttpOcrClient
    from bibr.scan.consensus import start_recognizers

    async def down(self):
        raise ConnectionError("refused")

    monkeypatch.setattr(HttpOcrClient, "wait_for_server", down)
    settings = _ocr_settings(consensus_backend="glm-http", consensus_url="http://127.0.0.1:9")
    assert asyncio.run(start_recognizers(settings)) is None


def test_consensus_identity_names_what_decides_the_output():
    from bibr.scan.consensus import identity

    assert identity(_ocr_settings()) == "off"
    key = identity(_ocr_settings(consensus_backend="glm-http", escalation_backend="paddle-http"))
    assert key.startswith("glm-http:") and "paddle-http" in key
