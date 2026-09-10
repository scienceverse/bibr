"""Stage Protocol structural conformance."""

from bibr.pipeline.stage import Stage


class _FakeStage:
    name = "fake"

    async def run(self, ctx) -> None:
        return None


def test_fake_stage_is_a_stage():
    assert isinstance(_FakeStage(), Stage)


def test_missing_run_is_not_a_stage():
    class Bad:
        name = "bad"

    assert not isinstance(Bad(), Stage)


def test_missing_name_is_not_a_stage():
    class Bad:
        async def run(self, ctx) -> None:
            return None

    assert not isinstance(Bad(), Stage)
