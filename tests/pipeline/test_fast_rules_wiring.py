"""The local rules a fast run leans on are switched by the run, not by its callers."""

from __future__ import annotations

from bibr.config import GlobalSettings
from bibr.run_mode import fast_settings


def test_the_rules_are_off_in_a_default_run():
    settings = GlobalSettings()
    assert settings.FRONT_TAGGER_RULES is False
    assert settings.EQUATION_REPAIR_GLITCHES is False


def test_a_fast_run_turns_the_rules_on():
    fast = fast_settings(GlobalSettings())
    assert fast.FRONT_TAGGER_RULES is True
    assert fast.EQUATION_REPAIR_GLITCHES is True


async def test_the_parse_stage_reads_the_rules_flag_from_the_runs_settings(monkeypatch):
    from bibr.front_tagger import runtime as runtime_mod
    from bibr.pipeline.stages import parse_segment

    seen: dict[str, bool] = {}

    def fake_tag_front_matter(tagger, regions, *, first_page_index=0, rules=False):
        seen["rules"] = rules
        return None

    monkeypatch.setattr(runtime_mod, "tag_front_matter", fake_tag_front_matter)

    class _RM:
        def ensure_front_tagger(self, settings=None):
            return object()

    await parse_segment._tag_front_matter(
        _RM(), [], first_page_index=0, settings=GlobalSettings(FRONT_TAGGER_RULES=True)
    )
    assert seen == {"rules": True}

    await parse_segment._tag_front_matter(
        _RM(), [], first_page_index=0, settings=GlobalSettings()
    )
    assert seen == {"rules": False}
