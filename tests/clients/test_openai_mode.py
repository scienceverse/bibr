"""OpenAI provider instructor-mode selection for custom (local) base_urls."""

import instructor

from bibr.clients.providers.openai import _resolve_instructor_mode


class TestResolveInstructorMode:
    def test_default_is_json_schema(self):
        assert _resolve_instructor_mode("") == instructor.Mode.JSON_SCHEMA

    def test_json_selects_json_object_mode(self):
        assert _resolve_instructor_mode("json") == instructor.Mode.JSON

    def test_md_json_selects_prompt_only_json_mode(self):
        assert _resolve_instructor_mode("md_json") == instructor.Mode.MD_JSON

    def test_tools_selects_tools(self):
        assert _resolve_instructor_mode("tools") == instructor.Mode.TOOLS

    def test_case_insensitive(self):
        assert _resolve_instructor_mode("JSON") == instructor.Mode.JSON

    def test_unknown_falls_back_to_json_schema(self):
        assert _resolve_instructor_mode("bogus") == instructor.Mode.JSON_SCHEMA
