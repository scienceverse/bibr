"""Raw-completion-text extraction from a truncation error's chain.

Bridges instructor's ``IncompleteOutputException`` (possibly wrapped in
``UpstreamServiceError``) to the reference-batch salvage across the provider
completion shapes the pipeline actually sees.
"""

import json
from types import SimpleNamespace

from bibr.clients.llm import incomplete_output_text
from bibr.exceptions import UpstreamServiceError
from bibr.extract.ref_extractor import IncompleteOutputException


def _openai(content=None, tool_args=None):
    function = SimpleNamespace(arguments=tool_args) if tool_args is not None else None
    tool_calls = [SimpleNamespace(function=function)] if function is not None else None
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_message_content():
    exc = IncompleteOutputException(last_completion=_openai(content='{"references": ['))
    assert incomplete_output_text(exc) == '{"references": ['


def test_openai_tool_call_arguments():
    exc = IncompleteOutputException(last_completion=_openai(tool_args='{"references": [{"index"'))
    assert incomplete_output_text(exc) == '{"references": [{"index"'


def test_genai_text_shape():
    completion = SimpleNamespace(text='{"references": [{}]}', choices=None)
    exc = IncompleteOutputException(last_completion=completion)
    assert incomplete_output_text(exc) == '{"references": [{}]}'


def test_anthropic_tool_use_block():
    block = SimpleNamespace(type="tool_use", input={"references": [{"index": 1}]})
    completion = SimpleNamespace(content=[block], choices=None, text=None)
    exc = IncompleteOutputException(last_completion=completion)
    assert json.loads(incomplete_output_text(exc)) == {"references": [{"index": 1}]}


def test_unwraps_upstream_service_error():
    inner = IncompleteOutputException(last_completion=_openai(content="raw"))
    wrapped = UpstreamServiceError("LLM", "Failed to extract references", inner)
    assert incomplete_output_text(wrapped) == "raw"


def test_no_completion_returns_empty():
    assert incomplete_output_text(RuntimeError("boom")) == ""
    assert incomplete_output_text(IncompleteOutputException()) == ""
