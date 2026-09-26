"""Offline batch layer: docs must not promise a prefill that isn't wired (clients-llm-15).

The fix corrects the config/docs text instead of deleting code, so these
tests pin the corrected claims: nothing outside tests uses the batch
adapter to fill the LLM cache, and neither text promises that it does.
"""

import bibr.clients.batch as batch
import bibr.clients.llm_cache as llm_cache
from bibr.config import GlobalSettings


def test_llm_cache_docstring_promises_no_batch_prefill():
    doc = llm_cache.__doc__ or ""
    assert "never touches" in doc
    assert "writes them here" not in doc
    assert "half" not in doc and "price" not in doc


def test_cache_llm_setting_text_promises_no_batch_prefill():
    description = GlobalSettings.model_fields["cache"].annotation.model_fields["llm"].description
    assert "Not written by the offline batch layer" in description
    assert "prefill target" not in description


def test_batch_adapter_writes_to_no_cache():
    import inspect

    source = inspect.getsource(batch)
    assert "llm_cache" not in source
    assert "cache_dir" not in source
