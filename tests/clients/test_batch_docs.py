"""The LLM response cache has one prefill path, and the texts describe it.

``bibr batch --llm-batch`` (bibr.batch.llm_batch) fills the cache from
Anthropic Message Batches. These tests pin that the cache docstring and the
CACHE_LLM description say so, and that the batch client itself still writes
no cache: only the orchestration does, through ``llm_cache.response_body``.
"""

import bibr.clients.batch as batch
import bibr.clients.llm_cache as llm_cache
from bibr.config import GlobalSettings


def test_llm_cache_docstring_names_the_one_prefill_path():
    doc = llm_cache.__doc__ or ""
    assert "``bibr batch --llm-batch``" in doc
    assert ":func:`response_body`" in doc
    assert "half" not in doc and "price" not in doc


def test_cache_llm_setting_text_names_the_prefill_path():
    description = GlobalSettings.model_fields["cache"].annotation.model_fields["llm"].description
    assert "`bibr batch --llm-batch` fills it" in description


def test_the_filler_stores_through_the_shared_body():
    import inspect

    import bibr.batch.llm_batch as llm_batch

    assert "response_body(answer)" in inspect.getsource(llm_batch)


def test_batch_adapter_writes_to_no_cache():
    import inspect

    source = inspect.getsource(batch)
    assert "llm_cache" not in source
    assert "cache_dir" not in source
