"""Serve caches only final results.

A result shaped by a failure a retry could avoid (a timed-out enrichment, a
failed LLM call, a blocking issue, a failed field) used to be cached for the
full TTL, so every later request for the same PDF got the degraded export back.
"""

import pytest

from bibr.pipeline.context import RunConfig
from bibr.serve.deployments.pipeline import BibrPipelineAPI, _is_final_result
from tests.serve.test_cache_singleflight import _inputs, _MemoryCache


def _payload(**extraction) -> dict:
    return {"paper_id": "p", "source": {"file_name": "paper.pdf"}, "extraction": extraction}


class _SequencePipeline:
    def __init__(self, results):
        self._config = RunConfig()
        self._results = list(results)
        self.calls = 0

    async def process_file(self, filename, paper_id, content, config):  # noqa: ARG002
        self.calls += 1
        return self._results.pop(0)


_DEGRADED = _payload(
    warnings=[{"code": "CROSSREF_ENRICHMENT_FAILED", "message": "Crossref enrichment failed"}]
)
_CLEAN = _payload(warnings=[])


async def test_degraded_result_is_not_cached_and_the_next_request_reruns(tmp_path):
    api = BibrPipelineAPI(upload_root=tmp_path)
    pipeline = _SequencePipeline([_DEGRADED, _CLEAN, _CLEAN])
    cache = _MemoryCache()
    api._pipeline, api._cache, api._cache_inited, api._inflight_sem = pipeline, cache, True, None

    first = await api.predict(_inputs())
    second = await api.predict(_inputs())
    third = await api.predict(_inputs())

    assert first["paper_json"]["extraction"]["warnings"]
    assert second["paper_json"]["extraction"]["warnings"] == []
    assert third["paper_json"]["extraction"]["warnings"] == []
    # The degraded run was not cached; the clean one was, and served the third.
    assert pipeline.calls == 2
    assert cache.set_calls == 1


@pytest.mark.parametrize(
    ("payload", "final"),
    [
        (_CLEAN, True),
        (_payload(), True),
        ({"paper_id": "p"}, True),
        (_payload(warnings=[{"code": "LOW_TEXT_QUALITY", "message": "m"}]), True),
        (_payload(warnings=[{"code": "OCR_REGION_FAILED", "message": "m"}]), False),
        (_payload(warnings=[{"code": "CITATION_LLM_FAILED", "message": "m"}]), False),
        (
            _payload(
                validation={
                    "promotable": False,
                    "issues": [{"code": "VAL_METADATA_FIELD_FAILED", "blocking": True}],
                }
            ),
            False,
        ),
        (
            _payload(
                validation={
                    "promotable": False,
                    "issues": [{"code": "VAL_REFERENCES_INCOMPLETE", "blocking": True}],
                }
            ),
            False,
        ),
        # A front-matter abstention is a decision the same input makes again.
        (
            _payload(
                validation={
                    "promotable": False,
                    "issues": [{"code": "VAL_METADATA_MULTI_ITEM", "blocking": True}],
                }
            ),
            True,
        ),
        (_payload(validation={"promotable": True, "issues": []}), True),
        (_payload(enrichment={"complete": False, "refs_enriched": 1, "refs_total": 3}), False),
        (_payload(fields={"title": {"state": "failed", "source": "llm", "issues": []}}), False),
        (_payload(fields={"title": {"state": "absent", "source": None, "issues": []}}), True),
    ],
)
def test_final_result_rule(payload, final):
    assert _is_final_result(payload) is final
