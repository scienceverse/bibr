from bibr.export.json_export import export_paper_to_json
from bibr.ocr.postprocess import merge_text_blocks
from tests.test_export_units import _minimal_paper


def test_source_evidence_is_exported_only_with_regions():
    paper = _minimal_paper()
    source = {"pages": [{"index": 3, "source": {"counts": {"unassigned": 12}}}]}
    paper.contents.native_source = source
    assert "_native_source" not in export_paper_to_json(paper, validate=False)
    debug = export_paper_to_json(paper, validate=False, include_regions=True)
    assert debug["_native_source"] == source


def test_joined_text_retains_each_source_and_repair_receipt():
    pages = [
        {
            "label": "text",
            "content": "Inter-",
            "_source_region_ids": ["p0:r0"],
            "_native_spans": [{"span_id": "p0:r0:s0"}],
        },
        {
            "label": "text",
            "content": "national.",
            "_source_region_ids": ["p0:r1"],
            "_native_spans": [{"span_id": "p0:r1:s0"}],
            "_formula_proposals": [{"content": "x"}],
        },
    ]
    joined = merge_text_blocks(pages)
    assert len(joined) == 1
    assert joined[0]["_source_region_ids"] == ["p0:r0", "p0:r1"]
    assert [s["span_id"] for s in joined[0]["_native_spans"]] == ["p0:r0:s0", "p0:r1:s0"]
    assert joined[0]["_formula_proposals"] == [{"content": "x"}]
