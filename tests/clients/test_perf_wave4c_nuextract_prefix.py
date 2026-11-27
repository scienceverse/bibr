"""Wave 4C tests: NuExtract references_parse keeps instructions prefix byte-identical across batches."""

from __future__ import annotations

from bibr.clients.nuextract import build_native_request_kwargs
from bibr.clients.prompts import PROMPTS, prompt_text
from bibr.config import GlobalSettings
from bibr.schemas import PaperReferenceList


def test_nuextract_references_parse_prefix_is_byte_identical_across_batches() -> None:
    settings = GlobalSettings()
    settings.llm.model = "numind/NuExtract3-FP8"
    spec = PROMPTS["references_parse"]

    parts_1 = spec.build_user(boundary="B1", text="1. First ref", start_index=1)
    parts_26 = spec.build_user(boundary="B2", text="26. Later ref", start_index=26)

    kw_1 = build_native_request_kwargs(
        settings=settings,
        response_model=PaperReferenceList,
        system=spec.system,
        messages=[{"role": "user", "content": parts_1}],
        max_tokens=None,
    )
    kw_26 = build_native_request_kwargs(
        settings=settings,
        response_model=PaperReferenceList,
        system=spec.system,
        messages=[{"role": "user", "content": parts_26}],
        max_tokens=None,
    )

    tk_1 = kw_1["extra_body"]["chat_template_kwargs"]
    tk_26 = kw_26["extra_body"]["chat_template_kwargs"]
    # Template and instructions (rendered before the document in NuExtract's chat template)
    # stay 100% byte-identical across batches so vLLM/MLX prefix caches hit through the end
    # of instructions.
    assert tk_1["template"] == tk_26["template"]
    assert tk_1["instructions"] == tk_26["instructions"]
    assert "has index" not in tk_1["instructions"]

    # The start_index trailer follows the closing fence in the projected document payload.
    doc_1 = kw_1["messages"][1]["content"]
    doc_26 = kw_26["messages"][1]["content"]
    assert doc_1.endswith(
        "--- B1 END ---\n\nThe first numbered entry in the supplied document has index 1."
    )
    assert doc_26.endswith(
        "--- B2 END ---\n\nThe first numbered entry in the supplied document has index 26."
    )

    # Flat non-NuExtract prompt rendering is unchanged.
    assert prompt_text(parts_26).endswith(
        "--- B2 END ---\n\nThe first numbered entry in the supplied document has index 26."
    )
