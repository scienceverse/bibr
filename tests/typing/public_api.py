from pathlib import Path
from typing import assert_type

from bibr.api import Chewer, ChewFailure, Result, chew_file, chew_many
from bibr.config import snapshot_settings
from bibr.document_api import DocumentRecord, DocumentResult, chew_document
from bibr.export import DocumentExport, PaperExport

settings = snapshot_settings()

single = chew_file(Path("paper.pdf"), refs="off", settings=settings)
assert_type(single, Result)
assert_type(single.model, PaperExport)

batch = chew_many([Path("a.pdf"), Path("b.pdf")], refs="off", settings=settings)
assert_type(batch, list[Result | ChewFailure])

chewer = Chewer(settings=settings, refs="off")
assert_type(chewer, Chewer)

document = chew_document(Path("proceedings.pdf"), refs="off", settings=settings)
assert_type(document, DocumentResult)
assert_type(document.model, DocumentExport)
assert_type(document.records, list[DocumentRecord])
assert_type(document.records[0].paper, Result | None)
