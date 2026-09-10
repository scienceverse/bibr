from pathlib import Path
from typing import assert_type

from bibr.api import Chewer, ChewFailure, Result, chew_file, chew_many
from bibr.config import GlobalSettings
from bibr.export import PaperExport

settings = GlobalSettings()

single = chew_file(Path("paper.pdf"), refs="off", settings=settings)
assert_type(single, Result)
assert_type(single.model, PaperExport)

batch = chew_many([Path("a.pdf"), Path("b.pdf")], refs="off", settings=settings)
assert_type(batch, list[Result | ChewFailure])

chewer = Chewer(settings=settings, refs="off")
assert_type(chewer, Chewer)
