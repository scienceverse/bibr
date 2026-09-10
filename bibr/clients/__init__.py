"""Client modules — explicit-import only.

Heavy clients (``llm.py`` imports Instructor + provider SDKs;
``crossref.py`` imports habanero) are intentionally NOT re-exported
from this ``__init__`` so a bare ``import bibr.clients`` stays cheap.
Import the submodule you need:

    from bibr.clients.llm import LLMClient
    from bibr.clients.crossref import CrossrefClient
"""
