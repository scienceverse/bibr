"""Front-matter tagger: label printed first-page words instead of generating them.

A multilingual encoder + CRF tags every word of the first pages' OCR regions
(title, names, affiliations, abstract, keywords, the paper's own journal
line, page furniture). :mod:`.assemble` turns the tags into the same response
objects the front-matter LLM calls return, so everything downstream (grounding,
field decisions, validation) runs unchanged. The LLM is called only when the
tagger is unsure (``ML_FRONT_TAGGER_MODE=primary``).
"""
