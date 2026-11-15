"""Statement tagger: an opt-in ONNX classifier that marks the blocks holding paper statements.

:mod:`.runtime` is the model itself, with no dependency on the rest of bibr
(onnxruntime, ``tokenizers`` and numpy): ordered block texts in, typed
statements out. :mod:`.pipeline` feeds it bibr's paragraphs and joins what it
finds to statement resolution when ``ML_STATEMENT_TAGGER_MODEL_ID`` is set.
"""
