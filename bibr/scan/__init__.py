"""The scan path: page triage, article splitting and OCR consensus.

Scanned and old papers fail in ways born-digital ones do not: a page of a
journal issue carries the end of the previous article or the start of the
next, and a single recognizer misreads degraded print with nothing to say it
did. This package holds the three pieces that act on scanned pages only:

- :mod:`bibr.scan.page_kind` classes each page as born-digital, born-digital
  with a broken text layer, or a scan.
- :mod:`bibr.scan.articles` finds article boundaries on scanned pages and
  drops the regions of neighbouring articles before parsing.
- :mod:`bibr.scan.consensus` reads scan regions with a second recognizer and
  escalates the regions where the two disagree most.
"""
