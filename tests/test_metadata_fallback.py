from unittest import mock

import pandas as pd

from bibr.extract.extractor import MetadataExtractor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection


def test_metadata_fallback():
    # Create dummy sentences DF with no recognizable sections
    data = {
        "text": [f"Sentence {i}" for i in range(300)],
        "section_name": ["Unknown Section"] * 300,
    }
    df = pd.DataFrame(data)

    # Mock PaperContents
    mock_contents = mock.Mock(spec=PaperContents)
    mock_contents.sentences_df = df
    mock_contents.sections = [
        PaperSection(0, "Unknown Section", 2, None, CanonicalSection.UNKNOWN, 0.0),
    ]

    extractor = MetadataExtractor(mock_contents)

    # Direct test
    cutoff = extractor.locator.get_cutoff_index()
    assert cutoff == 250
