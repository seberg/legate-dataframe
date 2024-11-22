import legate_dataframe


def test_version_constants_are_populated():
    # __version__ should always be non-empty
    assert isinstance(legate_dataframe.__version__, str)
    assert len(legate_dataframe.__version__) > 0
