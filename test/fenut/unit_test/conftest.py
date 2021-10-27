import pytest
import pathlib


@pytest.fixture
def odf_sheet_test2():
    return pathlib.Path("test/fenut/unit_test/test_sheet2.ods")


@pytest.fixture
def odf_sheet_test():
    return pathlib.Path("test/fenut/unit_test/test_sheet.ods")
