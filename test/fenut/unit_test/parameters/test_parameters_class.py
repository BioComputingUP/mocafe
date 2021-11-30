import pytest
import pathlib
from mocafe.fenut.parameters import from_ods_sheet, Parameters


def test_parameters_init(odf_sheet_test):
    p = from_ods_sheet(odf_sheet_test, "Sheet1")
    assert type(p) is Parameters, "p should be of Parameters class"


def test_parameters_get_value(odf_sheet_test):
    p = from_ods_sheet(odf_sheet_test, "Sheet1")
    value = p.get_value("lattice_unit")
    assert value == 1.25, "It should be 1.25"


def test_parameters_set_value(odf_sheet_test):
    p = from_ods_sheet(odf_sheet_test, "Sheet1")
    new_value = 1.0
    p.set_value("lattice_unit", new_value)
    value = p.get_value("lattice_unit")
    assert value == new_value, "The parameter value should have been setted to 1."

