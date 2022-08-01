import fenics
import pytest
from mocafe.angie.af_sourcing import RandomSourceMap
import pathlib
from mocafe.fenut.parameters import from_ods_sheet, Parameters
import os


@pytest.fixture
def parameters() -> Parameters:
    # the saved mesh is 300x300
    return from_ods_sheet(pathlib.Path(f"{os.path.dirname(__file__)}/test_parameters.ods"), "SimParams")


@pytest.fixture
def mesh():
    mesh = fenics.RectangleMesh(fenics.Point(0, 0),
                                fenics.Point(300, 300),
                                300,
                                300)
    return mesh


@pytest.fixture
def source_map(mesh, parameters):
    n_sources = 10
    x_lim = 10
    return RandomSourceMap(mesh,
                           n_sources,
                           parameters,
                           where=lambda x: x[0] > x_lim)


@pytest.fixture
def setup_function_space(mesh):
    V = fenics.FunctionSpace(mesh, "CG", 1)
    return V


@pytest.fixture
def phi_vessel_total(setup_function_space):
    phi = fenics.interpolate(fenics.Constant(1.), setup_function_space)
    return phi


@pytest.fixture
def phi_vessel_half(setup_function_space):
    phi = fenics.interpolate(fenics.Expression("x[0] < 150 ? 1. : -1", degree=1),
                             setup_function_space)
    return phi
