import fenics
import pytest
from fenut.fenut import RectangleMeshWrapper
from src_traang.af_sourcing import SourceMap
import pathlib
from fenut.parameters import Parameters
from src_traang import load_random_state

"""Global fixtures"""
load_random_state("test/test_randomstate.pickle")


@pytest.fixture
def parameters() -> Parameters:
    return Parameters(pathlib.Path("input_files/parameters.ods"), "SimParams")


@pytest.fixture
def mesh_wrapper():
    mesh_wrapper = RectangleMeshWrapper(fenics.Point(0., 0.),
                                        fenics.Point(300., 300.),
                                        100, 100)
    return mesh_wrapper


@pytest.fixture
def source_map(mesh_wrapper, parameters):
    n_sources = 10
    x_lim = 10
    return SourceMap(n_sources,
                     x_lim,
                     mesh_wrapper,
                     0,
                     parameters)


@pytest.fixture
def setup_function_space(mesh_wrapper):
    mesh = mesh_wrapper.get_local_mesh()
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
