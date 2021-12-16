import fenics
import pytest
from mocafe.fenut.fenut import MeshWrapper
from mocafe.angie.af_sourcing import SourceMap
import pathlib
from mocafe.fenut.parameters import from_ods_sheet, Parameters
from mocafe.angie import setup_random_state
import os

"""Global fixtures"""
setup_random_state(load=True, load_path=f"{os.path.dirname(__file__)}")


@pytest.fixture
def parameters() -> Parameters:
    return from_ods_sheet(pathlib.Path(f"{os.path.dirname(__file__)}/test_parameters.ods"), "SimParams")


@pytest.fixture
def mesh_wrapper():
    mesh = fenics.RectangleMesh(fenics.Point(0., 0.),
                                fenics.Point(300., 300.),
                                100,
                                100)
    mesh_wrapper = MeshWrapper(mesh)
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
