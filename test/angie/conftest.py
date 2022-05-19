import dolfinx
import numpy as np
from mpi4py import MPI
import pytest
from mocafe.angie.af_sourcing import RandomSourceMap
import pathlib
from mocafe.fenut.parameters import from_ods_sheet, Parameters
from mocafe.angie import _setup_random_state
import os

"""Global fixtures"""
_setup_random_state(load=True, load_path=f"{os.path.dirname(__file__)}")


@pytest.fixture
def parameters() -> Parameters:
    # the saved mesh is 300x300
    return from_ods_sheet(pathlib.Path(f"{os.path.dirname(__file__)}/test_parameters.ods"), "SimParams")


@pytest.fixture
def mesh():
    mesh = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                                         points=[[0, 0], [300, 300]],
                                         n=[300, 300])
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
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    return V


@pytest.fixture
def phi_vessel_total(mesh):
    phi = dolfinx.fem.Constant(mesh, 1.)
    return phi


@pytest.fixture
def phi_vessel_half(setup_function_space):
    phi = dolfinx.fem.Function(setup_function_space)
    phi.interpolate(lambda x: np.where(x[0] < 150, 1., -1.))
    return phi
