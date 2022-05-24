import dolfinx
import numpy as np
from mpi4py import MPI
from mocafe.math import estimate_field01_integral

_comm = MPI.COMM_WORLD


def test_estimate_field01_integral():
    # create a unit square mesh
    mesh = dolfinx.mesh.create_unit_square(_comm, 10, 10)
    # define function space
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    # define a constant function on it
    constant_function = dolfinx.fem.Function(V)
    constant_function.interpolate(lambda x: np.ones(x.shape[1]))
    constant_function.x.scatter_forward()
    # compute the integral
    integral = estimate_field01_integral(constant_function)
    # should be one
    assert np.isclose(integral, 1.), "It should be 1."
