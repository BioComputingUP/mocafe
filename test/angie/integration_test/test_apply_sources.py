import dolfinx
from dolfinx.mesh import create_rectangle
from dolfinx.fem import Constant, FunctionSpace, Function
from ufl import VectorElement
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from mocafe.angie.af_sourcing import SourceMap, SourcesManager
from mocafe.fenut.fenut import get_colliding_cells_for_points
import mocafe.fenut.fenut as fu
import numpy as np


def test_apply_sources(parameters):
    # define mesh
    n_x = n_y = 300
    mesh = create_rectangle(MPI.COMM_WORLD, points=[[0., 0.], [n_x, n_y]], n=[n_x, n_y])

    # define function space
    V = FunctionSpace(mesh, ("CG", 1))

    # define source map
    source_points = [np.array([num, num, 0]) for num in range(0, 310, 10)]
    sources_map = SourceMap(mesh, source_points, parameters)

    # define sources manager
    sources_manager = SourcesManager(sources_map, mesh, parameters)

    # define T
    T = Function(V)
    T.interpolate(lambda x: np.zeros(x[0].shape))
    assert isinstance(T, Function)

    # apply sources on T
    sources_manager.apply_sources(T)

    # get the T value at each source point
    mesh_bbt = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    T_vals = T.eval(*get_colliding_cells_for_points(source_points, mesh, mesh_bbt))

    # check if all points on this proc are close to T_s
    are_all_close_on_proc = np.allclose(T_vals, parameters.get_value("T_s"))

    # check if all point in the global mesh are close to T_s
    are_all_close = MPI.COMM_WORLD.allreduce(are_all_close_on_proc, MPI.LAND)

    # check
    assert are_all_close, "It should be true"


def test_apply_sources_mixed_function_space(parameters):
    # define mesh
    n_x = n_y = 300
    mesh = create_rectangle(MPI.COMM_WORLD, points=[[0., 0.], [n_x, n_y]], n=[n_x, n_y])

    # define function space
    mixed_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, mixed_elem)

    # define source map
    source_point = [np.array([150, 150, 0])]
    sources_map = SourceMap(mesh, source_point, parameters)

    # define sources manager
    sources_manager = SourcesManager(sources_map, mesh, parameters)

    # define T
    u = Function(V)
    u.interpolate(lambda x: np.vstack((np.zeros(x[0].shape), np.zeros(x[0].shape))))
    T, foo = u.split()

    # apply sources on T
    sources_manager.apply_sources(T)

    # get the T value at each source point
    mesh_bbt = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    T_vals = T.eval(*get_colliding_cells_for_points(source_point, mesh, mesh_bbt))

    # check if all points on this proc are close to T_s
    are_all_close_on_proc = np.allclose(T_vals, parameters.get_value("T_s"))

    # check if all point in the global mesh are close to T_s
    are_all_close = MPI.COMM_WORLD.allreduce(are_all_close_on_proc, MPI.LAND)

    # check
    assert are_all_close, "It should be true"
