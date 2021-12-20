import fenics
import pytest


def get_n_procs_for_point(point, mesh_wrapper):
    root = 0
    is_in_local = mesh_wrapper.is_inside_local_mesh(point)
    is_in_local_array = fenics.MPI.comm_world.gather(is_in_local, root)
    if fenics.MPI.comm_world.Get_rank() == root:
        n_procs = sum(is_in_local_array)
    else:
        n_procs = None
    n_procs = fenics.MPI.comm_world.bcast(n_procs, root)
    return n_procs


def test_mesh_wrapper_partitioning(mesh_wrapper):
    if fenics.MPI.comm_world.Get_size() == 1:
        assert mesh_wrapper.n_local_mesh_vertices() == mesh_wrapper.n_global_mesh_vertices(), "With one process" \
                                                                                              "should be equal"
    else:
        assert mesh_wrapper.n_local_mesh_vertices() < mesh_wrapper.n_global_mesh_vertices(), "With two or more" \
                                                                                             "n_local_points should" \
                                                                                             "be less"


def test_point_in_local_mesh(mesh_wrapper):
    point = fenics.Point(1, 1)
    if fenics.MPI.comm_world.Get_size() == 1:
        assert mesh_wrapper.is_inside_local_mesh(point) is True
    else:
        n_procs_for_point = get_n_procs_for_point(point, mesh_wrapper)
        test_result = n_procs_for_point == 1 or n_procs_for_point == 2
        assert test_result, "A point can be in one or 2 meshes maximum"


def test_point_in_global_mesh(mesh_wrapper):
    point = fenics.Point(1, 1)
    assert mesh_wrapper.is_inside_global_mesh(point) is True, "For every process the point should be inside"
