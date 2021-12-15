import fenics
import mshr
import numpy as np

from mocafe.fenut.fenut import get_global_mesh_coordinates


def test_get_global_mesh_coordinates_unitsquare_mesh():
    # local mesh
    local_mesh = fenics.UnitSquareMesh(10, 10)
    # global mesh
    global_mesh = fenics.UnitSquareMesh(fenics.MPI.comm_self, 10, 10)
    # get global coordinates from local
    gc_from_local = get_global_mesh_coordinates(local_mesh)
    # get global coordinates
    gc = global_mesh.coordinates()
    assert len(gc_from_local) == len(gc)
    assert type(gc) == np.ndarray
    assert type(gc) == type(gc_from_local)
    gc_list = list(gc)
    gc_list.sort(key=lambda x: hash(tuple(x)))
    gc_from_local_list = list(gc_from_local)
    gc_from_local_list.sort(key=lambda x: hash(tuple(x)))
    for elem_gc, elem_gc_from_local in zip(gc_list, gc_from_local_list):
        assert np.allclose(elem_gc, elem_gc_from_local)


def test_get_global_mesh_coordinates_mshr_mesh():
    # local mash
    domain = mshr.Rectangle(fenics.Point(0, 0), fenics.Point(1, 1))
    mesh = mshr.generate_mesh(domain, 10)
    # get global coordinates
    lc = mesh.coordinates()
    gc_from_local = get_global_mesh_coordinates(mesh)
    assert len(lc) <= len(gc_from_local)
    # check if no element got lost
    lc_tuples = [tuple(elem) for elem in lc]
    gc_from_local_tuples = [tuple(elem) for elem in gc_from_local]
    for elem in lc_tuples:
        assert elem in gc_from_local_tuples

