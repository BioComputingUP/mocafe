import fenics
from mocafe.angie.af_sourcing import SourceMap, SourcesManager
import mocafe.fenut.fenut as fu
import numpy as np


def test_apply_sources(parameters):
    # define mesh
    n_x = n_y = 300
    mesh = fenics.RectangleMesh(fenics.Point(0., 0.), fenics.Point(n_x, n_y), n_x, n_y)

    # define function space
    element = fenics.FiniteElement("CG", fenics.triangle, 1)
    V = fenics.FunctionSpace(mesh, element)

    # define source map
    source_points = [np.array([num, num]) for num in range(0, 310, 10)]
    sources_map = SourceMap(mesh, source_points, parameters)

    # define sources manager
    sources_manager = SourcesManager(sources_map, mesh, parameters)

    # define T
    T = fenics.interpolate(fenics.Constant(0.0), V)

    # apply sources on T
    sources_manager.apply_sources(T)

    test_result = True
    for source_point in source_points:
        if fu.is_point_inside_mesh(mesh, source_point):
            test_result = np.isclose(T(source_point), parameters.get_value("T_s"))

    # confront
    assert test_result, "It should be true"


def test_apply_sources_mixed_function_space(parameters):
    # define mesh
    n_x = n_y = 300
    mesh = fenics.RectangleMesh(fenics.Point(0., 0.), fenics.Point(n_x, n_y), n_x, n_y)

    # define function space
    element = fenics.FiniteElement("CG", fenics.triangle, 1)
    mixed_elem = fenics.MixedElement([element, element])
    V = fenics.FunctionSpace(mesh, mixed_elem)

    # define source map
    source_point = [np.array([150, 150])]
    sources_map = SourceMap(mesh, source_point, parameters)

    # define sources manager
    sources_manager = SourcesManager(sources_map, mesh, parameters)

    # define T
    exp = fenics.Expression(("0.0", "0."), degree=1)
    u = fenics.interpolate(exp, V)
    T, foo = u.split()

    # apply sources on T
    sources_manager.apply_sources(T)

    # confront
    test_result = True
    if fu.is_point_inside_mesh(mesh, source_point[0]):
        test_result = np.isclose(T(source_point[0]), parameters.get_value("T_s"))

    assert test_result, "It should be 1"
