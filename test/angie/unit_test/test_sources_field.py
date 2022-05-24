import numpy as np
import pytest
import dolfinx
import mocafe.fenut.fenut as fu
from mocafe.angie.af_sourcing import ConstantSourcesField


@pytest.fixture
def sources_field(source_map, parameters, setup_function_space):
    # init T
    T_exp = ConstantSourcesField(source_map, parameters)
    # interpolate
    T = dolfinx.fem.Function(setup_function_space)
    T.interpolate(T_exp.eval)
    T.x.scatter_forward()
    return T


@pytest.fixture()
def mesh_bbt(mesh):
    return dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)


def test_value_in_source_points(source_map, sources_field, mesh, mesh_bbt):
    test_result = True
    for source_cell in source_map.get_global_source_cells():
        # get source cell position (scp)
        scp = source_cell.get_position()
        # get collisions
        scp_on_proc, scp_cells = fu.get_colliding_cells_for_points([scp],
                                                                   mesh,
                                                                   mesh_bbt)
        # check if scp is on current proc
        is_scp_on_proc = (len(scp_on_proc) > 0)
        # check if empty
        if is_scp_on_proc:
            if sources_field.eval(scp_on_proc, scp_cells)[0] < 0.9:
                test_result = False
                break

    assert test_result is True, "At the center of each source the value should be 1."


def test_value_near_source_points(source_map, sources_field, mesh, parameters, mesh_bbt):
    test_result = True
    R_c = parameters.get_value("R_c") * 0.5
    random_angles = [2 * np.pi * 0.3,
                     2 * np.pi * 0.9,
                     2 * np.pi * 0.653,
                     2 * np.pi * 0.5535,
                     2 * np.pi * 0.1]
    random_scales = [0.1, 0.5, 0.6, 0.2, 0.8]
    random_vectors = [R_c * scale * np.array([np.cos(angle), np.sin(angle), 0.])
                      for scale, angle in zip(random_scales, random_angles)]
    for source_cell in source_map.get_global_source_cells():
        source_cell_position = source_cell.get_position()
        test_positions = [source_cell_position + vector for vector in random_vectors]
        for test_position in test_positions:
            # get collisions
            tp_on_proc, tp_cell = fu.get_colliding_cells_for_points([test_position],
                                                                    mesh,
                                                                    mesh_bbt)
            # check if point on proc
            is_tp_on_proc = (len(tp_on_proc) > 0)
            # check if empty
            if is_tp_on_proc:
                if sources_field.eval(tp_on_proc, tp_cell)[0] < 0.5:
                    test_result = False
                    break

    assert test_result is True, "Near each source cell the value should be 1."


def test_vector_values(sources_field):
    test_result = True
    local_vector = sources_field.x.array
    for elem in local_vector:
        if (~np.isnan(elem)) and (~np.isclose(elem, 1.)):
            test_result = False

    assert test_result is True, "Each vector element should be 0 or one."
