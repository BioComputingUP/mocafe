import numpy as np
import pytest
import fenics
from src_traang.af_sourcing import SourcesField


@pytest.fixture
def sources_field(source_map, parameters, setup_function_space):
    # init T
    T_exp = SourcesField(source_map, parameters)
    # interpolate
    T = fenics.interpolate(T_exp, setup_function_space)
    return T


def test_value_in_source_points(source_map, sources_field, mesh_wrapper):
    test_result = True
    for source_cell in source_map.get_global_source_cells():
        source_cell_position = source_cell.get_position()
        if mesh_wrapper.is_inside_local_mesh(source_cell_position):
            if sources_field(source_cell.get_position()) < 0.9:
                test_result = False
                break
    assert test_result is True, "At the center of each source the value should be 1."


def test_value_near_source_points(source_map, sources_field, mesh_wrapper, parameters):
    test_result = True
    R_c = parameters.get_value("R_c") * 0.5
    random_angles = [2 * np.pi * 0.3,
                     2 * np.pi * 0.9,
                     2 * np.pi * 0.653,
                     2 * np.pi * 0.5535,
                     2 * np.pi * 0.1]
    random_scales = [0.1, 0.5, 0.6, 0.2, 0.8]
    random_vectors = [R_c * scale * np.array([np.cos(angle), np.sin(angle)])
                      for scale, angle in zip(random_scales, random_angles)]
    for source_cell in source_map.get_global_source_cells():
        source_cell_position = source_cell.get_position()
        test_positions = [source_cell_position + vector for vector in random_vectors]
        for test_position in test_positions:
            if mesh_wrapper.is_inside_local_mesh(test_position):
                if sources_field(test_position) < 0.5:
                    test_result = False
                    break
    assert test_result is True, "Near each source cell the value should be 1."


def test_vector_values(sources_field):
    test_result = True
    local_vector = sources_field.vector().get_local()
    for elem in local_vector:
        if (not np.isclose(elem, 0.)) and (not np.isclose(elem, 1.)):
            test_result = False

    assert test_result is True, "Each vector element should be 0 or one."
