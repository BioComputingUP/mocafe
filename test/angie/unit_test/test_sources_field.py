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
    return T


@pytest.fixture()
def mesh_bbt(mesh):
    return dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)


def test_value_in_source_points(source_map, sources_field, mesh, mesh_bbt):
    test_result = True
    for source_cell in source_map.get_global_source_cells():
        source_cell_position = source_cell.get_position()
        # get fem_cell for point
        candidate_fem_cells = dolfinx.geometry.compute_collisions(mesh_bbt, source_cell_position)
        colliding_fem_cells = dolfinx.geometry.compute_colliding_cells(mesh, candidate_fem_cells, source_cell_position)
        # check if empty
        if len(colliding_fem_cells) > 0:
            # if not, pick one cell for evaluation (no matter which)
            current_fem_cell = colliding_fem_cells[0]
            if sources_field.eval(source_cell_position, current_fem_cell) < 0.9:
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
            # get fem_cell for point
            candidate_fem_cells = dolfinx.geometry.compute_collisions(mesh_bbt, test_position)
            colliding_fem_cells = dolfinx.geometry.compute_colliding_cells(mesh, candidate_fem_cells,
                                                                           test_position)
            # check if empty
            if len(colliding_fem_cells) > 0:
                # if not, pick one cell for evaluation (no matter which)
                current_fem_cell = colliding_fem_cells[0]
                if sources_field.eval(test_position, current_fem_cell) < 0.5:
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
