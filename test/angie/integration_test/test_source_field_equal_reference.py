import fenics
import numpy as np
from src_traang.af_sourcing import SourceMap, SourcesField
import fenut.fenut as fu


def test_source_field_equals_reference(parameters):
    # setup data folder
    data_folder = "test/integration_test/input_test_files/test_source_field_equal_reference"

    # define files
    file_names = ["T_checkpoint", "T"]
    file_T, file_T_checkpoint = fu.setup_xdmf_files(file_names, data_folder)

    # define mesh
    n_x = n_y = 300
    mesh_wrapper = fu.RectangleMeshWrapper(fenics.Point(0., 0.), fenics.Point(n_x, n_y), n_x, n_y)
    mesh = mesh_wrapper.get_local_mesh()

    # define function space
    element = fenics.FiniteElement("CG", fenics.triangle, 1)
    V = fenics.FunctionSpace(mesh, element)

    # define source map
    source_points = [np.array([num, num]) for num in range(0, 310, 10)]
    sources_map = SourceMap(0, 10, mesh_wrapper, 0, parameters, source_points=source_points)

    # define source field
    T = fenics.interpolate(SourcesField(sources_map, parameters), V)

    # read
    T_ref = fenics.Function(V)
    file_T_checkpoint.read_checkpoint(T_ref, "T", 0)

    # confront
    assert np.allclose(T.vector().get_local(), T_ref.vector().get_local()), "They should be the same"
