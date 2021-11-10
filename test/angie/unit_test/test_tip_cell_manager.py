import fenics
import numpy as np
import pytest
import mocafe.fenut.fenut as fu
from mocafe.angie.tipcells import TipCellManager


# @pytest.fixture
# def file_list():
#     # setup root folder
#     data_folder = "runtime/test/tipcell"
#
#     # define files
#     file_names = ["phi", "T", "grad_T"]
#     file_list = fu.setup_xdmf_files(file_names, data_folder)
#     return file_list


@pytest.fixture
def mesh_wrapper():
    # define mesh
    nx = ny = 300
    mesh_wrapper = fu.RectangleMeshWrapper(fenics.Point(0., 0.),
                                           fenics.Point(nx, ny),
                                           nx, ny)
    return mesh_wrapper


@pytest.fixture
def mesh(mesh_wrapper):
    return mesh_wrapper.get_local_mesh()


@pytest.fixture
def T0(parameters, mesh):
    T_c = parameters.get_value("T_c")
    G_M = parameters.get_value("G_M")
    V = fenics.FunctionSpace(mesh, "CG", 1)
    c = T_c + 0.01
    m = 4 * (G_M + 0.01)
    T_exp = fenics.Expression("c + (m * x[0])",
                              degree=1,
                              c=c,
                              m=m)
    T0 = fenics.interpolate(T_exp, V)
    return T0


@pytest.fixture
def phi0(mesh):
    V = fenics.FunctionSpace(mesh, "CG", 1)
    phi_min = -1
    phi_max = 1
    phi_exp = fenics.Expression("x[0] < 30 ? phi_max : phi_min",
                                degree=1,
                                phi_max=phi_max,
                                phi_min=phi_min)
    phi0 = fenics.interpolate(phi_exp, V)
    return phi0


@pytest.fixture
def gradT0(mesh, T0):
    V_vec = fenics.VectorFunctionSpace(mesh, "CG", 1)
    gradT0 = fenics.project(fenics.grad(T0), V_vec, mesh=mesh)
    return gradT0


@pytest.fixture
def tip_cell_manager(mesh_wrapper, parameters):
    return TipCellManager(mesh_wrapper, parameters)


def test_activate_tip_cell(T0, phi0, gradT0, tip_cell_manager):
    # activate
    tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, 0)

    test_result = False
    # check if ok
    if len(tip_cell_manager.get_global_tip_cells_list()) == 1:
        if tip_cell_manager.get_global_tip_cells_list()[0].get_position()[0] < 30:
            print(f"p{fenics.MPI.comm_world.Get_rank()}: "
                  f"Activated tip cell in {tip_cell_manager.get_global_tip_cells_list()[0].get_position()}")
            test_result = True
    else:
        print(f"p{fenics.MPI.comm_world.Get_rank()}: "
              f" n activated tip cells = {len(tip_cell_manager.get_global_tip_cells_list())}")
        test_result = False

    assert test_result is True, "There should be just one tip cell activated"


def test_activate_3_tip_cells(parameters, T0, phi0, gradT0, tip_cell_manager):
    for i in range(3):
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)

    tip_cell_list = tip_cell_manager.get_global_tip_cells_list()
    tip_cell_list_len_is_3 = len(tip_cell_list) == 3
    are_cells_distant = True
    for index, tip_cell in enumerate(tip_cell_list):
        print(f"p{fenics.MPI.comm_world.Get_rank()}: activated tip cells in pos:"
              f"    {tip_cell.get_position()}")
        other_indexes = [i for i in range(len(tip_cell_list))]
        other_indexes.remove(index)
        for i in other_indexes:
            distance = tip_cell.get_distance(tip_cell_list[i].get_position())
            if distance < 4 * parameters.get_value("R_c"):
                are_cells_distant = False

    assert tip_cell_list_len_is_3 and are_cells_distant, "There should be 3 cells distant to each other"


def test_revert_tip_cells(phi0, T0, gradT0, tip_cell_manager):
    # set test result
    ref_len = [1, 2, 0]
    actual_len = []
    for i in range(3):
        if i == 2:
            T0.assign(fenics.Constant(0.))
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)
        tip_cell_manager.revert_tip_cells(T0, gradT0)
        n_tip_cells = len(tip_cell_manager.get_global_tip_cells_list())
        print(f"p{fenics.MPI.comm_world.Get_rank()}: step {i}: "
              f"len = {len(tip_cell_manager.get_global_tip_cells_list())}")
        actual_len.append(n_tip_cells)
    assert np.allclose(ref_len, actual_len), \
        "There should be 1 tip cell at step 0, 2 tip cells at step 1, and 0 at step 2"
