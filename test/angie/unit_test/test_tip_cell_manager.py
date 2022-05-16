import fenics
import sys
import numpy as np
import pytest
from mocafe.angie.tipcells import TipCellManager, TipCell


# @pytest.fixture
# def file_list():
#     # setup root folder
#     data_folder = "runtime/test/tipcell"
#
#     # define files
#     file_names = ["c", "T", "grad_T"]
#     file_list = fu.setup_xdmf_files(file_names, data_folder)
#     return file_list


@pytest.fixture
def mesh():
    # define mesh
    nx = ny = 300
    mesh = fenics.RectangleMesh(fenics.Point(0., 0.),
                                fenics.Point(nx, ny),
                                nx, ny)
    return mesh


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


def test_activate_tip_cell(T0, phi0, gradT0, mesh, parameters):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # activate
    tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, 0)

    test_result = False
    # check if ok
    if len(tip_cell_manager.get_global_tip_cells_list()) == 1:
        if tip_cell_manager.get_global_tip_cells_list()[0].get_position()[0] < 30:
            print(f"p{fenics.MPI._comm_world.Get_rank()}: "
                  f"Activated tip cell in {tip_cell_manager.get_global_tip_cells_list()[0].get_position()}")
            test_result = True
    else:
        print(f"p{fenics.MPI._comm_world.Get_rank()}: "
              f" n activated tip cells = {len(tip_cell_manager.get_global_tip_cells_list())}")
        test_result = False

    assert test_result is True, "There should be just one tip cell activated"


def test_activate_3_tip_cells(parameters, T0, phi0, gradT0, mesh):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    for i in range(3):
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)

    tip_cell_list = tip_cell_manager.get_global_tip_cells_list()
    tip_cell_list_len_is_3 = len(tip_cell_list) == 3
    are_cells_distant = True
    for index, tip_cell in enumerate(tip_cell_list):
        print(f"p{fenics.MPI._comm_world.Get_rank()}: activated tip cells in pos:"
              f"    {tip_cell.get_position()}")
        other_indexes = [i for i in range(len(tip_cell_list))]
        other_indexes.remove(index)
        for i in other_indexes:
            distance = tip_cell.get_distance(tip_cell_list[i].get_position())
            if distance < 4 * parameters.get_value("R_c"):
                are_cells_distant = False

    assert tip_cell_list_len_is_3 and are_cells_distant, "There should be 3 cells distant to each other"


def test_revert_tip_cells(phi0, T0, gradT0, mesh, parameters):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # set test result
    ref_len = [1, 2, 0]
    actual_len = []
    for i in range(3):
        if i == 2:
            T0.assign(fenics.Constant(0.))
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)
        tip_cell_manager.revert_tip_cells(T0, gradT0)
        n_tip_cells = len(tip_cell_manager.get_global_tip_cells_list())
        print(f"p{fenics.MPI._comm_world.Get_rank()}: step {i}: "
              f"len = {len(tip_cell_manager.get_global_tip_cells_list())}")
        actual_len.append(n_tip_cells)
    assert np.allclose(ref_len, actual_len), \
        "There should be 1 tip cell at step 0, 2 tip cells at step 1, and 0 at step 2"


def test_delta_notch_reversion(T0, gradT0, mesh, parameters):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # create 2 tip cells close to each other
    tipcell1_pos = np.array([100., 100.])
    tipcell1 = TipCell(tipcell1_pos, 4., 0)
    tipcell2_pos = np.array([104., 100.])
    tipcell2 = TipCell(tipcell2_pos, 4., 0)

    # check if tip cells have been created correcty
    assert np.allclose(tipcell1.get_position(), tipcell1_pos)
    assert np.allclose(tipcell2.get_position(), tipcell2_pos)

    # add tip cells to tip cell manager
    tip_cell_manager._add_tip_cell(tipcell1)
    tip_cell_manager._add_tip_cell(tipcell2)

    # check if the tip cells have been added
    assert len(tip_cell_manager.get_global_tip_cells_list()) == 2

    # modify T0 and gradT0 to ensure the removal is not due to T0 or gradT0 level
    T0.assign(fenics.Constant(1.))
    gradT0.assign(fenics.Expression(("1.", "1"), degree=1))

    # remove tip cells
    tip_cell_manager.revert_tip_cells(T0, gradT0)
    # exit(0)
    assert len(tip_cell_manager.get_global_tip_cells_list()) == 1, \
        "One Tip Cell should be removed due to Delta-Notch signaling."


def test_delta_notch_3_cells(parameters, T0, gradT0, mesh):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # get min tip cell distance
    min_distance = parameters.get_value("min_tipcell_distance")

    # create 2 tip cells close to each other
    tipcell1_pos = np.array([100., 100.])
    tipcell1 = TipCell(tipcell1_pos, 4., 0)
    tipcell2_pos = tipcell1_pos + np.array([min_distance / 1.5, 0.])  # <-- only this should be removed
    tipcell2 = TipCell(tipcell2_pos, 4., 0)
    tipcell3_pos = tipcell2_pos + np.array([min_distance / 1.5, 0.])
    tipcell3 = TipCell(tipcell3_pos, 4., 0)

    # check if tip cells have been created correctly
    assert np.allclose(tipcell1.get_position(), tipcell1_pos)
    assert np.allclose(tipcell2.get_position(), tipcell2_pos)
    assert np.allclose(tipcell3.get_position(), tipcell3_pos)

    # add tip cells to tip cell manager
    tip_cell_manager._add_tip_cell(tipcell1)
    tip_cell_manager._add_tip_cell(tipcell2)
    tip_cell_manager._add_tip_cell(tipcell3)

    # check if the tip cells have been added
    assert len(tip_cell_manager.get_global_tip_cells_list()) == 3

    # modify T0 and gradT0 to ensure the removal is not due to T0 or gradT0 level
    T0.assign(fenics.Constant(1.))
    gradT0.assign(fenics.Expression(("1.", "1"), degree=1))

    # remove tip cells
    tip_cell_manager.revert_tip_cells(T0, gradT0)
    assert len(tip_cell_manager.get_global_tip_cells_list()) == 2, \
        "Only one Tip Cell should be removed due to Delta-Notch signaling."
    assert tipcell1 in tip_cell_manager.get_global_tip_cells_list(), \
        f"Tip Cell in position {tipcell1_pos} should be in the list"
    assert tipcell3 in tip_cell_manager.get_global_tip_cells_list(), \
        f"Tip Cell in position {tipcell3_pos} should be in the list"
