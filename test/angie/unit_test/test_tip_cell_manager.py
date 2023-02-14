import dolfinx
import ufl
import numpy as np
import pytest
from mocafe.angie.tipcells import TipCellManager, TipCell, load_tip_cells_from_json
from mocafe.math import project
from mpi4py import MPI
from pathlib import Path


@pytest.fixture
def T0(parameters, mesh):
    T_c = parameters.get_value("T_c")
    G_M = parameters.get_value("G_M")
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    c = T_c + 0.01  # a bit more than the threshold
    m = 4 * (G_M + 0.01)  # more than the threshold
    T0 = dolfinx.fem.Function(V)
    T0.interpolate(lambda x: c + (m * x[0]))
    T0.x.scatter_forward()
    return T0


@pytest.fixture
def phi0(mesh):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    phi_min = -1
    phi_max = 1
    phi0 = dolfinx.fem.Function(V)
    phi0.interpolate(lambda x: np.where(x[0] < 30, phi_max, phi_min))
    phi0.x.scatter_forward()
    return phi0


@pytest.fixture
def gradT0(mesh, T0):
    V_vec = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
    gradT0 = dolfinx.fem.Function(V_vec)
    project(ufl.grad(T0), gradT0, [])
    return gradT0


def test_activate_tip_cell(T0, phi0, gradT0, mesh, parameters, tmpdir):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # activate
    tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, 0)

    # get global tip cell list
    g_tc_list = tip_cell_manager.get_global_tip_cells_list()

    # test length
    assert len(g_tc_list) == 1, "There should be just one tip cell"

    # test if tc is in the right position
    activated_tc = g_tc_list[0]
    assert activated_tc.get_position()[0] < 30, f"Tip cell should be where phi0 is high (x[0] < 30). " \
                                                f"Tip cell found in position {activated_tc.get_position()}"

    # test if a new tip_cell_manager can load the global tip cell list
    del tip_cell_manager
    tip_cell_manager2 = TipCellManager(mesh, parameters, initial_tcs=g_tc_list)
    g_tc_list2 = tip_cell_manager2.get_global_tip_cells_list()
    assert g_tc_list2 == g_tc_list, f"The two gloabl tip cells list should be equal."

    # check if the new tip cell manager can activate another tip cell
    tip_cell_manager2.activate_tip_cell(phi0, T0, gradT0, 0)
    g_tc_list2 = tip_cell_manager2.get_global_tip_cells_list()
    assert len(g_tc_list2) == 2, f"There should be two tip cells now. Found {len(g_tc_list2)} instead."
    assert all([tc.get_position()[0] < 30 for tc in g_tc_list2]), f"All Tip Cells should be where phi0 is high " \
                                                                  f"(x[0] < 30). Tip Cells positions are: " \
                                                                  f"{[tc.get_position() for tc in g_tc_list2]}"

    # test if I can save tip cells
    tc_file_r0 = f"{tmpdir}/tipcells.json"
    tc_file_r0 = MPI.COMM_WORLD.bcast(tc_file_r0, root=0)
    tip_cell_manager2.save_tip_cells(tc_file_r0)
    assert Path(tc_file_r0).exists(), f"The tc file {tc_file_r0} should have been created. "

    # test if I can load tip cells
    tc_list = load_tip_cells_from_json(tc_file_r0)
    assert tc_list == g_tc_list2

def test_activate_3_tip_cells(parameters, T0, phi0, gradT0, mesh):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    for i in range(3):
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)

    tip_cell_list = tip_cell_manager.get_global_tip_cells_list()
    tip_cell_list_len_is_3 = len(tip_cell_list) == 3
    are_cells_distant = True
    for index, tip_cell in enumerate(tip_cell_list):
        print(f"p{MPI.COMM_WORLD.Get_rank()}: activated tip cells in pos:"
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
    for i in range(3):
        # change T0 at step 2
        if i == 2:
            T0.interpolate(lambda x: np.zeros(x.shape[1]))
            T0.x.scatter_forward()
        # activate
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)
        # revert
        tip_cell_manager.revert_tip_cells(T0, gradT0)
        # check if the number of tip cells is correct
        assert len(tip_cell_manager.get_global_tip_cells_list()) == ref_len[i], \
            f"The global tip cell list should be {ref_len[i]}"


def test_delta_notch_revert(T0, gradT0, mesh, parameters):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # create 2 tip cells close to each other
    tipcell1_pos = np.array([100., 100., 0.])
    tipcell1 = TipCell(tipcell1_pos, 4., 0)
    tipcell2_pos = np.array([104., 100., 0])
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
    T0.interpolate(lambda x: np.ones(x.shape[1]))
    T0.x.scatter_forward()
    gradT0.interpolate(lambda x: [np.ones(x.shape[1]), np.ones(x.shape[1])])
    gradT0.x.scatter_forward()

    # remove tip cells
    tip_cell_manager.revert_tip_cells(T0, gradT0)
    # exit(0)
    assert len(tip_cell_manager.get_global_tip_cells_list()) == 1, \
        "One Tip Cell should be removed due to Delta-Notch signaling."


def test_delta_notch_3_cells_revert(parameters, T0, gradT0, mesh):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # get min tip cell distance
    min_distance = parameters.get_value("min_tipcell_distance")

    # create 2 tip cells close to each other
    tipcell1_pos = np.array([100., 100., 0.])
    tipcell1 = TipCell(tipcell1_pos, 4., 0)
    tipcell2_pos = tipcell1_pos + np.array([min_distance / 1.5, 0., 0.])  # <-- only this should be removed
    tipcell2 = TipCell(tipcell2_pos, 4., 0)
    tipcell3_pos = tipcell2_pos + np.array([min_distance / 1.5, 0., 0.])
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
    T0.interpolate(lambda x: np.ones(x.shape[1]))
    T0.x.scatter_forward()
    gradT0.interpolate(lambda x: [np.ones(x.shape[1]), np.ones(x.shape[1])])
    gradT0.x.scatter_forward()

    # remove tip cells
    tip_cell_manager.revert_tip_cells(T0, gradT0)
    assert len(tip_cell_manager.get_global_tip_cells_list()) == 2, \
        "Only one Tip Cell should be removed due to Delta-Notch signaling."
    assert tipcell1 in tip_cell_manager.get_global_tip_cells_list(), \
        f"Tip Cell in position {tipcell1_pos} should be in the list"
    assert tipcell3 in tip_cell_manager.get_global_tip_cells_list(), \
        f"Tip Cell in position {tipcell3_pos} should be in the list"


def test_move_tip_cells(parameters, phi0, T0, gradT0, mesh):
    # create tip cell manager
    tip_cells_manager = TipCellManager(mesh, parameters)
    # activate tip cell
    tip_cells_manager.activate_tip_cell(phi0, T0, gradT0, 0)
    # move
    tip_cells_manager.move_tip_cells(phi0, T0, gradT0)
    # get last tip cells field
    latest_tcf = tip_cells_manager.get_latest_tip_cell_function()

    # check if any is nan
    is_any_local_value_nan = np.any(np.isnan(latest_tcf.x.array))
    # check if any global nan
    is_any_local_value_nan = MPI.COMM_WORLD.gather(is_any_local_value_nan, 0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        is_any_global_value_nan = np.all(is_any_local_value_nan)
    else:
        is_any_global_value_nan = None
    is_any_global_value_nan = MPI.COMM_WORLD.bcast(is_any_global_value_nan, 0)
    assert not is_any_global_value_nan, "All value should be not nan"

    # check if there are positive points
    is_any_local_value_positive = np.any(latest_tcf.x.array > 0)
    # check if any global is positive
    is_any_local_value_positive = MPI.COMM_WORLD.gather(is_any_local_value_positive, 0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        is_any_global_value_positive = np.any(is_any_local_value_positive)
    else:
        is_any_global_value_positive = None
    is_any_global_value_positive = MPI.COMM_WORLD.bcast(is_any_global_value_positive, 0)
    assert is_any_global_value_positive, "At least some values should be positive"
