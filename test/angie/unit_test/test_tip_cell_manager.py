import fenics
import numpy as np
import pytest
import itertools
from mocafe.angie.tipcells import TipCellManager, TipCell


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


def test_activate_3_tip_cells(parameters, T0, phi0, gradT0, mesh):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # create 3 tip cells
    for i in range(3):
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)

    # check if len is correct
    tip_cell_list = tip_cell_manager.get_global_tip_cells_list()
    assert len(tip_cell_list) == 3, f"There should be 3 Tip Cells. Found {len(tip_cell_list)} instead"

    # check if tcs are distant
    min_tc_distance = parameters.get_value("min_tipcell_distance")
    for tc, other_tc in itertools.combinations(tip_cell_list, 2):
        # compute distance
        tc_distance = tc.get_distance(other_tc.get_position())
        # check if is correct
        assert tc_distance >= min_tc_distance, f"Min distance between tip cells should be" \
                                               f"{min_tc_distance}. Is {tc_distance} instead."

    # repeat the tests for another tip cell manager, which loads the previous tc_list
    del tip_cell_manager
    tip_cell_manager2 = TipCellManager(mesh, parameters, initial_tcs=tip_cell_list)
    for i in range(3):
        tip_cell_manager2.activate_tip_cell(phi0, T0, gradT0, i)
    tip_cell_list = tip_cell_manager2.get_global_tip_cells_list()
    assert len(tip_cell_list) == 6, f"There should be 6 Tip Cells. Found {len(tip_cell_list)} instead"
    for tc, other_tc in itertools.combinations(tip_cell_list, 2):
        # compute distance
        tc_distance = tc.get_distance(other_tc.get_position())
        # check if is correct
        assert tc_distance >= min_tc_distance, f"Min distance between tip cells should be" \
                                               f"{min_tc_distance}. Is {tc_distance} instead."


def test_revert_tip_cells(phi0, T0, gradT0, mesh, parameters):
    # create tip cell manager
    tip_cell_manager = TipCellManager(mesh, parameters)

    # set test result
    ref_len = [1, 2, 0]
    # init list for test results
    actual_len = []
    for i in range(3):
        if i == 2:
            T0.assign(fenics.Constant(0.))
        tip_cell_manager.activate_tip_cell(phi0, T0, gradT0, i)
        tip_cell_manager.revert_tip_cells(T0, gradT0)
        n_tip_cells = len(tip_cell_manager.get_global_tip_cells_list())
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


def test_different_initial_tc_lists_on_different_processes(mesh, parameters):
    # create different tip cell list in different processes
    init_tc_list = [TipCell(np.array([4, 4]), 4, 100),
                    TipCell(np.array([10, 10]), 4, 101),
                    TipCell(np.array([20, 20]), 4, fenics.MPI.comm_world.Get_rank())]
    # check if error raises
    with pytest.raises(RuntimeError):
        TipCellManager(mesh, parameters, initial_tcs=init_tc_list)
