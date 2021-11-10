import numpy as np
import pytest
from mocafe.angie.tipcells import TipCell


@pytest.fixture
def tip_cell_in_0():
    tip_cell = TipCell(np.array([0., 0.]), 0.5, 4, 0)
    return tip_cell


def test_tip_cell_move(tip_cell_in_0):
    new_pos = np.array([0., 0.]) + np.array([1., 1.])
    tip_cell_in_0.move(new_pos)
    assert np.allclose(new_pos, tip_cell_in_0.get_position()), "The new position should have been set"


def test_is_point_inside(tip_cell_in_0):
    pos = np.array([1., 1.])
    assert tip_cell_in_0.is_point_inside(pos), "pos should be inside"
