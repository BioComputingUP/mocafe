import numpy as np
from src_traang.af_sourcing import SourceCell


def test_distance_source_cells():
    position1 = np.array([1, 1])
    position2 = np.array([2, 1])
    source_cell1 = SourceCell(position1, 0.5, 0)
    source_cell2 = SourceCell(position2, 0.5, 0)
    distance = source_cell1.get_distance(source_cell2.get_position())
    assert np.isclose(distance, 1), "Source cells should be distant 1 unit."


def test_distance_source_cells2():
    position1 = np.array([1, 1])
    position2 = np.array([0, 1])
    source_cell1 = SourceCell(position1, 0.5, 0)
    source_cell2 = SourceCell(position2, 0.5, 0)
    distance = source_cell1.get_distance(source_cell2.get_position())
    assert np.isclose(distance, 1), "Source cells should be distant 1 unit."
