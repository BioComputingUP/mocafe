import pytest
from mocafe.angie.af_sourcing import SourceCell


@pytest.fixture
def source_cells(source_map):
    return source_map.get_global_source_cells()


def test_n_sources(source_map):
    n_sources = len(source_map.get_global_source_cells())
    assert n_sources == 10


def test_source_uniqueness(source_map):
    source_points = [source_cell.get_position() for source_cell in source_map.get_global_source_cells()]
    n_sources = len(source_points)
    source_points.sort(key=lambda x: x[0])
    points_are_equal = False
    for i in range(1, n_sources):
        point_prec = source_points[i-1]
        point = source_points[i]
        if ((point[0] - point_prec[0]) ** 2) + ((point[1] - point_prec[1]) ** 2) < 0.5:
            points_are_equal = True

    assert points_are_equal is False, "Each source should be unique"


def test_source_cells(source_cells):
    test_result = True
    for source_cell in source_cells:
        if type(source_cell) is not SourceCell:
            test_result = False

    assert test_result is True, "Each element should be of type SourceCell"


def test_remove_source_cell(source_map):
    first_source_cell = source_map.get_global_source_cells()[0]
    source_map.remove_global_source(first_source_cell)
    assert (first_source_cell in source_map.get_global_source_cells()) is False, "This cell should not be in list"
