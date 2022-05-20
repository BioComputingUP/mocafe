import pytest
import numpy as np
from mocafe.angie.af_sourcing import SourcesManager, SourceMap


@pytest.fixture
def non_random_source_map(mesh, parameters):
    # define source points
    source_points = [
        np.array([100, 100, 0]),
        np.array([100, 200, 0]),
        np.array([157, 200, 0]),
        np.array([167, 200, 0]),
        np.array([250, 250, 0]),
        np.array([250, 270, 0])
    ]
    return SourceMap(mesh, source_points, parameters)


@pytest.fixture
def sources_manager(non_random_source_map, mesh, parameters):
    return SourcesManager(non_random_source_map, mesh, parameters)


def test_remove_source_near_vessels_total(sources_manager, phi_vessel_total):
    sources_manager.remove_sources_near_vessels(phi_vessel_total)
    source_map = sources_manager.source_map
    assert len(source_map.get_global_source_cells()) == 0, "There should be no cell in source map"


def test_remove_source_near_vessels_total_local(sources_manager, phi_vessel_total):
    sources_manager.remove_sources_near_vessels(phi_vessel_total)
    source_map = sources_manager.source_map
    assert len(source_map.get_local_source_cells()) == 0, "There should be no cell in source map"


def test_remove_source_near_vessels_half(sources_manager, phi_vessel_half):
    # test if initial len is 6
    assert len(sources_manager.source_map.get_global_source_cells()) == 6, "Initially there should be 6 cells"

    # remove source cells
    sources_manager.remove_sources_near_vessels(phi_vessel_half)
    source_map = sources_manager.source_map

    # only cells in position [250, 250, 0] and [250, 270, 0] should remain
    assert len(source_map.get_global_source_cells()) == 2, "Only two cells are far from the vessel"
