import pytest
from mocafe.angie.af_sourcing import SourcesManager


@pytest.fixture
def sources_manager(source_map, mesh_wrapper, parameters):
    return SourcesManager(source_map, mesh_wrapper, parameters)


def test_remove_source_near_vessels_total(sources_manager, phi_vessel_total):
    sources_manager.remove_sources_near_vessels(phi_vessel_total)
    source_map = sources_manager.source_map
    assert len(source_map.get_global_source_cells()) == 0, "There should be no cell in source map"


def test_remove_source_near_vessels_total_local(sources_manager, phi_vessel_total):
    sources_manager.remove_sources_near_vessels(phi_vessel_total)
    source_map = sources_manager.source_map
    assert len(source_map.get_local_source_cells()) == 0, "There should be no cell in source map"


def test_remove_source_near_vessels_half(sources_manager, phi_vessel_half):
    sources_manager.remove_sources_near_vessels(phi_vessel_half)
    source_map = sources_manager.source_map
    test_result = True
    for source_cell in source_map.get_global_source_cells():
        source_cell_position = source_cell.get_position()
        print(f"Source cell position  = {source_cell_position}")
        if source_cell_position[0] < 150 + 17:
            test_result = False
            break
    assert test_result is True, "There should be no point near vessel"
