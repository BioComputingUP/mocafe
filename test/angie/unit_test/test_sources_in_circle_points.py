import numpy as np
from mocafe.angie.af_sourcing import sources_in_circle_points


def test_one():
    source_points = sources_in_circle_points(np.array([10, 0]), 2, 4)
    assert len(source_points) == 1
