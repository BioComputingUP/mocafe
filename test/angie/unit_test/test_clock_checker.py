import numpy as np
from mpi4py import MPI
from mocafe.angie.base_classes import ClockChecker

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def test_clock_checker_vessel_half(mesh, phi_vessel_half):
    # define clock checker
    clock_checker_radius = 4.
    clock_checker = ClockChecker(mesh, clock_checker_radius)

    # point on the border
    point1 = np.array([150, 150, 0.])
    # point distant less than radius from the border
    point2 = np.array([150. - (clock_checker_radius / 2), 150., 0.])
    # point distant more than a radius from the border
    point3 = np.array([150. - (clock_checker_radius * 2), 150., 0.])

    # ckeck the surroundings: is there any point1 lower than 0 around point1 (spoiler: yes)
    result = clock_checker.clock_check(point1,
                                       phi_vessel_half,
                                       condition=lambda fval: fval < 0.)
    # check result on all MPI procs
    result = comm.allreduce(result, MPI.LOR)
    assert result, "There should be a point around point1 lower than 0"

    # is there any point around point1 higher than 1.5 (spoiler: no)
    result = clock_checker.clock_check(point1,
                                       phi_vessel_half,
                                       condition=lambda fval: fval > 1.5)
    result = comm.allreduce(result, MPI.LOR)
    assert result is False, "There should be no point higher than 1.5"

    # is there any point around point2 lower than 0? (spoiler: yes)
    result = clock_checker.clock_check(point2,
                                       phi_vessel_half,
                                       condition=lambda fval: fval < 0.)
    result = comm.allreduce(result, MPI.LOR)
    assert result, "There should be a point around point2 lower than 0"

    # is there any point around point3 lower than 0? (spoiler: no)
    result = clock_checker.clock_check(point3,
                                       phi_vessel_half,
                                       condition=lambda fval: fval < 0.)
    result = comm.allreduce(result, MPI.LOR)
    assert result is False, "There should no point around point3 lower than 0"
