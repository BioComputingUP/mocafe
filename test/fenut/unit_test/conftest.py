import pytest
import pathlib
import fenics

comm = fenics.MPI.comm_world
rank = comm.Get_rank()


@pytest.fixture
def odf_sheet_test2():
    return pathlib.Path("test/fenut/unit_test/test_sheet2.ods")


@pytest.fixture
def odf_sheet_test():
    return pathlib.Path("test/fenut/unit_test/test_sheet.ods")


@pytest.fixture
def get_p0_tmpdir(tmpdir):
    """
    Utility method to get tmpdir from MPI process 0 and broadcast it to all processes
    :param tmpdir: current process tmpdir
    :return: the tmpdir for MPI process 0
    """
    if rank == 0:
        p0_tmpdir = tmpdir
    else:
        p0_tmpdir = None
    p0_tmpdir = comm.bcast(p0_tmpdir, 0)
    return p0_tmpdir
