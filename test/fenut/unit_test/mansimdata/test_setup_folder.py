import pathlib
import fenics
import mocafe.fenut.mansimdata as mansim

comm = fenics.MPI.comm_world
rank = comm.Get_rank()


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


def test_setup_data_folder(tmpdir):
    data_folder_path = str(tmpdir/pathlib.Path("test"))
    data_folder = mansim.setup_data_folder(folder_path=tmpdir/pathlib.Path("test"))
    assert data_folder.exists()
    assert str(data_folder) == str(get_p0_tmpdir(tmpdir)/pathlib.Path("test/0000"))


def test_setup_multiple_data_folder_auto_enumerate_true(tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name), auto_enumerate=True)
    data_folder2 = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name), auto_enumerate=True)

    assert data_folder1.exists(), "Data folder 1 should exist"
    assert str(data_folder1) == f"{get_p0_tmpdir(tmpdir)}" \
                                f"/{mansim.test_sim_name}/0000", \
        f"The data folder 1 should be {get_p0_tmpdir(tmpdir)}" \
                                f"/{mansim.test_sim_name}/0000"
    assert data_folder2.exists(), "Data folder 2 should exist"
    assert str(data_folder2) == f"{get_p0_tmpdir(tmpdir)}" \
                                f"/{mansim.test_sim_name}/0001", \
        f"The data folder 2 should be {get_p0_tmpdir(tmpdir)}" \
                                f"/{mansim.test_sim_name}/0001"


def test_setup_multiple_data_folder_auto_enumerate_false(tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name), auto_enumerate=False)
    data_folder2 = mansim.setup_data_folder(str(tmpdir/mansim.test_sim_name), auto_enumerate=False)

    assert data_folder1.exists(), "Data folder 1 should exist"
    assert data_folder2.exists(), "Data folder 2 should exist"
    assert str(data_folder1) == str(data_folder2), "The two should be equal"
