import fenics
import mocafe.fenut.mansimdata as mansim

comm = fenics.MPI._comm_world
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
    # setup data_folder
    data_folder = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}")
    # test if exists
    assert data_folder.exists(), "The folder should exist"
    # test if the name is the one expected
    p0_tmpdir = get_p0_tmpdir(tmpdir)
    assert str(data_folder) == f"{p0_tmpdir}/{mansim.test_sim_name}/0000"


def test_setup_multiple_data_folder(tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}")
    data_folder2 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}")

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


def test_setup_multiple_data_folder_no_enumerate(tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}", auto_enumerate=False)
    data_folder2 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}", auto_enumerate=False)

    assert data_folder1.exists(), "Data folder 1 should exist"
    assert data_folder2.exists(), "Data folder 2 should exist"
    assert str(data_folder1) == str(data_folder2), "The two should be equal"
