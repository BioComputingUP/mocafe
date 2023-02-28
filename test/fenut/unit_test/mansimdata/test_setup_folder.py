import fenics
import mocafe.fenut.mansimdata as mansim

comm = fenics.MPI.comm_world
rank = comm.Get_rank()


def test_setup_data_folder(tmpdir, get_p0_tmpdir):
    # setup data_folder
    data_folder = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}")
    # test if exists
    assert data_folder.exists(), "The folder should exist"
    assert str(data_folder) == f"{get_p0_tmpdir}/{mansim.test_sim_name}/0000"


def test_setup_multiple_data_folder(tmpdir, get_p0_tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}")
    data_folder2 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}")

    assert data_folder1.exists(), "Data folder 1 should exist"
    assert str(data_folder1) == f"{get_p0_tmpdir}" \
                                f"/{mansim.test_sim_name}/0000", \
        f"The data folder 1 should be {get_p0_tmpdir}" \
                                f"/{mansim.test_sim_name}/0000"
    assert data_folder2.exists(), "Data folder 2 should exist"
    assert str(data_folder2) == f"{get_p0_tmpdir}" \
                                f"/{mansim.test_sim_name}/0001", \
        f"The data folder 2 should be {get_p0_tmpdir}" \
                                f"/{mansim.test_sim_name}/0001"


def test_setup_multiple_data_folder_no_enumerate(tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}", auto_enumerate=None)
    data_folder2 = mansim.setup_data_folder(folder_path=f"{tmpdir}/{mansim.test_sim_name}", auto_enumerate=None)

    assert data_folder1.exists(), "Data folder 1 should exist"
    assert data_folder2.exists(), "Data folder 2 should exist"
    assert str(data_folder1) == str(data_folder2), "The two should be equal"
