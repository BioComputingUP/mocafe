import pathlib
import shutil
import fenics
import mocafe.fenut.mansimdata as mansim


def clean_folder(test_folder):
    """
    Clean the test folder
    :param test_folder: name of the test folder
    :return:
    """
    folder = mansim.saved_sim_folder / pathlib.Path(f"{test_folder}")
    if fenics.MPI.comm_world.Get_rank() == 0:
        shutil.rmtree(folder)


def test_setup_data_folder(tmpdir):
    # setup data_folder
    data_folder = mansim.setup_data_folder(mansim.test_sim_name, other_location=tmpdir)
    assert data_folder.exists() and str(data_folder) == f"{tmpdir}/{mansim.test_sim_name}/0000", \
        "The folder should exist and its name should be saved_sim/test/0000"


def test_setup_multiple_data_folder(tmpdir):
    # define multiple data folders
    data_folder1 = mansim.setup_data_folder(mansim.test_sim_name, other_location=tmpdir)
    data_folder2 = mansim.setup_data_folder(mansim.test_sim_name, other_location=tmpdir)

    assert data_folder1.exists() \
           and str(data_folder1) == f"{tmpdir}/{mansim.test_sim_name}/0000" \
           and data_folder2.exists() \
           and str(data_folder2) == f"{tmpdir}/{mansim.test_sim_name}/0001", \
           "Both folder should exist and their name should be saved_sim/test/0000 and saved_sim/test/0001"


def test_setup_runtime(tmpdir):
    data_folder = mansim.setup_data_folder(other_location=tmpdir)
    assert data_folder.exists() \
           and str(data_folder) == f"{mansim.runtime_folder}", \
           "The folder should be './runtime/'"
