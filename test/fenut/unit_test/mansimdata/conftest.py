import pytest
import pathlib
import fenics
import shutil
import mocafe.fenut.mansimdata as mansim


@pytest.fixture
def test_folder():
    """
    Define the test folder
    :return: the name of the test folder ('test')
    """
    # def sim name for test
    sim_name = mansim.test_sim_name
    # define test folder
    test_folder = mansim.saved_sim_folder / pathlib.Path(f"{sim_name}")
    # remove in case the folder already exists
    if test_folder.exists():
        if fenics.MPI.comm_world.Get_rank() == 0:
            shutil.rmtree(test_folder)
    return sim_name
