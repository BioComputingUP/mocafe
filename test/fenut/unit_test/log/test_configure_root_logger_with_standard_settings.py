import fenics
import logging
from pathlib import Path
from mocafe.fenut.log import \
    confgure_root_logger_with_standard_settings, \
    default_debug_log_folder, \
    default_debug_log_file, default_info_log_folder, default_info_log_file

comm = fenics.MPI.comm_world
rank = comm.Get_rank()

test_logger = logging.getLogger("test")


def test_log_files(tmpdir):
    confgure_root_logger_with_standard_settings(Path(str(tmpdir)))
    test_debug_msg = "debug test"
    test_logger.debug(test_debug_msg)
    test_info_msg = "info test"
    test_logger.info(test_info_msg)

    # check debug_log_file_exists
    current_debug_log_file = tmpdir / default_debug_log_folder / default_debug_log_file
    assert current_debug_log_file.exists(), "It should exist"

    # checl info log file exists
    current_info_log_file = tmpdir / default_info_log_folder / default_info_log_file
    assert current_info_log_file.exists(), "It should exist"


def test_log_files_cleaning(tmpdir):
    fake_old_log_file = Path(str(tmpdir)) / default_info_log_folder / Path("p1000_info_log_file.csv")
    if rank == 0:
        fake_old_log_file.parent.mkdir(parents=True, exist_ok=True)
        fake_old_log_file.touch()
    assert fake_old_log_file.exists(), "It should exist"

    confgure_root_logger_with_standard_settings(Path(str(tmpdir)))
    assert not fake_old_log_file.exists(), "It should have been removed"
