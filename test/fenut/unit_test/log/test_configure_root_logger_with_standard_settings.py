from mpi4py import MPI
import logging
from pathlib import Path
from mocafe.fenut.log import \
    confgure_root_logger_with_standard_settings, \
    default_debug_log_folder, \
    default_debug_log_file, default_info_log_folder, default_info_log_file

comm = MPI.COMM_WORLD
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
    if rank == 0:
        # create fake old log file for _rank 0 (tmpdir changes across MPI processes)
        fake_old_log_file = Path(str(tmpdir)) / default_info_log_folder / Path("p1000_info_log_file.csv")
        # create parent folder and file
        fake_old_log_file.parent.mkdir(parents=True, exist_ok=True)
        fake_old_log_file.touch()
    else:
        fake_old_log_file = None
    # bcast fake log file to all processes
    fake_old_log_file = comm.bcast(fake_old_log_file, 0)
    # check if file exists
    assert fake_old_log_file.exists(), "It should exist"
    # wait until all processes have checked fake old file existence, otherwise the following statement may eliminate it
    # before testing
    comm.Barrier()
    # configure root _logger
    confgure_root_logger_with_standard_settings(Path(str(tmpdir)))
    # check if old file exists
    assert not fake_old_log_file.exists(), "It should have been removed"
