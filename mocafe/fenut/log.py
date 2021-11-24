from pathlib import Path
import logging
import fenics

# get header for csv log file
log_header = ["timestamp", "mpi_p", "module", "activity"]

# get rank
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# define default log folder tree
default_main_log_folder = Path("log")
default_debug_log_folder = default_main_log_folder / Path("DEBUG")
default_info_log_folder = default_main_log_folder / Path("INFO")
default_debug_log_file = Path(f"p{rank}_debug_log.log")
default_info_log_file = Path(f"p{rank}_info_log.csv")


class InfoCsvAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs):
        return f"p{self.extra['rank']};{self.extra['module']};{msg}", kwargs


class DebugAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs):
        return f"p{self.extra['rank']} - {self.extra['module']}\n \t{msg}", kwargs


def _create_clean_log_folder(folder: Path):
    # create folder
    folder.mkdir(parents=True, exist_ok=True)
    # clean the folder
    if rank == 0:
        for file in folder.iterdir():
            file_name = file.name
            file_process = file_name.split("_")[0]
            file_process_rank = int(file_process.replace("p", ''))
            if file_process_rank >= comm.Get_size():
                file.unlink()
    return folder


def confgure_root_logger_with_standard_settings(data_folder: Path):
    # define debug log folder - more verbose
    current_debug_log_folder = data_folder / default_debug_log_folder
    # define info log folder - for optimization
    current_info_log_folder = data_folder / default_info_log_folder

    # create log folder or clean previous content that might be in the folder
    current_debug_log_folder = _create_clean_log_folder(current_debug_log_folder)
    current_info_log_folder = _create_clean_log_folder(current_info_log_folder)


    # get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Debug
    # create debug log file - one for each mpi process
    debug_fh = logging.FileHandler(str(current_debug_log_folder / default_debug_log_file), mode="w")
    debug_fh.setLevel(logging.DEBUG)
    # create dubug format
    debug_formatter = logging.Formatter("\033[94m{asctime}\033[0m - {message}", style="{")
    debug_fh.setFormatter(debug_formatter)
    # add handler to root logger
    root_logger.addHandler(debug_fh)

    # Info
    # create info log file - one for each mpi process
    info_fh = logging.FileHandler(str(current_info_log_folder / default_info_log_file), mode="w")
    info_fh.setLevel(logging.INFO)
    # create info format
    info_formatter = logging.Formatter("{asctime};{message}", style="{")
    info_fh.setFormatter(info_formatter)
    # add handler to root logger
    root_logger.addHandler(info_fh)

    # Stream
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
