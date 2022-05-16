from pathlib import Path
import logging
import fenics

# get header for csv log file
log_header = ["timestamp", "mpi_p", "module", "activity"]

# get _rank
comm = fenics.MPI._comm_world
rank = comm.Get_rank()

# define default log folder tree
default_main_log_folder = Path("log")
default_debug_log_folder = default_main_log_folder / Path("DEBUG")
default_info_log_folder = default_main_log_folder / Path("INFO")
default_debug_log_file = Path(f"p{rank}_debug_log.log")
default_info_log_file = Path(f"p{rank}_info_log.csv")


class InfoCsvAdapter(logging.LoggerAdapter):
    """
    Standard logging adapter for logging the "INFO" level as a csv file.

    This is mainly used for internal purposes, but can be used from the user as well.
    """
    def process(self, msg: str, kwargs):
        return f"p{self.extra['_rank']};{self.extra['module']};{msg}", kwargs


class DebugAdapter(logging.LoggerAdapter):
    """
    Standerd logging adapter ffor logging the "DEBUG" level.

    This is mainly used for internal purposes, but can be used from the user as well.
    """
    def process(self, msg: str, kwargs):
        return f"p{self.extra['_rank']} - {self.extra['module']}\n \t{msg}", kwargs


def _create_clean_log_folder(folder: Path):
    """
    INTERNAL USE

    Creates a clean log folder, eliminating all possible files inside.

    :param folder: the log folder to clean
    :return: the clean folder
    """
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
    """
    Configures the root _logger with the standard Mocafe settings.

    This is mainly used for internal purposes, but can be used from the user as well.

    The user can define its own logging configurations using the ``logging`` Python module.

    :param data_folder: the folder where to place the logging files and folder.
    :return: nothing
    """
    # define debug log folder - more verbose
    current_debug_log_folder = data_folder / default_debug_log_folder
    # define info log folder - for optimization
    current_info_log_folder = data_folder / default_info_log_folder

    # create log folder or clean previous content that might be in the folder
    current_debug_log_folder = _create_clean_log_folder(current_debug_log_folder)
    current_info_log_folder = _create_clean_log_folder(current_info_log_folder)


    # get root _logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Debug
    # create debug log file - one for each mpi process
    debug_fh = logging.FileHandler(str(current_debug_log_folder / default_debug_log_file), mode="w")
    debug_fh.setLevel(logging.DEBUG)
    # create dubug format
    debug_formatter = logging.Formatter("{asctime} - {message}", style="{")
    debug_fh.setFormatter(debug_formatter)
    # add handler to root _logger
    root_logger.addHandler(debug_fh)

    # Info
    # create info log file - one for each mpi process
    info_fh = logging.FileHandler(str(current_info_log_folder / default_info_log_file), mode="w")
    info_fh.setLevel(logging.INFO)
    # create info format
    info_formatter = logging.Formatter("{asctime};{message}", style="{")
    info_fh.setFormatter(info_formatter)
    # add handler to root _logger
    root_logger.addHandler(info_fh)

    # Stream
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
