from pathlib import Path
import logging
import fenics

# get header for csv log file
log_header = ["timestamp", "mpi_p", "module", "activity"]

# log folder
logfolder = Path("log")

# get rank
rank = fenics.MPI.comm_world.Get_rank()


class CsvAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs):
        return f"p{self.extra['rank']};{self.extra['module']};{msg}", kwargs


class InfoAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs):
        return f"p{self.extra['rank']} - {self.extra['module']}\n \t{msg}", kwargs


def get_progress_adapter(module_name: str, is_main: bool = False):
    """
    Create a logger to report the progress of the operations as csv file, with timestamp, the rank, the module and the
    activity performed. This module is thought for measuring program performance and for optimization.
    :param module_name: the module name which calls the method
    :param is_main: to be set to True if the module is the main simulation module, else is False. Default is False.
    :return: and adapter to use as a logger to log progress message.
    """
    main_module_name = "progress_simulation"
    main_log_folder = logfolder / Path("progress")
    if is_main:
        # create logger
        logger = logging.getLogger(main_module_name)
        logger.setLevel(logging.INFO)
        # create file handler
        main_log_folder.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(str(main_log_folder / Path(f"p_{rank}_progress_log.csv")), mode="w")
        fh.setLevel(logging.INFO)
        # create stream Handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        # create format
        formatter = logging.Formatter("{asctime};{message}", style="{")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        # add file handler to logger
        logger.addHandler(fh)
        logger.addHandler(sh)
    else:
        logger = logging.getLogger(main_module_name + f".{module_name}")

    adapter = CsvAdapter(logger, {"rank": rank, "module": module_name})
    return adapter


def get_info_adapter(module_name: str, is_main: bool = False):
    """
        Create a logger to report the progress of the operations as log file with timestamp, the rank, the module and
        the message. This module is though to save general information messages on the program execution.
        :param module_name: the module name which calls the method
        :param is_main: to be set to True if the module is the main simulation module, else is False. Default is False.
        :return: and adapter to use as a logger to log progress message.
        """
    main_module_name = "info_simulation"
    main_log_folder = logfolder / Path("info")
    if is_main:
        # create logger
        logger = logging.getLogger(main_module_name)
        logger.setLevel(logging.INFO)
        # create file handler
        main_log_folder.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(str(main_log_folder / Path(f"p_{rank}_info_log.log")), mode="w")
        fh.setLevel(logging.INFO)# create stream Handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        # create format
        formatter = logging.Formatter("{asctime} - {message}", style="{")
        fh.setFormatter(formatter)
        # add file handler to logger
        logger.addHandler(fh)
        logger.addHandler(sh)
    else:
        logger = logging.getLogger(main_module_name + f".{module_name}")

    adapter = InfoAdapter(logger, {"rank": rank, "module": module_name})
    return adapter

