"""
Useful methods to manage simulation data
"""

import datetime
import fenics
import mocafe
from mocafe.fenut.parameters import Parameters
import pathlib

comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# def macros
default_data_folder_name = "sim_data"
test_sim_name = "test"
sim_info_file = pathlib.Path("sim_info.html")
default_runtime_folder_name = "runtime"
default_saved_sim_folder_name = "saved_sim"


def setup_data_folder(folder_path: str,
                      auto_enumerate: bool = True) -> pathlib.Path:
    """
    Creates a folder at the given folder path and returns it as pathlib.Path object to use for the simulation
    files.

    If just the folder path is specified the folder will be simply created. For instance, calling the following:

    .. code-block:: default

        setup_data_folder(folder_name="my_sim_data")

    Will just create the folder "my_sim_data" in the current root directory.

    The ``auto_enumerate`` argument can be used to specify if you need coded folders for the same simulation. This
    is useful especially when one needs to save multiple simulation results under the same folder without having
    to decide a new folder path for each simulation. Indeed, if you call:

    .. code-block:: default

        setup_data_folder(folder_path="my_sim_data", auto_enumerate=True)

    This method will return:

    - the folder ``./my_sim_data/0000`` the first time the method is called
    - the folder ``./my_sim_data/0001`` the second time the method is called
    - ... and so on

    Works in parallel with MPI.

    :param folder_path: the path of the folder to generate
    :param auto_enumerate: if set to True, this method will create coded folder under the given ``folder_path`` in the
        order: 0000, 0001, ..., and so on.
    :return: the generated folder
    """
    if rank == 0:
        # rename folder_path
        data_folder = pathlib.Path(folder_path)
        # if you wank to keep all sim files, generate a new folder for each simulation
        if auto_enumerate:
            base_code = "0000"
            data_folder_coded = data_folder / pathlib.Path(f"{base_code}")
            if data_folder_coded.exists():
                sim_index = 1
                len_code = len(base_code)
                while data_folder_coded.exists():
                    new_code = str(sim_index).zfill(len_code)
                    data_folder_coded = data_folder / pathlib.Path(f"{new_code}")
                    sim_index += 1
            # the coded data folder at the end of the iteration is the data folder
            data_folder = data_folder_coded
        # create data folder
        data_folder.mkdir(parents=True, exist_ok=True)
    else:
        data_folder = None
    data_folder = comm.bcast(data_folder, 0)
    return data_folder


def save_sim_info(data_folder: pathlib.Path,
                  parameters: Parameters or dict,
                  execution_time: float or None = None,
                  sim_name: str = default_data_folder_name,
                  dateandtime: str = "auto",
                  sim_description: str = None,
                  error_msg: str = None) -> None:
    """
    Save simulation infos as html file. The simulation infos will be stored in the provided data_folder.

    :param data_folder: the data folder containing the simulation files
    :param execution_time: the execution time of the simulation
    :param parameters: the parameters used for the simulation (can be a dict of parameters files, each identified
        by a name)
    :param sim_name: the simulation name
    :param dateandtime: date and time of the simulation. If it is equal to "auto" the time and date are automatically
        added by the method
    :param sim_description: rationale of the simulation. If set to input the method will ask the user to type the
        rationale in the command line; otherwise the given rationale will be set in the "rationale" field of the
        sim_info.html file. Default is None.
    :param error_msg: if an error occurred during the simulation, save the error message
    :return: nothing
    """
    # if sim_name is not default, ask user the rationale for the simulation
    if rank == 0:
        if sim_name == default_data_folder_name or sim_name == test_sim_name:
            sim_description = sim_name
        else:
            if sim_description == "input":
                print("--- Simulation Rationale --- ")
                sim_description = input("Type the rationale for the simulation: ")
        with open(data_folder / sim_info_file, "w+") as report_file:
            report_file.write(f"<article>\n")
            report_file.write(f"  <h1>Simulation report </h1>\n")
            report_file.write(f"  <h2>Basic informations </h2>\n")
            report_file.write(f"  <p>Simulation name: {sim_name} </p>\n")
            report_file.write(f"  <p>Execution time: {execution_time / 60} min </p>\n")
            report_file.write(f"  <p>Mocafe version: {mocafe.__version__} min </p>\n")
            report_file.write(f"  <p>Date and time: "
                              f"{str(datetime.datetime.now()) if dateandtime == 'auto' else dateandtime} </p>\n")
            report_file.write(f"  <h2>Simulation rationale </h2>\n")
            report_file.write(f"  <p>{sim_description} </p>\n")
            report_file.write(f"  <h2>Parameters used </h2>\n")
            if type(parameters) is Parameters:
                report_file.write(parameters.as_dataframe().to_html())
                report_file.write("\n")
            elif type(parameters) is dict:
                for parameters_set in parameters:
                    report_file.write(f"  <h3>Parameters set {parameters_set} </h3> \n")
                    report_file.write(parameters[parameters_set].as_dataframe().to_html())
                    report_file.write("\n")
            else:
                raise ValueError(f"parameters can be only of type Parameters or dict. "
                                 f"Found type {type(parameters)} instead")
            if error_msg is not None:
                report_file.write(f"  <h2>Errors </h2>\n")
                report_file.write(f"  <p>\n"
                                  f"    Error message: \n"
                                  f"    {error_msg} \n"
                                  f"  </p>\n")
            report_file.write(f"</article>")
    # wait until MPI process 0 has written the simulation info file
    comm.Barrier()
