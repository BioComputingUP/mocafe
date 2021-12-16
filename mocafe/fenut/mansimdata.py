import datetime
import fenics
from mocafe.fenut.parameters import Parameters
import pathlib
"""
Useful methods to manage simulation data
"""

comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# def macros
default_data_folder_name = "sim_data"
test_sim_name = "test"
sim_info_file = pathlib.Path("sim_info.html")
default_runtime_folder_name = "runtime"
default_saved_sim_folder_name = "saved_sim"


def setup_data_folder(folder_name: str or None = None,
                      base_location: str or None = "./",
                      enumerate: bool = True) -> pathlib.Path:
    """
    Creates and returns the folder where the simulation files will be saved.

    One can specify the folder name; in case that is None, the simulation will be saved to the
    'sim_data' folder.

    The ``base_location`` is where the folder it is based. Default is './'. Thus, if one calls

        setup_data_folder(base_location="./saved_simulations")

    The data folder will be ``./saved_simulations/sim_data``

    The ``enumerate`` argument can be used to specify if you need coded folders for the same simulation. If
    you call:

        setup_data_folder(folder_name="my_sim_data", enumerate=True)

    This method will return:

    - the folder ``./my_sim_data/0000`` the first time the method is called
    - the folder ``./my_sim_data/0001`` the second time the method is called
    - ... and so on

    Works in parallel with MPI.

    :param folder_name: the name of the folder where the simulation data will be saved
    :param base_location: The base folder where the simulation will be saved. Default is './'
    :param enumerate: specify if you need the method to return coded folders for each simulation with the same
    folder name
    :return: the data folder
    """
    if rank == 0:
        # get base location path
        base_location_path = pathlib.Path(base_location)
        # get the data folder name
        if folder_name is None:
            data_folder = base_location_path / pathlib.Path(default_data_folder_name)
        else:
            data_folder = base_location_path / pathlib.Path(folder_name)
        # if you wank to keep all sim files, generate a new folder for each simulation
        if enumerate:
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

        data_folder.mkdir(parents=True, exist_ok=True)
    else:
        data_folder = None
    data_folder = comm.bcast(data_folder, 0)
    return data_folder


def save_sim_info(data_folder: pathlib.Path,
                  execution_time: float,
                  parameters: Parameters or dict,
                  sim_name: str = default_data_folder_name,
                  dateandtime: str = "auto",
                  sim_rationale: str = "input",
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
    :param sim_rationale: rationale of the simulation. If set to input the method will ask the user to type the
        rationale in the command line; otherwise the given rationale will be set in the "rationale" field of the
        sim_info.html file. Default is "input".
    :param error_msg: if an error occurred during the simulation, save the error message
    :return: nothing
    """
    # if sim_name is not default, ask user the rationale for the simulation
    if rank == 0:
        if sim_name == default_data_folder_name or sim_name == test_sim_name:
            sim_rationale = sim_name
        else:
            if sim_rationale == "input":
                print("--- Simulation Rationale --- ")
                sim_rationale = input("Type the rationale for the simulation: ")
        with open(data_folder / sim_info_file, "w+") as report_file:
            report_file.write(f"<article>\n")
            report_file.write(f"  <h1>Simulation report </h1>\n")
            report_file.write(f"  <h2>Basic informations </h2>\n")
            report_file.write(f"  <p>Simulation name: {sim_name} </p>\n")
            report_file.write(f"  <p>Execution time: {execution_time / 60} min </p>\n")
            report_file.write(f"  <p>Date and time: "
                              f"{str(datetime.datetime.now()) if dateandtime == 'auto' else dateandtime} </p>\n")
            report_file.write(f"  <h2>Simulation rationale </h2>\n")
            report_file.write(f"  <p>{sim_rationale} </p>\n")
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
