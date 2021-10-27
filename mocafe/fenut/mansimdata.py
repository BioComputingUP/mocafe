import datetime
import fenics
from mocafe.fenut.parameters import Parameters
import pathlib
"""
Useful methods to manage simulation data
"""

# def macros
default_sim_name = "default"
test_sim_name = "test"
runtime_folder = pathlib.Path("./runtime")
saved_sim_folder = pathlib.Path("./saved_sim")
sim_info_file = pathlib.Path("sim_info.html")


def setup_data_folder(sim_name: str or None = default_sim_name,
                      other_location: str or None = None) -> pathlib.Path:
    """
    Setup the folder where the simulation will be saved. One can specify the simulation name; in case that is None the
    simulation will be saved to the './runtime' folder, otherwise it will be saved to ./saved_sim/sim_name/0000.
    In case the same sim_name is given for multiple simulation, the result will be saved to ./saved_sim/sim_name/0001,
    0002, ... and so on.
    Works in parallel.
    :param sim_name: The name of the simulation. If None, the data_folder is './runtime'
    :param other_location: instead of ./saved_sim, specify another location for your saved simulations
    :return: the data folder
    """
    comm = fenics.MPI.comm_world
    rank = comm.Get_rank()
    if rank == 0:
        if sim_name == default_sim_name or sim_name is None:
            data_folder = runtime_folder
        else:
            local_saved_sim_folder = saved_sim_folder if other_location is None else pathlib.Path(other_location)
            base_code = "0000"
            data_folder = local_saved_sim_folder / pathlib.Path(f"{sim_name}/{base_code}")
            if data_folder.exists():
                sim_index = 1
                len_code = len(base_code)
                while data_folder.exists():
                    new_code = str(sim_index).zfill(len_code)
                    data_folder = local_saved_sim_folder / pathlib.Path(f"{sim_name}/{new_code}")
                    sim_index += 1
        data_folder.mkdir(parents=True, exist_ok=True)
    else:
        data_folder = None
    data_folder = comm.bcast(data_folder, 0)
    return data_folder


def save_sim_info(data_folder: pathlib.Path,
                  execution_time: float,
                  parameters: Parameters or dict,
                  sim_name: str = default_sim_name,
                  dateandtime: str = "auto",
                  sim_rationale: str = "input",
                  error_msg: str = None) -> None:
    """
    Save simulation infos as json file. The simulation infos will be stored in the provided data_folder.
    :param data_folder: the data folder containing the simulation files
    :param execution_time: the execution time of the simulation
    :param parameters: the parameters used for the simulation (can be a dict of parameters files, each identified
                       by a name)
    :param sim_name: the simulation name
    :param dateandtime: date and time of the simulation. If equals "auto" the time and date are automatically added by
        the method
    :param sim_rationale: rationale of the simulation. If set to input the method will ask the user to type the
        rationale in the command line; otherwise the given rationale will be set in the "rationale" field of the
        sim_info.json file
    :param error_msg: if an error occurred during the simulation, save the error message
    :return: nothing
    """
    comm = fenics.MPI.comm_world
    rank = comm.Get_rank()
    # if sim_name is not default, ask user the rationale for the simulation
    if rank == 0:
        if sim_name == default_sim_name or sim_name == test_sim_name:
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

