import fenics
from pathlib import Path
import random
import pickle

comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# default random state file
default_randomstate_file = Path(f"p{rank}.randomstate")


def set_equal_randomstate_for_all_p():
    """
    Set the randomstate of to MPI process 0 to all processes.

    Each MPI process has a different randomstate. This makes it difficult to reproduce the same simulation
    multiple times, especially if the simulation is run with a different number of processes. Thus, it is useful for
    reproducibility to set the same random state to all processes.

    By default, this method runs at the start of each simulation. You can eliminate the effect of this statement
    simply placing at the end of your simulation::
        import random
        random.setstate(None)

    :return: nothing
    """
    if rank == 0:
        # get randomstate for p0
        p0_randomstate = random.getstate()
    else:
        p0_randomstate = None
    # broadcast randomstate to all processes
    p0_randomstate = comm.bcast(p0_randomstate, 0)
    # set p0_randomstate to all procesess
    random.setstate(p0_randomstate)


def load_random_state(folder_name: str or Path,
                      equal_for_all_p: bool = True):
    """
    Load a random state for each MPI process from a given folder. The default behaviour is to load the randomstate
    for MPI process 0 and set it to all processes. Otherwise, it can also load a different randomstate for each
    MPI process. In any case, the randomstates must be saved in the given folder as 'p{rank}.randomstate' (e.g.
    p0.randomstate for process 0).
    *Note: using MPI, each process has a different random state.*
    :param folder_name: the folder containing the randomstate to load
    :param equal_for_all_p: default True. Set to false to load a different randomstate for each process. There must be a
    file for each MPI process.
    :return:
    """
    # get folder as Path object
    folder_path = folder_name if type(folder_name) is Path else Path(folder_name)
    # get randomstate file path
    if equal_for_all_p:
        randomstate_file_path = folder_path / Path("p0.randomstate")
    else:
        randomstate_file_path = folder_path / default_randomstate_file
    # use the file to set the randomstate
    with open(str(randomstate_file_path), "rb") as f:
        random.setstate(pickle.load(f))


def save_random_state(folder_name: str or Path,
                      equal_for_all_p: bool = True):
    """
    Save the random state for the simulation to reproduce it. The default behaviour is to save the randomstate for
    MPI process 0, that should be the same for all processes. However, it can also save a different randomstate for
    each process. The default format for randomstate file is 'p{rank}.randomstate' (e.g. p0.randomstate)
    :param folder_name: the folder name or Path where to save the randomstate
    :param equal_for_all_p: default is True. Set to false to save a different randomstate for each process.
    :return: nothing
    """
    if rank == 0:
        # create folder in rank 0
        folder_path = folder_name if type(folder_name) is Path else Path(folder_name)
        folder_path.mkdir(exist_ok=True, parents=True)
    else:
        folder_path = None
    # broadcast folder to all processes
    folder_path = comm.bcast(folder_path, 0)
    if equal_for_all_p:
        # save only randomstate for p0
        if rank == 0:
            randomstate_file_path = folder_path / default_randomstate_file
            randomstate = random.getstate()
            with open(randomstate_file_path, "wb") as f:
                pickle.dump(randomstate, f)
        # for all other processes, wait until p0 has finished saving
        comm.Barrier()
    else:
        # save a different randomstate file for each process
        randomstate_file_path = folder_path / default_randomstate_file
        with open(randomstate_file_path, "wb") as f:
            pickle.dump(random.getstate(), f)
