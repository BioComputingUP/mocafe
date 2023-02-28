"""
Module containing some useful methods and Classes to work with FEniCS.
Requires FEniCS 2019.1 to work.

"""

import fenics
import json
import numpy as np

_comm_world = fenics.MPI.comm_world
_rank = _comm_world.Get_rank()


def setup_pvd_files(file_names: list, data_folder):
    """
    returns a list of ``.pvd`` files with the given list of names. ``.pvd`` files can be used to store FEniCS Functions,
    even though XDMF files are preferred.

    This method is useful when a relative long list of files is needed for a simulation.

    :param file_names: the list of strings containing the files name.
    :param data_folder: the folder where to place the files
    :return: the FEniCS objects representing the files.
    """
    vtk_files = [fenics.File(str(data_folder) + "/" + file_name + ".pvd", "compressed") for file_name in file_names]
    return vtk_files


def setup_xdmf_files(file_names: list,
                     data_folder,
                     xdmf_files_parameters: dict = None,
                     comm=fenics.MPI.comm_world):
    """
    returns a list of ``.xdmf`` files with the given list of names. ``.xdmf`` files are the preferred way to store
    FEniCS functions for elaboration and visualization.

    This method is useful when a relative long list of files is needed for a simulation.

    :param file_names: the list of strings containing the files name.
    :param data_folder: the folder where to place the files
    :param xdmf_files_parameters: (new in 1.4) set parameters to generated xdmf files
    :param comm: MPI communicator. Default is COMM_WORLD and this is usually the best choice for normal simulations.
    :return: the FEniCS objects representing the files.
    """
    xdmf_files = [fenics.XDMFFile(comm, str(data_folder) + "/" + file_name + ".xdmf") for file_name in file_names]
    if xdmf_files_parameters is not None:
        for xdmf_f in xdmf_files:
            for k, i in xdmf_files_parameters.items():
                xdmf_f.parameters[k] = i
    return xdmf_files


def divide_in_chunks(given_list, n_chnks):
    """
    Divides the elements of a given list in ``n_chnks`` list with almost the same number of elements. The result is
    a list of lists.

    More precisely, the total number of element of the given list is divided by ``n_chnks``. If the number is not
    divisible by ``n_chnks``, the remainder will be distributed over the lists.

    Below are provided some examples:

    >>> a = [1, 2, 3, 4, 5, 6, 7]
    >>> divide_in_chunks(a, 3)
    [[1, 2, 3], [4, 5], [6, 7]]

    >>> b = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> divide_in_chunks(b, 2)
    [[1, 2, 3, 4], [5, 6, 7, 8]]

    Notice that the order is conserved.

    :param given_list: the list to be divided in chunks
    :param n_chnks: the number of desidered chunks.
    :return: a list of lists, where each element is one of the chunks.
    """
    quot, rem = divmod(len(given_list), n_chnks)
    chunks_list = []
    bookmark = 0
    for i_chunk in range(n_chnks):
        if i_chunk < rem:
            len_chunk = quot + 1
        else:
            len_chunk = quot

        chunk = given_list[bookmark:bookmark + len_chunk]
        chunks_list.append(chunk)
        bookmark += len_chunk
    return chunks_list


def load_parameters(parameters_file="parameters.json"):
    """
    DEPRECATED: use Parameters class instead
    :param parameters_file:
    :return:
    """
    input_files_folder = "./input_files"
    with open(input_files_folder + "/" + parameters_file) as json_file:
        parameters = json.load(json_file)
    return parameters


def get_mixed_function_space(mesh: fenics.Mesh,
                             n_variables: int,
                             element_type: str = "CG",
                             degree: int = 1):
    """
    Builds a mixed function space for the given mesh with n_variables elements of the same given element_type.
    :param mesh: the mesh to build the function space on.
    :param n_variables: the number of elements composing the mixed elements space, usually equal to the number of
    variables you need to simulate.
    :param element_type: the type of element you want to use, identified with the fenics string. Default is Continuos
    Glaerkin, 'CG'.
    :param degree: the degree of the elements you want to use. Default is 1.
    :return: the function space for the given mesh.
    """
    if mesh.geometric_dimension() == 2:
        element = fenics.FiniteElement(element_type, fenics.triangle, degree)
    elif mesh.geometric_dimension() == 3:
        element = fenics.FiniteElement(element_type, fenics.tetrahedron, degree)
    else:
        raise RuntimeError
    mixed_element = fenics.MixedElement([element] * n_variables)
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    return function_space


def build_local_box(local_mesh: fenics.Mesh,
                    border_width: float):
    """
    Builds a local box for a given mesh.

    A local box is useful for parallel computation and might be useful for different purposes. In general, defines
    a square space, bigger than the local mesh, which may contain elements of interest for the local mesh

    :param local_mesh: the local mesh for the current MPI process
    :param border_width: the width of the border of the local box
    :return: the local box
    """
    local_coordinates = local_mesh.coordinates()
    max_values = np.max(local_coordinates, axis=0) + border_width
    min_values = np.min(local_coordinates, axis=0) - border_width
    if len(max_values) == 2:
        local_box = {
            "dim": 2,
            "x_min": min_values[0],
            "x_max": max_values[0],
            "y_min": min_values[1],
            "y_max": max_values[1]
        }
    elif len(max_values) == 3:
        local_box = {
            "dim": 3,
            "x_min": min_values[0],
            "x_max": max_values[0],
            "y_min": min_values[1],
            "y_max": max_values[1],
            "z_min": min_values[2],
            "z_max": max_values[2]
        }
    else:
        raise RuntimeError("Found a geometric dimension different than 2 or 3. Local box cannot be defined.")
    return local_box


def is_in_local_box(local_box, position):
    """
    Given a local box, checks if the given point is inside that local box.

    :param local_box: the local box
    :param position: the position to check
    :return: True if the position is inside the local box. False otherwise
    """
    if local_box["dim"] == 2:
        try:
            is_inside = (local_box["x_min"] < position[0] < local_box["x_max"]) \
                and (local_box["y_min"] < position[1] < local_box["y_max"])
        except IndexError as e:
            print(position, type(position))
            raise e
    else:
        try:
            is_inside = (local_box["x_min"] < position[0] < local_box["x_max"]) \
                and (local_box["y_min"] < position[1] < local_box["y_max"]) \
                and (local_box["z_min"] < position[2] < local_box["z_max"])
        except IndexError as e:
            print(position, type(position))
            raise e
    return is_inside


def flatten_list_of_lists(list_of_lists):
    """
    Flattens a list of lists in a flat list
    :param list_of_lists: the given list of lists
    :return: the flat list
    """
    return [elem for sublist in list_of_lists for elem in sublist]


def is_point_inside_mesh(mesh: fenics.Mesh,
                         point):
    """
    Check if the given point is inside the given mesh. In parallel, checks if the point is inside the local mesh.

    :param mesh: the given Mesh
    :param point: the given point
    :return: True if the point is inside the mesh; False otherwise
    """
    if not isinstance(point, fenics.Point):
        if isinstance(point, np.ndarray):
            # convert in fenics point
            point = fenics.Point(point)
        else:
            raise TypeError(f"given point of unknown type {type(point)}")
    bbt = mesh.bounding_box_tree()
    is_point_inside = bbt.compute_first_entity_collision(point) <= mesh.num_cells()
    return is_point_inside
