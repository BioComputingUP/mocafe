import fenics
import json

"""
Module containing some useful methods and Classes to work with FEniCS.
Requires FEniCS 2019.1 to work.

"""


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


def setup_xdmf_files(file_names: list, data_folder, comm=fenics.MPI.comm_world):
    """
    returns a list of ``.xdmf`` files with the given list of names. ``.xdmf`` files are the preferred way to store
    FEniCS functions for elaboration and visualization.

    This method is useful when a relative long list of files is needed for a simulation.

    :param file_names: the list of strings containing the files name.
    :param data_folder: the folder where to place the files
    :param comm: MPI communicator. Default is COMM_WORLD and this is usually the best choice for normal simulations.
    :return: the FEniCS objects representing the files.
    """
    xdmf_files = [fenics.XDMFFile(comm, str(data_folder) + "/" + file_name + ".xdmf") for file_name in file_names]
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


class RectangleMeshWrapper:
    """ Wrapper for fenics.RectangleMesh with some utility method """
    def __init__(self, limit_point1: fenics.Point, limit_point2: fenics.Point, nx, ny):
        """
        inits a wrapper for rectangle mesh that is useful for the parallel implementation of the source cells and
        the tip cells.
        :param limit_point1: first point defining the rectangle
        :param limit_point2: second point defining the rectangle
        :param nx: number of points in the x direction
        :param ny: number of points in the y direction
        """
        self.local_mesh = fenics.RectangleMesh(limit_point1,
                                               limit_point2,
                                               nx, ny)
        self.local_bounding_box_tree = self.local_mesh.bounding_box_tree()
        self.global_mesh = fenics.RectangleMesh(fenics.MPI.comm_self,
                                                limit_point1,
                                                limit_point2,
                                                nx, ny)
        self.global_bounding_box_tree = self.global_mesh.bounding_box_tree()
        self.dim = 2

    def get_local_mesh(self):
        """
        get the mesh visible to the local MPI process
        :return: the local Mesh object
        """
        return self.local_mesh

    def get_global_mesh(self):
        """
        get the global mesh shared among all the MPI processes
        :return: the global mesh shared among all the MPI processes
        """
        return self.global_mesh

    def is_inside_local_mesh(self, point):
        """
        check if the given point is inside the local mesh
        :param point: the point to check
        :return: True if the point is inside the local mesh. False otherwise.
        """
        point_to_check = point if type(point) is fenics.Point else fenics.Point(point)
        return \
            self.local_bounding_box_tree.compute_first_entity_collision(point_to_check) <= self.local_mesh.num_cells()

    def is_inside_global_mesh(self, point):
        """
        check if the given point is inside the global mesh
        :param point: the point to check
        :return: True if the point is inside the local mesh. False otherwise
        """
        point_to_check = point if type(point) is fenics.Point else fenics.Point(point)
        return \
            self.global_bounding_box_tree.compute_first_entity_collision(point_to_check) <= self.global_mesh.num_cells()

    def n_local_mesh_vertices(self):
        """
        get the number of points in the local mesh
        :return: the number of points in the local mesh
        """
        return self.local_mesh.num_vertices()

    def n_global_mesh_vertices(self):
        """
        get the number of points in the global mesh
        :return: the number of points in the global mesh
        """
        return self.global_mesh.num_vertices()

    def get_dim(self):
        """
        get the number of dimensions of the mesh
        :return: 2 if the mesh is 2 dimensonal.
        """
        return self.dim


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
    element = fenics.FiniteElement(element_type, fenics.triangle, degree)
    mixed_element = fenics.MixedElement([element] * n_variables)
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    return function_space


def build_local_box(local_mesh, border_width):
    """
    Builds a local box for a given mesh.

    A local box is useful for parallel computation and might be useful for different purposes. In general, defines
    a square space, bigger than the local mesh, which may contain elements of interest for the local mesh
    :param local_mesh: the local mesh for the current MPI process
    :param border_width: the width of the border of the local box
    :return: the local box
    """
    x_list = []
    y_list = []
    for point in local_mesh.coordinates():
        x_list.append(point[0])
        y_list.append(point[1])
    x_min = min(x_list) - border_width
    x_max = max(x_list) + border_width
    y_min = min(y_list) - border_width
    y_max = max(y_list) + border_width
    local_box = {"x_min": x_min,
                 "x_max": x_max,
                 "y_min": y_min,
                 "y_max": y_max}
    return local_box


def is_in_local_box(local_box, position):
    """
    Given a local box, checks if the given point is inside that local box.
    :param local_box: the local box
    :param position: the position to check
    :return: True if the position is inside the local box. False otherwise
    """
    return (local_box["x_min"] < position[0] < local_box["x_max"]) \
           and (local_box["y_min"] < position[1] < local_box["y_max"])


def flatten_list_of_lists(list_of_lists):
    """
    Flattens a list of lists in a flat list
    :param list_of_lists: the given list of lists
    :return: the flat list
    """
    return [elem for sublist in list_of_lists for elem in sublist]
