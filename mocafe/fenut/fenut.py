import fenics
import json

"""
Module containing some useful methods and Classes to work with FEniCS.
Requires FEniCS 2019.1 to work.

"""


def setup_vtk_files(file_names: list, data_folder):
    vtk_files = [fenics.File(str(data_folder) + "/" + file_name + ".pvd", "compressed") for file_name in file_names]
    return vtk_files


def setup_xdmf_files(file_names: list, data_folder, comm=fenics.MPI.comm_world):
    xdmf_files = [fenics.XDMFFile(comm, str(data_folder) + "/" + file_name + ".xdmf") for file_name in file_names]
    return xdmf_files


def divide_in_chunks(given_list, n_chnks):
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
    input_files_folder = "./input_files"
    with open(input_files_folder + "/" + parameters_file) as json_file:
        parameters = json.load(json_file)
    return parameters


class RectangleMeshWrapper:
    """ Wrapper for fenics.RectangleMesh with some utility method """
    def __init__(self, limit_point1: fenics.Point, limit_point2: fenics.Point, nx, ny):
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
        return self.local_mesh

    def get_global_mesh(self):
        return self.global_mesh

    def is_inside_local_mesh(self, point):
        point_to_check = point if type(point) is fenics.Point else fenics.Point(point)
        return \
            self.local_bounding_box_tree.compute_first_entity_collision(point_to_check) <= self.local_mesh.num_cells()

    def is_inside_global_mesh(self, point):
        point_to_check = point if type(point) is fenics.Point else fenics.Point(point)
        return \
            self.global_bounding_box_tree.compute_first_entity_collision(point_to_check) <= self.global_mesh.num_cells()

    def n_local_mesh_vertices(self):
        return self.local_mesh.num_vertices()

    def n_global_mesh_vertices(self):
        return self.global_mesh.num_vertices()

    def get_dim(self):
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
    return (local_box["x_min"] < position[0] < local_box["x_max"]) \
           and (local_box["y_min"] < position[1] < local_box["y_max"])


def flatten_list_of_lists(list_of_lists):
    return [elem for sublist in list_of_lists for elem in sublist]
