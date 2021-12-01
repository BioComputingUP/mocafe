import fenics
import random
import numpy as np
import logging
import mocafe.fenut.fenut as fu
from mocafe.angie import base_classes
from mocafe.fenut.parameters import Parameters
from mocafe.fenut.log import InfoCsvAdapter, DebugAdapter

"""
This module contains classes and methods to manage discrete angiogenic factor sources in ``mocafe``. More precisely, 
it provides useful tools to create cells (described as circles with a given radius) which act as angiogenic factor
sources, inducing angiogenesis.

This module have been implemented to reproduce the results given by the angiogenesis model presented by Lorenzo et al. 
(2011) [Travasso2011] _, where they where testing their model using multiple sources in random positions.

.. [Travasso2011] Travasso, R. D. M., Poiré, E. C., Castro, M., Rodrguez-Manzaneque, J. C., & Hernández-Machado, A. 
   (2011). Tumor angiogenesis and vascular patterning: A mathematical model. PLoS ONE, 6(5), e19989. 
   https://doi.org/10.1371/journal.pone.0019989
"""

# Get MPI communicator and rank to be used in the module
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# configure logger
logger = logging.getLogger(__name__)
info_adapter = InfoCsvAdapter(logger, {"rank": rank, "module": __name__})  # mainly for process optimization
debug_adapter = DebugAdapter(logger, {"rank": rank, "module": __name__})  # all kinds of information


class SourceCell(base_classes.BaseCell):
    """
    Class representing a source cell, i.e. a cell of the non-vascular tissue which expresses an angiogenic factor. It
    is just a wrapper of BaseCell.
    """

    def __init__(self,
                 point: np.ndarray,
                 creation_step):
        """
        inits a source cell centered in a given point.
        :param point: center of the tip cell, as ndarray
        :param creation_step: the step of the simulation at which the cell is created. It is used together with the
        poisition to generate an unique identifier of the cell.
        """
        super(SourceCell, self).__init__(point, creation_step)


class SourceMap:
    """
    Class representing the spatial map of the source cell positions. This class is responsible for keeping the position
    of each source cell at any point of the simulation, providing access to it to other objects and methods.
    """

    def __init__(self,
                 n_sources: int,
                 x_lim,
                 mesh_wrapper: fu.RectangleMeshWrapper,
                 current_step: int,
                 parameters: Parameters,
                 source_points=None):
        """
        inits a SourceMap, i.e. a class responsible for keeping the current position of each source cell at any point
        of the simulation. By default, the sources are placed randomly in the spatial domain.
        :param n_sources: number of random sources to place
        :param x_lim: defines the x value where to start placing sources. if x_lim is 5, the source cells will be placed
        at any point of the domain if x[0] > x_lim
        :param mesh_wrapper: the mesh wrapper
        :param current_step: initialization step of the source map. It is used to initialize tip cells.
        :param parameters: simulation parameters
        :source_points: if the user does not want to have random sources, ha can also define a list of positions where
        to place source cells.
        """
        self.mesh_wrapper = mesh_wrapper
        self.local_box = self._build_local_box(parameters)
        # compute source point
        if source_points is None:
            global_source_points = self._get_randomy_sorted_mesh_points(n_sources, x_lim, mesh_wrapper)
        else:
            global_source_points = source_points
        # initialize SourceCells
        self.global_source_cells = [SourceCell(point, current_step)
                                    for point in global_source_points]
        # compute local source cells
        self.local_source_cells = self._devide_source_cells()

    def _get_randomy_sorted_mesh_points(self, n_sources: int,
                                        x_lim,
                                        mesh_wrapper: fu.RectangleMeshWrapper):
        """
        INTERNAL USE
        Return the source points selected randomly and sorted along the x axis
        :param n_sources: number of sources to select
        :param x_lim: part of the x axis to select
        :param mesh_wrapper: mesh wrapper
        :return: the ndarray of the selected points
        """
        # get comm size
        n_procs = comm.Get_size()
        # define root proc
        root = 0
        # get global coordinates
        global_coords = mesh_wrapper.get_global_mesh().coordinates()
        # devide coordinates among processes
        if rank == root:
            coords_chunks_list = fu.devide_in_chunks(global_coords, n_procs)
        else:
            coords_chunks_list = None
        local_coords_chunk = comm.scatter(coords_chunks_list, root)
        # remove points with x[0] less than x_lim
        pickable_points = [point for point in local_coords_chunk if point[0] > x_lim]
        # compute n local sources
        local_n_sources = self._compute_local_n_sources(n_sources, len(pickable_points), root)
        # pick the source point randomly
        local_sources_array = random.sample(pickable_points, local_n_sources)

        # gather picked sources and share among processes
        local_sources_array_list = comm.gather(local_sources_array, root)
        if rank == root:
            sources_array = fu.flatten_list_of_lists(local_sources_array_list)
        else:
            sources_array = None
        sources_array = comm.bcast(sources_array, root)

        # sort them along the x axes
        sources_array.sort(key=lambda x: x[0])
        return sources_array

    def _compute_local_n_sources(self, global_n_sources, n_local_pickable_points, root):
        """
        INTERNAL USE
        Computes the number of sources to randomly select for the local process, considering that the number
        of pickable points may differ between different processes and that the global number of sources is fixed.
        :param global_n_sources: number of global sources
        :param n_local_pickable_points: number of pickable points
        :param root: MPI root process
        """
        # compute global number of pickable points
        n_local_pickable_points_array = comm.gather(n_local_pickable_points, root)
        if rank == root:
            n_global_pickable_points = sum(n_local_pickable_points_array)
        else:
            n_global_pickable_points = None
        n_global_pickable_points = comm.bcast(n_global_pickable_points, root)

        # compute local process weight
        loc_w = int(np.floor(global_n_sources * (n_local_pickable_points / n_global_pickable_points)))

        # manage missing sources
        loc_w_list = comm.gather(loc_w, root)
        if rank == root:
            # compute missing_sources
            missing_sources = global_n_sources - sum(loc_w_list)
            # pair each process rank with its weight
            proc_loc_w_pairs = list(enumerate(loc_w_list))
            # sort for loc_w
            proc_loc_w_pairs.sort(key=lambda pair: pair[1], reverse=True)
            # mark processes who should take care of the missing sources
            marked_procs = []
            for i in range(int(missing_sources)):
                marked_procs.append(proc_loc_w_pairs[i][0])
        else:
            marked_procs = None
        # get marked processes
        marked_procs = comm.bcast(marked_procs, root)

        # compute local n_sources
        local_n_sources = loc_w
        if rank in marked_procs:
            local_n_sources += 1

        return local_n_sources

    def _devide_source_cells(self):
        competence_source_cells = []
        for source_cell in self.global_source_cells:
            position = source_cell.get_position()
            if self._is_in_local_box(position):
                competence_source_cells.append(source_cell)
        return competence_source_cells

    def _build_local_box(self, parameters: Parameters):
        d = parameters.get_value("d")
        return fu.build_local_box(self.mesh_wrapper.get_local_mesh(), d)

    def _is_in_local_box(self, position):
        return fu.is_in_local_box(self.local_box, position)

    def get_global_source_cells(self):
        return self.global_source_cells

    def get_local_source_cells(self):
        return self.local_source_cells

    def remove_global_source(self, source_cell: SourceCell):
        # remove from global list
        self.global_source_cells.remove(source_cell)
        debug_adapter.debug(f"Removed source cell {source_cell.__hash__()} at position {source_cell.get_position()}"
                            f"from the global list")
        # if in local, remove from local list too
        if source_cell in self.local_source_cells:
            self.local_source_cells.remove(source_cell)
            debug_adapter.debug(
                "Removed source cell {source_cell.__hash__()} at position {source_cell.get_position()}"
                f"from the local list")


class SourcesManager:
    def __init__(self, source_map: SourceMap,
                 mesh_wrapper: fu.RectangleMeshWrapper,
                 parameters: Parameters,
                 expression_function_parameters: dict):
        self.source_map = source_map
        self.mesh_wrapper = mesh_wrapper
        self.parameters: Parameters = parameters
        self.clock_checker = ClockChecker(mesh_wrapper, parameters.get_value("d"))
        if expression_function_parameters["type"] == "rotational":
            ref_parameters = expression_function_parameters["parameters"]
            self.expression_function = RotationalExpressionFunction(ref_parameters)
        else:
            self.expression_function = None

    def remove_sources_near_vessels(self, phi: fenics.Function):
        """
        Remove source cells near the blood vessels
        :param phi: vessel field
        :return:
        """
        # prepare list of cells to remove
        to_remove = []
        debug_adapter.debug(f"Starting to remove source cells")
        for source_cell in self.source_map.get_local_source_cells():
            source_cell_position = source_cell.get_position()
            debug_adapter.debug(f"Checking cell {source_cell.__hash__()} at position {source_cell_position}")
            clock_check_test_result = self.clock_checker.clock_check(source_cell_position,
                                                                     phi,
                                                                     self.parameters.get_value("phi_th"),
                                                                     lambda val, thr: val > thr)
            debug_adapter.debug(f"Clock Check test result is {clock_check_test_result}")
            if clock_check_test_result:
                to_remove.append(source_cell)
                debug_adapter.debug(f"Appended source cell {source_cell.__hash__()} at position "
                                    f"{source_cell_position} to the 'to_remove' list")

        self._remove_sources(to_remove)

    def _remove_sources(self, local_to_remove):
        # get root rank
        root = 0

        # share cells to remove among processes
        local_to_remove_list = comm.gather(local_to_remove, root)
        if rank == root:
            # remove sublists
            global_to_remove = [item for sublist in local_to_remove_list for item in sublist]
            # remove duplicates
            global_to_remove = list(set(global_to_remove))
        else:
            global_to_remove = None
        global_to_remove = comm.bcast(global_to_remove, root)

        # each process cancels the sources
        for source_cell in global_to_remove:
            self.source_map.remove_global_source(source_cell)

    def apply_sources(self, T: fenics.Function, V: fenics.FunctionSpace, is_V_sub_space, t):
        """
        Set the source points to 1. in the given field T.
        :param T:
        :param V:
        :param is_V_sub_space:
        :param t: current time
        :return:
        """
        # update time of expression function
        if type(self.expression_function) is RotationalExpressionFunction:
            self.expression_function.set_time(t)
        # get function space
        if not is_V_sub_space:
            # interpolate source field
            s_f = fenics.interpolate(SourcesField(self.source_map, self.parameters, self.expression_function),
                                     V)
            # assign s_f to T where s_f equals 1
            self._assign_values_to_vector(T, s_f)
        else:
            # collapse subspace
            V_collapsed = V.collapse()
            # interpolate source field
            s_f = fenics.interpolate(SourcesField(self.source_map, self.parameters, self.expression_function),
                                     V_collapsed)
            # create assigner to collapsed
            assigner_to_collapsed = fenics.FunctionAssigner(V_collapsed, V)
            # assign T to local variable T_temp
            T_temp = fenics.Function(V_collapsed)
            assigner_to_collapsed.assign(T_temp, T)
            # assign values to T_temp
            self._assign_values_to_vector(T_temp, s_f)
            # create inverse assigner
            assigner_to_sub = fenics.FunctionAssigner(V, V_collapsed)
            # assign T_temp to T
            assigner_to_sub.assign(T, T_temp)

    def _assign_values_to_vector(self, T, s_f):
        # get local values for T and source_field
        s_f_loc_values = s_f.vector().get_local()
        T_loc_values = T.vector().get_local()
        # change T value only where s_f is grater than 0
        where_s_f_is_over_zero = s_f_loc_values > 0.
        T_loc_values[where_s_f_is_over_zero] = \
            T_loc_values[where_s_f_is_over_zero] + s_f_loc_values[where_s_f_is_over_zero]
        T.vector().set_local(T_loc_values)
        T.vector().update_ghost_values()  # necessary, otherwise I get errors


class SourcesField(fenics.UserExpression):
    def __floordiv__(self, other):
        pass

    def __init__(self, source_map: SourceMap, parameters: Parameters, expression_function=None):
        super(SourcesField, self).__init__()
        self.source_map: SourceMap = source_map
        self.value_min = parameters.get_value("T_min")
        self.value_max = parameters.get_value("T_s")
        self.radius = parameters.get_value("R_c")
        if expression_function is None:
            self.expression_function = ConstantExpressionFunction(self.value_max)
        else:
            self.expression_function = expression_function

    def eval(self, values, x):
        point_value = self.value_min
        for source_cell in self.source_map.get_local_source_cells():
            if source_cell.get_distance(x) <= self.radius:
                point_value = self.expression_function.get_point_value_at_source_cell(source_cell)
                break
        values[0] = point_value

    def value_shape(self):
        return ()


class ClockChecker:
    def __init__(self, mesh_wrapper: fu.RectangleMeshWrapper, radius, start_point="east"):
        self.radius = radius
        angles = np.arange(0, 2 * np.pi, (2 * np.pi) / 30)
        circle_vectors = [np.array([radius * np.cos(angle), radius * np.sin(angle)]) for angle in angles]
        if start_point == "east":
            circle_vectors.sort(key=lambda x: x[0])
        elif start_point == "west":
            circle_vectors.sort(key=lambda x: x[0], reverse=True)
        else:
            raise ValueError("ClockChecker can be just 'east' or 'west' type")
        self.circle_vectors = circle_vectors
        self.mesh_wrapper = mesh_wrapper

    def clock_check(self, point, function: fenics.Function, threshold, condition):
        # cast point to the right type
        if type(point) is fenics.Point:
            point = np.array([point.array()[i] for i in range(self.mesh_wrapper.get_dim())])
        for vector in self.circle_vectors:
            for scale in np.arange(1., 0., -(1 / 20)):
                ppv = point + (scale * vector)
                debug_adapter.debug(f"Checking point {ppv}")
                if self.mesh_wrapper.is_inside_local_mesh(fenics.Point(ppv)):
                    debug_adapter.debug(f"Point {ppv} is inside local mesh.")
                    if condition(function(list(ppv)), threshold):  # ppv translated to List to avoid FutureWarning
                        return True
                else:
                    pass
        return False


class RotationalExpressionFunction:
    """Defines an angiogenic expression function which reproduces a spiral activation of the source cells around a
    center"""
    def __init__(self, rotational_expression_function_parameters):
        """
        :param rotational_expression_function_parameters: parameters of the expression function
        """
        x_center = rotational_expression_function_parameters["x_center"]
        y_center = rotational_expression_function_parameters["y_center"]
        self.center = np.array([x_center, y_center])
        self.radius = rotational_expression_function_parameters["radius"]
        self.time = 0.
        self.period = rotational_expression_function_parameters["period"]
        self.reference_point = self.center + np.array([self.radius, 0.])
        self.mean_value = rotational_expression_function_parameters["mean_value"]
        self.amplitude = rotational_expression_function_parameters["amplitude"]
        self.value_out_of_rotational_loop = rotational_expression_function_parameters["value_out_of_rotational_loop"]

    def get_point_value_at_source_cell(self, source_cell):
        # get position of the source cell
        cell_position = source_cell.get_position()

        # if cell is inside the rotational loop
        if source_cell.get_distance(self.center) <= self.radius:
            # evaluate cos phase based on position with carnot theorem
            if np.allclose(cell_position, self.center):
                cos_phase = 1.
            else:
                cos_phase = ((np.linalg.norm(cell_position - self.center) ** 2) +
                             (np.linalg.norm(self.reference_point - self.center) ** 2) -
                             (np.linalg.norm(cell_position - self.reference_point) ** 2)) / \
                            (2 *
                             np.linalg.norm(cell_position - self.center) *
                             np.linalg.norm(self.reference_point - self.center))

            # evaluate phase
            if cell_position[1] < self.center[1]:
                phase = - np.arccos(cos_phase)
            else:
                phase = np.arccos(cos_phase)

            # set point value
            point_value = self.mean_value + self.amplitude * np.sin(2 * np.pi * (self.time / self.period) + phase)
        else:
            point_value = self.value_out_of_rotational_loop
        return point_value

    def set_time(self, t):
        self.time = t


class ConstantExpressionFunction:
    def __init__(self, constant_value):
        self.constant_value = constant_value

    def get_point_value_at_source_cell(self, source_cell):
        return self.constant_value


def sources_in_circle_points(center: np.ndarray, circle_radius, cell_radius):
    # initialize source points
    source_points = [center]
    # eval cell diameter
    cell_diameter = 2 * cell_radius
    # floor radius
    circle_radius = int(np.floor(circle_radius))
    # evaluate all the radiuses along the circle radius
    radius_array = np.arange(cell_diameter, circle_radius, cell_diameter)
    # set cell positions
    for radius in radius_array:
        # evaluate number of cells in the current circle
        n_cells = int(np.floor((2 * np.pi * radius) / cell_diameter))
        # for each cell
        for k in range(n_cells):
            # eval cell position
            cell_position = np.array([center[0] + radius * np.cos(2 * np.pi * (k / n_cells)),
                                      center[1] + radius * np.sin(2 * np.pi * (k / n_cells))])
            # append it to source points
            source_points.append(cell_position)
    return source_points
