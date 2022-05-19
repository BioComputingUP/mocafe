"""
This module contains classes and methods to manage discrete angiogenic factor sources in Mocafe. This discrete agents
might represent hypoxic cells inducing angiogenesis, as presented by Travasso et al. :cite:`Travasso2011a`.

If you use this model in your research, remember to cite the original paper describing the model:

    Travasso, R. D. M., Poiré, E. C., Castro, M., Rodrguez-Manzaneque, J. C., & Hernández-Machado, A. (2011).
    Tumor angiogenesis and vascular patterning: A mathematical model. PLoS ONE, 6(5), e19989.
    https://doi.org/10.1371/journal.pone.0019989

For a use example see the :ref:`Angiogenesis <Angiogenesis 2D Demo>` and the
:ref:`Angiogenesis 3D <Angiogenesis 2D Demo>` demos.
"""
import sys
import types
import dolfinx.mesh
import dolfinx.fem
import random
import numpy as np
import logging
import mocafe.fenut.fenut as fu
from mpi4py import MPI
from mocafe.angie import base_classes
from mocafe.fenut.parameters import Parameters
from mocafe.fenut.log import InfoCsvAdapter, DebugAdapter

# Get MPI communicator and _rank to be used in the module
_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()

# configure _logger
_logger = logging.getLogger(__name__)
_info_adapter = InfoCsvAdapter(_logger, {"_rank": _rank, "module": __name__})  # mainly for process optimization
_debug_adapter = DebugAdapter(_logger, {"_rank": _rank, "module": __name__})  # all kinds of information


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
            position to generate an unique identifier of the cell.
        """
        super(SourceCell, self).__init__(point, creation_step)


class SourceMap:
    """
    Class representing the spatial map of the source cell positions. This class is responsible for keeping the position
    of each source cell at any point of the simulation, providing access to it to other objects and methods.
    """
    def __init__(self,
                 mesh: dolfinx.mesh.Mesh,
                 source_points: list,
                 parameters: Parameters):
        """
        inits a SourceMap, i.e. a class responsible for keeping the current position of each source cell at any point
        of the simulation, with a SourceCell in all the positions listed in source_points.

        :param mesh: the simulation mesh
        :param source_points: list of positions where to place the source cells.
        :param parameters: simulation parameters
        """
        # initialize global SourceCells list in the given points
        self.mesh = mesh
        self.local_box = self._build_local_box(parameters)
        global_source_points = source_points
        # sort global source point for distance from origin
        global_source_points.sort(key=lambda x: np.sqrt(sum(x**2)))
        self.global_source_cells = [SourceCell(point, 0) for point in global_source_points]
        self.local_source_cells = self._divide_source_cells()

    def _divide_source_cells(self):
        """
        INTERNAL USE
        Divides the source cells among the MPI process. Each process has to take care of the source cells inside its
        local box

        :return: the list of source cell which has to be handled by the current MPI process
        """
        competence_source_cells = []
        for source_cell in self.global_source_cells:
            position = source_cell.get_position()
            if self._is_in_local_box(position):
                competence_source_cells.append(source_cell)
        return competence_source_cells

    def _build_local_box(self, parameters: Parameters):
        """
        INTERNAL USE
        Builds the local box for the MPI process. The local box is a square spatial domain that is used to check
        if the source cells are near a blood vessel.

        :param parameters:
        :return:
        """
        d = parameters.get_value("d")
        return fu.build_local_box(self.mesh, d)

    def _is_in_local_box(self, position):
        """
        INTERNAL USE
        Determines if the given position is inside the local box.

        :param position: position to check
        :return: True if the position is inside the local box. False otherwise
        """
        return fu.is_in_local_box(self.local_box, position)

    def get_global_source_cells(self):
        """
        Get the global list of source cell (equal for each MPI process)

        :return: the global list of source cells
        """
        return self.global_source_cells

    def get_local_source_cells(self):
        """
        Get the local list of source cells (for the current MPI process)

        :return:
        """
        return self.local_source_cells

    def remove_global_source(self, source_cell: SourceCell):
        """
        Remove a source cell from the global source cell list. If the cell is part of the local source cells,
        it is removed also from that list.

        :param source_cell: the source cell to remove
        :return:
        """
        # remove from global list
        self.global_source_cells.remove(source_cell)
        _debug_adapter.debug(f"Removed source cell {source_cell.__hash__()} at position {source_cell.get_position()}"
                            f"from the global list")
        # if in local, remove from local list too
        if source_cell in self.local_source_cells:
            self.local_source_cells.remove(source_cell)
            _debug_adapter.debug(
                f"Removed source cell {source_cell.__hash__()} at position {source_cell.get_position()}"
                f"from the local list")


class RandomSourceMap(SourceMap):
    """
    A SourceMap of randomly distributed sources in a given spatial domain
    """
    def __init__(self,
                 mesh: dolfinx.mesh.Mesh,
                 n_sources: int,
                 parameters: Parameters,
                 where: types.FunctionType or types.LambdaType):
        """
        inits a SourceMap of randomly distributed source cell in the mesh. One can specify the number of source
        cells to be initialized (argument ``n_sources``) and where the source cells will be placed
        (argument ``where``). The argument where can be:

        - None; in this case, any point in the mesh can be picked as a rondom source
        - a Python function; in this case, only the points for which the function returns True can be picked as
            random sources. The given Python function must have a single input of the type fenics.Point.
        - a mesh.cpp.Geometry (e.g. mshr.Rectangle or mshr.Circle) or a fenics.SubDomain object; in this case, the
            point will be selected the defined space.

        :param mesh: the given mesh
        :param n_sources: the number of sources to select
        :param parameters: simulation parameters
        :param where: where to place the randomly distributed source cells. Default is None.
        """
        # check type of function
        if isinstance(where, types.FunctionType) or isinstance(where, types.LambdaType):
            where_fun = where
        else:
            raise TypeError(f"Argument 'where' can be only of type {types.FunctionType}, {types.LambdaType} \n"
                            f"Detected {type(where)} instead")

        # check type of n_sources
        if not isinstance(n_sources, int):
            raise TypeError(f"n_sources must be of type int. Found {type(n_sources)} instead.")

        # get randomly distributed mesh point
        self.mesh = mesh
        global_source_points_list = self._pick_n_global_vertices_where_asked(n_sources, where_fun)
        # inits source map
        super(RandomSourceMap, self).__init__(mesh,
                                              global_source_points_list,
                                              parameters)

    def _pick_n_global_vertices_where_asked(self, n_points, where_fun):
        # get global vertex index (unique identifier of coordinates for all the procs)
        global_vertex_indices = self.mesh.geometry.input_global_indices
        # get local mesh coordinates
        lmc = self.mesh.geometry.x
        # extract the index of the pickable points
        global_pickable_vertex_index = \
            [index for index, coordinate in zip(global_vertex_indices, lmc) if where_fun(coordinate)]
        # gather lmc in proc 0
        lmc_arrays = _comm.gather(lmc, 0)
        # gather indices in proc 0
        vertex_maps = _comm.gather(global_vertex_indices, 0)
        # gather pickable coordinats indices in proc 0
        pickable_vertex_maps = _comm.gather(global_pickable_vertex_index, 0)

        if _rank == 0:
            # init global coordinates array
            n_global_vertices = self.mesh.topology.index_map(0).size_global
            global_coordinates = np.zeros((n_global_vertices, lmc.shape[1]))
            # get global pickable coordinates
            for coordinates, indices in zip(lmc_arrays, vertex_maps):
                global_coordinates[indices] = coordinates
            # get pickable points indices (removing duplicates)
            pickable_vertex_indices = list(set(fu.flatten_list_of_lists(pickable_vertex_maps)))
            # select pickable points
            pickable_points = global_coordinates[pickable_vertex_indices]
            # convert in list
            pickable_points = list(pickable_points)
            # pick n of them (if available)
            n_pickable_points = len(pickable_points)
            if n_pickable_points <= n_points:
                _logger.warning(f"The mesh looks too small for selecting {n_points} random source, since it was asked"
                               f"to select {n_points} among {n_pickable_points}"
                               f"pickable points. Returning all available pickable points.")
                global_sources_coordinates = pickable_points
            else:
                global_sources_coordinates = random.sample(pickable_points, n_points)
        else:
            global_sources_coordinates = None
        global_sources_coordinates = _comm.bcast(global_sources_coordinates, 0)
        return global_sources_coordinates


class SourcesManager:
    """
    Class representing the manager of the position of the source cells. This class takes care of removing the source
    cells when they are near the blood vessels and of translating the source cell map in a FEniCS phase field function
    """
    def __init__(self, source_map: SourceMap,
                 mesh: dolfinx.mesh.Mesh,
                 parameters: Parameters):
        """
        inits a source cells manager for a given source map.

        :param source_map: the source map (i.e. the source cells) to manage
        :param mesh: the mesh
        :param parameters: the simulation parameters
        """
        self.source_map = source_map
        self.mesh = mesh
        self.parameters: Parameters = parameters
        self.value_min = parameters.get_value("T_min")
        self.value_max = parameters.get_value("T_s")
        if parameters.is_parameter("d") and parameters.is_value_present("d"):
            self.default_clock_checker = base_classes.ClockChecker(mesh, parameters.get_value("d"))
            self.default_clock_checker_is_present = True
        elif not parameters.is_parameter("d"):
            _logger.debug("Reference for the parameter 'd' not found. Can't init the default clock checker.")
            self.default_clock_checker = None
            self.default_clock_checker_is_present = False
        elif not parameters.is_value_present("d"):
            _logger.debug("The parameter 'd' is present in the parameters object but the value is not set. "
                         "Can't init the default clock checker.")
            self.default_clock_checker = None
            self.default_clock_checker_is_present = False

    def remove_sources_near_vessels(self, c: dolfinx.fem.Function, **kwargs):
        """
        Removes the source cells near the blood vessels

        :param c: blood vessel field
        :return:
        """
        # prepare list of cells to remove
        to_remove = []
        _debug_adapter.debug(f"Starting to remove source cells")

        # if distance is specified
        if "d" in kwargs.keys():
            clock_checker = base_classes.ClockChecker(self.mesh, kwargs["d"])
        else:
            if self.default_clock_checker_is_present:
                clock_checker = self.default_clock_checker
            else:
                raise RuntimeError("The min distance for removing the source cells has not be defined. "
                                   "Pass it in the constructor of the class through the parameters object or "
                                   "input it to the method using the key 'd'")

        for source_cell in self.source_map.get_local_source_cells():
            source_cell_position = source_cell.get_position()
            _debug_adapter.debug(f"Checking cell {source_cell.__hash__()} at position {source_cell_position}")
            clock_check_test_result = clock_checker.clock_check(source_cell_position,
                                                                c,
                                                                self.parameters.get_value("phi_th"),
                                                                lambda val, thr: val > thr)
            _debug_adapter.debug(f"Clock Check test result is {clock_check_test_result}")
            # if the clock test is positive, add the source cells in the list of the cells to remove
            if clock_check_test_result:
                to_remove.append(source_cell)
                _debug_adapter.debug(f"Appended source cell {source_cell.__hash__()} at position "
                                    f"{source_cell_position} to the 'to_remove' list")

        self._remove_sources(to_remove)

    def _remove_sources(self, local_to_remove):
        """
        INTERNAL USE
        Given a list of source cells to remove, it takes care that those cells will be removed from the SourceMap

        :param local_to_remove:
        :return:
        """
        # get root _rank
        root = 0

        # share cells to remove among processes
        local_to_remove_list = _comm.gather(local_to_remove, root)
        if _rank == root:
            # remove sublists
            global_to_remove = [item for sublist in local_to_remove_list for item in sublist]
            # remove duplicates
            global_to_remove = list(set(global_to_remove))
        else:
            global_to_remove = None
        global_to_remove = _comm.bcast(global_to_remove, root)

        # each process cancels the sources
        for source_cell in global_to_remove:
            self.source_map.remove_global_source(source_cell)

    def apply_sources(self, af: dolfinx.fem.Function):
        """
        Apply the sources at the current time to the angiogenic factor field af, respecting the expression function.

        :param af: FEniCS function representing the angiogenic factor
        :return: nothing

        """
        # create a field for sources copying af
        s_f = af.copy()
        # interpolate the custom sources expression on it
        s_f_exp = ConstantSourcesField(self.source_map, self.parameters)
        s_f.interpolate(s_f_exp.eval)
        # overwrite af only where s_f is not nan
        s_f_not_nan = ~np.isnan(np.array(s_f.vector.array))
        af.vector.array[s_f_not_nan] = s_f.vector.array[s_f_not_nan]

        # # get Function Space of af
        # V_af = af.function_space
        #
        # # check if V_af is sub space
        # if V_af.component():
        #     is_V_sub_space = True
        # else:
        #     is_V_sub_space = False

        # # interpolate according to V_af
        # if not is_V_sub_space:
        #     # interpolate source field
        #     s_f = dolfinx.fem.Function(V_af)
        #     s_f.interpolate(ConstantSourcesField(self.source_map, self.parameters))
        #     # assign s_f to T where s_f equals 1
        #     self._assign_values_to_vector(af, s_f)
        # else:
        #     # collapse subspace
        #     V_collapsed = V_af.collapse()[0]
        #     # interpolate source field
        #     s_f = dolfinx.fem.Function(V_collapsed)
        #     s_f.interpolate(ConstantSourcesField(self.source_map, self.parameters))
        #     # create assigner to collapsed
        #     assigner_to_collapsed = fenics.FunctionAssigner(V_collapsed, V_af)
        #     # assign T to local variable T_temp
        #     T_temp = fenics.Function(V_collapsed)
        #     assigner_to_collapsed.assign(T_temp, af)
        #     # assign values to T_temp
        #     self._assign_values_to_vector(T_temp, s_f)
        #     # create inverse assigner
        #     assigner_to_sub = fenics.FunctionAssigner(V_af, V_collapsed)
        #     # assign T_temp to T
        #     assigner_to_sub.assign(af, T_temp)

    # def _assign_values_to_vector(self, af, s_f):
    #     """
    #     INTERNAL USE
    #     Assign the positive values of the source field function to the angiogenic factor field.
    #
    #     :param af: angiogenic factor function
    #     :param s_f: source field function
    #     :return: nothing
    #     """
    #     # get local values for T and source_field
    #     s_f_loc_values = s_f.vector().get_local()
    #     T_loc_values = af.vector().get_local()
    #     # change T value only where s_f is grater than 0
    #     where_s_f_is_over_zero = s_f_loc_values > 0.
    #     T_loc_values[where_s_f_is_over_zero] = \
    #         T_loc_values[where_s_f_is_over_zero] + s_f_loc_values[where_s_f_is_over_zero]
    #     af.vector().set_local(T_loc_values)
    #     af.vector().update_ghost_values()  # necessary, otherwise I get errors


class ConstantSourcesField:
    """
    Custom Expression representing the distribution of the angiogenic factor expressed by the source cells.
    """

    def __init__(self, source_map: SourceMap, parameters: Parameters):
        """
        inits a SourceField for the given SourceMap, in respect of the simulation parameters and of the expression
        function

        :param source_map:
        :param parameters:
        """
        super(ConstantSourcesField, self).__init__()
        self.sources_positions = [source_cell.get_position() for source_cell in source_map.get_local_source_cells()]
        self.sources_positions_not_empty = len(self.sources_positions) != 0
        self.value_max = parameters.get_value("T_s")
        self.radius = parameters.get_value("R_c")

    def eval(self, x):
        # if there are source cells
        if self.sources_positions_not_empty:
            # create a list to store booleans
            is_inside_any_source = None
            # for each source check if there are points inside
            for source_position in self.sources_positions:
                is_inside_current_source = np.sum((x.T - source_position) ** 2, axis=1) < (self.radius ** 2)
                if is_inside_any_source is None:  # True at first iteration
                    is_inside_any_source = is_inside_current_source
                else:
                    is_inside_any_source = is_inside_any_source | is_inside_current_source
            # return max value if the point is inside, nan everywhere else
            return np.where(is_inside_any_source, self.value_max, np.nan)
        else:
            nan_array = np.empty(x.shape[1])
            nan_array[:] = np.nan
            return nan_array


# class AFExpressionFunction:
#     def __init__(self):
#         pass
#
#
# class RotationalAFExpressionFunction(AFExpressionFunction):
#     """
#     Defines an angiogenic factor expression function which reproduces a spiral activation of the source cells around a
#     center
#     """
#     def __init__(self, rotational_expression_function_parameters):
#         """
#         :param rotational_expression_function_parameters: parameters of the expression function
#         """
#         super(RotationalAFExpressionFunction, self).__init__()
#         x_center = rotational_expression_function_parameters["x_center"]
#         y_center = rotational_expression_function_parameters["y_center"]
#         self.center = np.array([x_center, y_center])
#         self.radius = rotational_expression_function_parameters["radius"]
#         self.time = 0.
#         self.period = rotational_expression_function_parameters["period"]
#         self.reference_point = self.center + np.array([self.radius, 0.])
#         self.mean_value = rotational_expression_function_parameters["mean_value"]
#         self.amplitude = rotational_expression_function_parameters["amplitude"]
#         self.value_out_of_rotational_loop = rotational_expression_function_parameters["value_out_of_rotational_loop"]
#
#     def get_point_value_at_source_cell(self, source_cell):
#         """
#         Returns the concentration of the angiogenic factor expressed for the given source cell.
#
#         :param source_cell: the source cell considered
#         :return:
#         """
#         # get position of the source cell
#         cell_position = source_cell.get_position()
#
#         # if cell is inside the rotational loop
#         if source_cell.get_distance(self.center) <= self.radius:
#             # evaluate cos phase based on position with carnot theorem
#             if np.allclose(cell_position, self.center):
#                 cos_phase = 1.
#             else:
#                 cos_phase = ((np.linalg.norm(cell_position - self.center) ** 2) +
#                              (np.linalg.norm(self.reference_point - self.center) ** 2) -
#                              (np.linalg.norm(cell_position - self.reference_point) ** 2)) / \
#                             (2 *
#                              np.linalg.norm(cell_position - self.center) *
#                              np.linalg.norm(self.reference_point - self.center))
#
#             # evaluate phase
#             if cell_position[1] < self.center[1]:
#                 phase = - np.arccos(cos_phase)
#             else:
#                 phase = np.arccos(cos_phase)
#
#             # set point value
#             point_value = self.mean_value + self.amplitude * np.sin(2 * np.pi * (self.time / self.period) + phase)
#         else:
#             point_value = self.value_out_of_rotational_loop
#         return point_value
#
#     def set_time(self, t):
#         self.time = t


# class ConstantAFExpressionFunction(AFExpressionFunction):
#     """
#     Defines an angiogenic factor expression where each source cell has a constant angiogenic factor expression
#     """
#     def __init__(self, constant_value):
#         super(ConstantAFExpressionFunction, self).__init__()
#         self.constant_value = constant_value
#
#     def get_point_value_at_source_cell(self, source_cell):
#         """
#         Returns the concentration of the angiogenic factor expressed for the given source cell.
#
#         :param source_cell: the source cell considered
#         :return:
#         """
#         return self.constant_value


def sources_in_circle_points(center: np.ndarray, circle_radius, cell_radius):
    """
    Generate the points where to place the source cells to place the source cells in a circle. The circle is full of
    source cells.

    :param center: center of the circle
    :param circle_radius: radius of the circle
    :param cell_radius: radius of the cells
    :return: the list of source cells positions
    """
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
