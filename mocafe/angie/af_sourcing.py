"""
This module contains classes and methods to manage discrete angiogenic factor sources in ``mocafe``. More precisely,
it provides useful tools to create cells (described as circles with a given radius) which act as angiogenic factor
sources, inducing angiogenesis.

This module have been implemented to reproduce the results given by the angiogenesis model presented by Travasso et al.
:cite:`Travasso2011a`, where they where testing their model using multiple sources in random positions.
"""

import types
import fenics
import random
import mshr.cpp
import numpy as np
import logging
import mocafe.fenut.fenut as fu
from mocafe.angie import base_classes
from mocafe.fenut.parameters import Parameters
from mocafe.fenut.log import InfoCsvAdapter, DebugAdapter

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
        position to generate an unique identifier of the cell.
        """
        super(SourceCell, self).__init__(point, creation_step)


class SourceMap:
    """
    Class representing the spatial map of the source cell positions. This class is responsible for keeping the position
    of each source cell at any point of the simulation, providing access to it to other objects and methods.
    """
    def __init__(self,
                 mesh_wrapper: fu.MeshWrapper,
                 source_points: list,
                 parameters: Parameters):
        """
        inits a SourceMap, i.e. a class responsible for keeping the current position of each source cell at any point
        of the simulation, with a SourceCell in all the positions listed in source_points.

        :param mesh_wrapper: the mesh wrapper of the simulation mesh
        :param source_points: list of positions where to place the source cells.
        :param parameters: simulation parameters
        """
        # initialize global SourceCells list in the given points
        self.mesh_wrapper = mesh_wrapper  # todo:RMW
        self.local_box = self._build_local_box(parameters)
        global_source_points = source_points
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
        return fu.build_local_box(self.mesh_wrapper.get_local_mesh(), d)  # todo:RMW can be sobsitituted with mesh

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
        debug_adapter.debug(f"Removed source cell {source_cell.__hash__()} at position {source_cell.get_position()}"
                            f"from the global list")
        # if in local, remove from local list too
        if source_cell in self.local_source_cells:
            self.local_source_cells.remove(source_cell)
            debug_adapter.debug(
                f"Removed source cell {source_cell.__hash__()} at position {source_cell.get_position()}"
                f"from the local list")


class RandomSourceMap(SourceMap):
    """
    A SourceMap of randomly distributed sources in a given spatial domain
    """
    def __init__(self,
                 mesh_wrapper: fu.MeshWrapper,
                 n_sources: int,
                 parameters: fenics.Parameters,
                 where: types.FunctionType or mshr.cpp.CSGGeometry or fenics.SubDomain or None = None):
        """
        inits a SourceMap of randomly distributed source cell in the mesh. One can specify the number of source
        cells to be initialized (argument ``n_sources``) and where the source cells will be placed
        (argument ``where``). The argument where can be:

        - None; in this case, any point in the mesh can be picked as a rondom source
        - a Python function; in this case, only the points for which the function returns True can be picked as
            random sources. The given Python function must have a single input of the type fenics.Point.
        - a mesh.cpp.Geometry (e.g. mshr.Rectangle or mshr.Circle) or a fenics.SubDomain object; in this case, the
            point will be selected the defined space.

        :param mesh_wrapper: the mesh wrapper for the given mesh
        :param n_sources: the number of sources to select
        :param parameters: simulation parameters
        :param where: where to place the randomly distributed source cells. Default is None.
        """
        # compute random points
        if isinstance(where, types.FunctionType):
            where_fun = where
        elif isinstance(where, mshr.cpp.CSGGeometry) \
                or issubclass(type(where), mshr.cpp.CSGGeometry):
            where_fun = where.inside
        else:
            raise TypeError(f"Argument 'where' can be only of type {types.FunctionType}, {mshr.cpp.CSGGeometry},"
                            f"{fenics.SubDomain} or None. "
                            f"Detected {type(where)} instead")

        # get randomly distributed mesh point
        self.mesh_wrapper = mesh_wrapper  # todo:RMW
        global_source_points_list = self._get_randomy_sorted_mesh_points(n_sources, where_fun)
        # inits source map
        super(RandomSourceMap, self).__init__(mesh_wrapper,
                                              global_source_points_list,
                                              parameters)

    def _get_randomy_sorted_mesh_points(self, n_sources: int, where_fun):
        """
        INTERNAL USE
        Return the source points selected randomly and sorted along the x axis

        :param n_sources: number of sources to select
        :param where_fun: the function to use to determine if a point can be picked or not
        """
        # get comm size
        n_procs = comm.Get_size()
        # define root proc
        root = 0
        # get global coordinates
        global_coords = self.mesh_wrapper.get_global_mesh().coordinates()  # todo:RMW I can gather global coords without using mesh_wrapper
        # divide coordinates among processes
        if rank == root:
            coords_chunks_list = fu.divide_in_chunks(global_coords, n_procs)
        else:
            coords_chunks_list = None
        local_coords_chunk = comm.scatter(coords_chunks_list, root)
        # if where_fun is given
        if where_fun is not None:
            # get the pickable points
            pickable_points = [point for point in local_coords_chunk if where_fun(fenics.Point(point))]
        else:
            # else all points are pickable
            pickable_points = [point for point in local_coords_chunk]
        # compute n local sources
        n_pickable_points = len(pickable_points)
        local_n_sources = self._compute_local_n_sources(n_sources, n_pickable_points)
        # pick the source point randomly
        if n_pickable_points <= local_n_sources:
            logger.warning(f"The mesh looks too small for selecting {n_sources} random source, since it was asked"
                           f"to select {local_n_sources} among {n_pickable_points}"
                           f"local pickable points. Returning all available pickable points.")
            local_sources_array = pickable_points
        else:
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

    def _compute_local_n_sources(self, global_n_sources, n_local_pickable_points):
        """
        INTERNAL USE
        Computes the number of sources to randomly select for the local process, considering that the number
        of pickable points may differ between different processes and that the global number of sources is fixed.

        :param global_n_sources: number of global sources
        :param n_local_pickable_points: number of pickable points
        """
        root = 0
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


class SourcesManager:
    """
    Class representing the manager of the position of the source cells. This class takes care of removing the source
    cells when they are near the blood vessels and of translating the source cell map in a FEniCS phase field function
    """
    def __init__(self, source_map: SourceMap,
                 mesh_wrapper: fu.MeshWrapper,
                 parameters: Parameters):
        """
        inits a source cells manager for a given source map.

        :param source_map: the source map (i.e. the source cells) to manage
        :param mesh_wrapper: the mesh wrapper
        :param parameters: the simulation parameters
        """
        self.source_map = source_map
        self.mesh_wrapper = mesh_wrapper  # todo:RMW
        self.parameters: Parameters = parameters
        if parameters.is_parameter("d") and parameters.is_value_present("d"):
            self.default_clock_checker = ClockChecker(mesh_wrapper, parameters.get_value("d"))  # todo:RMW
            self.default_clock_checker_is_present = True
        elif not parameters.is_parameter("d"):
            logger.debug("Reference for the parameter 'd' not found. Can't init the default clock checker.")
            self.default_clock_checker = None
            self.default_clock_checker_is_present = False
        elif not parameters.is_value_present("d"):
            logger.debug("The parameter 'd' is present in the parameters object but the value is not set. "
                         "Can't init the default clock checker.")
            self.default_clock_checker = None
            self.default_clock_checker_is_present = False

        self.default_af_expression_function = ConstantAFExpressionFunction(self.parameters.get_value("T_s"))

    def remove_sources_near_vessels(self, c: fenics.Function, **kwargs):
        """
        Removes the source cells near the blood vessels

        :param c: blood vessel field
        :return:
        """
        # prepare list of cells to remove
        to_remove = []
        debug_adapter.debug(f"Starting to remove source cells")
        # if distance is specified
        if "min_distance" in kwargs.keys():
            clock_checker = ClockChecker(self.mesh_wrapper, kwargs["d"])  # todo:RMW
        else:
            if self.default_clock_checker_is_present:
                clock_checker = self.default_clock_checker
            else:
                raise RuntimeError("The min distance for removing the source cells has not be defined. "
                                   "Pass it in the constructor of the class through the parameters object or "
                                   "input it to the method using the key 'd'")

        for source_cell in self.source_map.get_local_source_cells():
            source_cell_position = source_cell.get_position()
            debug_adapter.debug(f"Checking cell {source_cell.__hash__()} at position {source_cell_position}")
            clock_check_test_result = clock_checker.clock_check(source_cell_position,
                                                                c,
                                                                self.parameters.get_value("phi_th"),
                                                                lambda val, thr: val > thr)
            debug_adapter.debug(f"Clock Check test result is {clock_check_test_result}")
            # if the clock test is positive, add the source cells in the list of the cells to remove
            if clock_check_test_result:
                to_remove.append(source_cell)
                debug_adapter.debug(f"Appended source cell {source_cell.__hash__()} at position "
                                    f"{source_cell_position} to the 'to_remove' list")

        self._remove_sources(to_remove)

    def _remove_sources(self, local_to_remove):
        """
        INTERNAL USE
        Given a list of source cells to remove, it takes care that those cells will be removed from the SourceMap

        :param local_to_remove:
        :return:
        """
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

    def apply_sources(self, af: fenics.Function,
                      af_expression_function=None):
        """
        Apply the sources at the current time to the angiogenic factor field af, respecting the expression function.

        :param af: FEniCS function representing the angiogenic factor
        :param af_expression_function: an object of type AFExpressionFunction which tells this method the value of
        the angiogenic factor inside the cells.
        :return: nothing

        """
        current_af_expression_function = self.default_af_expression_function \
            if af_expression_function is None else af_expression_function
        if not issubclass(type(current_af_expression_function), AFExpressionFunction):
            raise TypeError(f"af expression function is of type {type(current_af_expression_function)}. "
                            f"Must be a subclass of AFExpressionFunction")
        # get Function Space of af
        V_af = af.function_space()
        # check if V_af is sub space
        try:
            V_af.collapse()
            is_V_sub_space = True
        except RuntimeError:
            is_V_sub_space = False

        # get function space
        if not is_V_sub_space:
            # interpolate source field
            s_f = fenics.interpolate(SourcesField(self.source_map, self.parameters, current_af_expression_function),
                                     V_af)
            # assign s_f to T where s_f equals 1
            self._assign_values_to_vector(af, s_f)
        else:
            # collapse subspace
            V_collapsed = V_af.collapse()
            # interpolate source field
            s_f = fenics.interpolate(SourcesField(self.source_map, self.parameters, current_af_expression_function),
                                     V_collapsed)
            # create assigner to collapsed
            assigner_to_collapsed = fenics.FunctionAssigner(V_collapsed, V_af)
            # assign T to local variable T_temp
            T_temp = fenics.Function(V_collapsed)
            assigner_to_collapsed.assign(T_temp, af)
            # assign values to T_temp
            self._assign_values_to_vector(T_temp, s_f)
            # create inverse assigner
            assigner_to_sub = fenics.FunctionAssigner(V_af, V_collapsed)
            # assign T_temp to T
            assigner_to_sub.assign(af, T_temp)

    def _assign_values_to_vector(self, af, s_f):
        """
        INTERNAL USE
        Assign the positive values of the source field function to the angiogenic factor field.

        :param af: angiogenic factor function
        :param s_f: source field function
        :return: nothing
        """
        # get local values for T and source_field
        s_f_loc_values = s_f.vector().get_local()
        T_loc_values = af.vector().get_local()
        # change T value only where s_f is grater than 0
        where_s_f_is_over_zero = s_f_loc_values > 0.
        T_loc_values[where_s_f_is_over_zero] = \
            T_loc_values[where_s_f_is_over_zero] + s_f_loc_values[where_s_f_is_over_zero]
        af.vector().set_local(T_loc_values)
        af.vector().update_ghost_values()  # necessary, otherwise I get errors


class SourcesField(fenics.UserExpression):
    """
    FEniCS Expression representing the distribution of the angiogenic factor expressed by the source cells.
    """
    def __floordiv__(self, other):
        pass

    def __init__(self, source_map: SourceMap, parameters: Parameters, af_expression_function):
        """
        inits a SourceField for the given SourceMap, in respect of the simulation parameters and of the expression
        function

        :param source_map:
        :param parameters:
        :param af_expression_function:
        """
        super(SourcesField, self).__init__()
        self.source_map: SourceMap = source_map
        self.value_min = parameters.get_value("T_min")
        self.value_max = parameters.get_value("T_s")
        self.radius = parameters.get_value("R_c")
        self.af_expression_function = af_expression_function

    def eval(self, values, x):
        point_value = self.value_min
        for source_cell in self.source_map.get_local_source_cells():
            if source_cell.get_distance(x) <= self.radius:
                point_value = self.af_expression_function.get_point_value_at_source_cell(source_cell)
                break
        values[0] = point_value

    def value_shape(self):
        return ()


class ClockChecker:
    """
    Class representing a clock checker, i.e. an object that checks if a given condition is met in the surroundings of
    a point of the mesh.
    """
    def __init__(self, mesh_wrapper: fu.MeshWrapper, radius, start_point="east"):
        """
        inits a ClockChecker, which will check if a condition is met inside the given radius

        :param mesh_wrapper: mesh wrapper
        :param radius: radius where to check if the condition is met
        :param start_point: starting point where to start checking. If the point is `east`, the clock checker will
        start checking from the point with the lower value of x[0]; if the point is `west` the clock cheker will start
        from the point with higher value of x[0]
        """
        self.radius = radius
        # define vectors to check
        angles = np.arange(0, 2 * np.pi, (2 * np.pi) / 30)
        circle_vectors = [np.array([radius * np.cos(angle), radius * np.sin(angle)]) for angle in angles]
        if start_point == "east":
            circle_vectors.sort(key=lambda x: x[0])
        elif start_point == "west":
            circle_vectors.sort(key=lambda x: x[0], reverse=True)
        else:
            raise ValueError("ClockChecker can be just 'east' or 'west' type")
        self.circle_vectors = circle_vectors
        self.mesh_wrapper = mesh_wrapper  # todo:RMW

    def clock_check(self, point, function: fenics.Function, threshold, condition):
        """
        clock-check the given function in the surrounding of the given point

        :param point: center of the clock-check
        :param function: function to check
        :param threshold: threshold that the function has to surpass
        :param condition: lambda function representing the condition to be met
        :return: True if the condition is met; False otherwise
        """
        # cast point to the right type
        if type(point) is fenics.Point:
            point = np.array([point.array()[i] for i in range(self.mesh_wrapper.get_dim())])  # todo:RMW can be replaced with get geometric dimension
        for vector in self.circle_vectors:
            for scale in np.arange(1., 0., -(1 / 20)):
                ppv = point + (scale * vector)
                debug_adapter.debug(f"Checking point {ppv}")
                if self.mesh_wrapper.is_inside_local_mesh(fenics.Point(ppv)):  # todo:RMW can be replaced with mesh
                    debug_adapter.debug(f"Point {ppv} is inside local mesh.")
                    if condition(function(list(ppv)), threshold):  # ppv translated to List to avoid FutureWarning
                        return True
                else:
                    pass
        return False


class AFExpressionFunction:
    def __init__(self):
        pass


class RotationalAFExpressionFunction(AFExpressionFunction):
    """
    Defines an angiogenic factor expression function which reproduces a spiral activation of the source cells around a
    center
    """
    def __init__(self, rotational_expression_function_parameters):
        """
        :param rotational_expression_function_parameters: parameters of the expression function
        """
        super(RotationalAFExpressionFunction, self).__init__()
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
        """
        Returns the concentration of the angiogenic factor expressed for the given source cell.

        :param source_cell: the source cell considered
        :return:
        """
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


class ConstantAFExpressionFunction(AFExpressionFunction):
    """
    Defines an angiogenic factor expression where each source cell has a constant angiogenic factor expression
    """
    def __init__(self, constant_value):
        super(ConstantAFExpressionFunction, self).__init__()
        self.constant_value = constant_value

    def get_point_value_at_source_cell(self, source_cell):
        """
        Returns the concentration of the angiogenic factor expressed for the given source cell.

        :param source_cell: the source cell considered
        :return:
        """
        return self.constant_value


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
