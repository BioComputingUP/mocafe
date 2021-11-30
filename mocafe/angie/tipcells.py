import fenics
import numpy as np
import mocafe.fenut.fenut as fu
from mocafe.angie import af_sourcing
from mocafe.angie.base_classes import BaseCell
import random
import logging
from mocafe.fenut.parameters import Parameters
from mocafe.fenut.log import InfoCsvAdapter, DebugAdapter

"""
Test script for tip cell activation

Franco Pradelli
20 May 2021
"""

# get rank
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# configure logger
logger = logging.getLogger(__name__)
info_adapter = InfoCsvAdapter(logger, {"rank": rank, "module": __name__})
debug_adapter = DebugAdapter(logger, {"rank": rank, "module": __name__})


class TipCell(BaseCell):
    def __init__(self, position, radius, creation_step):
        super(TipCell, self).__init__(position, creation_step)
        self.radius = radius

    def move(self, new_position):
        self.position = new_position

    def get_radius(self):
        return self.radius

    def is_point_inside(self, x):
        return self.get_distance(x) <= self.radius


class TipCellsField(fenics.UserExpression):
    def __floordiv__(self, other):
        pass

    def __init__(self, parameters: Parameters):
        super(TipCellsField, self).__init__()
        self.alpha_p = parameters.get_value("alpha_p")
        self.T_p = parameters.get_value("T_p")
        self.phi_min = parameters.get_value("phi_min")
        self.phi_max = parameters.get_value("phi_max")
        self.tip_cells = []
        self.velocities = []
        self.T_values = []

    def add_tip_cell(self, tip_cell, velocity, T_at_point):
        self.tip_cells.append(tip_cell)
        self.velocities.append(velocity)
        self.T_values.append(self.T_p if T_at_point > self.T_p else T_at_point)

    def eval(self, values, x):
        point_value = self.phi_min
        for tip_cell, velocity, T_value in zip(self.tip_cells, self.velocities, self.T_values):
            if tip_cell.is_point_inside(x):
                phi_c = ((self.alpha_p * T_value * np.pi * tip_cell.get_radius()) / (2 * np.linalg.norm(velocity))) + 1
                point_value = phi_c
                break
        values[0] = point_value

    def value_shape(self):
        return ()


class TipCellManager:
    def __init__(self, mesh_wrapper: fu.RectangleMeshWrapper,
                 parameters: Parameters):
        self.global_tip_cells_list = []
        self.local_tip_cells_list = []
        self.mesh_wrapper = mesh_wrapper
        self.parameters = parameters
        self.T_c = parameters.get_value("T_c")
        self.G_m = parameters.get_value("G_m")
        self.phi_th = parameters.get_value("phi_th")
        self.cell_radius = parameters.get_value("R_c")
        self.G_M = parameters.get_value("G_M")
        self.alpha_p = parameters.get_value("alpha_p")
        self.T_p = parameters.get_value("T_p")
        self.min_tipcell_distance = parameters.get_value("min_tipcell_distance")
        self.clock_checker = af_sourcing.ClockChecker(mesh_wrapper, self.cell_radius, start_point="west")
        self.local_box = self._build_local_box(self.cell_radius)

    def get_global_tip_cells_list(self):
        return self.global_tip_cells_list

    def _point_distant_to_tip_cells(self, point: fenics.Point):
        if self.global_tip_cells_list:
            for tip_cell in self.global_tip_cells_list:
                if point.distance(fenics.Point(tip_cell.get_position())) < self.min_tipcell_distance:
                    return False
        return True

    def _build_local_box(self, cell_radius):
        return fu.build_local_box(self.mesh_wrapper.get_local_mesh(), cell_radius)

    def _is_in_local_box(self, position):
        return fu.is_in_local_box(self.local_box, position)

    def _add_tip_cell(self, tip_cell: TipCell):
        self.global_tip_cells_list.append(tip_cell)
        if self._is_in_local_box(tip_cell.get_position()):
            self.local_tip_cells_list.append(tip_cell)

    def activate_tip_cell(self, phi, T, gradT, current_step):
        # define root rank
        root = 0
        # logging
        info_adapter.info(f"Called {self.activate_tip_cell.__name__}")
        # get local mesh points
        local_mesh_points = self.mesh_wrapper.get_local_mesh().coordinates()
        # initialize local possible locations list
        local_possible_locations = []
        # Debug: setup cunters to check which test is not passed
        debug_adapter.debug(f"Searching for new tip cells")
        n_points_to_check = len(local_mesh_points)
        n_points_distant = 0
        n_points_phi_09 = 0
        n_points_over_Tc = 0
        n_points_over_Gm = 0
        for point in local_mesh_points:
            if self._point_distant_to_tip_cells(fenics.Point(point)):
                n_points_distant += 1
                if phi(point) > self.phi_th:
                    n_points_phi_09 += 1
                    if T(point) > self.T_c:
                        n_points_over_Tc += 1
                        if np.linalg.norm(gradT(point)) > self.G_m:
                            n_points_over_Gm += 1
                            local_possible_locations.append(point)
        debug_msg = \
            f"Finished checking. I found: \n" \
            f"\t* {n_points_distant} / {n_points_to_check} distant to the current tip cells \n" \
            f"\t* {n_points_phi_09} / {n_points_to_check} which were at phi > {self.phi_th} \n" \
            f"\t* {n_points_over_Tc} / {n_points_to_check} which were at T > {self.T_c} \n" \
            f"\t* {n_points_over_Gm} / {n_points_to_check} which were at G > {self.G_m} \n" \
            f"\t* {n_points_over_Gm} / {n_points_to_check} new possible locations"
        for line in debug_msg.split("\n"):
            debug_adapter.debug(line)

        # gather possible locations on root
        local_possible_locations_lists = comm.gather(local_possible_locations, root)

        # pick new cell position on root
        if rank == root:
            possible_locations = [item for sublist in local_possible_locations_lists for item in sublist]
            # get one, if there is a possible location
            if possible_locations:
                new_tip_cell_position = random.choice(possible_locations)
            else:
                new_tip_cell_position = None
        else:
            new_tip_cell_position = None

        # broadcast new tip cell position
        new_tip_cell_position = comm.bcast(new_tip_cell_position, root)

        # create tip cell and add it to the list
        if new_tip_cell_position is not None:
            new_tip_cell = TipCell(new_tip_cell_position,
                                   self.cell_radius,
                                   current_step)
            self._add_tip_cell(new_tip_cell)
            debug_adapter.debug(f"Created new tip cell at point {new_tip_cell_position}")

    def _remove_tip_cells(self, local_to_remove):
        # define root rank
        root = 0
        # get global to remove list
        local_to_remove_array = comm.gather(local_to_remove, root)
        if rank == root:
            global_to_remove = fu.flatten_list_of_lists(local_to_remove_array)
            # remove duplicates
            global_to_remove = list(set(global_to_remove))
        else:
            global_to_remove = None
        global_to_remove = comm.bcast(global_to_remove, root)
        debug_adapter.debug(f"Created global_to_remove list. It includes:")
        for tip_cell in global_to_remove:
            debug_adapter.debug(f"\t* tip_cell at position {tip_cell.get_position()}")

        # remove cells from global and local
        for tip_cell in global_to_remove:
            debug_adapter.debug(f"Removing tip cell at position {tip_cell.get_position()}")
            self.global_tip_cells_list.remove(tip_cell)
            if tip_cell in self.local_tip_cells_list:
                self.local_tip_cells_list.remove(tip_cell)

    def revert_tip_cells(self, T, gradT):
        info_adapter.info(f"Called {self.revert_tip_cells.__name__}")
        local_to_remove = []
        for tip_cell in self.local_tip_cells_list:
            position = tip_cell.get_position()
            if self.mesh_wrapper.is_inside_local_mesh(position):  # check only if point is in local mesh
                if (T(position) < self.T_c) or (np.linalg.norm(gradT(position)) < self.G_m):
                    local_to_remove.append(tip_cell)
            else:
                if not self.mesh_wrapper.is_inside_global_mesh(position):  # remove tip cell if not in global mesh
                    local_to_remove.append(tip_cell)
        # if not empty
        self._remove_tip_cells(local_to_remove)

    def _update_tip_cell_positions_and_get_field(self, T, gradT):
        # init tip cell field
        tip_cells_field_expression = TipCellsField(self.parameters)

        # define root rank
        root = 0
        # initialize cells went out of mesh
        tip_cells_out_of_mesh = []

        # iterate on all tip cells
        for tip_cell in self.global_tip_cells_list:
            # get position
            tip_cell_position = tip_cell.get_position()
            # the process that has access to tip cell position computes the mesh_related values
            if self.mesh_wrapper.is_inside_local_mesh(tip_cell_position):
                # compute velocity
                velocity = self.compute_tip_cell_velocity(gradT,
                                                          self.parameters.get_value("chi"),
                                                          tip_cell_position)
                # compute value of T in position
                T_at_point = T(tip_cell_position)
            else:
                velocity = None
                T_at_point = None

            # gather valocity and T_at_point
            velocity_array = comm.gather(velocity, root)
            T_at_point_array = comm.gather(T_at_point, root)
            if rank == root:
                # remove nones
                velocity_array_no_nones = [v for v in velocity_array
                                           if v is not None]
                T_at_point_array_no_nones = [val for val in T_at_point_array
                                             if val is not None]
                # get velocity
                velocity = velocity_array_no_nones[0]
                # get T_value
                T_at_point = T_at_point_array_no_nones[0]
            else:
                velocity = None
                T_at_point = None
            # bcast velocity and T_at_point
            velocity = comm.bcast(velocity, root)
            T_at_point = comm.bcast(T_at_point, root)

            # compute new position
            dt = self.parameters.get_value("dt")
            new_position = tip_cell_position + (dt * velocity)
            debug_msg = \
                f"DEBUG: p{fenics.MPI.comm_world.Get_rank()}: computing new tip cell position: \n" \
                f"\t*[tip cell position] + [dt] * [velocity] = \n" \
                f"\t*{tip_cell_position} + {dt} * {velocity} = {new_position}"
            for line in debug_msg.split("\n"):
                debug_adapter.debug(line)

            # if new position is not in global mesh
            if not self.mesh_wrapper.is_inside_global_mesh(new_position):
                tip_cells_out_of_mesh.append(tip_cell)  # set cell as to remove
            else:
                # else update local lists
                if tip_cell in self.local_tip_cells_list:
                    if not self._is_in_local_box(new_position):  # if tip cell is no motre in the local box
                        self.local_tip_cells_list.remove(tip_cell)  # remove it
                else:
                    if self._is_in_local_box(new_position):  # if tip cell is now in the local box
                        self.local_tip_cells_list.append(tip_cell)  # append it

            # move tip cell
            tip_cell.move(new_position)

            # append everything to tip_cell_field
            tip_cells_field_expression.add_tip_cell(tip_cell, velocity, T_at_point)

        # remove tip cells went out of mesh
        self._remove_tip_cells(tip_cells_out_of_mesh)

        # return tip cells field
        return tip_cells_field_expression

    def _assign_values_to_vector(self, phi, t_c_f_function):
        # get local values for T and source_field
        t_c_f_loc_values = t_c_f_function.vector().get_local()
        phi_loc_values = phi.vector().get_local()
        # set T to 1. where source_field is 1
        where_t_c_f_greater_than_0 = t_c_f_loc_values > 0.
        phi_loc_values[where_t_c_f_greater_than_0] = t_c_f_loc_values[where_t_c_f_greater_than_0]
        phi.vector().set_local(phi_loc_values)
        phi.vector().update_ghost_values()  # necessary, otherwise errors

    def _apply_tip_cells_field(self, phi, tip_cells_field_expression, V, is_V_sub_space):
        if not is_V_sub_space:
            # interpolate tip_cells_field
            t_c_f_function = fenics.interpolate(tip_cells_field_expression, V)
            # assign t_c_f_function to phi where is greater than 0
            self._assign_values_to_vector(phi, t_c_f_function)
        else:
            # collapse subspace
            V_collapsed = V.collapse()
            # interpolate tip cells field
            t_c_f_function = fenics.interpolate(tip_cells_field_expression, V_collapsed)
            # create assigner to collapsed
            assigner_to_collapsed = fenics.FunctionAssigner(V_collapsed, V)
            # assign phi to local variable phi_temp
            phi_temp = fenics.Function(V_collapsed)
            assigner_to_collapsed.assign(phi_temp, phi)
            # assign values to phi_temp
            self._assign_values_to_vector(phi_temp, t_c_f_function)
            # create inverse assigner
            assigner_to_sub = fenics.FunctionAssigner(V, V_collapsed)
            # assign phi_temp to phi
            assigner_to_sub.assign(phi, phi_temp)

        # return tip cells field function for monitoring
        return t_c_f_function

    def move_tip_cells(self, phi, T, gradT, V, is_V_sub_space) -> fenics.Function:
        info_adapter.info(f"Called {self.move_tip_cells.__name__}")
        # update tip cell positions
        tip_cells_field_expression = self._update_tip_cell_positions_and_get_field(T, gradT)
        # apply tip_cells_field to phi
        t_c_f_fucntion = self._apply_tip_cells_field(phi, tip_cells_field_expression, V, is_V_sub_space)
        # return tip cell field function for monitoring
        return t_c_f_fucntion

    def compute_tip_cell_velocity(self, gradT, chi, tip_cell_position):
        grad_T_at_point = gradT(tip_cell_position)
        G_at_point = np.linalg.norm(grad_T_at_point)
        if G_at_point < self.G_M:
            velocity = chi * grad_T_at_point
        else:
            velocity = chi * grad_T_at_point * (self.G_M / G_at_point)

        return velocity
