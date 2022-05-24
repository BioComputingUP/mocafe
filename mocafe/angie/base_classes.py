"""
Base classes used only by mocafe.angie
"""
import math
import dolfinx
import numpy as np
from mpi4py import MPI

_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()


class BaseCell:
    """
    Class representing a discrete cell agent. This class is inherited by both Tip Cells and Source Cells.
    """
    def __init__(self,
                 point: np.ndarray,
                 creation_step):
        """
        inits a BaseCell in a given point, the base class used for both source cells and tip cells

        :param point: position of the cell
        :param creation_step: creation step of the cell, used for internal purposes
        """
        if not isinstance(point, np.ndarray):
            raise TypeError(r"A cell position can be only an array of type np.ndarray. Please change array type")
        if len(point) != 3:
            raise ValueError(f"In FEniCSx each point must be of length 3, even in 2D")

        self.initial_position = point
        self.creation_step = creation_step
        self.position = point

    def __eq__(self, other):
        return hash(other) == self.__hash__()

    def __hash__(self):
        init_pos = [coord for coord in self.initial_position]
        id_tuple = tuple([*init_pos, self.creation_step])
        return hash(id_tuple)

    def get_position(self):
        """
        get cell position

        :return: tip cell position as numpy.ndarray
        """
        return self.position

    def get_dimension(self):
        """
        get dimension (2D or 3D) for the cell.

        :return: 2 for 2D, 3 for 3D
        """
        return len(self.position)

    def get_distance(self, point):
        """
        get the distance of the given point from the the cell

        :param point: the point to check the distance with
        :return: the distance
        """
        distance = np.sqrt(np.sum((point - self.position) ** 2))
        return distance


def fibonacci_sphere(n_points):
    """
    Returns ``n_points`` points evely distributed on a sphere of radius 1 using the fibonacci algorithm.

    :param n_points: number of points to spread on a sphere.
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append(np.array([x, y, z]))

    return points


class ClockChecker:
    """
    Class representing a clock checker, i.e. an object that checks if a given condition is met in the surroundings of
    a point of the mesh.
    """
    def __init__(self, mesh: dolfinx.mesh.Mesh, radius, start_point="east"):
        """
        inits a ClockChecker, which will check if a condition is met inside the given radius

        :param mesh: mesh
        :param radius: radius where to check if the condition is met
        :param start_point: starting point where to start checking. If the point is `east`, the clock checker will
            start checking from the point with the lower value of x[0]; if the point is `west` the clock cheker will
            start from the point with higher value of x[0]
        """
        self.radius = radius
        self.mesh = mesh
        self.mesh_dim = mesh.geometry.dim
        self.mesh_bbt = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
        if (start_point == "east") or (start_point == "west"):
            self.check_points = self._build_surrounding_points(start_point)
        else:
            raise ValueError("ClockChecker can be just 'east' or 'west' type")

    def _build_surrounding_points(self, start_point):
        """
        Internal use.

        Builds the points that need to be checked in the surrounding of the given point during the clock check.
        The points are already given in an order that should optimize the search, namely from the further to
        the closer.

        In 2D, the points are evenly distributed around n circles surrounding the given point. n is computed as the
        closest integer to self.radius / self.hmin. For each circle, the number of p

        :param start_point: east or west
        """
        # init points list
        points_list = []

        # compute min cell dimension across all the domains
        local_hmin = dolfinx.cpp.mesh.h(self.mesh,
                                        self.mesh_dim,
                                        range(self.mesh.topology.index_map(0).size_local)).min()
        global_hmin_values = _comm.gather(local_hmin, 0)
        if _rank == 0:
            hmin = min(global_hmin_values)
        else:
            hmin = None
        hmin = _comm.bcast(hmin, 0)

        if self.mesh_dim == 2:
            # compute number of circles
            n_circles = int(np.round(self.radius / hmin))
            if n_circles == 0:
                n_circles = 1
            # compute number of points for circles, from the largest to the shortest (the order is for optimization)
            n_points_for_circle = \
                [int(np.round(2 * np.pi * circle_number)) for circle_number in range(n_circles, 0, -1)]
            # compute radiuses of circles, from the largest to the shortest
            shortest_radius = self.radius / n_circles
            circles_radiuses = [circle_number * shortest_radius for circle_number in range(n_circles, 0, -1)]
            # create points for each circle and append them to the list
            reverse = (start_point == "west")
            for n_points, radius in zip(n_points_for_circle, circles_radiuses):
                angle_step = (2 * np.pi) / n_points
                angles = np.arange(0, (2 * np.pi) + angle_step, angle_step)
                circle_points = [radius * np.array([np.cos(angle), np.sin(angle), 0.]) for angle in angles]
                circle_points.sort(key=lambda x: x[0], reverse=reverse)
                points_list.extend(circle_points)
            # append origin
            points_list.append(np.array([0., 0., 0.]))

        elif self.mesh_dim == 3:
            # compute number of spheres
            n_spheres = int(np.round(self.radius / hmin))
            if n_spheres == 0:
                n_spheres = 1
            # compute sphere radiuses, from largest to shortest
            shortest_radius = self.radius / n_spheres
            sphere_radiuses = [sphere_number * shortest_radius for sphere_number in range(n_spheres, 0, -1)]
            # compute the number of points for each sphere, from the largest to the shortest
            sqrt_pi = np.sqrt(np.pi)
            n_points_for_sphere = \
                [int(np.round(np.round(((2 * sqrt_pi * rad) / hmin)) + 1)) ** 2 for rad in sphere_radiuses]
            # evaluate points
            reverse = (start_point == "west")
            for n_points, radius in zip(n_points_for_sphere, sphere_radiuses):
                # evaluate points with fibonacci algorithm
                fibonacci_points = fibonacci_sphere(n_points)
                # rescale points
                sphere_points = [radius * point for point in fibonacci_points]
                # sort points
                sphere_points.sort(key=lambda x: x[0], reverse=reverse)
                points_list.extend(sphere_points)
            # append origin
            points_list.append(np.array([0., 0., 0.]))
        else:
            raise NotImplementedError(f"Clock checker is not implemented for meshes of dim {self.mesh_dim}. "
                                      f"Only for dim 2 and 3.")

        return points_list

    def clock_check(self, point, function: dolfinx.fem.Function, condition):
        """
        Clock-check the given function in the surrounding of the given point

        For instance, the following:

            clock_checker = ClockChecker(mesh, radius=4.)
            clock_checker.clock_check(point, foo, lambda: foo_val: foo_val < 0.)

        Will return True if exist at least one point inside the circle of radius 4 which surrounds the point
        for which the function foo has a value lower than 0. Notice that all the points inside the circle will
        be checked, not just those on the border. Also the point ``point`` is included.

        :param point: center of the clock-check
        :param function: function to check
        :param condition: lambda function representing the condition to be met
        :return: True if the condition is met; False otherwise
        """

        for check_point in self.check_points:
            # compute current check point
            ccp = point + check_point
            # define list (otherwise errors)
            ccp_list = np.array([ccp, ])
            # compute cells near the current check points
            candidate_cells = dolfinx.geometry.compute_collisions(self.mesh_bbt, ccp_list)
            # compute the cells actually colliding with the current check points
            colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, candidate_cells, ccp_list)
            # get an array of cells
            colliding_cells_array = colliding_cells.links(0)
            # check if empty
            if len(colliding_cells_array) > 0:
                # if not, pick one cell for evaluation (no matter which one)
                current_cell = colliding_cells_array[0]
                # evaluate condition on point and cell
                if condition(function.eval(ccp, current_cell)):
                    # if condition is true, break cycle and return True
                    return True
        # if no point meets the condition, return False
        return False
