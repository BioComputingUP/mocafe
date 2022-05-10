"""
Base classes used only by mocafe.angie
"""
import math
import fenics
import numpy as np


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
        if type(point) is not np.ndarray:
            raise TypeError(r"A cell position can be only an array of type np.ndarray. Please change array type")

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
