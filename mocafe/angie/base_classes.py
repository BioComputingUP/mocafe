import numpy as np


class BaseCell:
    def __init__(self,
                 point: np.ndarray,
                 creation_step):
        """
        inits a BaseCell in a given point, the base class used for both source cells and tip cells
        :param point: position of the cell
        :param creation_step: creation step of the cell, used for internal purposes
        """
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
        distance_squared = 0.
        for i in range(self.get_dimension()):
            distance_squared += (point[i] - self.position[i]) ** 2
        distance = np.sqrt(distance_squared)
        return distance
