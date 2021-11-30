import numpy as np


class BaseCell:
    def __init__(self,
                 point: np.ndarray,
                 creation_step):
        self.initial_position = point
        self.creation_step = creation_step
        self.position = point
        # self.hmin = hmin

    def __eq__(self, other):
        return hash(other) == self.__hash__()

    def __hash__(self):
        init_pos = [coord for coord in self.initial_position]
        id_tuple = tuple([*init_pos, self.creation_step])
        return hash(id_tuple)

    def get_position(self):
        return self.position

    def get_dimension(self):
        return len(self.position)

    def get_distance(self, point):
        distance_sqared = 0.
        for i in range(self.get_dimension()):
            distance_sqared += (point[i] - self.position[i]) ** 2
        distance = np.sqrt(distance_sqared)
        return distance
