import fenics
import numpy as np

from mocafe.math import perrycioc


class SmoothCircularTumor(fenics.UserExpression):
    """
    Circular tumor positioned in a given center of a given radius with a smooth border
    """
    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 steepness: float = 100.):
        super(SmoothCircularTumor, self).__init__()
        self.center = center
        self.radius = radius
        self.max_value = 1.
        self.min_value = 0.
        self.steepness = steepness

    def eval(self, values, x):
        distance_from_center = fenics.sqrt(((x[0] - self.center[0]) ** 2) + ((x[1] - self.center[1]) ** 2))
        values[0] = perrycioc(distance_from_center,
                              self.radius,
                              self.max_value,
                              self.min_value,
                              self.steepness)

    def value_shape(self):
        return ()

    def __floordiv__(self, other):
        pass
