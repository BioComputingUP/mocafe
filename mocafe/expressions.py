import fenics
import numpy as np

from mocafe.math import perrycioc


class SmoothCircle(fenics.UserExpression):
    """
    Circle positioned in a given center, with given values inside and outside the circle. It has a smooth border.
    """
    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 inside_value: float,
                 outside_value: float,
                 slope: float = 100.):
        super(SmoothCircle, self).__init__()
        self.center = center
        self.radius = radius
        self.inside_value = inside_value
        self.outside_value = outside_value
        self.slope = slope

    def eval(self, values, x):
        distance_from_center = fenics.sqrt(((x[0] - self.center[0]) ** 2) + ((x[1] - self.center[1]) ** 2))
        values[0] = perrycioc(distance_from_center,
                              self.radius,
                              self.inside_value,
                              self.outside_value,
                              self.slope)

    def value_shape(self):
        return ()

    def __floordiv__(self, other):
        pass


class SmoothCircularTumor(SmoothCircle):
    """
    Circular tumor positioned in a given center of a given radius with a smooth border
    """
    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 slope: float = 100.):
        super(SmoothCircularTumor, self).__init__(center, radius, 1., 0., slope)

    def eval(self, values, x):
        super(SmoothCircularTumor, self).eval(values, x)
