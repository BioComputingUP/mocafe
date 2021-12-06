import fenics
import numpy as np

from mocafe.math import sigmoid


class SmoothCircle(fenics.UserExpression):
    """
    Expression representing a Circle with a given center and radius, with given values inside and outside the circle.
    The border of the circle is a smooth sigmoidal function with the given slope.
    """
    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 inside_value: float,
                 outside_value: float,
                 slope: float = 100.):
        """
        inits a SmoothCircle expression.

        :param center: center of the circle
        :param radius: radius of the circle
        :param inside_value: value of the expression inside the circle
        :param outside_value: value of the expression outside the circle
        :param slope: slope of the sigmoid function at the circle borderd. Default is 100
        """
        super(SmoothCircle, self).__init__()
        self.center = center
        self.radius = radius
        self.inside_value = inside_value
        self.outside_value = outside_value
        self.slope = slope

    def eval(self, values, x):
        distance_from_center = fenics.sqrt(((x[0] - self.center[0]) ** 2) + ((x[1] - self.center[1]) ** 2))
        values[0] = sigmoid(distance_from_center, self.radius, self.inside_value, self.outside_value, self.slope)

    def value_shape(self):
        return ()

    def __floordiv__(self, other):
        pass


class SmoothCircularTumor(SmoothCircle):
    """
    Expression representing a circular phase field tumor with a given center and radius, which has value 1. inside
    the circle and 0. outside. The border of the circle is a smooth sigmoidal function with the given slope.
    """
    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 slope: float = 100.):
        """
        inits a SmoothCircularTumor.

        :param center: center of the tumor.
        :param radius: radius of the tumor.
        :param slope: slope of the smooth border. Default is 100.
        """
        super(SmoothCircularTumor, self).__init__(center, radius, 1., 0., slope)

    def eval(self, values, x):
        super(SmoothCircularTumor, self).eval(values, x)
