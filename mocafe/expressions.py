import numpy as np
from mocafe.math import sigmoid


class EllipseField:
    """
    Expression representing an ellipse with a given value inside and a given value outside. The user can set
    the semiaxes and the center.
    """
    def __init__(self,
                 center: np.ndarray,
                 semiax_x: float,
                 semiax_y: float,
                 inside_value: float,
                 outside_value: float):
        super(EllipseField, self).__init__()
        self.center = center
        self.semiax_x = semiax_x
        self.semiax_y = semiax_y
        self.inside_value = inside_value
        self.outside_value = outside_value

    def eval(self, x):
        x_in_ellipse = (((x[0] - self.center[0]) / self.semiax_x) ** 2) + \
                       (((x[1] - self.center[1]) / self.semiax_y) ** 2) <= 1.
        return np.where(x_in_ellipse, self.inside_value, self.outside_value)


class EllipsoidField:
    """
    Expression representing an ellipsoid with a given value inside and a given value outside. The user can set
    the semiaxes and the center.
    """

    def __init__(self,
                 center: np.ndarray,
                 semiax_x: float,
                 semiax_y: float,
                 semiax_z: float,
                 inside_value: float,
                 outside_value: float):
        super(EllipsoidField, self).__init__()
        self.center = center
        self.semiax_x = semiax_x
        self.semiax_y = semiax_y
        self.semiax_z = semiax_z
        self.inside_value = inside_value
        self.outside_value = outside_value

    def eval(self, x):
        x_in_ellipsoid = (((x[0] - self.center[0]) / self.semiax_x) ** 2) + \
                         (((x[1] - self.center[1]) / self.semiax_y) ** 2) + \
                         (((x[2] - self.center[2]) / self.semiax_z) ** 2) <= 1.
        return np.where(x_in_ellipsoid, self.inside_value, self.outside_value)


class SmoothCircle:
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

    def eval(self, x):
        distance_from_center = np.sqrt(((x[0] - self.center[0]) ** 2) + ((x[1] - self.center[1]) ** 2))
        return sigmoid(distance_from_center, self.radius, self.inside_value, self.outside_value, self.slope)


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

    def eval(self, x):
        super(SmoothCircularTumor, self).eval(x)
