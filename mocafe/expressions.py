import fenics
import numpy as np
from mocafe.math import sigmoid


class EllipseField(fenics.UserExpression):
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

    def eval(self, values, x):
        x_in_ellipse = (((x[0] - self.center[0]) / self.semiax_x) ** 2) + \
                       (((x[1] - self.center[1]) / self.semiax_y) ** 2) <= 1.
        if x_in_ellipse:
            values[0] = self.inside_value
        else:
            values[0] = self.outside_value

    def value_shape(self):
        return ()

    def __floordiv__(self, other):
        pass


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


class PythonFunctionField(fenics.UserExpression):
    """
    Expression representing a field with values determined on the basis of the given python function.
    """
    def __init__(self,
                 python_fun,
                 *python_fun_params):
        self.python_fun = python_fun
        self.python_fun_params = python_fun_params

    def eval(self, values, x):
        values[0] = self.python_fun(*self.python_fun_params)

    def value_shape(self):
        return ()

    def __floordiv__(self, other):
        pass
