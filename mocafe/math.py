import fenics


def sigmoid(variable,
            c: float,
            a: float,
            b: float,
            slope: float):
    r"""
    Simple sigmoid function defined as:

    .. math::
        y(x) = \frac{a \cdot e^{slope \cdot c} + b \cdot e^{r \cdot x}}{e^{slope \cdot c} + e^{slope \cdot c}}

    Where: :math:`a` represent the value of the sigmoid for x -> -Inf; :math:`b` represent the value of the sigmoid
    for x -> +Inf; :math:`c` represent the center of the sigmoid (where the derivative is max); .math:`slope`
    represent the slope of the sigmoid.


    :param variable: variable x for which the sigmoid is defined
    :param c: central value of the sigmoid
    :param a: value of the sigmoid for variable -> -inf
    :param b: value of the sigmoid for variable -> +inf
    :param slope: slope of the sigmoid
    :return: the value of the sigmoid for the given value of the variable
    """
    return ((a * fenics.exp(c * slope)) + (b * fenics.exp(slope * variable))) / \
           (fenics.exp(c * slope) + fenics.exp(slope * variable))


def shf(variable, slope: float = 100):
    r"""
    Smoothed Heavyside Function (SHF) using the sigmoid function, which reads:

    .. math::
        \frac{e^{slope * variable}}{(1 + e^{slope * variable})}


    :param variable: varible for the SHF
    :param slope: slope of the SHF. Default is 100
    :return: the value of the sigmoid for the given value of the variable
    """
    return fenics.exp(slope * variable) / (1 + fenics.exp(slope * variable))


def estimate_cancer_area(phi: fenics.Function):
    r"""
    Estimate cancer area for the phase field :math:`\varphi` representing the cancer, where the phase field must
    have values between 0 and 1

    :param phi: phase field representing the cancer
    :return: the area of the given cancer
    """
    return fenics.assemble(phi * fenics.dx)


def estimate_capillaries_area(c: fenics.Function,
                              V: fenics.FunctionSpace):
    """
    Estimate capillaries area for the phase field c representing the vessels. In order to work c must be
    between -1 and 1.

    :param c: phase field representing capillaries
    :param V: function space of c
    :return: the area of the given field
    """
    # rescale phase field c
    c_rescaled = fenics.project(
        fenics.Constant(1) + (fenics.Constant(0.5) * c),
        c.function_space
    )
    # estimate area of the rescaled field
    return estimate_cancer_area(c_rescaled)
