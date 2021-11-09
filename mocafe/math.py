import fenics


def perrycioc(variable,
              c: float,
              a: float,
              b: float,
              slope: float):
    """
    Sigmoid function as formalized by J.W.Perry on StackExchange1 [1].
    References:
     - [1] https://math.stackexchange.com/questions/535860/what-is-the-simplest-formula-for-activation-smooth-step-function
    :param variable: variable along which the sigmoid will be defined
    :param c: value of the variable for which the derivative of the sigmoid is max
    :param a: value of the sigmoid for variable -> -inf
    :param b: value of the sigmoid for variable -> +inf
    :param slope: slope of the sigmoid
    :return: the value of the sigmoid for the given value of the variable
    """
    return ((a * fenics.exp(c * slope)) + (b * fenics.exp(slope * variable))) / \
           (fenics.exp(c * slope) + fenics.exp(slope * variable))


def shf(variable, slope: float = 100):
    """
    Smoothed Heavyside Function (SHF) using the sigmoid function, that is:
            exp(slope * variable) / (1 + exp(slope * variable))
    :param variable: variable along which a step must be considered
    :param slope: slope of the SHF (100 by default)
    :return: symbolic form of the SHF compatible with FEniCS Form definition
    """
    return fenics.exp(slope * variable) / (1 + fenics.exp(slope * variable))


def estimate_cancer_area(phi: fenics.Function):
    """
    Estimate cancer area for the phase field phi representing the cancer.
    In order to work phi must be between 0 and 1.
    :param phi: phase field representing the cancer
    :return: the area of the given cancer
    """
    return fenics.assemble(phi * fenics.dx)


def estimate_capillaries_area(c: fenics.Function,
                              V: fenics.FunctionSpace):
    """
    Estimate capillaries area for the phase field c representing the vessels.
    In order to work c must be between -1 and 1
    :param c: phase field representing capillaries
    :param V: function space of c
    :return: the area of the given field
    """
    # rescale phase field c
    c_rescaled = fenics.project(
        fenics.Constant(1) + (fenics.Constant(0.5) * c),
        V
    )
    # estimate area of the rescaled field
    return estimate_cancer_area(c_rescaled)
