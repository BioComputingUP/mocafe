import fenics
from fenut.parameters import Parameters

"""FEniCS forms used throughout the simulation"""


def vascular_proliferation_form(alpha_p, af, T_p, phi, v):
    """
    Returns the UFL form for the proliferation of the vascular tissue as defined by the paper of Travasso et al.
    (2011) [1].
    Reference:
    - [1]: 
    :param alpha_p:
    :param af:
    :param T_p:
    :param phi:
    :param v:
    :return:
    """
    proliferation_function = alpha_p * af
    proliferation_function_max = alpha_p * T_p
    proliferation_function_hysteresis = fenics.conditional(fenics.gt(proliferation_function,
                                                                     proliferation_function_max),
                                                           proliferation_function_max,
                                                           proliferation_function)
    proliferation_term = proliferation_function_hysteresis * phi
    proliferation_term_heaviside = fenics.conditional(fenics.gt(proliferation_term, 0.),
                                                      proliferation_term,
                                                      fenics.Constant(0.))
    proliferation_term_form = proliferation_term_heaviside * v * fenics.dx
    return proliferation_term_form


def cahn_hillard_form(c: fenics.Variable,
                      c0: fenics.Function,
                      mu: fenics.Function,
                      mu0: fenics.Function,
                      q: fenics.TestFunction,
                      v: fenics.TestFunction,
                      dt,
                      theta,
                      chem_potential,
                      lmbda,
                      M):
    """
    Returns the UFL variational form of a general Cahn-Hillard system, discretized in time with the theta method.
    :param mu:
    :param mu0:
    :param c: variable for which the variational form is solved
    :param c0: previous value of the c variable
    :param q: test function for c
    :param v: test function for mu
    :param dt: time step
    :param theta: theta value for theta method
    :param chem_potential: UFL form for the chemical potential
    :param lmbda: energetic weight for the gradient of c
    :param M: motility of vessels
    :return: the UFL form of the Cahn-Hillard Equation
    """
    # Define form for mu (theta method)
    mu_mid = (1.0 - theta) * mu0 + theta * mu

    # chem potential derivative
    dfdc = fenics.diff(chem_potential, c)

    # define form
    l0 = ((c - c0) / dt) * q * fenics.dx + M * fenics.dot(fenics.grad(mu_mid), fenics.grad(q)) * fenics.dx
    l1 = mu * v * fenics.dx - dfdc * v * fenics.dx - lmbda * fenics.dot(fenics.grad(c), fenics.grad(v)) * fenics.dx
    form = l0 + l1

    # return form
    return form


def angiogenesis_form(c: fenics.Function,
                      c0: fenics.Function,
                      mu: fenics.Function,
                      mu0: fenics.Function,
                      v1: fenics.TestFunction,
                      v2: fenics.TestFunction,
                      af: fenics.Function,
                      parameters: Parameters):
    # define theta
    theta = 0.5

    # define chemical potential for the phase field
    c = fenics.variable(c)
    chem_potential = ((c ** 4) / 4) - ((c ** 2) / 2)

    # define total form
    form_cahn_hillard = cahn_hillard_form(c, c0, mu, mu0, v1, v2, parameters.get_value("dt"), theta, chem_potential,
                                          parameters.get_value("epsilon"), parameters.get_value("M"))
    form_proliferation = vascular_proliferation_form(parameters.get_value("alpha_p"), af, parameters.get_value("T_p"),
                                                     c, v1)
    form = form_cahn_hillard - form_proliferation

    return form


def angiogenic_factor_form(T: fenics.Function,
                           T0: fenics.Function,
                           phi: fenics.Function,
                           v: fenics.TestFunction,
                           parameters: Parameters):
    # get parameters
    alfa = parameters.get_value("alpha_T")
    D = parameters.get_value("D")
    dt = parameters.get_value("dt")
    # define reaction term
    reaction_term = alfa * T * phi
    reaction_term_non_negative = fenics.conditional(fenics.gt(reaction_term, fenics.Constant(0.0)),
                                                    reaction_term,
                                                    fenics.Constant(0.))
    reaction_term_form = reaction_term_non_negative * v * fenics.dx
    # define time discretization
    time_discretization = ((T - T0) / dt) * v * fenics.dx
    # define diffusion
    diffusion = D * fenics.dot(fenics.grad(T), fenics.grad(v)) * fenics.dx
    # add terms
    F = time_discretization + diffusion + reaction_term_form

    return F
