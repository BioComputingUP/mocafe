"""
Weak forms of the Phase-Field models related to angiogenesis. Each weak form is a FEniCS UFL Form which can be used
calling a specific method, that returns the form itself.

If you use this model in your research, remember to cite the original paper describing the model:

    Travasso, R. D. M., Poiré, E. C., Castro, M., Rodrguez-Manzaneque, J. C., & Hernández-Machado, A. (2011).
    Tumor angiogenesis and vascular patterning: A mathematical model. PLoS ONE, 6(5), e19989.
    https://doi.org/10.1371/journal.pone.0019989

For a use example see the :ref:`Angiogenesis <Angiogenesis 2D Demo>` and the
:ref:`Angiogenesis 3D <Angiogenesis 2D Demo>` demos.
"""

import ufl
import dolfinx
from petsc4py import PETSc
from mocafe.fenut.parameters import Parameters


def vascular_proliferation_form(alpha_p,
                                af: dolfinx.fem.Function,
                                af_p,
                                c: dolfinx.fem.Function or ufl.variable.Variable,
                                v: ufl.TestFunction):
    r"""
    Returns the UFL Form for the proliferation term of the vascular tissue as defined by the paper of Travasso et al.
    (2011) :cite:`Travasso2011a`.

    The corresponding term of the equation is (H is the Heaviside function):

    .. math::
       \alpha_p(af_p) \cdot c \cdot H(c)

    Where :math: `af` is the angiogenic factor concentration, and :math: `\alpha_p(af)` represents the proliferation
    rate, that is defined as the follwing function of :math: `af`. The definition of the latter function is the
    following:

    .. math::
       \alpha_p(af) &= \alpha_p \cdot af_p \quad \textrm{if} \quad af>af_p \\
                    &= \alpha_p \cdot af  \quad \textrm{if} \quad 0<af \le af_p \\
                    & = 0 \quad \textrm{if} \quad af \le 0

    Where :math: `\alpha-p` and :math: `af_p` are constants.

    :param alpha_p: costant of the proliferation rate function for the capillaries
    :param af: FEniCS function representing the angiogenic factor distribution
    :param af_p: maximum concentration of angiogenic factor leading to proliferation. If af > af_p, the proliferation
        rate remains alpha_p * af_p
    :param c: FEniCS function representing the capillaries
    :param v: FEniCS test function
    :return: the UFL form for the proliferation term
    """
    # def the proliferation function
    proliferation_function = alpha_p * af
    # def the max value for the proliferation function
    proliferation_function_max = alpha_p * af_p
    # take the bigger between the two of them
    proliferation_function_hysteresis = ufl.conditional(ufl.gt(proliferation_function,
                                                               proliferation_function_max),
                                                        proliferation_function_max,
                                                        proliferation_function)
    # multiply the proliferation term with the vessel field
    proliferation_term = proliferation_function_hysteresis * c
    # take it oly if bigger than 0
    proliferation_term_heaviside = ufl.conditional(ufl.gt(proliferation_term, 0.),
                                                   proliferation_term,
                                                   0.)
    # build the form
    proliferation_term_form = proliferation_term_heaviside * v * ufl.dx
    return proliferation_term_form


def cahn_hillard_form(c,
                      c0: dolfinx.fem.Function,
                      mu: dolfinx.fem.Function,
                      mu0: dolfinx.fem.Function,
                      q: dolfinx.fem.Function,
                      v: dolfinx.fem.Function,
                      dt,
                      theta,
                      chem_potential,
                      lmbda,
                      M):
    r"""
    Returns the UFL form of a for a general Cahn-Hillard equation, discretized in time using the theta method. The
    method is the same reported by the FEniCS team in one of their demo `1. Cahn-Hillard equation`_ and is briefly
    discussed below for your conveneince.

    .. _1. Cahn-Hillard equation:
       https://fenicsproject.org/olddocs/dolfin/2016.2.0/cpp/demo/documented/cahn-hilliard/cpp/documentation.html

    The Cahn-Hillard equation reads as follows:

    .. math::
       \frac{\partial c}{\partial t} - \nabla \cdot M (\nabla(\frac{d f}{d c}
             - \lambda \nabla^{2}c)) = 0 \quad \textrm{in} \ \Omega

    Where :math: `c` is the unknown field to find, :math: `f` is some kind of energetic potential which defines the
    phase separation, and :math: `M` is a scalar parameter.

    The equation involves 4th order derivatives, so its weak form could not be handled with the standard Lagrange
    finite element basis. However, the equation can be splitted in two second-order equations adding a second unknown
    auxiliary field :math: `\mu`:

    .. math::
       \frac{\partial c}{\partial t} - \nabla \cdot M \nabla\mu  &= 0 \quad \textrm{in} \ \Omega, \\
       \mu -  \frac{d f}{d c} + \lambda \nabla^{2}c &= 0 \quad \textrm{ in} \ \Omega.

    In this way, it is possible to solve this equation using the standard Lagrange basis and, indeed, this
    implementation uses this form.

    :param c: main Cahn-Hillard field
    :param c0: initial condition for the main Cahn-Hillard field
    :param mu: auxiliary field for the Cahn-Hillard equation
    :param mu0: initial condition for the auxiliary field
    :param q: test function for c
    :param v: test function for mu
    :param dt: time step
    :param theta: theta value for theta method
    :param chem_potential: UFL form for the Cahn-Hillard potential
    :param lmbda: energetic weight for the gradient of c
    :param M: scalar parameter
    :return: the UFL form of the Cahn-Hillard Equation
    """
    # Define form for mu (theta method)
    mu_mid = (1.0 - theta) * mu0 + theta * mu

    # chem potential derivative
    dfdc = ufl.diff(chem_potential, c)

    # define form
    l0 = ((c - c0) / dt) * q * ufl.dx + M * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
    l1 = mu * v * ufl.dx - dfdc * v * ufl.dx - lmbda * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx
    form = l0 + l1

    # return form
    return form


def angiogenesis_form(c: dolfinx.fem.Function,
                      c0: dolfinx.fem.Function,
                      mu: dolfinx.fem.Function,
                      mu0: dolfinx.fem.Function,
                      v1: ufl.TestFunction,
                      v2: ufl.TestFunction,
                      af: dolfinx.fem.Function,
                      parameters: Parameters):
    r"""
    Returns the UFL form for the Phase-Field model for angiogenesis reported by Travasso et al. (2011)
    :cite:`Travasso2011a`.

    The equation reads simply as the sum of a Cahn-Hillard term and a proliferation term (for further details see
    the original paper):

    .. math::
       \frac{\partial c}{\partial t} = M \cdot \nabla^2 [\frac{df}{dc}\ - \epsilon \nabla^2 c]
       + \alpha_p(T) \cdot c H(c)

    Where :math: `c` is the unknown field representing the capillaries, and :

    .. math:: f = \frac{1}{4} \cdot c^4 - \frac{1}{2} \cdot c^2

    .. math::
       \alpha_p(af) &= \alpha_p \cdot af_p \quad \textrm{if} \quad af>af_p \\
                    &= \alpha_p \cdot af  \quad \textrm{if} \quad 0<af \le af_p \\
                    & = 0 \quad \textrm{if} \quad af \le 0

    In this implementation, the equation is splitted in two equations of lower order, in order to make the weak form
    solvable using standard Lagrange finite elements:

    .. math::
       \frac{\partial c}{\partial t} &= M \nabla^2 \cdot \mu + \alpha_p(T) \cdot c H(c) \\
       \mu &= \frac{d f}{d c} - \epsilon \nabla^{2}c

    :param c: capillaries field
    :param c0: initial condition for the capillaries field
    :param mu: auxiliary field
    :param mu0: initial condition for the auxiliary field
    :param v1: test function for c
    :param v2: test function  for mu
    :param af: angiogenic factor field
    :param parameters: simulation parameters
    :return:
    """
    # define theta
    theta = 0.5

    # define chemical potential for the phase field
    c = ufl.variable(c)
    chem_potential = ((c ** 4) / 4) - ((c ** 2) / 2)

    # define total form
    form_cahn_hillard = cahn_hillard_form(c, c0, mu, mu0, v1, v2, parameters.get_value("dt"), theta, chem_potential,
                                          parameters.get_value("epsilon"), parameters.get_value("M"))
    form_proliferation = vascular_proliferation_form(parameters.get_value("alpha_p"), af, parameters.get_value("T_p"),
                                                     c, v1)
    form = form_cahn_hillard - form_proliferation

    return form


def angiogenic_factor_form(af: dolfinx.fem.Function,
                           af_0: dolfinx.fem.Function,
                           c: dolfinx.fem.Function,
                           v: ufl.TestFunction,
                           parameters: Parameters):
    r"""
    Returns the UFL form for the equation for the angiogenic factor reported by Travasso et al. (2011)
    :cite:`Travasso2011a`.

    The equation simply considers the diffusion of the angiogenic factor and its consumption by the capillaries
    (for further details see the original paper):

    .. math::
       \frac{\partial af}{\partial t} = D \nabla^2 af - \alpha_T \cdot af \cdot c \cdot H(c)

    Where :math: `af` is the angiogenic factor field, :math: `c` is the capillaries field, and :math: `H(c)` is the
    Heaviside function

    :param af: angiogenic factor field
    :param af_0: initial condition for the angiogenic factor field
    :param c: capillaries field
    :param v: test function for the equation
    :param parameters: simulation parameters
    :return:
    """
    # get parameters
    alfa = parameters.get_value("alpha_T")
    D = parameters.get_value("D")
    dt = parameters.get_value("dt")
    # define reaction term
    reaction_term = alfa * af * c
    reaction_term_non_negative = ufl.conditional(
        condition=ufl.gt(reaction_term, dolfinx.fem.Constant(af_0.function_space.mesh, PETSc.ScalarType(0.))),
        true_value=reaction_term,
        false_value=dolfinx.fem.Constant(af_0.function_space.mesh, PETSc.ScalarType(0.)))
    reaction_term_form = reaction_term_non_negative * v * ufl.dx
    # define time discretization
    time_discretization = ((af - af_0) / dt) * v * ufl.dx
    # define diffusion
    diffusion = D * ufl.dot(ufl.grad(af), ufl.grad(v)) * ufl.dx
    # add terms
    F = time_discretization + diffusion + reaction_term_form

    return F
