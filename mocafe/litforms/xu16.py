"""
Weak forms of the Phase-Field models to the tumor growth model presented by Xu and collaborators in 2016 [Xu2016] _.
Each weak form is a FEniCS UFL Form which can be used calling a specific method, that returns the form itself.

References:

.. [Xu2016] Xu, J., Vilanova, G., & Gomez, H. (2016). A Mathematical Model Coupling Tumor Growth and Angiogenesis. PLOS
   ONE, 11(2), e0149422. https://doi.org/10.1371/JOURNAL.PONE.0149422
"""
import fenics
from mocafe.math import shf
from mocafe.fenut.parameters import Parameters


def xu2016_nutrient_form(sigma: fenics.Function,
                         sigma_0: fenics.Function,
                         c: fenics.Function,
                         phi: fenics.Function,
                         v: fenics.TestFunction,
                         params: dict):
    r"""
    Equation describing the nutrient evolution in time based on the equation (6) of the paper by Xu et al (2016)
    [Xu2016] _.

    The equation simply describes a general nutrient :math:`\sigma` which diffuses from the capillaries field :math:`c`
    and is consumed by the cancer and the healthy tissue (represented by value 1 and 0 of the field :math:`\varphi`:).

    .. math::
       \frac{\partial \sigma}{\partial t} = \nabla (D_{\sigma} \nabla \sigma) +
       V_{p}^{c} \cdot (1 - \sigma) \cdot c \cdot H(c) - V_{u}^{T} \cdot \sigma \varphi -
       V_{u}^{H} \cdot \sigma H(1 -\varphi)

    References:

    .. [Xu2016] Xu, J., Vilanova, G., & Gomez, H. (2016). A Mathematical Model Coupling Tumor Growth and Angiogenesis. PLOS
       ONE, 11(2), e0149422. https://doi.org/10.1371/JOURNAL.PONE.0149422

    :param sigma: the nutrient field
    :param sigma_0: the initial nutrient field
    :param c: the capillaries field
    :param phi: the cancer field
    :param v: the test function for the equation
    :param params: the simulation parameters
    :return: the weak form of the equation
    """
    # set parameters
    # v_pc = params["V_pc"] * (0.1 + 0.9 * shf(af - params["af_th"]))
    v_pc = params["V_pv"]
    # define form
    form = (((sigma - sigma_0) / params["dt"]) * v * fenics.dx) + \
           (params["D_sigma"] * fenics.dot(fenics.grad(sigma), fenics.grad(v)) * fenics.dx) - \
           (v_pc * c * shf(c) * (fenics.Constant(1.) - sigma) * v * fenics.dx) + \
           (params["V_uT"] * sigma * phi * v * fenics.dx) + \
           (params["V_uH"] * sigma * shf(fenics.Constant(1.) - phi) * v * fenics.dx)

    return form


def xu_2016_cancer_form(phi: fenics.Function,
                        phi_0: fenics.Function,
                        sigma: fenics.Function,
                        v: fenics.TestFunction,
                        params: Parameters):
    r"""
    Equation describing cancer evolution according to equation (5) of the paper of Xu et al. (2016) [Xu2016] _.

    The equation describes a pretty versatile phase field cancer model based on a double well potential. The model
    reads:

    .. math::
        \frac{\partial \varphi}{\partial t} = M_{\varphi} \cdot (\lambda^2 \cdot \nabla^2 \varphi -
        \mu_{\varphi}(\varphi, \sigma))

    Where:

    .. math::
        \mu_{\varphi}(\varphi, \sigma) = \frac{\partial \Psi}{\partial \varphi}

        \Psi(\varphi, \sigma) = g(\varphi) + m(\sigma) \cdot h(\varphi)

        g(\varphi) = \varphi^2 \cdot (1 - \varphi)^2

        h(\varphi) = \varphi^2 \cdot (3 - 2\varphi)

        m(\sigma) = \frac{2}{3.01\pi} \cdot arctan(15 \cdot (\sigma - \sigma^{h-v}))

    :param phi: tumor field
    :param phi_0: initial tumor field
    :param sigma: nutrient field
    :param v: test function for the equation
    :param params: parameters of the equation
    :return: the weak form of the equation
    """
    # transform phi and sigma in variables
    phi = fenics.variable(phi)
    sigma = fenics.variable(sigma)

    # define chem potential
    g = (phi ** 2) * ((1 - phi) ** 2)
    h = (phi ** 2) * (3 - 2 * phi)
    m_sigma = (- 2 / (3.01 * fenics.pi)) * fenics.atan(15 * (sigma - params.get_value("sigma^(h-v)")))
    chem_poteintial = g + (h * m_sigma)

    # define mu
    mu = fenics.diff(chem_poteintial, phi)

    # define form
    form = \
        ((phi - phi_0) / params.get_value("dt")) * v * fenics.dx + \
        (params.get_value("M_phi") * (params.get_value("lambda_phi") ** 2) * fenics.dot(fenics.grad(phi), fenics.grad(v)) * fenics.dx) + \
        (params.get_value("M_phi") * mu * v * fenics.dx)

    return form
