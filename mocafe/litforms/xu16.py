"""
Weak forms derived from Xu et al. (2016) paper.
In the following the complete reference:

Xu, J., Vilanova, G., & Gomez, H. (2016).
A Mathematical Model Coupling Tumor Growth and Angiogenesis.
PLOS ONE, 11(2), e0149422.
https://doi.org/10.1371/JOURNAL.PONE.0149422
"""
import fenics
from mocafe.math import shf
from mocafe.fenut.parameters import Parameters


def xu2016_nutrient_form(sigma: fenics.Function,
                         sigma_old: fenics.Function,
                         c: fenics.Function,
                         phi: fenics.Function,
                         v: fenics.TestFunction,
                         params: dict):
    """
    Equation describing the nutrient evolution in time based on the equation (6) of the paper by Xu et al.
    """
    # set parameters
    # v_pc = params["V_pc"] * (0.1 + 0.9 * shf(af - params["af_th"]))
    v_pc = params["V_pv"]
    # define form
    form = (((sigma - sigma_old) / params["dt"]) * v * fenics.dx) + \
           (params["D_sigma"] * fenics.dot(fenics.grad(sigma), fenics.grad(v)) * fenics.dx) - \
           (v_pc * c * shf(c) * (fenics.Constant(1.) - sigma) * v * fenics.dx) + \
           (params["V_uT"] * sigma * phi * v * fenics.dx) + \
           (params["V_uH"] * sigma * shf(fenics.Constant(1.) - phi) * v * fenics.dx)

    return form


def xu_2016_cancer_form(phi: fenics.Function,
                        phi_old: fenics.Function,
                        sigma: fenics.Function,
                        v: fenics.TestFunction,
                        params: Parameters):
    """
    Equation describing cancer evolution according to equation (5) of the paper of Xu et al. (2016).
    :param phi: function representing the tumor at a given time t_n
    :param phi_old: function representing the tumor at the step before, i.e. t_(n-1) (initial condition)
    :param sigma: function representing the distribution of the nutrient at a given time
    :param v: test function necessary to define the weak form of the equation
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
        ((phi - phi_old) / params.get_value("dt")) * v * fenics.dx + \
        (params.get_value("M_phi") * (params.get_value("lambda_phi") ** 2) * fenics.dot(fenics.grad(phi), fenics.grad(v)) * fenics.dx) + \
        (params.get_value("M_phi") * mu * v * fenics.dx)

    return form
