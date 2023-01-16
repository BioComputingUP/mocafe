"""
This module contains methods to build the weak form of a Phase Field cancer model described by
Xu and collaborators :cite:`Xu2016`.

If you use this model in your research, remember to cite the original paper describing the model:

    Xu, J., Vilanova, G., & Gomez, H. (2016). A Mathematical Model Coupling Tumor Growth and Angiogenesis.
    PLOS ONE, 11(2), e0149422. https://doi.org/10.1371/JOURNAL.PONE.0149422

For a complete description of the model, please refer to the original publication. Also, if you use this model
for your scientific work, remember to cite the original paper.
"""
import dolfinx
import ufl
from mocafe.math import shf
from mocafe.fenut.parameters import Parameters, _unpack_parameters_list


def xu2016_nutrient_form(sigma: dolfinx.fem.Function,
                         sigma_0: dolfinx.fem.Function,
                         c: dolfinx.fem.Function,
                         phi: dolfinx.fem.Function,
                         v: ufl.TestFunction,
                         params: Parameters or None,
                         **kwargs):
    r"""
    Equation describing the nutrient evolution in time based on the equation (6) of the paper by Xu et al.
    :cite:`Xu2016`.

    The equation simply describes a general nutrient :math:`\sigma` which diffuses from the capillaries field :math:`c`
    and is consumed by the cancer and the healthy tissue (represented by value 1 and 0 of the field :math:`\varphi`:).

    The parameters required for the equation must be specified in the ``parameters`` object. The name for the
    required parameters ore:

    - ``dt``: the time step (time discretization: backward Euler)
    - ``D_sigma``: diffusion constant for \sigma (correspond to :math:`D_{\sigma}` in the original paper)
    - ``V_pc``: nutrient production rate at capillaries (correspond to :math:`V_{p}^{c}` in the original paper)
    - ``V_uT``: nutrient uptake rate by the tumor (correspond to :math:`V_{u}^{T}` in the original paper)
    - ``V_uH``: nutrient uptake rate by the healthy tissue (correspond to :math:`V_{u}^{H}` in the original paper)

    (New in version 1.4) Specify a parameter for the form calling the function, e.g. with
    ``xu2016_nutrient_form(sigma, sigma0, c, phi, v, params, D_sigma=0.1)``. If both a Parameters
    object and a parameter as input are given, the function will choose the input parameter.

    :param sigma: the FEniCS ``Function`` for the nutrient
    :param sigma_0: the FEniCS ``Function`` for the nutrient initial value
    :param c: the FEniCS ``Function`` for the capillaries
    :param phi: the FEniCS ``Function`` for the cancer
    :param v: the Test Function for the equation
    :param params: the simulation parameters as ``Parameters`` object
    :return: the FEniCS UFL form of the equation
    """
    # get parameters
    dt, D_sigma, V_pc, V_uT, V_uH = _unpack_parameters_list(["dt", "D_sigma", "V_pc", "V_uT", "V_uH"],
                                                            params,
                                                            kwargs)
    # define form
    form = (((sigma - sigma_0) / dt) * v * ufl.dx) + \
           (D_sigma * ufl.dot(ufl.grad(sigma), ufl.grad(v)) * ufl.dx) - \
           (V_pc * c * shf(c) * (1. - sigma) * v * ufl.dx) + \
           (V_uT * sigma * phi * v * ufl.dx) + \
           (V_uH * sigma * shf(1. - phi) * v * ufl.dx)

    return form


def xu_2016_cancer_form(phi: dolfinx.fem.Function,
                        phi_0: dolfinx.fem.Function,
                        sigma: dolfinx.fem.Function,
                        v: ufl.TestFunction,
                        params: Parameters or None,
                        **kwargs):
    r"""
    Equation describing cancer evolution according to equation (5) of the paper of Xu et al. :cite:`Xu2016`.

    The equation describes a pretty versatile phase field cancer model based on a double well potential. The cancer
    is represented by the variable \varphi.

    The parameters required for the equation must be specified in the ``parameters`` object. The name for the
    required parameters ore:

    - ``dt``: the time step (time discretization: backward Euler)
    - ``M_phi``: mobility constant for \varphi (correspond to :math:`M_{\phi}` in the original paper)
    - ``lambda_phi``: interface constant (correspond to :math:`\lambda_{phi}` in the original paper)
    - ``sigma^(h-v)``: nutrient value separating the proliferative rim and the hypoxic rim (correspond to
      :math:`\sigma^{h - v}` in the original paper)

    (New in version 1.4) Specify a parameter for the form calling the function, e.g. with
    ``xu_2016_cancer_form(phi, phi0, sigma, v, params, M_phi=0.1)``. If both a Parameters
    object and a parameter as input are given, the function will choose the input parameter.

    :param phi: the FEniCS ``Function`` for the tumor
    :param phi_0: the FEniCS ``Function`` for the tumor initial condition
    :param sigma: the FEniCS ``Function`` for the nutrient
    :param v: Test Function for the equation
    :param params: parameters of the equation
    :return: the FEniCS UFL form of the equation
    """
    # get parameters
    sigma_h_v, dt, M_phi, lambda_phi = _unpack_parameters_list(
        ["sigma_h_v", "dt", "M_phi", "lambda_phi"],
        params,
        kwargs
    )

    # transform phi and sigma in variables
    phi = ufl.variable(phi)
    sigma = ufl.variable(sigma)

    # define chem potential
    g = (phi ** 2) * ((1 - phi) ** 2)
    h = (phi ** 2) * (3 - 2 * phi)
    m_sigma = (- 2 / (3.01 * ufl.pi)) * ufl.atan(15 * (sigma - sigma_h_v))
    chem_poteintial = g + (h * m_sigma)

    # define mu
    mu = ufl.diff(chem_poteintial, phi)

    # define form
    form = \
        ((phi - phi_0) / dt) * v * ufl.dx + \
        (M_phi * (lambda_phi ** 2) * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx) + \
        (M_phi * mu * v * ufl.dx)

    return form
