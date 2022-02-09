"""
This module contains methods to build the weak form of a Phase Field prostate cancer model described by
Lorenzo and collaborators :cite:`Lorenzo2016`.

For a complete description of the model, please refer to the original publication. Also, if you use this model
for your scientific work, remember to cite the original paper.
"""
import fenics
import mocafe.fenut.parameters as mcfp


def prostate_cancer_chem_potential(var_phi: fenics.Variable,
                                   chem_potential_constant):
    r"""
    Returns the "chemical potential" associated with the cancer equation. It is used to build the weak form for
    the prostate cancer model.

    :param var_phi: the \varphi variable
    :param chem_potential_constant: the constant of the chemical potential (equals to 16 in :cite:`Lorenzo2016`)
    :return: the chemical potential, as FEniCS UFL equation
    """
    chem_potential = chem_potential_constant * (var_phi ** 2) * ((1 - var_phi) ** 2)
    return chem_potential


def df_dphi(phi, chem_potential_constant):
    r"""
    Returns the derivative of the "chemical potential" associated with the cancer equation. It is used to build the weak form for
    the prostate cancer model.

    :param phi: the FEniCS ``Function`` for \varphi
    :param chem_potential_constant: the constant of the chemical potential (equals to 16 in :cite:`Lorenzo2016`)
    :return: the derivative of the chemical potential, as FEniCS UFL equation
    """
    var_phi = fenics.variable(phi)
    chem_potential = prostate_cancer_chem_potential(var_phi, chem_potential_constant)
    return fenics.diff(chem_potential, var_phi)


def prostate_cancer_form(phi: fenics.Function,
                         phi_prec: fenics.Function,
                         sigma: fenics.Function,
                         v: fenics.TestFunction,
                         parameters: mcfp.Parameters):
    r"""
    Builds the FEniCS UFL weak form for the prostate cancer equation reported by Lorenzo and collaborators
    :cite:`Lorenzo2016`. The name of the variable for the cancer is \varphi

    The parameters required for the equation must be specified in the ``parameters`` object. The name for the
    required parameters ore:
     - ``dt``: the time step (time discretization: backward Euler)
     - ``lambda``: diffusion constant for \varphi (correspond to \lambda in the original paper)
     - ``tau``: the time constant for the double-well potential (correspond to \tau in the original paper)
     - ``chempot_constant``: the double-well potential constant (no name in the original paper; its value was 16)
     - ``chi``: proliferation rate for prostate cancer (correspond to \chi in the original publication)
     - ``A``: the apoptosis rate for the prostate cancer (correspond to :math:`A` in the original publication

    :param phi: the FEniCS ``Function`` for \varphi
    :param phi_prec: the initial value for \varphi
    :param sigma: the FEniCS ``Function`` for the nutrients (\sigma)
    :param v: the Test Function to define the weak form
    :param parameters: the parameters of the equation as ``Parameters`` object. All the values listed in the
        documentation must be present
    :return: the UFL weak form for the prostate cancer equation
    """
    F = (((phi - phi_prec) / parameters.get_value("dt")) * v * fenics.dx) \
        + (parameters.get_value("lambda") * fenics.dot(fenics.grad(phi), fenics.grad(v)) * fenics.dx) \
        + ((1 / parameters.get_value("tau")) * df_dphi(phi, parameters.get_value("chempot_constant")) * v * fenics.dx) \
        + (- parameters.get_value("chi") * sigma * v * fenics.dx) \
        + (parameters.get_value("A") * phi * v * fenics.dx)

    return F


def prostate_cancer_nutrient_form(sigma: fenics.Function,
                                  sigma_old: fenics.Function,
                                  phi: fenics.Function,
                                  v: fenics.TestFunction,
                                  s: fenics.Function,
                                  parameters: mcfp.Parameters):
    r"""
    Builds the FEniCS UFL weak form for the nutrient equation reported by Lorenzo and collaborators
    :cite:`Lorenzo2016`. The name of the nutrient is \sigma.

    The parameters required for the equation must be specified in the ``parameters`` object. The name for the
    required parameters ore:
     - ``dt``: the time step (time discretization: backward Euler)
     - ``epsilon``: diffusion constant for \sigma (correspond to \lambda in the original paper)
     - ``delta``: the uptake rate of the nutrient by the cancer (correspond to \delta in the original paper)
     - ``gamma``: the decay rate for \sigma (correspong to \gamma in the original paper)

    :param sigma: the FEniCS ``Function`` for the \sigma variable
    :param sigma_old: the FEniCS ``Function`` for the initial value of the \sigma value
    :param phi: the FEniCS ``Function`` for the cancer
    :param v: the Test Function to define the weak form
    :param s: the FEniCS ``Function`` for the nutrient supply
    :param parameters: the parameters fo the equation as ``Parameters`` object. All the values listed in the
        documentation must be present
    :return: the UFL weak form for the nutrient equation
    """
    F = (((sigma - sigma_old) / parameters.get_value("dt")) * v * fenics.dx) \
        + (parameters.get_value("epsilon") * fenics.dot(fenics.grad(sigma), fenics.grad(v)) * fenics.dx) \
        + (- s * v * fenics.dx) \
        + (parameters.get_value("delta") * phi * v * fenics.dx) \
        + (parameters.get_value("gamma") * sigma * v * fenics.dx)

    return F
