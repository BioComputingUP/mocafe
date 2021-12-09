import fenics
import mocafe.fenut.parameters as mcfp


def prostate_cancer_chem_potential(phi: fenics.Function,
                                   chem_potential_constant):
    var_phi = fenics.variable(phi)
    chem_potential = chem_potential_constant * (var_phi ** 2) * ((1 - var_phi) ** 2)
    return chem_potential


def df_dphi(phi, chem_potential_constant):
    # var_phi = fenics.variable(phi)
    # c1 = potential_constant * 2
    # c2 = - potential_constant * 6
    # c3 = potential_constant * 4
    chem_potential = prostate_cancer_chem_potential(phi, chem_potential_constant)
    return fenics.diff(chem_potential)


def prostate_cancer_form(phi: fenics.Function,
                         phi_prec: fenics.Function,
                         sigma: fenics.Function,
                         v: fenics.TestFunction,
                         parameters: mcfp.Parameters):
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
                                  s_expression: fenics.Expression,
                                  parameters: mcfp.Parameters):
    F = (((sigma - sigma_old) / parameters.get_value("dt")) * v * fenics.dx) \
        + (parameters.get_value("epsilon") * fenics.dot(fenics.grad(sigma), fenics.grad(v)) * fenics.dx) \
        + (- s_expression * v * fenics.dx) \
        + (parameters.get_value("delta") * phi * v * fenics.dx) \
        + (parameters.get_value("gamma") * sigma * v * fenics.dx)

    return F
