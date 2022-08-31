"""
Test calling all forms defined in ``mocafe.litforms``
"""
import fenics
import pytest
from mocafe.litforms.prostate_cancer import prostate_cancer_form, prostate_cancer_nutrient_form
from mocafe.litforms.xu16 import xu_2016_cancer_form, xu2016_nutrient_form
from mocafe.fenut.parameters import from_dict


def test_calling_litforms():
    # define mesh
    n_x = n_y = 50
    mesh = fenics.UnitSquareMesh(n_x, n_y)

    # define function space
    element = fenics.FiniteElement("CG", fenics.triangle, 1)
    V = fenics.FunctionSpace(mesh, element)

    # define T, T0 and phi
    foo = fenics.Function(V)

    # define test functions
    v_foo = fenics.TestFunction(V)

    # define correct parameters for pc
    pc_parameters = from_dict({
        "dt": 0.001,  # years
        "lambda": 1.6E5,  # (um^2) / years
        "tau": 0.01,  # years
        "chempot_constant": 16,  # adimensional
        "chi": 600.0,  # Liters / (gram * years)
        "A": 600.0,  # 1 / years
        "epsilon": 5.0E6,  # (um^2) / years
        "delta": 1003.75,  # grams / (Liters * years)
        "gamma": 1000.0,  # grams / (Liters * years)
    })

    # try calling pc forms
    prostate_cancer_form(foo, foo, foo, v_foo, pc_parameters)
    prostate_cancer_nutrient_form(foo, foo, foo, v_foo, foo, pc_parameters)
    with pytest.warns(Warning):
        prostate_cancer_form(foo, foo, foo, v_foo, pc_parameters, dt=0.1)
    with pytest.warns(Warning):
        prostate_cancer_nutrient_form(foo, foo, foo, v_foo, foo, pc_parameters, dt=0.1)

    # try calling pc forms with wrong parameters
    pc_parameters = from_dict({"foo": 10})
    with pytest.raises(RuntimeError):
        prostate_cancer_form(foo, foo, foo, v_foo, pc_parameters)
    with pytest.raises(RuntimeError):
        prostate_cancer_nutrient_form(foo, foo, foo, v_foo, foo, pc_parameters)

    # define correct parameters for xu
    xu_parameters = from_dict({
        "dt": 10.,
        "D_sigma": 10.,
        "V_pc": 20.,
        "V_uT": 10.,
        "V_uH": 30.,
        "sigma_h_v": 10.,
        "M_phi": 10.,
        "lambda_phi": 10.
    })

    # try calling xu forms
    xu_2016_cancer_form(foo, foo, foo, v_foo, xu_parameters)
    xu2016_nutrient_form(foo, foo, foo, foo, v_foo, xu_parameters)
    with pytest.warns(Warning):
        xu_2016_cancer_form(foo, foo, foo, v_foo, xu_parameters, dt=0.1)
    with pytest.warns(Warning):
        xu2016_nutrient_form(foo, foo, foo, foo, v_foo, xu_parameters, dt=0.1)

    # try calling it with wrong parameters
    xu_parameters = from_dict({"foo": 10.})
    with pytest.raises(RuntimeError):
        xu_2016_cancer_form(foo, foo, foo, v_foo, xu_parameters)
    with pytest.raises(RuntimeError):
        xu2016_nutrient_form(foo, foo, foo, foo, v_foo, xu_parameters)
