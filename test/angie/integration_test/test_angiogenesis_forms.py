import fenics
from mocafe.angie.forms import angiogenic_factor_form, angiogenesis_form
import pytest
from mocafe.fenut.parameters import from_dict


def test_angiogenesis_forms(parameters):
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

    # test if forms are called correctly (the following statements should not raise error)
    F1 = angiogenic_factor_form(foo, foo, foo, v_foo, parameters)
    F2 = angiogenesis_form(foo, foo, foo, foo, v_foo, v_foo, foo, parameters)
    F = F1 + F2

    # test if overriding parameter raise warning
    with pytest.warns(Warning):
        angiogenic_factor_form(foo, foo, foo, v_foo, parameters, alpha_T=10)
    with pytest.warns(Warning):
        angiogenesis_form(foo, foo, foo, foo, v_foo, v_foo, foo, parameters, alpha_p=10)

    # test forms when parameters are wrong
    parameters = from_dict({"foo": 10})
    with pytest.raises(RuntimeError):
        angiogenic_factor_form(foo, foo, foo, v_foo, parameters)
    with pytest.raises(RuntimeError):
        angiogenesis_form(foo, foo, foo, foo, v_foo, v_foo, foo, parameters)
