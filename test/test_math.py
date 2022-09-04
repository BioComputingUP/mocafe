import fenics
import numpy as np
from mocafe.math import estimate_cancer_area, estimate_capillaries_area


def test_area_estimators():
    # create mesh
    mesh = fenics.UnitSquareMesh(20, 20)
    # create function space
    V = fenics.FunctionSpace(mesh, "CG", 1)
    # create test fields
    cancer_field_full = fenics.interpolate(fenics.Constant(1), V)
    cancer_field_half = fenics.interpolate(fenics.Expression("x[0] < 0.5 ? 1 : 0.", degree=1),
                                           V)
    cancer_field_empty = fenics.interpolate(fenics.Constant(0.), V)
    capillaries_field_half = fenics.interpolate(fenics.Expression("x[0] < 0.5 ? 1 : -1", degree=1),
                                                V)
    capillaries_field_half2 = fenics.interpolate(fenics.Expression("x[0] < 0.5 ? 2 : -1", degree=1),
                                                 V)
    # test
    assert np.isclose(estimate_cancer_area(cancer_field_full), 1.), f"Should be 1."
    assert np.isclose(estimate_capillaries_area(cancer_field_full), 1.), "Should be 1."
    assert np.isclose(estimate_cancer_area(cancer_field_half), 0.5, atol=0.05), "Should be 0.5"
    assert np.isclose(estimate_capillaries_area(capillaries_field_half), 0.5, atol=0.05), "Should be 0.5"
    assert np.isclose(estimate_capillaries_area(capillaries_field_half2), 0.5, atol=0.05), "Should be 0.5"
    assert estimate_cancer_area(cancer_field_empty) < 1e-8, "Should be close to 0."
    assert estimate_capillaries_area(cancer_field_empty) < 1e8, "Should be close to 0."