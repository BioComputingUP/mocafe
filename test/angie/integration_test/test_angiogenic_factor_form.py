import fenics
import mocafe.fenut.fenut as fu
from mocafe.angie.forms import angiogenic_factor_form
from mocafe.angie.af_sourcing import SourceMap, SourcesManager
import numpy as np
import os
import logging

logging.StreamHandler().setLevel(logging.ERROR)


def test_angiogenic_factor_form(parameters):
    # setup data folder
    data_folder = f"{os.path.dirname(__file__)}/input_test_files/test_angiogenic_factor_form"

    # define files
    file_names = ["T_checkpoint"]
    file_T_checkpoint, = fu.setup_xdmf_files(file_names, data_folder)

    # define mesh
    n_x = n_y = 300
    mesh_wrapper = fu.RectangleMeshWrapper(fenics.Point(0., 0.), fenics.Point(n_x, n_y), n_x, n_y)
    mesh = mesh_wrapper.get_local_mesh()

    # define function space
    element = fenics.FiniteElement("CG", fenics.triangle, 1)
    V = fenics.FunctionSpace(mesh, element)

    # define source map
    source_points = [np.array([num, num]) for num in range(0, 310, 10)]
    sources_map = SourceMap(0, 10, mesh_wrapper, 0, parameters, source_points=source_points)

    # define sources manager
    sources_manager = SourcesManager(sources_map, mesh_wrapper, parameters, {"type": "None"})

    # define T and T0
    T = fenics.Function(V)
    T0 = fenics.Function(V)

    # apply sources on T0
    sources_manager.apply_sources(T0, V, False, 0)

    # define test functions
    v1 = fenics.TestFunction(V)

    # define phi function (static)
    phi_exp = fenics.Expression("x[0] < 150 ? 1. : 0.", degree=1)

    # interpolate function
    phi = fenics.interpolate(phi_exp, V)

    # define variational formulation
    F = angiogenic_factor_form(T, T0, phi, v1, parameters)

    # define jacobian
    J = fenics.derivative(F, T)

    # start iteraton
    t = 0.
    step = 0
    n_steps = 2
    dt = parameters.get_value("dt")
    while step <= n_steps - 1:
        # update time and step
        t += dt
        step += 1

        # solve variational problem
        fenics.solve(F == 0, T, J=J)

        # remove sources near the vessels
        sources_manager.remove_sources_near_vessels(phi)

        # assign T to T0
        T0.assign(T)

        # re-apply sources to T0
        sources_manager.apply_sources(T0, V, False, t)

    # write reference file (use only to reset test)
    # file_T_checkpoint.write_checkpoint(T0, "T", 0, fenics.XDMFFile.Encoding.HDF5, False)
    # file_T_checkpoint.close()

    # read
    T_ref = fenics.Function(V)
    file_T_checkpoint.read_checkpoint(T_ref, "T", 0)

    # create files for T and T ref and write them for visual check
    file_T0 = fenics.XDMFFile(data_folder + "/T.xdmf")
    file_T_ref = fenics.XDMFFile(data_folder + "/T_ref.xdmf")
    file_T0.write(T0)
    file_T_ref.write(T_ref)

    # compare
    assert np.allclose(T0.vector().get_local(), T_ref.vector().get_local()), "They should be the same"
