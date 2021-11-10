import fenics
import time
import mocafe.fenut.fenut as fu
import mocafe.fenut.mansimdata as mansimd
import logging
import pathlib
from mocafe.angie import af_sourcing, tipcells
from mocafe.angie.expressions import AngiogenesisInitialCondition
from mocafe.angie.forms import angiogenesis_form, angiogenic_factor_form
from tqdm import tqdm
from mocafe.fenut.parameters import Parameters

"""
FEniCS implementation of the angiogenesis model formulated by Travasso et al. 
Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0019989

Finally working.
"""


def run_simulation(parameters_file: pathlib.Path, expression_function_parameters: dict,
                   sim_name=None, sim_rationale=None, parameter_to_vary=None, new_parameter_value=None):
    """
    Run the simulation of the 2D Travasso Angiogenesis model based on the given parameters file. The model is defined
    according to 2011 paper by Travassi et al (url:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0019989).
    It is possible to define a name for the simulation, that will be used for creating the folder with the result of the
    simulation inside the "runtime" folder.

    :param parameters_file: name of the .json parameter file to use to set up the simulation
    :param expression_function_parameters: parameters for the expression of VEGF
    :param sim_name: name of the simulation. It will be used to create the folder where the rusults of the simulation
        will be stored
    :param sim_rationale: rationale for the simulation
    :param parameter_to_vary: if necessary, the user can specify a parameter to vary in the simulation
    :param new_parameter_value: new value for the parameter
    :return: nothing.
    """
    # get process rank
    rank = fenics.MPI.comm_world.Get_rank()

    # measure time for simulation
    init_time = time.time()
    # only process 0 logs
    fenics.parameters["std_out_all_processes"] = False
    # set log level
    fenics.set_log_level(fenics.LogLevel.ERROR)
    logging.getLogger("mocafe.angie.tipcells").setLevel(logging.ERROR)
    logging.getLogger("mocafe.angie.af_sourcing").setLevel(logging.ERROR)
    # define data folder
    data_folder = mansimd.setup_data_folder(sim_name)

    # define files for fields
    file_names = ["phi", "T", "grad_T", "tipcells"]
    file_phi, file_T, file_grad_T, tipcells_xdmf = fu.setup_xdmf_files(file_names, data_folder)

    # import parametes
    parameters = Parameters(parameters_file, "SimParams")

    # change parameter value if necessary
    if parameter_to_vary is not None:
        parameters.set_value(parameter_to_vary, new_parameter_value)

    # define mesh
    Lx = parameters.get_value("Lx")
    Ly = parameters.get_value("Ly")
    nx = int(parameters.get_value("nx"))
    ny = int(parameters.get_value("ny"))
    mesh_wrapper = fu.RectangleMeshWrapper(fenics.Point(0., 0.),
                                           fenics.Point(Lx, Ly),
                                           nx,
                                           ny)
    mesh = mesh_wrapper.get_local_mesh()

    # define additional parameters
    initial_vessel_width = parameters.get_value("initial_vessel_width")
    n_sources = parameters.get_value("n_sources")

    # define function space for T and phi
    CG1 = fenics.FiniteElement("CG", fenics.triangle, 1)
    mixed_element = fenics.MixedElement([CG1, CG1, CG1])
    V1 = fenics.FunctionSpace(mesh, mixed_element)
    # define function space for grad_T
    V2 = fenics.VectorFunctionSpace(mesh, "CG", 1)

    # define test functions
    v1, v2, v3 = fenics.TestFunctions(V1)

    # define functions
    u_curr = fenics.Function(V1)
    T_old, phi_old, mu_old = fenics.split(u_curr)
    u = fenics.Function(V1)
    T, phi, mu = fenics.split(u)
    grad_T = fenics.Function(V2)
    tipcells_field = fenics.Function(V1.sub(0).collapse())

    # define initial condition
    phi_max = parameters.get_value("phi_max")
    phi_min = parameters.get_value("phi_min")
    T0 = 0.
    u_curr.interpolate(AngiogenesisInitialCondition(initial_vessel_width, phi_max, phi_min, T0))

    # set sources
    # c_x = parameters["ExpressionFunction"]["parameters"]["x_center"]
    # c_y = parameters["ExpressionFunction"]["parameters"]["y_center"]
    # circle_center = np.array([c_x, c_y])
    # circle_radius = parameters["ExpressionFunction"]["parameters"]["radius"]
    # source_points = angiogenicFactors.sources_in_circle_points(circle_center,
    #                                                            circle_radius,
    #                                                            parameters["R_c"]["value"])
    sources_map = af_sourcing.SourceMap(n_sources,
                                        initial_vessel_width + parameters.get_value("d"),
                                        mesh_wrapper,
                                        0,
                                        parameters)

    # init sources manager
    sources_manager = af_sourcing.SourcesManager(sources_map, mesh_wrapper, parameters, expression_function_parameters)

    # split u_curr
    T_curr, phi_curr, mu_curr = u_curr.split()

    # add sources to initial condition of T
    sources_manager.apply_sources(T_curr, V1.sub(0), True, 0.)

    # compute gradient of T
    grad_T.assign(fenics.project(fenics.grad(T_curr), V2))

    # save to file
    file_T.write(T_curr, 0)
    file_phi.write(phi_curr, 0)
    file_grad_T.write(grad_T, 0)

    # define form for angiogenic factor
    form_T = angiogenic_factor_form(T, T_old, phi, v1, parameters)

    # define form for angiogenesis
    form_ang = angiogenesis_form(phi, phi_old, mu, mu_old, v2, v3, T, parameters)

    # define complete form
    F = form_T + form_ang

    # define Jacobian
    J = fenics.derivative(F, u)

    # initialize time iteration
    t = 0.
    n_steps = int(parameters.get_value("n_steps"))
    tip_cell_manager = tipcells.TipCellManager(mesh_wrapper,
                                               parameters)

    # initialize progress bar
    if rank == 0:
        pbar = tqdm(total=n_steps, ncols=100, position=1, desc=sim_name)

    # start iteration
    for step in range(1, n_steps + 1):
        # update time
        t += parameters.get_value("dt")

        # turn off near sources
        sources_manager.remove_sources_near_vessels(phi_curr)

        # activate tip cell
        tip_cell_manager.activate_tip_cell(phi_curr, T_curr, grad_T, step)

        # revert tip cells
        tip_cell_manager.revert_tip_cells(T_curr, grad_T)

        # move tip cells
        fenics.assign(tipcells_field,
                      tip_cell_manager.move_tip_cells(phi_curr, T_curr, grad_T, V1.sub(1), True))

        # update fields
        try:
            fenics.solve(F == 0, u, J=J)
        except RuntimeError as e:
            mansimd.save_sim_info(data_folder,
                                  time.time() - init_time,
                                  parameters,
                                  sim_name,
                                  sim_rationale="RUNTIME ERROR OCCURRED DURING THE SIMULATION. \n" + sim_rationale)
            raise e

        # assign u to u_curr
        u_curr.assign(u)

        # split components of u
        T_curr, phi_curr, mu_curr = u_curr.split()

        # update source field
        sources_manager.apply_sources(T_curr, V1.sub(0), True, t)

        # compute grad_T
        grad_T.assign(fenics.project(fenics.grad(T_curr), V2))

        # save data
        if step % parameters.get_value("save_rate") == 0:
            file_T.write(T_curr, t)
            file_phi.write(phi_curr, t)
            file_grad_T.write(grad_T, t)
            tipcells_xdmf.write(tipcells_field, t)

        if rank == 0:
            pbar.update(1)

    # save sim info
    mansimd.save_sim_info(data_folder,
                          time.time() - init_time,
                          parameters,
                          sim_name,
                          sim_rationale=sim_rationale)
