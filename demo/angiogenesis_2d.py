import sys
import fenics
import time
import mshr
from tqdm import tqdm
from pathlib import Path
file_folder = Path(__file__).parent.resolve()
mocafe_folder = file_folder.parent
sys.path.append(str(mocafe_folder))
import mocafe.fenut.fenut as fu
import mocafe.fenut.mansimdata as mansimd
from mocafe.angie import af_sourcing, tipcells
from mocafe.angie.forms import angiogenesis_form, angiogenic_factor_form
from mocafe.fenut.parameters import from_ods_sheet
from mocafe.expressions import PythonFunctionField


# get comm and rank
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# measure time for simulation
init_time = time.time()
# only process 0 logs
fenics.parameters["std_out_all_processes"] = False
# set log level
fenics.set_log_level(fenics.LogLevel.ERROR)
# define data folder
data_folder = mansimd.setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/angiogenesis_2d",
                                        auto_enumerate=False)

# define files for fields
file_names = ["c", "af", "tipcells"]
file_c, file_af, tipcells_xdmf = fu.setup_xdmf_files(file_names, data_folder)

# import parametes
parameters_file = file_folder/Path("demo_in/angiogenesis_2d/parameters.ods")
parameters = from_ods_sheet(parameters_file, "SimParams")

# define mesh
Lx = parameters.get_value("Lx")
Ly = parameters.get_value("Ly")
nx = int(parameters.get_value("nx"))
ny = int(parameters.get_value("ny"))
mesh = fenics.RectangleMesh(fenics.Point(0., 0.),
                            fenics.Point(Lx, Ly),
                            nx,
                            ny)
mesh_wrapper = fu.MeshWrapper(mesh)

# define additional parameters
initial_vessel_width = parameters.get_value("initial_vessel_width")
n_sources = parameters.get_value("n_sources")

# define function space for T and c
function_space = fu.get_mixed_function_space(mesh, 3, "CG", 1)
# define function space for grad_T
grad_T_function_space = fenics.VectorFunctionSpace(mesh, "CG", 1)

# define initial conditions for af
af_0 = fenics.interpolate(fenics.Constant(0.), function_space.sub(0).collapse())
random_sources_domain = mshr.Rectangle(fenics.Point(initial_vessel_width + parameters.get_value("d"), 0),
                                       fenics.Point(Lx, Ly))
sources_map = af_sourcing.RandomSourceMap(mesh_wrapper,
                                          n_sources,
                                          parameters,
                                          where=random_sources_domain)
sources_manager = af_sourcing.SourcesManager(sources_map, mesh_wrapper, parameters)
sources_manager.apply_sources(af_0)

# define initial condition for c
c_0 = fenics.interpolate(PythonFunctionField(python_fun=lambda x: 1. if x[0] < initial_vessel_width else -1.),
                         function_space.sub(0).collapse())

# define initial condition for mu
mu_0 =fenics.interpolate(fenics.Constant(0.), function_space.sub(0).collapse())


# define test functions
v1, v2, v3 = fenics.TestFunctions(function_space)

# define functions
u = fenics.Function(function_space)
af, c, mu = fenics.split(u)
grad_af = fenics.Function(grad_T_function_space)
tipcells_field = fenics.Function(function_space.sub(0).collapse())

# compute gradient of T
grad_af.assign(fenics.project(fenics.grad(af_0), grad_T_function_space))

# save to file
file_af.write(af_0, 0)
file_c.write(c_0, 0)

# define form for angiogenic factor
form_af = angiogenic_factor_form(af, af_0, c, v1, parameters)

# define form for angiogenesis
form_ang = angiogenesis_form(c, c_0, mu, mu_0, v2, v3, af, parameters)

# define complete form
F = form_af + form_ang

# define Jacobian
J = fenics.derivative(F, u)

# initialize time iteration
t = 0.
n_steps = int(parameters.get_value("n_steps"))
tip_cell_manager = tipcells.TipCellManager(mesh_wrapper,
                                           parameters)

# initialize progress bar
if rank == 0:
    pbar = tqdm(total=n_steps, ncols=100, position=1, desc="angiogenesis_2d")
else:
    pbar = None

# start iteration
for step in range(1, n_steps + 1):
    # update time
    t += parameters.get_value("dt")

    # turn off near sources
    sources_manager.remove_sources_near_vessels(c_0)

    # activate tip cell
    tip_cell_manager.activate_tip_cell(c_0, af_0, grad_af, step)

    # revert tip cells
    tip_cell_manager.revert_tip_cells(af_0, grad_af)

    # move tip cells
    tip_cell_manager.move_tip_cells(c_0, af_0, grad_af)

    # get tip cells field
    tipcells_field.assign(tip_cell_manager.get_latest_tip_cell_function())

    # update fields
    try:
        fenics.solve(F == 0, u, J=J)
    except RuntimeError as e:
        mansimd.save_sim_info(data_folder,
                              time.time() - init_time,
                              parameters,
                              "angiogenesis_2d",
                              error_msg=str(e),
                              sim_rationale="Simulating angiogenesis 2d demo.")
        raise e

    # assign u to u_curr
    fenics.assign([af_0, c_0, mu_0], u)

    # update source field
    sources_manager.apply_sources(af_0)

    # compute grad_T
    grad_af.assign(fenics.project(fenics.grad(af_0), grad_T_function_space))

    # save data
    if step % parameters.get_value("save_rate") == 0:
        file_af.write(af_0, t)
        file_c.write(c_0, t)
        tipcells_xdmf.write(tipcells_field, t)

    if rank == 0:
        pbar.update(1)

# save sim info
mansimd.save_sim_info(data_folder,
                      time.time() - init_time,
                      parameters,
                      "angiogenesis_2d",
                      sim_rationale="Simulating angiogenesis 2d demo.")
