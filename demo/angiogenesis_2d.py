import sys
import fenics
import time
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


class AngiogenesisInitialCondition(fenics.UserExpression):
    """Initial condition for the vessel field, the tumor field (c) and the angiogenic factor (T)"""
    def __init__(self, vessel_width, phi_max, phi_min, T0):
        super(AngiogenesisInitialCondition, self).__init__()
        self.vessel_width = vessel_width
        self.phi_max = phi_max
        self.phi_min = phi_min
        self.T0 = T0

    def eval(self, values, x):
        # set initial value to T
        values[0] = self.T0
        # set initial value to c
        if x[0] < self.vessel_width:
            values[1] = self.phi_max
        else:
            values[1] = self.phi_min
        # set initial value to um
        values[2] = 0

    def value_shape(self):
        return (3,)

    def __floordiv__(self, other):
        pass


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
data_folder = mansimd.setup_data_folder(folder_name="angiogenesis_2d",
                                        base_location=file_folder/Path("demo_out"),
                                        enumerate=False)

# define files for fields
file_names = ["c", "T", "grad_T", "tipcells"]
file_phi, file_T, file_grad_T, tipcells_xdmf = fu.setup_xdmf_files(file_names, data_folder)

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

# define test functions
v1, v2, v3 = fenics.TestFunctions(function_space)

# define functions
u_curr = fenics.Function(function_space)
T_old, c_old, mu_old = fenics.split(u_curr)
u = fenics.Function(function_space)
T, c, mu = fenics.split(u)
grad_T = fenics.Function(grad_T_function_space)
tipcells_field = fenics.Function(function_space.sub(0).collapse())

# define initial condition
phi_max = parameters.get_value("phi_max")
phi_min = parameters.get_value("phi_min")
T0 = 0.
u_curr.interpolate(AngiogenesisInitialCondition(initial_vessel_width, phi_max, phi_min, T0))

sources_map = af_sourcing.SourceMap(n_sources,
                                    initial_vessel_width + parameters.get_value("d"),
                                    mesh_wrapper,
                                    0,
                                    parameters)

# init sources manager
sources_manager = af_sourcing.SourcesManager(sources_map, mesh_wrapper, parameters, {"type": "None"})

# split u_curr
T_curr, phi_curr, mu_curr = u_curr.split()

# add sources to initial condition of T
sources_manager.apply_sources(T_curr, function_space.sub(0), True, 0.)

# compute gradient of T
grad_T.assign(fenics.project(fenics.grad(T_curr), grad_T_function_space))

# save to file
file_T.write(T_curr, 0)
file_phi.write(phi_curr, 0)
file_grad_T.write(grad_T, 0)

# define form for angiogenic factor
form_T = angiogenic_factor_form(T, T_old, c, v1, parameters)

# define form for angiogenesis
form_ang = angiogenesis_form(c, c_old, mu, mu_old, v2, v3, T, parameters)

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
    pbar = tqdm(total=n_steps, ncols=100, position=1, desc="angiogenesis_2d")
else:
    pbar = None

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
                  tip_cell_manager.move_tip_cells(phi_curr, T_curr, grad_T, function_space.sub(1), True))

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
    u_curr.assign(u)

    # split components of u
    T_curr, phi_curr, mu_curr = u_curr.split()

    # update source field
    sources_manager.apply_sources(T_curr, function_space.sub(0), True, t)

    # compute grad_T
    grad_T.assign(fenics.project(fenics.grad(T_curr), grad_T_function_space))

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
                      "angiogenesis_2d",
                      sim_rationale="Simulating angiogenesis 2d demo.")