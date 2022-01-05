import sys
import time
import fenics
from tqdm import tqdm
from pathlib import Path
file_folder = Path(__file__).parent.resolve()
mocafe_folder = file_folder.parent
sys.path.append(str(mocafe_folder))
import mocafe.fenut.fenut as fu
import mocafe.fenut.mansimdata as mansimd
from mocafe.angie import af_sourcing, tipcells
from mocafe.angie.forms import angiogenesis_form, angiogenic_factor_form
import mocafe.fenut.parameters as mpar

# initial setup
comm = fenics.MPI.comm_world
rank = comm.Get_rank()


def print_p0(msg: str):
    if rank == 0:
        print(msg)


# get simulation time
init_time = time.time()

n = 5
print_p0(f"1/{n} Setting up...")
# only process 0 logs
fenics.parameters["std_out_all_processes"] = False
# set log level ERROR
fenics.set_log_level(fenics.LogLevel.ERROR)
# define data folder
data_folder = mansimd.setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/angiogenesis_3d",
                                        auto_enumerate=False)

# setup xdmf files
file_names = ["c", "af", "tipcells", "mesh"]
file_c, file_af, tipcells_xdmf, mesh_xdmf = fu.setup_xdmf_files(file_names, data_folder)

# setup parameters
parameters_file = file_folder/Path("demo_in/angiogenesis_2d/parameters.ods")
parameters = mpar.from_ods_sheet(parameters_file, "SimParams")

# create mesh
Lx = parameters.get_value("Lx")
Ly = parameters.get_value("Ly")
Lz = 40  # 1/3 of Ly and Lx
nx = int(parameters.get_value("nx"))
ny = int(parameters.get_value("ny"))
nz = 40  # 1/3 of nx and ny
mesh_file = data_folder / Path("mesh.xdmf")
if mesh_file.exists():
    print_p0(f"2/{n} Detected mesh file, reading mesh data...")
    mesh = fenics.Mesh()
    mesh_xdmf.read(mesh)
else:
    print_p0(f"2/{n} Creating 3D mesh; this might take some time...")
    mesh = fenics.BoxMesh(fenics.Point(0., 0., 0.),
                          fenics.Point(Lx, Ly, Lz),
                          nx,
                          ny,
                          nz)
    mesh_xdmf.write(mesh)

# define function spaces
# for c and af
function_space = fu.get_mixed_function_space(mesh, 3, "CG", 1)
# for grad_T
grad_af_function_space = fenics.VectorFunctionSpace(mesh, "CG", 1)

# define initial conditions for c and mu
print_p0(f"3/{n} Defining initial conditions... ")
initial_vessel_radius = parameters.get_value("initial_vessel_width")
c_exp = fenics.Expression("((pow(x[0], 2) + pow(x[2] - Lz/2, 2)) < pow(R_v, 2)) ? 1 : -1",
                          degree=2,
                          R_v=initial_vessel_radius,
                          Lz=Lz)
c_0 = fenics.interpolate(c_exp, function_space.sub(0).collapse())
mu_0 = fenics.interpolate(fenics.Constant(0.), function_space.sub(0).collapse())

# define source map
n_sources = int(parameters.get_value("n_sources"))
cylinder_radius = initial_vessel_radius + parameters.get_value("d")
sources_map = af_sourcing.RandomSourceMap(mesh,
                                          n_sources,
                                          parameters,
                                          where=lambda x: (x[0]**2 + (x[2] - Lz/2)**2) > (cylinder_radius**2))
# define sources manager
sources_manager = af_sourcing.SourcesManager(sources_map, mesh, parameters)
# apply sources to af
af_0 = fenics.interpolate(fenics.Constant(0.), function_space.sub(0).collapse())
sources_manager.apply_sources(af_0)

# write initial conditions
file_af.write(af_0, 0)
file_c.write(c_0, 0)

# init tipcell field
tipcells_field = fenics.Function(function_space.sub(0).collapse())

# init grad af
grad_af = fenics.Function(grad_af_function_space)
grad_af.assign(  # assign to grad_af
    fenics.project(fenics.grad(af_0), grad_af_function_space)  # the projection on the fun space of grad(af_0)
)

print_p0(f"4/{n} Defining weak forms... ")
# init test functions
v1, v2, v3 = fenics.TestFunctions(function_space)

# init variables
u = fenics.Function(function_space)
af, c, mu = fenics.split(u)

# form
form_af = angiogenic_factor_form(af, af_0, c, v1, parameters)
form_ang = angiogenesis_form(c, c_0, mu, mu_0, v2, v3, af, parameters)
weak_form = form_af + form_ang

# init tip cell manager
tip_cell_manager = tipcells.TipCellManager(mesh,
                                           parameters)

jacobian = fenics.derivative(weak_form, u)

print_p0(f"5/{n} Starting iteration... ")
t = 0.
n_steps = int(parameters.get_value("n_steps"))
if rank == 0:
    pbar = tqdm(total=n_steps, ncols=100, position=1, desc="angiogenesis_3d")
else:
    pbar = None

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
    fenics.solve(weak_form == 0, u, J=jacobian)

    # assign u to the initial conditions functions
    fenics.assign([af_0, c_0, mu_0], u)

    # update source field
    sources_manager.apply_sources(af_0)

    # compute grad_T
    grad_af.assign(fenics.project(fenics.grad(af_0), grad_af_function_space))

    # save data
    file_af.write(af_0, t)
    file_c.write(c_0, t)
    tipcells_xdmf.write(tipcells_field, t)

    if rank == 0:
        pbar.update(1)

print("Ok!")

