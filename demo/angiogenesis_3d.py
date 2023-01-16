r"""
.. _Angiogenesis 3D Demo:

Angiogenesis 3D
===============

In this short demo we will show you how to simulate a phase field model described by Travasso et al. in 2011
:cite:`Travasso2011a` using FEniCS and Mocafe in 3D. You'll notice that the script is just the same of the 2D
demo: you just need to change the spatial domain!

.. contents:: Table of Contents
   :local:

How to run this example on Mocafe
---------------------------------
Make sure you have FEniCS and Mocafe and download the source script of this page (see above for the link).

Then, download the parameters file for the simulation from
:download:`this link<./demo_in/angiogenesis_3d/parameters.ods>` and place it inside the folder
``demo_in/angiogenesis_3d``:

.. code-block:: console

    mkdir demo_in
    mkdir demo_in/angiogenesis_3d
    mv parameters.ods demo_in/angiogenesis_3d/

Then, simply run it using python:

.. code-block:: console

    python3 angiogenesis_3d.py

However, it is recommended to exploit parallelization to save simulation time:

.. code-block:: console

    mpirun -n 4 python3 angiogenesis_3d.py

Notice that the number following the ``-n`` option is the number of MPI processes you're using for parallelizing the
simulation. You can change it accordingly with your CPU.

Note on 3D simulations
----------------------
The computational effort required to solve a system in 3D is of orders of magnitude higher than to solve the same
system in 2D. Thus, a normal laptop might be not able to compute the solution in reasonable time. Consider using
a powerful desktop computer or an HPC to simulate the system.

Visualize the results of this simulation
----------------------------------------
You need to have `Paraview <https://www.paraview.org/>`_ to visualize the results. Once you have installed it,
you can easly import the ``.xdmf`` files generated during the simulation and visualize the result.
"""

# %%
# Implementation
# --------------
#
# One of the great things of differential equations is that they are not really constrained to a specific space
# dimension. With appropriate initial and boundary conditions, you can possibly find the solution of a differential
# equation in any possible space. This is not always true for the software implementations of such differential
# equations; however, FEniCS and Mocafe are designed to follow just the same philosophy. So, you'll notice this
# script is extremely similar to the one used for the 2D simulation.
#
# Setup
# ^^^^^
# The setup is just the same as before; we just added a progress bar to follow the setup
# (that might take a while) and we changed the data folder, in order to separate the generated
# data. Also, notice that the parameters file is different. However, if you compare the file 
# with the one we provided you for the 2D examples, you'll notice that there are just small
# variations.
import fenics
from tqdm import tqdm
from pathlib import Path
import petsc4py
import mocafe.fenut.fenut as fu
import mocafe.fenut.mansimdata as mansimd
from mocafe.angie import af_sourcing, tipcells
from mocafe.angie.forms import angiogenesis_form, angiogenic_factor_form
import mocafe.fenut.parameters as mpar
from mocafe.fenut.solvers import SNESProblem

# get MPI _comm and _rank
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# create pbar for setup
if rank == 0:
    setup_pbar = tqdm(total=8, desc="setting up")
else:
    setup_pbar = None

# only process 0 logs
fenics.parameters["std_out_all_processes"] = False
# set log level ERROR
fenics.set_log_level(fenics.LogLevel.ERROR)
# define data folder
file_folder = Path(__file__).parent.resolve()
data_folder = mansimd.setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/angiogenesis_3d",
                                        auto_enumerate=False)

# setup xdmf files
file_names = ["c", "af", "tipcells", "mesh"]
file_c, file_af, tipcells_xdmf, mesh_xdmf = fu.setup_xdmf_files(file_names, data_folder)

# setup parameters
file_folder = Path(__file__).parent.resolve()
parameters_file = file_folder/Path("demo_in/angiogenesis_3d/parameters.ods")
parameters = mpar.from_ods_sheet(parameters_file, "SimParams")

# %%
# Definition of the spatial domain and the function space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This is the one of the only changes we need to do: we need to define a 3D domain. However, we can do that with ease
# using a ``BoxMesh``. Of course, creating a 3D mesh takes longer than a 2D mesh; thus, we placed an if statement to
# make FEniCS generate the mesh at the first run of the script, save it in the data folder, and reload it in all the
# following runs.
Lx = parameters.get_value("Lx")
Ly = parameters.get_value("Ly")
Lz = parameters.get_value("Lz")
nx = int(parameters.get_value("nx"))
ny = int(parameters.get_value("ny"))
nz = int(parameters.get_value("nz"))
mesh_file = data_folder / Path("mesh.xdmf")

# check if mesh has already been created
if mesh_file.exists():
    if rank == 0:
        setup_pbar.update(1)
        setup_pbar.set_description("loading mesh")

    # in the case, load it
    mesh = fenics.Mesh()
    mesh_xdmf.read(mesh)
else:
    if rank == 0:
        setup_pbar.update(1)
        setup_pbar.set_description("creating mesh")

    # create mesh
    mesh = fenics.BoxMesh(fenics.Point(0., 0., 0.),
                          fenics.Point(Lx, Ly, Lz),
                          nx,
                          ny,
                          nz)
    # read it to file for following runs
    mesh_xdmf.write(mesh)

# %%
# From the mesh, we can again define the function space in the same way we did in the 2D simulation. Indeed, the
# system of differential equations is the same and FEniCS will take care of defining the "3D-version" of the polynomial
# functions. Remember that, even though there are just two variables :math:`c` and :math:`af`, we also need to
# consider an auxiliary variable :math:`\mu` for the :math:`c` field (see demo for the 2D case).

# for c and af
function_space = fu.get_mixed_function_space(mesh, 3, "CG", 1)
# for grad_T
grad_af_function_space = fenics.VectorFunctionSpace(mesh, "CG", 1)

# %%
# Initial & boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Again, in this implementation we will consider natural Neumann boundary conditions for both :math:`c` and
# :math`af`.
#
# As initial condition for :math:`c`, the most natural choice to resemble the results of Travasso and his collaborators
# :cite:`Travasso2011a` is to define a cylindrical blood vessel on one side of the mesh. To do so, we will use again
# the standard fenics interface for defining an ``Expression``:

if rank == 0:
    setup_pbar.update(1)
    setup_pbar.set_description("generating initial conditions")

initial_vessel_radius = parameters.get_value("initial_vessel_width")
c_exp = fenics.Expression("((pow(x[0], 2) + pow(x[2] - Lz/2, 2)) < pow(R_v, 2)) ? 1 : -1",
                          degree=2,
                          R_v=initial_vessel_radius,
                          Lz=Lz)

if rank == 0:
    setup_pbar.update(1)
    setup_pbar.set_description("interpolating c_0 and af_0")

c_0 = fenics.interpolate(c_exp, function_space.sub(0).collapse())
mu_0 = fenics.interpolate(fenics.Constant(0.), function_space.sub(0).collapse())

# %%
# As initial condition for :math:`af`, we can just use the ``RandomSourceMap`` object and the ``SourcesManager`` just
# as we did in the 2D demo. Both of them are indeed designed to work just the same in 2D and 3D, with the only
# difference that, in 3D, the cells are spheres instead of circles.

# define source map
if rank == 0:
    setup_pbar.update(1)
    setup_pbar.set_description("creating sources map")

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

if rank == 0:
    setup_pbar.update(1)
    setup_pbar.set_description("applying sources")

sources_manager.apply_sources(af_0)

# write initial conditions
file_af.write(af_0, 0)
file_c.write(c_0, 0)

# init tipcell field
tipcells_field = fenics.Function(function_space.sub(0).collapse())

# init grad af
if rank == 0:
    setup_pbar.update(1)
    setup_pbar.set_description("projecting grad_af")

grad_af = fenics.Function(grad_af_function_space)
grad_af.assign(  # assign to grad_af
    fenics.project(fenics.grad(af_0), grad_af_function_space,
                   solver_type="gmres", preconditioner_type="amg")  # the projection on the fun space of grad(af_0)
)

# %%
# PDE System definition
# ^^^^^^^^^^^^^^^^^^^^^
# Exactly how the differential equations don't change from 2D to 3D, the PDE definition remains the same. Indeed,
# you can notice that the code it's just identical to the 2D demo, except for the update of ``setup_pbar``:

if rank == 0:
    setup_pbar.update(1)
    setup_pbar.set_description("defining weak form")

# init test functions
v1, v2, v3 = fenics.TestFunctions(function_space)

# init variables
u = fenics.Function(function_space)
af, c, mu = fenics.split(u)

# form
form_af = angiogenic_factor_form(af, af_0, c, v1, parameters)
form_ang = angiogenesis_form(c, c_0, mu, mu_0, v2, v3, af, parameters)
weak_form = form_af + form_ang

# %%
# Simulation setup
# ^^^^^^^^^^^^^^^^
# Now that everything is set up we can proceed to the actual simulation, that, just as before, will start with the
# definition of the ``TipCellsManager``:
tip_cell_manager = tipcells.TipCellManager(mesh,
                                           parameters)

# update
if rank == 0:
    setup_pbar.update(1)
    setup_pbar.set_description("starting simulation")

# %%
# After that, everything will just work the same. For efficiency, we make use of the PETSc SNES solver to solve the
# differential equations this time, but this is the only change we made to the 2D demo code.
t = 0.
n_steps = 200
if rank == 0:
    pbar = tqdm(total=n_steps, ncols=100, position=1, desc="simulation")
else:
    pbar = None

petsc4py.init([__name__,
               "-snes_type", "newtonls",
               "-ksp_type", "gmres",
               "-pc_type", "gamg"])
from petsc4py import PETSc

# create snes solver
snes_solver = PETSc.SNES().create(comm)
snes_solver.setFromOptions()

# start iteration in time
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

    # solve the problem with the solver defined by the given parameters
    problem = SNESProblem(weak_form, u, [])
    b = fenics.PETScVector()
    J_mat = fenics.PETScMatrix()
    snes_solver.setFunction(problem.F, b.vec())
    snes_solver.setJacobian(problem.J, J_mat.mat())
    snes_solver.solve(None, u.vector().vec())

    # assign u to the initial conditions functions
    fenics.assign([af_0, c_0, mu_0], u)

    # update source field
    sources_manager.apply_sources(af_0)

    # compute grad_T
    grad_af.assign(
        fenics.project(fenics.grad(af_0), grad_af_function_space,
                       solver_type="gmres", preconditioner_type="amg")
    )

    # save data
    file_af.write(af_0, t)
    file_c.write(c_0, t)
    tipcells_xdmf.write(tipcells_field, t)

    if rank == 0:
        pbar.update(1)

# %%
# Result
# ------
# We uploaded on YouTube the result on this simulation. You can check it out below or at
# `this link <https://youtu.be/ho-V58mqDv8>`_
#
# ..  youtube:: ho-V58mqDv8
#
# Visualize the result with ParaView
# ----------------------------------
# The result of the simulation is stored in the ``.xdmf`` file generated, which are easy to load and visualize in
# external software as ParaView. If you don't know how to do it, you can check out the tutorial below or at
# `this YouTube link <https://youtu.be/ATzlVEIjicI>`_.
#
# ..  youtube:: ATzlVEIjicI
#

# %%
# Full code
# ---------
#
# .. code-block:: default
#
#   import fenics
#   from tqdm import tqdm
#   from pathlib import Path
#   import petsc4py
#   import mocafe.fenut.fenut as fu
#   import mocafe.fenut.mansimdata as mansimd
#   from mocafe.angie import af_sourcing, tipcells
#   from mocafe.angie.forms import angiogenesis_form, angiogenic_factor_form
#   import mocafe.fenut.parameters as mpar
#   from mocafe.fenut.solvers import SNESProblem
#
#   # get MPI _comm and _rank
#   _comm = fenics.MPI.comm_world
#   _rank = _comm.Get_rank()
#
#   # create pbar for setup
#   if _rank == 0:
#       setup_pbar = tqdm(total=8, desc="setting up")
#   else:
#       setup_pbar = None
#
#   # only process 0 logs
#   fenics.parameters["std_out_all_processes"] = False
#   # set log level ERROR
#   fenics.set_log_level(fenics.LogLevel.ERROR)
#   # define data folder
#   file_folder = Path(__file__).parent.resolve()
#   data_folder = mansimd.setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/angiogenesis_3d",
#                                           auto_enumerate=False)
#
#   # setup xdmf files
#   file_names = ["c", "af", "tipcells", "mesh"]
#   file_c, file_af, tipcells_xdmf, mesh_xdmf = fu.setup_xdmf_files(file_names, data_folder)
#
#   # setup parameters
#   file_folder = Path(__file__).parent.resolve()
#   parameters_file = file_folder/Path("demo_in/angiogenesis_3d/parameters.ods")
#   parameters = mpar.from_ods_sheet(parameters_file, "SimParams")
#
#   Lx = parameters.get_value("Lx")
#   Ly = parameters.get_value("Ly")
#   Lz = parameters.get_value("Lz")
#   nx = int(parameters.get_value("nx"))
#   ny = int(parameters.get_value("ny"))
#   nz = int(parameters.get_value("nz"))
#   mesh_file = data_folder / Path("mesh.xdmf")
#
#   # check if mesh has already been created
#   if mesh_file.exists():
#       if _rank == 0:
#           setup_pbar.update(1)
#           setup_pbar.set_description("loading mesh")
#
#       # in the case, load it
#       mesh = fenics.Mesh()
#       mesh_xdmf.read(mesh)
#   else:
#       if _rank == 0:
#           setup_pbar.update(1)
#           setup_pbar.set_description("creating mesh")
#
#       # create mesh
#       mesh = fenics.BoxMesh(fenics.Point(0., 0., 0.),
#                             fenics.Point(Lx, Ly, Lz),
#                             nx,
#                             ny,
#                             nz)
#       # read it to file for following runs
#       mesh_xdmf.write(mesh)
#
#
#   # for c and af
#   function_space = fu.get_mixed_function_space(mesh, 3, "CG", 1)
#   # for grad_T
#   grad_af_function_space = fenics.VectorFunctionSpace(mesh, "CG", 1)
#
#
#   if _rank == 0:
#       setup_pbar.update(1)
#       setup_pbar.set_description("generating initial conditions")
#
#   initial_vessel_radius = parameters.get_value("initial_vessel_width")
#   c_exp = fenics.Expression("((pow(x[0], 2) + pow(x[2] - Lz/2, 2)) < pow(R_v, 2)) ? 1 : -1",
#                             degree=2,
#                             R_v=initial_vessel_radius,
#                             Lz=Lz)
#
#   if _rank == 0:
#       setup_pbar.update(1)
#       setup_pbar.set_description("interpolating c_0 and af_0")
#
#   c_0 = fenics.interpolate(c_exp, function_space.sub(0).collapse())
#   mu_0 = fenics.interpolate(fenics.Constant(0.), function_space.sub(0).collapse())
#
#
#   # define source map
#   if _rank == 0:
#       setup_pbar.update(1)
#       setup_pbar.set_description("creating sources map")
#
#   n_sources = int(parameters.get_value("n_sources"))
#   cylinder_radius = initial_vessel_radius + parameters.get_value("d")
#   sources_map = af_sourcing.RandomSourceMap(mesh,
#                                             n_sources,
#                                             parameters,
#                                             where=lambda x: (x[0]**2 + (x[2] - Lz/2)**2) > (cylinder_radius**2))
#   # define sources manager
#   sources_manager = af_sourcing.SourcesManager(sources_map, mesh, parameters)
#   # apply sources to af
#   af_0 = fenics.interpolate(fenics.Constant(0.), function_space.sub(0).collapse())
#
#   if _rank == 0:
#       setup_pbar.update(1)
#       setup_pbar.set_description("applying sources")
#
#   sources_manager.apply_sources(af_0)
#
#   # write initial conditions
#   file_af.write(af_0, 0)
#   file_c.write(c_0, 0)
#
#   # init tipcell field
#   tipcells_field = fenics.Function(function_space.sub(0).collapse())
#
#   # init grad af
#   if _rank == 0:
#       setup_pbar.update(1)
#       setup_pbar.set_description("projecting grad_af")
#
#   grad_af = fenics.Function(grad_af_function_space)
#   grad_af.assign(  # assign to grad_af
#       fenics.project(fenics.grad(af_0), grad_af_function_space,
#                      solver_type="gmres", preconditioner_type="amg")  # the projection on the fun space of grad(af_0)
#   )
#
#
#   if _rank == 0:
#       setup_pbar.update(1)
#       setup_pbar.set_description("defining weak form")
#
#   # init test functions
#   v1, v2, v3 = fenics.TestFunctions(function_space)
#
#   # init variables
#   u = fenics.Function(function_space)
#   af, c, mu = fenics.split(u)
#
#   # form
#   form_af = angiogenic_factor_form(af, af_0, c, v1, parameters)
#   form_ang = angiogenesis_form(c, c_0, mu, mu_0, v2, v3, af, parameters)
#   weak_form = form_af + form_ang
#
#   tip_cell_manager = tipcells.TipCellManager(mesh,
#                                              parameters)
#
#   # update
#   if _rank == 0:
#       setup_pbar.update(1)
#       setup_pbar.set_description("starting simulation")
#
#   t = 0.
#   n_steps = 200
#   if _rank == 0:
#       pbar = tqdm(total=n_steps, ncols=100, position=1, desc="simulation")
#   else:
#       pbar = None
#
#   petsc4py.init([__name__,
#                  "-snes_type", "newtonls",
#                  "-ksp_type", "gmres",
#                  "-pc_type", "gamg"])
#   from petsc4py import PETSc
#
#   # create snes solver
#   snes_solver = PETSc.SNES().create(_comm)
#   snes_solver.setFromOptions()
#
#   # start iteration in time
#   for step in range(1, n_steps + 1):
#       # update time
#       t += parameters.get_value("dt")
#
#       # turn off near sources
#       sources_manager.remove_sources_near_vessels(c_0)
#
#       # activate tip cell
#       tip_cell_manager.activate_tip_cell(c_0, af_0, grad_af, step)
#
#       # revert tip cells
#       tip_cell_manager.revert_tip_cells(af_0, grad_af)
#
#       # move tip cells
#       tip_cell_manager.move_tip_cells(c_0, af_0, grad_af)
#
#       # get tip cells field
#       tipcells_field.assign(tip_cell_manager.get_latest_tip_cell_function())
#
#       # solve the problem with the solver defined by the given parameters
#       problem = SNESProblem(weak_form, u, [])
#       b = fenics.PETScVector()
#       J_mat = fenics.PETScMatrix()
#       snes_solver.setFunction(problem.F, b.vec())
#       snes_solver.setJacobian(problem.J, J_mat.mat())
#       snes_solver.solve(None, u.vector().vec())
#
#       # assign u to the initial conditions functions
#       fenics.assign([af_0, c_0, mu_0], u)
#
#       # update source field
#       sources_manager.apply_sources(af_0)
#
#       # compute grad_T
#       grad_af.assign(
#           fenics.project(fenics.grad(af_0), grad_af_function_space,
#                          solver_type="gmres", preconditioner_type="amg")
#       )
#
#       # save data
#       file_af.write(af_0, t)
#       file_c.write(c_0, t)
#       tipcells_xdmf.write(tipcells_field, t)
#
#       if _rank == 0:
#           pbar.update(1)
