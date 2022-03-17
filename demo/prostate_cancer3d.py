r"""
.. _Prostate Cancer 3D Demo:

Prostate cancer 3D
==================

In this short demo we will show you how to simulate a phase field model described by G. Lorenzo and collaborators
in 2016 :cite:`Lorenzo2016` using FEniCS and Mocafe in 3D. You'll notice that the script is just the same of the 2D
demo: you just need to change the spatial domain!

.. contents:: Table of Contents
   :local:

How to run this example on Mocafe
---------------------------------
Make sure you have FEniCS and Mocafe and download the source script of this page (see above for the link).
Then, simply run it using python:

.. code-block:: console

    python3 prostate_cancer3d.py

However, it is recommended to exploit parallelization to save simulation time:

.. code-block:: console

    mpirun -n 4 python3 prostate_cancer3d.py

Notice that the number following the ``-n`` option is the number of MPI processes you using for parallelizing the
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
# ------------------------------------------
#
# One of the great things of differential equations is that they are not really constrained to a specific space
# dimension. With appropriate initial and boundary conditions, you can find the solution of a differential equation
# in any possible space. This is not always true for the software implementations of such differential equations;
# however, using FEniCS the script is just slightly different from the one we've presented in the 2D case.

# Setup
# ^^^^^
# The setup is just the same as in the 2D case; we can even use the same parameters. Of course, the data folder
# changed, in order to store the 2D and 3D simulation result in different locations.
import numpy as np
import fenics
import petsc4py
from tqdm import tqdm
from pathlib import Path
from mocafe.fenut.solvers import SNESProblem
from mocafe.fenut.fenut import get_mixed_function_space, setup_xdmf_files
from mocafe.fenut.mansimdata import setup_data_folder
from mocafe.expressions import EllipsoidField
from mocafe.fenut.parameters import from_dict
import mocafe.litforms.prostate_cancer as pc_model

comm = fenics.MPI.comm_world
rank = comm.Get_rank()

file_folder = Path(__file__).parent.resolve()
data_folder = setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/prostate_cancer_3d",
                                auto_enumerate=False)

phi_xdmf, sigma_xdmf = setup_xdmf_files(["phi", "sigma"], data_folder)

parameters = from_dict({
    "phi0_in": 1.,  # adimentional
    "phi0_out": 0.,  # adimdimentional
    "sigma0_in": 0.2,  # adimentional
    "sigma0_out": 1.,  # adimentional
    "dt": 0.001,  # years
    "lambda": 1.6E5,  # (um^2) / years
    "tau": 0.01,  # years
    "chempot_constant": 16,  # adimensional
    "chi": 600.0,  # Liters / (gram * years)
    "A": 600.0,  # 1 / years
    "epsilon": 5.0E6,  # (um^2) / years
    "delta": 1003.75,  # grams / (Liters * years)
    "gamma": 1000.0,  # grams / (Liters * years)
    "s_average": 961.2,  # grams / (Liters * years)
    "s_max": 73.,
    "s_min": -73.
})

# %%
# Mesh definition and spatial discretization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The mesh definition is different from the 2D case, because this time we need to define a 3D domain.
# However, we can do that with ease using a FEniCS ``BoxMesh`` with a side of 2000 :math:`\mu m`:
nx = 130
nz = ny = nx
x_max = 1000  # um
x_min = -1000  # um
z_max = y_max = x_max
z_min = y_min = x_min

mesh = fenics.BoxMesh(fenics.Point(x_min, y_min, z_min),
                      fenics.Point(x_max, y_max, z_max),
                      nx,
                      ny,
                      nz)

# %%
# From the mesh, we can again define the function space in the same way we did in the 2D simulation. Indeed, the
# system of differential equations is the same and FEniCS will take care of defining the "3D-version" of the finite
# element:
function_space = get_mixed_function_space(mesh, 2, "CG", 1)

# %%
# Initial & boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Again, in this implementation we will consider natural Neumann boundary conditions for both :math:`\varphi` and
# :math`\sigma`.
#
# As initial condition for :math:`\varphi` and :math:`\sigma`, the most natural choice to resemble the results of
# Lorenzo and collaborators :cite:`Lorenzo2016` is to define an Ellipsoid, instead of an Ellipse. This can be done
# with ease using Mocafe:
semiax_x = 100  # um
semiax_y = 150  # um
semiax_z = 100  # um

phi0 = EllipsoidField(center=np.array([0., 0., 0.]),
                      semiax_x=semiax_x,
                      semiax_y=semiax_y,
                      semiax_z=semiax_z,
                      inside_value=parameters.get_value("phi0_in"),
                      outside_value=parameters.get_value("phi0_out"))
phi0 = fenics.interpolate(phi0, function_space.sub(0).collapse())
phi_xdmf.write(phi0, 0)

sigma0 = EllipsoidField(center=np.array([0., 0., 0.]),
                      semiax_x=semiax_x,
                      semiax_y=semiax_y,
                      semiax_z=semiax_z,
                      inside_value=parameters.get_value("sigma0_in"),
                      outside_value=parameters.get_value("sigma0_out"))
sigma0 = fenics.interpolate(sigma0, function_space.sub(0).collapse())
sigma_xdmf.write(sigma0, 0)

# %%
# PDE System definition
# ^^^^^^^^^^^^^^^^^^^^^
# Exactly how the differential equations don't change from 2D to 3D, the PDE definition remains the same. Indeed,
# you can notice that the code it's just identical to the 2D demo:
u = fenics.Function(function_space)

phi, sigma = fenics.split(u)

s_exp = fenics.Expression("(s_av + s_min) + ((s_max - s_min)*(random()/((double)RAND_MAX)))",
                          degree=2,
                          s_av=parameters.get_value("s_average"),
                          s_min=parameters.get_value("s_min"),
                          s_max=parameters.get_value("s_max"))
s = fenics.interpolate(s_exp, function_space.sub(0).collapse())

v1, v2 = fenics.TestFunctions(function_space)
weak_form = pc_model.prostate_cancer_form(phi, phi0, sigma, v1, parameters) + \
    pc_model.prostate_cancer_nutrient_form(sigma, sigma0, phi, v2, s, parameters)


# %%
# Simulation setup
# ^^^^^^^^^^^^^^^^
# And, again, the simulation setup is the same as the 2D case. We just choose a lower number of step in order to reduce
# the simulation time:
n_steps = 500

# %%
# Then, the code remains the same. However, remember what we remarked in the 2D demo: you might need to change the
# solver configuration in order to solve the system on your computer, and it's not guaranteed that the configuration
# you choose for the 2D system is the best for the 3D system as well.

# set up progress bar
if rank == 0:
    progress_bar = tqdm(total=n_steps, ncols=100)
else:
    progress_bar = None

# configure PETSc
petsc4py.init([__name__,
               "-snes_type", "newtonls",
               "-ksp_type", "gmres",
               "-pc_type", "gamg"])
from petsc4py import PETSc

# create snes solver
snes_solver = PETSc.SNES().create(comm)
snes_solver.setFromOptions()

# iterate in time
t = 0
for current_step in range(n_steps):
    # update time
    t += parameters.get_value("dt")

    # solve the problem with the solver defined by the given parameters
    problem = SNESProblem(weak_form, u, [])
    b = fenics.PETScVector()
    J_mat = fenics.PETScMatrix()
    snes_solver.setFunction(problem.F, b.vec())
    snes_solver.setJacobian(problem.J, J_mat.mat())
    snes_solver.solve(None, u.vector().vec())

    # save new values to phi0 and sigma0, in order for them to be the initial condition for the next step
    fenics.assign([phi0, sigma0], u)

    # save current solutions to file
    phi_xdmf.write(phi0, t)  # write the value of phi at time t
    sigma_xdmf.write(sigma0, t)  # write the value of sigma at time t

    # update progress bar
    if rank == 0:
        progress_bar.update(1)


# %%
# Result
# ------
# We uploaded on Youtube the result on this simulation. You can check it out below or at
# `this link <https://youtu.be/pcT0Vf-kHt0>`_
#
# ..  youtube:: pcT0Vf-kHt0
#
# Visualize the result with ParaView
# ----------------------------------
# The result of the simulation is stored in the ``.xdmf`` file generated, which are easy to load and visualize in
# expernal softwares as ParaView. If you don't now how to do it, you can check out the tutorial below or at
# `this Youtube link <https://youtu.be/ghx5MNZesvQ>`_.
#
# ..  youtube:: ghx5MNZesvQ
#

# %%
# Full code
# ----------
#
# .. code-block:: default
#
#   import numpy as np
#   import fenics
#   import petsc4py
#   from tqdm import tqdm
#   from pathlib import Path
#   from mocafe.fenut.solvers import SNESProblem
#   from mocafe.fenut.fenut import get_mixed_function_space, setup_xdmf_files
#   from mocafe.fenut.mansimdata import setup_data_folder
#   from mocafe.expressions import EllipsoidField
#   from mocafe.fenut.parameters import from_dict
#   import mocafe.litforms.prostate_cancer as pc_model
#
#   comm = fenics.MPI.comm_world
#   rank = comm.Get_rank()
#
#   file_folder = Path(__file__).parent.resolve()
#   data_folder = setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/prostate_cancer_3d",
#                                   auto_enumerate=False)
#
#   phi_xdmf, sigma_xdmf = setup_xdmf_files(["phi", "sigma"], data_folder)
#
#   parameters = from_dict({
#       "phi0_in": 1.,  # adimentional
#       "phi0_out": 0.,  # adimdimentional
#       "sigma0_in": 0.2,  # adimentional
#       "sigma0_out": 1.,  # adimentional
#       "dt": 0.001,  # years
#       "lambda": 1.6E5,  # (um^2) / years
#       "tau": 0.01,  # years
#       "chempot_constant": 16,  # adimensional
#       "chi": 600.0,  # Liters / (gram * years)
#       "A": 600.0,  # 1 / years
#       "epsilon": 5.0E6,  # (um^2) / years
#       "delta": 1003.75,  # grams / (Liters * years)
#       "gamma": 1000.0,  # grams / (Liters * years)
#       "s_average": 961.2,  # grams / (Liters * years)
#       "s_max": 73.,
#       "s_min": -73.
#   })
#
#   # Mesh definition
#   nx = 130
#   nz = ny = nx
#   x_max = 1000  # um
#   x_min = -1000  # um
#   z_max = y_max = x_max
#   z_min = y_min = x_min
#
#   mesh = fenics.BoxMesh(fenics.Point(x_min, y_min, z_min),
#                         fenics.Point(x_max, y_max, z_max),
#                         nx,
#                         ny,
#                         nz)
#
#   # Spatial discretization
#   function_space = get_mixed_function_space(mesh, 2, "CG", 1)
#
#   # Initial conditions
#   semiax_x = 100  # um
#   semiax_y = 150  # um
#   semiax_z = 100  # um
#
#   phi0 = EllipsoidField(center=np.array([0., 0., 0.]),
#                         semiax_x=semiax_x,
#                         semiax_y=semiax_y,
#                         semiax_z=semiax_z,
#                         inside_value=parameters.get_value("phi0_in"),
#                         outside_value=parameters.get_value("phi0_out"))
#   phi0 = fenics.interpolate(phi0, function_space.sub(0).collapse())
#   phi_xdmf.write(phi0, 0)
#
#   sigma0 = EllipsoidField(center=np.array([0., 0., 0.]),
#                         semiax_x=semiax_x,
#                         semiax_y=semiax_y,
#                         semiax_z=semiax_z,
#                         inside_value=parameters.get_value("sigma0_in"),
#                         outside_value=parameters.get_value("sigma0_out"))
#   sigma0 = fenics.interpolate(sigma0, function_space.sub(0).collapse())
#   sigma_xdmf.write(sigma0, 0)
#
#   # Weak form
#   u = fenics.Function(function_space)
#
#   phi, sigma = fenics.split(u)
#
#   s_exp = fenics.Expression("(s_av + s_min) + ((s_max - s_min)*(random()/((double)RAND_MAX)))",
#                             degree=2,
#                             s_av=parameters.get_value("s_average"),
#                             s_min=parameters.get_value("s_min"),
#                             s_max=parameters.get_value("s_max"))
#   s = fenics.interpolate(s_exp, function_space.sub(0).collapse())
#
#   v1, v2 = fenics.TestFunctions(function_space)
#   weak_form = pc_model.prostate_cancer_form(phi, phi0, sigma, v1, parameters) + \
#       pc_model.prostate_cancer_nutrient_form(sigma, sigma0, phi, v2, s, parameters)
#
#
#   # Simulation
#   n_steps = 500
#
#   # set up progress bar
#   if rank == 0:
#       progress_bar = tqdm(total=n_steps, ncols=100)
#   else:
#       progress_bar = None
#
#   # configure PETSc
#   petsc4py.init([__name__,
#                  "-snes_type", "newtonls",
#                  "-ksp_type", "gmres",
#                  "-pc_type", "gamg"])
#   from petsc4py import PETSc
#
#   # create snes solver
#   snes_solver = PETSc.SNES().create(comm)
#   snes_solver.setFromOptions()
#
#   # iterate in time
#   t = 0
#   for current_step in range(n_steps):
#       # update time
#       t += parameters.get_value("dt")
#
#       # solve the problem with the solver defined by the given parameters
#       problem = SNESProblem(weak_form, u, [])
#       b = fenics.PETScVector()
#       J_mat = fenics.PETScMatrix()
#       snes_solver.setFunction(problem.F, b.vec())
#       snes_solver.setJacobian(problem.J, J_mat.mat())
#       snes_solver.solve(None, u.vector().vec())
#
#       # save new values to phi0 and sigma0, in order for them to be the initial condition for the next step
#       fenics.assign([phi0, sigma0], u)
#
#       # save current solutions to file
#       phi_xdmf.write(phi0, t)  # write the value of phi at time t
#       sigma_xdmf.write(sigma0, t)  # write the value of sigma at time t
#
#       # update progress bar
#       if rank == 0:
#           progress_bar.update(1)