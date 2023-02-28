r"""
.. _Prostate Cancer 2D Demo:

Prostate cancer
===============

In this short demo we will show you how to simulate a phase field model described by G. Lorenzo and collaborators
in 2016 :cite:`Lorenzo2016` using FEniCS and Mocafe. The model was published on PNAS in 2016 and presents a
continuous mathematical model able to reproduce the growth pattern of prostate cancer at tissue scale.

.. contents:: Table of Contents
   :local:

How to run this example on Mocafe
---------------------------------
Make sure you have FEniCS and Mocafe installed and download the source script of this page (see above for the link).
Then, simply run it using python:

.. code-block:: console

    python3 prostate_cancer2d.py

However, it is recommended to exploit parallelization to save simulation time:

.. code-block:: console

    mpirun -n 4 python3 prostate_cancer2d.py

Notice that the number following the ``-n`` option is the number of MPI processes you using for parallelizing the
simulation. You can change it accordingly with your CPU.

Visualize the results of this simulation
----------------------------------------
You need to have `Paraview <https://www.paraview.org/>`_ to visualize the results. Once you have installed it,
you can easly import the ``.xdmf`` files generated during the simulation and visualize the result.

Brief introduction to the mathematical model
--------------------------------------------
The model is composed of just two partial differential equations (PDEs). The first describes the evolution of the
cancer phase field  :math:`\varphi`, and reads:

.. math::
    \frac{\partial \varphi}{\partial t} = \lambda \nabla^2 \varphi - \frac{1}{\tau}\frac{dF(\varphi)}{d\varphi}
    + \chi \sigma - A \varphi

The second describes the variation of nutrients concentration :math:`\sigma` in time:

.. math::
    \frac{\partial \sigma}{\partial t} = \epsilon \nabla^2\sigma + s - \delta\cdot\varphi - \gamma\cdot\sigma

A complete discussion of these two equations is above the purpose of this short demo so, if you're interested, we
suggest you to refer to the original paper :cite:`Lorenzo2016`. However, we just mention some of their main features.

The first equation describes a cancer development driven by both proliferation, and apoptosis. Cancer cells are
assumed to duplicate in presence of nutrient and their proliferation is, indeed, described by the term
:math:`\chi \sigma`, which contains the nutrients concentration. Apoptosis is, instead assumed to occur at a constant
rate and is represented in the equation by the term :math:`-A \varphi`.

The second equation describes the diffusion of the nutrients with the Fick's low of diffusion
:cite:`enwiki:1058693490`. The equation assumes that the nutrient is supplied constantly in the domain with
a distribution :math:`s`, and that the nutrient is consumed at a constant rate by the cancer cells (term
:math:`-\delta\varphi`). Additionaly, the nutrient is supposed to decay at a constant rate, described by the term
:math:`\gamma \sigma`.

"""

# %%
# Implementation
# ------------------------------------------
# After the math, let's see the code. The implementation of this model in scientific code just follow the general
# FEniCS workflow we outlined in the demo :ref:`FENICS_INTRO`.
#
# Initial setup
# ^^^^^^^^^^^^^
# To reproduce this model we need first to import everything we need throughout
# the simulation. Notice that while most of the packages are provided by Mocafe, we also use some other stuff.
import numpy as np
import fenics
from tqdm import tqdm
from pathlib import Path
import petsc4py
from mocafe.fenut.solvers import SNESProblem
from mocafe.fenut.fenut import get_mixed_function_space, setup_xdmf_files
from mocafe.fenut.mansimdata import setup_data_folder
from mocafe.expressions import EllipseField
from mocafe.fenut.parameters import from_dict
import mocafe.litforms.prostate_cancer as pc_model

# %%
# Then, it is useful (even though not necessary) to do a number of operations before running our simulation.
#
# First of all, we shut down the logging messages from FEniCS, leaving only the error messages in case something goes
# *really* wrong. If you want to see the FEniCS logging messages, you can comment this line.
fenics.set_log_level(fenics.LogLevel.ERROR)

# %%
# Then, we define the MPI _rank for each process. Generally speaking, this is necessary for running the simulation in
# parallel using ``mpirun``, even though in this simulation is not largely used, as we are going to see.
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# %%
# Then, we can define the files where to save our result for visualization and post-processing. The recommended format
# for saving FEniCS simulations is using ``.xdmf`` files, which can easily be visualized in
# `Paraview <https://www.paraview.org/>`_.
#
# Even though FEniCS provides its own classes and method to define these files, in the following we use two Mocafe
# methods for defining:
#
# - first, the folder where to save the result of the simulation. In this case, the folder will be based inside
#   the current folder (``file_folder``) and it's called demo_out/prostate_cancer2d;
#
file_folder = Path(__file__).parent.resolve()
data_folder = setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/prostate_cancer_2d",
                                auto_enumerate=None)

# %%
# - then, the two files for the cancer :math:`\varphi` and for the nutrients :math:`\sigma`, which will be called
#   ``phi.xdmf`` and ``sigma.xdmf``.
#
phi_xdmf, sigma_xdmf = setup_xdmf_files(["phi", "sigma"], data_folder)

# %%
# Finally, we define the parameters of the differential equation using a Mocafe ``Parameter`` object, which is created
# for this purpose.
#
# A Parameters object can be initialized in several ways. In the following, we define it from a
# dictionary where each key is the parameter name and the value is the actual value of the parameter. The values
# chosen for this simulation are in agreement with those reported by Lorenzo et al. by two papers regarding this
# model :cite:`Lorenzo2016` :cite:`Lorenzo2017`.
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
    "s_average": 2.75 * 365,  # 961.2,  # grams / (Liters * years)
    "s_max": 73.,
    "s_min": -73.
})

# %%
# Mesh definition and spatial discretization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The first step toward the simulation of our system is the definition of the space where the simulation takes
# place. Similarly to the original paper, we are going to simulate the model on a 2D square mesh of dimension
# 2000 x 2000 :math:`\mu m`. This is pretty simple to do using FEniCs, which provides the class ``RectangleMesh``
# to do this job.
#
# More precisely, in the following we are going to define a mesh of the dimension described above, with ``nx``
# points for each side.
#
nx = 130
ny = nx
x_max = 1000  # um
x_min = -1000  # um
y_max = x_max
y_min = x_min

mesh = fenics.RectangleMesh(fenics.Point(x_min, y_min),
                            fenics.Point(x_max, y_max),
                            nx,
                            ny)

# %%
# From the mesh defined above, we can then define the ``FunctionSpace``.
# Since the model we wish to simulate is composed of two coupled equations, we need to define a MixedElement function
# space with two different elements. In this implementation, we will use for both equations the same element
# type, "CG" (Continuous Galerking), of the first order, which can be created in FEniCS simply using::
#
#     cg1_element = fenics.FiniteElement("CG", fenics.triangle, 1)  # define element
#     mixed_element = fenics.MixedElement([cg1_element] * 2)  # define mixed element
#     function_space = fenics.FunctionSpace(mesh, mixed_element)  # define function space for the given mesh
#
# However, the very same operation can be performed in just one line using the following method provided by
# Mocafe:
#
function_space = get_mixed_function_space(mesh, 2, "CG", 1)

# %%
# Initial & boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this implementation we will consider natural Neumann boundary conditions for both :math:`\varphi` and
# :math`\sigma`, which means that the derivative in space of the two fields is zero along the entire boundary.
# This is an easy pick for FEniCS, since it will automatically apply this condition for us without requiring any
# command from the user.
#
# As initial condition for :math:`\varphi`, according to the author :cite:`Lorenzo2017`, we will define an elliptical
# tumor with the given semiaxes:
semiax_x = 100  # um
semiax_y = 150  # um

# %%
# With FEniCS is not hard to define such a function leveraging the ``Expression`` class. However, given how common
# this initial condition is in cancer mathematical modeling, we provided our own built-in expression for defining a
# general elliptic field expression:
phi0 = EllipseField(center=np.array([0., 0.]),
                    semiax_x=semiax_x,
                    semiax_y=semiax_y,
                    inside_value=parameters.get_value("phi0_in"),
                    outside_value=parameters.get_value("phi0_out"))

# %%
# Which can be then interpolated in our function space.
#
# The interpolation can be done simply calling the FEniCS method ``interpolate``, which takes as arguments the
# expression to be interpolated and the function space where to do the interpolation. Notice that, since the function
# space we defined is mixed, we must choose one of the sub-field to define the function.
phi0 = fenics.interpolate(phi0, function_space.sub(0).collapse())

# %%
# Notice also that since the mixed function space is defined by two identical function spaces, it makes no
# difference to pick sub(0) or sub(1).
#
# Then, we can save the initial condition of the :math:`\varphi` field in the `.xdmf` file we defined at the
# beginning:
phi_xdmf.write(phi0, 0)

# %%
# Finally, after having defined the initial condition for :math:`\varphi`, let's define the initial for
# :math:`\sigma`. Following the hypothesis of original author :cite:`Lorenzo2017`, we will assume a nutrient
# distribution that is 0.2 inside the cancer and 1. outside. So, we can define this distribution similarly to
# what we just did for ``phi0``:
sigma0 = EllipseField(center=np.array([0., 0.]),
                      semiax_x=semiax_x,
                      semiax_y=semiax_y,
                      inside_value=parameters.get_value("sigma0_in"),
                      outside_value=parameters.get_value("sigma0_out"))
sigma0 = fenics.interpolate(sigma0, function_space.sub(0).collapse())
sigma_xdmf.write(sigma0, 0)

# %%
# Weak form definition
# ^^^^^^^^^^^^^^^^^^^^^
# After having defined the initial conditions for the system, we continue with the definition of the weak form of
# the system itself.
#
# First of all, we define the two variables, ``phi`` and ``sigma``, for which the system will be solved. Since the
# two equations are coupled (i.e. they depend on each other) the easiest way to do so is to define a 'vector'
# function ``u`` on the mixed function space:
u = fenics.Function(function_space)

# %%
# And then to split the vector in its two components, which represent :math:`\varphi` and :math:`\sigma`:
phi, sigma = fenics.split(u)

# %%
# After having defined phi and sigma, we defined the :math:`s` function, which represent the distribution of
# nutrient that is supplied to the system.
#
# In the original paper they simulated the model for both a constant distribution and for a 'capillary-like'
# distribution based on an picture :cite:`Lorenzo2016`.
#
# In this implementation we just chose to simulate the model with a random distribution of the nutrient, with
# values included in the range :math:`[s_{average} + s_{min}, s_{average} + s_{max}]`, where
# :math:`s_{max} = - s_{min}`.
# The specific values we need are specified in the parameters object we created above, so we use that to retrieve the
# values.
#
# The most efficient way to do so in FEniCS is to use the ``Expression`` class and a C++ code with the
# function ``(random()/((double)RAND_MAX))`` to generate uniform random numbers between 0 and 1. Of course, there are
# ways to do the same thing in Python using the ``random`` module, but in our experience the use of C++ code with the
# FEniCS interface reduces significantly the time required for the interpolation.
s_exp = fenics.Expression("(s_av + s_min) + ((s_max - s_min)*(random()/((double)RAND_MAX)))",
                          degree=2,
                          s_av=parameters.get_value("s_average"),
                          s_min=parameters.get_value("s_min"),
                          s_max=parameters.get_value("s_max"))
s = fenics.interpolate(s_exp, function_space.sub(0).collapse())

# %%
# Now, we have everything in place to define our PDE system exploiting the related Mocafe functions contained in the
# module ``pc_model``:

v1, v2 = fenics.TestFunctions(function_space)
weak_form = pc_model.prostate_cancer_form(phi, phi0, sigma, v1, parameters) + \
    pc_model.prostate_cancer_nutrient_form(sigma, sigma0, phi, v2, s, parameters)

# %%
# These functions are nothing more than a self-contained definition of the UFL form of the model's equations, which you
# can inspect yourself if you like.

# %%
# Simulation: setup
# ^^^^^^^^^^^^^^^^^
# Now that everything is ready, simulating this mathematical model is just a matter of solving the PDE system defined
# above for each time step.
#
# To do so, we start defining the total number of steps to simulate. We choose that in order to have a total
# simulated time of one year, given the dt of the system (see its value in the ``Parameters`` object)
n_steps = 1000

# %%
# Then, we define a progress bar with ``tqdm`` in order to monitor the iteration progress. Notice that the progress
# bar is defined only if the _rank of the process is 0. This is necessary to avoid every process to print out a
# different progress bar.
if rank == 0:
    progress_bar = tqdm(total=n_steps, ncols=100)
else:
    progress_bar = None

# %%
# Then, we need to define how we want FEniCS to solve or PDE system. To do so, we first need to define the solver we
# want to use.
# For that, we can take advantage of the `PETSc <https://petsc.org/release/>`_
# (Portable, Extensible Toolkit for Scientific Computation) library, implemented in Python as ``petsc4py``, which
# is one of the most used suites of routines for solving partial differential equations.
# More precisely, since our model is non-linear, we will take advantage of the PETSc SNES solver
# (which is optimized for nonlinear systems).
#
# The standard way to create a SNES solver is to set it up from the command line, using:
#
# .. code-block:: default
#
#   petsc4py.init(sys.argv)
#
# However, for your convenience, we just hard coded the SNES configuration that worked better for us.
petsc4py.init([__name__,
               "-snes_type", "newtonls",
               "-ksp_type", "gmres",
               "-pc_type", "gamg"])
from petsc4py import PETSc

# define solver
snes_solver = PETSc.SNES().create(comm)
snes_solver.setFromOptions()


# %%
# Still, notice that the best configuration for your system might change, since it is well known that it is very hard
# to tell which solver will perform the best given the PDEs, the mesh, the CPU, the cores number and so on (see
# `this post
# <https://fenicsproject.discourse.group/t/how-to-choose-the-optimal-solver-for-a-pde-problem/7477>`_).
#
# If errors occur, please consider using a different configuration for SNES. For a complete list, you can refer to
# the documentation of `petsc4py <https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/index.html>`_. If you
# need more information on the use of SNES in FEniCS, you can also refer to this
# `excellent discussion <https://fenicsproject.discourse.group/t/using-petsc4py-petsc-snes-directly/2368>`_ in the
# FEniCS forum.
#

# %%
# Simulation
# ^^^^^^^^^^
# Finally, we can iterate in time to solve the system with the given solver at each time step.
t = 0
for current_step in range(n_steps):
    # update time
    t += parameters.get_value("dt")

    # define problem
    problem = SNESProblem(weak_form, u, [])

    # set up algebraic system for SNES
    b = fenics.PETScVector()
    J_mat = fenics.PETScMatrix()
    snes_solver.setFunction(problem.F, b.vec())
    snes_solver.setJacobian(problem.J, J_mat.mat())

    # solve system
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
# Let's analyze everything step-by-step. First, we update the simulation time:
#
# .. code-block:: default
#
#   # update time
#   t += parameters.get_value("dt")
#
# Then, we define the "problem" we want to be solved by the SNES solver:
#
# .. code-block:: default
#
#   # define problem
#   problem = SNESProblem(weak_form, u, [])
#
#   # set up algebraic system for SNES
#   b = fenics.PETScVector()
#   J_mat = fenics.PETScMatrix()
#   snes_solver.setFunction(problem.F, b.vec())
#   snes_solver.setJacobian(problem.J, J_mat.mat())
#
# The few lines above might look a bit obscure if you're not experienced with FEM and numerical methods in general,
# but we will do our best to clarify a bit.
#
# Like every numerical method, FEM translates a system of PDEs in an algebraic system of linear equations of which
# the solution is an estimate of the real PDE system solution. The job of the class ``PETScProblem`` is exactly to
# construct the algebraic system of equations from the weak form, the function we want to find, and the boundary
# conditions. For our example:
#
# - we already defined the weak form above, so we can use it as it is;
# - the function we want to find is ``u``, which contains both ``phi`` and ``sigma``;
# - we left the list of boundary conditions empty (``[]``) because we are considering natural Neumann boundary
#   conditions, which are applied by default by the FEM method.
#
# Once we did that, we simply need to tell SNES to solve our system, specifying the weak form (``problem.F``) and
# its jacobian matrix (``problem.J``) as a ``PETScVector`` and a ``PETScMatrix``, respectively. This is indeed what
# we're doing with the methods ``setFunction`` and ``setJacobian``.
#
# Then, we can solve our system placing the result in the ``u`` function:
#
# .. code-block:: default
#
#   # solve system
#   snes_solver.solve(None, u.vector().vec())
#
# Assign the result at the current step as the new values of ``phi0`` and ``sigma0``, in order to be the initial
# condition for the next iteration:
#
# .. code-block:: default
#
#   fenics.assign([phi0, sigma0], u)
#
# And finally, we write the result on the ``.xdmf`` files and update the progress bar:
#
# .. code-block:: default
#
#   # save current solutions to file
#   phi_xdmf.write(phi0, t)  # write the value of phi at time t
#   sigma_xdmf.write(sigma0, t)  # write the value of sigma at time t
#
#   # update progress bar
#   if _rank == 0:
#       progress_bar.update(1)
#
#
# Result
# ------
# We uploaded on Youtube the result on this simulation. You can check it out below or at `this link <https://youtu.be/2isujSzWH1A>`_
#
# ..  youtube:: 2isujSzWH1A
#
# Visualize the result with ParaView
# ----------------------------------
# The result of the simulation is stored in the ``.xdmf`` file generated, which are easy to load and visualize in
# expernal softwares as ParaView. If you don't now how to do it, you can check out the tutorial below or at
# `this Youtube link <https://youtu.be/d-BwfqNltN4>`_.
#
# ..  youtube:: d-BwfqNltN4
#
# Full code
# ---------
#
# .. code-block:: default
#
#   import numpy as np
#   import fenics
#   from tqdm import tqdm
#   from pathlib import Path
#   import petsc4py
#   from mocafe.fenut.solvers import SNESProblem
#   from mocafe.fenut.fenut import get_mixed_function_space, setup_xdmf_files
#   from mocafe.fenut.mansimdata import setup_data_folder
#   from mocafe.expressions import EllipseField
#   from mocafe.fenut.parameters import from_dict
#   import mocafe.litforms.prostate_cancer as pc_model
#
#   # initial setup
#   fenics.set_log_level(fenics.LogLevel.ERROR)
#   _comm = fenics.MPI.comm_world
#   _rank = _comm.Get_rank()
#
#   file_folder = Path(__file__).parent.resolve()
#   data_folder = setup_data_folder(folder_path=f"{file_folder / Path('demo_out')}/prostate_cancer_2d",
#                                     auto_enumerate=None)
#   phi_xdmf, sigma_xdmf = setup_xdmf_files(["phi", "sigma"], data_folder)
#
#   parameters = from_dict({
#         "phi0_in": 1.,  # adimentional
#         "phi0_out": 0.,  # adimdimentional
#         "sigma0_in": 0.2,  # adimentional
#         "sigma0_out": 1.,  # adimentional
#         "dt": 0.001,  # years
#         "lambda": 1.6E5,  # (um^2) / years
#         "tau": 0.01,  # years
#         "chempot_constant": 16,  # adimensional
#         "chi": 600.0,  # Liters / (gram * years)
#         "A": 600.0,  # 1 / years
#         "epsilon": 5.0E6,  # (um^2) / years
#         "delta": 1003.75,  # grams / (Liters * years)
#         "gamma": 1000.0,  # grams / (Liters * years)
#         "s_average": 2.75 * 365,  # 961.2,  # grams / (Liters * years)
#         "s_max": 73.,
#         "s_min": -73.
#   })
#
#   # Mesh definition
#   nx = 130
#   ny = nx
#   x_max = 1000  # um
#   x_min = -1000  # um
#   y_max = x_max
#   y_min = x_min
#
#   mesh = fenics.RectangleMesh(fenics.Point(x_min, y_min),
#                               fenics.Point(x_max, y_max),
#                               nx,
#                               ny)
#
#   # Spatial discretization
#   function_space = get_mixed_function_space(mesh, 2, "CG", 1)
#
#   # Initial conditions
#   semiax_x = 100  # um
#   semiax_y = 150  # um
#   phi0 = EllipseField(center=np.array([0., 0.]),
#                       semiax_x=semiax_x,
#                       semiax_y=semiax_y,
#                       inside_value=parameters.get_value("phi0_in"),
#                       outside_value=parameters.get_value("phi0_out"))
#   phi0 = fenics.interpolate(phi0, function_space.sub(0).collapse())
#   phi_xdmf.write(phi0, 0)
#
#   sigma0 = EllipseField(center=np.array([0., 0.]),
#                         semiax_x=semiax_x,
#                         semiax_y=semiax_y,
#                         inside_value=parameters.get_value("sigma0_in"),
#                         outside_value=parameters.get_value("sigma0_out"))
#   sigma0 = fenics.interpolate(sigma0, function_space.sub(0).collapse())
#   sigma_xdmf.write(sigma0, 0)
#
#   # Weak form definition
#   u = fenics.Function(function_space)
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
#                 pc_model.prostate_cancer_nutrient_form(sigma, sigma0, phi, v2, s, parameters)
#
#   # Simulation: setup
#   n_steps = 1000
#
#   if _rank == 0:
#         progress_bar = tqdm(total=n_steps, ncols=100)
#   else:
#         progress_bar = None
#
#   petsc4py.init([__name__,
#                  "-snes_type", "newtonls",
#                  "-ksp_type", "gmres",
#                  "-pc_type", "gamg"])
#   from petsc4py import PETSc
#
#   # define solver
#   snes_solver = PETSc.SNES().create(_comm)
#   snes_solver.setFromOptions()
#
#   t = 0
#   for current_step in range(n_steps):
#         # update time
#         t += parameters.get_value("dt")
#
#         # define problem
#         problem = SNESProblem(weak_form, u, [])
#
#         # set up algebraic system for SNES
#         b = fenics.PETScVector()
#         J_mat = fenics.PETScMatrix()
#         snes_solver.setFunction(problem.F, b.vec())
#         snes_solver.setJacobian(problem.J, J_mat.mat())
#
#         # solve system
#         snes_solver.solve(None, u.vector().vec())
#
#         # save new values to phi0 and sigma0, in order for them to be the initial condition for the next step
#         fenics.assign([phi0, sigma0], u)
#
#         # save current solutions to file
#         phi_xdmf.write(phi0, t)  # write the value of phi at time t
#         sigma_xdmf.write(sigma0, t)  # write the value of sigma at time t
#
#         # update progress bar
#         if _rank == 0:
#             progress_bar.update(1)
