r"""
Prostate cancer phase field model
==================================

In this short demo we will show you how to simulate a phase field model described by G. Lorenzo and collaborators
in 2016 :cite:`Lorenzo2016` using FEniCS and mocafe. The model was published on PNAS in 2016 and presents a
continuous mathematical model able to reproduce the growth pattern of prostate cancer at tissue scale.

How to run this example on mocafe
---------------------------------
Make sure you have FEniCS and mocafe installed and download the source script of this page (see above for the link).
Then, simply run it using python:

.. code-block:: console

    python3 prostate_cancer2d.py

If you are in a hurry, you can exploit parallelization to run the simulation faster:

.. code-block:: console

    mpirun -n 4 python3 prostate_cancer2d.py

Notice that the number following the ``-n`` option is the number of MPI processes you using for parallelizing the
simulation. You can change it accordingly with your CPU.

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
:math:`\chi \sigma`, which contains the nutrients concentration. Apoptosis is, instead assumed to occurr at a constant
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
#
# Setup
# ^^^^^
# After the math, let's see the code. To reproduce this model we need first to import everything we need throughout
# the simulation. Notice that while most of the packages are provided by mocafe, we also use some other stuff.
import sys
import numpy as np
import fenics
import random
from tqdm import tqdm
from pathlib import Path
file_folder = Path(__file__).parent.resolve()
mocafe_folder = file_folder.parent
sys.path.append(str(mocafe_folder))  # appending mocafe path. Must be removed
from mocafe.fenut.solvers import PETScProblem, PETScNewtonSolver
from mocafe.fenut.fenut import get_mixed_function_space, setup_xdmf_files
from mocafe.fenut.mansimdata import setup_data_folder
from mocafe.expressions import EllipseField, PythonFunctionField
from mocafe.fenut.parameters import from_dict
import mocafe.litforms.prostate_cancer as pc_model

# %%
# Then, it is useful (even though not necessary) to do a number of operations before running our simulation.
#
# First of all, we shut down the logging messages from FEniCS, leaving only the error messages in case something goes
# *really* wrong. If you want to check out the FEniCS messages, you can comment this line.
fenics.set_log_level(fenics.LogLevel.ERROR)

# %%
# Then, we define the MPI rank for each process. Generally speaking, this is necessary for running the simulation in
# parallel using ``mpirun``, even though in this simulation is not largely used, as we are going to see.
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# %%
# Then, we can define the files where to save our result for visualization and post-processing. The suggested format
# for saving FEniCS simulations is using ``.xdmf`` files, which can easily be visualized in
# `Paraview <https://www.paraview.org/>`_.
#
# Even though FEniCS provides its own classes and method to define these files, in the following we use two mocafe
# methods for defining:
#
# - first, the folder where to save the result of the simulation. In this case, the folder will be based inside
#   the current folder (``base_location``) and it's called demo_out/prostate_cancer2d;
#
data_folder = setup_data_folder(folder_path=f"{file_folder/Path('demo_out')}/prostate_cancer_2d",
                                auto_enumerate=False)

# %%
# - then, the two files for the cancer :math:`\varphi` and for the nutrients :math:`\sigma`, which will be called
#   ``phi.xdmf`` and ``sigma.xdmf``.
#
phi_xdmf, sigma_xdmf = setup_xdmf_files(["phi", "sigma"], data_folder)

# %%
# Finally, we define the parameters of the differential equation using a mocafe ``Parameter`` object, which is created
# for this purpose.
#
# A Parameters object can be initialized in several ways. In the following, we define it from a
# dictionary where each key is the parameter name and the value is the actual value of the parameter.
parameters = from_dict({
    "phi0_in": 1.,  # adimentional
    "phi0_out": 0.,  # adimdimentional
    "sigma0_in": 0.2,  # adimentional
    "sigma0_out": 1.,  # adimentional
    "dt": 0.01,  # years
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
# Definition of the spatial domain and the function space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The first step toward the simulation of our system is the definition of the space where the simulation takes
# place. Similarly to the original paper, we are going to simulate the model on a 2D square mesh of dimension
# 2000 x 2000 :math:`\mu m`. This is pretty simple to do using FEniCs, which provides the class ``RectangleMesh``
# to do this job.
#
# More precisely, in the following we are going to define a mesh of the dimension described above, with 512
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
# From the mesh defined above, we can then define the ``FunctionSpace``. If your not familiar enough with FEniCS
# to know what a function space is, we suggest you to have a look to the first pages of The Fenics Tutorial
# :cite:`LangtangenLogg2017`, but basically the function space defines the set of the piece-wise
# polynomial function that will be used to approximate the solutions of our PDEs.
#
# Since the model we wish to simulate is composed of two coupled equations, we need to define a MixedElement function
# space with two different elements. In this implementation, we will use for both equations the same element
# type, "CG" (Continuous Galerking), of the first order, which can be created in FEniCS simply using::
#
#     cg1_element = fenics.FiniteElement("CG", fenics.triangle, 1)  # define element
#     mixed_element = fenics.MixedElement([cg1_element] * 2)  # define mixed element
#     function_space = fenics.FunctionSpace(mesh, mixed_element)  # define function space for the given mesh
#
# However, the very same operation can be performed in just one line using the following method provided by
# mocafe:
#
function_space = get_mixed_function_space(mesh, 2, "CG", 1)

# %%
# Initial & boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Since the model is a system of PDEs, we need both initial and boundary conditions to find a unique solution.
#
# In this implementation we will consider natural Neumann boundary conditions for both :math:`\varphi` and
# :math`\sigma`, which means that the derivative in space of the two fields is zero along the entire boundary.
# This is an easy pick for FEniCS, since it will automatically apply this condition for us without requiring any
# command from the user.
#
# As initial condition for :math:`\varphi`, according to another paper of the same author :cite:`Lorenzo2017`, we
# will define an elliptical tumor with the given semiaxes:
semiax_x = 100  # um
semiax_y = 150  # um

# %%
# With FEniCS we can do so by defining an expression which 'mathematically' represent our initial condition.
# Indeed, ``Espression``s are the FEniCS way to define symbolic mathematical functions and they can be defined
# using simple C++ code as follows:
#
# .. code-block:: default
#
#    phi0_max = 1
#    phi0_min = 0
#    # cpp code that returns True if the point x is inside the ellipse, and False otherwise
#    is_in_ellipse_cpp_code = "((pow(x[0] / semiax_x, 2)) + (pow(x[1] / semiax_y, 2)) <= 1)"
#    # cpp code that returns 1 if the above statement is True, and 0 otherwise
#    phi0_cpp_code = is_in_ellipse_cpp_code + " ? phi0_max : phi0_min"
#    # FEniCS expression, built from cpp code defined above
#    phi0 = fenics.Expression(phi0_cpp_code,
#                             degree=2,
#                             semiax_x=semiax_x,
#                             semiax_y=semiax_y,
#                             phi0_max=phi0_max,
#                             phi0_min=phi0_min)
#
# However, if you don't feel confident in defining your own expression with the FEniCS interface, you can use
# the one provided by mocafe:
phi0 = EllipseField(center=np.array([0., 0.]),
                    semiax_x=semiax_x,
                    semiax_y=semiax_y,
                    inside_value=parameters.get_value("phi0_in"),
                    outside_value=parameters.get_value("phi0_out"))

# %%
# The FEniCS expression must then be interpolated in the function space in order to obtain a
# FEniCS Function. Again, explaining why we need to do so is something that goes beyond the purpose of this small
# demo, but think about it as a necessary operation required to transform the 'symbolic' function provided by the
# ``Expression`` into the actual set of values of our expression in our spatial domain, so we can use them to
# calculate our solution.
#
# The interpolation can be done simply calling the FEniCS method ``interpolate``, which takes as arguments the
# expression to be projected and the function space where to do the projection. Notice that, since the function space
# we defined is mixed, we must choose one of the sub-field to define the function.
phi0 = fenics.interpolate(phi0, function_space.sub(0).collapse())

# %%
# Notice also that since the mixed function space is defined by two identical function spaces, it makes no
# difference to pick sub(0) or sub(1).
#
# Then, we can save the initial condition of the :math:`\varphi` field in the `.xdmf` file we defined at the
# beginning, simply calling the method ``write(phi0, 0)``. The second argument, 0, just represent the fact that
# this is the value of the field for the time 0. As we're going to see in the simulation, the file ``phi_xdmf`` can
# collect the values of phi for each time.
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
# PDE System definition
# ^^^^^^^^^^^^^^^^^^^^^
# After having defined the initial conditions for the system, we continue with the definition of the system
# itself.
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
# In the original paper they simulated the model for both a constant distibution and for a randomic one. In
# this implementation we chose to do the the latter, which is slightly more complex, even though made
# simplier by the mocafe ``Expression`` ``PythonFunctionField``.
#
# This class allows us to use a python function, such as a lambda function, to define the values of a FEniCS function.
# In the following, indeed, we make use of a lambda function and of the methods provided by the module ``random``
# to define the random distribution mentioned above. Indeed, The pyhton function it is used by this class to evaluate
# the value of the FEniCS function at each point of the mesh. Notice that the function given as imput must always have
# at least on input (x in this case), representing the spatial point.
s_expression = PythonFunctionField(
    python_fun=lambda x: parameters.get_value("s_average") + random.uniform(parameters.get_value("s_min"),
                                                                          parameters.get_value("s_max")),
)

# %%
# Now, we have everything in place to define our PDE system. Since FEniCS uses the Finite Element Method (FEM) to
# approximate the solution we need to define the so called 'weak form' of our system. This operation is not difficult
# to do with the Unified Form Language (UFL) of FEniCS and, if you're not experienced with that, you are encouraged to
# have a look to The Fenics Tutorial to start :cite:`LangtangenLogg2017`. However, the weak form of this system
# is already defined in mocefe, so we can exploit that without wondering too much about weak form construction:
#
v1, v2 = fenics.TestFunctions(function_space)
weak_form = pc_model.prostate_cancer_form(phi, phi0, sigma, v1, parameters) + \
    pc_model.prostate_cancer_nutrient_form(sigma, sigma0, phi, v2, s_expression, parameters)

# %%
# Still, you are invited to notice a couple of interesting things:
#
# - the trial function necessary to define every weak form are simply variables in FEniCS;
# - the variable ``weak_form`` is defined as the sum of two elements ``prostate_cancer_form`` and
#   ``prostate_cancer_nutrient_form``, which represent, of course, the two differential equations of the system
# - the variable ``weak_form`` depends on ``phi``, ``sigma``, their initial values, ``s``, and the model parameters,
#   exactly like the equations defined above
#
# This was just to give you a taste of how simple it is to use UFL do define systems of differential equation, and how
# well is integrated in Python. If you want to know more about it, you're again invited to have a look to The FEniCS
# Tutorial :cite:`LangtangenLogg2017`.

# %%
# Simulation setup
# ^^^^^^^^^^^^^^^^
# Now that everything is set up, simulating this mathematical model is just a matter of solving the PDE system defined
# above for each time step.
#
# To do so, we start defining the total number of steps to simulate:
n_steps = 100
# %%
# Then, we define a progress bar with ``tqdm`` in order to monitor the iteration progress. Notice that the progress
# bar is defined only if the rank of the process is 0. This is necessary to avoid every process to print out a
# different progress bar.
if rank == 0:
    progress_bar = tqdm(total=n_steps, ncols=100)
else:
    progress_bar = None

# %%
# Then, we need to define how we want FEniCS to solve or PDE system. This can be done with just a few lines of code in
# mocafe, which are necessary to set up the right solver for our problem:
jacobian = fenics.derivative(weak_form, u)
problem = PETScProblem(jacobian, weak_form, [])
solver = PETScNewtonSolver({"ksp_type": "gmres", "pc_type": "asm"},
                           mesh.mpi_comm())

# %%
# The few lines above might look a bit obscure if you're not experienced with FEM and numerical methods in general,
# but we will do our best to clarify a bit.
#
# Like every numerical method, FEM translates a system of PDEs in an algebraic system of linear equations of which
# the solution is an estimate of the real PDE system solution. FEniCS delegates the construction and the solution of
# this system to `PETSc <https://petsc.org/release/>`_ (Portable, Extensible Toolkit for Scientific Computation),
# its default algebraic backend.
#
# The job of the class ``PETScProblem`` is to construct the algebraic system of equations from the weak form,
# its jacobian matrix, and the boundary conditions. For our example:
#
# - we already defined the weak form above, so we can use it as it is;
# - we can retrieve the Jacobian matrix, which is a multidimensional version of the traditional matematical derivative,
#   symply calling the FEniCS command ``derivative``;
# - we left the list of boundary conditions empty (``[]``) because we are considering natural Neumann boundary
#   conditions, which are applied by default by the FEM method.
#
# The job of the class ``PETScNewtonSolver``, instead, is to define the algorithm to be used to solve the 'problem'
# defined above, and to apply it for the computation of the actual solution. The algorithm may be one of the many
# available for solving algebraic systems of equations. The reason of the name 'Newton Solver` is just because
# the system of PDEs we are solving is non-linear and thus it requires this class of solvers. More precisely, in this
# implementation we are asking to PETSc to solve our system with a Krylov solver of type 'gmres'
# (``"ksp_type": "gmres"``) using a preconditioner called "asm" (``"pc_type": "asm"``). For further details, you
# are suggested to have a look to chapter 9 of the book "The Finite Element Method: Theory, Implementation,
# and Applications", by Larson and Bengzon :cite:`Larson2013`

# %%
# Simulation
# ^^^^^^^^^^
# Finally, we can iterate in time to solve the system with the given solver at each time step.
t = 0
for current_step in range(n_steps):
    # update time
    t += parameters.get_value("dt")

    # solve the problem with the solver defined by the given parameters
    solver.solve(problem, u.vector())

    # save new values to phi0 and sigma0, in order for them to be the initial condition for the next step
    fenics.assign([phi0, sigma0], u)

    # save current solutions to file
    phi_xdmf.write(phi0, t)  # write the value of phi at time t
    sigma_xdmf.write(sigma0, t)  # write the value of sigma at time t

    # update progress bar
    if rank == 0:
        progress_bar.update(1)
