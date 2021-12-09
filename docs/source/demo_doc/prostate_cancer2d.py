r"""
Prostate cancer phase field model
============================================

This demo presents how to simulate a prostate cancer phase field model presented by G. Lorenzo and collaborators
in 2016 :cite:`Lorenzo2016` using FEniCS and mocafe.

How to run this example on mocafe
-------------------------------------------
todo

Brief introduction to the mathematical model
--------------------------------------------

The model was published on PNAS in 2016 and presentes a continuous mathematical model able to reproduce the
growth pattern of prostate cancer at tissue scale :cite:`Lorenzo2016`.

The model is composed of just two equations. The first one is for the cancer phase field :math:`\varphi`, and reads:

.. math::
    \frac{\partial \varphi}{\partial t} = \lambda \nabla^2 \varphi - \frac{1}{\tau}\frac{dF(\varphi)}{d\varphi}
    + \chi \sigma - A \varphi

Where :math:`F(\varphi)` is the following double-well potential:

.. math::
    F(\varphi) = 16\cdot \varphi^2 \cdot (1 - \varphi)^2

The second equation is for the nutrient concentration :math:`\sigma`:

.. math::
    \frac{\partial \sigma}{\partial t} = \epsilon \nabla^2\sigma + s - \delta\cdot\varphi - \gamma\cdot\sigma
"""

# %%
# Implementation
# ------------------------------------------
#
# Setup
# ^^^^^
# First of all, we import all we need to run the simulation.
import sys
import numpy as np
import fenics
import random
from tqdm import tqdm
from pathlib import Path
file_folder = Path(__file__).parent.resolve()
mocafe_folder = file_folder.parent
sys.path.append(str(mocafe_folder))  # appending mocafe path. Must be removed
from mocafe.fenut.solvers import PETScProblem, PETScSolver
from mocafe.fenut.fenut import get_mixed_function_space, setup_xdmf_files
from mocafe.fenut.mansimdata import setup_data_folder
from mocafe.expressions import EllipseField, PythonFunctionField
from mocafe.fenut.parameters import from_dict
import mocafe.litforms.prostate_cancer as pc_model



# %%
# Then, it is useful (even though not necessary) to do a number of operations befor running our simulation.
#
# First of all, we shut down the logging messages from FEniCS. You can comment this line if you want.
fenics.set_log_level(fenics.LogLevel.ERROR)

# %%
# Then, we define the MPI rank for each process. Generally speaking, this is necessary for running the simulation in
# parallel using ``mpirun``, even though in this simulation is not largely used, as we are going to see.
comm = fenics.MPI.comm_world
rank = comm.Get_rank()

# %%
# Then, we can define the files where to save our result. The suggested format for saving simulations is using
# ``.xdmf`` files, which can easily visualized in `Paraview <https://www.paraview.org/>`_.
#
# In the following, we use two mocafe methods for defining:
#
# - first, the folder where to save the result of the simulation. In this case, the folder will be based inside
#   the current folder (``base_location``) and it's called demo_out/prostate_cancer2d;
# - then, the two files for the cancer :math:`\varphi` and for the nutrients :math:`\sigma`, which will be called
#   ``phi.xdmf`` and ``sigma.xdmf``.
#
data_folder = setup_data_folder(sim_name=str(__file__).replace(".py", ""),
                                base_location=file_folder,
                                saved_sim_folder="demo_out")

phi_xdmf, sigma_xdmf = setup_xdmf_files(["phi", "sigma"], data_folder)

# %%
# Finally, we define the parameters of the differential equation using a mocafe object created for this purpose,
# Parameters. A Parameters object can be initialized in several ways. In the following, we define it from a
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
#
# Similarly to the original paper, we are going to simulate the model on a 2D square mesh of dimension
# 2000 x 2000 :math:`\mu m`. This is pretty simple to do using FEniCs, which provides the class ``RectangleMesh``
# to do this job.
#
# More precisely, in the following we are going to define a mesh of the dimension described above, with 512
# points for each side.
#
nx = 512
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
# From the mesh defined above, we can then define the ``FunctionSpace``, that is the set of the piece-wise
# polynomial function to be used to represent our solution computed using the finite element method (FEM). Since
# the model we wish to simulate is composed of two coupled equations, we need to define a MixedElement function
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
# Initial conditions
# ^^^^^^^^^^^^^^^^^^^
# Since the system of differential equations involves time, we need to define initial conditions for both
# :math:`\varphi` and :math`\sigma`. According to the original paper as initial condition for :math:`\varphi`
# we will define an elliptical tumor with the given semiaxes:
semiax_x = 100  # um
semiax_y = 150  # um

# %%
# With FEniCS we can do so by defining an expression which represent mathematically our initial condition:
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
# However, if you don't feel confident in defining your own expression, you can use the one provided by mocafe:
phi0 = EllipseField(center=np.array([0., 0.]),
                    semiax_x=semiax_x,
                    semiax_y=semiax_y,
                    inside_value=parameters.get_value("phi0_in"),
                    outside_value=parameters.get_value("phi0_out"))

# %%
# The FEniCS expression must then be projected or interpolated in the function space in order to obtain a
# fenics Function. Notice that since the function space we defined is mixed, we must choose one of the
# sub-field to define the function.
phi0 = fenics.interpolate(phi0, function_space.sub(0).collapse())
phi_xdmf.write(phi0, 0)

# %%
# Notice also that since the mixed function space is defined by two identical function spaces, it makes no
# difference to pick sub(0) or sub(1).
#
# After having defined the initial condition for :math:`\varphi`, let's define the initial for :math:`\sigma` in a
# similar fashion:
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
# After having defined the initial conditions for the system, we can proceed with the definition of the system
# itself.
#
# First of all, we define the two variables, ``phi`` and ``sigma``, for which the system will be solved:
# define bidim function
u = fenics.Function(function_space)
fenics.assign(u, [phi0, sigma0])
phi, sigma = fenics.split(u)

# %%
# Then, we define the test functions for defining the weak forms of the PDEs:
v1, v2 = fenics.TestFunctions(function_space)

# %%
# Now, let's define an expression for s
s_expression = PythonFunctionField(
    python_fun=lambda: parameters.get_value("s_average") + random.uniform(parameters.get_value("s_min"),
                                                                          parameters.get_value("s_max")),
)

# %%
# And the weak form of the system, which is already defined in mocafe
weak_form = pc_model.prostate_cancer_form(phi, phi0, sigma, v1, parameters) + \
    pc_model.prostate_cancer_nutrient_form(sigma, sigma0, phi, v2, s_expression, parameters)

# %%
# Simulation
# ^^^^^^^^^^
# Simulating this mathematical model is just a matter of solving the PDE system defined above for each time step.
# To do so, we define a Problem and a Solver directly calling the PETSc backend.
jacobian = fenics.derivative(weak_form, u)  # jacobian of the system

problem = PETScProblem(jacobian, weak_form, [])
solver = PETScSolver({"ksp_type": "gmres", "pc_type": "asm"}, mesh.mpi_comm())

# %%
# Then we initialize a progress bar with tqdm
n_steps = 100
if rank == 0:
    progress_bar = tqdm(total=n_steps, ncols=100)
else:
    progress_bar = None

# %%
# And finally we iterate in time and solve the system at each time step
t = 0
for current_step in range(n_steps):
    # update time
    t += parameters.get_value("dt")

    # solve problem for current time
    solver.solve(problem, u.vector())

    # update values
    fenics.assign([phi0, sigma0], u)

    # save current solutions to file
    phi_xdmf.write(phi0, t)
    sigma_xdmf.write(sigma0, t)

    # update pbar
    if rank == 0:
        progress_bar.update(1)


