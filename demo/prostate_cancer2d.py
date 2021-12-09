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
# First of all, we import all we need to run the simulation.
import sys
import numpy as np
import fenics
from pathlib import Path
from mocafe.fenut.fenut import get_mixed_function_space, setup_xdmf_files
from mocafe.fenut.mansimdata import setup_data_folder
from mocafe.expressions import EllipseField
from mocafe.fenut.parameters import from_dict

# %%
# We also need to append the mocafe path to make it work.
file_folder = Path(__file__).parent.resolve()
mocafe_folder = file_folder.parent.parent
sys.path.append(str(mocafe_folder))

data_folder = setup_data_folder(sim_name=str(__file__).replace(".py", ""),
                                base_location=file_folder,
                                saved_sim_folder="demo_out")

phi_xdmf, sigma_xdmf = setup_xdmf_files(["phi", "sigma"], data_folder)

parameters = from_dict({
    "phi0_in": 1.,
    "phi0_out": 0.,
    "sigma0_in": 0.2,
    "sigma0_out": 1.
})

# %%
# Definition of the spatial domain and the function space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Simularly to the original paper, we are going to simulate the model on a 2D square mesh of dimension
# 2000 x 2000 :math:`\mu m`. This is pretty simple to do using FEniCs, which provides the class ``RectangleMesh``
# to do this job.
#
# More precisely, in the following we are going to define a mesh of the dimension described above, with 512 elements for
# each side.
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
# From the mesh defined above, we can then define the ``FunctionSpace``, that is the set of the piece-wise polynomial
# function to be used to represent our solution computed using the finite element method (FEM). Since the model we wish
# to simulate is composed of two coupled equations, we need to define a MixedElement function space with two different
# elements. In this implementation, we will used for both equations the same element type, "CG" (Continuous Galerking),
# of the first order, which can be created in FEniCS simply using::
#
#     cg1_element = fenics.FiniteElement("CG", fenics.triangle, 1)
#     mixed_element = fenics.MixedElement([cg1_element] * 2)
#     function_space = fenics.FunctionSpace(mesh, mixed_element)
#
# However, the very same operation can be performed quicker using a util method of mocafe, which can
# be used as follows:
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
#    phi0_max = 1
#    phi0_min = 0
#    # cpp code that returns True if the point x is inside the ellipse, and False otherwise
#    is_in_ellipse_cpp_code = "((pow(x[0] / semiax_x, 2)) + (pow(x[1] / semiax_y, 2)) <= 1)"
#    # cpp code that returns 1 if the above statement is True, and 0 otherwise
#    phi0_cpp_code = is_in_ellipse_cpp_code + " ? phi0_max : phi0_min"
#    # FEniCS expression, built from cpp code defined above
#    phi0 = fenics.Expression(phi0_cpp_code,
#                             degree=2,
#                             semiax_x=semiax_x, semiax_y=semiax_y,
#                             phi0_max=phi0_max, phi0_min=phi0_min)
#
# However, if you don't feel confident in defininf your own expression, you can use the one provided by mocafe:
phi0 = EllipseField(center=np.array([0., 0.]),
                    semiax_x=semiax_x,
                    semiax_y=semiax_y,
                    inside_value=parameters.get_value("phi0_in"),
                    outside_value=parameters.get_value("phi0_out"))

# %%
# The FEniCS expression must then be projected or interpolated in the function space. Notice that since the
# function space we defined is mixed, we must choose one of the sub-field to define the function.
phi0 = fenics.interpolate(phi0, function_space.sub(0).collapse())
phi_xdmf.write(phi0, 0)

# %%
# Notice also that since the mixed function space is defined by two identical function spaces, it makes no difference
# to pick sub(0) or sub(1).
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

u_old = fenics.Function(function_space)
fenics.assign(u_old, [phi0, sigma0])
u = fenics.Function(function_space)
fenics.assign(u, [phi0, sigma0])
phi, sigma = fenics.split(u)





