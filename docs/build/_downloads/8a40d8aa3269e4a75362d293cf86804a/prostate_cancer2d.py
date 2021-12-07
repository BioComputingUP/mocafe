r"""
Prostate cancer phase field model
============================================

This demo presents how to simulate a prostate cancer phase field model presented by G. Lorenzo and collaborators
in 2016 :cite:`Lorenzo2016` using FEniCS and mocafe.

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

.. bibliography:: references.bib

Implementation
------------------------------------------

In the following we will implement the prostate cancer model using FEniCS - with the help of mocafe, starting from the
import of the needed modules and the definition of the MPI communicator and rank, which is needed for the parallel
computation:
"""

import sys
from pathlib import Path
file_folder = Path(__file__).parent.resolve()
mocafe_folder = file_folder.parent.parent
sys.path.append(str(mocafe_folder))

import fenics
from mocafe.fenut.fenut import get_mixed_function_space


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

