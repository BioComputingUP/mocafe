import ufl
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc


def sigmoid(variable,
            c: float,
            a: float,
            b: float,
            slope: float):
    r"""
    Simple sigmoid function defined as:

    .. math::
        y(x) = \frac{a \cdot e^{slope \cdot c} + b \cdot e^{r \cdot x}}{e^{slope \cdot c} + e^{slope \cdot c}}

    Where: :math:`a` represent the value of the sigmoid for x -> -Inf; :math:`b` represent the value of the sigmoid
    for x -> +Inf; :math:`c` represent the center of the sigmoid (where the derivative is max); .math:`slope`
    represent the slope of the sigmoid.


    :param variable: variable x for which the sigmoid is defined
    :param c: central value of the sigmoid
    :param a: value of the sigmoid for variable -> -inf
    :param b: value of the sigmoid for variable -> +inf
    :param slope: slope of the sigmoid
    :return: the value of the sigmoid for the given value of the variable
    """
    return ((a * ufl.exp(c * slope)) + (b * ufl.exp(slope * variable))) / \
           (ufl.exp(c * slope) + ufl.exp(slope * variable))


def shf(variable, slope: float = 100):
    r"""
    Smoothed Heavyside Function (SHF) using the sigmoid function, which reads:

    .. math::
        \frac{e^{slope * variable}}{(1 + e^{slope * variable})}


    :param variable: varible for the SHF
    :param slope: slope of the SHF. Default is 100
    :return: the value of the sigmoid for the given value of the variable
    """
    return ufl.exp(slope * variable) / (1 + ufl.exp(slope * variable))


def estimate_field01_integral(phi: dolfinx.fem.Function):
    r"""
    Estimate phase field area for the input function, where the phase field must
    have values between 0 and 1

    :param phi: phase field representing the cancer
    :return: the area of the given cancer
    """
    # compute integral (only local on dolfinx)
    local_integral = dolfinx.fem.assemble_scalar(dolfinx.fem.form(phi * ufl.dx))
    # sum across all MPI domain and store the result on 0
    total_integral = MPI.COMM_WORLD.reduce(local_integral, op=MPI.SUM, root=0)
    # broadcast to all processes
    total_integral = MPI.COMM_WORLD.bcast(total_integral, root=0)
    return total_integral


def estimate_capillaries_area(c: dolfinx.fem.Function,
                              V: dolfinx.fem.FunctionSpace):
    """
    Estimate capillaries area for the phase field c representing the vessels. In order to work c must be
    between -1 and 1.

    :param c: phase field representing capillaries
    :param V: function space of c
    :return: the area of the given field
    """
    # rescale phase field c
    c_rescaled = dolfinx.fem.Function(V)
    c_rescaled.vector.array[:] = (c.vector.array + 1.) / 2
    # estimate area
    area = estimate_field01_integral(c_rescaled)
    # estimate area of the rescaled field
    return area


def project(v, target_func, bcs=[]):
    """
    Project a given form and store the result on the target function.

    Thanks to @michalhabera for the implementation.

    :param v: the form to be projected
    :param target_func: the function where the result will be stored
    :param bcs: boundary conditions, if any
    :return: nothing
    """
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = dolfinx.fem.form(ufl.inner(Pv, w) * dx)
    L = dolfinx.fem.form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)
    dolfinx.fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
