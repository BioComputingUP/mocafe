import fenics
import numpy as np
import scipy.sparse as sps
import cupy
import cupyx
import cupyx.scipy.sparse.linalg
import logging
import pathlib

mempool = cupy.get_default_memory_pool()

with cupy.cuda.Device(0):
    mempool.set_limit(size=10.5*1024**3)

# get process rank
rank = fenics.MPI.comm_world.Get_rank()

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create log file
logfolder = pathlib.Path(f"./runtime/log")
logfolder.mkdir(parents=True, exist_ok=True)
logfile = logfolder / pathlib.Path(f"{__name__}.log")
if fenics.MPI.comm_world.Get_rank() == 0:
    f = open(str(logfile), "w")
    f.close()
else:
    pass

# define file handler
ch = logging.FileHandler(filename=str(logfile), mode='a')
ch.setLevel(logging.DEBUG)

# add ch to logger
logger.addHandler(ch)


def cupy_newton_solver(form: fenics.Form,
                       sol: fenics.Function,
                       V: fenics.FunctionSpace,
                       u_prec: fenics.Function):

    # define initial solution u_k
    u_k = fenics.Function(V)
    fenics.assign(u_k, u_prec)

    # define function du
    du = fenics.Function(V)

    # define function u_kp1
    u_kp1 = fenics.Function(V)

    # start iterating
    omega = 1.0
    eps = 1
    tol = 1E-5
    ind = 0
    itermax = 100
    while (eps > tol) and (ind < itermax):
        # compute the action of the form on the initial vector u_k
        F = fenics.action(form, u_k)
        # assemble F, that will be the right hand side of the system
        b = fenics.assemble(-F)
        # get b as numpy array
        b = b[:]
        logger.debug(f"step {ind}, Assembled F: {b}")
        # compute the Jacobian of F
        J = fenics.derivative(F, u_k)
        # assemble J
        A = fenics.assemble(J)
        # convert A to sparse matrix
        row, col, val = fenics.as_backend_type(A).mat().getValuesCSR()
        A = sps.csr_matrix((val, col, row))
        # then convert A nd b to cupy-compatible data:
        bs = cupy.array(b)
        As = cupyx.scipy.sparse.csr_matrix(A)
        logger.debug(f"step {ind}, Assembled J: {A}")
        # solve linear system
        du.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As, bs)[:1][0])
        # compute norm
        eps = np.linalg.norm(du.vector().get_local(), ord=np.Inf)
        logger.debug(f"step {ind}, eps = {eps}")
        # compute new solution
        u_kp1.vector()[:] = u_k.vector() + omega * du.vector()
        fenics.assign(u_k, u_kp1)
        ind = ind + 1

    fenics.assign(sol, u_k)
