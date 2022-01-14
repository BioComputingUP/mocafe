import fenics
import time
from tqdm import tqdm
from itertools import product

comm = fenics.MPI.comm_world
rank = comm.Get_rank()


class PETScProblem(fenics.NonlinearProblem):
    """
    INTERNAL USE

    Defines a nonlinear problem directly calling the PETSc linear algebra backend. This is the preferred way of
    defining a problem.
    """
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        fenics.NonlinearProblem.__init__(self)

    def F(self, b, x):
        fenics.assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        fenics.assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class PETScNewtonSolver(fenics.NewtonSolver):
    """
    INTERNAL USE

    Defines a solver for a nonlinear problem directly calling the PETSc linear algebra backend.
    """
    def __init__(self, solver_parameters: dict, comm=fenics.MPI.comm_world):
        self.solver_parameters = solver_parameters
        fenics.NewtonSolver.__init__(self, comm,
                                     fenics.PETScKrylovSolver(), fenics.PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        for option in self.solver_parameters:
            if self.solver_parameters[option] is None:
                fenics.PETScOptions.set(option)
            else:
                fenics.PETScOptions.set(option, self.solver_parameters[option])

        self.linear_solver().set_from_options()


class BestSolverFinder:
    def __init__(self):
        self.default_lu_methods: dict = fenics.lu_solver_methods()
        self.default_krylov_methods: dict = fenics.krylov_solver_methods()
        self.default_pc: dict = fenics.krylov_solver_preconditioners()

    def _run_default_solver_with_parameters(self, weak_form, function, jacobian, lsp):
        init_time = time.time()
        fenics.solve(weak_form == 0, function, J=jacobian, solver_parameters=lsp)
        # try:
        #     fenics.solve(weak_form == 0, function, J=jacobian, solver_parameters=lsp)
        #     error_occurred = False
        # except RuntimeError as e:
        #     error_occurred = True
        error_occurred = False
        total_time = time.time() - init_time
        return error_occurred, total_time

    def find_quicker_nonlinear_solver(self, weak_form, function, jacobian):
        # init dict
        performance_dict = {
            "solver": [],  # solver type
            "preconditioner": [],  # preconditioner (None if not krylov solver)
            "error": [],
            "duration": []
        }
        # set progress bar
        if rank == 0:
            pbar = tqdm(iterable=[*list(product(self.default_lu_methods, [None])),
                              *list(product(self.default_krylov_methods, self.default_pc))])
            pbar.set_description("testing lu solvers:")
        else:
            pbar = None
        # measure performance for lu methods
        for method in self.default_lu_methods:
            lsp = {"newton_solver": {"linear_solver": method}}
            error_occurred, total_time = self._run_default_solver_with_parameters(weak_form, function, jacobian, lsp)
            # update dict
            performance_dict["solver"].append(method)
            performance_dict["preconditioner"].append(None)
            performance_dict["error"] = error_occurred
            performance_dict["duration"] = None if error_occurred else total_time
            # update pbar
            if rank == 0:
                pbar.update(1)

        # measure performance for krylov solvers
        if rank == 0:
            pbar.set_description("testing krylov solvers:")
        for method in self.default_krylov_methods:
            for pc in self.default_pc:
                lsp = {"newton_solver": {"linear_solver": method, "preconditioner": pc}}
                error_occurred, total_time = self._run_default_solver_with_parameters(weak_form, function, jacobian,
                                                                                      lsp)
                # update dict
                performance_dict["solver"].append(method)
                performance_dict["preconditioner"].append(pc)
                performance_dict["error"] = error_occurred
                performance_dict["duration"] = None if error_occurred else total_time
                # update pbar
                if rank == 0:
                    pbar.update(1)

        return performance_dict






