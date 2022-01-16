import sys
import fenics
import time
from tqdm import tqdm
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
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


class SNESProblem:
    def __init__(self, F, u, bcs):
        V = u.function_space()
        du = fenics.TrialFunction(V)
        self.L = F
        self.a = fenics.derivative(F, u, du)
        self.bcs = bcs
        self.u = u

    def F(self, snes, x, F):
        x = fenics.PETScVector(x)
        F = fenics.PETScVector(F)
        x.vec().copy(self.u.vector().vec())
        self.u.vector().apply("")
        fenics.assemble(self.L, tensor=F)
        for bc in self.bcs:
            bc.apply(F, x)
            bc.apply(F, self.u.vector())

    def J(self, snes, x, J, P):
        J = fenics.PETScMatrix(J)
        x.copy(self.u.vector().vec())
        self.u.vector().apply("")
        fenics.assemble(self.a, tensor=J)
        for bc in self.bcs:
            bc.apply(J)


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
        try:
            fenics.solve(weak_form == 0, function, J=jacobian, solver_parameters=lsp)
            error_occurred = False
        except RuntimeError:
            error_occurred = True
        total_time = time.time() - init_time
        return error_occurred, total_time

    def find_quicker_gmres_pc(self, weak_form, function, jacobian):
        # init dict
        performance_dict = {
            "solver": [],  # solver type
            "preconditioner": [],  # preconditioner (None if not krylov solver)
            "error": [],
            "duration": []
        }
        # set progress bar
        if rank == 0:
            pbar = tqdm(iterable=list(self.default_pc))
        else:
            pbar = None
        # measure performance for all pcs
        for pc in self.default_pc:
            if rank == 0:
                pbar.set_description(f"testing gmres with {pc}")

            lsp = {"newton_solver": {"linear_solver": "gmres", "preconditioner": pc}}
            error_occurred, total_time = self._run_default_solver_with_parameters(weak_form, function, jacobian,
                                                                                  lsp)
            # update dict
            performance_dict["solver"].append("gmres")
            performance_dict["preconditioner"].append(pc)
            performance_dict["error"].append(error_occurred)
            performance_dict["duration"].append(None if error_occurred else total_time)
            # update pbar
            if rank == 0:
                pbar.update(1)

        return performance_dict

    def _run_snes_solver_with_parameters(self, problem: SNESProblem, parameters_list):
        # init snes solver
        petsc4py.init([__name__, *parameters_list])
        # define vector and matrix
        b = fenics.PETScVector()
        J_mat = fenics.PETScMatrix()
        # define snes solver
        snes = PETSc.SNES().create(comm)
        snes.setFromOptions()
        snes.setFunction(problem.F, b.vec())
        snes.setJacobian(problem.J, J_mat.mat())
        # try to solve
        init_time = time.time()
        try:
            snes.solve(None, problem.u.vector().vec())
            error_occurred = False
        except:
            error_occurred = True
        total_time = time.time() - init_time
        return error_occurred, total_time

    def find_qucker_snes_solver(self, weak_form,  function):
        # init performance dict
        performance_dict = {
            "snes_type": [],  # solver type
            "error": [],
            "duration": [],
        }
        # # get ksp types
        # ksp_types = [PETSc.KSP.Type().__getattribute__(attribute)
        #              for attribute in dir(PETSc.KSP.Type) if attribute.isupper()]
        # # get pc types
        # pc_types = [PETSc.PC.Type().__getattribute__(attribute)
        #             for attribute in dir(PETSc.PC.Type) if attribute.isupper()]
        # # build parameters dicts
        # parameters_lists = [["-ksp_type", ksp_type, "-pc_type", pc_type]
        #                     for ksp_type, pc_type in product(ksp_types, pc_types)]
        snes_types = [PETSc.SNES.Type().__getattribute__(attribute)
                      for attribute in dir(PETSc.SNES.Type) if attribute.isupper()]
        parameters_lists = [["-snes_type", snes_type] for snes_type in snes_types]
        # def problem
        problem = SNESProblem(weak_form, function, [])
        # init pbar
        if rank == 0:
            pbar = tqdm(iterable=parameters_lists)
        else:
            pbar = None
        # solve for each
        for p_list in parameters_lists:
            error_occurred, total_time = self._run_snes_solver_with_parameters(problem, p_list)

            # update dict
            performance_dict["snes_type"].append(p_list[1])
            performance_dict["error"].append(error_occurred)
            performance_dict["duration"].append(None if error_occurred else total_time)

            # update pbar
            if rank == 0:
                pbar.update(1)

        return performance_dict










