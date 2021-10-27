import fenics


class PETScProblem(fenics.NonlinearProblem):
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


class PETScSolver(fenics.NewtonSolver):
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
