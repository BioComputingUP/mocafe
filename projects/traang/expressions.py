import fenics


"""FEniCS expression used throughout the simulation"""


class AngiogenesisInitialCondition(fenics.UserExpression):
    """Initial condition for the vessel field, the tumor field (phi) and the angiogenic factor (T)"""
    def __init__(self, vessel_width, phi_max, phi_min, T0):
        super(AngiogenesisInitialCondition, self).__init__()
        self.vessel_width = vessel_width
        self.phi_max = phi_max
        self.phi_min = phi_min
        self.T0 = T0

    def eval(self, values, x):
        # set initial value to T
        values[0] = self.T0
        # set initial value to phi
        if x[0] < self.vessel_width:
            values[1] = self.phi_max
        else:
            values[1] = self.phi_min
        # set initial value to um
        values[2] = 0

    def value_shape(self):
        return (3,)

    def __floordiv__(self, other):
        pass
