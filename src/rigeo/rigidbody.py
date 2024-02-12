import numpy as np
import cvxpy as cp

import rigeo.util as util


# TODO the only real use of this seems to be as a container that permits
# manipulation of shapes and inertial parameters together
class RigidBody:
    """A rigid body in three dimensions.

    The rigid body is defined by a list of shapes and a set of inertial parameters.
    """

    def __init__(self, shapes, params):
        if not isinstance(shapes, Iterable):
            shapes = [shapes]
        self.shapes = shapes
        self.params = params

    def __add__(self, other):
        shapes = self.shapes + other.shapes
        params = self.params + other.params
        return RigidBody(shapes=shapes, params=params)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def density_realizable(self):
        """Check if the rigid body is density realizable."""
        pass

    def transform(self, rotation=None, translation=None):
        pass


def density_realizable(shapes, params):
    # one shape we can just check
    if not isinstance(shapes, Iterable):
        return shapes.can_realize(params)

    # otherwise we need to solve an opt problem
    Js = []
    constraints = []
    for shape in shapes:
        J = cp.Variable((4, 4), PSD=True)
        constraints.extend(shape.must_realize(J))
        Js.append(J)
    constraints.append(params.J == cp.sum(Js))

    # feasibility problem
    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.status == "optimal"


def must_be_density_realizable(shapes, param_var):
    """Generate the constraints required for a set of inertial parameters
    to be realizable on the this rigid body.

    param_var can be either θ or J

    Returns
    -------
    : list
        A list of cvxpy constraints.
    """
    pass


def body_regressor(V, A):
    """Compute regressor matrix Y given body frame velocity V and acceleration A.

    The regressor maps the inertial parameters to the body inertial wrench: w = Yθ.
    """
    return util.lift6(A) + util.skew6(V) @ util.lift6(V)
