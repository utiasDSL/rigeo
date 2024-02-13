from collections.abc import Iterable

import numpy as np
import cvxpy as cp

import rigeo.util as util
from rigeo.inertial import InertialParameters


# TODO the only real use of this seems to be as a container that permits
# manipulation of shapes and inertial parameters together
class RigidBody:
    """A rigid body in three dimensions.

    The rigid body is defined by a list of shapes and a set of inertial parameters.

    Attributes
    ----------
    shapes : list
        A list of shapes, the union of which defines the shape of the body.
    params : InertialParameters
        The inertial parameters of the body. If none are provided, then they
        default to zero and the body acts like an "empty" (massless) shape.
    """

    def __init__(self, shapes, params=None):
        # TODO default value for params?
        if not isinstance(shapes, Iterable):
            shapes = [shapes]
        if params is None:
            params = InertialParameters(mass=0, h=np.zeros(0), H=np.zeros((3, 3)))
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

    def is_realizable(self, solver=None):
        """Check if the rigid body is density realizable."""
        return self.can_realize(self.params, solver=solver)

    def can_realize(self, params, solver=None):
        """Check if the rigid body can realize a set of inertial parameters."""
        # with one shape, we can just check
        if len(self.shapes) == 1:
            return shapes[0].can_realize(params)

        # otherwise we need to solve an opt problem
        Js = []
        constraints = []
        for shape in self.shapes:
            J = cp.Variable((4, 4), PSD=True)
            constraints.extend(shape.must_realize(J))
            Js.append(J)
        constraints.append(params.J == cp.sum(Js))

        # feasibility problem
        objective = cp.Minimize(0)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        return problem.status == "optimal"

    def must_realize(self, param_var):
        if len(self.shapes) == 1:
            return shapes[0].must_realize(param_var)
        # TODO
        pass

    def transform(self, rotation=None, translation=None):
        """Apply a rigid transform to the body.

        Parameters
        ----------
        rotation : np.ndarray, shape (d, d)
            Rotation matrix.
        translation : np.ndarray, shape (d,)
            Translation vector.

        Returns
        -------
        : RigidBody
            A new rigid body that has been rigidly transformed.
        """
        rotation, translation = util.clean_transform(
            rotation=rotation, translation=translation, dim=3
        )
        shapes = [
            shape.transform(rotation=rotation, translation=translation)
            for shape in self.shapes
        ]
        params = self.params.transform(rotation=rotation, translation=translation)
        return RigidBody(shapes=shapes, params=params)


# def density_realizable(shapes, params):
#     # one shape we can just check
#     if not isinstance(shapes, Iterable):
#         return shapes.can_realize(params)
#
#     # otherwise we need to solve an opt problem
#     Js = []
#     constraints = []
#     for shape in shapes:
#         J = cp.Variable((4, 4), PSD=True)
#         constraints.extend(shape.must_realize(J))
#         Js.append(J)
#     constraints.append(params.J == cp.sum(Js))
#
#     # feasibility problem
#     objective = cp.Minimize(0)
#     problem = cp.Problem(objective, constraints)
#     problem.solve()
#     return problem.status == "optimal"


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
