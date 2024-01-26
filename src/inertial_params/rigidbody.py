import inertial_params.util as util


class RigidBody:
    """A rigid body.

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

    def realizability_constraints(self):
        """Generate the constraints required for a set of inertial parameters
        to be realizable on the this rigid body.

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
