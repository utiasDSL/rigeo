"""Three-dimensional rigid bodies."""
from collections.abc import Iterable

import numpy as np
import cvxpy as cp

import rigeo.util as util
from rigeo.inertial import InertialParameters
from rigeo.constraint import pim_must_equal_param_var


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
        if not isinstance(shapes, Iterable):
            shapes = [shapes]
        if params is None:
            params = InertialParameters.zero()

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
        """Check if the rigid body is density realizable.

        Parameters
        ----------
        solver : str or None
            If checking realizability requires solving an optimization problem,
            one can optionally be specified.

        Returns
        -------
        : bool
            ``True`` if ``self.params`` is realizable on ``self.shapes``,
            ``False`` otherwise.
        """
        return self.can_realize(self.params, solver=solver)

    def can_realize(self, params, solver=None):
        """Check if the rigid body can realize a set of inertial parameters.

        Parameters
        ----------
        params : InertialParameters
            The inertial parameters to check.
        solver : str or None
            If checking realizability requires solving an optimization problem,
            one can optionally be specified.

        Returns
        -------
        : bool
            ``True`` if the parameters are realizable, ``False`` otherwise.
        """
        # with one shape, we can just check
        if len(self.shapes) == 1:
            return self.shapes[0].can_realize(params, solver=solver)

        # otherwise we need to solve an opt problem
        J = cp.Variable((4, 4), PSD=True)
        constraints = self.must_realize(J, eps=0) + [J == params.J]

        # feasibility problem
        objective = cp.Minimize(0)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        return problem.status == "optimal"

    def must_realize(self, param_var, eps=0):
        """Generate cvxpy constraints for inertial parameters to be realizable
        on this body.

        Parameters
        ----------
        param_var : cp.Expression, shape (4, 4) or shape (10,)
            The cvxpy inertial parameter variable. If shape is ``(4, 4)``, this
            is interpreted as the pseudo-inertia matrix. If shape is ``(10,)``,
            this is interpreted as the inertial parameter vector.
        eps : float, non-negative
            Pseudo-inertia matrix ``J`` is constrained such that ``J - eps *
            np.eye(4)`` is positive semidefinite and J is symmetric.

        Returns
        -------
        : list
            List of cvxpy constraints.
        """
        if len(self.shapes) == 1:
            return shapes[0].must_realize(param_var, eps=eps)

        J, psd_constraints = pim_must_equal_param_var(param_var, eps=eps)
        Js = [cp.Variable((4, 4), PSD=True) for _ in self.shapes]
        return (
            [
                c
                for shape, J in zip(self.shapes, Js)
                for c in shape.must_realize(J, eps=0)
            ]
            + [J == cp.sum(Js)]
            + psd_constraints
        )

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

    @staticmethod
    def regressor(V, A):
        """Compute regressor matrix ``Y`` for the body.

        The regressor maps the inertial parameters to the body inertial wrench: w = Yθ.

        Parameters
        ----------
        V : np.ndarray, shape (6,)
            Body-frame velocity.
        A : np.ndarray, shape (6,)
            Body-frame acceleration.

        Returns
        -------
        : np.ndarray, shape (6, 10)
            The regressor matrix.
        """
        return util.lift6(A) + util.skew6(V) @ util.lift6(V)
