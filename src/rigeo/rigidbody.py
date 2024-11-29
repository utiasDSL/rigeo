"""Three-dimensional rigid bodies."""
from collections.abc import Iterable
from dataclasses import dataclass
import time

import numpy as np
import cvxpy as cp

from .util import clean_transform, lift6
from .inertial import InertialParameters
from .constraint import pim_must_equal_param_var
from .shape.ellipsoid import Ellipsoid


@dataclass
class VerificationStats:
    """Stats of parameter verification optimization."""

    iters: int
    solve_time: float


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

    # def is_realizable(self, eps=0, verbose=False, **kwargs):
    #     """Check if the rigid body is density realizable.
    #
    #     Parameters
    #     ----------
    #     solver : str or None
    #         If checking realizability requires solving an optimization problem,
    #         one can optionally be specified.
    #
    #     Returns
    #     -------
    #     : bool
    #         ``True`` if ``self.params`` is realizable on ``self.shapes``,
    #         ``False`` otherwise.
    #     """
    #     return self.can_realize(self.params, eps=eps, verbose=verbose, **kwargs)
    #
    # def can_realize(self, params, eps=0, verbose=False, **kwargs):
    #     """Check if the rigid body can realize a set of inertial parameters.
    #
    #     Parameters
    #     ----------
    #     params : InertialParameters
    #         The inertial parameters to check.
    #     eps : float, non-negative
    #         Pseudo-inertia matrix ``J`` is constrained such that ``J - eps *
    #         np.eye(4)`` is positive semidefinite and J is symmetric.
    #
    #     Additional arguments are passed to the solver.
    #
    #     Returns
    #     -------
    #     : bool
    #         ``True`` if the parameters are realizable, ``False`` otherwise.
    #     """
    #     # with one shape, we can just check
    #     if len(self.shapes) == 1:
    #         # TODO also return stats here if verbose=True
    #         return self.shapes[0].can_realize(params, **kwargs)
    #
    #     # otherwise we need to solve an opt problem
    #     J = cp.Variable((4, 4), PSD=True)
    #     constraints = self.must_realize(J, eps=eps) + [J == params.J]
    #
    #     # feasibility problem
    #     objective = cp.Minimize(0)
    #     problem = cp.Problem(objective, constraints)
    #
    #     t0 = time.time()
    #     problem.solve(**kwargs)
    #     t1 = time.time()
    #
    #     solved = problem.status == "optimal"
    #
    #     if verbose:
    #         stats = VerificationStats(
    #             iters=problem.solver_stats.num_iters,
    #             solve_time=t1 - t0,
    #         )
    #         return solved, stats
    #     return solved

    def moment_sdp_constraints(self, param_var, eps=0, d=2):
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
        def moment_constraints(shape, J, eps=0):
            if isinstance(shape, Ellipsoid):
                return shape.moment_constraints(J, eps=eps)
            return shape.moment_sdp_constraints(J, eps=eps, d=d)

        if len(self.shapes) == 1:
            return moment_constraints(self.shapes[0], param_var, eps=eps)

        J, psd_constraints = pim_must_equal_param_var(param_var, eps=eps)
        Js = [cp.Variable((4, 4), PSD=True) for _ in self.shapes]

        return (
            [
                c
                for shape, J in zip(self.shapes, Js)
                for c in moment_constraints(shape, J)
            ]
            + [J == cp.sum(Js)]
            + psd_constraints
        )

    def mbes(self, sphere=False, solver=None):
        """Generate a new rigid body with each shape replaced with its bounding ellipsoid.

        Parameters
        ----------
        sphere : bool
            If ``True``, use bounding spheres rather than ellipsoids.
        solver : str or None
            If generating the minimum bounding ellipsoid requires solving an
            optimization problem, a solver can optionally be specified.

        Returns
        -------
        : RigidBody
            The new body with the same inertial parameters but each shapes
            replaced by its minimum-volume bounding ellipsoid.
        """
        shapes = [shape.mbe(sphere=sphere, solver=solver) for shape in self.shapes]
        return RigidBody(shapes=shapes, params=self.params)

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
        rotation, translation = clean_transform(
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
        V : SV
            Body-frame spatial velocity.
        A : SV
            Body-frame spatial acceleration.

        Returns
        -------
        : np.ndarray, shape (6, 10)
            The regressor matrix.
        """
        return lift6(A) - V.adjoint().T @ lift6(V)
