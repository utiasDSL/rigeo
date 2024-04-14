from dataclasses import dataclass
import time

import numpy as np
import cvxpy as cp

from .constraint import pim_psd, pim_must_equal_vec
from .inertial import InertialParameters


def entropic_regularizer(Js, J0s):
    """Entropic regularizer for inertial parameter identification.

    See :cite:t:`lee2019geometric`. The regularizer is convex in ``Js``.

    Parameters
    ----------
    Js : cxvpy.Expression or Iterable[cxvpy.Expression]
        The pseudo-inertia matrix variables to regularize.
    J0s : np.ndarray, shape (4, 4), or Iterable[np.ndarray]
        The nominal values for the pseudo-inertia matrices.

    Returns
    -------
    : cxvpy.Expression
        The cvxpy expression for the regularizer.

    """
    J0s = np.array(J0s)
    if J0s.ndim == 2:
        J0s = [J0s]
        Js = [Js]
    assert len(Js) == len(J0s)
    assert Js[0].shape == (4, 4)
    assert J0s[0].shape == (4, 4)

    return cp.sum(
        [-cp.log_det(J) + cp.trace(np.linalg.inv(J0) @ J) for J, J0 in zip(Js, J0s)]
    )


def least_squares_objective(θs, As, bs, W0=None):
    """Least squares objective function

    .. math::
       \\sum_i \\|\\boldsymbol{A}_i\\boldsymbol{\\theta} - \\boldsymbol{b}_i\\|^2

    where :math:`\\boldsymbol{\\theta}=[\\boldsymbol{\\theta}_1,\\dots,\\boldsymbol{\\theta}_m]`.

    Parameters
    ----------
    θs : cxvpy.Expression or Iterable[cxvpy.Expression]
        The regressor variables.
    As : np.ndarray or Iterable[np.ndarray]
        The regressor matrices.
    bs : np.ndarray or Iterable[np.ndarray]
        The regressor vectors.
    W0 : np.ndarray or None
        The inverse measurement covariance matrix. Defaults to identity if not
        provided.

    Returns
    -------
    : cxvpy.Expression
        The cxvpy expression for the objective.
    """
    if W0 is None:
        W0 = np.eye(bs.shape[1])

    # psd_wrap fixes an occasional internal scipy error
    # https://github.com/cvxpy/cvxpy/issues/1421#issuecomment-865977139
    W = cp.psd_wrap(np.kron(np.eye(bs.shape[0]), W0))

    A = np.vstack(As)
    b = np.concatenate(bs)
    θ = cp.hstack(θs)
    return cp.quad_form(A @ θ - b, W)


@dataclass
class IdentificationResult:
    """Result of parameter identification optimization."""

    params: list
    objective: float
    iters: int
    solve_time: float


class IdentificationProblem:
    """Inertial parameter identification problem.

    The problem is formulated as a convex, constrained least-squares problem
    and solved via cxvpy.

    Attributes
    ----------
    As : np.ndarray or Iterable[np.ndarray]
        The regressor matrices.
    bs : np.ndarray or Iterable[np.ndarray]
        The regressor vectors.
    γ : float, non-negative
        The coefficient of the regularization term.
    ε : float, non-negative
        The value such that each body satisfies
        :math:`\\boldsymbol{J}\\succcurlyeq\\epsilon\\boldsymbol{1}_4`.
    solver : str or None
        The underlying solver for cvxpy to use.
    problem : cxvpy.Problem
        Once ``solve`` has been called, the underlying ``cvxpy.Problem``
        instance is made available for inspection.
    """

    def __init__(self, As, bs, γ=0, ε=0, **kwargs):
        assert As.shape[0] == bs.shape[0]
        assert γ >= 0
        assert ε >= 0

        self.no = As.shape[0]  # number of observations

        self.As = As
        self.bs = bs

        self.γ = γ
        self.ε = ε

        self.solve_kwargs = kwargs

    def solve(self, bodies, must_realize=True):
        """Solve the identification problem.

        Additional ``kwargs`` are passed to the `solve` method of the
        `cvxpy.Problem` instance.

        Parameters
        ----------
        bodies : Iterable[RigidBody]
            The rigid bodies used to (1) constrain the parameters to be
            realizable within their shapes and (2) to provide nominal
            parameters for regularization.
        must_realize : bool
            If ``True``, enforce density realizable constraints. If ``False``,
            the problem is unconstrained except that each pseudo-inertia matrix
            must be positive definite.

        Returns
        -------
        : Iterable[InertialParameters]
            The identified inertial parameters for each body.
        """
        # variables
        θs = [cp.Variable(10) for _ in bodies]
        Js = [pim_must_equal_vec(θ) for θ in θs]

        # objective
        J0s = [body.params.J for body in bodies]
        regularizer = entropic_regularizer(Js, J0s)

        lstsq = least_squares_objective(θs, self.As, self.bs)
        cost = 0.5 / self.no * lstsq + self.γ * regularizer
        objective = cp.Minimize(cost)

        # constraints
        constraints = [c for J in Js for c in pim_psd(J, self.ε)]
        if must_realize:
            for body, J in zip(bodies, Js):
                constraints.extend(body.must_realize(J))

        problem = cp.Problem(objective, constraints)

        # no warm start because we don't want the different problems
        # influencing each other
        # solve_kwargs = {"solver": self.solver, "warm_start": False, **kwargs}

        t0 = time.time()
        problem.solve(**self.solve_kwargs)
        t1 = time.time()

        assert (
            problem.status == "optimal"
        ), f"Optimization failed with status {problem.status}"

        return IdentificationResult(
            params=[InertialParameters.from_vec(θ.value) for θ in θs],
            objective=objective.value,
            iters=problem.solver_stats.num_iters,
            solve_time=t1 - t0,
        )
