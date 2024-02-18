import numpy as np
import cvxpy as cp

from rigeo.constraint import pim_psd, pim_must_equal_vec
from rigeo.inertial import InertialParameters


def entropic_regularizer(Js, J0s):
    # TODO handle single values
    assert len(Js) == len(J0s)
    return cp.sum(
        [-cp.log_det(J) + cp.trace(np.linalg.inv(J0) @ J) for J, J0 in zip(Js, J0s)]
    )


def least_squares_objective(θs, As, bs, W0=None):
    if W0 is None:
        W0 = np.eye(bs.shape[1])

    # psd_wrap fixes an occasional internal scipy error
    # https://github.com/cvxpy/cvxpy/issues/1421#issuecomment-865977139
    W = cp.psd_wrap(np.kron(np.eye(bs.shape[0]), W0))

    A = np.vstack(As)
    b = np.concatenate(bs)
    θ = cp.hstack(θs)
    return cp.quad_form(A @ θ - b, W)


class IdentificationProblem:
    def __init__(self, As, bs, γ=0, ε=0, solver=None):
        assert As.shape[0] == bs.shape[0]
        assert γ >= 0
        assert ε >= 0

        self.no = As.shape[0]  # number of observations

        self.As = As
        self.bs = bs

        self.γ = γ
        self.ε = ε

        self.solver = solver

    def solve(self, bodies, must_realize=True, **kwargs):
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

        self.problem = cp.Problem(objective, constraints)
        solve_kwargs = {"solver": self.solver, **kwargs}
        self.problem.solve(**solve_kwargs)
        assert (
            self.problem.status == "optimal"
        ), f"Optimization failed with status {problem.status}"
        return [InertialParameters.from_vector(θ.value) for θ in θs]
