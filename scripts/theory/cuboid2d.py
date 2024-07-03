import numpy as np
import cvxpy as cp
import rigeo as rg


def move_mass_out_counterexample():
    """2D example that shows that moving mass outside the shape does not
    necessarily increase the variance along any direction."""
    # mass at the vertices
    V = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    μ = np.ones(4) / 4
    S = sum([m * np.outer(v, v) for m, v in zip(μ, V)])

    # mass along the axes, outside of the shape
    # this results in a strictly smaller covariance S!
    V2 = np.array([[1.1, 0], [0, 1.1], [-1.1, 0], [0, -1.1]])
    S2 = sum([m * np.outer(v, v) for m, v in zip(μ, V2)])


def cuboid_trace_counterexample():
    """A 2D counter-example to the proposed trace inequality conditions for
    density realizability on cuboids, which shows that they are *not*
    sufficient for implying the proposed vertex conditions.

    See the script cuboid_trace_counterexample.py for a 3D version.
    """
    x = 1
    y = 1
    xy = x * y
    verts = np.array([[x, y], [-x, y], [-y, x], [-x, -y]])

    # solve for a particular value that satisfies the trace inequalities
    J = cp.Variable((3, 3), PSD=True)
    H = J[:2, :2]
    h = J[:2, 2]

    objective = cp.Maximize(-H[0, 1] + h[0] + h[1])
    constraints = [cp.diag(H) <= np.ones(2), J[2, 2] == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # H = J.value[:2, :2]
    # h = J.value[:2, 2]

    # without numerical errors, these are the optimal values
    # they satisfy the trace constraints
    H = np.array([[1, -0.5], [-0.5, 1]])
    h = np.array([0.5, 0.5])

    # check if these values can be realized according to the vertex-based
    # constraints
    μ = cp.Variable(4)
    objective = cp.Minimize(0)
    constraints = [
        H << cp.sum([μi * np.outer(vi, vi) for μi, vi in zip(μ, verts)]),
        h == cp.sum([μi * vi for μi, vi in zip(μ, verts)]),
        cp.sum(μ) == 1,
        μ >= 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # they cannot be realized, which means that the trace constraint is not
    # sufficient for the vertex constraints to be satisfied
    assert problem.status == "infeasible"

    # check if there is a "bigger" H that is realizable (this would disprove
    # the conjectured sufficiency direction of my main theorem)
    J = cp.Variable((3, 3), PSD=True)
    μ = cp.Variable(4)

    objective = cp.Minimize(0)
    constraints = [
        cp.sum([μi * np.outer(vi, vi) for μi, vi in zip(μ, verts)]) >> H,
        h == cp.sum([μi * vi for μi, vi in zip(μ, verts)]),
        cp.sum(μ) == 1,
        μ >= 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # there is no such H
    assert problem.status == "infeasible"


cuboid_trace_counterexample()
