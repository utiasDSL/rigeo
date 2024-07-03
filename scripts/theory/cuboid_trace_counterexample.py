"""A counter-example to the proposed trace inequality conditions for density
realizability on cuboids, which shows that they are *not* sufficient for
implying the proposed vertex conditions."""
import numpy as np
import cvxpy as cp
import rigeo as rg

H = np.array([[1, -0.45, 0], [-0.45, 1, 0], [0, 0, 1]])
h = np.array([0.5, 0.5, 0])
poly = rg.Box(half_extents=[1, 1, 1])

# J is positive definite and satisfies the proposed trace constraints
J = np.zeros((4, 4))
J[:3, :3] = H
J[:3, 3] = h
J[3, :3] = h
J[3, 3] = 1
assert np.all(np.linalg.eigvals(J) > 0)

# but we cannot find a set of masses at the vertices to achieve the vertex
# realizability conditions
μ = cp.Variable(8)
objective = cp.Minimize(0)
constraints = [
    H << cp.sum([μi * np.outer(vi, vi) for μi, vi in zip(μ, poly.vertices)]),
    h == cp.sum([μi * vi for μi, vi in zip(μ, poly.vertices)]),
    cp.sum(μ) == 1,
    μ >= 0,
]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK)
assert problem.status == "infeasible"
