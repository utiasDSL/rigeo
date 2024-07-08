"""Examining the cuboid special case."""
import numpy as np
import cvxpy as cp
import rigeo as rg

import IPython


def different_Hs():
    mass = 1
    box = rg.Box(half_extents=[0.5, 1, 2])

    points1 = box.vertices
    masses1 = mass * np.ones(8) / 8
    params1 = rg.InertialParameters.from_point_masses(
        masses=masses1, points=points1
    )

    # this has same mass and CoM but different H
    # (but the diagonal elements are the same)
    masses2 = mass * np.ones(4) / 4
    points2 = np.array(
        [[0.5, 1, 2], [0.5, -1, -2], [-0.5, 1, 2], [-0.5, -1, -2]]
    )
    params2 = rg.InertialParameters.from_point_masses(
        masses=masses2, points=points2
    )

    masses3 = mass * np.ones(2) / 2
    points3 = np.array([[-0.5, 1, 2], [0.5, -1, -2]])
    params3 = rg.InertialParameters.from_point_masses(
        masses=masses3, points=points3
    )

    H = np.array([[0.2, 0, 0], [0, 0.9, 0.5], [0, 0.5, 3]])

    # 0.75 masses1 + 0.25 masses2 (linear combo of 1 and 2)
    masses4 = 0.75 * masses1 + 0.25 * np.array(
        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25]
    )
    points4 = box.vertices
    params4 = rg.InertialParameters.from_point_masses(
        masses=masses4, points=points4
    )


def lp_conjecture():
    """Testing the conjecture that density realizability for cuboids can
    actually be solved with linear programming rather than semidefinite
    programming."""
    np.random.seed(0)
    mass = 1
    box = rg.Box(half_extents=[0.5, 1, 2])
    d = mass * box.half_extents**2

    n = 10

    for _ in range(1000):
        # random density realizable parameters
        points = box.random_points(n)
        masses = np.random.random(n)
        masses = mass * masses / np.sum(masses)
        params = rg.InertialParameters.from_point_masses(
            masses=masses, points=points
        )

        h = mass * box.random_points(1)
        Vs = [np.outer(v, v) for v in box.vertices]

        μ = cp.Variable(8)
        H = cp.sum([μi * V for μi, V in zip(μ, Vs)])

        # find vertex point masses that exactly realize the off-diagonal terms
        # of H, while also achieving the desired mass and CoM
        objective = cp.Minimize(0)
        constraints = [
            H[0, 1] == params.H[0, 1],
            H[0, 2] == params.H[0, 2],
            H[1, 2] == params.H[1, 2],
            params.h == cp.sum([μi * vi for μi, vi in zip(μ, box.vertices)]),
            cp.sum(μ) == params.mass,
            μ >= 0,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status != "optimal":
            print(problem.status)
            IPython.embed()

        Hopt = sum([μi * V for μi, V in zip(μ.value, Vs)])

        # diagonal is always the same
        assert np.allclose(np.diag(Hopt), d)

        # NOTE: here I tried to just solve the least squares solution to see if
        # this always worked, but it does not
        # A0 = np.vstack([[V[0, 1], V[0, 2], V[1, 2]] for V in Vs]).T
        # A = np.vstack([A0, box.vertices.T, np.ones(8)])
        # b = np.array([params.H[0, 1], params.H[0, 2], params.H[1, 2], h[0], h[1], h[2], mass])
        #
        # x = np.linalg.lstsq(A, b, rcond=None)[0]
        # Hopt2 = sum([xi * V for xi, V in zip(x, Vs)])
        # if not np.all(x >= 0):
        #     print("negative!")

        # IPython.embed()
        # return

lp_conjecture()
