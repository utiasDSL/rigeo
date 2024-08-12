import numpy as np
import cvxpy as cp

import rigeo as rg


def test_can_realize():
    rng = np.random.default_rng(0)

    N = 100  # number of trials
    n = 10  # number of point masses per trial

    # random convex polyhedron
    points = rng.random((10, 3)) - 0.5
    poly = rg.ConvexPolyhedron.from_vertices(points, prune=True)

    for i in range(N):
        points = poly.random_points(n, rng=rng)
        masses = rng.random(n)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
        assert poly.can_realize(params, solver=cp.MOSEK)


def test_tetrahedron_can_realize():
    rng = np.random.default_rng(0)

    N = 1000  # number of trials
    n = 10  # number of point masses per trial

    poly = rg.Simplex(np.ones(3))

    # test a bunch of feasible params
    for i in range(N):
        points = poly.random_points(n, rng=rng)
        masses = rng.random(n)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
        assert poly.can_realize(params, solver=cp.MOSEK)

    # test some infeasible cases too
    masses = [0.5, 0.5]

    points = np.array([[0, 0, 0], [1.1, 0, 0]])
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not poly.can_realize(params, solver=cp.MOSEK)

    points = np.array([[0, 0, 0], [0, 1.1, 0]])
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not poly.can_realize(params, solver=cp.MOSEK)

    points = np.array([[0, 0, 0], [0, 0, 1.1]])
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not poly.can_realize(params, solver=cp.MOSEK)


def test_tetrahedron_must_realize_pim():
    poly = rg.Simplex(np.ones(3)).transform(translation=[-0.5, 0, 0])

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[0, 0])

    # need a mass constraint to bound the problem
    constraints = poly.must_realize(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    # H_xx is maximized by splitting mass between the face at x = -0.5 and the
    # vertex at x = 0.5
    assert np.isclose(objective.value, 0.5**2)


def test_tetrahedron_must_realize_vec():
    poly = rg.Simplex(np.ones(3)).transform(translation=[-0.5, 0, 0])

    θ = cp.Variable(10)
    m = θ[0]

    J = rg.pim_must_equal_vec(θ)
    objective = cp.Maximize(J[0, 0])

    # need a mass constraint to bound the problem
    constraints = poly.must_realize(θ) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    assert np.isclose(objective.value, 0.5**2)
