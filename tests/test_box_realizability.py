import numpy as np
import cvxpy as cp

import rigeo as rg


def test_cube_at_origin_can_realize():
    rng = np.random.default_rng(0)

    N = 100  # number of trials
    n = 10  # number of point masses per trial

    box = rg.Box(half_extents=[0.5, 0.5, 0.5])

    for i in range(N):
        points = box.random_points(n, rng=rng)
        masses = rng.random(n)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
        assert box.can_realize(params)

    points = np.array([-box.half_extents, box.half_extents])
    masses = [0.5, 0.5]
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert box.can_realize(params, eps=-1e-12)

    masses = np.ones(8)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=box.vertices)
    assert box.can_realize(params)

    # infeasible case
    points = np.array([-box.half_extents, 1.1 * box.half_extents])
    masses = [0.5, 0.5]
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not box.can_realize(params)


def test_box_offset_from_origin_can_realize():
    rng = np.random.default_rng(0)

    N = 100  # number of trials
    n = 10  # number of point masses per trial

    box = rg.Box(half_extents=[1.0, 0.5, 0.1], center=[1, 1, 0])

    for i in range(N):
        points = box.random_points(n, rng=rng)
        masses = rng.random(n)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
        assert box.can_realize(params)


def test_cube_must_realize_J():
    box = rg.Box(half_extents=[0.5, 0.5, 0.5])

    # convert to a general convex polyhedron
    poly = rg.ConvexPolyhedron.from_vertices(box.vertices)

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[0, 0])

    # need a mass constraint to bound the problem
    constraints = box.must_realize(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5**2)

    # objective should be the same if we use the general convex polyhedron
    # formulation
    constraints = poly.must_realize(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5**2)


def test_cube_must_realize_vec():
    box = rg.Box(half_extents=[0.5, 0.5, 0.5])
    poly = rg.ConvexPolyhedron.from_vertices(box.vertices)

    θ = cp.Variable(10)
    m = θ[0]

    objective = cp.Maximize(θ[4])

    # need a mass constraint to bound the problem
    constraints = box.must_realize(θ) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5)

    # objective should be the same if we use the general convex polyhedron
    # formulation
    constraints = poly.must_realize(θ) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5)
