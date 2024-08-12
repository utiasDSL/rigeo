import numpy as np
import cvxpy as cp

import rigeo as rg


def test_must_contain():
    cap = rg.Cylinder(length=2, radius=1).capsule()

    point = cp.Variable(3)

    # longitudinal direction
    objective = cp.Maximize(point[2])
    constraints = cap.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.allclose(point.value, [0, 0, 0.5 * cap.full_length])

    # transverse direction (point is not in one of the spheres)
    objective = cp.Maximize(point[0])
    constraints = cap.must_contain(point) + [point[2] == 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, cap.radius)

    # with scale
    h = cp.Variable(3)
    m = cp.Variable(1)

    objective = cp.Maximize(h[2])
    constraints = cap.must_contain(h, scale=m) + [m >= 0, m <= 2]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, cap.full_length)


def test_random_points():
    rng = np.random.default_rng(0)

    cap = rg.Cylinder(length=2, radius=1).capsule()

    # one point
    point = cap.random_points(rng=rng)
    assert point.shape == (3,)
    assert cap.contains(point)

    # multiple points
    points = cap.random_points(shape=10, rng=rng)
    assert points.shape == (10, 3)
    assert cap.contains(points).all()

    # grid of points
    points = cap.random_points(shape=(10, 10), rng=rng)
    assert points.shape == (10, 10, 3)
    assert cap.contains(points.reshape((100, 3))).all()

    # grid with one dimension 1
    points = cap.random_points(shape=(10, 1), rng=rng)
    assert points.shape == (10, 1, 3)
    assert cap.contains(points.reshape((10, 3))).all()

    # capsule which does not contain the origin
    # there was a bug where the underlying rejection sampling algorithm was
    # returning all zeros
    cap = rg.Cylinder(length=2, radius=1, center=[5, 5, 5]).capsule()
    assert not cap.contains([0, 0, 0])
    points = cap.random_points(shape=10, rng=rng)
    assert points.shape == (10, 3)
    assert cap.contains(points).all()
