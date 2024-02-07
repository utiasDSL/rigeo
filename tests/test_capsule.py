import numpy as np
import cvxpy as cp

import inertial_params as ip


def test_must_contain():
    cap = ip.Cylinder(length=2, radius=1).capsule()

    point = cp.Variable(3)

    # longitudinal direction
    objective = cp.Maximize(point[2])
    constraints = cap.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.allclose(point.value, [0, 0, 0.5 * cap.full_length])

    # transverse direction (point is not in one of the spheres)
    objective = cp.Maximize(point[0])
    constraints = cap.must_contain(point) + [point[2] == 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, cap.radius)

    # with scale
    h = cp.Variable(3)
    m = cp.Variable(1)

    objective = cp.Maximize(h[2])
    constraints = cap.must_contain(h, scale=m) + [m >= 0, m <= 2]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, cap.full_length)


def test_random_points():
    np.random.seed(0)

    cap = ip.Cylinder(length=2, radius=1).capsule()

    # one point
    point = cap.random_points()
    assert point.shape == (3,)
    assert cap.contains(point)

    # multiple points
    points = cap.random_points(shape=10)
    assert points.shape == (10, 3)
    assert cap.contains(points).all()
