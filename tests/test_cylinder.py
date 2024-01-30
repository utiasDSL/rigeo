import numpy as np
import cvxpy as cp

import inertial_params as ip


def test_cylinder_contains():
    cylinder = ip.Cylinder(length=1, radius=0.5)

    # a single point
    assert cylinder.contains([0, 0, 0])

    # multiple points
    points = np.array([[0, 0, 0.5], [0, 0, -0.5], [0.5, 0, 0.5], [0.5, 0, -0.5]])
    assert np.all(cylinder.contains(points))

    # multiple points not inside the cylinder
    points = np.array([[0, 0, 0.6], [0, 0, -0.6], [0.6, 0, 0], [0, -0.6, 0]])
    assert not np.any(cylinder.contains(points))

    # a mix of points inside and outside
    points = np.array([[0, 0, 0.5], [0, 0, -0.6]])
    contained = cylinder.contains(points)
    assert contained[0] and not contained[1]


def test_cylinder_must_contain():
    cylinder = ip.Cylinder(length=1, radius=0.5)
    point = cp.Variable(3)

    objective = cp.Maximize(point[0])
    constraints = cylinder.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 0.5)
