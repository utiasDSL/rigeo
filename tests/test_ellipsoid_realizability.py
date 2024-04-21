import numpy as np
import cvxpy as cp

import rigeo as rg


# TODO need some tests with rotated cylinders


def test_disk_can_realize():
    disk = rg.Ellipsoid(half_extents=[0, 1, 1])

    points = np.array([[0, 0, 0.5], [0, 0, -0.5], [0, 0.5, 0], [0, -0.5, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert disk.can_realize(params)

    points = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert disk.can_realize(params)

    points = np.array([[0, 0, 1.1], [0, 0, -1], [0, 1, 0], [0, -1, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not disk.can_realize(params)

    points = np.array([[0.1, 0, 0.5], [0, 0, -0.5], [0, 0.5, 0], [0, -0.5, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not disk.can_realize(params)


def test_disk_can_realize_translated():
    disk = rg.Ellipsoid(half_extents=[0, 1, 1], center=[1, 0, 0])

    points = np.array([[1, 0, 0.5], [1, 0, -0.5], [1, 0.5, 0], [1, -0.5, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert disk.can_realize(params)

    points = np.array([[0, 0, 0.5], [0, 0, -0.5], [0, 0.5, 0], [0, -0.5, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not disk.can_realize(params)


def test_disk_must_realize():
    disk = rg.Ellipsoid(half_extents=[0, 1, 1])

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[1, 1])

    # need a mass constraint to bound the problem
    constraints = disk.must_realize(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 1.0)

    # now try offset from the origin
    disk = rg.Ellipsoid(half_extents=[0, 1, 1], center=[1, 0, 0])

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[1, 1])

    # need a mass constraint to bound the problem
    constraints = disk.must_realize(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 1.0)
