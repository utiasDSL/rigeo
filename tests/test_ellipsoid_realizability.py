import numpy as np
import cvxpy as cp
from spatialmath.base import roty

import rigeo as rg


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

    # different sized disks
    disk = rg.Ellipsoid(half_extents=[0, 0.5, 0.5])
    points = np.array([[0, 0, 0.5], [0, 0, -0.5], [0, 0.5, 0], [0, -0.5, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert disk.can_realize(params)

    disk = rg.Ellipsoid(half_extents=[0, 2, 2])
    points = np.array([[0, 0, 2], [0, 0, -2], [0, 2, 0], [0, -2, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert disk.can_realize(params)


def test_line_segment_can_realize():
    segment = rg.Ellipsoid(half_extents=[0, 0, 1])

    points = np.array([[0, 0, 1], [0, 0, -1]])
    masses = np.ones(2)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert segment.can_realize(params)

    points = np.array([[0, 0, 1.1], [0, 0, -1]])
    masses = np.ones(2)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not segment.can_realize(params)


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


def test_disk_can_realize_rotated():
    disk = rg.Ellipsoid(half_extents=[0, 1, 1], rotation=roty(np.pi / 4))

    points = np.array([[0.5, 0, 0.5], [-0.5, 0, -0.5], [0, 1, 0], [0, -1, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert disk.can_realize(params)

    points = np.array([[1, 0, 1], [-1, 0, -1], [0, 1, 0], [0, -1, 0]])
    masses = np.ones(4)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not disk.can_realize(params)


def test_disk_moment_constraints():
    disk = rg.Ellipsoid(half_extents=[0, 1, 1])

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[1, 1])

    # need a mass constraint to bound the problem
    constraints = disk.moment_constraints(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 1.0)

    # now try offset from the origin
    disk = rg.Ellipsoid(half_extents=[0, 1, 1], center=[1, 0, 0])

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[1, 1])

    # need a mass constraint to bound the problem
    constraints = disk.moment_constraints(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 1.0)
