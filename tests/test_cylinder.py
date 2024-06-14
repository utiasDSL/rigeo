import numpy as np
import cvxpy as cp
from spatialmath.base import rotx, roty, rotz

import rigeo as rg


def test_contains():
    cylinder = rg.Cylinder(length=1, radius=0.5)

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


def test_must_contain():
    cylinder = rg.Cylinder(length=1, radius=0.5)
    point = cp.Variable(3)

    objective = cp.Maximize(point[0])
    constraints = cylinder.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5)


def test_aabb():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    cyl = rg.Cylinder(length=2, radius=0.5, center=[1, 0, 1], rotation=C)
    box = cyl.aabb()

    # check that the point farther along each axis (in both directions) in the
    # cylinder is contained in the AABB
    vecs = np.vstack((np.eye(3), -np.eye(3)))
    p = cp.Variable(3)
    constraints = cyl.must_contain(p)
    for v in vecs:
        objective = cp.Maximize(v @ p)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        assert box.contains(p.value, tol=1e-7)


def test_random_points():
    np.random.seed(0)

    cyl = rg.Cylinder(length=2, radius=0.5, center=[1, 0, 1])

    # one point
    point = cyl.random_points()
    assert point.shape == (3,)
    assert cyl.contains(point)

    # multiple points
    points = cyl.random_points(shape=10)
    assert points.shape == (10, 3)
    assert cyl.contains(points).all()

    # grid of points
    points = cyl.random_points(shape=(10, 10))
    assert points.shape == (10, 10, 3)
    assert cyl.contains(points.reshape((100, 3))).all()

    # grid with one dimension 1
    points = cyl.random_points(shape=(10, 1))
    assert points.shape == (10, 1, 3)
    assert cyl.contains(points.reshape((10, 3))).all()

    # test that the sampling is uniform by seeing if the number of points that
    # fall in an inscribed box is proportional to its relative volume
    n = 10000
    points = cyl.random_points(shape=n)
    mib = cyl.mib()
    n_box = np.sum(mib.contains(points))

    # accuracy can be increased by increasing n
    assert np.isclose(n_box / n, mib.volume / cyl.volume, rtol=0, atol=0.01)


def test_maximum_inscribed_box():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    cyl = rg.Cylinder(length=2, radius=0.5, center=[1, 0, 1], rotation=C)
    mib = cyl.mib()
    assert np.allclose(mib.center, cyl.center)
    assert cyl.contains_polyhedron(mib)

    # vertices should all be at the extreme ends of the cylinder
    h = (mib.vertices - mib.center) @ cyl.longitudinal_axis
    assert np.allclose(np.abs(h), cyl.length / 2)


def test_mbe():
    np.random.seed(0)

    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    cyl = rg.Cylinder(length=2, radius=0.5, center=[1, 0, 1], rotation=C)
    mib = cyl.mib()
    ell = cyl.mbe()
    assert np.allclose(ell.center, cyl.center)

    # any bounding shape should contain an inscribed shape
    assert ell.contains_polyhedron(mib)

    # should also contain any point in the shape
    points = cyl.random_points(1000)
    assert ell.contains(points).all()


def test_minimum_bounding_box():
    np.random.seed(0)

    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    cyl = rg.Cylinder(length=2, radius=0.5, center=[1, 0, 1], rotation=C)
    mib = cyl.mib()
    mbb = cyl.mbb()
    assert np.allclose(mbb.center, cyl.center)

    # any bounding shape should contain an inscribed shape
    assert mbb.contains_polyhedron(mib)

    # should also contain any point in the shape
    points = cyl.random_points(1000)
    assert mbb.contains(points).all()
