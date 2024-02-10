import numpy as np
import cvxpy as cp
from spatialmath.base import rotx, roty, rotz

import inertial_params as ip


def test_ellipsoid_sphere():
    ell = ip.Ellipsoid.sphere(radius=0.5)
    assert ell.contains([0.4, 0, 0])
    assert not ell.contains([0.6, 0, 0])
    assert ell.rank == 3
    assert not ell.degenerate()

    ell2 = ip.Ellipsoid.from_Ab(A=ell.A, b=ell.b)
    assert np.allclose(ell.Einv, ell2.Einv)
    assert np.allclose(ell.center, ell2.center)

    ell3 = ip.Ellipsoid.from_Q(Q=ell.Q)
    assert np.allclose(ell.Einv, ell3.Einv)
    assert np.allclose(ell.center, ell3.center)


def test_ellipsoid_hypersphere():
    dim = 4
    ell = ip.Ellipsoid.sphere(radius=0.5, center=np.zeros(dim))
    assert ell.contains([0.4, 0, 0, 0])
    assert not ell.contains([0.6, 0, 0, 0])
    assert ell.rank == 4
    assert not ell.degenerate()

    ell2 = ip.Ellipsoid.from_Ab(A=ell.A, b=ell.b)
    assert np.allclose(ell.Einv, ell2.Einv)
    assert np.allclose(ell.center, ell2.center)

    ell3 = ip.Ellipsoid.from_Q(Q=ell.Q)
    assert np.allclose(ell.Einv, ell3.Einv)
    assert np.allclose(ell.center, ell3.center)


def test_ellipsoid_degenerate_contains():
    ell_0 = ip.Ellipsoid(half_extents=[1, 1, 0])
    ell_inf = ip.Ellipsoid(half_extents=[1, 1, np.inf])

    assert ell_0.contains([0.5, 0.5, 0])
    assert np.all(ell_0.contains([[0.5, 0.5, 0], [-0.1, 0.5, 0]]))
    assert not np.any(ell_0.contains([[0.5, 0.5, -0.1], [0.5, 0.5, 0.1]]))

    assert ell_inf.contains([0.5, 0.5, 0])
    assert np.all(ell_inf.contains([[0.5, 0.5, 0], [0.5, 0.5, 10]]))


def test_cube_bounding_ellipsoid():
    h = 0.5
    half_lengths = h * np.ones(3)
    ell = ip.Box(half_lengths).mbe()
    elld = ip.cube_bounding_ellipsoid(h)
    assert ell.is_same(elld)


def test_cube_bounding_ellipsoid_translated():
    h = 0.5
    offset = np.array([1, 1, 0])

    half_lengths = h * np.ones(3)
    points = ip.Box(half_lengths).vertices
    points += offset

    ell = ip.mbe_of_points(points)
    elld = ip.cube_bounding_ellipsoid(h).transform(translation=offset)
    assert ell.is_same(elld)


def test_cube_bounding_ellipsoid_rotated():
    h = 0.5
    C = rotx(np.pi / 2) @ roty(np.pi / 4)

    half_lengths = h * np.ones(3)
    points = ip.Box(half_lengths).vertices
    points = (C @ points.T).T

    ell = ip.mbe_of_points(points)
    elld = ip.cube_bounding_ellipsoid(h).transform(rotation=C)
    assert ell.is_same(elld)


def test_bounding_ellipoid_4d():
    np.random.seed(0)

    dim = 4
    points = np.random.random((20, dim))
    ell = ip.mbe_of_points(points)
    assert np.all(ell.contains(points))


def test_bounding_ellipsoid_degenerate():
    points = np.array([[0.5, 0, 0], [-0.5, 0, 0]])
    ell = ip.mbe_of_points(points)
    assert ell.rank == 1
    assert ell.degenerate()
    assert np.all(ell.contains(points))


def test_cube_inscribed_ellipsoid():
    h = 0.5
    half_lengths = h * np.ones(3)
    vertices = ip.Box(half_lengths).vertices
    ell = ip.maximum_inscribed_ellipsoid(vertices)
    elld = ip.cube_inscribed_ellipsoid(h)
    assert ell.is_same(elld)


def test_inscribed_sphere():
    box = ip.Box(half_extents=[1, 2, 3])
    ell = box.maximum_inscribed_ellipsoid(sphere=True)
    assert np.allclose(ell.Einv, np.eye(3))


def test_inscribed_ellipsoid_4d():
    np.random.seed(0)

    dim = 4
    points = np.random.random((20, dim))
    vertices = ip.convex_hull(points)
    ell = ip.maximum_inscribed_ellipsoid(vertices)

    # check that all the vertices are on the border or outside of the
    # ellipsoid
    for x in vertices:
        assert (x - ell.center).T @ ell.Einv @ (x - ell.center) >= 1


def test_inscribed_ellipsoid_degenerate():
    vertices = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    ell = ip.maximum_inscribed_ellipsoid(vertices)
    assert ell.rank == 2

    # check that all the vertices are on the border or outside of the
    # ellipsoid
    for x in vertices:
        assert (x - ell.center).T @ ell.Einv @ (x - ell.center) >= 1


def test_ellipsoid_must_contain():
    ell = ip.Ellipsoid.sphere(radius=1)

    point = cp.Variable(3)

    objective = cp.Maximize(point[0])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 1.0)

    # with scale
    h = cp.Variable(3)
    m = cp.Variable(1)

    objective = cp.Maximize(h[0])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 1.0)

    # offset and smaller radius
    ell = ip.Ellipsoid.sphere(radius=0.5, center=[1, 1, 1])

    objective = cp.Maximize(h[0])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 1.5)


# TODO test with rotations
def test_ellipsoid_must_contain_degenerate():
    # zero half extent
    ell = ip.Ellipsoid(half_extents=[0.5, 0.5, 0])

    point = cp.Variable(3)

    objective = cp.Maximize(point[0])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 0.5)

    objective = cp.Maximize(point[2])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 0.0, rtol=0, atol=1e-7)

    # with scale
    h = cp.Variable(3)
    m = cp.Variable(1)

    objective = cp.Maximize(h[2])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 0.0, rtol=0, atol=5e-7)

    # infinite half extent
    ell = ip.Ellipsoid(half_extents=[0.5, 0.5, np.inf])

    h = cp.Variable(3)
    m = cp.Variable(1)

    objective = cp.Maximize(h[0])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 0.5)

    objective = cp.Maximize(h[2])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert problem.status == "unbounded"


def test_ellipsoid_random_points():
    np.random.seed(0)

    ell = ip.Ellipsoid(half_extents=[2, 1, 0.5], center=[1, 0, 1])

    # one point
    point = ell.random_points()
    assert point.shape == (3,)
    assert ell.contains(point)

    # multiple points
    points = ell.random_points(shape=10)
    assert points.shape == (10, 3)
    assert ell.contains(points).all()

    # grid of points
    points = ell.random_points(shape=(10, 10))
    assert points.shape == (10, 10, 3)
    assert ell.contains(points.reshape((100, 3))).all()

    # grid with one dimension 1
    points = ell.random_points(shape=(10, 1))
    assert points.shape == (10, 1, 3)
    assert ell.contains(points.reshape((10, 3))).all()

    # test that the sampling is uniform by seeing if the number of points that
    # fall in an inscribed box is proportional to its relative volume
    n = 10000
    points = ell.random_points(shape=n)
    mib = ell.mib()
    n_box = np.sum(mib.contains(points))

    # accuracy can be increased by increasing n
    assert np.isclose(n_box / n, mib.volume / ell.volume, rtol=0, atol=0.01)


def test_grid():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = ip.Ellipsoid(half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C)
    grid = ell.grid(10)
    assert ell.contains(grid).all()


def test_aabb():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = ip.Ellipsoid(half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C)
    box = ell.aabb()

    # check that the point farthest along each axis (in both directions) in the
    # ellipsoid is contained in the AABB
    vecs = np.vstack((np.eye(3), -np.eye(3)))
    p = cp.Variable(3)
    constraints = ell.must_contain(p)
    for v in vecs:
        objective = cp.Maximize(v @ p)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        assert box.contains(p.value)


def test_mib():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = ip.Ellipsoid(half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C)
    box = ell.mib()
    assert ell.contains_polyhedron(box)


def test_mbb():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = ip.Ellipsoid(half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C)
    box = ell.mbb()

    # check that the point farthest along each axis (in both directions) in the
    # ellipsoid is contained in the AABB
    vecs = np.vstack((np.eye(3), -np.eye(3)))
    p = cp.Variable(3)
    constraints = ell.must_contain(p)
    for v in vecs:
        objective = cp.Maximize(v @ p)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        assert box.contains(p.value)


def test_contains_ellipsoid():
    # TODO more testing
    ell1 = ip.Ellipsoid(half_extents=[2, 1, 1])
    ell2 = ip.Ellipsoid(half_extents=[1, 1, 1])
    ell3 = ip.Ellipsoid(half_extents=[1, 1.1, 1])
    assert ell1.contains_ellipsoid(ell2)
    assert not ell1.contains_ellipsoid(ell3)


def test_mbe_of_ellipsoids():
    ell1 = ip.Ellipsoid(half_extents=[2, 1, 1], center=[1, 1, 0])
    ell2 = ip.Ellipsoid(half_extents=[0.5, 1.5, 1], center=[-0.5, 0, 0.5])
    ell = ip.mbe_of_ellipsoids([ell1, ell2])

    assert ell.contains_ellipsoid(ell1)
    assert ell.contains_ellipsoid(ell2)
