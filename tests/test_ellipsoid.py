import numpy as np
import cvxpy as cp
from spatialmath.base import rotx, roty, rotz

import rigeo as rg


def test_ellipsoid_sphere():
    ell = rg.Ellipsoid.sphere(radius=0.5)
    assert ell.contains([0.4, 0, 0])
    assert not ell.contains([0.6, 0, 0])
    assert ell.rank == 3
    assert not ell.is_degenerate()

    ell2 = rg.Ellipsoid.from_affine(A=ell.A, b=ell.b)
    assert np.allclose(ell.S, ell2.S)
    assert np.allclose(ell.center, ell2.center)


def test_ellipsoid_hypersphere():
    dim = 4
    ell = rg.Ellipsoid.sphere(radius=0.5, center=np.zeros(dim))
    assert ell.contains([0.4, 0, 0, 0])
    assert not ell.contains([0.6, 0, 0, 0])
    assert ell.rank == 4
    assert not ell.is_degenerate()

    ell2 = rg.Ellipsoid.from_affine(A=ell.A, b=ell.b)
    assert np.allclose(ell.S, ell2.S)
    assert np.allclose(ell.center, ell2.center)


def test_ellipsoid_degenerate_contains():
    ell = rg.Ellipsoid(half_extents=[1, 1, 0])

    assert ell.contains([0.5, 0.5, 0])
    assert np.all(ell.contains([[0.5, 0.5, 0], [-0.1, 0.5, 0]]))
    assert not np.any(ell.contains([[0.5, 0.5, -0.1], [0.5, 0.5, 0.1]]))


def test_box_bounding_ellipsoid():
    # MBE for box and for general set of points are implemented differently, so
    # it is worth testing them against each other
    half_extents = [0.5, 1, 2]
    offset = np.array([1, 1, 0])
    C = rotx(np.pi / 2) @ roty(np.pi / 4)

    # translated
    box = rg.Box(half_extents, center=offset)
    ell = rg.mbe_of_points(box.vertices)
    assert ell.is_same(box.mbe(), tol=1e-4)

    # rotated
    box = rg.Box(half_extents, rotation=C)
    ell = rg.mbe_of_points(box.vertices)
    assert ell.is_same(box.mbe(), tol=1e-4)

    # translated + rotated
    box = rg.Box(half_extents, rotation=C, center=offset)
    ell = rg.mbe_of_points(box.vertices)
    assert ell.is_same(box.mbe(), tol=1e-4)


def test_bounding_ellipsoid_4d():
    rng = np.random.default_rng(0)

    dim = 4
    points = rng.random((20, dim))
    ell = rg.mbe_of_points(points)

    assert np.all(ell.contains(points))


def test_bounding_ellipsoid_degenerate():
    # 1D line segment
    points = np.array([[1.0, 0.5, 0], [0, 0.5, 0]])
    ell = rg.mbe_of_points(points)
    assert ell.rank == 1
    assert ell.is_degenerate()
    assert np.all(ell.contains(points))

    assert np.allclose(ell.center, [0.5, 0.5, 0])
    assert not np.any(
        ell.contains([[1.1, 0.5, 0], [-0.1, 0.5, 0], [0, 0.6, 0], [0, 0, 0.1]])
    )

    # 2D plane
    points = np.array([[1.0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0]])
    ell = rg.mbe_of_points(points)
    assert ell.rank == 2
    assert ell.is_degenerate()
    assert np.all(ell.contains(points))

    assert np.allclose(ell.center, [0.5, 0.5, 0.5])
    assert not np.any(ell.contains([[0.45, 0.5, 0.55], [0.55, 0.5, 0.45]]))


def test_box_inscribed_ellipsoid():
    # MIE for box and for general set of points are implemented differently, so
    # it is worth testing them against each other
    half_extents = [0.5, 1, 2]
    offset = np.array([1, 1, 0])
    C = rotx(np.pi / 2) @ roty(np.pi / 4)

    # translated
    box = rg.Box(half_extents, center=offset)
    ell = box.as_poly().mie()
    assert ell.is_same(box.mie(), tol=1e-4)

    # rotated
    box = rg.Box(half_extents, rotation=C)
    ell = box.as_poly().mie()
    assert ell.is_same(box.mie(), tol=1e-4)

    # translated + rotated
    box = rg.Box(half_extents, rotation=C, center=offset)
    ell = box.as_poly().mie()
    assert ell.is_same(box.mie(), tol=1e-4)


def test_inscribed_sphere():
    box = rg.Box(half_extents=[1, 2, 3])
    ell = box.mie(sphere=True)
    assert np.allclose(ell.S, np.eye(3))


def test_inscribed_ellipsoid_4d():
    rng = np.random.default_rng(0)

    dim = 4
    points = rng.random((20, dim))
    poly = rg.ConvexPolyhedron.from_vertices(points, prune=True)
    ell = poly.mie()

    # check that all the vertices are on the border or outside of the
    # ellipsoid
    for x in poly.vertices:
        assert (x - ell.center).T @ ell.S @ (x - ell.center) >= 1


def test_inscribed_ellipsoid_degenerate():
    vertices = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    poly = rg.ConvexPolyhedron.from_vertices(vertices)
    ell = poly.mie()
    assert ell.rank == 2

    # check that all the vertices are on the border or outside of the
    # ellipsoid
    # TODO this does not work in the case of a degenerate ellipsoid!
    # for x in vertices:
    #     assert (x - ell.center).T @ ell.S @ (x - ell.center) >= 1


def test_must_contain():
    ell = rg.Ellipsoid.sphere(radius=1)

    point = cp.Variable(3)

    objective = cp.Maximize(point[0])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 1.0)

    # with scale
    h = cp.Variable(3)
    m = cp.Variable(1)

    objective = cp.Maximize(h[0])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 1.0)

    # offset and smaller radius
    ell = rg.Ellipsoid.sphere(radius=0.5, center=[1, 1, 1])

    objective = cp.Maximize(h[0])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 1.5)

    # make mass h and mass general cvxpy expressions
    # recall that there was a bug where the schur complement function would
    # fail in this case
    J = cp.Variable((4, 4), PSD=True)
    h = J[:3, 3]
    m = J[3, 3]

    objective = cp.Maximize(h[0])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 1.5)


# TODO test with rotations
def test_must_contain_degenerate():
    # zero half extent
    ell = rg.Ellipsoid(half_extents=[0.5, 0.5, 0])

    point = cp.Variable(3)

    objective = cp.Maximize(point[0])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5)

    objective = cp.Maximize(point[2])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.0, rtol=0, atol=1e-7)

    # with scale
    h = cp.Variable(3)
    m = cp.Variable(1)

    objective = cp.Maximize(h[2])
    constraints = ell.must_contain(h, scale=m) + [m >= 0, m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.0, rtol=0, atol=5e-7)


def test_must_contain_degenerate_rotated():
    # zero half extent
    C = roty(np.pi / 4)
    ell = rg.Ellipsoid(half_extents=[0.5, 0.5, 0], rotation=C)

    point = cp.Variable(3)

    # max y is unaffected
    objective = cp.Maximize(point[1])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5)

    # max x is reduced
    objective = cp.Maximize(point[0])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.25 * np.sqrt(2))

    # max z is not zero anymore: it is same as x
    objective = cp.Maximize(point[2])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.25 * np.sqrt(2))


def test_random_points():
    rng = np.random.default_rng(0)

    ell = rg.Ellipsoid(half_extents=[2, 1, 0.5], center=[1, 0, 1])

    # one point
    point = ell.random_points(rng=rng)
    assert point.shape == (3,)
    assert ell.contains(point)

    # multiple points
    points = ell.random_points(shape=10, rng=rng)
    assert points.shape == (10, 3)
    assert ell.contains(points).all()

    # grid of points
    points = ell.random_points(shape=(10, 10), rng=rng)
    assert points.shape == (10, 10, 3)
    assert ell.contains(points.reshape((100, 3))).all()

    # grid with one dimension 1
    points = ell.random_points(shape=(10, 1), rng=rng)
    assert points.shape == (10, 1, 3)
    assert ell.contains(points.reshape((10, 3))).all()

    # test that the sampling is uniform by seeing if the number of points that
    # fall in an inscribed box is proportional to its relative volume
    n = 10000
    points = ell.random_points(shape=n, rng=rng)
    mib = ell.mib()
    n_box = np.sum(mib.contains(points))

    # accuracy can be increased by increasing n
    assert np.isclose(n_box / n, mib.volume / ell.volume, rtol=0, atol=0.01)


def test_random_points_on_surface():
    rng = np.random.default_rng(0)

    ell = rg.Ellipsoid(half_extents=[2, 1, 0.5], center=[1, 0, 1])

    # one point
    point = ell.random_points_on_surface(rng=rng)
    assert point.shape == (3,)
    assert ell.on_surface(point)

    # multiple points
    points = ell.random_points_on_surface(shape=10, rng=rng)
    assert points.shape == (10, 3)
    assert ell.on_surface(points).all()

    # grid of points
    points = ell.random_points_on_surface(shape=(10, 10), rng=rng)
    assert points.shape == (10, 10, 3)
    points = points.reshape((100, 3))
    assert ell.on_surface(points).all()

    # 2D ellipsoid
    ell = rg.Ellipsoid(half_extents=[2, 1], center=[1, 0])
    points = ell.random_points_on_surface(shape=10, rng=rng)
    assert points.shape == (10, 2)
    assert ell.on_surface(points).all()

    # 4D ellipsoid
    ell = rg.Ellipsoid(half_extents=[2, 1, 0.5, 0.5], center=[1, 0, 1, 0])
    points = ell.random_points_on_surface(shape=10, rng=rng)
    assert points.shape == (10, 4)
    assert ell.on_surface(points).all()

    # degenerate ellipsoid
    # TODO: degenerate not supported yet
    # ell = rg.Ellipsoid(half_extents=[1, 1, 0])
    # points = ell.random_points_on_surface(shape=10, rng=rng)
    # assert ell.on_surface(points).all()


def test_on_surface():
    ell = rg.Ellipsoid(half_extents=[1, 2, 3])

    assert ell.on_surface([1, 0, 0])

    points = np.array([[1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, 0, -3]])
    assert ell.on_surface(points).all()

    # not contained in the ellipsoid
    assert not ell.on_surface([1, 2, 0])

    # degenerate ellipsoid
    # now any point inside the ellipsoid is also on the surface
    rng = np.random.default_rng(0)
    ell = rg.Ellipsoid(half_extents=[1, 2, 0])
    points = ell.random_points(1000, rng=rng)
    assert ell.on_surface(points).all()

    # in contrast, the equivalent lower-dimensional ellipsoid does require the
    # points on the (lower-dimensional) surface
    ell = rg.Ellipsoid(half_extents=[1, 2])
    points = points[:, :2]
    assert not ell.on_surface(points).all()

    # test with translation
    ell = rg.Ellipsoid(half_extents=[1, 2, 3], center=[4, 4, 4])
    points = ell.center + np.array([[1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, 0, -3]])
    res = ell.on_surface(points)
    assert ell.on_surface(points).all()


def test_grid():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = rg.Ellipsoid(
        half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C
    )
    grid = ell.grid(10)
    assert ell.contains(grid).all()


def test_aabb():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = rg.Ellipsoid(
        half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C
    )
    box = ell.aabb()

    # check that the point farthest along each axis (in both directions) in the
    # ellipsoid is contained in the AABB
    vecs = np.vstack((np.eye(3), -np.eye(3)))
    p = cp.Variable(3)
    constraints = ell.must_contain(p)
    for v in vecs:
        objective = cp.Maximize(v @ p)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        assert box.contains(p.value)


def test_mib():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = rg.Ellipsoid(
        half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C
    )
    box = ell.mib()
    assert ell.contains_polyhedron(box)


def test_mbb():
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    ell = rg.Ellipsoid(
        half_extents=[1, 0.5, 0.25], center=[1, 0, 1], rotation=C
    )
    box = ell.mbb()

    # check that the point farthest along each axis (in both directions) in the
    # ellipsoid is contained in the AABB
    vecs = np.vstack((np.eye(3), -np.eye(3)))
    p = cp.Variable(3)
    constraints = ell.must_contain(p)
    for v in vecs:
        objective = cp.Maximize(v @ p)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        assert box.contains(p.value)


def test_contains_ellipsoid():
    # TODO more testing
    ell1 = rg.Ellipsoid(half_extents=[2, 1, 1])
    ell2 = rg.Ellipsoid(half_extents=[1, 1, 1])
    ell3 = rg.Ellipsoid(half_extents=[1, 1.1, 1])
    assert ell1.contains_ellipsoid(ell2)
    assert not ell1.contains_ellipsoid(ell3)


def test_mbe_of_ellipsoids():
    ell1 = rg.Ellipsoid(half_extents=[2, 1, 1], center=[1, 1, 0])
    ell2 = rg.Ellipsoid(half_extents=[0.5, 1.5, 1], center=[-0.5, 0, 0.5])
    ell = rg.mbe_of_ellipsoids([ell1, ell2])

    assert ell.contains_ellipsoid(ell1)
    assert ell.contains_ellipsoid(ell2)


def test_hollow_density_params():
    ell = rg.Ellipsoid(half_extents=[1, 0.75, 0.5])

    # linear scaling with mass
    p1 = ell.hollow_density_params(mass=1.0)
    p2 = ell.hollow_density_params(mass=2.0)
    assert np.allclose(p2.J, 2 * p1.J)
