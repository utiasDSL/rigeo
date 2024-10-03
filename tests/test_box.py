import numpy as np
from spatialmath.base import rotx, roty, rotz
import rigeo as rg


def test_from_two_vertices():
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    box = rg.Box.from_two_vertices(v1, v2)
    assert np.allclose(box.center, [0.5, 0.5, 0.5])
    assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])
    assert np.isclose(box.volume, 1)

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 1])
    box = rg.Box.from_two_vertices(v1, v2)
    assert np.allclose(box.center, [0.5, 0.5, 0.5])
    assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])
    assert np.isclose(box.volume, 1)


def test_vertices():
    h = np.array([0.5, 0.5, 0.5])
    box = rg.Box(h)
    assert np.allclose(np.max(box.vertices, axis=0), h)
    assert np.allclose(np.min(box.vertices, axis=0), -h)

    c = np.array([1, 0, 0])
    box = rg.Box(h, c)
    assert np.allclose(np.max(box.vertices, axis=0), h + c)
    assert np.allclose(np.min(box.vertices, axis=0), -h + c)


def test_contains():
    h = np.array([0.5, 0.5, 0.5])
    box = rg.Box(h)
    assert box.contains(box.vertices).all()

    # random points
    rng = np.random.default_rng(0)
    points = rng.random((10, 3)) + [0.51, 0, 0]
    res = box.contains(points)
    assert res.shape == (10,)
    assert not res.any()

    # offset the box
    box = rg.Box(h, center=[1.5, 0, 0])
    assert box.contains(box.vertices).all()
    assert box.contains(box.center)


def test_from_points_to_bound():
    rng = np.random.default_rng(0)
    points = rng.random((100, 3))
    box = rg.Box.from_points_to_bound(points)
    assert box.contains(points).all()

    # at least two points must be on the surface (defining opposing vertices)
    assert np.sum(box.on_surface(points)) >= 2


def test_grid():
    box = rg.Box(half_extents=[1, 1, 1])
    n = 20
    points = box.grid(n)
    assert points.shape == (n**3, 3)
    assert box.contains(points).all()

    # translated
    box = rg.Box(half_extents=[1, 1, 1], center=[2, 2, 2])
    points = box.grid(n)
    assert box.contains(points).all()

    # different number of points along each dimensions
    n = [10, 6, 3]
    box = rg.Box(half_extents=[1, 1, 1])
    points = box.grid(n)
    assert points.shape == (np.prod(n), 3)
    assert box.contains(points).all()


def test_rotation():
    C = rotz(np.pi / 8)
    half_extents = np.ones(3)
    box1 = rg.Box(half_extents=half_extents)
    box2 = rg.Box(half_extents=half_extents, rotation=C)
    box3 = box1.transform(rotation=C)

    assert np.allclose(box2.half_extents, box1.half_extents)
    assert np.allclose(box2.center, box1.center)

    points = (C @ box1.vertices.T).T
    assert box2.contains(points).all()

    # box2 and box3 should be the same
    assert np.allclose(box3.half_extents, box1.half_extents)
    assert np.allclose(box3.center, box1.center)
    assert np.allclose(box3.vertices, box2.vertices)


def test_random_points():
    rng = np.random.default_rng(0)

    box = rg.Box.cube(half_extent=0.5)
    points = box.random_points(10, rng=rng)
    assert points.shape == (10, 3)
    assert box.contains(points).all()

    point = box.random_points(rng=rng)
    assert point.shape == (3,)
    assert box.contains(point)

    box = rg.Box(half_extents=[0.5, 1, 2])
    points = box.random_points(10, rng=rng)
    assert box.contains(points).all()

    # translated from origin
    box = rg.Box(half_extents=[0.5, 1, 2], center=[10, 2, 3])
    points = box.random_points(10, rng=rng)
    assert box.contains(points).all()

    # translated and rotated
    C = rotz(np.pi / 4)
    box = rg.Box(half_extents=[0.5, 1, 2], center=[10, 2, 5], rotation=C)
    points = box.random_points(10, rng=rng)
    assert box.contains(points).all()

    # multi-dimensional set of points
    points = box.random_points((10, 5), rng=rng)
    assert points.shape == (10, 5, 3)
    assert box.contains(points.reshape((50, 3))).all()


def test_on_surface():
    box = rg.Box(half_extents=[1, 2, 3])

    assert box.on_surface([1, 0, 0])

    points = np.array([[1, 0, 0], [-1, 0, 0], [1, 2, 3], [-1, -2, -3]])
    assert box.on_surface(points).all()

    # not contained in the box
    assert not box.on_surface([1, 3, 0])

    # inside but not on the boundary
    assert not box.on_surface([0.9, 0, 0])

    # test with rotation and translation
    C = rotz(np.pi / 4)
    r = np.array([4, 4, 4])
    box = rg.Box(half_extents=[1, 2, 3], center=r, rotation=C)

    points = (C @ points.T).T + r
    assert box.on_surface(points).all()


def test_random_points_on_surface():
    rng = np.random.default_rng(0)
    box = rg.Box(half_extents=[1, 0.75, 0.5])
    points = box.random_points_on_surface(10, rng=rng)
    assert box.on_surface(points).all()


def test_hollow_density_params():
    box = rg.Box(half_extents=[1, 0.75, 0.5])
    ell = rg.Ellipsoid(half_extents=[1, 0.75, 0.5])

    # the inertia for hollow boxes and ellipsoids differs by only a constant
    # factor
    p1 = box.hollow_density_params(mass=1.0)
    p2 = ell.hollow_density_params(mass=1.0)
    assert np.allclose(p1.H * 3 / 5, p2.H)

    p3 = box.hollow_density_params(mass=2.0)
    assert np.allclose(p3.J, 2 * p1.J)
