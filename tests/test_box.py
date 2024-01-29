import numpy as np
from spatialmath.base import rotx, roty, rotz
import inertial_params as ip


def test_axis_aligned_box_from_two_vertices():
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    box = ip.AxisAlignedBox.from_two_vertices(v1, v2)
    assert np.allclose(box.center, [0.5, 0.5, 0.5])
    assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])
    assert np.isclose(box.volume, 1)

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 1])
    box = ip.AxisAlignedBox.from_two_vertices(v1, v2)
    assert np.allclose(box.center, [0.5, 0.5, 0.5])
    assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])
    assert np.isclose(box.volume, 1)


def test_axis_aligned_box_vertices():
    h = np.array([0.5, 0.5, 0.5])
    box = ip.AxisAlignedBox(h)
    assert np.allclose(np.max(box.vertices, axis=0), h)
    assert np.allclose(np.min(box.vertices, axis=0), -h)

    c = np.array([1, 0, 0])
    box = ip.AxisAlignedBox(h, c)
    assert np.allclose(np.max(box.vertices, axis=0), h + c)
    assert np.allclose(np.min(box.vertices, axis=0), -h + c)


def test_axis_aligned_box_contains():
    h = np.array([0.5, 0.5, 0.5])
    box = ip.AxisAlignedBox(h)
    assert box.contains(box.vertices).all()

    points = np.random.random((10, 3)) + [0.51, 0, 0]
    res = box.contains(points)
    assert res.shape == (10,)
    assert not res.any()


def test_axis_aligned_box_random_points():
    np.random.seed(0)

    h = np.array([0.5, 0.5, 0.5])
    box = ip.AxisAlignedBox(h)

    points = box.random_points(10)
    assert box.contains(points).all()


def test_axis_aligned_box_from_points_to_bound():
    np.random.seed(0)
    points = np.random.random((100, 3))
    box = ip.AxisAlignedBox.from_points_to_bound(points)
    assert box.contains(points).all()


def test_axis_aligned_box_grid():
    box = ip.AxisAlignedBox(half_extents=(1, 1, 1))
    n = 20
    points = box.grid(n)
    assert points.shape == (n**3, 3)
    assert box.contains(points).all()


def test_box_rotation():
    C = rotz(np.pi / 8)
    half_extents = np.ones(3)
    box1 = ip.AxisAlignedBox(half_extents=half_extents)
    box2 = ip.AxisAlignedBox(half_extents=half_extents, rotation=C)
    box3 = box1.transform(rotation=C)

    assert np.allclose(box2.half_extents, box1.half_extents)
    assert np.allclose(box2.center, box1.center)

    points = (C @ box1.vertices.T).T
    assert box2.contains(points).all()

    # box2 and box3 should be the same
    assert np.allclose(box3.half_extents, box1.half_extents)
    assert np.allclose(box3.center, box1.center)
    assert np.allclose(box3.vertices, box2.vertices)

    # check that ellipsoidal intersection still holds
    ells = box2.as_ellipsoidal_intersection()
    for ell in ells:
        assert ell.contains(box2.vertices).all()
