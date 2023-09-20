import numpy as np
import inertial_params as ip


def test_axis_aligned_box_from_two_vertices():
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    box = ip.AxisAlignedBox.from_two_vertices(v1, v2)
    assert np.allclose(box.center, [0.5, 0.5, 0.5])
    assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 1])
    box = ip.AxisAlignedBox.from_two_vertices(v1, v2)
    assert np.allclose(box.center, [0.5, 0.5, 0.5])
    assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])


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
