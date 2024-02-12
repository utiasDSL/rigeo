import numpy as np

import rigeo as rg


def test_closest_points():
    # box-box
    box = rg.Box(half_extents=[0.5, 0.5, 0.5])
    box2 = rg.Box(half_extents=[0.5, 0.5, 0.5], center=[2, 2, 2])
    info = rg.closest_points(box, box2)

    p1 = 0.5 * np.ones(3)
    p2 = 1.5 * np.ones(3)
    assert np.isclose(info.dist, np.linalg.norm(p2 - p1))
    assert np.allclose(info.p1, p1)
    assert np.allclose(info.p2, p2)

    # box-sphere
    sphere = rg.Ellipsoid.sphere(radius=0.5, center=[2, 0, 0])
    info = rg.closest_points(box, sphere)

    assert np.isclose(info.dist, 1.0)
    assert np.allclose(info.p1, [0.5, 0, 0])
    assert np.allclose(info.p2, [1.5, 0, 0])

    # box-cylinder
    # shapes are overlapping
    cylinder = rg.Cylinder(length=1.0, radius=0.5, center=[0.9, 0, 0])
    info = rg.closest_points(box, cylinder)

    assert np.isclose(info.dist, 0.0)
    assert np.allclose(info.p1, info.p2)
    assert box.contains(info.p1)
    assert cylinder.contains(info.p1)

    # TODO ellipsoid-ellipsoid
