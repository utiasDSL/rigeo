import numpy as np

import inertial_params as ip


def test_collision():
    box = ip.Box(half_extents=[0.5, 0.5, 0.5])
    sphere = ip.Ellipsoid.sphere(radius=0.5, center=[2, 0, 0])
    info = ip.closest_points(box, sphere)

    assert np.isclose(info.dist, 1.0)
    assert np.allclose(info.p1, [0.5, 0, 0])
    assert np.allclose(info.p2, [1.5, 0, 0])

    # shapes are overlapping
    cylinder = ip.Cylinder(length=1.0, radius=0.5, center=[0.9, 0, 0])
    info = ip.closest_points(box, cylinder)

    assert np.isclose(info.dist, 0.0)
    assert np.allclose(info.p1, info.p2)
    assert box.contains(info.p1)
    assert cylinder.contains(info.p1)
