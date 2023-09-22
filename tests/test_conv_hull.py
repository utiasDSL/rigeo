import numpy as np
import inertial_params as ip


def test_conv_hull_cube():
    half_extents = 0.5 * np.ones(3)
    points = ip.AxisAlignedBox(half_extents).vertices
    vertices = ip.convex_hull(points)
    assert np.allclose(vertices, points)

    # generate some random points inside the hull
    np.random.seed(0)
    extras = ip.AxisAlignedBox(half_extents).random_points(10)
    points_extra = np.vstack((points, extras))
    vertices = ip.convex_hull(points_extra)
    assert np.allclose(vertices, points)
