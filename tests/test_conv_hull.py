import numpy as np
import inertial_params as ip


def test_conv_hull_cube():
    half_extents = 0.5 * np.ones(3)
    points = ip.cuboid_vertices(half_extents)
    vertices = ip.convex_hull(points)
    assert np.allclose(vertices, points)

    # generate some random points inside the hull
    np.random.seed(0)
    extras = ip.random_point_in_box(half_extents, n=10)
    points_extra = np.vstack((points, extras))
    vertices = ip.convex_hull(points_extra)
    assert np.allclose(vertices, points)
